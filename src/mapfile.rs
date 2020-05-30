use crate::cache::TERRA_DIRECTORY;
use crate::terrain::quadtree::node::VNode;
use crate::terrain::tile_cache::{LayerParams, LayerType, TextureDescriptor, TileHeader};
use failure::Error;
use memmap::MmapMut;
use std::path::PathBuf;
use vec_map::VecMap;
use std::collections::HashMap;
use std::fs;

pub(crate) enum TileState {
    Missing,
    Base,
    Generated,
    GpuOnly,
}

pub struct MapFile {
    header: TileHeader,
    file: MmapMut,
    reverse: HashMap<VNode, usize>,
}
impl MapFile {
    pub(crate) fn new(header: TileHeader, file: MmapMut) -> Self {
        let reverse = header.nodes.iter().enumerate().map(|(i, n)| (*n, i)).collect();
        Self { header, file, reverse }
    }

    pub(crate) fn tile_state(&self, layer: LayerType, node: VNode) -> TileState {
        let bitmap = &self.header.layers[layer.index()].tile_valid_bitmap;
        let tile = self.reverse.get(&node);

        if tile.is_none() || *tile.unwrap() >= bitmap.length {
            return TileState::GpuOnly;
        }

        match self.file[bitmap.offset + tile.unwrap()] {
            0 => TileState::Missing,
            1 => TileState::Base,
            2 => TileState::Generated,
            _ => unreachable!(),
        }
    }
    pub(crate) fn read_tile(&self, layer: LayerType, node: VNode) -> Option<Vec<u8>> {
        let filename = Self::tile_name(layer, node);
        match layer {
            LayerType::Albedo => {
                if !filename.exists() {
                    return None;
                }
                let image = image::open(filename).ok()?;
                Some(image.to_rgba().into_vec())
            }
            LayerType::Heightmaps | LayerType::Normals | LayerType::Displacements => {
                if !filename.exists() {
                    return None;
                }
                fs::read(filename).ok()
            }
            // _ => {
            //     let tile = self.reverse[&node];
            //     let params = &self.header.layers[layer.index()];
            //     let offset = params.tile_locations[tile].offset;
            //     let length = params.tile_locations[tile].length;
            //     Some(self.file[offset..][..length].to_vec())
            // }
        }
    }
    pub(crate) fn write_tile(
        &mut self,
        layer: LayerType,
        node: VNode,
        data: &[u8],
    ) -> Result<(), Error> {
        let filename = Self::tile_name(layer, node);
        match layer {
            LayerType::Albedo => {
                Ok(image::save_buffer_with_format(
                    &filename,
                    data,
                    self.header.layers[layer].texture_resolution as u32,
                    self.header.layers[layer].texture_resolution as u32,
                    image::ColorType::Rgba8,
                    image::ImageFormat::Bmp,
                )?)
            }
            LayerType::Heightmaps | LayerType::Normals | LayerType::Displacements => {
                Ok(fs::write(filename, data)?)
            }
            // _ => {
            //     let tile = self.reverse[&node];
            //     let params = &self.header.layers[layer.index()];
            //     let offset = params.tile_locations[tile].offset;
            //     let length = params.tile_locations[tile].length;

            //     assert_eq!(length, data.len());

            //     self.file[offset..][..length].copy_from_slice(data);
            //     self.file.flush_range(offset, length)?;
            //     self.file[params.tile_valid_bitmap.offset + tile] = 2;
            //     self.file.flush_range(params.tile_valid_bitmap.offset + tile, 1)?;
            //     Ok(())
            // }
        }
    }

    pub(crate) fn clear_generated(&mut self, layer: LayerType) -> Result<(), Error> {
        let bitmap = self.header.layers[layer.index()].tile_valid_bitmap;
        for i in 0..bitmap.length {
            if self.file[bitmap.offset + i] == 2 {
                self.file[bitmap.offset + i] = 0;
            }
        }
        self.file.flush_range(bitmap.offset, bitmap.length)?;
        Ok(())
    }

    fn load_texture(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        desc: &TextureDescriptor,
    ) -> wgpu::Texture {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width: desc.resolution, height: desc.resolution, depth: 1 },
            format: desc.format.to_wgpu(),
            mip_level_count: 1,
            array_layer_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::STORAGE,
            label: None,
        });

        let resolution = desc.resolution as usize;
        let bytes_per_texel = desc.format.bytes_per_texel();
        let row_bytes = resolution * bytes_per_texel;
        let row_pitch = (row_bytes + 255) & !255;
        let data = &self.file[desc.offset..][..desc.bytes];

        let mapped = device.create_buffer_mapped(&wgpu::BufferDescriptor {
            size: (row_pitch * resolution) as u64,
            usage: wgpu::BufferUsage::MAP_WRITE | wgpu::BufferUsage::COPY_SRC,
            label: None,
        });

        for row in 0..resolution {
            mapped.data[row * row_pitch..][..row_bytes]
                .copy_from_slice(&data[row * row_bytes..][..row_bytes]);
        }

        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &mapped.finish(),
                offset: 0,
                bytes_per_row: row_pitch as u32,
                rows_per_image: resolution as u32,
            },
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            },
            wgpu::Extent3d { width: resolution as u32, height: resolution as u32, depth: 1 },
        );

        texture
    }

    pub(crate) fn noise_texture(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Texture {
        self.load_texture(device, encoder, &self.header.noise.texture)
    }

    pub(crate) fn layers(&self) -> &VecMap<LayerParams> {
        &self.header.layers
    }

    pub(crate) fn tile_name(layer: LayerType, node: VNode) -> PathBuf {
        let face = match node.face() {
            0 => "0E",
            1 => "180E",
            2 => "N",
            3 => "S",
            4 => "90E",
            5 => "90W",
            _ => unreachable!(),
        };
        let (layer, ext) = match layer {
            LayerType::Displacements => ("displacements", "raw"),
            LayerType::Albedo => ("albedo", "bmp"),
            LayerType::Normals => ("normals", "raw"),
            LayerType::Heightmaps => ("heightmaps", "raw"),
        };
        TERRA_DIRECTORY.join(&format!(
            "tiles/{}_{}_{}_{}x{}.{}",
            layer,
            node.level(),
            face,
            node.x(),
            node.y(),
            ext
        ))
    }
}
