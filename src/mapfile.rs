use crate::cache::TERRA_DIRECTORY;
use crate::terrain::quadtree::node::VNode;
use crate::terrain::tile_cache::{
    LayerParams, LayerType, TextureDescriptor, TextureFormat, TileHeader,
};
use failure::Error;
use memmap::MmapMut;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use vec_map::VecMap;

#[derive(Serialize, Deserialize)]
pub(crate) enum TileState {
    Missing,
    Base,
    Generated,
    GpuOnly,
}

#[derive(Serialize, Deserialize)]
struct TileMeta {
    crc32: u32,
    state: TileState,
}

pub struct MapFile {
    header: TileHeader,
    db: sled::Db,
}
impl MapFile {
    pub(crate) fn new(header: TileHeader) -> Self {
        let db = sled::open(TERRA_DIRECTORY.join("tiles/meta")).unwrap();
        Self { header, db }
    }

    pub(crate) fn tile_state(&self, layer: LayerType, node: VNode) -> Result<TileState, Error> {
        Ok(match self.lookup_tile_meta(layer, node)? {
            Some(meta) => meta.state,
            None => TileState::GpuOnly,
        })
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
        }
    }
    pub(crate) fn write_tile(
        &mut self,
        layer: LayerType,
        node: VNode,
        data: &[u8],
        base: bool,
    ) -> Result<(), Error> {
        let filename = Self::tile_name(layer, node);
        match layer {
            LayerType::Albedo => image::save_buffer_with_format(
                &filename,
                data,
                self.header.layers[layer].texture_resolution as u32,
                self.header.layers[layer].texture_resolution as u32,
                image::ColorType::Rgba8,
                image::ImageFormat::Bmp,
            )?,
            LayerType::Heightmaps | LayerType::Normals | LayerType::Displacements => {
                fs::write(filename, data)?;
            }
        }

        self.update_tile_meta(
            layer,
            node,
            TileMeta { crc32: 0, state: if base { TileState::Base } else { TileState::Generated } },
        )
    }

    pub(crate) fn clear_generated(&mut self, layer: LayerType) -> Result<(), Error> {
        let prefix = bincode::serialize(&layer).unwrap();
        for i in self.db.scan_prefix(&prefix) {
            let (k, v) = i?;
            if let TileState::Generated = bincode::deserialize::<TileMeta>(&v)?.state {
                self.db.remove(k)?;
            }
        }
        Ok(())
    }

    fn load_texture(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        desc: &TextureDescriptor,
        name: &str,
    ) -> Result<wgpu::Texture, Error> {
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
        // let data = &self.file[desc.offset..][..desc.bytes];

        let filename = TERRA_DIRECTORY.join(name);
        let image = image::open(filename)?;
        let data = image.to_rgba().into_vec();

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

        Ok(texture)
    }

    pub(crate) fn noise_texture(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<wgpu::Texture, Error> {
        self.load_texture(device, encoder, &self.header.noise.texture, "noise.bmp")
    }

    pub(crate) fn save_noise_texture(desc: &TextureDescriptor, data: &[u8]) -> Result<(), Error> {
        assert_eq!(desc.format, TextureFormat::RGBA8);
        Ok(image::save_buffer_with_format(
            &TERRA_DIRECTORY.join("noise.bmp"),
            data,
            desc.resolution as u32,
            desc.resolution as u32,
            image::ColorType::Rgba8,
            image::ImageFormat::Bmp,
        )?)
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

    pub(crate) fn set_missing(&self, layer: LayerType, node: VNode) -> Result<(), Error> {
        self.update_tile_meta(layer, node, TileMeta {crc32: 0, state: TileState::Missing })
    }

    fn lookup_tile_meta(&self, layer: LayerType, node: VNode) -> Result<Option<TileMeta>, Error> {
        let key = bincode::serialize(&(layer, node)).unwrap();
        Ok(self.db.get(key)?.map(|value| bincode::deserialize(&value).unwrap()))
    }
    fn update_tile_meta(&self, layer: LayerType, node: VNode, meta: TileMeta) -> Result<(), Error> {
        let key = bincode::serialize(&(layer, node)).unwrap();
        let value = bincode::serialize(&meta).unwrap();
        self.db.insert(key, value)?;
        Ok(())
    }
}
