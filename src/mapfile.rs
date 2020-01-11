use crate::terrain::tile_cache::{LayerParams, LayerType, TextureDescriptor, TileHeader};
use failure::Error;
use memmap::MmapMut;
use vec_map::VecMap;

pub(crate) enum TileState {
    Missing,
    Base,
    Generated,
}

pub struct MapFile {
    header: TileHeader,
    file: MmapMut,
}
impl MapFile {
    pub(crate) fn new(header: TileHeader, file: MmapMut) -> Self {
        Self { header, file }
    }

    pub(crate) fn tile_state(&self, layer: LayerType, tile: usize) -> TileState {
        let offset = self.header.layers[layer.index()].tile_valid_bitmap.offset;
        match self.file[offset + tile] {
            0 => TileState::Missing,
            1 => TileState::Base,
            2 => TileState::Generated,
            _ => unreachable!(),
        }
    }
    pub(crate) fn read_tile(&self, layer: LayerType, tile: usize) -> Option<&[u8]> {
        let params = &self.header.layers[layer.index()];
        let offset = params.tile_locations[tile].offset;
        let length = params.tile_locations[tile].length;

        Some(&self.file[offset..][..length])
    }
    pub(crate) fn write_tile(
        &mut self,
        layer: LayerType,
        tile: usize,
        data: &[u8],
    ) -> Result<(), Error> {
        let params = &self.header.layers[layer.index()];
        let offset = params.tile_locations[tile].offset;
        let length = params.tile_locations[tile].length;

        assert_eq!(length, data.len());

        self.file[offset..][..length].copy_from_slice(data);
        self.file.flush_range(offset, length)?;
        self.file[params.tile_valid_bitmap.offset + tile] = 2;
        self.file.flush_async_range(params.tile_valid_bitmap.offset + tile, 1)?;
        Ok(())
    }

    pub(crate) fn clear_generated(&mut self, layer: LayerType) {
        let bitmap = self.header.layers[layer.index()].tile_valid_bitmap;
        for i in 0..bitmap.length {
            if self.file[bitmap.offset + i] == 2 {
                self.file[bitmap.offset + i] = 0;
            }
        }
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
        });

        let resolution = desc.resolution as usize;
        let bytes_per_texel = desc.format.bytes_per_texel();
        let row_bytes = resolution * bytes_per_texel;
        let row_pitch = (row_bytes + 255) & !255;
        let data = &self.file[desc.offset..][..desc.bytes];

        let mapped = device.create_buffer_mapped(
            row_pitch * resolution,
            wgpu::BufferUsage::MAP_WRITE | wgpu::BufferUsage::COPY_SRC,
        );
        for row in 0..resolution {
            mapped.data[row * row_pitch..][..row_bytes]
                .copy_from_slice(&data[row * row_bytes..][..row_bytes]);
        }

        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &mapped.finish(),
                offset: 0,
                row_pitch: row_pitch as u32,
                image_height: resolution as u32,
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
    pub(crate) fn planet_mesh_texture(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Texture {
        self.load_texture(device, encoder, &self.header.planet_mesh_texture)
    }
    pub(crate) fn base_heights_texture(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Texture {
        self.load_texture(device, encoder, &self.header.base_heights)
    }

    pub(crate) fn layers(&self) -> &VecMap<LayerParams> {
        &self.header.layers
    }
}
