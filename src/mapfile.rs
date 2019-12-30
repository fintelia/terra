use crate::coordinates::CoordinateSystem;
use crate::terrain::quadtree::{Node, NodeId};
use crate::terrain::tile_cache::{LayerType, TextureDescriptor, LayerParams, TextureFormat, TileHeader};
use memmap::Mmap;
use std::sync::Arc;

pub enum TileState {
    Present,
    Missing,
}

pub(crate) struct MapFile {
    header: TileHeader,
    file: Arc<Mmap>,
}
impl MapFile {
    pub fn new(header: TileHeader, file: Arc<Mmap>) -> Self {
        Self { header, file }
    }

    pub fn tile_state(&self, layer: LayerType, node: NodeId) -> TileState {
        todo!()
    }

    pub fn read_tile(&self, layer: LayerType, node: NodeId) -> Option<&[u8]> {
        todo!()
    }
    pub fn write_tile(&mut self, layer: LayerType, node: NodeId, data: &[u8]) {
        todo!()
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
                origin: wgpu::Origin3d { x: 0.0, y: 0.0, z: 0.0 },
            },
            wgpu::Extent3d { width: resolution as u32, height: resolution as u32, depth: 1 },
        );

		texture
    }

    pub fn noise_texture(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Texture {
		self.load_texture(device, encoder, &self.header.noise.texture)
	}

    pub fn planet_mesh_texture(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Texture {
		self.load_texture(device, encoder, &self.header.planet_mesh_texture)
	}

	pub fn system(&self) -> CoordinateSystem {
		self.header.system.clone()
	}
	pub fn layers(&self) -> &[LayerParams] {
		&self.header.layers
	}
	pub fn data_file(&self) -> Arc<Mmap> {
		self.file.clone()
	}
}
