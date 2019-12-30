use std::sync::Arc;

use cgmath::Point3;
use memmap::Mmap;
use serde::{Deserialize, Serialize};
use vec_map::VecMap;

use crate::coordinates::CoordinateSystem;
// use runtime_texture::{TextureArray, TextureFormat};
use crate::terrain::quadtree::{Node, NodeId};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TextureFormat {
    R8,
    RG8,
    R32F,
    RG32F,
    RGBA8,
    SRGBA,
}
impl TextureFormat {
    pub fn bytes_per_texel(&self) -> usize {
        match *self {
            TextureFormat::R8 => 1,
            TextureFormat::RG8 => 2,
            TextureFormat::R32F => 4,
            TextureFormat::RG32F => 8,
            TextureFormat::RGBA8 => 4,
            TextureFormat::SRGBA => 4,
        }
    }
	pub fn to_wgpu(&self) -> wgpu::TextureFormat {
		match *self {
            TextureFormat::R8 => wgpu::TextureFormat::R8Unorm,
            TextureFormat::RG8 => wgpu::TextureFormat::Rg8Unorm,
            TextureFormat::R32F => wgpu::TextureFormat::R32Float,
            TextureFormat::RG32F => wgpu::TextureFormat::Rg32Float,
            TextureFormat::RGBA8 => wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::SRGBA => wgpu::TextureFormat::Rgba8UnormSrgb,
		}
	}
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Priority(f32);
impl Priority {
    pub fn cutoff() -> Self {
        Priority(1.0)
    }
    pub fn none() -> Self {
        Priority(-1.0)
    }
    pub fn from_f32(value: f32) -> Self {
        assert!(value.is_finite());
        Priority(value)
    }
}
impl Eq for Priority {}
impl Ord for Priority {
    fn cmp(&self, other: &Self) -> ::std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub const NUM_LAYERS: usize = 5;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) enum LayerType {
    Heights = 0,
    Colors = 1,
    Normals = 2,
    Splats = 3,
    Foliage = 4,
}
impl LayerType {
    pub fn cache_size(&self) -> u16 {
        match *self {
            LayerType::Heights => 512,
            LayerType::Colors => 384,
            LayerType::Normals => 384,
            LayerType::Splats => 32,
            LayerType::Foliage => 64,
        }
    }
    pub fn index(&self) -> usize {
        *self as usize
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ByteRange {
    pub offset: usize,
    pub length: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum PayloadType {
    Texture {
        /// Number of samples in each dimension, per tile.
        resolution: u32,
        /// Number of samples outside the tile on each side.
        border_size: u32,
        /// Format used by this layer.
        format: TextureFormat,
    },
    InstancedMesh {
        /// Actual mesh that is being instanced.
        mesh: MeshDescriptor,
        /// Texture map for the mesh.
        texture: TextureDescriptor,
        /// Maximum number of instances in any tile.
        max_instances: usize,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct LayerParams {
    /// What kind of layer this is. There can be at most one of each layer type in a file.
    pub layer_type: LayerType,
    // Array of bytes indicating whether each tile is valid.
    pub tile_valid_bitmap: ByteRange,
    /// Where each tile is located in the file.
    pub tile_locations: Vec<ByteRange>,
    /// What kind of data is stored in this layer.
    pub payload_type: PayloadType,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct NoiseParams {
	pub texture: TextureDescriptor,
    pub wavelength: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MeshDescriptor {
    pub offset: usize,
    pub bytes: usize,
    pub num_vertices: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct TextureDescriptor {
    pub offset: usize,
    pub resolution: u32,
    pub format: TextureFormat,
    pub bytes: usize,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct TileHeader {
    pub layers: Vec<LayerParams>,
    pub noise: NoiseParams,
    pub nodes: Vec<Node>,
    pub planet_mesh: MeshDescriptor,
    pub planet_mesh_texture: TextureDescriptor,
    pub system: CoordinateSystem,
}

// gfx_vertex_struct!(MeshInstance {
//     position: [f32; 3] = "vPosition",
//     color: [f32; 3] = "vColor",
//     rotation: f32 = "vRotation",
//     scale: f32 = "vScale",
//     normal: [f32; 3] = "vNormal",
//     padding1: f32 = "vPadding1",
//     padding2: [f32; 4] = "vPadding2",
// });

// enum PayloadSet<B: Backend> {
//     Texture {
//         texture: Texture<B>,
//         resolution: u16,
//     },
//     InstancedMesh {
//         buffer: gfx::handle::Buffer<R, MeshInstance>,
//         max_elements_per_slot: usize,
//     },
// }

struct Entry {
    priority: Priority,
    id: NodeId,
    valid: bool,
}

pub(crate) struct TileCache {
    size: usize,
    slots: Vec<Entry>,
    reverse: VecMap<usize>,

    /// Nodes that should be added to the cache.
    missing: Vec<(Priority, NodeId)>,
    /// Smallest priority among all nodes in the cache.
    min_priority: Priority,

    /// Slots which still need to be loaded
    pending_uploads: Vec<(usize, ByteRange)>,

    /// Resolution of each tile in this cache.
    layer_params: LayerParams,

    /// Section of memory map that holds the data for this layer.
    data_file: Arc<Mmap>,
}
impl TileCache {
    pub fn new(params: LayerParams, data_file: Arc<Mmap>) -> Self {
        let size = params.layer_type.cache_size() as usize;

        Self {
            size,
            slots: Vec::new(),
            reverse: VecMap::new(),
            missing: Vec::new(),
            pending_uploads: Vec::new(),
            min_priority: Priority::none(),
            layer_params: params,
            data_file,
        }
    }

    pub fn update_priorities(&mut self, nodes: &mut Vec<Node>, camera: Point3<f32>) {
        for entry in &mut self.slots {
            entry.priority = nodes[entry.id].priority(camera);
        }

        self.min_priority = self.slots.iter().map(|s| s.priority).min().unwrap_or(Priority::none());
    }

    pub fn add_missing(&mut self, element: (Priority, NodeId)) {
        if element.0 > self.min_priority || self.slots.len() < self.size {
            self.missing.push(element);
        }
    }

    pub fn process_missing(&mut self, nodes: &mut Vec<Node>) {
        // Find slots for missing entries.
        self.missing.sort();
        while !self.missing.is_empty() && self.slots.len() < self.size {
            let m = self.missing.pop().unwrap();
            self.reverse.insert(m.1.index(), self.slots.len());
            self.slots.push(Entry { priority: m.0, id: m.1, valid: false });
        }
        if !self.missing.is_empty() {
            let mut possible: Vec<_> = self
                .slots
                .iter()
                .map(|e| e.priority)
                .chain(self.missing.iter().map(|m| m.0))
                .collect();
            possible.sort();

            // Anything >= to cutoff should be included.
            let cutoff = possible[possible.len() - self.size];

            let mut index = 0;
            'outer: while let Some(m) = self.missing.pop() {
                if cutoff >= m.0 {
                    continue;
                }

                // Find the next element to evict.
                while self.slots[index].priority >= cutoff {
                    index += 1;
                    if index == self.slots.len() {
                        break 'outer;
                    }
                }

                self.reverse.remove(self.slots[index].id.index());
                self.reverse.insert(m.1.index(), index);
                self.slots[index] = Entry { priority: m.0, id: m.1, valid: false };
                index += 1;
                if index == self.slots.len() {
                    break;
                }
            }
        }
        self.missing.clear();

        // Figure out which entries need to be uploaded
        self.pending_uploads.clear();
        for (i, entry) in self.slots.iter_mut().enumerate() {
            if entry.valid {
                continue;
            }

            let tile = nodes[entry.id].tile_indices[self.layer_params.layer_type.index()].unwrap()
                as usize;
            let tile_valid = self.data_file[self.layer_params.tile_valid_bitmap.offset + tile] != 0;

            if tile_valid {
                // Tile is on disk, just needs to be uploaded
                let offset = self.layer_params.tile_locations[tile].offset;
                let length = self.layer_params.tile_locations[tile].length;
                self.pending_uploads.push((i, ByteRange { offset, length }));
            } else {
                // TODO: Tile needs to be generated.
            }
        }
    }

    pub(crate) fn upload_tiles(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
    ) {
        if self.pending_uploads.is_empty() {
            return;
        }

        let resolution = self.resolution() as usize;
        let bytes_per_texel = self.bytes_per_texel();
        let row_bytes = resolution * bytes_per_texel;
        let row_pitch = (row_bytes + 255) & !255;
        let tiles = self.pending_uploads.len();

        let buffer = device.create_buffer_mapped(
            row_pitch * resolution * tiles,
            wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::MAP_WRITE,
        );

        let mut i = 0;
        for upload in &self.pending_uploads {
            let data = &self.data_file[upload.1.offset..][..upload.1.length];
            for row in 0..resolution {
                buffer.data[i..][..row_bytes]
                    .copy_from_slice(&data[row * row_bytes..][..row_bytes]);
                i += row_pitch;
            }
        }

        let buffer = buffer.finish();

        for (index, (slot, _)) in self.pending_uploads.drain(..).enumerate() {
            encoder.copy_buffer_to_texture(
                wgpu::BufferCopyView {
                    buffer: &buffer,
                    offset: (index * row_pitch * resolution) as u64,
                    row_pitch: row_pitch as u32,
                    image_height: resolution as u32,
                },
                wgpu::TextureCopyView {
                    texture,
                    mip_level: 0,
                    array_layer: slot as u32,
                    origin: wgpu::Origin3d { x: 0.0, y: 0.0, z: 0.0 },
                },
                wgpu::Extent3d { width: resolution as u32, height: resolution as u32, depth: 1 },
            );
            self.slots[slot].valid = true;
        }
    }

    pub fn contains(&self, id: NodeId) -> bool {
        self.reverse.get(id.index())
            .and_then(|&slot| self.slots.get(slot))
            .map(|entry| entry.valid)
            .unwrap_or(false)
    }

    pub fn get_slot(&self, id: NodeId) -> Option<usize> {
        self.reverse.get(id.index()).cloned()
    }

    pub fn resolution(&self) -> u32 {
        match self.layer_params.payload_type {
            PayloadType::Texture { resolution, .. } => resolution,
            _ => unreachable!(),
        }
    }

    pub fn border(&self) -> u32 {
        match self.layer_params.payload_type {
            PayloadType::Texture { border_size, .. } => border_size,
            _ => unreachable!(),
        }
    }

    fn bytes_per_texel(&self) -> usize {
        match self.layer_params.payload_type {
            PayloadType::Texture { format, .. } => format.bytes_per_texel(),
            _ => unreachable!(),
        }
    }

    pub fn get_texel(&self, node: &Node, x: usize, y: usize) -> &[u8] {
        if let PayloadType::Texture { border_size, resolution, format } =
            self.layer_params.payload_type
        {
            let tile = node.tile_indices[self.layer_params.layer_type.index()].unwrap() as usize;
            let offset = self.layer_params.tile_locations[tile].offset;
            let length = self.layer_params.tile_locations[tile].length;
            let tile_data = &self.data_file[offset..(offset + length)];
            let bytes_per_texel = format.bytes_per_texel();
            let border = border_size as usize;
            let index = ((x + border) + (y + border) * resolution as usize) * bytes_per_texel;
            &tile_data[index..(index + bytes_per_texel)]
        } else {
            unreachable!()
        }
    }

    pub fn capacity(&self) -> usize {
        self.size
    }
    pub fn utilization(&self) -> usize {
        self.slots.iter().filter(|s| s.priority >= Priority::cutoff() && s.valid).count()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::mem;

//     #[test]
//     fn mesh_instance_size() {
//         assert_eq!(mem::size_of::<MeshInstance>(), 64);
//     }
// }
