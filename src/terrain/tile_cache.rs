use std::convert::TryInto;
use std::sync::Arc;

use gfx;
use gfx::traits::FactoryExt;
use gfx_core;
use memmap::Mmap;
use vec_map::VecMap;

use coordinates::CoordinateSystem;
use runtime_texture::{TextureArray, TextureFormat};
use terrain::quadtree::{Node, NodeId};

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
            LayerType::Colors => 256,
            LayerType::Normals => 256,
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
    /// Where each tile is located in the file.
    pub tile_locations: Vec<ByteRange>,
    /// What kind of data is stored in this layer.
    pub payload_type: PayloadType,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct NoiseParams {
    pub offset: usize,
    pub resolution: u32,
    pub format: TextureFormat,
    pub bytes: usize,
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

gfx_vertex_struct!(MeshInstance {
    position: [f32; 3] = "vPosition",
    color: [f32; 3] = "vColor",
    rotation: f32 = "vRotation",
    scale: f32 = "vScale",
    normal: [f32; 3] = "vNormal",
    padding1: f32 = "vPadding1",
    padding2: [f32; 4] = "vPadding2",
});

enum PayloadSet<R: gfx::Resources> {
    Texture {
        texture: TextureArray<R>,
        resolution: u16,
    },
    InstancedMesh {
        buffer: gfx::handle::Buffer<R, MeshInstance>,
        max_elements_per_slot: usize,
    },
}

pub(crate) struct TileCache<R: gfx::Resources> {
    /// Maximum number of slots in this `TileCache`.
    size: usize,
    /// Actually contents of the cache.
    slots: Vec<(Priority, NodeId)>,
    /// Which index each node is at in the cache (if any).
    reverse: VecMap<usize>,
    /// Nodes that should be added to the cache.
    missing: Vec<(Priority, NodeId)>,
    /// Smallest priority among all nodes in the cache.
    min_priority: Priority,

    /// Resolution of each tile in this cache.
    layer_params: LayerParams,

    payloads: PayloadSet<R>,

    /// Section of memory map that holds the data for this layer.
    data_file: Arc<Mmap>,
}
impl<R: gfx::Resources> TileCache<R> {
    pub fn new<F: gfx::Factory<R>>(
        params: LayerParams,
        data_file: Arc<Mmap>,
        factory: &mut F,
    ) -> Self {
        let cache_size = params.layer_type.cache_size();
        let payloads = match params.payload_type {
            PayloadType::Texture {
                resolution, format, ..
            } => PayloadSet::Texture {
                texture: TextureArray::new(format, resolution as u16, cache_size, factory),
                resolution: resolution.try_into().unwrap(),
            },
            PayloadType::InstancedMesh { max_instances, .. } => PayloadSet::InstancedMesh {
                buffer: factory.create_constant_buffer(max_instances * cache_size as usize),
                max_elements_per_slot: max_instances,
            },
        };

        Self {
            size: cache_size as usize,
            slots: Vec::new(),
            reverse: VecMap::new(),
            missing: Vec::new(),
            min_priority: Priority::none(),
            layer_params: params,
            data_file,
            payloads,
        }
    }

    pub fn update_priorities(&mut self, nodes: &mut Vec<Node>) {
        for &mut (ref mut priority, id) in self.slots.iter_mut() {
            *priority = nodes[id].priority();
        }

        self.min_priority = self
            .slots
            .iter()
            .map(|s| s.0)
            .min()
            .unwrap_or(Priority::none());
    }

    pub fn add_missing(&mut self, element: (Priority, NodeId)) {
        if element.0 > self.min_priority || self.slots.len() < self.size {
            self.missing.push(element);
        }
    }

    pub fn load_missing<C: gfx_core::command::Buffer<R>>(
        &mut self,
        nodes: &mut Vec<Node>,
        encoder: &mut gfx::Encoder<R, C>,
    ) {
        while !self.missing.is_empty() && self.slots.len() < self.size {
            let m = self.missing.pop().unwrap();
            let index = self.slots.len();
            self.slots.push(m.clone());
            self.reverse.insert(m.1.index(), index);
            self.load(&mut nodes[m.1], index, encoder);
        }

        if !self.missing.is_empty() {
            let mut possible: Vec<_> = self
                .slots
                .iter()
                .cloned()
                .chain(self.missing.iter().cloned())
                .collect();
            possible.sort();

            // Anything >= to cutoff should be included.
            let cutoff = possible[possible.len() - self.size];

            let mut index = 0;
            while let Some(m) = self.missing.pop() {
                if cutoff >= m {
                    continue;
                }

                // Find the next element to evict.
                while self.slots[index] >= cutoff {
                    index += 1;
                }

                self.reverse.remove(self.slots[index].1.index());
                self.reverse.insert(m.1.index(), index);
                self.slots[index] = m.clone();
                self.load(&mut nodes[m.1], index, encoder);
                index += 1;
            }
        }
    }

    fn load<C: gfx_core::command::Buffer<R>>(
        &mut self,
        node: &mut Node,
        slot: usize,
        encoder: &mut gfx::Encoder<R, C>,
    ) {
        let tile = node.tile_indices[self.layer_params.layer_type.index()].unwrap() as usize;
        let offset = self.layer_params.tile_locations[tile].offset;
        let length = self.layer_params.tile_locations[tile].length;
        let data = &self.data_file[offset..(offset + length)];

        match self.payloads {
            PayloadSet::Texture {
                ref mut texture,
                resolution,
            } => texture.update_layer(
                slot as u16,
                resolution,
                gfx::memory::cast_slice(data),
                encoder,
            ),
            PayloadSet::InstancedMesh {
                ref buffer,
                max_elements_per_slot,
            } => encoder
                .update_buffer(
                    buffer,
                    gfx::memory::cast_slice(data),
                    max_elements_per_slot * slot,
                ).unwrap(),
        }
    }

    pub fn contains(&self, id: NodeId) -> bool {
        self.reverse.contains_key(id.index())
    }

    pub fn get_slot(&self, id: NodeId) -> Option<usize> {
        self.reverse.get(id.index()).cloned()
    }

    pub fn get_texture_view_r8(&self) -> Option<&gfx_core::handle::ShaderResourceView<R, f32>> {
        match self.payloads {
            PayloadSet::Texture {
                texture: TextureArray::R8 { ref view, .. },
                ..
            } => Some(view),
            _ => None,
        }
    }

    pub fn get_texture_view_f32(&self) -> Option<&gfx_core::handle::ShaderResourceView<R, f32>> {
        match self.payloads {
            PayloadSet::Texture {
                texture: TextureArray::F32 { ref view, .. },
                ..
            } => Some(view),
            _ => None,
        }
    }

    pub fn get_texture_view_rgba8(
        &self,
    ) -> Option<&gfx_core::handle::ShaderResourceView<R, [f32; 4]>> {
        match self.payloads {
            PayloadSet::Texture {
                texture: TextureArray::RGBA8 { ref view, .. },
                ..
            } => Some(view),
            _ => None,
        }
    }

    pub fn get_texture_view_srgba(
        &self,
    ) -> Option<&gfx_core::handle::ShaderResourceView<R, [f32; 4]>> {
        match self.payloads {
            PayloadSet::Texture {
                texture: TextureArray::SRGBA { ref view, .. },
                ..
            } => Some(view),
            _ => None,
        }
    }

    pub fn get_buffer(&self) -> Option<&gfx::handle::Buffer<R, MeshInstance>> {
        match self.payloads {
            PayloadSet::InstancedMesh { ref buffer, .. } => Some(buffer),
            _ => None,
        }
    }

    pub fn get_instance_offset(&self, slot: usize) -> usize {
        if let PayloadSet::InstancedMesh {
            max_elements_per_slot,
            ..
        } = self.payloads
        {
            slot * max_elements_per_slot
        } else {
            unreachable!()
        }
    }

    pub fn get_instance_count(&self, node: &Node) -> usize {
        let tile = node.tile_indices[self.layer_params.layer_type.index()].unwrap() as usize;
        let length = self.layer_params.tile_locations[tile].length;
        (length as usize / ::std::mem::size_of::<MeshInstance>())
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

    pub fn get_texel(&self, node: &Node, x: usize, y: usize) -> &[u8] {
        if let PayloadType::Texture {
            border_size,
            resolution,
            format,
        } = self.layer_params.payload_type
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn mesh_instance_size() {
        assert_eq!(mem::size_of::<MeshInstance>(), 32);
    }
}
