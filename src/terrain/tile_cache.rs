use gfx;
use gfx_core;
use memmap::MmapViewSync;
use vec_map::VecMap;

use terrain::quadtree::{Node, NodeId};
use runtime_texture::{TextureArray, TextureFormat};

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

pub const NUM_LAYERS: usize = 4;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) enum LayerType {
    Heights = 0,
    Colors = 1,
    Normals = 2,
    Water = 3,
}
impl LayerType {
    pub fn cache_size(&self) -> u16 {
        match *self {
            LayerType::Heights => 256,
            LayerType::Colors => 256,
            LayerType::Normals => 32,
            LayerType::Water => 256,
        }
    }
    pub fn index(&self) -> usize {
        *self as usize
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct LayerParams {
    /// What kind of layer this is. There can be at most one of each layer type in a file.
    pub layer_type: LayerType,
    /// Byte offset from start of file.
    pub offset: usize,
    /// Number of tiles in layer.
    pub tile_count: usize,
    /// Number of samples in each dimension, per tile.
    pub tile_resolution: u32,
    /// Number of samples outside the tile on each side.
    pub border_size: u32,
    /// Format used by this layer.
    pub format: TextureFormat,
    /// Number of bytes per tile.
    pub tile_bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct NoiseParams {
    pub offset: usize,
    pub resolution: u32,
    pub format: TextureFormat,
    pub bytes: usize,
    pub wavelength: f32,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct MeshDescriptor {
    pub offset: usize,
    pub bytes: usize,
    pub num_vertices: usize,
}

#[derive(Serialize, Deserialize)]
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

    texture: TextureArray<R>,

    /// Section of memory map that holds the data for this layer.
    data_file: MmapViewSync,
}
impl<R: gfx::Resources> TileCache<R> {
    pub fn new<F: gfx::Factory<R>>(
        params: LayerParams,
        data_file: MmapViewSync,
        factory: &mut F,
    ) -> Self {
        let cache_size = params.layer_type.cache_size();
        let texture = TextureArray::new(
            params.format,
            params.tile_resolution as u16,
            cache_size,
            factory,
        );

        Self {
            size: cache_size as usize,
            slots: Vec::new(),
            reverse: VecMap::new(),
            missing: Vec::new(),
            min_priority: Priority::none(),
            layer_params: params,
            data_file,
            texture,
        }
    }

    pub fn update_priorities(&mut self, nodes: &mut Vec<Node>) {
        for &mut (ref mut priority, id) in self.slots.iter_mut() {
            *priority = nodes[id].priority();
        }

        self.min_priority = self.slots.iter().map(|s| s.0).min().unwrap_or(
            Priority::none(),
        );
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
            let mut possible: Vec<_> = self.slots
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
        let len = self.layer_params.tile_bytes;
        let offset = self.layer_params.offset + tile * len;
        let data = unsafe { &self.data_file.as_slice()[offset..(offset + len)] };

        self.texture.update_layer(
            slot as u16,
            self.layer_params.tile_resolution as u16,
            gfx::memory::cast_slice(data),
            encoder,
        )
    }

    pub fn contains(&self, id: NodeId) -> bool {
        self.reverse.contains_key(id.index())
    }

    pub fn get_slot(&self, id: NodeId) -> Option<usize> {
        self.reverse.get(id.index()).cloned()
    }

    pub fn get_texture_view_f32(&self) -> Option<&gfx_core::handle::ShaderResourceView<R, f32>> {
        match self.texture {
            TextureArray::F32 { ref view, .. } => Some(view),
            _ => None,
        }
    }

    pub fn get_texture_view_rgba8(
        &self,
    ) -> Option<&gfx_core::handle::ShaderResourceView<R, [f32; 4]>> {
        match self.texture {
            TextureArray::RGBA8 { ref view, .. } => Some(view),
            _ => None,
        }
    }

    pub fn get_texture_view_srgba(
        &self,
    ) -> Option<&gfx_core::handle::ShaderResourceView<R, [f32; 4]>> {
        match self.texture {
            TextureArray::SRGBA { ref view, .. } => Some(view),
            _ => None,
        }
    }

    pub fn resolution(&self) -> u32 {
        self.layer_params.tile_resolution
    }

    pub fn border(&self) -> u32 {
        self.layer_params.border_size
    }

    pub fn get_texel(&self, node: &Node, x: usize, y: usize) -> &[u8] {
        let tile = node.tile_indices[self.layer_params.layer_type.index()].unwrap() as usize;
        let len = self.layer_params.tile_bytes;
        let offset = self.layer_params.offset + tile * len;
        let tile_data = unsafe { &self.data_file.as_slice()[offset..(offset + len)] };
        let resolution = self.layer_params.tile_resolution as usize;
        let border = self.layer_params.border_size as usize;
        let bytes_per_texel = self.layer_params.format.bytes_per_texel();
        let index = ((x + border) + (y + border) * resolution) * bytes_per_texel;
        &tile_data[index..(index + bytes_per_texel)]
    }
}
