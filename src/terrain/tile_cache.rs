use gfx;
use gfx_core;
use memmap::MmapViewSync;
use vec_map::VecMap;

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

pub const HEIGHTS_LAYER: usize = 0;
#[allow(unused)]
pub const NORMALS_LAYER: usize = 1;
#[allow(unused)]
pub const SPLATS_LAYER: usize = 2;
pub const NUM_LAYERS: usize = 3;

#[derive(Clone, Default, Serialize, Deserialize)]
pub(crate) struct LayerParams {
    /// Byte offset from start of file.
    pub offset: usize,
    /// Number of tiles in layer.
    pub tile_count: usize,
    /// Number of samples in each dimension, per tile.
    pub tile_resolution: u32,
    /// Number of bytes in each sample.
    pub sample_bytes: usize,
    /// Number of bytes per tile
    pub tile_bytes: usize,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct TileHeader {
    pub layers: [LayerParams; NUM_LAYERS],
    pub nodes: Vec<Node>,
}

pub(crate) struct TileCache<R: gfx::Resources> {
    /// Which layer this is a tile cache for.
    layer_id: usize,
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

    texture: gfx_core::handle::Texture<R, gfx_core::format::R32>,
    pub(crate) texture_view: gfx_core::handle::ShaderResourceView<R, f32>,

    /// Section of memory map that holds the data for this layer.
    data_file: MmapViewSync,
}
impl<R: gfx::Resources> TileCache<R> {
    pub fn new<F: gfx::Factory<R>>(
        layer_id: usize,
        cache_size: u16,
        params: LayerParams,
        data_file: MmapViewSync,
        factory: &mut F,
    ) -> Self {
        let texture = factory
            .create_texture::<gfx::format::R32>(
                gfx::texture::Kind::D2Array(
                    params.tile_resolution as u16,
                    params.tile_resolution as u16,
                    cache_size,
                    gfx::texture::AaMode::Single,
                ),
                1,
                gfx::SHADER_RESOURCE,
                gfx::memory::Usage::Dynamic,
                Some(gfx::format::ChannelType::Float),
            )
            .unwrap();
        let texture_view = factory
            .view_texture_as_shader_resource::<f32>(&texture, (0, 0), gfx::format::Swizzle::new())
            .unwrap();

        Self {
            layer_id,
            size: cache_size as usize,
            slots: Vec::new(),
            reverse: VecMap::new(),
            missing: Vec::new(),
            min_priority: Priority::none(),
            layer_params: params,
            data_file,
            texture,
            texture_view,
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
        if self.slots.len() + self.missing.len() < self.size {
            while let Some(m) = self.missing.pop() {
                let index = self.slots.len();
                self.slots.push(m.clone());
                self.load(m.1, &mut nodes[m.1], index, encoder);
            }
        } else {
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

                self.slots[index] = m.clone();
                self.load(m.1, &mut nodes[m.1], index, encoder);
                index += 1;
            }
        }
    }

    fn load<C: gfx_core::command::Buffer<R>>(
        &mut self,
        id: NodeId,
        node: &mut Node,
        slot: usize,
        encoder: &mut gfx::Encoder<R, C>,
    ) {
        if slot < self.slots.len() {
            self.reverse.remove(self.slots[slot].1.index());
        }
        self.reverse.insert(id.index(), slot);

        let tile = node.tile_indices[self.layer_id].unwrap() as usize;
        let len = self.layer_params.tile_bytes;
        let offset = self.layer_params.offset + tile * len;
        let data = unsafe { &self.data_file.as_slice()[offset..(offset + len)] };

        encoder
            .update_texture::<gfx::format::R32, f32>(
                &self.texture,
                None,
                gfx_core::texture::NewImageInfo {
                    xoffset: 0,
                    yoffset: 0,
                    zoffset: slot as u16,
                    width: self.layer_params.tile_resolution as u16,
                    height: self.layer_params.tile_resolution as u16,
                    depth: 1,
                    format: (),
                    mipmap: 0,
                },
                gfx::memory::cast_slice(data),
            )
            .unwrap();
    }

    pub fn contains(&self, id: NodeId) -> bool {
        self.reverse.contains_key(id.index())
    }

    pub fn get_slot(&self, id: NodeId) -> Option<usize> {
        self.reverse.get(id.index()).cloned()
    }

    #[allow(unused)]
    pub fn resolution(&self) -> u32 {
        self.layer_params.tile_resolution
    }
}
