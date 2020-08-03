use crate::mapfile::{MapFile, TileState};
use crate::terrain::quadtree::VNode;
use cgmath::Point3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::{Index, IndexMut};
use vec_map::VecMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TextureFormat {
    R8,
    RG8,
    RGBA8,
    R32F,
    RG32F,
    RGBA32F,
    SRGBA,
    BC4,
    BC5,
}
impl TextureFormat {
    /// Returns the number of bytes in a single texel of the format. Actually reports bytes per
    /// block for compressed formats.
    pub fn bytes_per_block(&self) -> usize {
        match *self {
            TextureFormat::R8 => 1,
            TextureFormat::RG8 => 2,
            TextureFormat::RGBA8 => 4,
            TextureFormat::R32F => 4,
            TextureFormat::RG32F => 8,
            TextureFormat::RGBA32F => 16,
            TextureFormat::SRGBA => 4,
            TextureFormat::BC4 => 8,
            TextureFormat::BC5 => 16,
        }
    }
    pub fn to_wgpu(&self) -> wgpu::TextureFormat {
        match *self {
            TextureFormat::R8 => wgpu::TextureFormat::R8Unorm,
            TextureFormat::RG8 => wgpu::TextureFormat::Rg8Unorm,
            TextureFormat::RGBA8 => wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::R32F => wgpu::TextureFormat::R32Float,
            TextureFormat::RG32F => wgpu::TextureFormat::Rg32Float,
            TextureFormat::RGBA32F => wgpu::TextureFormat::Rgba32Float,
            TextureFormat::SRGBA => wgpu::TextureFormat::Rgba8UnormSrgb,
            TextureFormat::BC4 => wgpu::TextureFormat::Bc4RUnorm,
            TextureFormat::BC5 => wgpu::TextureFormat::Bc5RgUnorm,
        }
    }
    pub fn block_size(&self) -> u32 {
        match *self {
            TextureFormat::BC4 | TextureFormat::BC5 => 4,
            TextureFormat::R8
            | TextureFormat::RG8
            | TextureFormat::RGBA8
            | TextureFormat::R32F
            | TextureFormat::RG32F
            | TextureFormat::RGBA32F
            | TextureFormat::SRGBA => 1,
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) enum LayerType {
    Displacements = 0,
    Albedo = 1,
    Roughness = 2,
    Normals = 3,
    Heightmaps = 4,
}
impl LayerType {
    pub fn index(&self) -> usize {
        *self as usize
    }
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => LayerType::Displacements,
            1 => LayerType::Albedo,
            2 => LayerType::Roughness,
            3 => LayerType::Normals,
            4 => LayerType::Heightmaps,
            _ => unreachable!(),
        }
    }
    pub fn bit_mask(&self) -> u32 {
        1 << self.index() as u32
    }
}
impl<T> Index<LayerType> for VecMap<T> {
    type Output = T;
    fn index(&self, i: LayerType) -> &Self::Output {
        &self[i as usize]
    }
}
impl<T> IndexMut<LayerType> for VecMap<T> {
    fn index_mut(&mut self, i: LayerType) -> &mut Self::Output {
        &mut self[i as usize]
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ByteRange {
    pub offset: usize,
    pub length: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct LayerParams {
    /// What kind of layer this is. There can be at most one of each layer type in a file.
    pub layer_type: LayerType,
    /// Number of samples in each dimension, per tile.
    pub texture_resolution: u32,
    /// Number of samples outside the tile on each side.
    pub texture_border_size: u32,
    /// Format used by this layer.
    pub texture_format: TextureFormat,
}

struct Entry {
    priority: Priority,
    node: VNode,
    valid: u32,
    generated: u32,
}

pub(crate) struct TileCache {
    size: usize,
    slots: Vec<Entry>,
    reverse: HashMap<VNode, usize>,

    /// Nodes that should be added to the cache.
    missing: Vec<(Priority, VNode)>,
    /// Smallest priority among all nodes in the cache.
    min_priority: Priority,

    /// Resolution of each tile in this cache.
    layers: VecMap<LayerParams>,
}
impl TileCache {
    pub fn new(layers: VecMap<LayerParams>, size: usize) -> Self {
        Self {
            size,
            slots: Vec::new(),
            reverse: HashMap::new(),
            missing: Vec::new(),
            min_priority: Priority::none(),
            layers,
        }
    }

    pub fn update_priorities(&mut self, camera_cspace: Point3<f64>) {
        for entry in &mut self.slots {
            entry.priority = entry.node.priority(camera_cspace);
        }

        self.min_priority = self.slots.iter().map(|s| s.priority).min().unwrap_or(Priority::none());
    }

    pub fn add_missing(&mut self, element: (Priority, VNode)) {
        if !self.reverse.contains_key(&element.1)
            && (element.0 > self.min_priority || self.slots.len() < self.size)
        {
            self.missing.push(element);
        }
    }

    fn process_missing(&mut self) {
        // Find slots for missing entries.
        self.missing.sort();
        while !self.missing.is_empty() && self.slots.len() < self.size {
            let m = self.missing.pop().unwrap();
            self.reverse.insert(m.1, self.slots.len());
            self.slots.push(Entry { priority: m.0, node: m.1, valid: 0, generated: 0 });
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

                self.reverse.remove(&self.slots[index].node);
                self.reverse.insert(m.1, index);
                self.slots[index] = Entry { priority: m.0, node: m.1, valid: 0, generated: 0 };
                index += 1;
                if index == self.slots.len() {
                    break;
                }
            }
        }
        self.missing.clear();
    }

    pub(crate) fn upload_tiles(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        mapfile: &mut MapFile,
        ty: LayerType,
    ) -> Vec<VNode> {
        self.process_missing();

        // Figure out which entries need to be uploaded
        let mut pending_uploads = Vec::new();
        let mut pending_generate = Vec::new();

        for (i, ref mut entry) in self.slots.iter_mut().enumerate() {
            if entry.valid & ty.bit_mask() != 0 {
                continue;
            }

            match mapfile.tile_state(ty, entry.node).unwrap() {
                TileState::GpuOnly => {
                    entry.generated |= ty.bit_mask();
                    pending_generate.push(entry.node);
                }
                TileState::Base => {
                    entry.generated &= !ty.bit_mask();
                    pending_uploads.push((i, entry.node));
                }
                TileState::Generated => {
                    entry.generated |= ty.bit_mask();
                    pending_uploads.push((i, entry.node));
                }
                TileState::Missing => pending_generate.push(entry.node),
                TileState::MissingBase => unreachable!(),
            }
        }
        if pending_uploads.is_empty() {
            return pending_generate;
        }

        let resolution = self.resolution(ty) as usize;
        let resolution_blocks = self.resolution_blocks(ty) as usize;
        let bytes_per_block = self.layers[ty].texture_format.bytes_per_block();
        let row_bytes = resolution_blocks * bytes_per_block;
        let row_pitch = (row_bytes + 255) & !255;
        let tiles = pending_uploads.len();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: (row_pitch * resolution_blocks * tiles) as u64,
            usage: wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::MAP_WRITE,
            label: None,
            mapped_at_creation: true,
        });
        let mut buffer_view = buffer.slice(..).get_mapped_range_mut();

        let mut i = 0;
        for upload in &pending_uploads {
            let data = mapfile.read_tile(ty, upload.1).unwrap();
            for row in 0..resolution_blocks {
                buffer_view[i..][..row_bytes]
                    .copy_from_slice(&data[row * row_bytes..][..row_bytes]);
                i += row_pitch;
            }
        }

        drop(buffer_view);
        buffer.unmap();
        for (index, (slot, _)) in pending_uploads.drain(..).enumerate() {
            encoder.copy_buffer_to_texture(
                wgpu::BufferCopyView {
                    buffer: &buffer,
                    layout: wgpu::TextureDataLayout {
                        offset: (index * row_pitch * resolution_blocks) as u64,
                        bytes_per_row: row_pitch as u32,
                        rows_per_image: 0,
                    },
                },
                wgpu::TextureCopyView {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: slot as u32 },
                },
                wgpu::Extent3d { width: resolution as u32, height: resolution as u32, depth: 1 },
            );
            self.slots[slot].valid |= ty.bit_mask();
        }

        pending_generate
    }

    pub fn make_cache_textures(&self, device: &wgpu::Device) -> VecMap<wgpu::Texture> {
        self.layers
            .iter()
            .map(|(ty, layer)| {
                (
                    ty,
                    device.create_texture(&wgpu::TextureDescriptor {
                        size: wgpu::Extent3d {
                            width: layer.texture_resolution,
                            height: layer.texture_resolution,
                            depth: self.size as u32,
                        },
                        format: layer.texture_format.to_wgpu(),
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        usage: wgpu::TextureUsage::COPY_SRC
                            | wgpu::TextureUsage::COPY_DST
                            | wgpu::TextureUsage::SAMPLED
                            | wgpu::TextureUsage::STORAGE,
                        label: None,
                    }),
                )
            })
            .collect()
    }

    pub fn clear_generated(&mut self, ty: LayerType) {
        for i in 0..self.slots.len() {
            if self.slots[i].generated & ty.bit_mask() != 0 {
                self.slots[i].valid &= !ty.bit_mask();
            }
        }
    }

    pub fn contains(&self, node: VNode, ty: LayerType) -> bool {
        self.reverse
            .get(&node)
            .and_then(|&slot| self.slots.get(slot))
            .map(|entry| (entry.valid & ty.bit_mask()) != 0)
            .unwrap_or(false)
    }
    pub fn set_slot_valid(&mut self, slot: usize, ty: LayerType) {
        self.slots[slot].valid |= ty.bit_mask();
        self.slots[slot].generated |= ty.bit_mask();
    }

    pub fn get_slot(&self, node: VNode) -> Option<usize> {
        self.reverse.get(&node).cloned()
    }

    pub fn slot_valid(&self, slot: usize, ty: LayerType) -> bool {
        self.slots.get(slot).map(|entry| entry.valid & ty.bit_mask() != 0).unwrap_or(false)
    }

    pub fn resolution(&self, ty: LayerType) -> u32 {
        self.layers[ty].texture_resolution
    }
    pub fn resolution_blocks(&self, ty: LayerType) -> u32 {
        let resolution = self.layers[ty].texture_resolution ;
        let block_size = self.layers[ty].texture_format.block_size();
        assert_eq!(resolution % block_size, 0);
        resolution / block_size
    }
    pub fn bytes_per_block(&self, ty: LayerType) -> usize {
        self.layers[ty].texture_format.bytes_per_block()
    }
    pub fn row_pitch(&self, ty: LayerType) -> usize {
        let row_bytes = self.resolution_blocks(ty) as usize * self.bytes_per_block(ty);
        let row_pitch = (row_bytes + 255) & !255;
        row_pitch
    }

    pub fn border(&self, ty: LayerType) -> u32 {
        self.layers[ty].texture_border_size
    }

    // #[allow(unused)]
    // pub fn get_texel<'a>(
    //     &self,
    //     mapfile: &'a MapFile,
    //     node: VNode,
    //     ty: LayerType,
    //     x: usize,
    //     y: usize,
    // ) -> &'a [u8] {
    //     let tile = self.layers[ty].tile_indices[&node] as usize;
    //     let tile_data = &mapfile.read_tile(ty, node).unwrap();
    //     let border = self.border(ty) as usize;
    //     let index =
    //         ((x + border) + (y + border) * self.resolution(ty) as usize) * self.bytes_per_texel(ty);
    //     &tile_data[index..(index + self.bytes_per_texel(ty))]
    // }

    // pub fn capacity(&self) -> usize {
    //     self.size
    // }
    // pub fn utilization(&self) -> usize {
    //     self.slots.iter().filter(|s| s.priority >= Priority::cutoff() && s.valid != 0).count()
    // }
}
