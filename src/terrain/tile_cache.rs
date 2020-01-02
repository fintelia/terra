use crate::coordinates::CoordinateSystem;
use crate::mapfile::{MapFile, TileState};
use crate::terrain::quadtree::{Node, NodeId};
use cgmath::Point3;
use memmap::Mmap;
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};
use std::sync::Arc;
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
}
impl TextureFormat {
    pub fn bytes_per_texel(&self) -> usize {
        match *self {
            TextureFormat::R8 => 1,
            TextureFormat::RG8 => 2,
            TextureFormat::RGBA8 => 4,
            TextureFormat::R32F => 4,
            TextureFormat::RG32F => 8,
            TextureFormat::RGBA32F => 16,
            TextureFormat::SRGBA => 4,
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
impl Index<LayerType> for VecMap<TileCache> {
    type Output = TileCache;
    fn index(&self, i: LayerType) -> &Self::Output {
        &self[i as usize]
    }
}
impl IndexMut<LayerType> for VecMap<TileCache> {
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
    // Array of bytes indicating whether each tile is valid.
    pub tile_valid_bitmap: ByteRange,
    /// Where each tile is located in the file.
    pub tile_locations: Vec<ByteRange>,
    /// Number of samples in each dimension, per tile.
    pub texture_resolution: u32,
    /// Number of samples outside the tile on each side.
    pub texture_border_size: u32,
    /// Format used by this layer.
    pub texture_format: TextureFormat,
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
    pub layers: VecMap<LayerParams>,
    pub noise: NoiseParams,
    pub nodes: Vec<Node>,
    pub planet_mesh: MeshDescriptor,
    pub planet_mesh_texture: TextureDescriptor,
    pub base_heights: TextureDescriptor,
    pub system: CoordinateSystem,
}

struct Entry {
    priority: Priority,
    id: NodeId,
    valid: bool,
	generated: bool,
}

pub(crate) struct TileCache {
    size: usize,
    slots: Vec<Entry>,
    reverse: VecMap<usize>,

    /// Nodes that should be added to the cache.
    missing: Vec<(Priority, NodeId)>,
    /// Smallest priority among all nodes in the cache.
    min_priority: Priority,

    /// Resolution of each tile in this cache.
    layer_params: LayerParams,
}
impl TileCache {
    pub fn new(params: LayerParams) -> Self {
        let size = params.layer_type.cache_size() as usize;

        Self {
            size,
            slots: Vec::new(),
            reverse: VecMap::new(),
            missing: Vec::new(),
            min_priority: Priority::none(),
            layer_params: params,
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

    fn process_missing(&mut self) {
        // Find slots for missing entries.
        self.missing.sort();
        while !self.missing.is_empty() && self.slots.len() < self.size {
            let m = self.missing.pop().unwrap();
            self.reverse.insert(m.1.index(), self.slots.len());
            self.slots.push(Entry { priority: m.0, id: m.1, valid: false, generated: false });
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
                self.slots[index] = Entry { priority: m.0, id: m.1, valid: false, generated: false };
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
        nodes: &Vec<Node>,
        mapfile: &mut MapFile,
    ) -> Vec<NodeId> {
        self.process_missing();

        let ty = self.layer_params.layer_type;

        // Figure out which entries need to be uploaded
        let mut pending_uploads = Vec::new();
        let mut pending_generate = Vec::new();

        for (i, ref mut entry) in self.slots.iter_mut().enumerate() {
            if entry.valid {
                continue;
            }

            let tile = nodes[entry.id].tile_indices[ty.index()].unwrap() as usize;

            match mapfile.tile_state(ty, tile) {
                TileState::Base => {
					entry.generated = false;
					pending_uploads.push((i, tile));
				}
				TileState::Generated => {
					entry.generated = true;
					pending_uploads.push((i, tile));
				}
                TileState::Missing => pending_generate.push(entry.id),
            }
        }
        if pending_uploads.is_empty() {
            return pending_generate;
        }

        let resolution = self.resolution() as usize;
        let bytes_per_texel = self.layer_params.texture_format.bytes_per_texel();
        let row_bytes = resolution * bytes_per_texel;
        let row_pitch = (row_bytes + 255) & !255;
        let tiles = pending_uploads.len();

        let buffer = device.create_buffer_mapped(
            row_pitch * resolution * tiles,
            wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::MAP_WRITE,
        );

        let mut i = 0;
        for upload in &pending_uploads {
            let data = mapfile.read_tile(ty, upload.1).unwrap();
            for row in 0..resolution {
                buffer.data[i..][..row_bytes]
                    .copy_from_slice(&data[row * row_bytes..][..row_bytes]);
                i += row_pitch;
            }
        }

        let buffer = buffer.finish();

        for (index, (slot, _)) in pending_uploads.drain(..).enumerate() {
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

        pending_generate
    }

    pub fn make_cache_texture(&self, device: &wgpu::Device) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: self.layer_params.texture_resolution,
                height: self.layer_params.texture_resolution,
                depth: 1,
            },
            format: self.layer_params.texture_format.to_wgpu(),
            array_layer_count: self.size as u32,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::STORAGE,
        })
    }

	pub fn clear_generated(&mut self) {
		for i in 0..self.slots.len() {
			if self.slots[i].generated {
				self.slots[i].valid = false;
			}
		}
	}

    pub fn contains(&self, id: NodeId) -> bool {
        self.reverse
            .get(id.index())
            .and_then(|&slot| self.slots.get(slot))
            .map(|entry| entry.valid)
            .unwrap_or(false)
    }

    pub fn get_slot(&self, id: NodeId) -> Option<usize> {
        self.reverse.get(id.index()).cloned()
    }

    pub fn resolution(&self) -> u32 {
        self.layer_params.texture_resolution
    }
    pub fn bytes_per_texel(&self) -> usize {
        self.layer_params.texture_format.bytes_per_texel()
    }
    pub fn row_pitch(&self) -> usize {
        let row_bytes = self.resolution() as usize * self.bytes_per_texel();
        let row_pitch = (row_bytes + 255) & !255;
        row_pitch
    }

    pub fn border(&self) -> u32 {
        self.layer_params.texture_border_size
    }

    pub fn get_texel<'a>(&self, mapfile: &'a MapFile, node: &Node, x: usize, y: usize) -> &'a [u8] {
        let ty = self.layer_params.layer_type;
        let tile = node.tile_indices[ty.index()].unwrap() as usize;
        let tile_data = &mapfile.read_tile(ty, tile).unwrap();
        let bytes_per_texel = self.layer_params.texture_format.bytes_per_texel();
        let border = self.layer_params.texture_border_size as usize;
        let index = ((x + border) + (y + border) * self.layer_params.texture_resolution as usize)
            * bytes_per_texel;
        &tile_data[index..(index + bytes_per_texel)]
    }

    pub fn capacity(&self) -> usize {
        self.size
    }
    pub fn utilization(&self) -> usize {
        self.slots.iter().filter(|s| s.priority >= Priority::cutoff() && s.valid).count()
    }
}
