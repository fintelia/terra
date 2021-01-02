use crate::terrain::quadtree::VNode;
use crate::{
    coordinates,
    stream::{TileResult, TileStreamerEndpoint},
};
use crate::{
    generate::GenerateTile,
    gpu_state::GpuState,
    mapfile::{MapFile, TileState},
};
use cgmath::Vector3;
use futures::stream::futures_unordered::FuturesUnordered;
use futures::StreamExt;
use futures::{future::BoxFuture, select};
use futures::{future::FutureExt, TryFutureExt};
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};
use std::sync::Arc;
use std::{collections::HashMap, num::NonZeroU32};
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

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    #[allow(unused)]
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
    /// Maximum number of tiles for this layer to generate in a single frame.
    pub tiles_generated_per_frame: usize,
}

enum CpuHeightmap {
    I16(Arc<Vec<i16>>),
    F32(Arc<Vec<f32>>),
}

struct Entry {
    /// How imporant this entry is for the current frame.
    priority: Priority,
    /// The node this entry is for.
    node: VNode,
    /// bitmask of whether the tile for each layer is valid.
    valid: u32,
    /// bitmask of whether the tile for each layer was generated.
    generated: u32,
    /// bitmask of whether the tile for each layer is currently being streamed.
    streaming: u32,
    /// A CPU copy of the heightmap tile, useful for collision detection and such.
    heightmap: Option<CpuHeightmap>,
    /// Map from layer to the generators that were used (perhaps indirectly) to produce it.
    generators: VecMap<NonZeroU32>,
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

    streamer: TileStreamerEndpoint,
    generators: Vec<Box<dyn GenerateTile>>,

    planned_heightmap_downloads: Vec<(VNode, wgpu::Buffer)>,
    pending_heightmap_downloads:
        FuturesUnordered<BoxFuture<'static, Result<(VNode, wgpu::Buffer), ()>>>,
}
impl TileCache {
    pub fn new(mapfile: Arc<MapFile>, generators: Vec<Box<dyn GenerateTile>>, size: usize) -> Self {
        Self {
            size,
            slots: Vec::new(),
            reverse: HashMap::new(),
            missing: Vec::new(),
            min_priority: Priority::none(),
            layers: mapfile.layers().clone(),
            streamer: TileStreamerEndpoint::new(mapfile).unwrap(),
            generators,
            planned_heightmap_downloads: Vec::new(),
            pending_heightmap_downloads: FuturesUnordered::new(),
        }
    }

    pub fn update_priorities(&mut self, camera: Vector3<f64>) {
        for entry in &mut self.slots {
            entry.priority = entry.node.priority(camera);
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
            self.slots.push(Entry {
                priority: m.0,
                node: m.1,
                valid: 0,
                generated: 0,
                streaming: 0,
                heightmap: None,
                generators: VecMap::new(),
            });
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
                self.slots[index] = Entry {
                    priority: m.0,
                    node: m.1,
                    valid: 0,
                    generated: 0,
                    streaming: 0,
                    heightmap: None,
                    generators: VecMap::new(),
                };
                index += 1;
                if index == self.slots.len() {
                    break;
                }
            }
        }
        self.missing.clear();
    }

    pub(crate) fn refresh_tile_generators(&mut self) {
        for (i, gen) in self.generators.iter_mut().enumerate() {
            if gen.needs_refresh() {
                assert!(i < 32);
                let mask = 1u32 << i;
                for slot in self.slots.iter_mut() {
                    for (layer, generator_mask) in &slot.generators {
                        if (generator_mask.get() & mask) != 0 {
                            slot.valid &= !(1 << layer);
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn generate_tiles(
        &mut self,
        mapfile: &MapFile,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gpu_state: &GpuState,
    ) {
        self.process_missing();

        let mut pending_generate = VecMap::new();

        for layer in self.layers.values() {
            let ty = layer.layer_type;

            // Figure out which entries need to be uploaded
            let pending_generate = pending_generate.entry(ty.index()).or_insert(Vec::new());

            for ref mut entry in self.slots.iter_mut() {
                if (entry.valid | entry.streaming) & ty.bit_mask() != 0 {
                    continue;
                }

                match mapfile.tile_state(ty, entry.node).unwrap() {
                    TileState::GpuOnly => {
                        entry.generated |= ty.bit_mask();
                        pending_generate.push(entry.node);
                    }
                    TileState::MissingBase | TileState::Base => {
                        if self.streamer.num_inflight() < 128 {
                            entry.streaming |= ty.bit_mask();
                            entry.generated &= !ty.bit_mask();
                            self.streamer.request_tile(entry.node, ty);
                        }
                    }
                    TileState::Generated => {
                        if self.streamer.num_inflight() < 128 {
                            entry.streaming |= ty.bit_mask();
                            entry.generated |= ty.bit_mask();
                            self.streamer.request_tile(entry.node, ty);
                        }
                    }
                    TileState::Missing => {
                        entry.generated |= ty.bit_mask();
                        pending_generate.push(entry.node);
                    }
                }
            }

            pending_generate.sort_by_key(|n| n.level());
            if pending_generate.len() > layer.tiles_generated_per_frame {
                let _ = pending_generate.split_off(layer.tiles_generated_per_frame);
            }
        }

        for (i, nodes) in &mut pending_generate {
            for n in nodes {
                let slot = self.reverse[n];
                if self.slots[slot].valid & (1 << i) != 0 {
                    continue;
                }

                let parent_slot = n.parent().and_then(|(p, _)| self.reverse.get(&p).copied());

                for (generator_index, generator) in self.generators.iter_mut().enumerate() {
                    let outputs = generator.outputs(n.level());

                    let peer_inputs = generator.peer_inputs(n.level());
                    let parent_inputs = generator.parent_inputs(n.level());

                    let generates_layer = outputs & (1 << i) != 0;
                    let has_peer_inputs = peer_inputs & !self.slots[slot].valid == 0;
                    let root_input_missing = n.level() == 0 && generator.parent_inputs(0) != 0;
                    let parent_input_missing = n.level() > 0
                        && (parent_slot.is_none()
                            || parent_inputs & !self.slots[*parent_slot.as_ref().unwrap()].valid
                                != 0);

                    if generates_layer
                        && has_peer_inputs
                        && !root_input_missing
                        && !parent_input_missing
                    {
                        let output_mask = !self.slots[slot].valid & generator.outputs(n.level());
                        generator.generate(
                            device,
                            encoder,
                            gpu_state,
                            &self.layers,
                            *n,
                            slot,
                            parent_slot,
                            output_mask,
                        );

                        self.slots[slot].valid |= output_mask;
                        self.slots[slot].generated |= output_mask;

                        let mut input_generators = 1 << generator_index;
                        for j in (0..32).filter(|j| peer_inputs & (1 << j) != 0) {
                            input_generators |= self.slots[slot]
                                .generators
                                .get(j)
                                .copied()
                                .map(NonZeroU32::get)
                                .unwrap_or(0);
                        }
                        for j in (0..32).filter(|j| parent_inputs & (1 << j) != 0) {
                            input_generators |= self.slots[*parent_slot.as_ref().unwrap()]
                                .generators
                                .get(j)
                                .copied()
                                .map(NonZeroU32::get)
                                .unwrap_or(0);
                        }
                        for j in (0..32).filter(|j| output_mask & (1 << j) != 0) {
                            self.slots[slot]
                                .generators
                                .insert(j, NonZeroU32::new(input_generators).unwrap());
                        }

                        if output_mask & LayerType::Heightmaps.bit_mask() != 0
                            && n.level() <= crate::generate::TILE_CELL_1M
                        {
                            let bytes_per_pixel =
                                self.layers[LayerType::Heightmaps].texture_format.bytes_per_block()
                                    as u64;
                            let resolution =
                                self.layers[LayerType::Heightmaps].texture_resolution as u64;
                            let row_bytes = resolution * bytes_per_pixel;
                            let row_pitch = (row_bytes + 255) & !255;

                            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                                size: row_pitch * resolution,
                                usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
                                label: None,
                                mapped_at_creation: false,
                            });
                            encoder.copy_texture_to_buffer(
                                wgpu::TextureCopyView {
                                    texture: &gpu_state.tile_cache[LayerType::Heightmaps],
                                    mip_level: 0,
                                    origin: wgpu::Origin3d { x: 0, y: 0, z: slot as u32 },
                                },
                                wgpu::BufferCopyView {
                                    buffer: &buffer,
                                    layout: wgpu::TextureDataLayout {
                                        offset: 0,
                                        bytes_per_row: row_pitch as u32,
                                        rows_per_image: 0,
                                    },
                                },
                                wgpu::Extent3d {
                                    width: resolution as u32,
                                    height: resolution as u32,
                                    depth: 1,
                                },
                            );

                            self.planned_heightmap_downloads.push((*n, buffer));
                        }

                        break;
                    }
                }
            }
        }
    }

    pub(crate) fn upload_tiles(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        textures: &VecMap<wgpu::Texture>,
    ) {
        let mut buffer_size = 0;
        let mut pending_uploads = Vec::new();
        while let Some(tile) = self.streamer.try_complete() {
            let resolution_blocks = self.resolution_blocks(tile.layer()) as usize;
            let bytes_per_block = self.layers[tile.layer()].texture_format.bytes_per_block();
            let row_bytes = resolution_blocks * bytes_per_block;
            let row_pitch = (row_bytes + 255) & !255;
            buffer_size += row_pitch * resolution_blocks;
            pending_uploads.push(tile);
        }

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: buffer_size as u64,
            usage: wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::MAP_WRITE,
            label: None,
            mapped_at_creation: true,
        });
        let mut buffer_view = buffer.slice(..).get_mapped_range_mut();

        let mut i = 0;
        for upload in &pending_uploads {
            let resolution_blocks = self.resolution_blocks(upload.layer()) as usize;
            let bytes_per_block = self.layers[upload.layer()].texture_format.bytes_per_block();
            let row_bytes = resolution_blocks * bytes_per_block;
            let row_pitch = (row_bytes + 255) & !255;

            let data;
            let mut height_data;
            match upload {
                TileResult::Heightmaps(node, heights) => {
                    if let Some(slot) = self.reverse.get(node) {
                        self.slots[*slot].heightmap = Some(CpuHeightmap::I16(Arc::clone(&heights)));
                    }
                    let heights: Vec<_> = heights.iter().map(|&h| h as f32).collect();
                    height_data = vec![0; heights.len() * 4];
                    height_data.copy_from_slice(bytemuck::cast_slice(&heights));
                    data = &height_data;
                }
                TileResult::Albedo(_, ref d) | TileResult::Roughness(_, ref d) => data = &*d,
            }

            for row in 0..resolution_blocks {
                buffer_view[i..][..row_bytes]
                    .copy_from_slice(&data[row * row_bytes..][..row_bytes]);
                i += row_pitch;
            }
        }

        drop(buffer_view);
        buffer.unmap();

        let mut offset = 0;
        for tile in pending_uploads.drain(..) {
            let resolution = self.resolution(tile.layer()) as usize;
            let resolution_blocks = self.resolution_blocks(tile.layer()) as usize;
            let bytes_per_block = self.layers[tile.layer()].texture_format.bytes_per_block();
            let row_bytes = resolution_blocks * bytes_per_block;
            let row_pitch = (row_bytes + 255) & !255;
            if let Some(&slot) = self.reverse.get(&tile.node()) {
                encoder.copy_buffer_to_texture(
                    wgpu::BufferCopyView {
                        buffer: &buffer,
                        layout: wgpu::TextureDataLayout {
                            offset,
                            bytes_per_row: row_pitch as u32,
                            rows_per_image: 0,
                        },
                    },
                    wgpu::TextureCopyView {
                        texture: &textures[tile.layer()],
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: slot as u32 },
                    },
                    wgpu::Extent3d {
                        width: resolution as u32,
                        height: resolution as u32,
                        depth: 1,
                    },
                );
                self.slots[slot].valid |= tile.layer().bit_mask();
                self.slots[slot].streaming &= !tile.layer().bit_mask();
            }
            offset += (row_pitch * resolution_blocks) as u64
        }
    }

    pub(crate) fn download_tiles(&mut self) {
        for (n, buffer) in self.planned_heightmap_downloads.drain(..) {
            self.pending_heightmap_downloads.push(
                buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read)
                    .then(move |result| {
                        futures::future::ready(match result {
                            Ok(()) => Ok((n, buffer)),
                            Err(_) => Err(()),
                        })
                    })
                    .boxed(),
            );
        }

        loop {
            futures::select! {
                h = self.pending_heightmap_downloads.select_next_some() => {
                    if let Ok((node, buffer)) = h {
                        if let Some(slot) = self.reverse.get(&node) {
                            let bytes_per_pixel =
                            self.layers[LayerType::Heightmaps].texture_format.bytes_per_block()
                                as usize;
                            let resolution =
                                self.layers[LayerType::Heightmaps].texture_resolution as usize;
                            let row_bytes = resolution * bytes_per_pixel;
                            let row_pitch = (row_bytes + 255) & !255;
                            let mut heights = vec![0.0; resolution * resolution];

                            let t = std::time::Instant::now();
                            {
                                let mapped_buffer = buffer.slice(..).get_mapped_range();
                                for (h, b) in heights.chunks_exact_mut(resolution).zip(mapped_buffer.chunks_exact(row_pitch)) {
                                    bytemuck::cast_slice_mut(h).copy_from_slice(&b[..row_bytes]);
                                }
                            }
                            println!("{:.1}ms {}KB", t.elapsed().as_secs_f32() * 1000.0, heights.len() >> 8);

                            buffer.unmap();

                            self.slots[*slot].heightmap = Some(CpuHeightmap::F32(Arc::new(heights)));
                        }
                    }
                }
                default => break,
                complete => break,
            }
        }
    }

    pub fn make_cache_textures(&self, device: &wgpu::Device) -> VecMap<wgpu::Texture> {
        self.layers
            .iter()
            .map(|(ty, layer)| {
                let usage = match layer.texture_format {
                    TextureFormat::BC4 | TextureFormat::BC5 => {
                        wgpu::TextureUsage::COPY_SRC
                            | wgpu::TextureUsage::COPY_DST
                            | wgpu::TextureUsage::SAMPLED
                    }
                    _ => {
                        wgpu::TextureUsage::COPY_SRC
                            | wgpu::TextureUsage::COPY_DST
                            | wgpu::TextureUsage::SAMPLED
                            | wgpu::TextureUsage::STORAGE
                    }
                };

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
                        usage,
                        label: None,
                    }),
                )
            })
            .collect()
    }

    pub fn contains(&self, node: VNode, ty: LayerType) -> bool {
        self.reverse
            .get(&node)
            .and_then(|&slot| self.slots.get(slot))
            .map(|entry| (entry.valid & ty.bit_mask()) != 0)
            .unwrap_or(false)
    }

    pub fn get_slot(&self, node: VNode) -> Option<usize> {
        self.reverse.get(&node).cloned()
    }

    pub fn resolution(&self, ty: LayerType) -> u32 {
        self.layers[ty].texture_resolution
    }
    pub fn resolution_blocks(&self, ty: LayerType) -> u32 {
        let resolution = self.layers[ty].texture_resolution;
        let block_size = self.layers[ty].texture_format.block_size();
        assert_eq!(resolution % block_size, 0);
        resolution / block_size
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

    pub fn get_height(&self, latitude: f64, longitude: f64, level: u8) -> Option<f32> {
        let ecef = coordinates::polar_to_ecef(Vector3::new(latitude, longitude, 0.0));
        let cspace = ecef / ecef.x.abs().max(ecef.y.abs()).max(ecef.z.abs());

        let (node, x, y) = VNode::from_cspace(cspace, level);

        let border = self.layers[LayerType::Heightmaps].texture_border_size as usize;
        let resolution = self.layers[LayerType::Heightmaps].texture_resolution as usize;
        let x = (x * (resolution - 2 * border - 1) as f32) + border as f32;
        let y = (y * (resolution - 2 * border - 1) as f32) + border as f32;
        let index = x.round() as usize + y.round() as usize * resolution;

        // TODO: bilinear interpolate height.
        self.reverse.get(&node).and_then(|&slot| self.slots[slot].heightmap.as_ref()).map(|h| {
            match h {
                CpuHeightmap::I16(h) => h[index].max(0) as f32,
                CpuHeightmap::F32(h) => h[index].max(0.0),
            }
        })
    }
}
