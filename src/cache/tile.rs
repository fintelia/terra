use crate::{
    cache::{self, Priority, PriorityCacheEntry},
    terrain::quadtree::{QuadTree, VNode},
};
use crate::{
    coordinates,
    stream::{TileResult, TileStreamerEndpoint},
};
use crate::{
    generate::GenerateTile,
    gpu_state::GpuState,
    mapfile::{MapFile, TileState},
};
use cache::{LayerType, PriorityCache};
use cgmath::Vector3;
use futures::future::BoxFuture;
use futures::future::FutureExt;
use futures::stream::futures_unordered::FuturesUnordered;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::{num::NonZeroU32, sync::Arc};
use vec_map::VecMap;

use super::{GeneratorMask, LayerMask, UnifiedPriorityCache};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TextureFormat {
    R8,
    RG8,
    RGBA8,
    RGBA16F,
    R32,
    R32F,
    RG32F,
    RGBA32F,
    SRGBA,
    BC4,
    BC5,
    UASTC,
}
impl TextureFormat {
    /// Returns the number of bytes in a single texel of the format. Actually reports bytes per
    /// block for compressed formats.
    pub fn bytes_per_block(&self) -> usize {
        match *self {
            TextureFormat::R8 => 1,
            TextureFormat::RG8 => 2,
            TextureFormat::RGBA8 => 4,
            TextureFormat::RGBA16F => 8,
            TextureFormat::R32 => 4,
            TextureFormat::R32F => 4,
            TextureFormat::RG32F => 8,
            TextureFormat::RGBA32F => 16,
            TextureFormat::SRGBA => 4,
            TextureFormat::BC4 => 8,
            TextureFormat::BC5 => 16,
            TextureFormat::UASTC => 16,
        }
    }
    pub fn to_wgpu(&self, wgpu_features: wgpu::Features) -> wgpu::TextureFormat {
        match *self {
            TextureFormat::R8 => wgpu::TextureFormat::R8Unorm,
            TextureFormat::RG8 => wgpu::TextureFormat::Rg8Unorm,
            TextureFormat::RGBA8 => wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::RGBA16F => wgpu::TextureFormat::Rgba16Float,
            TextureFormat::R32 => wgpu::TextureFormat::R32Uint,
            TextureFormat::R32F => wgpu::TextureFormat::R32Float,
            TextureFormat::RG32F => wgpu::TextureFormat::Rg32Float,
            TextureFormat::RGBA32F => wgpu::TextureFormat::Rgba32Float,
            TextureFormat::SRGBA => wgpu::TextureFormat::Rgba8UnormSrgb,
            TextureFormat::BC4 => wgpu::TextureFormat::Bc4RUnorm,
            TextureFormat::BC5 => wgpu::TextureFormat::Bc5RgUnorm,
            TextureFormat::UASTC => {
                if wgpu_features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
                    wgpu::TextureFormat::Bc7RgbaUnorm
                } else if wgpu_features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC_LDR) {
                    wgpu::TextureFormat::Astc4x4RgbaUnorm
                } else {
                    unreachable!("Wgpu reports no texture compression support?")
                }
            }
        }
    }
    pub fn block_size(&self) -> u32 {
        match *self {
            TextureFormat::BC4 | TextureFormat::BC5 | TextureFormat::UASTC => 4,
            TextureFormat::R8
            | TextureFormat::RG8
            | TextureFormat::RGBA8
            | TextureFormat::RGBA16F
            | TextureFormat::R32F
            | TextureFormat::R32
            | TextureFormat::RG32F
            | TextureFormat::RGBA32F
            | TextureFormat::SRGBA => 1,
        }
    }
    pub fn is_compressed(&self) -> bool {
        match *self {
            TextureFormat::BC4 | TextureFormat::BC5 | TextureFormat::UASTC => true,
            TextureFormat::R8
            | TextureFormat::RG8
            | TextureFormat::RGBA8
            | TextureFormat::RGBA16F
            | TextureFormat::R32
            | TextureFormat::R32F
            | TextureFormat::RG32F
            | TextureFormat::RGBA32F
            | TextureFormat::SRGBA => false,
        }
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
    I16 { min: f32, max: f32, heights: Arc<Vec<i16>> },
    F32 { min: f32, max: f32, heights: Arc<Vec<f32>> },
}

pub(super) struct Entry {
    /// How imporant this entry is for the current frame.
    priority: Priority,
    /// The node this entry is for.
    node: VNode,
    /// bitmask of whether the tile for each layer is valid.
    pub(super) valid: LayerMask,
    /// bitmask of whether the tile for each layer was generated.
    generated: LayerMask,
    /// bitmask of whether the tile for each layer is currently being streamed.
    streaming: LayerMask,
    /// A CPU copy of the heightmap tile, useful for collision detection and such.
    heightmap: Option<CpuHeightmap>,
    /// Map from layer to the generators that were used (perhaps indirectly) to produce it.
    pub(super) generators: VecMap<GeneratorMask>,
}
impl Entry {
    fn new(node: VNode, priority: Priority) -> Self {
        Self {
            node,
            priority,
            valid: LayerMask::empty(),
            generated: LayerMask::empty(),
            streaming: LayerMask::empty(),
            heightmap: None,
            generators: VecMap::new(),
        }
    }
}
impl PriorityCacheEntry for Entry {
    type Key = VNode;
    fn priority(&self) -> Priority {
        self.priority
    }
    fn key(&self) -> VNode {
        self.node
    }
}

pub(crate) struct TileCache {
    pub(super) inner: PriorityCache<Entry>,
    pub(super) layers: VecMap<LayerParams>,
    pub(super) generators: Vec<Box<dyn GenerateTile>>,

    streamer: TileStreamerEndpoint,
    start_download:
        tokio::sync::mpsc::UnboundedSender<BoxFuture<'static, Result<(VNode, wgpu::Buffer), ()>>>,
    completed_downloads: crossbeam::channel::Receiver<(VNode, wgpu::Buffer, CpuHeightmap)>,
    free_download_buffers: Vec<wgpu::Buffer>,
    total_download_buffers: usize,
}
impl TileCache {
    pub fn new(mapfile: Arc<MapFile>, generators: Vec<Box<dyn GenerateTile>>, size: usize) -> Self {
        let layers = mapfile.layers().clone();

        let (start_tx, start_rx) = tokio::sync::mpsc::unbounded_channel();
        let (completed_tx, completed_rx) = crossbeam::channel::unbounded();

        let heightmap_resolution = layers[LayerType::Heightmaps].texture_resolution as usize;
        let heightmap_bytes_per_pixel =
            layers[LayerType::Heightmaps].texture_format.bytes_per_block() as usize;
        std::thread::spawn(move || {
            Self::download_thread(
                start_rx,
                completed_tx,
                heightmap_resolution,
                heightmap_bytes_per_pixel,
            )
        });

        Self {
            inner: PriorityCache::new(size),
            layers,
            streamer: TileStreamerEndpoint::new(mapfile).unwrap(),
            generators,
            start_download: start_tx,
            completed_downloads: completed_rx,
            free_download_buffers: Vec::new(),
            total_download_buffers: 0,
        }
    }

    pub(super) fn update(&mut self, quadtree: &QuadTree) {
        // Update priorities
        for entry in self.inner.slots_mut() {
            entry.priority = quadtree.node_priority(entry.node);
        }
        let min_priority =
            self.inner.slots().iter().map(|s| s.priority).min().unwrap_or(Priority::none());

        // Find any tiles that may need to be added.
        let mut missing = Vec::new();
        VNode::breadth_first(|node| {
            let priority = quadtree.node_priority(node);
            if priority < Priority::cutoff() {
                return false;
            }
            if !self.inner.contains(&node) && (priority > min_priority || !self.inner.is_full()) {
                missing.push(Entry::new(node, priority));
            }

            node.level() < VNode::LEVEL_CELL_5MM
        });
        self.inner.insert(missing);
    }

    pub(super) fn generate_tiles(
        cache: &mut UnifiedPriorityCache,
        mapfile: &MapFile,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_state: &GpuState,
    ) {
        let mut planned_heightmap_downloads = Vec::new();
        let mut pending_generate = VecMap::new();

        for layer in cache.tiles.layers.values() {
            let ty = layer.layer_type;

            // Figure out which entries need to be uploaded
            let pending_generate = pending_generate.entry(ty.index()).or_insert(Vec::new());

            for ref mut entry in cache.tiles.inner.slots_mut() {
                if (entry.valid | entry.streaming).intersects(ty.bit_mask()) {
                    continue;
                }

                match mapfile.tile_state(ty, entry.node).unwrap() {
                    TileState::GpuOnly => {
                        entry.generated |= ty.bit_mask();
                        pending_generate.push(entry.node);
                    }
                    TileState::MissingBase | TileState::Base => {
                        if cache.tiles.streamer.num_inflight() < 128 {
                            entry.streaming |= ty.bit_mask();
                            entry.generated &= !ty.bit_mask();
                            cache.tiles.streamer.request_tile(entry.node, ty);
                        }
                    }
                    TileState::Generated => {
                        if cache.tiles.streamer.num_inflight() < 128 {
                            entry.streaming |= ty.bit_mask();
                            entry.generated |= ty.bit_mask();
                            cache.tiles.streamer.request_tile(entry.node, ty);
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

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder.tiles.generate"),
        });

        let mut uniform_data = Vec::new();
        for (i, nodes) in &mut pending_generate {
            let layer = LayerType::from_index(i);
            for n in nodes {
                let entry = cache.tiles.inner.entry(&n).unwrap();
                let parent_entry =
                    if let Some(p) = n.parent() { cache.tiles.inner.entry(&p.0) } else { None };

                if entry.valid.contains_tile(layer) {
                    continue;
                }

                for (generator_index, generator) in cache.tiles.generators.iter_mut().enumerate() {
                    let outputs = generator.outputs(n.level());

                    let peer_inputs = generator.peer_inputs(n.level());
                    let parent_inputs = generator.parent_inputs(n.level());

                    let generates_layer = outputs.contains_tile(layer);
                    let has_peer_inputs = peer_inputs & !entry.valid == LayerMask::empty();
                    let root_input_missing =
                        n.level() == 0 && generator.parent_inputs(0) != LayerMask::empty();
                    let parent_input_missing = n.level() > 0
                        && (parent_entry.is_none()
                            || parent_inputs & !parent_entry.as_ref().unwrap().valid
                                != LayerMask::empty());
                    let missing_download_buffers = outputs.contains_tile(LayerType::Heightmaps)
                        && cache.tiles.free_download_buffers.is_empty()
                        && cache.tiles.total_download_buffers == 64;

                    if generates_layer
                        && has_peer_inputs
                        && !root_input_missing
                        && !parent_input_missing
                        && !missing_download_buffers
                    {
                        let slot = cache.tiles.inner.index_of(&n).unwrap();
                        let parent_slot = if let Some(p) = n.parent() {
                            cache.tiles.inner.index_of(&p.0)
                        } else {
                            None
                        };

                        let output_mask = !entry.valid & generator.outputs(n.level());
                        generator.generate(
                            device,
                            &mut encoder,
                            gpu_state,
                            &cache.tiles.layers,
                            *n,
                            slot,
                            parent_slot,
                            output_mask,
                            &mut uniform_data,
                        );

                        let mut input_generators = GeneratorMask::from_index(generator_index);
                        input_generators |= cache.generator_dependencies(*n, peer_inputs);
                        if parent_entry.is_some() {
                            input_generators |=
                                cache.generator_dependencies(n.parent().unwrap().0, parent_inputs);
                        }

                        let entry = cache.tiles.inner.entry_mut(&n).unwrap();
                        entry.valid |= output_mask;
                        entry.generated |= output_mask;
                        for layer in
                            LayerType::iter().filter(|&layer| output_mask.contains_tile(layer))
                        {
                            entry.generators.insert(layer.index(), input_generators);
                        }

                        if output_mask.contains_tile(LayerType::Heightmaps)
                            && n.level() <= VNode::LEVEL_CELL_1M
                        {
                            let bytes_per_pixel = cache.tiles.layers[LayerType::Heightmaps]
                                .texture_format
                                .bytes_per_block()
                                as u64;
                            let resolution =
                                cache.tiles.layers[LayerType::Heightmaps].texture_resolution as u64;
                            let row_bytes = resolution * bytes_per_pixel;
                            let row_pitch = (row_bytes + 255) & !255;

                            let buffer =
                                cache.tiles.free_download_buffers.pop().unwrap_or_else(|| {
                                    cache.tiles.total_download_buffers += 1;
                                    device.create_buffer(&wgpu::BufferDescriptor {
                                        size: row_pitch * resolution,
                                        usage: wgpu::BufferUsage::COPY_DST
                                            | wgpu::BufferUsage::MAP_READ,
                                        label: Some(&format!("buffer.tiles.download")),
                                        mapped_at_creation: false,
                                    })
                                });
                            encoder.copy_texture_to_buffer(
                                wgpu::ImageCopyTexture {
                                    texture: &gpu_state.tile_cache[LayerType::Heightmaps].0,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d { x: 0, y: 0, z: slot as u32 },
                                },
                                wgpu::ImageCopyBuffer {
                                    buffer: &buffer,
                                    layout: wgpu::ImageDataLayout {
                                        offset: 0,
                                        bytes_per_row: Some(
                                            NonZeroU32::new(row_pitch as u32).unwrap(),
                                        ),
                                        rows_per_image: None,
                                    },
                                },
                                wgpu::Extent3d {
                                    width: resolution as u32,
                                    height: resolution as u32,
                                    depth_or_array_layers: 1,
                                },
                            );

                            planned_heightmap_downloads.push((*n, buffer));
                        }

                        break;
                    }
                }
            }
        }
        queue.write_buffer(&gpu_state.generate_uniforms, 0, &uniform_data);
        queue.submit(Some(encoder.finish()));

        for (n, buffer) in planned_heightmap_downloads.drain(..) {
            let _ = cache.tiles.start_download.send(
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
    }

    pub(super) fn upload_tiles(&mut self, queue: &wgpu::Queue, textures: &VecMap<(wgpu::Texture, wgpu::TextureView)>) {
        while let Some(mut tile) = self.streamer.try_complete() {
            if let Some(entry) = self.inner.entry_mut(&tile.node()) {
                entry.valid |= tile.layer().bit_mask();
                entry.streaming &= !tile.layer().bit_mask();

                let index = self.inner.index_of(&tile.node()).unwrap();
                let layer = tile.layer();

                let resolution = self.resolution(tile.layer()) as usize;
                let resolution_blocks = self.resolution_blocks(tile.layer()) as usize;
                let bytes_per_block = self.layers[tile.layer()].texture_format.bytes_per_block();
                let row_bytes = resolution_blocks * bytes_per_block;

                let data;
                let mut height_data;
                match tile {
                    TileResult::Heightmaps(node, ref heights) => {
                        if let Some(entry) = self.inner.entry_mut(&node) {
                            let min = *heights.iter().min().unwrap() as f32;
                            let max = *heights.iter().max().unwrap() as f32;
                            entry.heightmap =
                                Some(CpuHeightmap::I16 { min, max, heights: Arc::clone(&heights) });
                        }
                        let heights: Vec<_> = heights
                            .iter()
                            .map(|&h| {
                                if h < 0 {
                                    0x800000 | (((h + 1024).max(0) as u32) << 9)
                                } else {
                                    (((h as u32) + 1024) << 9).min(0x7fffff)
                                }
                            })
                            .collect();
                        height_data = vec![0; heights.len() * 4];
                        height_data.copy_from_slice(bytemuck::cast_slice(&heights));
                        data = &mut height_data;
                    }
                    TileResult::Albedo(_, ref mut d) | TileResult::Roughness(_, ref mut d) => {
                        data = &mut *d
                    }
                }

                if cfg!(feature = "small-trace") {
                    for y in 0..resolution_blocks {
                        for x in 0..resolution_blocks {
                            if x % 16 == 0 && y % 16 == 0 {
                                continue;
                            }
                            let src = ((x & !15) + (y & !15) * resolution_blocks) * bytes_per_block;
                            let dst = (x + y * resolution_blocks) * bytes_per_block;
                            data.copy_within(src..src + bytes_per_block, dst);
                        }
                    }
                }

                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &textures[layer].0,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: index as u32 },
                    },
                    &*data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(NonZeroU32::new(row_bytes as u32).unwrap()),
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: resolution as u32,
                        height: resolution as u32,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }

    pub(super) fn download_tiles(&mut self) {
        while let Ok((node, buffer, heightmap)) = self.completed_downloads.try_recv() {
            if let Some(entry) = self.inner.entry_mut(&node) {
                self.free_download_buffers.push(buffer);
                entry.heightmap = Some(heightmap);
            }
        }
    }

    fn download_thread(
        start_rx: tokio::sync::mpsc::UnboundedReceiver<
            BoxFuture<'static, Result<(VNode, wgpu::Buffer), ()>>,
        >,
        completed_tx: crossbeam::channel::Sender<(VNode, wgpu::Buffer, CpuHeightmap)>,
        heightmap_resolution: usize,
        heightmap_bytes_per_pixel: usize,
    ) {
        let heightmap_row_bytes = heightmap_resolution * heightmap_bytes_per_pixel;
        let heightmap_row_pitch = (heightmap_row_bytes + 255) & !255;

        tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async move {
            let mut pending_heightmap_downloads = FuturesUnordered::new();
            let mut start_rx = tokio_stream::wrappers::UnboundedReceiverStream::new(start_rx).fuse();
            loop {
                futures::select! {
                    n = start_rx.select_next_some() => {
                        pending_heightmap_downloads.push(n);
                    }
                    h = pending_heightmap_downloads.select_next_some() => {
                        if let Ok((node, buffer)) = h {
                            let mut heights = vec![0u32; heightmap_resolution * heightmap_resolution];
                            {
                                let mapped_buffer = buffer.slice(..).get_mapped_range();
                                for (h, b) in heights.chunks_exact_mut(heightmap_resolution).zip(mapped_buffer.chunks_exact(heightmap_row_pitch)) {
                                    bytemuck::cast_slice_mut(h).copy_from_slice(&b[..heightmap_row_bytes]);
                                }
                            }
                            buffer.unmap();

                            let heights: Vec<f32> =
                                heights.into_iter().map(|h| ((h & 0x7fffff) as f32 / 512.0) - 1024.0).collect();
                            let (mut min, mut max) = (f32::MAX, 0.0);
                            for &h in &heights {
                                if min < h {
                                    min = h;
                                }
                                if max > h {
                                    max = h;
                                }
                            }

                            let _ = completed_tx.send((node, buffer, CpuHeightmap::F32 { min, max, heights: Arc::new(heights) }));
                        }
                    }
                    complete => break,
                }
            }
        });
    }

    pub(super) fn make_cache_textures(
        &self,
        device: &wgpu::Device,
    ) -> VecMap<(wgpu::Texture, wgpu::TextureView)> {
        self.layers
            .iter()
            .map(|(ty, layer)| {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    size: wgpu::Extent3d {
                        width: layer.texture_resolution,
                        height: layer.texture_resolution,
                        depth_or_array_layers: self.inner.size() as u32,
                    },
                    format: layer.texture_format.to_wgpu(device.features()),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    usage: wgpu::TextureUsage::COPY_SRC
                        | wgpu::TextureUsage::COPY_DST
                        | wgpu::TextureUsage::SAMPLED
                        | if !layer.texture_format.is_compressed() {
                            wgpu::TextureUsage::STORAGE
                        } else {
                            wgpu::TextureUsage::empty()
                        },
                    label: Some(&format!("texture.tiles.{}", LayerType::from_index(ty).name())),
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!(
                        "texture.tiles.{}.view",
                        LayerType::from_index(ty).name()
                    )),
                    ..Default::default()
                });
                (ty, (texture, view))
            })
            .collect()
    }

    pub fn contains(&self, node: VNode, ty: LayerType) -> bool {
        self.inner.entry(&node).map(|entry| entry.valid.contains_tile(ty)).unwrap_or(false)
    }
    pub fn contains_all(&self, node: VNode, layer_mask: LayerMask) -> bool {
        self.inner
            .entry(&node)
            .map(|entry| (entry.valid & layer_mask) == layer_mask)
            .unwrap_or(false)
    }

    pub fn get_slot(&self, node: VNode) -> Option<usize> {
        self.inner.index_of(&node)
    }

    fn resolution(&self, ty: LayerType) -> u32 {
        self.layers[ty].texture_resolution
    }
    fn resolution_blocks(&self, ty: LayerType) -> u32 {
        let resolution = self.layers[ty].texture_resolution;
        let block_size = self.layers[ty].texture_format.block_size();
        assert_eq!(resolution % block_size, 0);
        resolution / block_size
    }

    pub fn get_height(&self, latitude: f64, longitude: f64, level: u8) -> Option<f32> {
        let ecef = coordinates::polar_to_ecef(Vector3::new(latitude, longitude, 0.0));
        let cspace = ecef / ecef.x.abs().max(ecef.y.abs()).max(ecef.z.abs());

        let (node, x, y) = VNode::from_cspace(cspace, level);

        let border = self.layers[LayerType::Heightmaps].texture_border_size as usize;
        let resolution = self.layers[LayerType::Heightmaps].texture_resolution as usize;
        let x = (x * (resolution - 2 * border - 1) as f32) + border as f32;
        let y = (y * (resolution - 2 * border - 1) as f32) + border as f32;

        let w00 = (1.0 - x.fract()) * (1.0 - y.fract());
        let w10 = x.fract() * (1.0 - y.fract());
        let w01 = (1.0 - x.fract()) * y.fract();
        let w11 = x.fract() * y.fract();

        let i00 = x.floor() as usize + y.floor() as usize * resolution;
        let i10 = x.ceil() as usize + y.floor() as usize * resolution;
        let i01 = x.floor() as usize + y.ceil() as usize * resolution;
        let i11 = x.ceil() as usize + y.ceil() as usize * resolution;

        self.inner.entry(&node).and_then(|entry| Some(entry.heightmap.as_ref()?)).map(|h| match h {
            CpuHeightmap::I16 { heights: h, .. } => (h[i00] as f32 * w00
                + h[i10] as f32 * w10
                + h[i01] as f32 * w01
                + h[i11] as f32 * w11)
                .max(0.0),
            CpuHeightmap::F32 { heights: h, .. } => {
                (h[i00] * w00 + h[i10] * w10 + h[i01] * w01 + h[i11] * w11).max(0.0)
            }
        })
    }

    /// Returns a conservative estimate of the minimum and maximum heights in the given node.
    pub fn get_height_range(&self, node: VNode) -> (f32, f32) {
        let mut node = Some(node);
        while let Some(n) = node {
            if let Some(CpuHeightmap::I16 { min, max, .. } | CpuHeightmap::F32 { min, max, .. }) =
                self.inner.entry(&n).and_then(|entry| Some(entry.heightmap.as_ref()?))
            {
                return (min.min(0.0), *max + 6000.0);
            }
            node = n.parent().map(|p| p.0);
        }
        (0.0, 9000.0)
    }
}
