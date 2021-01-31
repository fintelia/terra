use crate::{
    cache::{self, Priority, PriorityCacheEntry},
    terrain::quadtree::VNode,
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
use cache::PriorityCache;
use cgmath::Vector3;
use futures::future::BoxFuture;
use futures::future::FutureExt;
use futures::stream::futures_unordered::FuturesUnordered;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
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
    pub fn is_compressed(&self) -> bool {
        match *self {
            TextureFormat::BC4 | TextureFormat::BC5 => true,
            TextureFormat::R8
            | TextureFormat::RG8
            | TextureFormat::RGBA8
            | TextureFormat::R32F
            | TextureFormat::RG32F
            | TextureFormat::RGBA32F
            | TextureFormat::SRGBA => false,
        }
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
    pub fn name(&self) -> &'static str {
        match *self {
            LayerType::Displacements => "displacements",
            LayerType::Albedo => "albedo",
            LayerType::Roughness => "roughness",
            LayerType::Normals => "normals",
            LayerType::Heightmaps => "heightmaps",
        }
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
impl Entry {
    fn new(node: VNode, priority: Priority) -> Self {
        Self {
            node,
            priority,
            valid: 0,
            generated: 0,
            streaming: 0,
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
    inner: PriorityCache<Entry>,

    /// Resolution of each tile in this cache.
    layers: VecMap<LayerParams>,

    streamer: TileStreamerEndpoint,
    generators: Vec<Box<dyn GenerateTile>>,

    pending_heightmap_downloads:
        FuturesUnordered<BoxFuture<'static, Result<(VNode, wgpu::Buffer), ()>>>,
}
impl TileCache {
    pub fn new(mapfile: Arc<MapFile>, generators: Vec<Box<dyn GenerateTile>>, size: usize) -> Self {
        Self {
            inner: PriorityCache::new(size),
            layers: mapfile.layers().clone(),
            streamer: TileStreamerEndpoint::new(mapfile).unwrap(),
            generators,
            pending_heightmap_downloads: FuturesUnordered::new(),
        }
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_state: &GpuState,
        mapfile: &MapFile,
        camera: mint::Point3<f64>,
    ) {
        let camera = Vector3::new(camera.x, camera.y, camera.z);

        self.refresh_tile_generators();

        // Update priorities
        for entry in self.inner.slots_mut() {
            entry.priority = entry.node.priority(camera);
        }
        let min_priority =
            self.inner.slots().iter().map(|s| s.priority).min().unwrap_or(Priority::none());

        // Find any tiles that may need to be added.
        let mut missing = Vec::new();
        VNode::breadth_first(|node| {
            let priority = node.priority(camera);
            if priority < Priority::cutoff() {
                return false;
            }
            if !self.inner.contains(&node) && (priority > min_priority || !self.inner.is_full()) {
                missing.push(Entry::new(node, priority));
            }

            node.level() < VNode::LEVEL_CELL_2CM
        });
        self.inner.insert(missing);

        self.upload_tiles(queue, &gpu_state.tile_cache);
        self.generate_tiles(mapfile, device, &queue, &gpu_state);
        self.download_tiles();
    }

    fn refresh_tile_generators(&mut self) {
        for (i, gen) in self.generators.iter_mut().enumerate() {
            if gen.needs_refresh() {
                assert!(i < 32);
                let mask = 1u32 << i;
                for slot in self.inner.slots_mut() {
                    for (layer, generator_mask) in &slot.generators {
                        if (generator_mask.get() & mask) != 0 {
                            slot.valid &= !(1 << layer);
                        }
                    }
                }
            }
        }
    }

    fn generate_tiles(
        &mut self,
        mapfile: &MapFile,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_state: &GpuState,
    ) {
        let mut planned_heightmap_downloads = Vec::new();
        let mut pending_generate = VecMap::new();

        for layer in self.layers.values() {
            let ty = layer.layer_type;

            // Figure out which entries need to be uploaded
            let pending_generate = pending_generate.entry(ty.index()).or_insert(Vec::new());

            for ref mut entry in self.inner.slots_mut() {
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

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder.tiles.generate"),
        });

        for (i, nodes) in &mut pending_generate {
            for n in nodes {
                let entry = self.inner.entry(&n).unwrap();
                let parent_entry =
                    if let Some(p) = n.parent() { self.inner.entry(&p.0) } else { None };

                if entry.valid & (1 << i) != 0 {
                    continue;
                }

                for (generator_index, generator) in self.generators.iter_mut().enumerate() {
                    let outputs = generator.outputs(n.level());

                    let peer_inputs = generator.peer_inputs(n.level());
                    let parent_inputs = generator.parent_inputs(n.level());

                    let generates_layer = outputs & (1 << i) != 0;
                    let has_peer_inputs = peer_inputs & !entry.valid == 0;
                    let root_input_missing = n.level() == 0 && generator.parent_inputs(0) != 0;
                    let parent_input_missing = n.level() > 0
                        && (parent_entry.is_none()
                            || parent_inputs & !parent_entry.as_ref().unwrap().valid != 0);

                    if generates_layer
                        && has_peer_inputs
                        && !root_input_missing
                        && !parent_input_missing
                    {
                        let slot = self.inner.index_of(&n).unwrap();
                        let parent_slot =
                            if let Some(p) = n.parent() { self.inner.index_of(&p.0) } else { None };

                        let output_mask = !entry.valid & generator.outputs(n.level());
                        generator.generate(
                            device,
                            &mut encoder,
                            gpu_state,
                            &self.layers,
                            *n,
                            slot,
                            parent_slot,
                            output_mask,
                        );

                        let mut input_generators = 1 << generator_index;
                        for j in (0..32).filter(|j| peer_inputs & (1 << j) != 0) {
                            input_generators |=
                                entry.generators.get(j).copied().map(NonZeroU32::get).unwrap_or(0);
                        }
                        for j in (0..32).filter(|j| parent_inputs & (1 << j) != 0) {
                            input_generators |= parent_entry
                                .as_ref()
                                .unwrap()
                                .generators
                                .get(j)
                                .copied()
                                .map(NonZeroU32::get)
                                .unwrap_or(0);
                        }

                        let entry = self.inner.entry_mut(&n).unwrap();
                        entry.valid |= output_mask;
                        entry.generated |= output_mask;
                        for j in (0..32).filter(|j| output_mask & (1 << j) != 0) {
                            entry.generators.insert(j, NonZeroU32::new(input_generators).unwrap());
                        }

                        if output_mask & LayerType::Heightmaps.bit_mask() != 0
                            && n.level() <= VNode::LEVEL_CELL_1M
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
                                label: Some(&format!("buffer.tiles.download.{}", i)),
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

                            planned_heightmap_downloads.push((*n, buffer));
                        }

                        break;
                    }
                }
            }
        }
        queue.submit(Some(encoder.finish()));

        for (n, buffer) in planned_heightmap_downloads.drain(..) {
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
    }

    fn upload_tiles(&mut self, queue: &wgpu::Queue, textures: &VecMap<wgpu::Texture>) {
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
                            entry.heightmap = Some(CpuHeightmap::I16(Arc::clone(&heights)));
                        }
                        let heights: Vec<_> = heights.iter().map(|&h| h as f32).collect();
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
                    wgpu::TextureCopyView {
                        texture: &textures[layer],
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: index as u32 },
                    },
                    &*data,
                    wgpu::TextureDataLayout {
                        offset: 0,
                        bytes_per_row: row_bytes as u32,
                        rows_per_image: 0,
                    },
                    wgpu::Extent3d {
                        width: resolution as u32,
                        height: resolution as u32,
                        depth: 1,
                    },
                );
            }
        }
    }

    pub(crate) fn download_tiles(&mut self) {
        loop {
            futures::select! {
                h = self.pending_heightmap_downloads.select_next_some() => {
                    if let Ok((node, buffer)) = h {
                        if let Some(entry) = self.inner.entry_mut(&node) {
                            let bytes_per_pixel =
                            self.layers[LayerType::Heightmaps].texture_format.bytes_per_block()
                                as usize;
                            let resolution =
                                self.layers[LayerType::Heightmaps].texture_resolution as usize;
                            let row_bytes = resolution * bytes_per_pixel;
                            let row_pitch = (row_bytes + 255) & !255;
                            let mut heights = vec![0.0; resolution * resolution];

                            {
                                let mapped_buffer = buffer.slice(..).get_mapped_range();
                                for (h, b) in heights.chunks_exact_mut(resolution).zip(mapped_buffer.chunks_exact(row_pitch)) {
                                    bytemuck::cast_slice_mut(h).copy_from_slice(&b[..row_bytes]);
                                }
                            }
                            buffer.unmap();

                            entry.heightmap = Some(CpuHeightmap::F32(Arc::new(heights)));
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
                (
                    ty,
                    device.create_texture(&wgpu::TextureDescriptor {
                        size: wgpu::Extent3d {
                            width: layer.texture_resolution,
                            height: layer.texture_resolution,
                            depth: self.inner.size() as u32,
                        },
                        format: layer.texture_format.to_wgpu(),
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
                    }),
                )
            })
            .collect()
    }

    pub fn contains(&self, node: VNode, ty: LayerType) -> bool {
        self.inner.entry(&node).map(|entry| (entry.valid & ty.bit_mask()) != 0).unwrap_or(false)
    }
    pub fn contains_all(&self, node: VNode, layer_mask: u32) -> bool {
        self.inner
            .entry(&node)
            .map(|entry| (entry.valid & layer_mask) == layer_mask)
            .unwrap_or(false)
    }

    pub fn get_slot(&self, node: VNode) -> Option<usize> {
        self.inner.index_of(&node)
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
            CpuHeightmap::I16(h) => (h[i00] as f32 * w00
                + h[i10] as f32 * w10
                + h[i01] as f32 * w01
                + h[i11] as f32 * w11)
                .max(0.0),
            CpuHeightmap::F32(h) => {
                (h[i00] * w00 + h[i10] * w10 + h[i01] * w01 + h[i11] * w11).max(0.0)
            }
        })
    }
}
