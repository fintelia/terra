pub mod generators;
mod mesh;
mod tile;

pub(crate) use crate::cache::mesh::{MeshCache, MeshCacheDesc};
use crate::stream::TileStreamerEndpoint;
use crate::{
    cache::tile::NodeSlot, compute_shader::ComputeShader, gpu_state::GpuState, mapfile::MapFile,
    quadtree::QuadTree,
};
use basis_universal::TranscoderTextureFormat;
use maplit::hashmap;
use serde::{Deserialize, Serialize};
use std::cmp::Eq;
use std::hash::Hash;
use std::num::NonZeroU64;
use std::ops::{Index, IndexMut};
use std::sync::Arc;
use std::{collections::HashMap, num::NonZeroU32};
pub(crate) use tile::{LayerParams, TextureFormat};
use types::{Priority, VNode, MAX_QUADTREE_LEVEL, NODE_OFFSETS};
use vec_map::VecMap;
use wgpu::util::DeviceExt;

use self::tile::Entry;
use self::{generators::DynamicGenerator, mesh::CullMeshUniforms};
use self::{generators::GenerateTile, tile::CpuHeightmap};

const SLOTS_PER_LEVEL: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum LayerType {
    Heightmaps = 0,
    Displacements = 1,
    AlbedoRoughness = 2,
    Normals = 3,
    GrassCanopy = 4,
    TreeAttributes = 5,
    AerialPerspective = 6,
    BentNormals = 7,
    TreeCover = 8,
    BaseAlbedo = 9,
    RootAerialPerspective = 10,
    LandFraction = 11,
    Slopes = 12,
}
impl LayerType {
    pub fn index(&self) -> usize {
        *self as usize
    }
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => LayerType::Heightmaps,
            1 => LayerType::Displacements,
            2 => LayerType::AlbedoRoughness,
            3 => LayerType::Normals,
            4 => LayerType::GrassCanopy,
            5 => LayerType::TreeAttributes,
            6 => LayerType::AerialPerspective,
            7 => LayerType::BentNormals,
            8 => LayerType::TreeCover,
            9 => LayerType::BaseAlbedo,
            10 => LayerType::RootAerialPerspective,
            11 => LayerType::LandFraction,
            12 => LayerType::Slopes,
            _ => unreachable!(),
        }
    }
    pub fn bit_mask(&self) -> LayerMask {
        (*self).into()
    }
    pub fn name(&self) -> &'static str {
        match *self {
            LayerType::Heightmaps => "heightmaps",
            LayerType::Displacements => "displacements",
            LayerType::AlbedoRoughness => "albedo",
            LayerType::Normals => "normals",
            LayerType::GrassCanopy => "grass_canopy",
            LayerType::TreeAttributes => "tree_attributes",
            LayerType::AerialPerspective => "aerial_perspective",
            LayerType::BentNormals => "bent_normals",
            LayerType::TreeCover => "treecover",
            LayerType::BaseAlbedo => "base_albedo",
            LayerType::RootAerialPerspective => "root_aerial_perspective",
            LayerType::LandFraction => "land_fraction",
            LayerType::Slopes => "slopes",
        }
    }
    pub fn streamed_levels(&self) -> u8 {
        match *self {
            LayerType::Heightmaps => VNode::LEVEL_CELL_76M + 1,
            LayerType::BaseAlbedo => VNode::LEVEL_CELL_610M + 1,
            LayerType::TreeCover => VNode::LEVEL_CELL_76M + 1,
            LayerType::LandFraction => VNode::LEVEL_CELL_76M + 1,
            _ => 0,
        }
    }
    pub fn dynamic(&self) -> bool {
        match *self {
            LayerType::AerialPerspective | LayerType::RootAerialPerspective => true,
            _ => false,
        }
    }
    pub fn iter() -> impl Iterator<Item = Self> {
        (0..=12).map(Self::from_index)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum MeshType {
    Terrain = 0,
    Grass = 1,
    TreeBillboards = 2,
}
impl MeshType {
    pub fn bit_mask(&self) -> LayerMask {
        (*self).into()
    }
    pub fn name(&self) -> &'static str {
        match *self {
            MeshType::Terrain => "terrain",
            MeshType::Grass => "grass",
            MeshType::TreeBillboards => "tree_billboards",
        }
    }
    fn from_index(i: usize) -> Self {
        match i {
            0 => MeshType::Terrain,
            1 => MeshType::Grass,
            2 => MeshType::TreeBillboards,
            _ => unreachable!(),
        }
    }
    pub fn iter() -> impl Iterator<Item = Self> {
        (0..=2).map(Self::from_index)
    }
}
impl<T> Index<MeshType> for VecMap<T> {
    type Output = T;
    fn index(&self, i: MeshType) -> &Self::Output {
        &self[i as usize]
    }
}
impl<T> IndexMut<MeshType> for VecMap<T> {
    fn index_mut(&mut self, i: MeshType) -> &mut Self::Output {
        &mut self[i as usize]
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) struct LayerMask(NonZeroU32);
impl LayerMask {
    const VALID: u32 = 0x80000000;

    pub fn empty() -> Self {
        Self(NonZeroU32::new(Self::VALID).unwrap())
    }
    pub fn contains_layer(&self, t: LayerType) -> bool {
        assert!((t as usize) < 16);
        self.0.get() & (1 << (t as usize)) != 0
    }
    pub fn contains_mesh(&self, t: MeshType) -> bool {
        assert!((t as usize) < 8);
        self.0.get() & (1 << (t as usize + 16)) != 0
    }
}
impl From<LayerType> for LayerMask {
    fn from(t: LayerType) -> Self {
        assert!((t as usize) < 16);
        Self(NonZeroU32::new(Self::VALID | (1 << (t as usize))).unwrap())
    }
}
impl From<MeshType> for LayerMask {
    fn from(t: MeshType) -> Self {
        assert!((t as usize) < 8);
        Self(NonZeroU32::new(Self::VALID | (1 << (t as usize + 16))).unwrap())
    }
}
impl std::ops::BitOr for LayerMask {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for LayerMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl std::ops::BitAnd for LayerMask {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(NonZeroU32::new(Self::VALID | (self.0.get() & rhs.0.get())).unwrap())
    }
}
impl std::ops::BitAndAssign for LayerMask {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 = NonZeroU32::new(Self::VALID | (self.0.get() & rhs.0.get())).unwrap();
    }
}
impl std::ops::Not for LayerMask {
    type Output = Self;
    fn not(self) -> Self {
        Self(NonZeroU32::new(Self::VALID | !self.0.get()).unwrap())
    }
}

lazy_static! {
    pub(crate) static ref LAYERS_BY_NAME: HashMap<&'static str, LayerType> =
        LayerType::iter().map(|t| (t.name(), t)).collect();
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) struct GeneratorMask(NonZeroU32);
impl GeneratorMask {
    const VALID: u32 = 0x80000000;

    pub fn empty() -> Self {
        Self(NonZeroU32::new(Self::VALID).unwrap())
    }
    pub fn from_index(i: usize) -> Self {
        assert!(i < 31);
        Self(NonZeroU32::new(Self::VALID | 1 << i).unwrap())
    }
    pub fn intersects(&self, other: Self) -> bool {
        self.0.get() & other.0.get() != Self::VALID
    }
}
impl std::ops::BitOr for GeneratorMask {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for GeneratorMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl std::ops::BitAnd for GeneratorMask {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(NonZeroU32::new(Self::VALID | (self.0.get() & rhs.0.get())).unwrap())
    }
}
impl std::ops::BitAndAssign for GeneratorMask {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 = NonZeroU32::new(Self::VALID | (self.0.get() & rhs.0.get())).unwrap();
    }
}
impl std::ops::Not for GeneratorMask {
    type Output = Self;
    fn not(self) -> Self {
        Self(NonZeroU32::new(Self::VALID | !self.0.get()).unwrap())
    }
}

pub trait PriorityCacheEntry {
    type Key: Hash + Eq;

    fn priority(&self) -> Priority;
    fn key(&self) -> Self::Key;
}

#[derive(Default)]
pub struct PriorityCache<T: PriorityCacheEntry> {
    size: usize,
    slots: Vec<T>,
    reverse: HashMap<T::Key, usize>,
}
impl<T: PriorityCacheEntry> PriorityCache<T> {
    pub fn new(size: usize) -> Self {
        Self { size, slots: Vec::new(), reverse: HashMap::new() }
    }
    pub fn insert(&mut self, mut entries: Vec<T>) {
        entries.sort_by_key(T::priority);

        // Add tiles until all cache entries are full.
        while !entries.is_empty() && self.slots.len() < self.size {
            let e = entries.pop().unwrap();
            self.reverse.insert(e.key(), self.slots.len());
            self.slots.push(e);
        }

        // If more tiles meet the threshold, start evicting some existing entries.
        if !entries.is_empty() {
            let mut possible: Vec<_> = self
                .slots
                .iter()
                .map(|e| e.priority())
                .chain(entries.iter().map(|m| m.priority()))
                .collect();
            possible.sort();

            // Anything >= to cutoff should be included.
            let cutoff = possible[possible.len() - self.size];
            entries.retain(|e| e.priority() > cutoff);

            let mut index = 0;
            'outer: while let Some(e) = entries.pop() {
                // Find the next element to evict.
                while self.slots[index].priority() > cutoff {
                    index += 1;
                    if index == self.slots.len() {
                        break 'outer;
                    }
                }

                self.reverse.remove(&self.slots[index].key());
                self.reverse.insert(e.key(), index);
                self.slots[index] = e;
                index += 1;
                if index == self.slots.len() {
                    break;
                }
            }
        }
    }

    pub fn is_full(&self) -> bool {
        self.slots.len() == self.size
    }
    pub fn contains(&self, key: &T::Key) -> bool {
        self.reverse.contains_key(key)
    }

    pub fn slots(&self) -> &[T] {
        &*self.slots
    }
    pub fn slots_mut(&mut self) -> &mut [T] {
        &mut *self.slots
    }
    pub fn entry(&self, key: &T::Key) -> Option<&T> {
        Some(&self.slots[*self.reverse.get(key)?])
    }
    pub fn entry_mut(&mut self, key: &T::Key) -> Option<&mut T> {
        Some(&mut self.slots[*self.reverse.get(key)?])
    }

    pub fn index_of(&self, key: &T::Key) -> Option<usize> {
        self.reverse.get(key).copied()
    }
}

pub(crate) struct TileCache {
    levels: Vec<PriorityCache<Entry>>,
    level_masks: Vec<LayerMask>,

    layers: VecMap<LayerParams>,
    meshes: VecMap<MeshCache>,
    generators: Vec<Box<dyn GenerateTile>>,
    dynamic_generators: Vec<DynamicGenerator>,

    streamer: TileStreamerEndpoint,
    completed_downloads_tx: crossbeam::channel::Sender<(VNode, wgpu::Buffer, CpuHeightmap)>,
    completed_downloads_rx: crossbeam::channel::Receiver<(VNode, wgpu::Buffer, CpuHeightmap)>,
    free_download_buffers: Vec<wgpu::Buffer>,
    total_download_buffers: usize,

    index_buffer_contents: Vec<u32>,
    cull_shader: ComputeShader<mesh::CullMeshUniforms>,
}

impl TileCache {
    pub fn new(
        device: &wgpu::Device,
        mapfile: Arc<MapFile>,
        mesh_layers: Vec<MeshCacheDesc>,
    ) -> Self {
        let layers = mapfile.layers().clone();

        let mut index_buffer_contents = Vec::new();

        let mut base_slot = 0;
        let mut meshes = Vec::new();
        for mut desc in mesh_layers {
            let num_slots = (TileCache::base_slot(desc.max_level + 1)
                - TileCache::base_slot(desc.min_level))
                * desc.entries_per_node;
            let index_buffer_offset = index_buffer_contents.len() as u64;
            index_buffer_contents.append(&mut desc.index_buffer);
            meshes.push((
                desc.ty as usize,
                MeshCache::new(
                    desc,
                    base_slot,
                    num_slots,
                    index_buffer_offset * 4..index_buffer_contents.len() as u64 * 4,
                ),
            ));
            base_slot += num_slots;
        }
        let meshes = meshes.into_iter().collect();

        let soft_float64 = !device.features().contains(wgpu::Features::SHADER_FLOAT64);
        let generators = generators::generators(device, &layers, &meshes, soft_float64);

        let mut level_masks = vec![LayerMask::empty(); 23];
        for layer in layers.values() {
            for i in layer.min_level..=layer.max_level {
                level_masks[i as usize] |= layer.layer_type.bit_mask();
            }
        }
        for mesh in meshes.values() {
            for i in mesh.desc.min_level..=mesh.desc.max_level {
                level_masks[i as usize] |= mesh.desc.ty.bit_mask();
            }
        }

        let mut levels = vec![PriorityCache::new(6), PriorityCache::new(24)];
        for _ in 2..=MAX_QUADTREE_LEVEL {
            levels.push(PriorityCache::new(SLOTS_PER_LEVEL));
        }

        let (completed_tx, completed_rx) = crossbeam::channel::unbounded();

        let transcode_format = if device.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
        {
            TranscoderTextureFormat::BC7_RGBA
        } else {
            TranscoderTextureFormat::ASTC_4x4_RGBA
        };

        Self {
            streamer: TileStreamerEndpoint::new(mapfile, transcode_format).unwrap(),
            level_masks,
            completed_downloads_tx: completed_tx,
            completed_downloads_rx: completed_rx,
            free_download_buffers: Vec::new(),
            total_download_buffers: 0,
            levels,
            layers,
            meshes,
            generators,
            dynamic_generators: generators::dynamic_generators(),
            index_buffer_contents,
            cull_shader: ComputeShader::new(
                rshader::shader_source!("../shaders", "cull-meshes.comp", "declarations.glsl"),
                "cull-meshes".to_owned(),
            ),
        }
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_state: &GpuState,
        quadtree: &mut QuadTree,
        camera: mint::Point3<f64>,
    ) {
        for (i, gen) in self.generators.iter_mut().enumerate() {
            if gen.needs_refresh() {
                assert!(i < 32);
                let mask = GeneratorMask::from_index(i);
                for cache in self.levels.iter_mut() {
                    for slot in cache.slots_mut() {
                        for (layer, generator_mask) in &slot.generators {
                            if generator_mask.intersects(mask) {
                                slot.valid &= !LayerType::from_index(layer).bit_mask();
                            }
                        }
                        // Directly remove any meshes that were generated by this.
                        slot.valid &= !gen.outputs();
                    }
                }
            }
        }

        for g in &mut self.dynamic_generators {
            if g.bindgroup_pipeline.is_none() || g.shader.refresh() {
                let (bindgroup, layout) = gpu_state.bind_group_for_shader(
                    device,
                    &g.shader,
                    hashmap!["ubo".into() => (true, wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &gpu_state.generate_uniforms,
                    offset: 0,
                    size: Some(NonZeroU64::new(4096).unwrap()),
                }))],
                HashMap::new(),
                &format!("generate.{}", g.name),
                );
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        bind_group_layouts: [&layout][..].into(),
                        push_constant_ranges: &[],
                        label: None,
                    })),
                    module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(&format!("shader.generate.{}", g.name)),
                        source: g.shader.compute(),
                    }),
                    entry_point: "main",
                    label: Some(&format!("pipeline.generate.{}", g.name)),
                });
                g.bindgroup_pipeline = Some((bindgroup, pipeline));
            }
        }

        TileCache::update_levels(self, quadtree);
        self.upload_tiles(queue, &gpu_state.tile_cache);

        let (command_buffer, mut planned_heightmap_downloads) =
            TileCache::generate_tiles(self, device, &queue, gpu_state);

        self.write_nodes(queue, gpu_state, camera);

        queue.submit(Some(command_buffer));

        let heightmap_resolution = self.layers[LayerType::Heightmaps].texture_resolution as usize;
        let heightmap_bytes_per_pixel =
            self.layers[LayerType::Heightmaps].texture_format[0].bytes_per_block() as usize;
        let heightmap_row_bytes = heightmap_resolution * heightmap_bytes_per_pixel;
        let heightmap_row_pitch = (heightmap_row_bytes + 255) & !255;
        for (node, buffer) in planned_heightmap_downloads.drain(..) {
            let buffer = Arc::new(buffer);
            let completed_downloads_tx = self.completed_downloads_tx.clone();
            buffer.clone().slice(..).map_async(wgpu::MapMode::Read, move |r| {
                if r.is_err() {
                    return;
                }

                let mut heights = vec![0u32; heightmap_resolution * heightmap_resolution];
                {
                    let mapped_buffer = buffer.slice(..).get_mapped_range();
                    for (h, b) in heights
                        .chunks_exact_mut(heightmap_resolution)
                        .zip(mapped_buffer.chunks_exact(heightmap_row_pitch))
                    {
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

                let _ = completed_downloads_tx.send((
                    node,
                    Arc::try_unwrap(buffer).unwrap(),
                    CpuHeightmap::F32 { min, max, heights: Arc::new(heights) },
                ));
            });
        }

        self.download_tiles();

        self.cull_shader.refresh(device, gpu_state);
    }

    fn write_nodes(&self, queue: &wgpu::Queue, gpu_state: &GpuState, camera: mint::Point3<f64>) {
        assert_eq!(std::mem::size_of::<NodeSlot>(), 1024);

        let mut frame_nodes: VecMap<HashMap<_, _>> = VecMap::new();
        for (index, mesh) in &self.meshes {
            if !mesh.desc.render_overlapping_levels {
                frame_nodes.insert(
                    index,
                    self.compute_visible(mesh.desc.ty.bit_mask()).into_iter().collect(),
                );
            }
        }

        let mut data: Vec<NodeSlot> = vec![
            NodeSlot {
                node_center: [0.0; 3],
                layer_origins: [[0.0; 2]; 48],
                layer_slots: [-1; 48],
                layer_ratios: [0.0; 48],
                relative_position: [0.0; 3],
                min_distance: 0.0,
                mesh_valid_mask: [0; 4],
                level: 0,
                face: 0,
                coords: [0; 2],
                parent: -1,
                padding0: 0.0,
                padding: [0; 43],
            };
            TileCache::base_slot(self.levels.len() as u8)
        ];
        for (level_index, level) in self.levels.iter().enumerate() {
            for (slot_index, slot) in level.slots().into_iter().enumerate() {
                let index = TileCache::base_slot(level_index as u8) + slot_index;

                data[index].node_center = slot.node.center_wspace().into();
                data[index].level = level_index as u32;
                data[index].face = slot.node.face() as u32;
                data[index].coords = [slot.node.x(), slot.node.y()];
                data[index].relative_position = {
                    (cgmath::Point3::from(camera) - slot.node.center_wspace())
                        .cast::<f32>()
                        .unwrap()
                        .into()
                };
                data[index].min_distance = slot.node.min_distance() as f32;
                data[index].parent = slot
                    .node
                    .parent()
                    .and_then(|(parent, _)| self.get_slot(parent))
                    .map(|s| s as i32)
                    .unwrap_or(-1);

                for (mesh_index, m) in &self.meshes {
                    assert!(m.desc.entries_per_node <= 32);
                    data[index].mesh_valid_mask[mesh_index] = if slot.valid.contains_mesh(m.desc.ty)
                    {
                        0xffffffff >> (32 - m.desc.entries_per_node)
                    } else {
                        0
                    };
                    if let Some(ref frame_nodes) = frame_nodes.get(mesh_index) {
                        data[index].mesh_valid_mask[mesh_index] &=
                            *frame_nodes.get(&slot.node).unwrap_or(&0) as u32;
                    }
                }

                let mut ancestor = slot.node;
                let mut base_offset = cgmath::Vector2::new(0.0, 0.0);
                let mut found_layers = LayerMask::empty();
                for ancestor_index in 0..=level_index {
                    if let Some(ancestor_slot) =
                        self.levels[ancestor.level() as usize].entry(&ancestor)
                    {
                        for (layer_index, layer) in LayerType::iter().enumerate() {
                            let layer_slot = if !found_layers.contains_layer(layer)
                                && (ancestor_slot.valid.contains_layer(layer)
                                    || (layer.dynamic()
                                        && ancestor.level() >= self.layers[layer].min_level
                                        && ancestor.level() <= self.layers[layer].max_level))
                            {
                                found_layers |= layer.bit_mask();
                                layer_index
                            } else if ancestor_index == 1
                                && slot.valid.contains_layer(layer)
                                && ancestor_slot.valid.contains_layer(layer)
                            {
                                layer_index + 24
                            } else {
                                continue;
                            };

                            let texture_resolution = self.layers[layer].texture_resolution as f32;
                            let texture_border = self.layers[layer].texture_border_size as f32;
                            let texture_ratio = if self.layers[layer].grid_registration {
                                (texture_resolution - 2.0 * texture_border - 1.0)
                                    / texture_resolution
                            } else {
                                (texture_resolution - 2.0 * texture_border) / texture_resolution
                            };
                            let texture_origin = if self.layers[layer].grid_registration {
                                (texture_border + 0.5) / texture_resolution
                            } else {
                                texture_border / texture_resolution
                            };

                            data[index].layer_origins[layer_slot] = [
                                texture_origin + texture_ratio * base_offset.x,
                                texture_origin + texture_ratio * base_offset.y,
                            ];
                            data[index].layer_slots[layer_slot] = (self.get_slot(ancestor).unwrap()
                                - Self::base_slot(self.layers[layer].min_level))
                                as i32;
                            data[index].layer_ratios[layer_slot] =
                                f32::powi(0.5, ancestor_index as i32) * texture_ratio;
                        }
                    }

                    if ancestor_index < level_index {
                        let parent = ancestor.parent().unwrap();
                        ancestor = parent.0;
                        base_offset =
                            (NODE_OFFSETS[parent.1 as usize].cast().unwrap() + base_offset) * 0.5;
                    }
                }
            }
        }
        queue.write_buffer(&gpu_state.nodes, 0, bytemuck::cast_slice(&data));
    }

    fn generator_dependencies(&self, node: VNode, mask: LayerMask) -> GeneratorMask {
        let mut generators = GeneratorMask::empty();

        if let Some(entry) = self.levels[node.level() as usize].entry(&node) {
            for layer in LayerType::iter().filter(|layer| mask.contains_layer(*layer)) {
                generators |= entry
                    .generators
                    .get(layer.index())
                    .copied()
                    .unwrap_or_else(GeneratorMask::empty);
            }
        }
        generators
    }

    pub fn make_gpu_tile_cache(
        &self,
        device: &wgpu::Device,
    ) -> VecMap<Vec<(wgpu::Texture, wgpu::TextureView)>> {
        self.make_cache_textures(device)
    }
    pub fn make_gpu_mesh_index(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(&self.index_buffer_contents),
            usage: wgpu::BufferUsages::INDEX,
            label: Some("buffer.index.mesh"),
        })
    }
    pub fn make_gpu_mesh_storage(&self, device: &wgpu::Device) -> VecMap<wgpu::Buffer> {
        self.meshes
            .iter()
            .filter(|(_, c)| c.desc.max_bytes_per_node > 0)
            .map(|(i, c)| {
                (
                    i,
                    device.create_buffer(&wgpu::BufferDescriptor {
                        size: c.desc.max_bytes_per_node
                            * (c.num_entries / c.desc.entries_per_node) as u64,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                        label: Some(&format!("buffer.storage.{}", c.desc.ty.name())),
                    }),
                )
            })
            .collect()
    }
    pub fn total_mesh_entries(&self) -> usize {
        self.meshes.values().map(|m| m.num_entries).sum()
    }

    pub fn cull_meshes<'a>(
        &'a self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gpu_state: &'a GpuState,
    ) {
        for (mesh_index, c) in &self.meshes {
            self.cull_shader.run(
                device,
                encoder,
                &gpu_state,
                ((c.num_entries as u32 + 63) / 64, 1, 1),
                &CullMeshUniforms {
                    base_entry: c.base_entry as u32,
                    entries_per_node: c.desc.entries_per_node as u32,
                    num_nodes: (c.num_entries / c.desc.entries_per_node) as u32,
                    base_slot: TileCache::base_slot(c.desc.min_level) as u32,
                    mesh_index: mesh_index as u32,
                },
            );
        }
    }

    pub fn update_meshes(&mut self, device: &wgpu::Device, gpu_state: &GpuState) {
        for (_, c) in &mut self.meshes {
            c.update(device, gpu_state);
        }
    }

    pub fn render_meshes<'a>(
        &'a self,
        device: &wgpu::Device,
        rpass: &mut wgpu::RenderPass<'a>,
        gpu_state: &'a GpuState,
    ) {
        for (_, c) in &self.meshes {
            c.render(device, rpass, gpu_state);
        }
    }

    pub fn render_mesh_shadows<'a>(
        &'a self,
        device: &wgpu::Device,
        rpass: &mut wgpu::RenderPass<'a>,
        gpu_state: &'a GpuState,
    ) {
        for (_, c) in &self.meshes {
            c.render_shadow(device, rpass, gpu_state);
        }
    }
}
