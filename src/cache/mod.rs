pub mod generators;
mod mesh;
mod tile;

pub(crate) use crate::cache::mesh::{MeshCache, MeshCacheDesc};
use crate::stream::TileStreamerEndpoint;
use crate::utils::math::InfiniteFrustum;
use crate::{
    cache::tile::NodeSlot,
    generate::ComputeShader,
    gpu_state::GpuState,
    mapfile::MapFile,
    terrain::quadtree::{QuadTree, VNode},
};
use futures::{future::BoxFuture, FutureExt};
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use std::ops::{Index, IndexMut};
use std::{
    cmp::{Eq, Ord, PartialOrd},
    sync::Arc,
};
use std::{collections::HashMap, num::NonZeroU32};
pub(crate) use tile::{LayerParams, TextureFormat};
use vec_map::VecMap;

use self::tile::Entry;
use self::{generators::DynamicGenerator, mesh::CullMeshUniforms};
use self::{generators::GenerateTile, tile::CpuHeightmap};

const SLOTS_PER_LEVEL: usize = 32;
pub(crate) const MAX_QUADTREE_LEVEL: u8 = VNode::LEVEL_CELL_5MM;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum LayerType {
    Displacements = 0,
    Albedo = 1,
    Roughness = 2,
    Normals = 3,
    Heightmaps = 4,
    GrassCanopy = 5,
    MaterialKind = 6,
    AerialPerspective = 7,
    BentNormals = 8,
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
            5 => LayerType::GrassCanopy,
            6 => LayerType::MaterialKind,
            7 => LayerType::AerialPerspective,
            8 => LayerType::BentNormals,
            _ => unreachable!(),
        }
    }
    pub fn bit_mask(&self) -> LayerMask {
        (*self).into()
    }
    pub fn name(&self) -> &'static str {
        match *self {
            LayerType::Displacements => "displacements",
            LayerType::Albedo => "albedo",
            LayerType::Roughness => "roughness",
            LayerType::Normals => "normals",
            LayerType::Heightmaps => "heightmaps",
            LayerType::GrassCanopy => "grass_canopy",
            LayerType::MaterialKind => "material_kind",
            LayerType::AerialPerspective => "aerial_perspective",
            LayerType::BentNormals => "bent_normals",
        }
    }
    fn iter() -> impl Iterator<Item = Self> {
        (0..=8).map(Self::from_index)
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
}
impl MeshType {
    pub fn bit_mask(&self) -> LayerMask {
        (*self).into()
    }
    pub fn name(&self) -> &'static str {
        match *self {
            MeshType::Terrain => "terrain",
            MeshType::Grass => "grass",
        }
    }
    #[allow(unused)]
    fn from_index(i: usize) -> Self {
        match i {
            0 => MeshType::Terrain,
            1 => MeshType::Grass,
            _ => unreachable!(),
        }
    }
    #[allow(unused)]
    fn iter() -> impl Iterator<Item = Self> {
        (0..=1).map(Self::from_index)
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
    start_download:
        tokio::sync::mpsc::UnboundedSender<BoxFuture<'static, Result<(VNode, wgpu::Buffer), ()>>>,
    completed_downloads: crossbeam::channel::Receiver<(VNode, wgpu::Buffer, CpuHeightmap)>,
    free_download_buffers: Vec<wgpu::Buffer>,
    total_download_buffers: usize,

    cull_shader: ComputeShader<mesh::CullMeshUniforms>,
}

impl TileCache {
    pub fn new(
        device: &wgpu::Device,
        mapfile: Arc<MapFile>,
        mesh_layers: Vec<MeshCacheDesc>,
    ) -> Self {
        let layers = mapfile.layers().clone();

        let mut base_slot = 0;
        let mut meshes = Vec::new();
        for desc in mesh_layers {
            let num_slots = (TileCache::base_slot(desc.max_level + 1)
                - TileCache::base_slot(desc.min_level))
                * desc.entries_per_node;
            meshes.push((desc.ty as usize, MeshCache::new(desc, base_slot, num_slots)));
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
            streamer: TileStreamerEndpoint::new(mapfile).unwrap(),
            level_masks,
            start_download: start_tx,
            completed_downloads: completed_rx,
            free_download_buffers: Vec::new(),
            total_download_buffers: 0,
            levels,
            layers,
            meshes,
            generators,
            dynamic_generators: generators::dynamic_generators(),
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
        mapfile: &MapFile,
        quadtree: &mut QuadTree,
        frustum: Option<InfiniteFrustum>,
        camera: mint::Point3<f64>,
    ) {
        for (i, gen) in self.generators.iter_mut().enumerate() {
            if gen.needs_refresh() {
                assert!(i < 32);
                let mask = GeneratorMask::from_index(i);
                for cache in self.levels.iter_mut() {
                    for slot in cache.slots_mut() {
                        // TODO: handle meshes here somehow?
                        for (layer, generator_mask) in &slot.generators {
                            if generator_mask.intersects(mask) {
                                slot.valid &= !LayerType::from_index(layer).bit_mask();
                            }
                        }
                    }
                }
            }
        }

        TileCache::update_levels(self, quadtree);
        self.upload_tiles(queue, &gpu_state.tile_cache);

        let (command_buffer, mut planned_heightmap_downloads) =
            TileCache::generate_tiles(self, mapfile, device, &queue, gpu_state, frustum);

        self.write_nodes(queue, gpu_state, camera);

        queue.submit(Some(command_buffer));

        for (n, buffer) in planned_heightmap_downloads.drain(..) {
            let _ = self.start_download.send(
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

        self.download_tiles();
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
                layer_steps: [0.0; 48],
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
                            let layer_slot = if (ancestor_index == 0
                                && slot.valid.contains_layer(layer))
                                || (!found_layers.contains_layer(layer)
                                    && ancestor_slot.valid.contains_layer(layer))
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
                            let texture_step = texture_ratio / 64.0;
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
                            data[index].layer_steps[layer_slot] =
                                f32::powi(0.5, ancestor_index as i32) * texture_step;
                        }
                    }

                    if ancestor_index < level_index {
                        let parent = ancestor.parent().unwrap();
                        ancestor = parent.0;
                        base_offset = (crate::terrain::quadtree::node::OFFSETS[parent.1 as usize]
                            .cast()
                            .unwrap()
                            + base_offset)
                            * 0.5;
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
    ) -> VecMap<(wgpu::Texture, wgpu::TextureView)> {
        self.make_cache_textures(device)
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
        &'a mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gpu_state: &'a GpuState,
    ) {
        self.cull_shader.refresh();
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

    pub fn render_meshes<'a>(
        &'a mut self,
        device: &wgpu::Device,
        rpass: &mut wgpu::RenderPass<'a>,
        gpu_state: &'a GpuState,
    ) {
        for (_, c) in &mut self.meshes {
            c.render(device, rpass, gpu_state);
        }
    }
}
