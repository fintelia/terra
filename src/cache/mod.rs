pub(crate) mod generators;
mod mesh;
mod tile;
pub(crate) mod layer;

pub(crate) use crate::cache::mesh::{MeshCache, MeshCacheDesc};
use crate::stream::TileStreamerEndpoint;
use crate::{
    cache::tile::NodeSlot, compute_shader::ComputeShader, gpu_state::GpuState, mapfile::MapFile,
};
use cgmath::Vector3;
use fnv::FnvHashMap;
use maplit::hashmap;
use std::cmp::Eq;
use std::hash::Hash;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::{collections::HashMap, num::NonZeroU32};
use types::{Priority, VNode, MAX_QUADTREE_LEVEL, NODE_OFFSETS};
use vec_map::VecMap;
use wgpu::util::DeviceExt;

use self::layer::{LayerType, LayerMask};
use self::tile::Entry;
use self::{generators::DynamicGenerator, mesh::CullMeshUniforms};
use self::{generators::GenerateTile, tile::CpuHeightmap};

const SLOTS_PER_LEVEL: usize = 32;

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

pub(crate) struct Levels(Vec<PriorityCache<Entry>>);
impl Levels {
    pub(crate) fn base_slot(level: u8) -> usize {
        if level == 0 {
            0
        } else if level == 1 {
            6
        } else {
            30 + SLOTS_PER_LEVEL * (level - 2) as usize
        }
    }

    fn contains(&self, node: VNode) -> bool {
        self.0[node.level() as usize].contains(&node)
    }
    fn get(&self, node: VNode) -> Option<&Entry> {
        self.0[node.level() as usize].entry(&node)
    }
    fn get_mut(&mut self, node: VNode) -> Option<&mut Entry> {
        self.0[node.level() as usize].entry_mut(&node)
    }
    fn get_slot(&self, node: VNode) -> Option<usize> {
        self.0[node.level() as usize].index_of(&node).map(|i| Self::base_slot(node.level()) + i)
    }

    fn contains_layer(&self, node: VNode, ty: LayerType) -> bool {
        self.0[node.level() as usize]
            .entry(&node)
            .map(|entry| entry.valid.contains_layer(ty))
            .unwrap_or(false)
    }
    fn contains_layers(&self, node: VNode, layer_mask: LayerMask) -> bool {
        self.0[node.level() as usize]
            .entry(&node)
            .map(|entry| (entry.valid & layer_mask) == layer_mask)
            .unwrap_or(false)
    }

    fn update(&mut self, node_priorities: FnvHashMap<VNode, Priority>) {
        let mut min_priorities = Vec::new();
        for cache in &mut self.0 {
            for entry in cache.slots_mut() {
                entry.priority =
                    node_priorities.get(&entry.node).cloned().unwrap_or(Priority::none());
            }
            min_priorities
                .push(cache.slots().iter().map(|s| s.priority).min().unwrap_or(Priority::none()));
        }

        let mut missing = vec![Vec::new(); self.0.len()];
        VNode::breadth_first(|node| {
            let priority = node_priorities.get(&node).cloned().unwrap_or(Priority::none());
            if priority < Priority::cutoff() {
                return false;
            }
            let level = node.level() as usize;
            if !self.contains(node)
                && (priority > min_priorities[level] || !self.0[level].is_full())
            {
                missing[level].push(Entry::new(node, priority));
            }

            node.level() < MAX_QUADTREE_LEVEL
        });

        for (cache, missing) in self.0.iter_mut().zip(missing.into_iter()) {
            cache.insert(missing);
        }
    }

    fn generator_dependencies(&self, node: VNode, mask: LayerMask) -> GeneratorMask {
        let mut generators = GeneratorMask::empty();

        if let Some(entry) = self.get(node) {
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
}

pub(crate) struct TileCache {
    levels: Levels,
    level_masks: Vec<LayerMask>,

    meshes: VecMap<MeshCache>,
    generators: Vec<Box<dyn GenerateTile>>,
    dynamic_generators: Vec<DynamicGenerator>,

    streamer: TileStreamerEndpoint,
    completed_downloads_tx: crossbeam::channel::Sender<(VNode, wgpu::Buffer, CpuHeightmap)>,
    completed_downloads_rx: crossbeam::channel::Receiver<(VNode, wgpu::Buffer, CpuHeightmap)>,
    free_download_buffers: Vec<wgpu::Buffer>,
    total_download_buffers: usize,
    last_camera_position: Option<mint::Point3<f64>>,

    index_buffer_contents: Vec<u32>,
    cull_shader: ComputeShader<mesh::CullMeshUniforms>,
}

impl TileCache {
    pub fn new(
        device: &wgpu::Device,
        mapfile: Arc<MapFile>,
        mesh_layers: Vec<MeshCacheDesc>,
    ) -> Self {
        let mut index_buffer_contents = Vec::new();

        let mut base_slot = 0;
        let mut meshes = Vec::new();
        for mut desc in mesh_layers {
            let num_slots = (Levels::base_slot(desc.max_level + 1)
                - Levels::base_slot(desc.min_level))
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

        let generators = generators::generators(device, &meshes);

        let mut level_masks = vec![LayerMask::empty(); 23];
        for layer in LayerType::iter() {
            for i in layer.level_range() {
                level_masks[i as usize] |= layer.bit_mask();
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
            wgpu::TextureFormat::Bc7RgbaUnorm
        } else {
            wgpu::TextureFormat::Astc {
                block: wgpu::AstcBlock::B4x4,
                channel: wgpu::AstcChannel::Unorm,
            }
        };

        Self {
            streamer: TileStreamerEndpoint::new(mapfile, transcode_format).unwrap(),
            level_masks,
            completed_downloads_tx: completed_tx,
            completed_downloads_rx: completed_rx,
            free_download_buffers: Vec::new(),
            total_download_buffers: 0,
            levels: Levels(levels),
            meshes,
            generators,
            dynamic_generators: generators::dynamic_generators(),
            index_buffer_contents,
            cull_shader: ComputeShader::new(
                rshader::shader_source!("../shaders", "cull-meshes.comp", "declarations.glsl"),
                "cull-meshes".to_owned(),
            ),
            last_camera_position: None,
        }
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_state: &GpuState,
        camera: mint::Point3<f64>,
    ) {
        for (i, gen) in self.generators.iter_mut().enumerate() {
            if gen.needs_refresh() {
                assert!(i < 32);
                let mask = GeneratorMask::from_index(i);
                for cache in self.levels.0.iter_mut() {
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

        self.cull_shader.refresh(device, gpu_state);

        if self.last_camera_position != Some(camera) {
            self.last_camera_position = Some(camera);
            let camera = Vector3::new(camera.x, camera.y, camera.z);

            let mut node_priorities = FnvHashMap::default();
            VNode::breadth_first(|node| {
                let priority = node.priority(camera, self.get_height_range(node));
                node_priorities.insert(node, priority);
                priority >= Priority::cutoff() && node.level() < MAX_QUADTREE_LEVEL
            });
            self.levels.update(node_priorities);
        }

        self.upload_tiles(queue, &gpu_state.tile_cache);

        let command_buffer = self.generate_tiles(device, &queue, gpu_state);
        self.write_nodes(queue, gpu_state, camera);
        queue.submit(Some(command_buffer));

        self.tile_readback(device, queue, gpu_state);
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
                padding: [0; 48],
            };
            Levels::base_slot(self.levels.0.len() as u8)
        ];
        for (level_index, level) in self.levels.0.iter().enumerate() {
            for (slot_index, slot) in level.slots().into_iter().enumerate() {
                let index = Levels::base_slot(level_index as u8) + slot_index;

                let node_center = slot.node.center_wspace();
                data[index].node_center[0] = node_center.x as f32;
                data[index].node_center[1] = node_center.y as f32;
                data[index].node_center[2] = node_center.z as f32;
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
                    .and_then(|(parent, _)| self.levels.get_slot(parent))
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
                    if let Some(ancestor_slot) = self.levels.get(ancestor) {
                        for (layer_index, layer) in LayerType::iter().enumerate() {
                            let layer_slot = if !found_layers.contains_layer(layer)
                                && (ancestor_slot.valid.contains_layer(layer)
                                    || (layer.dynamic()
                                        && layer.level_range().contains(&ancestor.level())))
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

                            let texture_resolution = layer.texture_resolution() as f32;
                            let texture_border = layer.texture_border_size() as f32;
                            let texture_ratio = if layer.grid_registration() {
                                (texture_resolution - 2.0 * texture_border - 1.0)
                                    / texture_resolution
                            } else {
                                (texture_resolution - 2.0 * texture_border) / texture_resolution
                            };
                            let texture_origin = if layer.grid_registration() {
                                (texture_border + 0.5) / texture_resolution
                            } else {
                                texture_border / texture_resolution
                            };

                            data[index].layer_origins[layer_slot] = [
                                texture_origin + texture_ratio * base_offset.x,
                                texture_origin + texture_ratio * base_offset.y,
                            ];
                            data[index].layer_slots[layer_slot] =
                                (self.levels.get_slot(ancestor).unwrap()
                                    - Levels::base_slot(layer.min_level()))
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
                    base_slot: Levels::base_slot(c.desc.min_level) as u32,
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

    pub fn contains_layers(&self, node: VNode, layers: LayerMask) -> bool {
        self.levels.contains_layers(node, layers)
    }
}
