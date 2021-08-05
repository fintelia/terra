pub mod generators;
mod mesh;
mod tile;

pub(crate) use mesh::{MeshCache, MeshCacheDesc};
pub(crate) use tile::{LayerParams, TextureFormat, TileCache};

use crate::{
    generate::ComputeShader,
    gpu_state::{GpuMeshLayer, GpuState},
    mapfile::MapFile,
    terrain::quadtree::{QuadTree, VNode},
    utils::math::InfiniteFrustum,
};
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use std::ops::{Index, IndexMut};
use std::{
    cmp::{Eq, Ord, PartialOrd},
    sync::Arc,
};
use std::{collections::HashMap, num::NonZeroU32};
use vec_map::VecMap;

use self::generators::GenerateTile;

pub(crate) use tile::SLOTS_PER_LEVEL;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum LayerType {
    Displacements = 0,
    Albedo = 1,
    Roughness = 2,
    Normals = 3,
    Heightmaps = 4,
    GrassCanopy = 5,
    MaterialKind = 6,
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
        }
    }
    fn iter() -> impl Iterator<Item = Self> {
        (0..=6).map(Self::from_index)
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
    Grass = 0,
}
impl MeshType {
    pub fn bit_mask(&self) -> LayerMask {
        (*self).into()
    }
    pub fn name(&self) -> &'static str {
        match *self {
            MeshType::Grass => "grass",
        }
    }
    fn from_index(i: usize) -> Self {
        match i {
            0 => MeshType::Grass,
            _ => unreachable!(),
        }
    }
    fn iter() -> impl Iterator<Item = Self> {
        (0..=0).map(Self::from_index)
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
    pub fn all_tiles() -> Self {
        Self(NonZeroU32::new(Self::VALID | 0x0000ff).unwrap())
    }
    #[allow(unused)]
    pub fn all_meshes() -> Self {
        Self(NonZeroU32::new(Self::VALID | 0x00ff00).unwrap())
    }

    pub fn intersects(&self, other: Self) -> bool {
        self.0.get() & other.0.get() != Self::VALID
    }

    pub fn contains_layer(&self, t: LayerType) -> bool {
        assert!((t as usize) < 8);
        self.0.get() & (1 << (t as usize)) != 0
    }
    #[allow(unused)]
    pub fn contains_mesh(&self, t: MeshType) -> bool {
        assert!((t as usize) < 8);
        self.0.get() & (1 << (t as usize + 8)) != 0
    }
}
impl From<LayerType> for LayerMask {
    fn from(t: LayerType) -> Self {
        assert!((t as usize) < 8);
        Self(NonZeroU32::new(Self::VALID | (1 << (t as usize))).unwrap())
    }
}
impl From<MeshType> for LayerMask {
    fn from(t: MeshType) -> Self {
        assert!((t as usize) < 8);
        Self(NonZeroU32::new(Self::VALID | (1 << (t as usize + 8))).unwrap())
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
    pub(crate) static ref LAYERS_BY_NAME: HashMap<&'static str, LayerType> = {
        LayerType::iter().map(|t| (t.name(), t)).collect()
    };
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

    pub fn size(&self) -> usize {
        self.size
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

pub(crate) struct UnifiedPriorityCache {
    pub tiles: TileCache,
    meshes: VecMap<MeshCache>,

    cull_shader: ComputeShader<mesh::CullMeshUniforms>,
}

impl UnifiedPriorityCache {
    pub fn new(
        device: &wgpu::Device,
        mapfile: Arc<MapFile>,
        generators: Vec<Box<dyn GenerateTile>>,
        mesh_layers: Vec<MeshCacheDesc>,
    ) -> Self {
        Self {
            tiles: TileCache::new(mapfile, generators),
            meshes: mesh_layers
                .into_iter()
                .map(|desc| (desc.ty as usize, MeshCache::new(device, desc)))
                .collect(),
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
        quadtree: &QuadTree,
    ) {
        for (i, gen) in self.tiles.generators.iter_mut().enumerate() {
            if gen.needs_refresh() {
                assert!(i < 32);
                let mask = GeneratorMask::from_index(i);
                for cache in self.tiles.levels.iter_mut() {
                    for slot in cache.slots_mut() {
                        for (layer, generator_mask) in &slot.generators {
                            if generator_mask.intersects(mask) {
                                slot.valid &= !LayerType::from_index(layer).bit_mask();
                            }
                        }
                    }
                }
            }
        }

        for mesh_cache in self.meshes.values_mut() {
            if mesh_cache.desc.generate.refresh() {
                for cache in self.tiles.levels.iter_mut() {
                    for slot in cache.slots_mut() {
                        slot.valid &= !mesh_cache.desc.ty.bit_mask();
                    }
                }
            }
        }

        self.tiles.update(quadtree);
        self.tiles.upload_tiles(queue, &gpu_state.tile_cache);
        TileCache::generate_tiles(self, mapfile, device, &queue, gpu_state);
        self.tiles.download_tiles();

        MeshCache::generate_all(self, device, queue, gpu_state);
    }

    fn generator_dependencies(&self, node: VNode, mask: LayerMask) -> GeneratorMask {
        let mut generators = GeneratorMask::empty();

        if let Some(entry) = self.tiles.levels[node.level() as usize].entry(&node) {
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

    pub fn make_gpu_tile_cache(&self, device: &wgpu::Device) -> VecMap<(wgpu::Texture, wgpu::TextureView)> {
        self.tiles.make_cache_textures(device)
    }
    pub fn make_gpu_mesh_cache(&self, device: &wgpu::Device) -> VecMap<GpuMeshLayer> {
        self.meshes.iter().map(|(i, c)| (i, c.make_buffers(device))).collect()
    }

    pub fn cull_meshes<'a>(
        &'a mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gpu_state: &'a GpuState,
        frustum: &InfiniteFrustum,
        camera: mint::Point3<f64>,
    ) {
        self.cull_shader.refresh();
        for (_, c) in &mut self.meshes {
            c.cull_meshes(device, encoder, gpu_state, &self.tiles, camera, frustum, &mut self.cull_shader);
        }
    }

    pub fn render_meshes<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &'a wgpu::Queue,
        rpass: &mut wgpu::RenderPass<'a>,
        gpu_state: &'a GpuState,
        camera: mint::Point3<f64>,
    ) {
        for (_, c) in &mut self.meshes {
            c.render(&self.tiles, device, queue, rpass, gpu_state, camera);
        }
    }

    pub fn tile_desc(&self, ty: LayerType) -> &LayerParams {
        &self.tiles.layers[ty]
    }

    #[allow(unused)]
    fn contains_all(&self, node: VNode, mask: LayerMask) -> bool {
        self.tiles.contains_all(node, mask)
    }
}
