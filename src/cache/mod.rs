mod mesh;
mod tile;

pub(crate) use mesh::{MeshCache, MeshCacheDesc};
pub(crate) use tile::{LayerParams, TextureFormat, TileCache};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::{Index, IndexMut};
use std::{
    cmp::{Eq, Ord, PartialOrd},
    sync::Arc,
};
use vec_map::VecMap;

use crate::{generate::GenerateTile, gpu_state::{GpuMeshLayer, GpuState}, mapfile::MapFile};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum MeshType {
    Grass = 0,
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
struct LayerMask(u32);
impl From<LayerType> for LayerMask {
    fn from(t: LayerType) -> Self {
        assert!((t as usize) < 8);
        Self(1 << (t as usize))
    }
}
impl From<MeshType> for LayerMask {
    fn from(t: MeshType) -> Self {
        assert!((t as usize) < 8);
        Self(1 << (t as usize + 8))
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
                while self.slots[index].priority() >= cutoff {
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
    pub meshes: VecMap<MeshCache>,
}

impl UnifiedPriorityCache {
    pub fn new(
        device: &wgpu::Device,
        mapfile: Arc<MapFile>,
        size: usize,
        generators: Vec<Box<dyn GenerateTile>>,
        mesh_layers: Vec<MeshCacheDesc>,
    ) -> Self {
        Self {
            tiles: TileCache::new(mapfile, generators, size),
            meshes: mesh_layers
                .into_iter()
                .map(|desc| (desc.ty as usize, MeshCache::new(device, desc)))
                .collect(),
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
        for (i, gen) in self.tiles.generators.iter_mut().enumerate() {
            if gen.needs_refresh() {
                assert!(i < 32);
                let mask = 1u32 << i;
                for slot in self.tiles.inner.slots_mut() {
                    for (layer, generator_mask) in &slot.generators {
                        if (generator_mask.get() & mask) != 0 {
                            slot.valid &= !(1 << layer);
                        }
                    }
                }
            }
        }

        for mesh_cache in self.meshes.values_mut() {
            if mesh_cache.desc.generate.refresh() {
                for entry in mesh_cache.inner.slots_mut() {
                    entry.valid = false;
                }
            }
        }

        self.tiles.update(device, queue, gpu_state, mapfile, camera);
        for m in self.meshes.values_mut() {
            m.update(device, queue, &self.tiles, gpu_state, camera);
        }
    }

    pub fn make_gpu_tile_cache(&self, device: &wgpu::Device) -> VecMap<wgpu::Texture> {
        self.tiles.make_cache_textures(device)
    }
    pub fn make_gpu_mesh_cache(&self, device: &wgpu::Device) -> VecMap<GpuMeshLayer> {
        self.meshes.iter().map(|(i, c)| (i, c.make_buffers(device))).collect()
    }

    pub fn tile_desc(&self, ty: LayerType) -> &LayerParams {
        &self.tiles.layers[ty]
    }
}
