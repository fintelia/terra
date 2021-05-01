use crate::{
    cache::{
        GeneratorMask, LayerMask, Priority, PriorityCache, PriorityCacheEntry, SingularLayerType,
        UnifiedPriorityCache,
        TextureFormat,
    },
    generate::ComputeShader,
    gpu_state::GpuState,
    terrain::quadtree::VNode,
};
use cgmath::Vector3;

pub(super) struct Entry {
    priority: Priority,
    node: VNode,
    pub(super) valid: bool,
    pub(super) generators: GeneratorMask,
}
impl Entry {
    fn new(node: VNode, priority: Priority) -> Self {
        Self { node, priority, valid: false, generators: GeneratorMask::empty() }
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

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct SingularLayerGenerateUniforms {
    input_slot: u32,
    output_slot: u32,
}
unsafe impl bytemuck::Zeroable for SingularLayerGenerateUniforms {}
unsafe impl bytemuck::Pod for SingularLayerGenerateUniforms {}

pub(crate) struct SingularLayerDesc {
    pub generate: ComputeShader<SingularLayerGenerateUniforms>,
    pub cache_size: usize,
    pub dependency_mask: LayerMask,
    pub level: u8,
    pub ty: SingularLayerType,
    pub texture_resolution: u32,
    pub texture_format: TextureFormat,
}

pub(crate) struct SingularLayerCache {
    pub(super) inner: PriorityCache<Entry>,
    pub(super) desc: SingularLayerDesc,
}
impl SingularLayerCache {
    pub fn new(desc: SingularLayerDesc) -> Self {
        Self { inner: PriorityCache::new(desc.cache_size), desc }
    }

    pub fn update(&mut self, camera: mint::Point3<f64>) {
        let camera = Vector3::new(camera.x, camera.y, camera.z);

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
            if node.level() == self.desc.level
                && !self.inner.contains(&node)
                && (priority > min_priority || !self.inner.is_full())
            {
                missing.push(Entry::new(node, priority));
            }

            node.level() < self.desc.level
        });
        self.inner.insert(missing);
    }

    pub(super) fn generate_all(
        cache: &mut UnifiedPriorityCache,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_state: &GpuState,
    ) {
        let mut generated = Vec::new();
        let mut command_buffers = Vec::new();
        for layer_type in SingularLayerType::iter() {
            let m = &mut cache.textures[layer_type];

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{}.command_encoder", layer_type.name())),
            });

            for (index, entry) in m.inner.slots_mut().into_iter().enumerate() {
                if entry.valid || entry.priority < Priority::cutoff() {
                    continue;
                }
                if !cache.tiles.contains_all(entry.node, m.desc.dependency_mask) {
                    continue;
                }

                m.desc.generate.run(
                    device,
                    &mut encoder,
                    gpu_state,
                    ((m.desc.texture_resolution + 7) / 8, (m.desc.texture_resolution + 7) / 8, 1),
                    &SingularLayerGenerateUniforms {
                        input_slot: cache.tiles.get_slot(entry.node).unwrap() as u32,
                        output_slot: index as u32,
                    },
                );
                entry.valid = true;
                generated.push((layer_type, entry.node));
            }
            command_buffers.push(encoder.finish());
        }
        for (layer_type, node) in generated {
            cache.textures[layer_type].inner.entry_mut(&node).unwrap().generators =
                cache.generator_dependencies(node, cache.textures[layer_type].desc.dependency_mask);
        }

        queue.submit(command_buffers);
    }

    pub(super) fn make_cache_texture(&self, device: &wgpu::Device) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: self.desc.texture_resolution,
                height: self.desc.texture_resolution,
                depth_or_array_layers: self.desc.cache_size as u32,
            },
            format: self.desc.texture_format.to_wgpu(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::SAMPLED
                | if !self.desc.texture_format.is_compressed() {
                    wgpu::TextureUsage::STORAGE
                } else {
                    wgpu::TextureUsage::empty()
                },
            label: Some(&format!("texture.singular.{}", self.desc.ty.name())),
        })
    }
}
