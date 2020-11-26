use std::collections::HashMap;

use crate::terrain::tile_cache::LayerType;
use vec_map::VecMap;

pub(crate) struct GpuState {
    pub noise: wgpu::Texture,
    pub sky: wgpu::Texture,
    pub transmittance: wgpu::Texture,
    pub inscattering: wgpu::Texture,

    pub tile_cache: VecMap<wgpu::Texture>,

    pub bc4_staging: wgpu::Texture,
    pub bc5_staging: wgpu::Texture,
}
impl GpuState {
    pub(crate) fn bind_group_for_shader(
        &self,
        device: &wgpu::Device,
        shader: &rshader::ShaderSet,
        uniform_buffers: HashMap<&str, (bool, wgpu::BindingResource)>,
    ) -> (wgpu::BindGroup, wgpu::BindGroupLayout) {
        let linear = &device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            label: Some("linear".into()),
            ..Default::default()
        });
        let linear_wrap = &device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            label: Some("linear_wrap".into()),
            ..Default::default()
        });

        let noise = &self.noise.create_view(&Default::default());
        let sky = &self.sky.create_view(&Default::default());
        let transmittance = &self.transmittance.create_view(&Default::default());
        let inscattering = &self.inscattering.create_view(&Default::default());
        let bc4_staging = &self.bc4_staging.create_view(&Default::default());
        let bc5_staging = &self.bc5_staging.create_view(&Default::default());
        let tile_cache_views: VecMap<_> = self
            .tile_cache
            .iter()
            .map(|(i, tex)| (i, tex.create_view(&Default::default())))
            .collect();

        let mut layout_descriptor_entries = shader.layout_descriptor().entries.to_vec();
        let mut bindings = Vec::new();
        for (name, layout) in shader.desc_names().iter().zip(layout_descriptor_entries.iter_mut()) {
            let name = &**name.as_ref().unwrap();
            bindings.push(wgpu::BindGroupEntry {
                binding: layout.binding,
                resource: match layout.ty {
                    wgpu::BindingType::Sampler { comparison: _ } => {
                        wgpu::BindingResource::Sampler(match name {
                            "linear" => &linear,
                            "linear_wrap" => &linear_wrap,
                            _ => unreachable!("unrecognized sampler: {}", name),
                        })
                    }
                    wgpu::BindingType::StorageTexture { .. }
                    | wgpu::BindingType::SampledTexture { .. } => {
                        wgpu::BindingResource::TextureView(match name {
                            "noise" => noise,
                            "sky" => sky,
                            "transmittance" => transmittance,
                            "inscattering" => inscattering,
                            "displacements" => &tile_cache_views[LayerType::Displacements],
                            "albedo" => &tile_cache_views[LayerType::Albedo],
                            "roughness" => &tile_cache_views[LayerType::Roughness],
                            "normals" => &tile_cache_views[LayerType::Normals],
                            "heightmaps" => &tile_cache_views[LayerType::Heightmaps],
                            "bc4_staging" => &bc4_staging,
                            "bc5_staging" => &bc5_staging,
                            _ => unreachable!("unrecognized image: {}", name),
                        })
                    }
                    wgpu::BindingType::UniformBuffer { ref mut dynamic, .. } => {
                        let (d, ref buf) = uniform_buffers[name];
                        *dynamic = d;
                        buf.clone()
                    }
                    wgpu::BindingType::StorageBuffer { .. } => unimplemented!(),
                },
            });
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &layout_descriptor_entries,
            label: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &*bindings,
            label: None,
        });

        (bind_group, bind_group_layout)
    }
}
