pub(crate) struct GpuState {
    pub base_heights: wgpu::Texture,

    pub heights_staging: wgpu::Texture,
    pub normals_staging: wgpu::Texture,

    pub noise: wgpu::Texture,
    pub _planet_mesh_texture: wgpu::Texture,

    pub heights: wgpu::Texture,
    pub normals: wgpu::Texture,
    pub albedo: wgpu::Texture,
}
impl GpuState {
    pub(crate) fn bind_group_for_shader(
        &self,
        device: &wgpu::Device,
        shader: &rshader::ShaderSet,
        ubo: Option<&wgpu::BindingResource>,
    ) -> (wgpu::BindGroup, wgpu::BindGroupLayout) {
        let linear = &device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare_function: wgpu::CompareFunction::Always,
        });
        let linear_wrap = &device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare_function: wgpu::CompareFunction::Always,
        });

        let base_heights = &self.base_heights.create_default_view();
        let heights_staging = &self.heights_staging.create_default_view();
        let normals_staging = &self.normals_staging.create_default_view();
        let noise = &self.noise.create_default_view();
        let heights = &self.heights.create_default_view();
        let normals = &self.normals.create_default_view();
        let albedo = &self.albedo.create_default_view();

        let bind_group_layout = device.create_bind_group_layout(&shader.layout_descriptor());
        let mut bindings = Vec::new();
        for (name, layout) in
            shader.desc_names().iter().zip(shader.layout_descriptor().bindings.iter())
        {
            let name = &**name.as_ref().unwrap();
            bindings.push(wgpu::Binding {
                binding: layout.binding,
                resource: match layout.ty {
                    wgpu::BindingType::Sampler => wgpu::BindingResource::Sampler(match name {
                        "linear" => &linear,
                        "linear_wrap" => &linear_wrap,
                        _ => unreachable!("unrecognized sampler: {}", name),
                    }),
                    wgpu::BindingType::UniformBuffer { .. } => ubo.cloned().unwrap(),
                    wgpu::BindingType::StorageTexture { .. }
                    | wgpu::BindingType::SampledTexture { .. } => {
                        wgpu::BindingResource::TextureView(match name {
                            "base_heights" => base_heights,
                            "heights_staging" => heights_staging,
                            "normals_staging" => normals_staging,
                            "noise" => noise,
                            "heights" => heights,
                            "normals" => normals,
                            "albedo" => albedo,
                            _ => unreachable!("unrecognized image: {}", name),
                        })
                    }
                    wgpu::BindingType::StorageBuffer { .. } => unimplemented!(),
                },
            })
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &*bindings,
        });

        (bind_group, bind_group_layout)
    }
}
