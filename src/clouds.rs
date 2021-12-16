pub(crate) struct Cloudscape {
    render_size: (u32, u32),
    render_target: wgpu::Texture,
    shader: rshader::ShaderSet,
    bind_groups: Option<[wgpu::BindGroup; 2]>,
    pipeline: Option<RenderPipeline>,

    taa_targets: [wgpu::Texture; 2],
}
impl Cloudscape {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = rshader::ShaderSet::simple(
            rshader::shader_source!("shaders", "cloudscape.vert", "declarations.glsl"),
            rshader::shader_source!("shaders", "cloudscape.frag", "declarations.glsl"),
        )
        .unwrap();

        let render_target = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("cloud_render"),
        });

        Self {
            render_size: (width, height),
            render_target,
            shader,
            bind_groups: None,
            pipeline: None,
            taa_targets,
        }
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_buffer: &wgpu::TextureView,
        depth_buffer: &wgpu::TextureView,
        _frame_size: (u32, u32),
        view_proj: mint::ColumnMatrix4<f32>,
        camera: mint::Point3<f64>,
    ) {
        if self.bind_groups.is_none() {
            let (bind_group, bind_group_layout) = self.gpu_state.bind_group_for_shader(
                device,
                &self.shader,
                HashMap::new(),
                HashMap::new(),
                "clouds",
            );
        }
    }
}
