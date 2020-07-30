use crate::GpuState;
use std::mem;

#[derive(Copy, Clone)]
pub(crate) struct GenHeightmapsUniforms {
    pub position: [f32; 2],
    pub origin: [i32; 2],
    pub spacing: f32,
    pub in_slot: i32,
    pub out_slot: i32,
}
unsafe impl bytemuck::Zeroable for GenHeightmapsUniforms {}
unsafe impl bytemuck::Pod for GenHeightmapsUniforms {}

#[derive(Copy, Clone)]
pub(crate) struct GenDisplacementsUniforms {
    pub origin: [i32; 2],
    pub stride: i32,
    pub heightmaps_slot: i32,
    pub displacements_slot: i32,
}
unsafe impl bytemuck::Zeroable for GenDisplacementsUniforms {}
unsafe impl bytemuck::Pod for GenDisplacementsUniforms {}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct GenNormalsUniforms {
    pub cspace_origin: [f32; 4],
    pub cspace_dx: [f32; 4],
    pub cspace_dy: [f32; 4],
    pub heightmaps_origin: [i32; 2],
    pub parent_origin: [u32; 2],
    pub heightmaps_slot: i32,
    pub normals_slot: i32,
    pub albedo_slot: i32,
    pub parent_slot: i32,
    pub spacing: f32,
    pub padding: i32,
}
unsafe impl bytemuck::Zeroable for GenNormalsUniforms {}
unsafe impl bytemuck::Pod for GenNormalsUniforms {}

pub(crate) struct ComputeShader<U> {
    shader: rshader::ShaderSet,
    bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::ComputePipeline)>,
    uniforms: wgpu::Buffer,
    _phantom: std::marker::PhantomData<U>,
}
impl<U: bytemuck::Pod> ComputeShader<U> {
    pub fn new(device: &wgpu::Device, shader: rshader::ShaderSet) -> Self {
        Self {
            shader,
            bindgroup_pipeline: None,
            uniforms: device.create_buffer(&wgpu::BufferDescriptor {
                size: mem::size_of::<U>() as u64,
                usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::UNIFORM,
                mapped_at_creation: false,
                label: None,
            }),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn refresh(&mut self, watcher: &mut rshader::ShaderDirectoryWatcher) -> bool {
        if self.shader.refresh(watcher) {
            self.bindgroup_pipeline = None;
            true
        } else {
            false
        }
    }

    pub fn run(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        state: &GpuState,
        dimensions: (u32, u32, u32),
        uniforms: &U,
    ) {
        if self.bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = state.bind_group_for_shader(
                device,
                &self.shader,
                Some(self.uniforms.slice(..mem::size_of::<U>() as u64)),
            );
            self.bindgroup_pipeline = Some((
                bind_group,
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    layout: &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    }),
                    compute_stage: wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
                            self.shader.compute(),
                        )),
                        entry_point: "main",
                    },
                }),
            ));
        }

        let staging = device
            .create_buffer_with_data(bytemuck::bytes_of(uniforms), wgpu::BufferUsage::COPY_SRC);
        encoder.copy_buffer_to_buffer(&staging, 0, &self.uniforms, 0, mem::size_of::<U>() as u64);

        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
        cpass.set_bind_group(0, &self.bindgroup_pipeline.as_ref().unwrap().0, &[]);
        cpass.dispatch(dimensions.0, dimensions.1, dimensions.2);
    }
}
