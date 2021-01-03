use maplit::hashmap;

use crate::GpuState;
use std::mem;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct GenHeightmapsUniforms {
    pub position: [i32; 2],
    pub origin: [i32; 2],
    pub spacing: f32,
    pub in_slot: i32,
    pub out_slot: i32,
    pub level_resolution: i32,
    pub face: u32,
}
unsafe impl bytemuck::Zeroable for GenHeightmapsUniforms {}
unsafe impl bytemuck::Pod for GenHeightmapsUniforms {}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct GenDisplacementsUniforms {
    pub node_center: [f64; 3],
    pub padding0: f64,
    pub origin: [i32; 2],
    pub position: [i32; 2],
    pub stride: i32,
    pub heightmaps_slot: i32,
    pub displacements_slot: i32,
    pub face: i32,
    pub level_resolution: u32,
}
unsafe impl bytemuck::Zeroable for GenDisplacementsUniforms {}
unsafe impl bytemuck::Pod for GenDisplacementsUniforms {}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct GenNormalsUniforms {
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

    pub fn refresh(&mut self) -> bool {
        if self.shader.refresh() {
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
                hashmap!["ubo" => (false, wgpu::BindingResource::Buffer {
                    buffer: &self.uniforms,
                    offset: 0,
                    size: None,
                })],
            );
            self.bindgroup_pipeline = Some((
                bind_group,
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        bind_group_layouts: [&bind_group_layout][..].into(),
                        push_constant_ranges: &[],
                        label: None,
                    })),
                    compute_stage: wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                            label: None,
                            source: wgpu::ShaderSource::SpirV(self.shader.compute().into()),
                            flags: wgpu::ShaderFlags::empty(),
                        }),
                        entry_point: "main".into(),
                    },
                    label: None,
                }),
            ));
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            size: mem::size_of::<U>() as u64,
            usage: wgpu::BufferUsage::COPY_SRC,
            label: None,
            mapped_at_creation: true,
        });
        let mut buffer_view = staging.slice(..).get_mapped_range_mut();
        buffer_view[..mem::size_of::<U>()].copy_from_slice(bytemuck::bytes_of(uniforms));
        drop(buffer_view);
        staging.unmap();

        encoder.copy_buffer_to_buffer(&staging, 0, &self.uniforms, 0, mem::size_of::<U>() as u64);

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
        cpass.set_bind_group(0, &self.bindgroup_pipeline.as_ref().unwrap().0, &[]);
        cpass.dispatch(dimensions.0, dimensions.1, dimensions.2);
    }
}
