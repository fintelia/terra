use std::collections::HashMap;

use crate::gpu_state::GpuState;

pub(crate) struct Cloudscape {
    shader: rshader::ShaderSet,
    bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,
}
impl Cloudscape {
    pub fn new() -> Self {
        let shader = rshader::ShaderSet::simple(
            rshader::shader_source!("shaders", "cloudscape.vert", "declarations.glsl"),
            rshader::shader_source!("shaders", "cloudscape.frag", "declarations.glsl"),
        )
        .unwrap();

        Self {
            shader,
            bindgroup_pipeline: None,
        }
    }

    pub fn render<'a>(
        &'a mut self,
        device: &wgpu::Device,
        rpass: &mut wgpu::RenderPass<'a>,
        gpu_state: &GpuState,
    ) {
        if self.shader.refresh() {
            self.bindgroup_pipeline = None;
        }
        if self.bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = gpu_state.bind_group_for_shader(
                device,
                &self.shader,
                HashMap::new(),
                HashMap::new(),
                "clouds",
            );

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pipeline_layout.cloudscape"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("rendertarget.cloudscape"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("shader.cloudscape.vertex"),
                        source: self.shader.vertex(),
                    }),
                    entry_point: "main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("shader.cloudscape.fragment"),
                        source: self.shader.fragment(),
                    }),
                    entry_point: "main",
                    targets: &[wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }],
                }),
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_compare: wgpu::CompareFunction::Always,
                    depth_write_enabled: false,
                    bias: Default::default(),
                    stencil: Default::default(),
                }),
                multisample: Default::default(),
                multiview: None,
            });

            self.bindgroup_pipeline = Some((bind_group, pipeline));
        }

        // let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        //     label: Some("renderpass.cloudscape"),
        //     color_attachments: &[wgpu::RenderPassColorAttachment {
        //         view: color_buffer,
        //         resolve_target: None,
        //         ops: wgpu::Operations {
        //             load: wgpu::LoadOp::Load,
        //             store: true,
        //         },
        //     }],
        //     depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
        //         view: depth_buffer,
        //         depth_ops: None,
        //         stencil_ops: None,
        //     }),
        // });

        rpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
        rpass.set_bind_group(0, &self.bindgroup_pipeline.as_ref().unwrap().0, &[]);
        rpass.draw(0..3, 0..1);
    }
}
