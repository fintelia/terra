//! Terra is a large scale terrain generation and rendering library built on top of rendy.
#![feature(custom_attribute)]
#![feature(stmt_expr_attributes)]
#![feature(float_to_from_bytes)]

#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate rshader;

mod coordinates;
mod generate;
// mod graph;
mod terrain;
mod utils;
mod runtime_texture;
mod srgb;
mod cache;

// pub mod plugin;
// pub mod compute;

pub use generate::{QuadTreeBuilder, VertexQuality, TextureQuality, GridSpacing};
pub use terrain::quadtree::QuadTree;

pub struct Terrain {
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,

    render_pipeline_layout: wgpu::PipelineLayout,
    render_pipeline: Option<wgpu::RenderPipeline>,

    // set_layouts: Vec<Handle<DescriptorSetLayout<B>>>,
    // descriptor_sets: Vec<Escape<DescriptorSet<B>>>,
    // pipeline_layout: B::PipelineLayout,
    // graphics_pipeline: B::GraphicsPipeline,

    watcher: rshader::ShaderDirectoryWatcher,
    shader: rshader::ShaderSet,

    // uniform_buffer: wgpu::Buffer,
    // vertex_buffer: wgpu::Buffer,
    // index_buffer: wgpu::Buffer,
    // index_buffer_partial: wgpu::Buffer,
    // layers: Vec<wgpu::Texture>,

    quadtree: QuadTree,
}
impl Terrain {
    pub fn new(device: &wgpu::Device, quadtree: QuadTree) -> Self {
        let mut watcher = rshader::ShaderDirectoryWatcher::new("src/shaders").unwrap();
        let shader = rshader::ShaderSet::simple(
            &mut watcher,
            rshader::shader_source!("src/shaders", "tri.vert"),
            rshader::shader_source!("src/shaders", "tri.frag"),
        ).unwrap();

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { bindings: &[] });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[],
        });
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        Self {
            bind_group_layout,
            bind_group,
            render_pipeline_layout,
            render_pipeline: None,
            watcher,
            shader,
            quadtree,
        }
    }

    pub fn render(&mut self, device: &wgpu::Device, queue: &mut wgpu::Queue, frame: &wgpu::SwapChainOutput) {
        if self.shader.refresh(&mut self.watcher) {
            self.render_pipeline = None;
        }

        if self.render_pipeline.is_none() {
            let vs_module = device.create_shader_module(
                &wgpu::read_spirv(std::io::Cursor::new(self.shader.vertex())).unwrap());
            let fs_module = device.create_shader_module(
                &wgpu::read_spirv(std::io::Cursor::new(self.shader.fragment())).unwrap());

            self.render_pipeline = Some(
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: &self.render_pipeline_layout,
                    vertex_stage: wgpu::ProgrammableStageDescriptor {
                        module: &vs_module,
                        entry_point: "main",
                    },
                    fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                        module: &fs_module,
                        entry_point: "main",
                    }),
                    rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: wgpu::CullMode::None,
                        depth_bias: 0,
                        depth_bias_slope_scale: 0.0,
                        depth_bias_clamp: 0.0,
                    }),
                    primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                    color_states: &[wgpu::ColorStateDescriptor {
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
                        color_blend: wgpu::BlendDescriptor::REPLACE,
                        alpha_blend: wgpu::BlendDescriptor::REPLACE,
                        write_mask: wgpu::ColorWrite::ALL,
                    }],
                    depth_stencil_state: None,
                    index_format: wgpu::IndexFormat::Uint16,
                    vertex_buffers: &[],
                    sample_count: 1,
                    sample_mask: !0,
                    alpha_to_coverage_enabled: false,
                }));
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::GREEN,
                }],
                depth_stencil_attachment: None,
            });
            rpass.set_pipeline(self.render_pipeline.as_ref().unwrap());
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0 .. 3, 0 .. 1);
        }

        queue.submit(&[encoder.finish()]);
    }
}
