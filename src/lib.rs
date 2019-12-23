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
mod cache;
mod runtime_texture;
mod srgb;
mod terrain;
mod utils;

// pub mod plugin;
// pub mod compute;

pub use generate::{GridSpacing, QuadTreeBuilder, TextureQuality, VertexQuality};
pub use terrain::quadtree::QuadTree;

use crate::terrain::quadtree::render::NodeState;
use std::mem;

#[repr(C)]
#[derive(Copy, Clone)]
struct UniformBlock {
    view_proj: mint::ColumnMatrix4<f32>,
    camera: mint::Point3<f32>,
    padding: f32,
}
unsafe impl bytemuck::Pod for UniformBlock {}
unsafe impl bytemuck::Zeroable for UniformBlock {}

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

    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_buffer_partial: wgpu::Buffer,
    layers: Vec<wgpu::Texture>,

    quadtree: QuadTree,
}
impl Terrain {
    pub fn new(device: &wgpu::Device, quadtree: QuadTree) -> Self {
        let mut watcher = rshader::ShaderDirectoryWatcher::new("src/shaders").unwrap();
        let shader = rshader::ShaderSet::simple(
            &mut watcher,
            rshader::shader_source!("shaders", "version", "a.vert"),
            rshader::shader_source!("shaders", "version", "a.frag"),
        )
        .unwrap();

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: std::mem::size_of::<UniformBlock>() as u64,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::UNIFORM,
        });
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: (std::mem::size_of::<NodeState>() * quadtree.total_nodes()) as u64,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::VERTEX,
        });
        let (index_buffer, index_buffer_partial) = quadtree.create_index_buffers(device);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &uniform_buffer,
                    range: 0..mem::size_of::<UniformBlock>() as u64,
                },
            }],
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            });

        Self {
            bind_group_layout,
            bind_group,
            render_pipeline_layout,
            render_pipeline: None,
            watcher,
            shader,

            uniform_buffer,
            vertex_buffer,
            index_buffer,
            index_buffer_partial,
            layers: Vec::new(),

            quadtree,
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        frame: &wgpu::SwapChainOutput,
        view_proj: mint::ColumnMatrix4<f32>,
        camera: mint::Point3<f32>,
    ) {
        self.quadtree.update(camera, None);
        if self.shader.refresh(&mut self.watcher) {
            self.render_pipeline = None;
        }

        if self.render_pipeline.is_none() {
            self.render_pipeline = Some(
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: &self.render_pipeline_layout,
                    vertex_stage: wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(
                            &wgpu::read_spirv(std::io::Cursor::new(self.shader.vertex())).unwrap(),
                        ),
                        entry_point: "main",
                    },
                    fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(
                            &wgpu::read_spirv(std::io::Cursor::new(self.shader.fragment()))
                                .unwrap(),
                        ),
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
                    vertex_buffers: &[wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<NodeState>() as u64,
                        step_mode: wgpu::InputStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttributeDescriptor {
                                offset: 0,
                                format: wgpu::VertexFormat::Float2,
                                shader_location: 0,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 8,
                                format: wgpu::VertexFormat::Float,
                                shader_location: 1,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 12,
                                format: wgpu::VertexFormat::Float,
                                shader_location: 2,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 16,
                                format: wgpu::VertexFormat::Float3,
                                shader_location: 3,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 28,
                                format: wgpu::VertexFormat::Float2,
                                shader_location: 4,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 36,
                                format: wgpu::VertexFormat::Float2,
                                shader_location: 5,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 44,
                                format: wgpu::VertexFormat::Float2,
                                shader_location: 6,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 52,
                                format: wgpu::VertexFormat::Float2,
                                shader_location: 7,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 60,
                                format: wgpu::VertexFormat::Float2,
                                shader_location: 8,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 68,
                                format: wgpu::VertexFormat::Float,
                                shader_location: 9,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 72,
                                format: wgpu::VertexFormat::Float,
                                shader_location: 10,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 76,
                                format: wgpu::VertexFormat::Int,
                                shader_location: 11,
                            },
                        ],
                    }],
                    sample_count: 1,
                    sample_mask: !0,
                    alpha_to_coverage_enabled: false,
                }),
            );
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        self.quadtree
            .prepare_vertex_buffer(device, &mut encoder, &mut self.vertex_buffer);

        let mapped = device.create_buffer_mapped(
            mem::size_of::<UniformBlock>(),
            wgpu::BufferUsage::MAP_WRITE | wgpu::BufferUsage::COPY_SRC,
        );
        bytemuck::cast_slice_mut(mapped.data)[0] = UniformBlock {
            view_proj,
            camera,
            padding: 0.0,
        };
        encoder.copy_buffer_to_buffer(
            &mapped.finish(),
            0,
            &self.uniform_buffer,
            0,
            mem::size_of::<UniformBlock>() as u64,
        );

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

            self.quadtree.render(
                &mut rpass,
                &self.vertex_buffer,
                &self.index_buffer,
                &self.index_buffer_partial,
            );
        }

        queue.submit(&[encoder.finish()]);
    }
}
