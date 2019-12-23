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
    bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,

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

        let vs_module = device.create_shader_module(
            &wgpu::read_spirv(std::io::Cursor::new(shader.vertex())).unwrap());

        let fs_module = device.create_shader_module(
            &wgpu::read_spirv(std::io::Cursor::new(shader.fragment())).unwrap());

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { bindings: &[] });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
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
        });

        Self {
            bind_group,
            render_pipeline,
            quadtree,
        }
    }

    pub fn render(&mut self, device: &wgpu::Device, queue: &mut wgpu::Queue, frame: &wgpu::SwapChainOutput) {
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
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0 .. 3, 0 .. 1);
        }

        queue.submit(&[encoder.finish()]);
    }
}

// async fn run(device: &wgpu::Device, queue: &mut wgpu::Queue) {
//     use std::{convert::TryInto as _};
//     use zerocopy::AsBytes as _;

//     env_logger::init();

//     // For now this just panics if you didn't pass numbers. Could add proper error handling.
//     let numbers: Vec<f32> = vec![0.0; 1024 * 1024];

//     let slice_size = numbers.len() * std::mem::size_of::<f32>();
//     let size = slice_size as wgpu::BufferAddress;

//     let cs = create_compute_shader(include_str!("shader.comp"));
//     let cs_module =
//         device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&cs[..])).unwrap());

//     let staging_buffer = device.create_buffer_with_data(
//         numbers.as_slice().as_bytes(),
//         wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
//     );

//     let input_texture = device.create_texture(&wgpu::TextureDescriptor {
//         size: wgpu::Extent3d {width: 1024, height: 1024, depth: 1},
//         array_layer_count: 1,
//         mip_level_count: 1,
//         sample_count: 1,
//         dimension: wgpu::TextureDimension::D2,
//         format: wgpu::TextureFormat::R32Float,
//         usage: wgpu::TextureUsage::STORAGE
//             | wgpu::TextureUsage::COPY_DST
//             | wgpu::TextureUsage::COPY_SRC,
//     });

//     let output_texture = device.create_texture(&wgpu::TextureDescriptor {
//         size: wgpu::Extent3d {width: 1024, height: 1024, depth: 1},
//         array_layer_count: 1,
//         mip_level_count: 1,
//         sample_count: 1,
//         dimension: wgpu::TextureDimension::D2,
//         format: wgpu::TextureFormat::R32Float,
//         usage: wgpu::TextureUsage::STORAGE
//             | wgpu::TextureUsage::COPY_DST
//             | wgpu::TextureUsage::COPY_SRC,
//     });

//     let input_texture_view = input_texture.create_default_view();
//     let output_texture_view = output_texture.create_default_view();

//     let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
//         bindings: &[
//             wgpu::BindGroupLayoutBinding {
//                 binding: 0,
//                 visibility: wgpu::ShaderStage::COMPUTE,
//                 ty: wgpu::BindingType::StorageTexture {
//                     dimension: wgpu::TextureViewDimension::D2,
//                 },
//             },
//             wgpu::BindGroupLayoutBinding {
//                 binding: 0,
//                 visibility: wgpu::ShaderStage::COMPUTE,
//                 ty: wgpu::BindingType::StorageTexture {
//                     dimension: wgpu::TextureViewDimension::D2,
//                 },
//             }
//         ],
//     });

//     let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
//         layout: &bind_group_layout,
//         bindings: &[
//             wgpu::Binding {
//                 binding: 0,
//                 resource: wgpu::BindingResource::TextureView(&input_texture_view),
//             },
//             wgpu::Binding {
//                 binding: 0,
//                 resource: wgpu::BindingResource::TextureView(&output_texture_view),
//             }
//         ],
//     });

//     let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
//         bind_group_layouts: &[&bind_group_layout],
//     });

//     let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
//         layout: &pipeline_layout,
//         compute_stage: wgpu::ProgrammableStageDescriptor {
//             module: &cs_module,
//             entry_point: "main",
//         },
//     });

//     let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
//     encoder.copy_buffer_to_texture(
//         wgpu::BufferCopyView {
//             buffer: &staging_buffer,
//             offset: 0,
//             row_pitch: 4096,
//             image_height: 1024,
//         },
//         wgpu::TextureCopyView {
//             texture: &input_texture,
//             mip_level: 0,
//             array_layer: 0,
//             origin: wgpu::Origin3d {x: 0.0, y: 0.0, z: 0.0 },
//         },
//         wgpu::Extent3d { width: 1024, height: 1024, depth: 1 },
//     );
//     {
//         let mut cpass = encoder.begin_compute_pass();
//         cpass.set_pipeline(&compute_pipeline);
//         cpass.set_bind_group(0, &bind_group, &[]);
//         cpass.dispatch(1024, 1024, 1);
//     }
//     encoder.copy_texture_to_buffer(
//         wgpu::TextureCopyView {
//             texture: &output_texture,
//             mip_level: 0,
//             array_layer: 0,
//             origin: wgpu::Origin3d {x: 0.0, y: 0.0, z: 0.0 },
//         },
//         wgpu::BufferCopyView {
//             buffer: &staging_buffer,
//             offset: 0,
//             row_pitch: 4096,
//             image_height: 1024,
//         },
//         wgpu::Extent3d { width: 1024, height: 1024, depth: 1 },
//     );

//     queue.submit(&[encoder.finish()]);

//     if let Ok(mapping) = staging_buffer.map_read(0u64, size).await {
//         let times : Box<[f32]> = mapping
//             .as_slice()
//             .chunks_exact(4)
//             .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
//             .collect();

//         println!("Times: {:?}", times);
//     }
// }
