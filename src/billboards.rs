use std::{collections::HashMap, num::NonZeroU32};

use wgpu::util::DeviceExt;

use crate::{gpu_state::GpuState, speedtree_xml::{SpeedTreeModel, parse_xml}};

const RESOLUTION: u32 = 1024;
const FRAMES_PER_SIDE: u32 = 6;

// #[derive(Copy, Clone, Debug, Default)]
// #[repr(C)]
// struct Vertex {
//     position: [f32; 3],
//     texcoord_u: f32,
//     normal: [f32; 3],
//     texcoord_v: f32,
// }
// unsafe impl bytemuck::Pod for Vertex {}
// unsafe impl bytemuck::Zeroable for Vertex {}

pub(crate) struct Models {
    tree: SpeedTreeModel,
}
impl Models {
    pub fn new() -> Self {
        let tree = parse_xml(include_str!("../assets/Tree/Oak_English_Sapling.xml")).unwrap();

        Self {
            tree,
        }
        // let model = unimplemented!();//proctree_rs::generate_tree(0);

        // let mut vertices =
        //     Vec::with_capacity(model.vertices.len() / 3 + model.twig_vertices.len() / 3);
        // let mut indices = Vec::with_capacity(model.faces.len() + model.twig_faces.len());

        // for i in 0..(model.vertices.len() / 3) {
        //     vertices.push(Vertex {
        //         position: [
        //             model.vertices[i * 3],
        //             model.vertices[i * 3 + 1],
        //             model.vertices[i * 3 + 2],
        //         ],
        //         normal: [model.normals[i * 3], model.normals[i * 3 + 1], model.normals[i * 3 + 2]],
        //         texcoord_u: model.texcoords[i * 2],
        //         texcoord_v: model.texcoords[i * 2 + 1] * 0.5,
        //     })
        // }
        // let twig_index_offset = vertices.len() as u32;
        // for i in 0..(model.twig_vertices.len() / 3) {
        //     vertices.push(Vertex {
        //         position: [
        //             model.twig_vertices[i * 3],
        //             model.twig_vertices[i * 3 + 1],
        //             model.twig_vertices[i * 3 + 2],
        //         ],
        //         normal: [
        //             model.twig_normals[i * 3],
        //             model.twig_normals[i * 3 + 1],
        //             model.twig_normals[i * 3 + 2],
        //         ],
        //         texcoord_u: model.twig_texcoords[i * 2],
        //         texcoord_v: model.twig_texcoords[i * 2 + 1] * 0.5 + 0.5,
        //     })
        // }
        // for i in 0..model.faces.len() {
        //     indices.push(model.faces[i] as u32);
        // }
        // for i in 0..model.twig_faces.len() {
        //     indices.push(model.twig_faces[i] as u32 + twig_index_offset);
        // }

        // Self { vertices, indices }
    }

    pub fn make_buffers(&self, device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer) {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer.tree.vertex"),
            contents: bytemuck::cast_slice(&self.tree.vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer.tree.index"),
            contents: bytemuck::cast_slice(&self.tree.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        (vertex_buffer, index_buffer)
    }

    pub fn make_models_albedo(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
        let img = image::load_from_memory(include_bytes!("../assets/Tree/Oak_English_Sapling_Color.png")).unwrap();
        let albedo = img.into_rgba8();
        device.create_texture_with_data(queue, &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 4096,
                height: 4096,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
        }, &*albedo)
    }

    fn default_billboard_desc() -> wgpu::TextureDescriptor<'static> {
        wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: RESOLUTION * FRAMES_PER_SIDE,
                height: RESOLUTION * FRAMES_PER_SIDE,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }
    pub fn make_billboards_albedo(&self, device: &wgpu::Device) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture.billboards.albedo"),
            format: wgpu::TextureFormat::Rgba8Unorm,
            ..Self::default_billboard_desc()
        })
    }
    pub fn make_billboards_normals(&self, device: &wgpu::Device) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture.billboards.normals"),
            format: wgpu::TextureFormat::Rg8Snorm,
            ..Self::default_billboard_desc()
        })
    }
    pub fn make_billboards_depth(&self, device: &wgpu::Device) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture.billboards.depth"),
            format: wgpu::TextureFormat::R16Float,
            ..Self::default_billboard_desc()
        })
    }

    fn default_topdown_desc() -> wgpu::TextureDescriptor<'static> {
        wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: RESOLUTION,
                height: RESOLUTION,
                depth_or_array_layers: 1,
            },
            ..Self::default_billboard_desc()
        }
    }
    pub fn make_topdown_albedo(&self, device: &wgpu::Device) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture.topdown.albedo"),
            format: wgpu::TextureFormat::Rgba8Unorm,
            ..Self::default_topdown_desc()
        })
    }
    pub fn make_topdown_normals(&self, device: &wgpu::Device) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture.topdown.normals"),
            format: wgpu::TextureFormat::Rg8Snorm,
            ..Self::default_topdown_desc()
        })
    }
    pub fn make_topdown_depth(&self, device: &wgpu::Device) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture.topdown.depth"),
            format: wgpu::TextureFormat::R16Float,
            ..Self::default_topdown_desc()
        })
    }

    pub fn render_billboards(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_state: &GpuState,
    ) {
        let shader = rshader::ShaderSet::simple(
            rshader::shader_source!("shaders", "model.vert", "declarations.glsl"),
            rshader::shader_source!("shaders", "model.frag", "declarations.glsl"),
        )
        .unwrap();

        let (bind_group, bind_group_layout) = gpu_state.bind_group_for_shader(
            device,
            &shader,
            HashMap::new(),
            HashMap::new(),
            "model",
        );
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: [&bind_group_layout][..].into(),
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::VERTEX,
                    range: 0..8,
                }],
                label: Some("pipeline.billboard-texture.layout"),
            });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                    label: Some("shader.billboard-texture.vertex"),
                    source: shader.vertex(),
                }),
                entry_point: "main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                    label: Some("shader.billboard-texture.fragment"),
                    source: shader.fragment(),
                }),
                entry_point: "main",
                targets: &[
                    wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent::REPLACE,
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    },
                    wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg8Snorm,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent::REPLACE,
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    },
                    wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent::REPLACE,
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    },
                ],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                depth_write_enabled: false,
                bias: Default::default(),
                stencil: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            label: Some("pipeline.tree-billboards"),
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder.tree-billboards"),
        });

        // Billboards
        for base_array_layer in 0..1 {
            let view_desc = wgpu::TextureViewDescriptor {
                base_array_layer,
                array_layer_count: Some(NonZeroU32::new(1).unwrap()),
                ..Default::default()
            };
            let albedo_view = gpu_state.billboards_albedo.0.create_view(&view_desc);
            let normals_view = gpu_state.billboards_normals.0.create_view(&view_desc);
            let linear_depth_view = gpu_state.billboards_depth.0.create_view(&view_desc);
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("billboard.depthbuffer"),
                size: wgpu::Extent3d {
                    width: RESOLUTION * FRAMES_PER_SIDE,
                    height: RESOLUTION * FRAMES_PER_SIDE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
            });
            let depth_view = depth.create_view(&Default::default());

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[
                        wgpu::RenderPassColorAttachment {
                            view: &albedo_view,
                            resolve_target: None,
                            ops: Default::default(),
                        },
                        wgpu::RenderPassColorAttachment {
                            view: &normals_view,
                            resolve_target: None,
                            ops: Default::default(),
                        },
                        wgpu::RenderPassColorAttachment {
                            view: &linear_depth_view,
                            resolve_target: None,
                            ops: Default::default(),
                        },
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(Default::default()),
                        stencil_ops: None,
                    }),
                    label: Some("renderpass"),
                });

                rpass.set_pipeline(&pipeline);
                rpass.set_bind_group(0, &bind_group, &[]);
                rpass
                    .set_index_buffer(gpu_state.model_indices.slice(..), wgpu::IndexFormat::Uint32);
                for y in 0..FRAMES_PER_SIDE {
                    for x in 0..FRAMES_PER_SIDE {
                        rpass.set_viewport(
                            (RESOLUTION * x) as f32,
                            (RESOLUTION * y) as f32,
                            RESOLUTION as f32,
                            RESOLUTION as f32,
                            0.0,
                            1.0,
                        );
                        rpass.set_push_constants(
                            wgpu::ShaderStages::VERTEX,
                            0,
                            bytemuck::cast_slice(&[
                                2.0 * x as f32 / FRAMES_PER_SIDE as f32 - 1.0,
                                2.0 * y as f32 / FRAMES_PER_SIDE as f32 - 1.0,
                            ]),
                        );
                        rpass.draw_indexed(self.tree.lods.last().unwrap().clone(), 0, 0..1);
                    }
                }
            }
        }

        // Topdown
        for base_array_layer in 0..1 {
            let view_desc = wgpu::TextureViewDescriptor {
                base_array_layer,
                array_layer_count: Some(NonZeroU32::new(1).unwrap()),
                ..Default::default()
            };
            let albedo_view = gpu_state.topdown_albedo.0.create_view(&view_desc);
            let normals_view = gpu_state.topdown_normals.0.create_view(&view_desc);
            let linear_depth_view = gpu_state.topdown_depth.0.create_view(&view_desc);
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("topdown.depthbuffer"),
                size: wgpu::Extent3d {
                    width: RESOLUTION,
                    height: RESOLUTION,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
            });
            let depth_view = depth.create_view(&Default::default());

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[
                        wgpu::RenderPassColorAttachment {
                            view: &albedo_view,
                            resolve_target: None,
                            ops: Default::default(),
                        },
                        wgpu::RenderPassColorAttachment {
                            view: &normals_view,
                            resolve_target: None,
                            ops: Default::default(),
                        },
                        wgpu::RenderPassColorAttachment {
                            view: &linear_depth_view,
                            resolve_target: None,
                            ops: Default::default(),
                        },
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(Default::default()),
                        stencil_ops: None,
                    }),
                    label: Some("renderpass"),
                });

                rpass.set_pipeline(&pipeline);
                rpass.set_bind_group(0, &bind_group, &[]);
                rpass
                    .set_index_buffer(gpu_state.model_indices.slice(..), wgpu::IndexFormat::Uint32);
                rpass.set_push_constants(
                    wgpu::ShaderStages::VERTEX,
                    0,
                    bytemuck::cast_slice(&[0.0f32, 0.0f32]),
                );
                rpass.draw_indexed(self.tree.lods.last().unwrap().clone(), 0, 0..1);
            }
        }

        queue.submit(Some(encoder.finish()));
    }
}
