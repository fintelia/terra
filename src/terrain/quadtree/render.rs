use super::*;
use crate::terrain::tile_cache::LayerType;
use gfx_hal::{Backend, IndexType};
use rendy::command::{QueueId, RenderPassEncoder};
use rendy::factory::Factory;
use std::iter;

#[repr(C, align(4))]
pub(crate) struct NodeState {
    position: glsl_layout::vec2,
    side_length: f32,
    min_distance: f32,
    heights_origin: glsl_layout::vec3,
    texture_origin: glsl_layout::vec2,
    parent_texture_origin: glsl_layout::vec2,
    colors_layer: glsl_layout::vec2,
    normals_layer: glsl_layout::vec2,
    splats_layer: glsl_layout::vec2,
    texture_step: f32,
    parent_texture_step: f32,
    resolution: i32,
}

// gfx_defines!{
//     vertex NodeState {
//         position: [f32; 2] = "vPosition",
//         side_length: f32 = "vSideLength",
//         min_distance: f32 = "vMinDistance",
//         heights_origin: [f32; 3] = "heightsOrigin",
//         texture_origin: [f32; 2] = "textureOrigin",
//         parent_texture_origin: [f32; 2] = "parentTextureOrigin",
//         colors_layer: [f32; 2] = "colorsLayer",
//         normals_layer: [f32; 2] = "normalsLayer",
//         splats_layer: [f32; 2] = "splatsLayer",
//         texture_step: f32 = "textureStep",
//         parent_texture_step: f32 = "parentTextureStep",
//     }

//     vertex PlanetMeshVertex {
//         position: [f32; 3] = "vPosition",
//     }

//     vertex MeshVertex {
//         position: [f32; 3] = "mPosition",
//         texcoord: [f32; 2] = "mTexcoord",
//         normal: [f32; 3] = "mNormal",
//     }
// }

// gfx_pipeline!( pipe {
//     instances: gfx::InstanceBuffer<NodeState> = (),

//     model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
//     resolution: gfx::Global<i32> = "resolution",
//     camera_position: gfx::Global<[f32;3]> = "cameraPosition",
//     sun_direction: gfx::Global<[f32;3]> = "sunDirection",
//     world_to_warped: gfx::Global<[[f32; 4]; 4]> = "worldToWarped",

//     heights: gfx::TextureSampler<f32> = "heights",
//     normals: gfx::TextureSampler<[f32; 4]> = "normals",
//     colors: gfx::TextureSampler<[f32; 4]> = "colors",
//     splats: gfx::TextureSampler<f32> = "splats",
//     materials: gfx::TextureSampler<[f32; 4]> = "materials",
//     sky: gfx::TextureSampler<[f32; 4]> = "sky",
//     ocean_surface: gfx::TextureSampler<[f32; 4]> = "oceanSurface",
//     noise: gfx::TextureSampler<[f32; 4]> = "noise",
//     noise_wavelength: gfx::Global<f32> = "noiseWavelength",
//     planet_radius: gfx::Global<f32> = "planetRadius",
//     atmosphere_radius: gfx::Global<f32> = "atmosphereRadius",
//     transmittance: gfx::TextureSampler<[f32; 4]> = "transmittance",
//     inscattering: gfx::TextureSampler<[f32; 4]> = "inscattering",
//     color_buffer: gfx::RenderTarget<Rgba16F> = "OutColor",
//     depth_buffer: gfx::DepthTarget<Depth32F> = state::Depth{fun: state::Comparison::GreaterEqual, write: true},
// });

// gfx_pipeline!( sky_pipe {
//     camera_position: gfx::Global<[f32;3]> = "cameraPosition",
//     sun_direction: gfx::Global<[f32;3]> = "sunDirection",
//     ray_bottom_left: gfx::Global<[f32;3]> = "rayBottomLeft",
//     ray_bottom_right: gfx::Global<[f32;3]> = "rayBottomRight",
//     ray_top_left: gfx::Global<[f32;3]> = "rayTopLeft",
//     ray_top_right: gfx::Global<[f32;3]> = "rayTopRight",
//     world_to_warped: gfx::Global<[[f32; 4]; 4]> = "worldToWarped",
//     sky: gfx::TextureSampler<[f32; 4]> = "sky",
//     planet_radius: gfx::Global<f32> = "planetRadius",
//     atmosphere_radius: gfx::Global<f32> = "atmosphereRadius",
//     transmittance: gfx::TextureSampler<[f32; 4]> = "transmittance",
//     inscattering: gfx::TextureSampler<[f32; 4]> = "inscattering",
//     color_buffer: gfx::RenderTarget<Rgba16F> = "OutColor",
//     depth_buffer: gfx::DepthTarget<Depth32F> = state::Depth{fun: state::Comparison::GreaterEqual, write: true},
// });

// gfx_pipeline!( planet_mesh_pipe {
//     vertices: gfx::VertexBuffer<PlanetMeshVertex> = (),
//     model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
//     camera_position: gfx::Global<[f32;3]> = "cameraPosition",
//     sun_direction: gfx::Global<[f32;3]> = "sunDirection",
//     planet_radius: gfx::Global<f32> = "planetRadius",
//     atmosphere_radius: gfx::Global<f32> = "atmosphereRadius",
//     world_to_warped: gfx::Global<[[f32; 4]; 4]> = "worldToWarped",
//     transmittance: gfx::TextureSampler<[f32; 4]> = "transmittance",
//     inscattering: gfx::TextureSampler<[f32; 4]> = "inscattering",
//     color: gfx::TextureSampler<[f32; 4]> = "color",
//     color_buffer: gfx::RenderTarget<Rgba16F> = "OutColor",
//     depth_buffer: gfx::DepthTarget<Depth32F> = state::Depth{fun: state::Comparison::GreaterEqual, write: true},
// });

// gfx_pipeline!( instanced_mesh_pipe {
//     vertices: gfx::VertexBuffer<MeshVertex> = (),
//     instances: gfx::InstanceBuffer<MeshInstance> = (),
//     model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
//     camera_position: gfx::Global<[f32;3]> = "cameraPosition",
//     sun_direction: gfx::Global<[f32;3]> = "sunDirection",
//     planet_radius: gfx::Global<f32> = "planetRadius",
//     atmosphere_radius: gfx::Global<f32> = "atmosphereRadius",
//     world_to_warped: gfx::Global<[[f32; 4]; 4]> = "worldToWarped",
//     albedo: gfx::TextureSampler<[f32; 4]> = "albedo",
//     transmittance: gfx::TextureSampler<[f32; 4]> = "transmittance",
//     inscattering: gfx::TextureSampler<[f32; 4]> = "inscattering",
//     color_buffer: gfx::RenderTarget<Rgba16F> = "OutColor",
//     depth_buffer: gfx::DepthTarget<Depth32F> = state::Depth{fun: state::Comparison::GreaterEqual, write: true},
// });

impl QuadTree {
    // pub(crate) fn make_pso(
    //     factory: &mut F,
    //     shader: &gfx::ShaderSet<R>,
    // ) -> Result<gfx::PipelineState<R, pipe::Meta>, Error> {
    //     Ok(factory.create_pipeline_state(
    //         shader,
    //         gfx::Primitive::TriangleList,
    //         gfx::state::Rasterizer {
    //             front_face: gfx::state::FrontFace::Clockwise,
    //             cull_face: gfx::state::CullFace::Back,
    //             method: gfx::state::RasterMethod::Fill,
    //             offset: None,
    //             samples: None,
    //         },
    //         pipe::new(),
    //     )?)
    // }

    // pub(crate) fn make_sky_pso(
    //     factory: &mut F,
    //     shader: &gfx::ShaderSet<R>,
    // ) -> Result<gfx::PipelineState<R, sky_pipe::Meta>, Error> {
    //     Ok(factory.create_pipeline_state(
    //         shader,
    //         gfx::Primitive::TriangleList,
    //         gfx::state::Rasterizer {
    //             front_face: gfx::state::FrontFace::Clockwise,
    //             cull_face: gfx::state::CullFace::Nothing,
    //             method: gfx::state::RasterMethod::Fill,
    //             offset: None,
    //             samples: None,
    //         },
    //         sky_pipe::new(),
    //     )?)
    // }

    // pub(crate) fn make_planet_mesh_pso(
    //     factory: &mut F,
    //     shader: &gfx::ShaderSet<R>,
    // ) -> Result<gfx::PipelineState<R, planet_mesh_pipe::Meta>, Error> {
    //     Ok(factory.create_pipeline_state(
    //         shader,
    //         gfx::Primitive::TriangleList,
    //         gfx::state::Rasterizer {
    //             front_face: gfx::state::FrontFace::Clockwise,
    //             cull_face: gfx::state::CullFace::Back,
    //             method: gfx::state::RasterMethod::Fill,
    //             offset: None,
    //             samples: None,
    //         },
    //         planet_mesh_pipe::new(),
    //     )?)
    // }

    // pub(crate) fn make_instanced_mesh_pso(
    //     factory: &mut F,
    //     shader: &gfx::ShaderSet<R>,
    // ) -> Result<gfx::PipelineState<R, instanced_mesh_pipe::Meta>, Error> {
    //     Ok(factory.create_pipeline_state(
    //         shader,
    //         gfx::Primitive::TriangleList,
    //         gfx::state::Rasterizer {
    //             front_face: gfx::state::FrontFace::Clockwise,
    //             cull_face: gfx::state::CullFace::Nothing,
    //             method: gfx::state::RasterMethod::Fill,
    //             offset: None,
    //             samples: None,
    //         },
    //         instanced_mesh_pipe::new(),
    //     )?)
    // }

    // pub(super) fn update_shaders(&mut self) -> Result<(), Error> {
    //     if self
    //         .shader
    //         .refresh(&mut self.factory, &mut self.shaders_watcher)
    //     {
    //         self.pso = Self::make_pso(&mut self.factory, self.shader.as_shader_set())?;
    //     }

    //     if self
    //         .sky_shader
    //         .refresh(&mut self.factory, &mut self.shaders_watcher)
    //     {
    //         self.sky_pso = Self::make_sky_pso(&mut self.factory, self.sky_shader.as_shader_set())?;
    //     }

    //     if self
    //         .planet_mesh_shader
    //         .refresh(&mut self.factory, &mut self.shaders_watcher)
    //     {
    //         self.planet_mesh_pso = Self::make_planet_mesh_pso(
    //             &mut self.factory,
    //             self.planet_mesh_shader.as_shader_set(),
    //         )?;
    //     }

    //     if self
    //         .instanced_mesh_shader
    //         .refresh(&mut self.factory, &mut self.shaders_watcher)
    //     {
    //         self.instanced_mesh_pso = Self::make_instanced_mesh_pso(
    //             &mut self.factory,
    //             self.instanced_mesh_shader.as_shader_set(),
    //         )?;
    //     }

    //     Ok(())
    // }

    pub fn prepare_vertex_buffer<B: Backend>(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        vertex_buffer: &mut Buffer<B>,
    ) {
        //     encoder.draw(
        //         &gfx::Slice {
        //             start: 0,
        //             end: self.num_planet_mesh_vertices as u32,
        //             base_vertex: 0,
        //             instances: None,
        //             buffer: gfx::IndexBuffer::Auto,
        //         },
        //         &self.planet_mesh_pso,
        //         &self.planet_mesh_pipeline_data,
        //     );

        assert_eq!(
            self.tile_cache_layers[LayerType::Colors.index()].resolution(),
            self.tile_cache_layers[LayerType::Normals.index()].resolution()
        );
        assert_eq!(
            self.tile_cache_layers[LayerType::Colors.index()].border(),
            self.tile_cache_layers[LayerType::Normals.index()].border()
        );

        let resolution = self.tile_cache_layers[LayerType::Heights.index()].resolution() - 1;
        let texture_resolution = self.tile_cache_layers[LayerType::Normals.index()].resolution();
        let texture_border = self.tile_cache_layers[LayerType::Normals.index()].border();
        let texture_ratio =
            (texture_resolution - 2 * texture_border) as f32 / texture_resolution as f32;
        let texture_step = texture_ratio / resolution as f32;
        let texture_origin = texture_border as f32 / texture_resolution as f32;

        fn find_texture_slots(
            _nodes: &Vec<Node>,
            _tile_cache_layers: &VecMap<TileCache>,
            _id: NodeId,
            _texture_ratio: f32,
        ) -> (f32, f32, f32, Vector2<f32>, f32) {
            // let (ancestor, generations, offset) = Node::find_ancestor(&nodes, id, |id| {
            //     tile_cache_layers[LayerType::Normals.index()].contains(id)
            // })
            // .unwrap();
            // let colors_slot = tile_cache_layers[LayerType::Colors.index()]
            //     .get_slot(ancestor)
            //     .map(|s| s as f32)
            //     .unwrap();
            // let normals_slot = tile_cache_layers[LayerType::Normals.index()]
            //     .get_slot(ancestor)
            //     .map(|s| s as f32)
            //     .unwrap();
            // let splats_slot = tile_cache_layers[LayerType::Splats.index()]
            //     .get_slot(ancestor)
            //     .map(|s| s as f32)
            //     .unwrap_or(-1.0);
            // let scale = (0.5f32).powi(generations as i32);
            // let offset = Vector2::new(
            //     offset.x as f32 * texture_ratio * scale,
            //     offset.y as f32 * texture_ratio * scale,
            // );
            // (colors_slot, normals_slot, splats_slot, offset, scale)
            (0.0, 0.0, 0.0, Vector2::new(0.0, 0.0), 0.0)
        };
        fn find_parent_texture_slots(
            _nodes: &Vec<Node>,
            _tile_cache_layers: &VecMap<TileCache>,
            _id: NodeId,
            _texture_ratio: f32,
        ) -> (f32, f32, f32, Vector2<f32>, f32) {
            // if let Some((parent, child_index)) = nodes[id].parent {
            //     let (c, n, s, offset, scale) =
            //         find_texture_slots(nodes, tile_cache_layers, parent, texture_ratio);
            //     let child_offset = node::OFFSETS[child_index as usize];
            //     let offset = offset
            //         + Vector2::new(child_offset.x as f32, child_offset.y as f32)
            //             * scale
            //             * texture_ratio
            //             * 0.5;
            //     (c, n, s, offset, scale * 0.5)
            // } else {
            (-1.0, -1.0, -1.0, Vector2::new(0.0, 0.0), 0.0)
            // }
        }

        self.node_states.clear();
        for &id in self.visible_nodes.iter() {
            // let heights_slot = self.tile_cache_layers[LayerType::Heights.index()]
            //     .get_slot(id)
            //     .unwrap() as f32;
            let heights_slot = 0.0;
            let (colors_layer, normals_layer, splats_layer, texture_offset, tex_step_scale) =
                find_texture_slots(&self.nodes, &self.tile_cache_layers, id, texture_ratio);
            let (pcolors_layer, pnormals_layer, psplats_layer, ptexture_offset, ptex_step_scale) =
                find_parent_texture_slots(&self.nodes, &self.tile_cache_layers, id, texture_ratio);
            self.node_states.push(NodeState {
                position: [self.nodes[id].bounds.min.x, self.nodes[id].bounds.min.z].into(),
                side_length: self.nodes[id].side_length,
                min_distance: self.nodes[id].min_distance,
                heights_origin: [0.0, 0.0, heights_slot].into(),
                texture_origin: [
                    texture_origin + texture_offset.x,
                    texture_origin + texture_offset.y,
                ]
                .into(),
                parent_texture_origin: [
                    texture_origin + ptexture_offset.x,
                    texture_origin + ptexture_offset.y,
                ]
                .into(),
                colors_layer: [colors_layer, pcolors_layer].into(),
                normals_layer: [normals_layer, pnormals_layer].into(),
                splats_layer: [splats_layer, psplats_layer].into(),
                texture_step: texture_step * tex_step_scale,
                parent_texture_step: texture_step * ptex_step_scale,
                resolution: resolution as i32,
            });
        }
        for &(id, mask) in self.partially_visible_nodes.iter() {
            assert!(mask < 15);
            for i in 0..4u8 {
                if mask & (1 << i) != 0 {
                    let side_length = self.nodes[id].side_length * 0.5;
                    let offset = ((i % 2) as f32, (i / 2) as f32);
                    // let heights_slot = self.tile_cache_layers[LayerType::Heights.index()]
                    //     .get_slot(id)
                    //     .unwrap() as f32;
                    let heights_slot = 0.0;
                    let (
                        colors_layer,
                        normals_layer,
                        splats_layer,
                        texture_offset,
                        texture_step_scale,
                    ) = find_texture_slots(&self.nodes, &self.tile_cache_layers, id, texture_ratio);
                    let (
                        pcolors_layer,
                        pnormals_layer,
                        psplats_layer,
                        ptexture_offset,
                        ptexture_step_scale,
                    ) = find_parent_texture_slots(
                        &self.nodes,
                        &self.tile_cache_layers,
                        id,
                        texture_ratio,
                    );
                    self.node_states.push(NodeState {
                        position: [
                            self.nodes[id].bounds.min.x + offset.0 * side_length,
                            self.nodes[id].bounds.min.z + offset.1 * side_length,
                        ]
                        .into(),
                        side_length,
                        min_distance: self.nodes[id].min_distance,
                        heights_origin: [
                            offset.0 * (0.5 - 0.5 / (resolution + 1) as f32),
                            offset.1 * (0.5 - 0.5 / (resolution + 1) as f32),
                            heights_slot,
                        ]
                        .into(),
                        texture_origin: [
                            texture_origin
                                + texture_offset.x
                                + offset.0 * (0.5 - texture_origin) * texture_step_scale,
                            texture_origin
                                + texture_offset.y
                                + offset.1 * (0.5 - texture_origin) * texture_step_scale,
                        ]
                        .into(),
                        parent_texture_origin: [
                            texture_origin
                                + ptexture_offset.x
                                + offset.0 * (0.5 - texture_origin) * ptexture_step_scale,
                            texture_origin
                                + ptexture_offset.y
                                + offset.1 * (0.5 - texture_origin) * ptexture_step_scale,
                        ]
                        .into(),
                        colors_layer: [colors_layer, pcolors_layer].into(),
                        normals_layer: [normals_layer, pnormals_layer].into(),
                        splats_layer: [splats_layer, psplats_layer].into(),
                        texture_step: texture_step * texture_step_scale,
                        parent_texture_step: texture_step * ptexture_step_scale,
                        resolution: resolution as i32 / 2,
                    });
                }
            }
        }

        unsafe {
            factory
                .upload_visible_buffer(vertex_buffer, 0, &self.node_states[..])
                .unwrap();
        }
        // encoder.update_buffer(&self.pipeline_data.instances, &self.node_states[..], 0)?;

        // self.pipeline_data.resolution = resolution as i32;
        // encoder.draw(
        //     &gfx::Slice {
        //         start: 0,
        //         end: (resolution * resolution * 6) as u32,
        //         base_vertex: 0,
        //         instances: Some((self.visible_nodes.len() as u32, 0)),
        //         buffer: self.index_buffer.clone(),
        //     },
        //     &self.pso,
        //     &self.pipeline_data,
        // );

        // self.pipeline_data.resolution = (resolution / 2) as i32;
        // encoder.draw(
        //     &gfx::Slice {
        //         start: 0,
        //         end: ((resolution / 2) * (resolution / 2) * 6) as u32,
        //         base_vertex: 0,
        //         instances: Some((
        //             (self.node_states.len() - self.visible_nodes.len()) as u32,
        //             self.visible_nodes.len() as u32,
        //         )),
        //         buffer: self.index_buffer_partial.clone(),
        //     },
        //     &self.pso,
        //     &self.pipeline_data,
        // );

        // for (id, node) in self.nodes.iter().enumerate() {
        //     if node.priority < Priority::cutoff() {
        //         continue;
        //     }

        //     let tile_cache = &self.tile_cache_layers[LayerType::Foliage.index()];
        //     if let Some(slot) = tile_cache.get_slot(NodeId::new(id as u32)) {
        //         let count = tile_cache.get_instance_count(node) as u32;
        //         let offset = tile_cache.get_instance_offset(slot) as u32;
        //         encoder.draw(
        //             &gfx::Slice {
        //                 start: 0,
        //                 end: 18,
        //                 base_vertex: 0,
        //                 instances: Some((count, offset)),
        //                 buffer: gfx::IndexBuffer::Auto,
        //             },
        //             &self.instanced_mesh_pso,
        //             &self.instanced_mesh_pipeline_data,
        //         );
        //     }
        // }
    }

    pub(crate) fn render<B: Backend>(
        &self,
        encoder: &mut RenderPassEncoder<B>,
        vertex_buffer: &Buffer<B>,
        index_buffer: &Buffer<B>,
        index_buffer_partial: &Buffer<B>,
    ) {
        let resolution =
            (self.tile_cache_layers[LayerType::Heights.index()].resolution() - 1) as u32;
        let visible_nodes = self.visible_nodes.len() as u32;
        let total_nodes = self.node_states.len() as u32;

        unsafe {
            encoder.bind_vertex_buffers(0, iter::once((vertex_buffer.raw(), 0)));

            encoder.bind_index_buffer(index_buffer.raw(), 0, IndexType::U16);
            encoder.draw_indexed(0..(resolution * resolution * 6), 0, 0..visible_nodes);

            encoder.bind_index_buffer(index_buffer_partial.raw(), 0, IndexType::U16);
            encoder.draw_indexed(
                0..((resolution / 2) * (resolution / 2) * 6),
                0,
                visible_nodes..total_nodes,
            );
        }
    }

    // pub fn render_sky<C: gfx_core::command::Buffer<R>>(
    //     &mut self,
    //     encoder: &mut gfx::Encoder<R, C>,
    // ) {
    //     encoder.draw(
    //         &gfx::Slice {
    //             start: 0,
    //             end: 3,
    //             base_vertex: 0,
    //             instances: None,
    //             buffer: gfx::IndexBuffer::Auto,
    //         },
    //         &self.sky_pso,
    //         &self.sky_pipeline_data,
    //     );
    // }
}
