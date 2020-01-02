use super::*;
use crate::terrain::tile_cache::LayerType;
use std::mem;

#[derive(Copy, Clone)]
#[repr(C, align(4))]
pub(crate) struct NodeState {
    position: glsl_layout::vec2,
    side_length: f32,
    min_distance: f32,
    heights_desc: [[f32; 4]; 2],
    albedo_desc: [[f32; 4]; 2],
    normals_desc: [[f32; 4]; 2],
    resolution: i32,
}
unsafe impl bytemuck::Pod for NodeState {}
unsafe impl bytemuck::Zeroable for NodeState {}

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
    pub fn prepare_vertex_buffer(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        vertex_buffer: &wgpu::Buffer,
        tile_cache: &VecMap<TileCache>,
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
            tile_cache[LayerType::Colors.index()].resolution(),
            tile_cache[LayerType::Normals.index()].resolution()
        );
        assert_eq!(
            tile_cache[LayerType::Colors.index()].border(),
            tile_cache[LayerType::Normals.index()].border()
        );

        let resolution = tile_cache[LayerType::Heights.index()].resolution() - 1;
        let texture_resolution = tile_cache[LayerType::Normals.index()].resolution();
        let texture_border = tile_cache[LayerType::Normals.index()].border();
        let texture_ratio =
            (texture_resolution - 2 * texture_border) as f32 / texture_resolution as f32;
        let texture_step = texture_ratio / resolution as f32;
        let texture_origin = texture_border as f32 / texture_resolution as f32;

        fn find_descs(
            nodes: &Vec<Node>,
            tile_cache: &TileCache,
            id: NodeId,
            texture_origin: Vector2<f32>,
            base_origin: Vector2<f32>,
            texture_ratio: f32,
            texture_step: f32,
        ) -> [[f32; 4]; 2] {
            if tile_cache.contains(id) {
                let child_slot = tile_cache.get_slot(id).unwrap() as f32;
                let child_offset = texture_origin + texture_ratio * base_origin;

                if let Some((parent, child_index)) = nodes[id].parent {
                    if tile_cache.contains(parent) {
                        let parent_slot = tile_cache.get_slot(parent).unwrap() as f32;
                        let parent_offset = node::OFFSETS[child_index as usize].cast().unwrap();
                        let parent_offset =
                            texture_origin + 0.5 * texture_ratio * (base_origin + parent_offset);

                        return [
                            [child_offset.x, child_offset.y, child_slot, texture_step],
                            [parent_offset.x, parent_offset.y, parent_slot, texture_step * 0.5],
                        ];
                    }
                }

                [[child_offset.x, child_offset.y, child_slot, texture_step], [0.0, 0.0, -1.0, 0.0]]
            } else {
                let (ancestor, generations, offset) =
                    Node::find_ancestor(&nodes, id, |id| tile_cache.contains(id)).unwrap();
                let slot = tile_cache.get_slot(ancestor).map(|s| s as f32).unwrap();
                let scale = (0.5f32).powi(generations as i32);
                let offset = Vector2::new(offset.x as f32, offset.y as f32);
                let offset = texture_origin + scale * texture_ratio * (base_origin + offset);

                [[offset.x, offset.y, slot, scale * texture_step], [0.0, 0.0, -1.0, 0.0]]
            }
        }

        self.node_states.clear();
        for &id in self.visible_nodes.iter() {
            let heights_desc = find_descs(
                &self.nodes,
                &tile_cache[LayerType::Heights.index()],
                id,
                Vector2::new(0.5, 0.5) / (resolution + 1) as f32,
                Vector2::new(0.0, 0.0),
                resolution as f32 / (resolution + 1) as f32,
                1.0 / (resolution + 1) as f32,
            );
            let albedo_desc = find_descs(
                &self.nodes,
                &tile_cache[LayerType::Colors.index()],
                id,
                Vector2::new(texture_origin, texture_origin),
                Vector2::new(0.0, 0.0),
                texture_ratio,
                texture_step,
            );
            let normals_desc = find_descs(
                &self.nodes,
                &tile_cache[LayerType::Normals.index()],
                id,
                Vector2::new(texture_origin, texture_origin),
                Vector2::new(0.0, 0.0),
                texture_ratio,
                texture_step,
            );
            self.node_states.push(NodeState {
                position: [self.nodes[id].bounds.min.x, self.nodes[id].bounds.min.z].into(),
                side_length: self.nodes[id].side_length,
                min_distance: self.nodes[id].min_distance,
                heights_desc,
                albedo_desc,
                normals_desc,
                resolution: resolution as i32,
            });
        }
        for &(id, mask) in self.partially_visible_nodes.iter() {
            assert!(mask < 15);
            for i in 0..4u8 {
                if mask & (1 << i) != 0 {
                    let side_length = self.nodes[id].side_length * 0.5;
                    let offset = ((i % 2) as f32, (i / 2) as f32);
                    let base_origin = Vector2::new(offset.0 * (0.5), offset.1 * (0.5));
                    let heights_desc = find_descs(
                        &self.nodes,
                        &tile_cache[LayerType::Heights.index()],
                        id,
                        Vector2::new(0.5, 0.5) / (resolution + 1) as f32,
                        Vector2::new(offset.0, offset.1) * 0.5,
                        resolution as f32 / (resolution + 1) as f32,
                        1.0 / (resolution + 1) as f32,
                    );
                    let albedo_desc = find_descs(
                        &self.nodes,
                        &tile_cache[LayerType::Colors.index()],
                        id,
                        Vector2::new(texture_origin, texture_origin),
                        base_origin,
                        texture_ratio,
                        texture_step,
                    );
                    let normals_desc = find_descs(
                        &self.nodes,
                        &tile_cache[LayerType::Normals.index()],
                        id,
                        Vector2::new(texture_origin, texture_origin),
                        base_origin,
                        texture_ratio,
                        texture_step,
                    );
                    self.node_states.push(NodeState {
                        position: [
                            self.nodes[id].bounds.min.x + offset.0 * side_length,
                            self.nodes[id].bounds.min.z + offset.1 * side_length,
                        ]
                        .into(),
                        side_length,
                        min_distance: self.nodes[id].min_distance,
                        heights_desc,
                        normals_desc,
                        albedo_desc,
                        resolution: resolution as i32 / 2,
                    });
                }
            }
        }

        let mapped = device.create_buffer_mapped(
            self.node_states.len() * mem::size_of::<NodeState>(),
            wgpu::BufferUsage::MAP_WRITE | wgpu::BufferUsage::COPY_SRC,
        );

        let slice = bytemuck::cast_slice_mut(mapped.data);
        slice.copy_from_slice(&self.node_states[..]);

        encoder.copy_buffer_to_buffer(
            &mapped.finish(),
            0,
            vertex_buffer,
            0,
            (self.node_states.len() * mem::size_of::<NodeState>()) as u64,
        );

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

        //     let tile_cache = &tile_cache[LayerType::Foliage.index()];
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

    pub(crate) fn render(
        &self,
        rpass: &mut wgpu::RenderPass,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        index_buffer_partial: &wgpu::Buffer,
    ) {
        let resolution = self.heights_resolution;
        let visible_nodes = self.visible_nodes.len() as u32;
        let total_nodes = self.node_states.len() as u32;

        rpass.set_vertex_buffers(0, &[(vertex_buffer, 0)]);

        rpass.set_index_buffer(index_buffer, 0);
        rpass.draw_indexed(0..(resolution * resolution * 6), 0, 0..visible_nodes);

        rpass.set_index_buffer(index_buffer_partial, 0);
        rpass.draw_indexed(
            0..((resolution / 2) * (resolution / 2) * 6),
            0,
            visible_nodes..total_nodes,
        );
    }
}
