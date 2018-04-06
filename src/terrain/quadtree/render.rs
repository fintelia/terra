use failure::Error;
use gfx;
use gfx_core;
use gfx::traits::*;
use gfx::format::*;

use terrain::tile_cache::LayerType;
use super::*;

gfx_defines!{
    vertex NodeState {
        position: [f32; 2] = "vPosition",
        side_length: f32 = "vSideLength",
        min_distance: f32 = "vMinDistance",
        heights_origin: [f32; 3] = "heightsOrigin",
        texture_origin: [f32; 2] = "textureOrigin",
        parent_texture_origin: [f32; 2] = "parentTextureOrigin",
        colors_layer: [f32; 2] = "colorsLayer",
        normals_layer: [f32; 2] = "normalsLayer",
        water_layer: [f32; 2] = "waterLayer",
        texture_step: f32 = "textureStep",
        parent_texture_step: f32 = "parentTextureStep",
    }

    vertex PlanetMeshVertex {
        position: [f32; 3] = "vPosition",
    }
}

gfx_pipeline!( pipe {
    instances: gfx::InstanceBuffer<NodeState> = (),

    model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
    resolution: gfx::Global<i32> = "resolution",
    camera_position: gfx::Global<[f32;3]> = "cameraPosition",
    sun_direction: gfx::Global<[f32;3]> = "sunDirection",

    heights: gfx::TextureSampler<f32> = "heights",
    normals: gfx::TextureSampler<[f32; 4]> = "normals",
    colors: gfx::TextureSampler<[f32; 4]> = "colors",
    water: gfx::TextureSampler<[f32; 4]> = "water",
    materials: gfx::TextureSampler<[f32; 4]> = "materials",
    sky: gfx::TextureSampler<[f32; 4]> = "sky",
    ocean_surface: gfx::TextureSampler<[f32; 4]> = "oceanSurface",
    noise: gfx::TextureSampler<[f32; 4]> = "noise",
    noise_wavelength: gfx::Global<f32> = "noiseWavelength",
    planet_radius: gfx::Global<f32> = "planetRadius",
    atmosphere_radius: gfx::Global<f32> = "atmosphereRadius",
    transmittance: gfx::TextureSampler<[f32; 4]> = "transmittance",
    inscattering: gfx::TextureSampler<[f32; 4]> = "inscattering",
    color_buffer: gfx::RenderTarget<Srgba8> = "OutColor",
    depth_buffer: gfx::DepthTarget<DepthStencil> = gfx::preset::depth::LESS_EQUAL_WRITE,
});

gfx_pipeline!( sky_pipe {
    camera_position: gfx::Global<[f32;3]> = "cameraPosition",
    sun_direction: gfx::Global<[f32;3]> = "sunDirection",
    ray_bottom_left: gfx::Global<[f32;3]> = "rayBottomLeft",
    ray_bottom_right: gfx::Global<[f32;3]> = "rayBottomRight",
    ray_top_left: gfx::Global<[f32;3]> = "rayTopLeft",
    ray_top_right: gfx::Global<[f32;3]> = "rayTopRight",
    sky: gfx::TextureSampler<[f32; 4]> = "sky",
    planet_radius: gfx::Global<f32> = "planetRadius",
    atmosphere_radius: gfx::Global<f32> = "atmosphereRadius",
    transmittance: gfx::TextureSampler<[f32; 4]> = "transmittance",
    inscattering: gfx::TextureSampler<[f32; 4]> = "inscattering",
    color_buffer: gfx::RenderTarget<Srgba8> = "OutColor",
    depth_buffer: gfx::DepthTarget<DepthStencil> = gfx::preset::depth::LESS_EQUAL_WRITE,
});

gfx_pipeline!( planet_mesh_pipe {
    vertices: gfx::VertexBuffer<PlanetMeshVertex> = (),
    model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
    camera_position: gfx::Global<[f32;3]> = "cameraPosition",
    sun_direction: gfx::Global<[f32;3]> = "sunDirection",
    planet_radius: gfx::Global<f32> = "planetRadius",
    atmosphere_radius: gfx::Global<f32> = "atmosphereRadius",
    transmittance: gfx::TextureSampler<[f32; 4]> = "transmittance",
    inscattering: gfx::TextureSampler<[f32; 4]> = "inscattering",
    color: gfx::TextureSampler<[f32; 4]> = "color",
    color_buffer: gfx::RenderTarget<Srgba8> = "OutColor",
    depth_buffer: gfx::DepthTarget<DepthStencil> = gfx::preset::depth::LESS_EQUAL_WRITE,
});

impl<R, F> QuadTree<R, F>
where
    R: gfx::Resources,
    F: gfx::Factory<R>,
{
    pub(crate) fn make_pso(
        factory: &mut F,
        shader: &gfx::ShaderSet<R>,
    ) -> Result<gfx::PipelineState<R, pipe::Meta>, Error> {
        Ok(factory.create_pipeline_state(
            shader,
            gfx::Primitive::TriangleList,
            gfx::state::Rasterizer {
                front_face: gfx::state::FrontFace::Clockwise,
                cull_face: gfx::state::CullFace::Back,
                method: gfx::state::RasterMethod::Fill,
                offset: None,
                samples: None,
            },
            pipe::new(),
        )?)
    }

    pub(crate) fn make_sky_pso(
        factory: &mut F,
        shader: &gfx::ShaderSet<R>,
    ) -> Result<gfx::PipelineState<R, sky_pipe::Meta>, Error> {
        Ok(factory.create_pipeline_state(
            shader,
            gfx::Primitive::TriangleList,
            gfx::state::Rasterizer {
                front_face: gfx::state::FrontFace::Clockwise,
                cull_face: gfx::state::CullFace::Nothing,
                method: gfx::state::RasterMethod::Fill,
                offset: None,
                samples: None,
            },
            sky_pipe::new(),
        )?)
    }

    pub(crate) fn make_planet_mesh_pso(
        factory: &mut F,
        shader: &gfx::ShaderSet<R>,
    ) -> Result<gfx::PipelineState<R, planet_mesh_pipe::Meta>, Error> {
        Ok(factory.create_pipeline_state(
            shader,
            gfx::Primitive::TriangleList,
            gfx::state::Rasterizer {
                front_face: gfx::state::FrontFace::Clockwise,
                cull_face: gfx::state::CullFace::Back,
                method: gfx::state::RasterMethod::Fill,
                offset: None,
                samples: None,
            },
            planet_mesh_pipe::new(),
        )?)
    }

    pub(super) fn update_shaders(&mut self) -> Result<(), Error> {
        if self.shader
            .refresh(&mut self.factory, &mut self.shaders_watcher)
        {
            self.pso = Self::make_pso(&mut self.factory, self.shader.as_shader_set())?;
        }

        if self.sky_shader
            .refresh(&mut self.factory, &mut self.shaders_watcher)
        {
            self.sky_pso = Self::make_sky_pso(&mut self.factory, self.sky_shader.as_shader_set())?;
        }

        if self.planet_mesh_shader
            .refresh(&mut self.factory, &mut self.shaders_watcher)
        {
            self.planet_mesh_pso = Self::make_planet_mesh_pso(
                &mut self.factory,
                self.planet_mesh_shader.as_shader_set(),
            )?;
        }

        Ok(())
    }

    pub fn render<C: gfx_core::command::Buffer<R>>(
        &mut self,
        encoder: &mut gfx::Encoder<R, C>,
    ) -> Result<(), Error> {
        encoder.draw(
            &gfx::Slice {
                start: 0,
                end: self.num_planet_mesh_vertices as u32,
                base_vertex: 0,
                instances: None,
                buffer: gfx::IndexBuffer::Auto,
            },
            &self.planet_mesh_pso,
            &self.planet_mesh_pipeline_data,
        );

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

        fn find_texture_slots<R: gfx::Resources>(
            nodes: &Vec<Node>,
            tile_cache_layers: &VecMap<TileCache<R>>,
            id: NodeId,
            texture_ratio: f32,
        ) -> (f32, f32, f32, Vector2<f32>, f32) {
            let (ancestor, generations, offset) = Node::find_ancestor(&nodes, id, |id| {
                tile_cache_layers[LayerType::Colors.index()].contains(id)
            }).unwrap();
            let colors_slot = tile_cache_layers[LayerType::Colors.index()]
                .get_slot(ancestor)
                .map(|s| s as f32)
                .unwrap();
            let normals_slot = tile_cache_layers[LayerType::Normals.index()]
                .get_slot(ancestor)
                .map(|s| s as f32)
                .unwrap_or(-1.0);
            let water_slot = tile_cache_layers[LayerType::Water.index()]
                .get_slot(ancestor)
                .map(|s| s as f32)
                .unwrap();
            let scale = (0.5f32).powi(generations as i32);
            let offset = Vector2::new(
                offset.x as f32 * texture_ratio * scale,
                offset.y as f32 * texture_ratio * scale,
            );
            (colors_slot, normals_slot, water_slot, offset, scale)
        };
        fn find_parent_texture_slots<R: gfx::Resources>(
            nodes: &Vec<Node>,
            tile_cache_layers: &VecMap<TileCache<R>>,
            id: NodeId,
            texture_ratio: f32,
        ) -> (f32, f32, f32, Vector2<f32>, f32) {
            if let Some((parent, child_index)) = nodes[id].parent {
                let (c, n, w, offset, scale) =
                    find_texture_slots(nodes, tile_cache_layers, parent, texture_ratio);
                let child_offset = node::OFFSETS[child_index as usize];
                let offset = offset
                    + Vector2::new(child_offset.x as f32, child_offset.y as f32) * scale
                        * texture_ratio * 0.5;
                (c, n, w, offset, scale * 0.5)
            } else {
                (-1.0, -1.0, -1.0, Vector2::new(0.0, 0.0), 0.0)
            }
        }

        self.node_states.clear();
        for &id in self.visible_nodes.iter() {
            let heights_slot = self.tile_cache_layers[LayerType::Heights.index()]
                .get_slot(id)
                .unwrap() as f32;
            let (colors_layer, normals_layer, water_layer, texture_offset, tex_step_scale) =
                find_texture_slots(&self.nodes, &self.tile_cache_layers, id, texture_ratio);
            let (pcolors_layer, pnormals_layer, pwater_layer, ptexture_offset, ptex_step_scale) =
                find_parent_texture_slots(&self.nodes, &self.tile_cache_layers, id, texture_ratio);
            self.node_states.push(NodeState {
                position: [self.nodes[id].bounds.min.x, self.nodes[id].bounds.min.z],
                side_length: self.nodes[id].side_length,
                min_distance: self.nodes[id].min_distance,
                heights_origin: [0.0, 0.0, heights_slot],
                texture_origin: [
                    texture_origin + texture_offset.x,
                    texture_origin + texture_offset.y,
                ],
                parent_texture_origin: [
                    texture_origin + ptexture_offset.x,
                    texture_origin + ptexture_offset.y,
                ],
                colors_layer: [colors_layer, pcolors_layer],
                normals_layer: [normals_layer, pnormals_layer],
                water_layer: [water_layer, pwater_layer],
                texture_step: texture_step * tex_step_scale,
                parent_texture_step: texture_step * ptex_step_scale,
            });
        }
        for &(id, mask) in self.partially_visible_nodes.iter() {
            assert!(mask < 15);
            for i in 0..4u8 {
                if mask & (1 << i) != 0 {
                    let side_length = self.nodes[id].side_length * 0.5;
                    let offset = ((i % 2) as f32, (i / 2) as f32);
                    let heights_slot = self.tile_cache_layers[LayerType::Heights.index()]
                        .get_slot(id)
                        .unwrap() as f32;
                    let (
                        colors_layer,
                        normals_layer,
                        water_layer,
                        texture_offset,
                        texture_step_scale,
                    ) = find_texture_slots(&self.nodes, &self.tile_cache_layers, id, texture_ratio);
                    let (
                        pcolors_layer,
                        pnormals_layer,
                        pwater_layer,
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
                        ],
                        side_length,
                        min_distance: self.nodes[id].min_distance,
                        heights_origin: [
                            offset.0 * (0.5 - 0.5 / (resolution + 1) as f32),
                            offset.1 * (0.5 - 0.5 / (resolution + 1) as f32),
                            heights_slot,
                        ],
                        texture_origin: [
                            texture_origin + texture_offset.x
                                + offset.0 * (0.5 - texture_origin) * texture_step_scale,
                            texture_origin + texture_offset.y
                                + offset.1 * (0.5 - texture_origin) * texture_step_scale,
                        ],
                        parent_texture_origin: [
                            texture_origin + ptexture_offset.x
                                + offset.0 * (0.5 - texture_origin) * ptexture_step_scale,
                            texture_origin + ptexture_offset.y
                                + offset.1 * (0.5 - texture_origin) * ptexture_step_scale,
                        ],
                        colors_layer: [colors_layer, pcolors_layer],
                        normals_layer: [normals_layer, pnormals_layer],
                        water_layer: [water_layer, pwater_layer],
                        texture_step: texture_step * texture_step_scale,
                        parent_texture_step: texture_step * ptexture_step_scale,
                    });
                }
            }
        }

        encoder.update_buffer(&self.pipeline_data.instances, &self.node_states[..], 0)?;

        self.pipeline_data.resolution = resolution as i32;
        encoder.draw(
            &gfx::Slice {
                start: 0,
                end: (resolution * resolution * 6) as u32,
                base_vertex: 0,
                instances: Some((self.visible_nodes.len() as u32, 0)),
                buffer: self.index_buffer.clone(),
            },
            &self.pso,
            &self.pipeline_data,
        );

        self.pipeline_data.resolution = (resolution / 2) as i32;
        encoder.draw(
            &gfx::Slice {
                start: 0,
                end: ((resolution / 2) * (resolution / 2) * 6) as u32,
                base_vertex: 0,
                instances: Some((
                    (self.node_states.len() - self.visible_nodes.len()) as u32,
                    self.visible_nodes.len() as u32,
                )),
                buffer: self.index_buffer_partial.clone(),
            },
            &self.pso,
            &self.pipeline_data,
        );

        Ok(())
    }

    pub fn render_sky<C: gfx_core::command::Buffer<R>>(
        &mut self,
        encoder: &mut gfx::Encoder<R, C>,
    ) {
        encoder.draw(
            &gfx::Slice {
                start: 0,
                end: 6,
                base_vertex: 0,
                instances: None,
                buffer: gfx::IndexBuffer::Auto,
            },
            &self.sky_pso,
            &self.sky_pipeline_data,
        );
    }
}
