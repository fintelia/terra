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
        normals_origin: [f32; 3] = "normalsOrigin",
        normals_step: f32 = "normalsStep",
    }
}

gfx_pipeline!( pipe {
    instances: gfx::InstanceBuffer<NodeState> = (),

    model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
    resolution: gfx::Global<i32> = "resolution",
    camera_position: gfx::Global<[f32;3]> = "cameraPosition",

    heights: gfx::TextureSampler<f32> = "heights",
    normals: gfx::TextureSampler<[f32; 4]> = "normals",

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
    ) -> gfx::PipelineState<R, pipe::Meta> {
        factory
            .create_pipeline_state(
                shader,
                gfx::Primitive::TriangleList,
                gfx::state::Rasterizer {
                    front_face: gfx::state::FrontFace::Clockwise,
                    cull_face: gfx::state::CullFace::Nothing,
                    method: gfx::state::RasterMethod::Fill,
                    offset: None,
                    samples: None,
                },
                pipe::new(),
            )
            .unwrap()
    }

    pub fn update_shaders(&mut self) {
        if self.shader.refresh(
            &mut self.factory,
            &mut self.shaders_watcher,
        )
        {
            self.pso = Self::make_pso(&mut self.factory, self.shader.as_shader_set());
        }
    }

    pub fn render<C: gfx_core::command::Buffer<R>>(&mut self, encoder: &mut gfx::Encoder<R, C>) {
        let resolution = self.tile_cache_layers[LayerType::Heights.index()].resolution() - 1;
        let normals_resolution = self.tile_cache_layers[LayerType::Normals.index()].resolution();
        let normals_border = self.tile_cache_layers[LayerType::Normals.index()].border();
        let normals_ratio = (normals_resolution - 2 * normals_border) as f32 /
            normals_resolution as f32;
        let normals_step = normals_ratio / resolution as f32;
        let normals_origin = normals_border as f32 / normals_resolution as f32;

        fn find_normals_slot<R: gfx::Resources>(
            nodes: &Vec<Node>,
            tile_cache_layers: &VecMap<TileCache<R>>,
            id: NodeId,
            normals_ratio: f32,
        ) -> (f32, Vector2<f32>, f32) {
            let (ancestor, generations, offset) = Node::find_ancestor(&nodes, id, |id| {
                tile_cache_layers[LayerType::Normals.index()].contains(id)
            }).unwrap();
            let slot = tile_cache_layers[LayerType::Normals.index()]
                .get_slot(ancestor)
                .unwrap();
            let scale = (0.5f32).powi(generations as i32);
            let offset = Vector2::new(
                offset.x as f32 * normals_ratio * scale,
                offset.y as f32 * normals_ratio * scale,
            );
            (slot as f32, offset, scale)
        };

        self.node_states.clear();
        for &id in self.visible_nodes.iter() {
            let heights_slot = self.tile_cache_layers[LayerType::Heights.index()]
                .get_slot(id)
                .unwrap() as f32;
            let (normals_slot, normals_offset, normals_step_scale) =
                find_normals_slot(&self.nodes, &self.tile_cache_layers, id, normals_ratio);
            self.node_states.push(NodeState {
                position: [self.nodes[id].bounds.min.x, self.nodes[id].bounds.min.z],
                side_length: self.nodes[id].side_length,
                min_distance: self.nodes[id].min_distance,
                heights_origin: [0.0, 0.0, heights_slot],
                normals_origin: [
                    normals_origin + normals_offset.x,
                    normals_origin + normals_offset.y,
                    normals_slot,
                ],
                normals_step: normals_step * normals_step_scale,
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
                    let (normals_slot, normals_offset, normals_step_scale) =
                        find_normals_slot(&self.nodes, &self.tile_cache_layers, id, normals_ratio);
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
                        normals_origin: [
                            normals_origin + normals_offset.x +
                                offset.0 * (0.5 - normals_origin) * normals_step_scale,
                            normals_origin + normals_offset.y +
                                offset.1 * (0.5 - normals_origin) * normals_step_scale,
                            normals_slot,
                        ],
                        normals_step: normals_step * normals_step_scale,
                    });
                }
            }
        }

        encoder
            .update_buffer(&self.pipeline_data.instances, &self.node_states[..], 0)
            .unwrap();

        self.pipeline_data.resolution = resolution as i32;
        encoder.draw(
            &gfx::Slice {
                start: 0,
                end: (resolution * resolution * 6) as u32,
                base_vertex: 0,
                instances: Some((self.visible_nodes.len() as u32, 0)),
                buffer: gfx::IndexBuffer::Auto,
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
                buffer: gfx::IndexBuffer::Auto,
            },
            &self.pso,
            &self.pipeline_data,
        );
    }
}
