use gfx;
use gfx_core;
use gfx::traits::*;
use gfx::format::*;

use terrain::tile_cache::HEIGHTS_LAYER;
use super::*;

gfx_defines!{
    vertex NodeState {
        position: [f32; 2] = "vPosition",
        side_length: f32 = "vSideLength",
        min_distance: f32 = "vMinDistance",
    }
}

gfx_pipeline!( pipe {
    instances: gfx::InstanceBuffer<NodeState> = (),

    model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
    resolution: gfx::Global<i32> = "resolution",
    camera_position: gfx::Global<[f32;3]> = "cameraPosition",

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
                    method: gfx::state::RasterMethod::Line(1),
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
        let resolution = self.tile_cache_layers[HEIGHTS_LAYER].resolution() - 1;

        self.node_states.clear();
        for &id in self.visible_nodes.iter() {
            self.node_states.push(NodeState {
                position: [self.nodes[id].bounds.min.x, self.nodes[id].bounds.min.z],
                side_length: self.nodes[id].side_length,
                min_distance: self.nodes[id].min_distance,
            });
        }
        for &(id, mask) in self.partially_visible_nodes.iter() {
            assert!(mask < 15);
            for i in 0..4u8 {
                if mask & (1 << i) != 0 {
                    let side_length = self.nodes[id].side_length * 0.5;
                    let offset = ((i % 2) as f32, (i / 2) as f32);
                    self.node_states.push(NodeState {
                        position: [
                            self.nodes[id].bounds.min.x + offset.0 * side_length,
                            self.nodes[id].bounds.min.z + offset.1 * side_length,
                        ],
                        side_length,
                        min_distance: self.nodes[id].min_distance,
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
