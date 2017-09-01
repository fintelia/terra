use gfx;
use gfx_core;
use gfx::traits::*;
use gfx::format::*;

use super::*;

gfx_defines!{
    vertex NodeState {
        position: [f32; 2] = "vPosition",
        side_length: f32 = "vSideLength",
    }
}

gfx_pipeline!( pipe {
    instances: gfx::InstanceBuffer<NodeState> = (),

    model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
    resolution: gfx::Global<i32> = "resolution",

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
        let node_states: Vec<_> = self.visible_nodes
            .iter()
            .cloned()
            .map(|id| {
                NodeState {
                    position: [self.nodes[id].bounds.min.x, self.nodes[id].bounds.min.z],
                    side_length: self.nodes[id].side_length,
                }
            })
            .collect();

        encoder
            .update_buffer(&self.pipeline_data.instances, &node_states[..], 0)
            .unwrap();

        encoder.draw(
            &gfx::Slice {
                start: 0,
                end: ((self.pipeline_data.resolution - 1) *
                          (self.pipeline_data.resolution - 1) * 6) as u32,
                base_vertex: 0,
                instances: Some((node_states.len() as u32, 0)),
                buffer: gfx::IndexBuffer::Auto,
            },
            &self.pso,
            &self.pipeline_data,
        )
    }
}
