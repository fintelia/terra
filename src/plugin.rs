use amethyst::core::ecs::Resources;
/// Plugin for Amethyst that renders a terrain.
use amethyst::renderer::{
    bundle::{RenderOrder, RenderPlan, Target},
    Factory, RenderPlugin,
};
use failure::Error;
use gfx_hal::device::Device;
use gfx_hal::pass::Subpass;
use gfx_hal::pso::{
    Comparison, DepthStencilDesc, DepthTest, InputAssemblerDesc, PrimitiveRestart, StencilTest,
};
use gfx_hal::{Backend, Primitive};
use rendy::command::{QueueId, RenderPassEncoder};
use rendy::graph::render::{Layout, PrepareResult, RenderGroup, RenderGroupDesc};
use rendy::graph::{DescBuilder, GraphContext, NodeBuffer, NodeId, NodeImage};
use rendy::resource::{DescriptorSetLayout, Handle};
use rendy::shader::ShaderSet;

#[derive(Debug)]
pub struct RenderTerrain {}
impl RenderTerrain {
    pub fn new() -> Self {
        Self {}
    }
}

impl<B: amethyst::renderer::types::Backend> RenderPlugin<B> for RenderTerrain {
    fn on_plan(
        &mut self,
        plan: &mut RenderPlan<B>,
        factory: &mut Factory<B>,
        res: &Resources,
    ) -> Result<(), amethyst::error::Error> {
        plan.extend_target(Target::Main, |ctx| {
            let builder: DescBuilder<B, Resources, _> = TerrainRenderGroupDesc {}.builder();
            ctx.add(RenderOrder::Opaque, builder)?;
            Ok(())
        });
        Ok(())
    }
}

#[derive(Debug)]
pub struct TerrainRenderGroupDesc {}
impl<B: Backend, T: ?Sized> RenderGroupDesc<B, T> for TerrainRenderGroupDesc {
    fn build(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        queue: QueueId,
        aux: &T,
        framebuffer_width: u32,
        framebuffer_height: u32,
        subpass: Subpass<B>,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
    ) -> Result<Box<dyn RenderGroup<B, T> + 'static>, Error> {
        let mut shader_set: ShaderSet<B> = unimplemented!();
        let layout: Layout = unimplemented!();
        let input_assembler = InputAssemblerDesc {
            primitive: Primitive::TriangleList,
            primitive_restart: PrimitiveRestart::Disabled,
        };
        let depth_stencil = DepthStencilDesc {
            depth: DepthTest::On {
                fun: Comparison::Less,
                write: true,
            },
            depth_bounds: true,
            stencil: StencilTest::Off,
        };

        let set_layouts = layout
            .sets
            .into_iter()
            .map(|set| {
                factory
                    .create_descriptor_set_layout(set.bindings)
                    .map(Handle::from)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                shader_set.dispose(factory);
                e
            })?;

        let pipeline_layout = unsafe {
            factory
                .device()
                .create_pipeline_layout(set_layouts.iter().map(|l| l.raw()), layout.push_constants)
        }
        .map_err(|e| {
            shader_set.dispose(factory);
            e
        })?;

        let mut vertex_buffers = Vec::new();
        let mut attributes: Vec<gfx_hal::pso::AttributeDesc> = Vec::new();

        // for &(ref elements, stride, rate) in &pipeline.vertices {
        //     let index = vertex_buffers.len() as gfx_hal::pso::BufferIndex;

        //     vertex_buffers.push(gfx_hal::pso::VertexBufferDesc {
        //         binding: index,
        //         stride,
        //         rate,
        //     });

        //     let mut location = attributes.last().map_or(0, |a| a.location + 1);
        //     for &element in elements {
        //         attributes.push(gfx_hal::pso::AttributeDesc {
        //             location,
        //             binding: index,
        //             element,
        //         });
        //         location += 1;
        //     }
        // }

        let rect = gfx_hal::pso::Rect {
            x: 0,
            y: 0,
            w: framebuffer_width as i16,
            h: framebuffer_height as i16,
        };

        let shaders = match shader_set.raw() {
            Err(e) => {
                shader_set.dispose(factory);
                return Err(e);
            }
            Ok(s) => s,
        };

        let graphics_pipeline = unsafe {
            factory.device().create_graphics_pipelines(
                Some(gfx_hal::pso::GraphicsPipelineDesc {
                    shaders,
                    rasterizer: gfx_hal::pso::Rasterizer::FILL,
                    vertex_buffers,
                    attributes,
                    input_assembler,
                    blender: gfx_hal::pso::BlendDesc {
                        logic_op: None,
                        targets: Vec::new(),
                    },
                    depth_stencil,
                    multisampling: None,
                    baked_states: gfx_hal::pso::BakedStates {
                        viewport: Some(gfx_hal::pso::Viewport {
                            rect,
                            depth: 0.0..1.0,
                        }),
                        scissor: Some(rect),
                        blend_color: None,
                        depth_bounds: None,
                    },
                    layout: &pipeline_layout,
                    subpass,
                    flags: gfx_hal::pso::PipelineCreationFlags::empty(),
                    parent: gfx_hal::pso::BasePipeline::None,
                }),
                None,
            )
        }
        .remove(0)
        .map_err(|e| {
            shader_set.dispose(factory);
            e
        })?;

        shader_set.dispose(factory);

        Ok(Box::new(TerrainRenderGroup {
            set_layouts,
            pipeline_layout,
            graphics_pipeline,
        }))
    }
}

#[derive(Debug)]
pub struct TerrainRenderGroup<B: Backend> {
    set_layouts: Vec<Handle<DescriptorSetLayout<B>>>,
    pipeline_layout: B::PipelineLayout,
    graphics_pipeline: B::GraphicsPipeline,
}
impl<B: Backend, T: ?Sized> RenderGroup<B, T> for TerrainRenderGroup<B> {
    fn prepare(
        &mut self,
        factory: &Factory<B>,
        queue: QueueId,
        index: usize,
        subpass: Subpass<B>,
        aux: &T,
    ) -> PrepareResult {
        // TODO: what needs to happen here?
        PrepareResult::DrawRecord
    }

    fn draw_inline(
        &mut self,
        mut encoder: RenderPassEncoder<B>,
        index: usize,
        subpass: Subpass<B>,
        aux: &T,
    ) {
        encoder.bind_graphics_pipeline(&self.graphics_pipeline);
        // TODO: render terrain.
    }

    fn dispose(self: Box<Self>, factory: &mut Factory<B>, aux: &T) {
        unsafe {
            factory
                .device()
                .destroy_graphics_pipeline(self.graphics_pipeline);
            factory
                .device()
                .destroy_pipeline_layout(self.pipeline_layout);
            drop(self.set_layouts);
        }
    }
}
