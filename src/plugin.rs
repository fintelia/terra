use amethyst::core::ecs::Resources;
/// Plugin for Amethyst that renders a terrain.
use amethyst::renderer::{
    submodules::gather::CameraGatherer,
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
use rendy::resource::{DescriptorSet, DescriptorSetLayout, Escape, Handle};
use rendy::shader::{ShaderSet, ShaderSetBuilder, SpecConstantSet, SpirvShader};

/// Collection of functions terra needs on an aux object to render a scene.
pub trait TerraAux {
    fn camera(&self) -> (glsl_layout::vec3, glsl_layout::mat4, glsl_layout::mat4);
}
impl TerraAux for Resources {
    fn camera(&self) -> (glsl_layout::vec3, glsl_layout::mat4, glsl_layout::mat4) {
        let CameraGatherer {
            camera_position,
            projview,
        } = CameraGatherer::gather(self);
        (camera_position, projview.proj, projview.view)
    }
}

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
impl<B: Backend, T: TerraAux> RenderGroupDesc<B, T> for TerrainRenderGroupDesc {
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
        // layouts
        let layout = Layout {
            sets: Vec::new(),
            push_constants: Vec::new(),
        };
        let set_layouts = layout
            .sets
            .into_iter()
            .map(|set| {
                factory
                    .create_descriptor_set_layout(set.bindings)
                    .map(Handle::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pipeline_layout = unsafe {
            factory
                .device()
                .create_pipeline_layout(set_layouts.iter().map(|l| l.raw()), layout.push_constants)?
        };

        // descriptor sets
        let descriptor_sets = set_layouts
            .iter()
            .map(|set| factory.create_descriptor_set(set.clone()))
            .collect::<Result<Vec<_>, _>>()?;

        // buffers
        let mut vertex_buffers = Vec::new();
        let mut attributes: Vec<gfx_hal::pso::AttributeDesc> = Vec::new();

        // for &(ref elements, stride, rate) in &pipeline.vertices {
        //     let index =

        // vertex_buffers.push(gfx_hal::pso::VertexBufferDesc {
        //     binding: vertex_buffers.len() as gfx_hal::pso::BufferIndex,
        //     stride: 0,
        //     rate: VertexInputRate::Vertex,
        // });

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

        // Shaders
        let mut watcher = rshader::ShaderDirectoryWatcher::new("src/shaders").unwrap();
        let shader_set = rshader::ShaderSet::simple(
            &mut watcher,
            shader_source!("../src/shaders", "version", "a.vert"),
            shader_source!("../src/shaders", "version", "a.frag"),
        )?;
        let mut shader_set: ShaderSet<B> = ShaderSetBuilder::default()
            .with_vertex(shader_set.vertex())?
            .with_fragment(shader_set.fragment())?
            .build(factory, SpecConstantSet::default())?;
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
                    vertex_buffers,
                    attributes,
                    layout: &pipeline_layout,
                    rasterizer: gfx_hal::pso::Rasterizer::FILL,
                    input_assembler: InputAssemblerDesc {
                        primitive: Primitive::TriangleList,
                        primitive_restart: PrimitiveRestart::Disabled,
                    },
                    blender: gfx_hal::pso::BlendDesc {
                        logic_op: None,
                        targets: vec![rendy::hal::pso::ColorBlendDesc::EMPTY],
                    },
                    depth_stencil: DepthStencilDesc {
                        depth: DepthTest::On {
                            fun: Comparison::Always,
                            write: true,
                        },
                        depth_bounds: false,
                        stencil: StencilTest::Off,
                    },
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
                    flags: gfx_hal::pso::PipelineCreationFlags::empty(),
                    parent: gfx_hal::pso::BasePipeline::None,
                    subpass,
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
            descriptor_sets,
            pipeline_layout,
            graphics_pipeline,
        }))
    }
}

#[derive(Debug)]
pub struct TerrainRenderGroup<B: Backend> {
    set_layouts: Vec<Handle<DescriptorSetLayout<B>>>,
    descriptor_sets: Vec<Escape<DescriptorSet<B>>>,
    pipeline_layout: B::PipelineLayout,
    graphics_pipeline: B::GraphicsPipeline,

    // patch_resolution: u16,
    // index_buffer: Escape<Buffer<B>>,
    // index_buffer_partial: Escape<Buffer<B>>,
}
impl<B: Backend, T: TerraAux> RenderGroup<B, T> for TerrainRenderGroup<B> {
    fn prepare(
        &mut self,
        factory: &Factory<B>,
        queue: QueueId,
        index: usize,
        subpass: Subpass<B>,
        aux: &T,
    ) -> PrepareResult {
        let camera = aux.camera();

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
        unsafe {
            encoder.bind_graphics_descriptor_sets(
                &self.pipeline_layout,
                0,
                self.descriptor_sets.iter().map(|s| s.raw()),
                std::iter::empty::<u32>(),
            );

            encoder.draw(0..3, 0..1);
        }
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
