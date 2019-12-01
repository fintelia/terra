///! Plugin for Amethyst that renders a terrain.

use crate::terrain::quadtree::render::NodeState;
use crate::terrain::quadtree::QuadTree;
use crate::compute::ComputeNode;
use amethyst::ecs::{DispatcherBuilder, WorldExt};
use amethyst::prelude::World;
use amethyst::renderer::{
    bundle::{RenderOrder, RenderPlan, Target},
    submodules::gather::CameraGatherer,
    Factory, RenderPlugin,
};
use failure::Error;
use gfx_hal::buffer::Usage;
use gfx_hal::device::Device;
use gfx_hal::format::Format;
use gfx_hal::pass::Subpass;
use gfx_hal::pso::{
    Comparison, DepthStencilDesc, DepthTest, Descriptor, DescriptorSetLayoutBinding,
    DescriptorSetWrite, DescriptorType, Element, InputAssemblerDesc, PrimitiveRestart,
    ShaderStageFlags, StencilTest, VertexInputRate,
};
use gfx_hal::{Backend, Primitive};
use rendy::command::{QueueId, RenderPassEncoder};
use rendy::graph::render::{PrepareResult, RenderGroup, RenderGroupBuilder};
use rendy::graph::{
    BufferAccess, BufferId, GraphContext, ImageAccess, ImageId, Node, NodeBuffer, NodeId,
    NodeImage,
};
use rendy::memory;
use rendy::resource::{BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle};
use rendy::shader::{ShaderSet, ShaderSetBuilder, SpecConstantSet};

#[derive(Debug)]
pub struct RenderTerrain(Option<QuadTree>);
impl RenderTerrain {
    pub fn new(quadtree: QuadTree) -> Self {
        Self(Some(quadtree))
    }
}

impl<B: amethyst::renderer::types::Backend> RenderPlugin<B> for RenderTerrain {
    fn on_build<'a, 'b>(
        &mut self,
        world: &mut World,
        _builder: &mut DispatcherBuilder<'a, 'b>,
    ) -> Result<(), amethyst::Error> {
        world.insert(self.0.take().expect("on_build called multiple times?!"));
        Ok(())
    }

    fn on_plan(
        &mut self,
        plan: &mut RenderPlan<B>,
        _factory: &mut Factory<B>,
        _res: &World,
    ) -> Result<(), amethyst::error::Error> {
        plan.extend_target(Target::Main, |ctx| {
            let compute_node = ctx.graph().add_node(ComputeNode::builder());
            ctx.add_dep(compute_node);
            ctx.add(
                RenderOrder::Opaque,
                TerrainRenderGroupBuilder {
                    buffers: Vec::new(),
                    images: Vec::new(),
                },
            )?;
            Ok(())
        });
        Ok(())
    }
}

#[repr(C)]
struct UniformBlock {
    view: glsl_layout::mat4,
    projection: glsl_layout::mat4,
    camera: glsl_layout::vec3,
    padding: f32,
}

#[derive(Debug)]
pub struct TerrainRenderGroupBuilder {
    buffers: Vec<(BufferId, BufferAccess)>,
    images: Vec<(ImageId, ImageAccess)>,
}
impl<B: gfx_hal::Backend> RenderGroupBuilder<B, World> for TerrainRenderGroupBuilder {
    fn colors(&self) -> usize {
        1
    }

    fn depth(&self) -> bool {
        true
    }

    fn buffers(&self) -> Vec<(BufferId, BufferAccess)> {
        self.buffers.clone()
    }

    fn images(&self) -> Vec<(ImageId, ImageAccess)> {
        self.images.clone()
    }

    fn dependencies(&self) -> Vec<NodeId> {
        Vec::new()
    }

    fn build(
        self: Box<Self>,
        _ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        queue: QueueId,
        aux: &World,
        framebuffer_width: u32,
        framebuffer_height: u32,
        subpass: Subpass<B>,
        _buffers: Vec<NodeBuffer>,
        _images: Vec<NodeImage>,
    ) -> Result<Box<dyn RenderGroup<B, World> + 'static>, Error> {
        let quadtree = aux.read_resource::<QuadTree>();

        // layouts
        let set_layouts = vec![vec![DescriptorSetLayoutBinding {
            binding: 0,
            ty: DescriptorType::UniformBuffer,
            count: 1,
            stage_flags: ShaderStageFlags::VERTEX,
            immutable_samplers: false,
        }]];

        let push_constants: Vec<(gfx_hal::pso::ShaderStageFlags, std::ops::Range<u32>)> =
            Vec::new();

        let set_layouts = set_layouts
            .into_iter()
            .map(|bindings| {
                factory
                    .create_descriptor_set_layout(bindings)
                    .map(Handle::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pipeline_layout = unsafe {
            factory
                .device()
                .create_pipeline_layout(set_layouts.iter().map(|l| l.raw()), push_constants)?
        };

        // descriptor sets
        let descriptor_sets = set_layouts
            .iter()
            .map(|set| factory.create_descriptor_set(set.clone()))
            .collect::<Result<Vec<_>, _>>()?;

        // buffers
        let mut vertex_buffers = Vec::new();
        vertex_buffers.push(gfx_hal::pso::VertexBufferDesc {
            binding: 0,
            stride: std::mem::size_of::<NodeState>() as u32,
            rate: VertexInputRate::Instance(1),
        });

        let attribute_offsets = [0, 8, 12, 16, 28, 36, 44, 52, 60, 68, 72, 76];
        let mut attributes: Vec<gfx_hal::pso::AttributeDesc> = Vec::new();
        for i in 0..12 {
            attributes.push(gfx_hal::pso::AttributeDesc {
                location: i,
                binding: 0,
                element: Element {
                    format: Format::Rg32Sfloat,
                    offset: attribute_offsets[i as usize],
                },
            });
        }
        attributes[0].element.format = Format::Rg32Sfloat;
        attributes[1].element.format = Format::R32Sfloat;
        attributes[2].element.format = Format::R32Sfloat;
        attributes[3].element.format = Format::Rgb32Sfloat;
        attributes[4].element.format = Format::Rg32Sfloat;
        attributes[5].element.format = Format::Rg32Sfloat;
        attributes[6].element.format = Format::Rg32Sfloat;
        attributes[7].element.format = Format::Rg32Sfloat;
        attributes[8].element.format = Format::Rg32Sfloat;
        attributes[9].element.format = Format::R32Sfloat;
        attributes[10].element.format = Format::R32Sfloat;
        attributes[11].element.format = Format::R32Sint;

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

        let vertex_buffer = factory.create_buffer(
            BufferInfo {
                size: (std::mem::size_of::<NodeState>() * quadtree.total_nodes()) as u64,
                usage: Usage::VERTEX,
            },
            memory::Dynamic,
        )?;

        // Index Buffers
        let (index_buffer, index_buffer_partial) = quadtree.create_index_buffers(factory, queue)?;

        // Uniforms
        let uniform_buffer = factory.create_buffer(
            BufferInfo {
                size: std::mem::size_of::<UniformBlock>() as u64,
                usage: Usage::UNIFORM,
            },
            memory::Dynamic,
        )?;

        unsafe {
            factory
                .device()
                .write_descriptor_sets(vec![DescriptorSetWrite {
                    set: descriptor_sets[0].raw(),
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(Descriptor::Buffer(uniform_buffer.raw(), None..None)),
                }]);
        }

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
                    rasterizer: gfx_hal::pso::Rasterizer {
                        polygon_mode: gfx_hal::pso::PolygonMode::Line(1.0),
                        cull_face: gfx_hal::pso::Face::NONE,
                        front_face: gfx_hal::pso::FrontFace::CounterClockwise,
                        depth_clamping: false,
                        depth_bias: None,
                        conservative: false,
                    },
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
            uniform_buffer,
            vertex_buffer,
            index_buffer,
            index_buffer_partial,
            layers: Vec::new(),
        }))
    }
}

#[derive(Debug)]
pub struct TerrainRenderGroup<B: Backend> {
    set_layouts: Vec<Handle<DescriptorSetLayout<B>>>,
    descriptor_sets: Vec<Escape<DescriptorSet<B>>>,
    pipeline_layout: B::PipelineLayout,
    graphics_pipeline: B::GraphicsPipeline,

    uniform_buffer: Escape<rendy::resource::Buffer<B>>,
    vertex_buffer: Escape<rendy::resource::Buffer<B>>,
    index_buffer: Escape<rendy::resource::Buffer<B>>,
    index_buffer_partial: Escape<rendy::resource::Buffer<B>>,
    layers: Vec<rendy::texture::Texture<B>>,
    // patch_resolution: u16,
    // index_buffer: Escape<Buffer<B>>,
    // index_buffer_partial: Escape<Buffer<B>>,
}
impl<B: Backend> RenderGroup<B, World> for TerrainRenderGroup<B> {
    fn prepare(
        &mut self,
        factory: &Factory<B>,
        queue: QueueId,
        _index: usize,
        _subpass: Subpass<B>,
        aux: &World,
    ) -> PrepareResult {
        let CameraGatherer {
            camera_position,
            projview,
        } = CameraGatherer::gather(aux);

        unsafe {
            factory
                .upload_visible_buffer(
                    &mut self.uniform_buffer,
                    0,
                    &[UniformBlock {
                        camera: camera_position,
                        view: projview.view,
                        projection: projview.proj,
                        padding: 0.0,
                    }],
                )
                .unwrap();
        }

        let mut quadtree = aux.fetch_mut::<QuadTree>();
        quadtree.update(
            *<&cgmath::Point3<f32>>::from(camera_position.as_ref() as &[f32; 3]),
            None,
        );
        quadtree.prepare_vertex_buffer(factory, queue, &mut self.vertex_buffer);

        PrepareResult::DrawRecord
    }

    fn draw_inline(
        &mut self,
        mut encoder: RenderPassEncoder<B>,
        _index: usize,
        _subpass: Subpass<B>,
        aux: &World,
    ) {
        encoder.bind_graphics_pipeline(&self.graphics_pipeline);
        unsafe {
            encoder.bind_graphics_descriptor_sets(
                &self.pipeline_layout,
                0,
                self.descriptor_sets.iter().map(|s| s.raw()),
                std::iter::empty::<u32>(),
            );
        }

        aux.read_resource::<QuadTree>().render(
            &mut encoder,
            &self.vertex_buffer,
            &self.index_buffer,
            &self.index_buffer_partial,
        );
    }

    fn dispose(self: Box<Self>, factory: &mut Factory<B>, _aux: &World) {
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
