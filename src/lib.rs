//! Terra is a large scale terrain generation and rendering library built on top of rendy.
#![feature(custom_attribute)]

use std::collections::BTreeMap;

use rendy::{
    command::{Families, QueueId, RenderPassEncoder},
    factory::{Config, Factory},
    graph::{
        present::PresentNode, render::*, Graph, GraphBuilder, GraphContext, NodeBuffer, NodeImage,
    },
    memory::MemoryUsageValue,
    mesh::{AsVertex, PosColor},
    resource::Buffer,
    shader::{Shader, ShaderKind, SourceLanguage, StaticShaderInfo},
};

use rendy::{
    shader::{PathBufShaderInfo, ShaderSetBuilder},
    graph::render::*,
    memory::Dynamic,
    resource::{BufferInfo, DescriptorSetLayout, Escape, Handle},
};

use gfx_hal::pso::ShaderStageFlags;

use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

mod config;

type Backend = rendy::vulkan::Backend;

lazy_static::lazy_static! {
    static ref VERTEX: PathBufShaderInfo = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/src/tri.vert")),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    );

    static ref FRAGMENT: PathBufShaderInfo = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/src/tri.frag")),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    );

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).unwrap()
        .with_fragment(&*FRAGMENT).unwrap();
}

pub fn main() {
    let config: BTreeMap<String, config::Node> = toml::from_str(include_str!("../examples/graph.toml")).unwrap();
    println!("{:#?}", config);

    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .filter_module("triangle", log::LevelFilter::Trace)
        .init();

    let config: Config = Default::default();

    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();

    let mut event_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rendy Triangle")
        .build(&event_loop)
        .unwrap();

    event_loop.poll_events(|_| ());

    let surface = factory.create_surface(&window);

    let mut graph_builder = GraphBuilder::<Backend, ()>::new();

    let color = graph_builder.create_image(
        rendy::resource::Kind::D2(640, 480, 1, 1),
        1,
        factory.get_surface_format(&surface),
        Some(gfx_hal::command::ClearValue::Color(
            [1.0, 1.0, 1.0, 1.0].into(),
        )),
    );

    let pass = graph_builder.add_node(
        TriangleRenderPipeline::builder()
            .into_subpass()
            .with_color(color)
            .into_pass(),
    );

    graph_builder
        .add_node(PresentNode::builder(&factory, surface, color).with_dependency(pass));

    let graph = graph_builder
        .build(&mut factory, &mut families, &mut ())
        .unwrap();

    run(&mut event_loop, &mut factory, &mut families, graph).unwrap();
}

fn run(
    event_loop: &mut EventsLoop,
    factory: &mut Factory<Backend>,
    families: &mut Families<Backend>,
    mut graph: Graph<Backend, ()>,
) -> Result<(), failure::Error> {
    loop {
        factory.maintain(families);

        let mut should_close = false;
        event_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => should_close = true,
            _ => (),
        });
        if should_close {
            break;
        }
        graph.run(factory, families, &mut ());
    }

    graph.dispose(factory, &mut ());
    Ok(())
}

#[derive(Debug, Default)]
struct TriangleRenderPipelineDesc;

#[derive(Debug)]
struct TriangleRenderPipeline<B: gfx_hal::Backend> {
    vertex: Escape<Buffer<B>>,
}

impl<B, T> SimpleGraphicsPipelineDesc<B, T> for TriangleRenderPipelineDesc
where
    B: gfx_hal::Backend,
    T: ?Sized,
{
    type Pipeline = TriangleRenderPipeline<B>;

    fn vertices(
        &self,
    ) -> Vec<(
        Vec<gfx_hal::pso::Element<gfx_hal::format::Format>>,
        u32,
        gfx_hal::pso::VertexInputRate,
    )> {
        vec![PosColor::vertex().gfx_vertex_input_desc(gfx_hal::pso::VertexInputRate::Vertex)]
    }

    fn depth_stencil(&self) -> Option<gfx_hal::pso::DepthStencilDesc> {
        None
    }

    fn load_shader_set<'a>(
        &self,
        factory: &mut Factory<B>,
        _aux: &T,
    ) -> rendy::shader::ShaderSet<B> {
        SHADERS.build(factory, Default::default()).unwrap()
    }

    fn build<'a>(
        self,
        _ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        queue: QueueId,
        aux: &T,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<TriangleRenderPipeline<B>, failure::Error> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert!(set_layouts.is_empty());

        let mut vbuf = factory
            .create_buffer(
                BufferInfo {
                    size: PosColor::vertex().stride as u64 * 3,
                    usage: gfx_hal::buffer::Usage::VERTEX,
                },
                Dynamic,
            )
            .unwrap();

        unsafe {
            factory
                .upload_visible_buffer(
                    &mut vbuf,
                    0,
                    &[
                        PosColor {
                            position: [0.0, -0.5, 0.0].into(),
                            color: [1.0, 0.0, 0.0, 1.0].into(),
                        },
                        PosColor {
                            position: [0.5, 0.5, 0.0].into(),
                            color: [0.0, 1.0, 0.0, 1.0].into(),
                        },
                        PosColor {
                            position: [-0.5, 0.5, 0.0].into(),
                            color: [0.0, 0.0, 1.0, 1.0].into(),
                        },
                    ],
                )
                .unwrap();
        }

        Ok(TriangleRenderPipeline { vertex: vbuf })
    }
}

impl<B, T> SimpleGraphicsPipeline<B, T> for TriangleRenderPipeline<B>
where
    B: gfx_hal::Backend,
    T: ?Sized,
{
    type Desc = TriangleRenderPipelineDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        index: usize,
        aux: &T,
    ) -> PrepareResult {
        PrepareResult::DrawReuse
    }

    fn draw(
        &mut self,
        _layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        _index: usize,
        _aux: &T,
    ) {
        encoder.bind_vertex_buffers(0, Some((self.vertex.raw(), 0)));
        encoder.draw(0..3, 0..1);
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &T) {}
}
