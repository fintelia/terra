
use gfx;
use gfx_core;
use gfx::traits::*;
use vecmath;

// Boilerplate to suppress warning about unused variables/imports
// within gfx_vertex_struct macro, caused by using an empty vertex
// type.
#[allow(unused_imports)]
#[allow(unused_variables)]
mod vertex {
    gfx_vertex_struct!( V {});
    pub type Vertex = V;
}
use self::vertex::Vertex;

type RenderTarget = gfx::RenderTarget<::gfx::format::Srgba8>;
type DepthTarget = gfx::DepthTarget<::gfx::format::DepthStencil>;

gfx_pipeline!( pipe {
    vbuf: gfx::VertexBuffer<Vertex> = (),
    model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
    resolution: gfx::Global<i32> = "resolution",
    t_color: gfx::TextureSampler<[f32; 4]> = "t_color",
    out_color: RenderTarget = "OutColor",
    out_depth: DepthTarget = gfx::preset::depth::LESS_EQUAL_WRITE,
});

pub struct Terrain <R> where R: gfx::Resources{
    pso: gfx::PipelineState<R, pipe::Meta>,
    slice: gfx::Slice<R>,
    data: pipe::Data<R>,
}

impl<R> Terrain <R> where R: gfx::Resources {
    pub fn new<F, >(factory: &mut F,
                    out_color: <RenderTarget as gfx::pso::DataBind<R>>::Data,
                    out_stencil: <DepthTarget as gfx::pso::DataBind<R>>::Data) -> Self
        where F: gfx::Factory<R>,
    {
        let resolution = 8;

        let mut vertex_data = Vec::new();
        vertex_data.resize(4 * resolution * resolution, Vertex{});

        let vbuf = factory.create_vertex_buffer(&vertex_data);
        let slice = gfx::Slice {
            start: 0,
            end: vertex_data.len() as u32,
            base_vertex: 0,
            instances: None,
            buffer: gfx::IndexBuffer::Auto,
        };

        let texels = [
            [0xff, 0xff, 0xff, 0x00],
            [0xff, 0x00, 0x00, 0x00],
            [0x00, 0xff, 0x00, 0x00],
            [0x00, 0x00, 0xff, 0x00]
        ];
        let (_, texture_view) = factory.create_texture_immutable::<gfx::format::Rgba8>(
            gfx::texture::Kind::D2(2, 2, gfx::texture::AaMode::Single),
            &[&texels]).unwrap();

        let sinfo = gfx::texture::SamplerInfo::new(
            gfx::texture::FilterMethod::Bilinear,
            gfx::texture::WrapMode::Clamp);


        let data = pipe::Data {
            vbuf: vbuf.clone(),
            model_view_projection: [[0.0; 4]; 4],
            resolution: resolution as i32,
            t_color: (texture_view, factory.create_sampler(sinfo)),
            out_color: out_color,
            out_depth: out_stencil,
        };

        let shaders = gfx::ShaderSet::Tessellated(
            factory.create_shader_vertex(
                include_str!("../assets/clipmap.glslv").as_bytes()).unwrap(),
            factory.create_shader_hull(
                include_str!("../assets/clipmap.glslh").as_bytes()).unwrap(),
            factory.create_shader_domain(
                include_str!("../assets/clipmap.glsld").as_bytes()).unwrap(),
            factory.create_shader_pixel(
                include_str!("../assets/clipmap.glslf").as_bytes()).unwrap()
        );
        let rasterizer = gfx::state::Rasterizer {
            front_face: gfx::state::FrontFace::Clockwise,
            cull_face: gfx::state::CullFace::Nothing,
            method: gfx::state::RasterMethod::Line(1),
            offset: None,
            samples: None,
        };
        let pso = factory.create_pipeline_state(
            &shaders,
            gfx::Primitive::PatchList(4),
            rasterizer,
            pipe::new()
        ).unwrap();

        Terrain {
            pso: pso,
            slice: slice,
            data: data,
        }
    }

    pub fn update(&mut self, mvp_mat: vecmath::Matrix4<f32>) {
        self.data.model_view_projection = mvp_mat;
    }
    
    pub fn render<C>(&self, encoder: &mut gfx::Encoder<R, C>)
        where C: gfx_core::command::Buffer<R>
    {
        encoder.draw(&self.slice, &self.pso, &self.data);
    }
}
