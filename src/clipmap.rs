
use gfx;
use gfx_device_gl;
use vecmath;

use piston_window::*;
use gfx::traits::*;
use sdl2_window::Sdl2Window;

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

gfx_pipeline!( pipe {
    vbuf: gfx::VertexBuffer<Vertex> = (),
    model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
    resolution: gfx::Global<i32> = "resolution",
    t_color: gfx::TextureSampler<[f32; 4]> = "t_color",
    out_color: gfx::RenderTarget<::gfx::format::Srgba8> = "OutColor",
    out_depth: gfx::DepthTarget<::gfx::format::DepthStencil> =
        gfx::preset::depth::LESS_EQUAL_WRITE,
});

pub struct Terrain {
    pso: gfx::PipelineState<gfx_device_gl::Resources, pipe::Meta>,
    slice: gfx::Slice<gfx_device_gl::Resources>,
    data: pipe::Data<gfx_device_gl::Resources>,
}

impl Terrain {
    pub fn new(window: &mut PistonWindow<Sdl2Window>) -> Self {
        let ref mut factory = window.factory;

        let resolution = 16;

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
            out_color: window.output_color.clone(),
            out_depth: window.output_stencil.clone(),
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
    
    pub fn render(&self, encoder: &mut GfxEncoder) {
        encoder.draw(&self.slice, &self.pso, &self.data);
    }
}
