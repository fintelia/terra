
use gfx;
use gfx_device_gl;
use vecmath;

use piston_window::*;
use gfx::traits::*;
use sdl2_window::Sdl2Window;

gfx_vertex_struct!( Vertex {
    pos: [i8; 4] = "pos",
});

impl Vertex {
    pub fn new(pos: [i8; 3]) -> Vertex {
        Vertex {
            pos: [pos[0], pos[1], pos[2], 1],
        }
    }
}

gfx_pipeline!( pipe {
    vbuf: gfx::VertexBuffer<Vertex> = (),
    model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
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

        let vertex_data = vec![
            Vertex::new([-3, 0, 3]),
            Vertex::new([ 3, 0, 3]),
            Vertex::new([ 3, 0, -3]),
            Vertex::new([-3, 0, -3]),
        ];
        
        let index_data:Vec<u16> = vec![
            0,  1,  3,  2,//2,  3,  0, // top
        ];

        let (vbuf, slice) = factory.create_vertex_buffer_with_slice
            (&vertex_data, &index_data[..]);

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
