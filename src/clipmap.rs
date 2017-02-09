
use gfx;
use gfx_core;
use gfx::traits::*;
use gfx::format::*;
use vecmath;

use heightmap::Heightmap;

type RenderTarget = gfx::RenderTarget<Srgba8>;
type DepthTarget = gfx::DepthTarget<DepthStencil>;

gfx_pipeline!( pipe {
    model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
    resolution: gfx::Global<i32> = "resolution",
    heights: gfx::TextureSampler<f32> = "heights",
    normals: gfx::TextureSampler<[f32; 4]> = "normals",
    out_color: RenderTarget = "OutColor",
    out_depth: DepthTarget = gfx::preset::depth::LESS_EQUAL_WRITE,
});

gfx_pipeline!( generate_textures {
    y_scale: gfx::Global<f32> = "yScale",
    heights: gfx::TextureSampler<f32> = "heights",
    normals: gfx::RenderTarget<Rgba8> = "normals",
});

pub struct Terrain <R, F> where R: gfx::Resources, F: gfx::Factory<R>{
    factory: F,
    pso: gfx::PipelineState<R, pipe::Meta>,
    slice: gfx::Slice<R>,
    data: pipe::Data<R>,

    normals: (gfx_core::handle::Texture<R, gfx_core::format::R8_G8_B8_A8>,
              gfx_core::handle::ShaderResourceView<R, [f32; 4]>,
              gfx_core::handle::RenderTargetView<R, Rgba8>),
}

impl<R, F> Terrain <R, F> where R: gfx::Resources, F: gfx::Factory<R> {
    pub fn new(heightmap: Heightmap<u16>,
               mut factory: F,
               out_color: <RenderTarget as gfx::pso::DataBind<R>>::Data,
               out_stencil: <DepthTarget as gfx::pso::DataBind<R>>::Data) -> Self
    {
        let resolution = 32;

        let slice = gfx::Slice {
            start: 0,
            end: resolution * resolution * 4,
            base_vertex: 0,
            instances: None,
            buffer: gfx::IndexBuffer::Auto,
        };

        let w = heightmap.width;
        let h = heightmap.height;
        let (_, texture_view) = factory.create_texture_immutable::<(R16, Unorm)>(
            gfx::texture::Kind::D2(w, h, gfx::texture::AaMode::Single),
            &[&heightmap.heights[..]]).unwrap();

        let sinfo = gfx::texture::SamplerInfo::new(
            gfx::texture::FilterMethod::Bilinear,
            gfx::texture::WrapMode::Clamp);
        let sampler = factory.create_sampler(sinfo);

        let normals = factory.create_render_target::<Rgba8>(w, h).unwrap();

        let data = pipe::Data {
            model_view_projection: [[0.0; 4]; 4],
            resolution: resolution as i32,
            heights: (texture_view, sampler.clone()),
            normals: (normals.1.clone(), sampler),
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
            method: gfx::state::RasterMethod::Fill,
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
            factory: factory,
            pso: pso,
            slice: slice,
            data: data,
            normals: normals,
        }
    }

    pub fn generate_textures<C>(&mut self, encoder: &mut gfx::Encoder<R, C>)
        where C: gfx_core::command::Buffer<R>
    {
        let slice = gfx::Slice {
            start: 0,
            end: 6,
            base_vertex: 0,
            instances: None,
            buffer: gfx::IndexBuffer::Auto,
        };

        let data = generate_textures::Data {
            heights: self.data.heights.clone(),
            normals: self.normals.2.clone(),
            y_scale: 320.0,
        };

        let pso = self.factory.create_pipeline_simple(
            include_str!("../assets/generate_textures.glslv").as_bytes(),
            include_str!("../assets/generate_textures.glslf").as_bytes(),
            generate_textures::new()
        ).unwrap();

        encoder.draw(&slice, &pso, &data);
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
