
use gfx;
use gfx_core;
use gfx::traits::*;
use gfx::format::*;
use vecmath;

use heightmap::Heightmap;

type RenderTarget = gfx::RenderTarget<Srgba8>;
type DepthTarget = gfx::DepthTarget<DepthStencil>;

gfx_defines!{
    vertex Vertex {
        pos: [f32; 2] = "vPosition",
    }
}
gfx_pipeline!( pipe {
    vertex: gfx::VertexBuffer<Vertex> = (),
    model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
    position: gfx::Global<[f32; 3]> = "position",
    scale: gfx::Global<[f32; 3]> = "scale",
    heights: gfx::TextureSampler<f32> = "heights",
    normals: gfx::TextureSampler<[f32; 4]> = "normals",
    shadows: gfx::TextureSampler<f32> = "shadows",
    out_color: RenderTarget = "OutColor",
    out_depth: DepthTarget = gfx::preset::depth::LESS_EQUAL_WRITE,
});

gfx_pipeline!( generate_textures {
    y_scale: gfx::Global<f32> = "yScale",
    heights: gfx::TextureSampler<f32> = "heights",
    normals: gfx::RenderTarget<Rgba8> = "normals",
    shadows: gfx::RenderTarget<(R16, Unorm)> = "shadows",
});

type HeightMap<R> = (gfx_core::handle::Texture<R, gfx_core::format::R16>,
                  gfx_core::handle::ShaderResourceView<R, f32>,
                  gfx_core::handle::RenderTargetView<R, (R16, Unorm)>);

type ColorMap<R> = (gfx_core::handle::Texture<R, gfx_core::format::R8_G8_B8_A8>,
                 gfx_core::handle::ShaderResourceView<R, [f32; 4]>,
                 gfx_core::handle::RenderTargetView<R, Rgba8>);

type NormalMap<R> = (gfx_core::handle::Texture<R, gfx_core::format::R8_G8_B8_A8>,
                  gfx_core::handle::ShaderResourceView<R, [f32; 4]>,
                  gfx_core::handle::RenderTargetView<R, Rgba8>);

type ShadowMap<R> = (gfx_core::handle::Texture<R, gfx_core::format::R16>,
                  gfx_core::handle::ShaderResourceView<R, f32>,
                  gfx_core::handle::RenderTargetView<R, (R16, Unorm)>);

struct Clipmap <R> where R: gfx::Resources {
    side_length: i64,
    vertex_resolution: i64,
    texture_resolution: i64,

    pso: gfx::PipelineState<R, pipe::Meta>,
    slice: gfx::Slice<R>,

    layers: Vec<ClipmapLayer<R>>,
}

enum ClipmapLayer <R> where R: gfx::Resources {
    Precomputed {
        x: i64,
        y: i64,
        pipeline_data: pipe::Data<R>,
        colors: ColorMap<R>,
        heights: HeightMap<R>,
    },
    Static {
        x: i64,
        y: i64,
        pipeline_data: pipe::Data<R>,
        // heights: HeightMap<R>,
        normals: NormalMap<R>,
        shadows: ShadowMap<R>,
    },
    Generated {
        x: i64,
        y: i64,
        pipeline_data: pipe::Data<R>,
        heights: HeightMap<R>,
        normals: NormalMap<R>,
        shadows: ShadowMap<R>,
    }
}

impl<R> Clipmap <R> where R: gfx::Resources {
    pub fn generate_textures<F, C>(&mut self, factory: &mut F, encoder: &mut gfx::Encoder<R, C>)
        where C: gfx_core::command::Buffer<R>,
              F: gfx::Factory<R>
    {
        let slice = gfx::Slice {
            start: 0,
            end: 6,
            base_vertex: 0,
            instances: None,
            buffer: gfx::IndexBuffer::Auto,
        };

        for layer in self.layers.iter_mut() {
            match layer {
                &mut ClipmapLayer::Static {x, y,
                                      ref mut pipeline_data,
                                      ref mut normals,
                                      ref mut shadows} => {
                    let data = generate_textures::Data {
                        heights: pipeline_data.heights.clone(),
                        normals: normals.2.clone(),
                        shadows: shadows.2.clone(),
                        y_scale: 320.0,
                    };

                    let pso = factory.create_pipeline_simple(
                        include_str!("../assets/generate_textures.glslv").as_bytes(),
                        include_str!("../assets/generate_textures.glslf").as_bytes(),
                        generate_textures::new()
                    ).unwrap();

                    encoder.draw(&slice, &pso, &data);
                }
                _ => unimplemented!(),
            }
        }
    }
    pub fn update(&mut self, mvp_mat: vecmath::Matrix4<f32>) {
        for layer in self.layers.iter_mut() {
            match layer {
                &mut ClipmapLayer::Precomputed {ref mut pipeline_data, ..} =>
                    pipeline_data.model_view_projection = mvp_mat,
                &mut ClipmapLayer::Static {ref mut pipeline_data, ..} =>
                    pipeline_data.model_view_projection = mvp_mat,
                &mut ClipmapLayer::Generated {ref mut pipeline_data, ..} =>
                    pipeline_data.model_view_projection = mvp_mat,
            }
        }
    }
    pub fn render<C>(&self, encoder: &mut gfx::Encoder<R, C>)
        where C: gfx_core::command::Buffer<R>
    {
        for layer in self.layers.iter() {
            match layer {
                &ClipmapLayer::Precomputed {ref pipeline_data, ..} =>
                    encoder.draw(&self.slice, &self.pso, pipeline_data),
                &ClipmapLayer::Static {ref pipeline_data, ..} =>
                    encoder.draw(&self.slice, &self.pso, pipeline_data),
                &ClipmapLayer::Generated {ref pipeline_data, ..} =>
                    encoder.draw(&self.slice, &self.pso, pipeline_data),
            }
        }
    }

}

pub struct Terrain <R, F> where R: gfx::Resources, F: gfx::Factory<R>{
    factory: F,
    clipmap: Clipmap<R>,
}

impl<R, F> Terrain <R, F> where R: gfx::Resources, F: gfx::Factory<R> {
    pub fn new(heightmap: Heightmap<u16>,
               mut factory: F,
               out_color: <RenderTarget as gfx::pso::DataBind<R>>::Data,
               out_stencil: <DepthTarget as gfx::pso::DataBind<R>>::Data) -> Self
    {
        let resolution = 64;

        let denom = (resolution + 1) as f32;
        let mut vertices = Vec::new();
        for x in 0..resolution {
            for y in 0..resolution {
                let fx = x as f32;
                let fy = y as f32;

                vertices.push(Vertex{pos: [fx / denom, fy / denom]});
                vertices.push(Vertex{pos: [(fx+1.0) / denom, fy / denom]});
                vertices.push(Vertex{pos: [fx / denom, (fy+1.0) / denom]});

                vertices.push(Vertex{pos: [(fx+1.0) / denom, (fy+1.0) / denom]});
                vertices.push(Vertex{pos: [(fx+1.0) / denom, fy / denom]});
                vertices.push(Vertex{pos: [fx / denom, (fy+1.0) / denom]});
            }
        }
        let (vertex_buffer, slice) = factory.create_vertex_buffer_with_slice(&vertices, ());

        let w = heightmap.width;
        let h = heightmap.height;
        let (_, texture_view) = factory.create_texture_immutable::<(R16, Unorm)>(
            gfx::texture::Kind::D2(w, h, gfx::texture::AaMode::Single),
            &[&heightmap.heights[..]]).unwrap();

        let sinfo = gfx::texture::SamplerInfo::new(
            gfx::texture::FilterMethod::Bilinear,
            gfx::texture::WrapMode::Clamp);
        let sampler = factory.create_sampler(sinfo);

        let shaders = gfx::ShaderSet::Simple(
            factory.create_shader_vertex(
                include_str!("../assets/clipmap.glslv").as_bytes()).unwrap(),
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
            gfx::Primitive::TriangleList,
            rasterizer,
            pipe::new()
        ).unwrap();

        let mut clipmap = Clipmap {
            side_length: 1024,
            vertex_resolution: 64,
            texture_resolution: 1024,
            pso: pso,
            slice: slice,
            layers: Vec::new(),
        };

        let normals = factory.create_render_target::<Rgba8>(w, h).unwrap();
        let shadows = factory.create_render_target::<(R16, Unorm)>(w, h).unwrap();
        clipmap.layers.push(ClipmapLayer::Static{
            x: 0, y: 0,
            pipeline_data: pipe::Data {
                vertex: vertex_buffer,
                model_view_projection: [[0.0; 4]; 4],
                position: [0.0, 0.0, 0.0],
                scale: [3.0, 3.0, 3.0],
                heights: (texture_view, sampler.clone()),
                normals: (normals.1.clone(), sampler.clone()),
                shadows: (shadows.1.clone(), sampler),
                out_color: out_color,
                out_depth: out_stencil,
            },
            normals: normals.clone(),
            shadows: shadows.clone(),
        });

        Terrain {
            factory: factory,
            clipmap: clipmap,
        }
    }

    pub fn generate_textures<C>(&mut self, encoder: &mut gfx::Encoder<R, C>)
        where C: gfx_core::command::Buffer<R>
    {
        self.clipmap.generate_textures(&mut self.factory, encoder);
    }

    pub fn update(&mut self, mvp_mat: vecmath::Matrix4<f32>) {
        self.clipmap.update(mvp_mat);
    }

    pub fn render<C>(&self, encoder: &mut gfx::Encoder<R, C>)
        where C: gfx_core::command::Buffer<R>
    {
        self.clipmap.render(encoder);
    }
}
