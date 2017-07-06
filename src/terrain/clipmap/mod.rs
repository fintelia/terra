use gfx;
use gfx_core;
use gfx::traits::*;
use gfx::format::*;
use vecmath::*;

use std::mem;

use terrain::dem;
use terrain::heightmap;
use terrain::vertex_buffer::{self, ClipmapLayerKind};

type RenderTarget = gfx::RenderTarget<Srgba8>;
type DepthTarget = gfx::DepthTarget<DepthStencil>;

gfx_defines!{
    vertex Vertex {
        pos: [u8; 2] = "vPosition",
    }
}
gfx_pipeline!( pipe {
    vertex: gfx::VertexBuffer<Vertex> = (),
    model_view_projection: gfx::Global<[[f32; 4]; 4]> = "modelViewProjection",
    resolution: gfx::Global<i32> = "resolution",
    position: gfx::Global<[f32; 3]> = "position",
    scale: gfx::Global<[f32; 3]> = "scale",
    flip_axis: gfx::Global<[i32; 2]> = "flipAxis",
    texture_step: gfx::Global<i32> = "textureStep",
    texture_offset: gfx::Global<[i32; 2]> = "textureOffset",
    heights: gfx::TextureSampler<f32> = "heights",
    normals: gfx::TextureSampler<[f32; 4]> = "normals",
    detail: gfx::TextureSampler<[f32; 3]> = "detail",
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

pub struct Clipmap<R, F>
    where R: gfx::Resources,
          F: gfx::Factory<R>
{
    spacing: f32,
    world_width: f32,
    world_height: f32,
    mesh_resolution: i64,

    factory: F,
    pso: gfx::PipelineState<R, pipe::Meta>,
    ring1_slice: gfx::Slice<R>,
    ring2_slice: gfx::Slice<R>,
    center_slice: gfx::Slice<R>,

    dem: dem::DigitalElevationModel,
    layers: Vec<ClipmapLayer<R>>,
}

enum ClipmapLayer<R>
    where R: gfx::Resources
{
    #[allow(unused)]
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
    #[allow(unused)]
    Generated {
        x: i64,
        y: i64,
        pipeline_data: pipe::Data<R>,
        heights: HeightMap<R>,
        normals: NormalMap<R>,
        shadows: ShadowMap<R>,
    },
}

impl<R, F> Clipmap<R, F>
    where R: gfx::Resources,
          F: gfx::Factory<R>
{
    pub fn new(dem: dem::DigitalElevationModel,
               mut factory: F,
               out_color: &<RenderTarget as gfx::pso::DataBind<R>>::Data,
               out_stencil: &<DepthTarget as gfx::pso::DataBind<R>>::Data)
               -> Self {

        let mesh_resolution: u8 = 63;
        let num_layers = 6;
        let spacing = 30.0;

        let ring1_vertices = vertex_buffer::generate(mesh_resolution, ClipmapLayerKind::Ring1);
        let ring2_vertices = vertex_buffer::generate(mesh_resolution, ClipmapLayerKind::Ring2);
        let center_vertices = vertex_buffer::generate(mesh_resolution, ClipmapLayerKind::Center);

        let ring1_slice = gfx::Slice {
            start: 0,
            end: ring1_vertices.len() as u32,
            base_vertex: 0,
            instances: None,
            buffer: gfx::IndexBuffer::Auto,
        };
        let ring2_slice = gfx::Slice {
            start: ring1_vertices.len() as u32,
            end: (ring1_vertices.len() + ring2_vertices.len()) as u32,
            base_vertex: 0,
            instances: None,
            buffer: gfx::IndexBuffer::Auto,
        };
        let center_slice = gfx::Slice {
            start: (ring1_vertices.len() + ring2_vertices.len()) as u32,
            end: (ring1_vertices.len() + ring2_vertices.len() + center_vertices.len()) as u32,
            base_vertex: 0,
            instances: None,
            buffer: gfx::IndexBuffer::Auto,
        };

        let combined_vertices: Vec<_> = ring1_vertices
            .into_iter()
            .chain(ring2_vertices.into_iter())
            .chain(center_vertices.into_iter())
            .collect();
        let vertex_buffer = factory.create_vertex_buffer(&combined_vertices);

        let w = dem.width as u16;
        let h = dem.height as u16;
        let heights: Vec<u32> = dem.elevations
            .iter()
            .map(|h| unsafe { mem::transmute::<f32, u32>(*h) })
            .collect();
        let (_, texture_view) = factory.create_texture_immutable::<(R32, Float)>(
            gfx::texture::Kind::D2(w, h, gfx::texture::AaMode::Single),
            &[&heights[..]]).unwrap();

        let sinfo = gfx::texture::SamplerInfo::new(gfx::texture::FilterMethod::Bilinear,
                                                   gfx::texture::WrapMode::Clamp);
        let sampler = factory.create_sampler(sinfo);

        let v = match factory
                  .create_shader_vertex(include_str!("../../shaders/glsl/clipmap.glslv")
                                            .as_bytes()) {
            Ok(s) => s,
            Err(msg) => {
                println!("{}", msg);
                panic!("Failed to compile clipmap.glslv");
            }
        };

        let f = match factory
                  .create_shader_pixel(include_str!("../../shaders/glsl/clipmap.glslf")
                                           .as_bytes()) {
            Ok(s) => s,
            Err(msg) => {
                println!("{}", msg);
                panic!("Failed to compile clipmap.glslf");
            }
        };

        let detail_heightmap = heightmap::detail_heightmap(512, 512);
        let detailmap = detail_heightmap.as_height_and_slopes(1.0);
        let detailmap: Vec<[u32; 3]> = detailmap
            .into_iter()
            .map(|n| unsafe {
                     [mem::transmute::<f32, u32>(n[0]),
                      mem::transmute::<f32, u32>(n[1]),
                      mem::transmute::<f32, u32>(n[2])]
                 })
            .collect();
        let detail_texture = factory
            .create_texture_immutable::<(R32_G32_B32, Float)>(gfx::texture::Kind::D2(512,
                                                                    512,
                                                                    gfx::texture::AaMode::Single),
                                             &[&detailmap[..], &(vec![[0,0,0]; 65536])[..]])
            .unwrap();
        let sinfo_wrap =
            gfx::texture::SamplerInfo::new(gfx::texture::FilterMethod::Anisotropic(16),
                                           gfx::texture::WrapMode::Tile);
        let sampler_wrap = factory.create_sampler(sinfo_wrap);

        let rasterizer = gfx::state::Rasterizer {
            front_face: gfx::state::FrontFace::Clockwise,
            cull_face: gfx::state::CullFace::Nothing,
            method: gfx::state::RasterMethod::Fill,
            offset: None,
            samples: None,
        };
        let pso = factory
            .create_pipeline_state(&gfx::ShaderSet::Simple(v, f),
                                   gfx::Primitive::TriangleList,
                                   rasterizer,
                                   pipe::new())
            .unwrap();

        let size = spacing * ((mesh_resolution as i64 - 1) << (num_layers - 1)) as f32;

        let mut layers = Vec::new();
        for layer in 0..num_layers {
            let normals = factory.create_render_target::<Rgba8>(w, h).unwrap();
            let shadows = factory
                .create_render_target::<(R16, Unorm)>(w, h)
                .unwrap();

            layers.push(ClipmapLayer::Static {
                            x: 0,
                            y: 0,
                            pipeline_data: pipe::Data {
                                vertex: vertex_buffer.clone(),
                                model_view_projection: [[0.0; 4]; 4],
                                resolution: mesh_resolution as i32,
                                position: [0.0, 0.0, 0.0],
                                scale: [size / (1u64 << layer) as f32,
                                        1.0,
                                        size / (1u64 << layer) as f32],
                                flip_axis: [0, 0],
                                texture_step: 1 << (num_layers - layer - 1),
                                texture_offset: [0, 0],
                                heights: (texture_view.clone(), sampler.clone()),
                                normals: (normals.1.clone(), sampler.clone()),
                                shadows: (shadows.1.clone(), sampler.clone()),
                                detail: (detail_texture.1.clone(), sampler_wrap.clone()),
                                out_color: out_color.clone(),
                                out_depth: out_stencil.clone(),
                            },
                            normals: normals.clone(),
                            shadows: shadows.clone(),
                        });
        }

        Clipmap {
            spacing,
            world_width: spacing * dem.width as f32,
            world_height: spacing * dem.height as f32,
            mesh_resolution: mesh_resolution as i64,

            factory,
            pso,
            ring1_slice,
            ring2_slice,
            center_slice,

            dem,
            layers,
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

        let pso = self.factory
            .create_pipeline_simple(include_str!("../../shaders/glsl/generate_textures.glslv")
                                        .as_bytes(),
                                    include_str!("../../shaders/glsl/generate_textures.glslf")
                                        .as_bytes(),
                                    generate_textures::new())
            .unwrap();

        for layer in self.layers.iter_mut() {
            match layer {
                &mut ClipmapLayer::Static {
                         ref mut pipeline_data,
                         ref mut normals,
                         ref mut shadows,
                         ..
                     } => {
                    let data = generate_textures::Data {
                        heights: pipeline_data.heights.clone(),
                        normals: normals.2.clone(),
                        shadows: shadows.2.clone(),
                        y_scale: 1.0 / self.spacing,
                    };


                    encoder.draw(&slice, &pso, &data);
                }
                _ => unimplemented!(),
            }
        }
    }

    pub fn update(&mut self, mvp_mat: Matrix4<f32>, center: Vector2<f32>) {
        let center = (((center[0] + self.world_width * 0.5) / self.spacing) as i64,
                      ((center[1] + self.world_height * 0.5) / self.spacing) as i64);
        // TODO: clamp center to bound

        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_scale = 1 << (num_layers - i - 1);
            let step = layer_scale * 2;
            let half_step = step / 2;

            let target_center = ((center.0 / step) * step + half_step,
                                 (center.1 / step) * step + half_step);

            let flip_axis = if i < num_layers - 1 {
                [((((center.0 / half_step + 1) % 2) + 2) % 2) as i32,
                 ((((center.1 / half_step + 1) % 2) + 2) % 2) as i32]
            } else {
                [0, 0]
            };

            match *layer {
                ClipmapLayer::Precomputed {
                    ref mut pipeline_data,
                    ref mut x,
                    ref mut y,
                    ..
                } |
                ClipmapLayer::Static {
                    ref mut pipeline_data,
                    ref mut x,
                    ref mut y,
                    ..
                } |
                ClipmapLayer::Generated {
                    ref mut pipeline_data,
                    ref mut x,
                    ref mut y,
                    ..
                } => {
                    *x = target_center.0 - (self.mesh_resolution - 1) / 2 * layer_scale;
                    *y = target_center.1 - (self.mesh_resolution - 1) / 2 * layer_scale;

                    pipeline_data.position = [*x as f32 * self.spacing - 0.5 * self.world_width,
                                              0.0,
                                              *y as f32 * self.spacing - 0.5 * self.world_width];
                    pipeline_data.model_view_projection = mvp_mat;
                    pipeline_data.flip_axis = flip_axis;
                    pipeline_data.texture_offset =
                        [((*x as i32) * pipeline_data.texture_step / half_step as i32),
                         ((*y as i32) * pipeline_data.texture_step / half_step as i32)];
                }
            }
        }
    }

    pub fn render<C>(&self, encoder: &mut gfx::Encoder<R, C>)
        where C: gfx_core::command::Buffer<R>
    {
        for (i, layer) in self.layers.iter().enumerate().rev() {
            match *layer {
                ClipmapLayer::Precomputed { ref pipeline_data, .. } |
                ClipmapLayer::Static { ref pipeline_data, .. } |
                ClipmapLayer::Generated { ref pipeline_data, .. } => {
                    let ref slice = if i == self.layers.len() - 1 {
                        &self.center_slice
                    } else if pipeline_data.flip_axis[0] ==
                              pipeline_data.flip_axis[1] {
                        &self.ring1_slice
                    } else {
                        &self.ring2_slice
                    };

                    encoder.draw(slice, &self.pso, pipeline_data)
                }
            }
        }
    }

    /// Returns the approximate height at `position`.
    pub fn get_approximate_height(&self, position: Vector2<f32>) -> Option<f32> {
        let x = position[0] / self.spacing + 0.5 * (self.dem.width - 1) as f32;
        let y = position[1] / self.spacing + 0.5 * (self.dem.height - 1) as f32;
        if x < 0.0 || y < 0.0 || x >= self.dem.width as f32 - 1.0 ||
           y >= self.dem.height as f32 - 1.0 {
            return None;
        }

        let ix = x.trunc() as usize;
        let iy = y.trunc() as usize;
        let fx = x.fract();
        let fy = y.fract();

        let h00 = self.dem.elevations[ix + iy * self.dem.width];
        let h10 = self.dem.elevations[(ix + 1) + iy * self.dem.width];
        let h01 = self.dem.elevations[ix + (iy + 1) * self.dem.width];
        let h11 = self.dem.elevations[(ix + 1) + (iy + 1) * self.dem.width];

        let h0 = h00 * (1.0 - fy) + h01 * fy;
        let h1 = h10 * (1.0 - fy) + h11 * fy;

        Some(h0 * (1.0 - fx) + h1 * fx)
    }
}
