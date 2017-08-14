use gfx;
use gfx_core;
use gfx::traits::*;
use gfx::format::*;
use notify::{DebouncedEvent, RecommendedWatcher, Watcher, RecursiveMode, watcher};
use vecmath::*;

use std::env;
use std::io::{self, Read};
use std::fs::File;
use std::path::Path;
use std::time::Duration;
use std::sync::mpsc::{channel, Receiver};

use terrain::heightmap;
use terrain::vertex_buffer::{self, ClipmapLayerKind};
use terrain::file::TerrainFile;
use terrain::material::MaterialSet;

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
    texture_step: gfx::Global<f32> = "textureStep",
    texture_offset: gfx::Global<[f32; 2]> = "textureOffset",
    eye_position: gfx::Global<[f32; 3]> = "eyePosition",
    heights: gfx::TextureSampler<f32> = "heights",
    slopes: gfx::TextureSampler<[f32; 2]> = "slopes",
    shadows: gfx::TextureSampler<f32> = "shadows",
    noise: gfx::TextureSampler<[f32; 3]> = "noise",
    noise_wavelength: gfx::Global<f32> = "noiseWavelength",
    materials: gfx::TextureSampler<[f32; 4]> = "materials",
    out_color: RenderTarget = "OutColor",
    out_depth: DepthTarget = gfx::preset::depth::LESS_EQUAL_WRITE,
});

pub struct Clipmap<R, F>
where
    R: gfx::Resources,
    F: gfx::Factory<R>,
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

    _heights: gfx_core::handle::Texture<R, gfx_core::format::R32>,
    _slopes: gfx_core::handle::Texture<R, gfx_core::format::R32_G32>,
    _shadows: gfx_core::handle::Texture<R, gfx_core::format::R32>,

    shaders_directory: Option<String>,
    _shaders_watcher: Option<RecommendedWatcher>,
    shaders_watcher_rx: Receiver<DebouncedEvent>,

    num_fractal_layers: i32,
    terrain_file: TerrainFile,
    layers: Vec<ClipmapLayer<R>>,
}

struct ClipmapLayer<R>
where
    R: gfx::Resources,
{
    x: i64,
    y: i64,
    pipeline_data: pipe::Data<R>,
}

impl<R, F> Clipmap<R, F>
where
    R: gfx::Resources,
    F: gfx::Factory<R>,
{
    pub fn new<C>(
        terrain_file: TerrainFile,
        materials: MaterialSet<R>,
        mut factory: F,
        encoder: &mut gfx::Encoder<R, C>,
        out_color: &<RenderTarget as gfx::pso::DataBind<R>>::Data,
        out_stencil: &<DepthTarget as gfx::pso::DataBind<R>>::Data,
    ) -> Self
    where
        C: gfx_core::command::Buffer<R>,
    {
        let mesh_resolution: u8 = 31;
        let num_layers = 9;
        let spacing = terrain_file.cell_size();
        let num_fractal_layers = 3;
        let num_static_layers = num_layers - num_fractal_layers;

        let (tx, shaders_watcher_rx) = channel();
        let shaders_directory = env::var("TERRA_SHADER_DIRECTORY").ok();
        let mut shaders_watcher = watcher(tx, Duration::from_millis(50)).ok();
        if shaders_directory.is_some() && shaders_watcher.is_some() {
            let _ = shaders_watcher.as_mut().unwrap().watch(
                shaders_directory.as_ref().unwrap(),
                RecursiveMode::Recursive,
            );
        }

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

        let w = terrain_file.width() as u16;
        let h = terrain_file.height() as u16;
        let heights: Vec<u32> = terrain_file
            .elevations()
            .iter()
            .map(|h| h.to_bits())
            .collect();
        let heights_texture = factory
            .create_texture_immutable::<(R32, Float)>(
                gfx::texture::Kind::D2(w, h, gfx::texture::AaMode::Single),
                &[&heights[..]],
            )
            .unwrap();

        let slopes: Vec<[u32; 2]> = terrain_file
            .slopes()
            .iter()
            .map(|&(x, y)| [x.to_bits(), y.to_bits()])
            .collect();
        let slopes_texture = factory
            .create_texture_immutable::<(R32_G32, Float)>(
                gfx::texture::Kind::D2(w, h, gfx::texture::AaMode::Single),
                &[&slopes[..]],
            )
            .unwrap();

        let shadows: Vec<u32> = terrain_file
            .shadows()
            .iter()
            .map(|&s| s.to_bits())
            .collect();
        let shadows_texture = factory
            .create_texture_immutable::<(R32, Float)>(
                gfx::texture::Kind::D2(w, h, gfx::texture::AaMode::Single),
                &[&shadows[..]],
            )
            .unwrap();

        let sinfo = gfx::texture::SamplerInfo::new(
            gfx::texture::FilterMethod::Bilinear,
            gfx::texture::WrapMode::Clamp,
        );
        let sampler = factory.create_sampler(sinfo);

        let v = match factory.create_shader_vertex(
            include_str!("../../shaders/glsl/clipmap.glslv")
                .as_bytes(),
        ) {
            Ok(s) => s,
            Err(msg) => {
                println!("{}", msg);
                panic!("Failed to compile clipmap.glslv");
            }
        };

        let f = match factory.create_shader_pixel(
            include_str!("../../shaders/glsl/clipmap.glslf")
                .as_bytes(),
        ) {
            Ok(s) => s,
            Err(msg) => {
                println!("{}", msg);
                panic!("Failed to compile clipmap.glslf");
            }
        };

        let noise_heightmap = heightmap::wavelet_noise(64, 8);
        let noisemap = noise_heightmap.as_height_and_slopes(1.0 / 8.0);
        let noisemap: Vec<[u32; 3]> = noisemap
            .into_iter()
            .map(|n| [n[0].to_bits(), n[1].to_bits(), n[2].to_bits()])
            .collect();
        let noise_texture = factory
            .create_texture_immutable::<(R32_G32_B32, Float)>(
                gfx::texture::Kind::D2(512, 512, gfx::texture::AaMode::Single),
                &[&noisemap[..]],
            )
            .unwrap();
        encoder.generate_mipmap(&noise_texture.1);
        let sinfo_wrap = gfx::texture::SamplerInfo::new(
            gfx::texture::FilterMethod::Anisotropic(16),
            gfx::texture::WrapMode::Tile,
        );
        let sampler_wrap = factory.create_sampler(sinfo_wrap);

        let rasterizer = gfx::state::Rasterizer {
            front_face: gfx::state::FrontFace::Clockwise,
            cull_face: gfx::state::CullFace::Nothing,
            method: gfx::state::RasterMethod::Fill,
            offset: None,
            samples: None,
        };
        let pso = factory
            .create_pipeline_state(
                &gfx::ShaderSet::Simple(v, f),
                gfx::Primitive::TriangleList,
                rasterizer,
                pipe::new(),
            )
            .unwrap();

        let size = spacing * ((mesh_resolution as i64 - 1) << (num_static_layers - 1)) as f32;

        let mut layers = Vec::new();
        for layer in 0..num_layers {
            layers.push(ClipmapLayer {
                x: 0,
                y: 0,
                pipeline_data: pipe::Data {
                    vertex: vertex_buffer.clone(),
                    model_view_projection: [[0.0; 4]; 4],
                    resolution: mesh_resolution as i32,
                    position: [0.0, 0.0, 0.0],
                    scale: [
                        size / (1u64 << layer) as f32,
                        1.0,
                        size / (1u64 << layer) as f32,
                    ],
                    flip_axis: [0, 0],
                    texture_step: (2.0f32).powi(num_static_layers - layer - 1),
                    texture_offset: [0.0, 0.0],
                    eye_position: [0.0, 0.0, 0.0],
                    heights: (heights_texture.1.clone(), sampler.clone()),
                    slopes: (slopes_texture.1.clone(), sampler.clone()),
                    shadows: (shadows_texture.1.clone(), sampler.clone()),
                    noise: (noise_texture.1.clone(), sampler_wrap.clone()),
                    noise_wavelength: 1.0 / 64.0,
                    materials: (materials.view.clone(), sampler_wrap.clone()),
                    out_color: out_color.clone(),
                    out_depth: out_stencil.clone(),
                },
            });
        }

        Clipmap {
            spacing,
            world_width: spacing * terrain_file.width() as f32,
            world_height: spacing * terrain_file.height() as f32,
            mesh_resolution: mesh_resolution as i64,

            pso,
            factory,
            ring1_slice,
            ring2_slice,
            center_slice,

            _heights: heights_texture.0,
            _slopes: slopes_texture.0,
            _shadows: shadows_texture.0,

            shaders_directory,
            _shaders_watcher: shaders_watcher,
            shaders_watcher_rx,

            terrain_file,
            layers,
            num_fractal_layers,
        }
    }

    fn refresh_shaders(&mut self) {
        if self.shaders_directory.is_none() {
            return;
        }

        let mut needs_refresh = false;
        while let Ok(event) = self.shaders_watcher_rx.try_recv() {
            if let DebouncedEvent::Write(_) = event {
                needs_refresh = true;
            }
        }
        if !needs_refresh {
            return;
        }

        let file_contents = |filename| -> io::Result<String> {
            let mut file = File::open(filename)?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            Ok(contents)
        };

        let shaders_directory = Path::new(self.shaders_directory.as_ref().unwrap());
        let vertex = file_contents(shaders_directory.join("clipmap.glslv"));
        let fragment = file_contents(shaders_directory.join("clipmap.glslf"));
        if vertex.is_err() || fragment.is_err() {
            return;
        }
        let vertex_shader = self.factory.create_shader_vertex(
            vertex.unwrap().as_bytes(),
        );
        let fragment_shader = self.factory.create_shader_pixel(
            fragment.unwrap().as_bytes(),
        );

        match (vertex_shader, fragment_shader) {
            (Ok(v), Ok(f)) => {
                let rasterizer = gfx::state::Rasterizer {
                    front_face: gfx::state::FrontFace::Clockwise,
                    cull_face: gfx::state::CullFace::Nothing,
                    method: gfx::state::RasterMethod::Fill,
                    offset: None,
                    samples: None,
                };
                self.pso = self.factory
                    .create_pipeline_state(
                        &gfx::ShaderSet::Simple(v, f),
                        gfx::Primitive::TriangleList,
                        rasterizer,
                        pipe::new(),
                    )
                    .unwrap();
            }
            (Err(msg), _) => println!("{}\nFailed to compile vertex shader.", msg),
            (_, Err(msg)) => println!("{}\nFailed to compile fragment shader.", msg),
        }
    }
    pub fn update(&mut self, mvp_mat: Matrix4<f32>, camera: Vector3<f32>) {
        self.refresh_shaders();

        let spacing = self.spacing * (0.5f32).powi(self.num_fractal_layers);
        let center = (
            ((camera[0] + self.world_width * 0.5) / spacing).round() as i64,
            ((camera[2] + self.world_height * 0.5) / spacing).round() as i64,
        );
        // TODO: clamp center to bound

        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_scale = 1 << (num_layers - i - 1);
            let step = layer_scale * 2;
            let half_step = step / 2;

            let target_center = (
                (center.0 / step) * step + half_step,
                (center.1 / step) * step + half_step,
            );

            layer.x = target_center.0 - (self.mesh_resolution - 1) / 2 * layer_scale;
            layer.y = target_center.1 - (self.mesh_resolution - 1) / 2 * layer_scale;

            layer.pipeline_data.position = [
                layer.x as f32 * spacing - 0.5 * self.world_width,
                0.0,
                layer.y as f32 * spacing - 0.5 * self.world_height,
            ];
            layer.pipeline_data.texture_offset =
                [
                    ((layer.x as f32) * layer.pipeline_data.texture_step / half_step as f32),
                    ((layer.y as f32) * layer.pipeline_data.texture_step / half_step as f32),
                ];

            layer.pipeline_data.model_view_projection = mvp_mat;
            layer.pipeline_data.flip_axis = if i < num_layers - 1 {
                [
                    ((((center.0 / half_step + 1) % 2) + 2) % 2) as i32,
                    ((((center.1 / half_step + 1) % 2) + 2) % 2) as i32,
                ]
            } else {
                [0, 0]
            };
            layer.pipeline_data.eye_position = camera;
        }
    }

    pub fn render<C>(&self, encoder: &mut gfx::Encoder<R, C>)
    where
        C: gfx_core::command::Buffer<R>,
    {
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let ref slice = if i == self.layers.len() - 1 {
                &self.center_slice
            } else if layer.pipeline_data.flip_axis[0] == layer.pipeline_data.flip_axis[1] {
                &self.ring1_slice
            } else {
                &self.ring2_slice
            };

            encoder.draw(slice, &self.pso, &layer.pipeline_data)
        }
    }

    /// Returns the approximate height at `position`.
    pub fn get_approximate_height(&self, position: Vector2<f32>) -> Option<f32> {
        let x = position[0] / self.spacing + 0.5 * (self.terrain_file.width() - 1) as f32;
        let y = position[1] / self.spacing + 0.5 * (self.terrain_file.height() - 1) as f32;
        if x < 0.0 || y < 0.0 || x >= self.terrain_file.width() as f32 - 1.0 ||
            y >= self.terrain_file.height() as f32 - 1.0
        {
            return None;
        }

        let ix = x.trunc() as usize;
        let iy = y.trunc() as usize;
        let fx = x.fract();
        let fy = y.fract();

        let h00 = self.terrain_file.elevation(ix, iy);
        let h10 = self.terrain_file.elevation(ix + 1, iy);
        let h01 = self.terrain_file.elevation(ix, iy + 1);
        let h11 = self.terrain_file.elevation(ix + 1, iy + 1);

        let h0 = h00 * (1.0 - fy) + h01 * fy;
        let h1 = h10 * (1.0 - fy) + h11 * fy;

        Some(h0 * (1.0 - fx) + h1 * fx)
    }
}
