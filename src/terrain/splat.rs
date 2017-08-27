use gfx;
use gfx::traits::FactoryExt;
use gfx_core::format::*;
use gfx_core::handle::{Texture, ShaderResourceView};
use gfx_core::command::Buffer;
use rshader;

gfx_pipeline!( pipe {
    heights: gfx::TextureSampler<f32> = "heights",
    slopes: gfx::TextureSampler<[f32; 2]> = "slopes",
    noise: gfx::TextureSampler<[f32; 3]> = "noise",
    noise_wavelength: gfx::Global<f32> = "noiseWavelength",
    texture_spacing: gfx::Global<f32> = "textureSpacing",
    texture_resolution: gfx::Global<i32> = "resolution",
    heights_spacing: gfx::Global<f32> = "heightsSpacing",
//    materials: gfx::TextureSampler<[f32; 4]> = "materials",
    out_color: gfx::RenderTarget<Srgba8> = "OutColor",
    out_splat: gfx::RenderTarget<(R8, Uint)> = "OutSplat",
});

pub struct Splat<R: gfx::Resources> {
    pso: gfx::PipelineState<R, pipe::Meta>,
    pipeline_data: pipe::Data<R>,
    shader: rshader::Shader<R>,

    _splatmap: Texture<R, R8>,
    _colormap: Texture<R, R8_G8_B8_A8>,

    splatmap_view: ShaderResourceView<R, u32>,
    colormap_view: ShaderResourceView<R, [f32; 4]>,
}

impl<R: gfx::Resources> Splat<R> {
    fn create_pso<F: gfx::Factory<R>>(
        factory: &mut F,
        shaders: &gfx::ShaderSet<R>,
    ) -> gfx::PipelineState<R, pipe::Meta> {
        factory
            .create_pipeline_state(
                shaders,
                gfx::Primitive::TriangleList,
                gfx::state::Rasterizer {
                    front_face: gfx::state::FrontFace::Clockwise,
                    cull_face: gfx::state::CullFace::Nothing,
                    method: gfx::state::RasterMethod::Fill,
                    offset: None,
                    samples: None,
                },
                pipe::new(),
            )
            .unwrap()
    }

    fn update<C: Buffer<R>>(&mut self, encoder: &mut gfx::Encoder<R, C>) {
        encoder.draw(
            &gfx::Slice {
                start: 0,
                end: 6,
                base_vertex: 0,
                instances: None,
                buffer: gfx::IndexBuffer::Auto,
            },
            &self.pso,
            &self.pipeline_data,
        );
        encoder.generate_mipmap(&self.colormap_view);
    }

    pub fn new<F: gfx::Factory<R>, C: Buffer<R>>(
        factory: &mut F,
        encoder: &mut gfx::Encoder<R, C>,
        shaders_watcher: &mut rshader::ShaderDirectoryWatcher,
        resolution: u16,
        texture_spacing: f32,
        heights: ShaderResourceView<R, f32>,
        slopes: ShaderResourceView<R, [f32; 2]>,
        noise: ShaderResourceView<R, [f32; 3]>,
        noise_wavelength: f32,
        heights_spacing: f32,
    ) -> Self {
        let shader = rshader::Shader::simple(
            factory,
            shaders_watcher,
            shader_source!("../shaders/glsl", "version", "splatmap.glslv"),
            shader_source!("../shaders/glsl", "version", "fractal", "splatmap.glslf"),
        ).unwrap();

        let num_mipmaps = 15 - (resolution - 1).leading_zeros() as u8;

        let splatmap = factory
            .create_texture::<R8>(
                gfx::texture::Kind::D2(resolution, resolution, gfx::texture::AaMode::Single),
                1,
                gfx::RENDER_TARGET | gfx::SHADER_RESOURCE,
                gfx::memory::Usage::Dynamic,
                Some(ChannelType::Uint),
            )
            .unwrap();
        let colormap = factory
            .create_texture::<R8_G8_B8_A8>(
                gfx::texture::Kind::D2(resolution, resolution, gfx::texture::AaMode::Single),
                num_mipmaps,
                gfx::RENDER_TARGET | gfx::SHADER_RESOURCE,
                gfx::memory::Usage::Dynamic,
                Some(ChannelType::Unorm),
            )
            .unwrap();

        let splatmap_view = factory
            .view_texture_as_shader_resource::<(R8, Uint)>(&splatmap, (0, 0), Swizzle::new())
            .unwrap();
        let colormap_view = factory
            .view_texture_as_shader_resource::<Rgba8>(&colormap, (0, 0), Swizzle::new())
            .unwrap();

        let sampler_clamp = factory.create_sampler(gfx::texture::SamplerInfo::new(
            gfx::texture::FilterMethod::Bilinear,
            gfx::texture::WrapMode::Clamp,
        ));
        let sampler_wrap = factory.create_sampler(gfx::texture::SamplerInfo::new(
            gfx::texture::FilterMethod::Bilinear,
            gfx::texture::WrapMode::Tile,
        ));

        let mut splat = Self {
            pso: Self::create_pso(factory, shader.as_shader_set()),
            pipeline_data: pipe::Data::<R> {
                heights: (heights, sampler_clamp.clone()),
                slopes: (slopes, sampler_clamp.clone()),
                noise: (noise, sampler_wrap),
                noise_wavelength,
                texture_spacing,
                texture_resolution: resolution as i32,
                heights_spacing,
                out_color: factory
                    .view_texture_as_render_target(&colormap, 0, None)
                    .unwrap(),
                out_splat: factory
                    .view_texture_as_render_target(&splatmap, 0, None)
                    .unwrap(),
            },
            shader,
            _splatmap: splatmap,
            splatmap_view,
            _colormap: colormap,
            colormap_view,
        };
        splat.update(encoder);
        splat
    }

    pub fn refresh<F: gfx::Factory<R>, C: Buffer<R>>(
        &mut self,
        factory: &mut F,
        encoder: &mut gfx::Encoder<R, C>,
        shaders_watcher: &mut rshader::ShaderDirectoryWatcher,
    ) -> bool {
        if self.shader.refresh(factory, shaders_watcher) {
            self.pso = Self::create_pso(factory, self.shader.as_shader_set());
            self.update(encoder);
            true
        } else {
            false
        }
    }

    pub fn splatmap_shader_resource_view(&self) -> ShaderResourceView<R, u32> {
        self.splatmap_view.clone()
    }
    pub fn colormap_shader_resource_view(&self) -> ShaderResourceView<R, [f32; 4]> {
        self.colormap_view.clone()
    }
}
