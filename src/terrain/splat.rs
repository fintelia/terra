use gfx;
use gfx::traits::FactoryExt;
use gfx::Primitive::TriangleList;
use gfx::texture::FilterMethod::Bilinear;
use gfx::texture::WrapMode::*;
use gfx::texture::SamplerInfo;
use gfx_core::format::*;
use gfx_core::handle::{Texture, ShaderResourceView};
use gfx_core::command::Buffer;
use rshader;

gfx_pipeline!( splat_pipe {
    heights: gfx::TextureSampler<f32> = "heights",
    slopes: gfx::TextureSampler<[f32; 2]> = "slopes",
    noise: gfx::TextureSampler<[f32; 3]> = "noise",
    noise_wavelength: gfx::Global<f32> = "noiseWavelength",
    texture_spacing: gfx::Global<f32> = "textureSpacing",
    texture_resolution: gfx::Global<i32> = "resolution",
    heights_spacing: gfx::Global<f32> = "heightsSpacing",
//    materials: gfx::TextureSampler<[f32; 4]> = "materials",
    out_splat: gfx::RenderTarget<(R8, Uint)> = "OutSplat",
});
gfx_pipeline!( color_pipe {
    heights: gfx::TextureSampler<f32> = "heights",
    slopes: gfx::TextureSampler<[f32; 2]> = "slopes",
    noise: gfx::TextureSampler<[f32; 3]> = "noise",
    noise_wavelength: gfx::Global<f32> = "noiseWavelength",
    texture_spacing: gfx::Global<f32> = "textureSpacing",
    texture_resolution: gfx::Global<i32> = "resolution",
    heights_spacing: gfx::Global<f32> = "heightsSpacing",
//    materials: gfx::TextureSampler<[f32; 4]> = "materials",
    out_color: gfx::RenderTarget<Srgba8> = "OutColor",
});

const RASTERIZER: gfx::state::Rasterizer = gfx::state::Rasterizer {
    front_face: gfx::state::FrontFace::Clockwise,
    cull_face: gfx::state::CullFace::Nothing,
    method: gfx::state::RasterMethod::Fill,
    offset: None,
    samples: None,
};

pub struct Splat<R: gfx::Resources> {
    splat_pso: gfx::PipelineState<R, splat_pipe::Meta>,
    splat_pipeline_data: splat_pipe::Data<R>,
    splat_shader: rshader::Shader<R>,

    color_pso: gfx::PipelineState<R, color_pipe::Meta>,
    color_pipeline_data: color_pipe::Data<R>,
    color_shader: rshader::Shader<R>,

    _splatmap: Texture<R, R8>,
    _colormap: Texture<R, R8_G8_B8_A8>,

    splatmap_view: ShaderResourceView<R, u32>,
    colormap_view: ShaderResourceView<R, [f32; 4]>,
}

impl<R: gfx::Resources> Splat<R> {
    fn create_splat_pso<F: gfx::Factory<R>>(
        factory: &mut F,
        shaders: &gfx::ShaderSet<R>,
    ) -> gfx::PipelineState<R, splat_pipe::Meta> {
        factory
            .create_pipeline_state(shaders, TriangleList, RASTERIZER, splat_pipe::new())
            .unwrap()
    }
    fn create_color_pso<F: gfx::Factory<R>>(
        factory: &mut F,
        shaders: &gfx::ShaderSet<R>,
    ) -> gfx::PipelineState<R, color_pipe::Meta> {
        factory
            .create_pipeline_state(shaders, TriangleList, RASTERIZER, color_pipe::new())
            .unwrap()
    }

    fn update<C: Buffer<R>>(&mut self, encoder: &mut gfx::Encoder<R, C>) {
        let slice = gfx::Slice {
            start: 0,
            end: 6,
            base_vertex: 0,
            instances: None,
            buffer: gfx::IndexBuffer::Auto,
        };
        encoder.draw(&slice, &self.splat_pso, &self.splat_pipeline_data);
        encoder.draw(&slice, &self.color_pso, &self.color_pipeline_data);
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
        let splat_shader = rshader::Shader::simple(
            factory,
            shaders_watcher,
            shader_source!("../shaders/glsl", "version", "fullscreen.glslv"),
            shader_source!("../shaders/glsl", "version", "fractal", "splatmap.glslf"),
        ).unwrap();
        let color_shader = rshader::Shader::simple(
            factory,
            shaders_watcher,
            shader_source!("../shaders/glsl", "version", "fullscreen.glslv"),
            shader_source!("../shaders/glsl", "version", "fractal", "colormap.glslf"),
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

        let sampler_clamp = factory.create_sampler(SamplerInfo::new(Bilinear, Clamp));
        let sampler_wrap = factory.create_sampler(SamplerInfo::new(Bilinear, Tile));

        let mut splat = Self {
            splat_pso: Self::create_splat_pso(factory, splat_shader.as_shader_set()),
            color_pso: Self::create_color_pso(factory, color_shader.as_shader_set()),
            splat_pipeline_data: splat_pipe::Data::<R> {
                heights: (heights.clone(), sampler_clamp.clone()),
                slopes: (slopes.clone(), sampler_clamp.clone()),
                noise: (noise.clone(), sampler_wrap.clone()),
                noise_wavelength,
                texture_spacing,
                texture_resolution: resolution as i32,
                heights_spacing,
                out_splat: factory
                    .view_texture_as_render_target(&splatmap, 0, None)
                    .unwrap(),
            },
            color_pipeline_data: color_pipe::Data::<R> {
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
            },
            splat_shader,
            color_shader,
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
        let mut refreshed = false;
        if self.splat_shader.refresh(factory, shaders_watcher) {
            self.splat_pso = Self::create_splat_pso(factory, self.splat_shader.as_shader_set());
            self.update(encoder);
            refreshed = true;
        }

        if self.color_shader.refresh(factory, shaders_watcher) {
            self.color_pso = Self::create_color_pso(factory, self.color_shader.as_shader_set());
            self.update(encoder);
            refreshed = true;
        }

        refreshed
    }

    pub fn splatmap_shader_resource_view(&self) -> ShaderResourceView<R, u32> {
        self.splatmap_view.clone()
    }
    pub fn colormap_shader_resource_view(&self) -> ShaderResourceView<R, [f32; 4]> {
        self.colormap_view.clone()
    }
}
