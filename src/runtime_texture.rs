use gfx;
use gfx_core::{self, format, handle};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum TextureFormat {
    R8,
    F32,
    RGBA8,
    SRGBA,
}
impl TextureFormat {
    pub fn bytes_per_texel(&self) -> usize {
        match *self {
            TextureFormat::R8 => 1,
            TextureFormat::F32 => 4,
            TextureFormat::RGBA8 => 4,
            TextureFormat::SRGBA => 4,
        }
    }
}

pub(crate) enum TextureArray<R: gfx::Resources> {
    R8 {
        texture: handle::Texture<R, format::R8>,
        view: handle::ShaderResourceView<R, f32>,
    },
    F32 {
        texture: handle::Texture<R, format::R32>,
        view: handle::ShaderResourceView<R, f32>,
    },
    RGBA8 {
        texture: handle::Texture<R, format::R8_G8_B8_A8>,
        view: handle::ShaderResourceView<R, [f32; 4]>,
    },
    SRGBA {
        texture: handle::Texture<R, format::R8_G8_B8_A8>,
        view: handle::ShaderResourceView<R, [f32; 4]>,
    },
}
impl<R: gfx::Resources> TextureArray<R> {
    pub fn new<F: gfx::Factory<R>>(
        format: TextureFormat,
        resolution: u16,
        depth: u16,
        factory: &mut F,
    ) -> Self {
        match format {
            TextureFormat::R8 => {
                let texture = factory
                    .create_texture::<gfx::format::R8>(
                        gfx::texture::Kind::D2Array(
                            resolution,
                            resolution,
                            depth,
                            gfx::texture::AaMode::Single,
                        ),
                        1,
                        gfx::memory::Bind::SHADER_RESOURCE,
                        gfx::memory::Usage::Dynamic,
                        Some(gfx::format::ChannelType::Unorm),
                    ).unwrap();

                let view = factory
                    .view_texture_as_shader_resource::<(gfx::format::R8, gfx::format::Unorm)>(
                        &texture,
                        (0, 0),
                        gfx::format::Swizzle::new(),
                    ).unwrap();

                TextureArray::R8 { texture, view }
            }
            TextureFormat::F32 => {
                let texture = factory
                    .create_texture::<gfx::format::R32>(
                        gfx::texture::Kind::D2Array(
                            resolution,
                            resolution,
                            depth,
                            gfx::texture::AaMode::Single,
                        ),
                        1,
                        gfx::memory::Bind::SHADER_RESOURCE,
                        gfx::memory::Usage::Dynamic,
                        Some(gfx::format::ChannelType::Float),
                    ).unwrap();

                let view = factory
                    .view_texture_as_shader_resource::<f32>(
                        &texture,
                        (0, 0),
                        gfx::format::Swizzle::new(),
                    ).unwrap();

                TextureArray::F32 { texture, view }
            }
            TextureFormat::RGBA8 => {
                let texture = factory
                    .create_texture::<gfx::format::R8_G8_B8_A8>(
                        gfx::texture::Kind::D2Array(
                            resolution,
                            resolution,
                            depth,
                            gfx::texture::AaMode::Single,
                        ),
                        1,
                        gfx::memory::Bind::SHADER_RESOURCE,
                        gfx::memory::Usage::Dynamic,
                        Some(gfx::format::ChannelType::Unorm),
                    ).unwrap();

                let view = factory
                    .view_texture_as_shader_resource::<gfx::format::Rgba8>(
                        &texture,
                        (0, 0),
                        gfx::format::Swizzle::new(),
                    ).unwrap();

                TextureArray::RGBA8 { texture, view }
            }
            TextureFormat::SRGBA => {
                let texture = factory
                    .create_texture::<gfx::format::R8_G8_B8_A8>(
                        gfx::texture::Kind::D2Array(
                            resolution,
                            resolution,
                            depth,
                            gfx::texture::AaMode::Single,
                        ),
                        1,
                        gfx::memory::Bind::SHADER_RESOURCE,
                        gfx::memory::Usage::Dynamic,
                        Some(gfx::format::ChannelType::Srgb),
                    ).unwrap();

                let view = factory
                    .view_texture_as_shader_resource::<gfx::format::Srgba8>(
                        &texture,
                        (0, 0),
                        gfx::format::Swizzle::new(),
                    ).unwrap();

                TextureArray::SRGBA { texture, view }
            }
        }
    }

    pub fn update_layer<C: gfx_core::command::Buffer<R>>(
        &self,
        layer: u16,
        resolution: u16,
        data: &[u8],
        encoder: &mut gfx::Encoder<R, C>,
    ) {
        let new_image_info = gfx_core::texture::NewImageInfo {
            xoffset: 0,
            yoffset: 0,
            zoffset: layer,
            width: resolution,
            height: resolution,
            depth: 1,
            format: (),
            mipmap: 0,
        };

        match *self {
            TextureArray::R8 { ref texture, .. } => {
                encoder
                    .update_texture::<gfx::format::R8, u8>(
                        texture,
                        None,
                        new_image_info,
                        gfx::memory::cast_slice(data),
                    ).unwrap();
            }
            TextureArray::F32 { ref texture, .. } => {
                encoder
                    .update_texture::<gfx::format::R32, f32>(
                        texture,
                        None,
                        new_image_info,
                        gfx::memory::cast_slice(data),
                    ).unwrap();
            }
            TextureArray::RGBA8 { ref texture, .. } => {
                encoder
                    .update_texture::<format::R8_G8_B8_A8, gfx::format::Rgba8>(
                        texture,
                        None,
                        new_image_info,
                        gfx::memory::cast_slice(data),
                    ).unwrap();
            }
            TextureArray::SRGBA { ref texture, .. } => {
                encoder
                    .update_texture::<format::R8_G8_B8_A8, gfx::format::Srgba8>(
                        texture,
                        None,
                        new_image_info,
                        gfx::memory::cast_slice(data),
                    ).unwrap();
            }
        }
    }
}
