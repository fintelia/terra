use gfx;
use gfx_core;
use gfx::format::*;

use std::path::Path;

use image;
use image::GenericImage;

pub struct MaterialSet<R: gfx::Resources> {
    pub(crate) texture_view: gfx_core::handle::ShaderResourceView<R, [f32; 4]>,
    pub(crate) _texture: gfx_core::handle::Texture<R, gfx_core::format::R8_G8_B8_A8>,
}

impl<R: gfx::Resources> MaterialSet<R> {
    pub fn load<F: gfx::Factory<R>, C: gfx_core::command::Buffer<R>>(
        path: &Path,
        factory: &mut F,
        encoder: &mut gfx::Encoder<R, C>,
    ) -> Self {
        let image_files = vec![
            "limestone-rock/limestone-rock-albedo.png",
            "grass1/grass1-albedo3.png",
        ];

        let resolution = 2048;
        let mipmaps = 12;

        let texture = factory
            .create_texture::<R8_G8_B8_A8>(
                gfx::texture::Kind::D2Array(
                    resolution,
                    resolution,
                    image_files.len() as u16,
                    gfx::texture::AaMode::Single,
                ),
                mipmaps,
                gfx::SHADER_RESOURCE,
                gfx::memory::Usage::Dynamic,
                Some(ChannelType::Srgb),
            )
            .unwrap();

        for (layer, file) in image_files.iter().enumerate() {
            let mut img = image::open(path.join(file)).unwrap();

            assert_eq!(img.width(), img.height());

            for level in 0..mipmaps {
                img = img.resize(
                    (resolution >> level) as u32,
                    (resolution >> level) as u32,
                    image::FilterType::Triangle,
                );

                if level >= 4 {
                    img = img.blur(2.0);
                }

                let image_data: Vec<[u8; 4]> = img.to_rgba().to_vec()[..]
                    .chunks(4)
                    .map(|c| [c[0], c[1], c[2], c[3]])
                    .collect();

                encoder
                    .update_texture::<R8_G8_B8_A8, gfx::format::Srgba8>(
                        &texture,
                        None,
                        gfx_core::texture::NewImageInfo {
                            xoffset: 0,
                            yoffset: 0,
                            zoffset: layer as u16,
                            width: resolution >> level,
                            height: resolution >> level,
                            depth: 1,
                            format: (),
                            mipmap: level,
                        },
                        &image_data[..],
                    )
                    .unwrap();
            }
        }

        let texture_view = factory
            .view_texture_as_shader_resource::<gfx::format::Srgba8>(
                &texture,
                (0, 12),
                Swizzle::new(),
            )
            .unwrap();

        Self {
            texture_view,
            _texture: texture,
        }
    }
}
