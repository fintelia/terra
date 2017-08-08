use gfx;
use gfx_core;
use gfx::format::*;

use std::path::Path;
use std::cmp;

use image;

pub struct MaterialSet<R: gfx::Resources> {
    pub(crate) view: gfx_core::handle::ShaderResourceView<R, [f32; 4]>,
    pub(crate) _texture: gfx_core::handle::Texture<R, gfx_core::format::R8_G8_B8_A8>,
}

impl<R: gfx::Resources> MaterialSet<R> {
    pub fn load<F: gfx::Factory<R>, C: gfx_core::command::Buffer<R>>(
        path: &Path,
        factory: &mut F,
        _encoder: &mut gfx::Encoder<R, C>,
    ) -> Self {
        let img = image::open(path.join("Just Add Bison.jpg")).unwrap();
        let img = img.to_rgba();

        let mut image_data = vec![img.to_vec()];

        let (width, height) = img.dimensions();
        let (mut width, mut height) = (width as usize, height as usize);

        while width > 1 || height > 1 {
            let nwidth = cmp::max(width >> 1, 1);
            let nheight = cmp::max(height >> 1, 1);

            let mut data = Vec::with_capacity(width * height * 4);
            for y in (0..height).step_by(2) {
                for x in (0..width).step_by(2) {
                    for t in 0..4 {
                        if image_data.len() < 40 {
                            let parent = &image_data[image_data.len() - 1];
                            let mut denominator = 1;
                            let mut sum = parent[(x + y * width) * 4 + t] as u32;
                            if x + 1 < width {
                                sum += parent[((x + 1) + y * width) * 4 + t] as u32;
                                denominator += 1;
                            }
                            if y + 1 < height {
                                sum += parent[(x + (y + 1) * width) * 4 + t] as u32;
                                denominator += 1;
                            }
                            if x + 1 < width && y + 1 < height {
                                sum += parent[((x + 1) + (y + 1) * width) * 4 + t] as u32;
                                denominator += 1;
                            }

                            data.push((sum / denominator) as u8);
                        } else {
                            data.push(0);
                        }
                    }
                }
            }
            image_data.push(data);

            width = nwidth;
            height = nheight;
        }

        let v: Vec<&[u8]> = image_data.iter().map(|d| &d[..]).collect();

        let texture = factory
            .create_texture_immutable_u8::<(R8_G8_B8_A8, Unorm)>(
                gfx::texture::Kind::D2Array(
                    img.dimensions().0 as u16,
                    img.dimensions().1 as u16,
                    1,
                    gfx::texture::AaMode::Single,
                ),
                &v[..],
            )
            .unwrap();
        // encoder.generate_mipmap(&texture.1);

        Self {
            view: texture.1,
            _texture: texture.0,
        }
    }
}
