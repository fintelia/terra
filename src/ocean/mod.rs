use std::f32::consts::PI;

use cgmath::*;
use gfx;
use gfx::format::*;
use gfx_core;
use rand;
use rand::distributions::{IndependentSample, Normal};
use rustfft::FFT;
use rustfft::algorithm::Radix4;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

// See: https://github.com/deiss/fftocean

const RESOLUTION: usize = 128;
const MIPMAPS: u8 = 8;

pub struct Ocean<R: gfx::Resources> {
    side_length: f32,
    time: f32,

    texture_data: Vec<Vec<[u8; 4]>>,

    /// Initial wave spectrum.
    spectrum: Vec<Complex<f32>>,
    /// Current wave heights.
    heights: Vec<f32>,

    pub(crate) texture_view: gfx_core::handle::ShaderResourceView<R, [f32; 4]>,
    texture: gfx_core::handle::Texture<R, gfx_core::format::R8_G8_B8_A8>,
}

impl<R: gfx::Resources> Ocean<R> {
    pub fn new<F: gfx::Factory<R>>(factory: &mut F) -> Self {
        let texture = factory
            .create_texture::<R8_G8_B8_A8>(
                gfx::texture::Kind::D2Array(
                    RESOLUTION as u16,
                    RESOLUTION as u16,
                    1,
                    gfx::texture::AaMode::Single,
                ),
                MIPMAPS,
                gfx::SHADER_RESOURCE,
                gfx::memory::Usage::Dynamic,
                Some(ChannelType::Unorm),
            )
            .unwrap();

        let texture_view = factory
            .view_texture_as_shader_resource::<gfx::format::Rgba8>(&texture, (0, 0), Swizzle::new())
            .unwrap();

        let side_length = 500.0;

        // Parameters for ocean simulation; currently all hardcoded.
        let g = 9.81;
        let wind_speed = 30.0;
        let min_wave_size = 1.0;
        let l = (wind_speed * wind_speed) / g;
        let (wx, wy) = (1.0, 0.0);

        // Initialize the wave spectrum, recalling that the DC terms will be zero.
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0);
        let mut spectrum = vec![Zero::zero(); (RESOLUTION + 1) * (RESOLUTION + 1)];
        for y in 0..(RESOLUTION + 1) {
            for x in 1..(RESOLUTION + 1) {
                spectrum[x + y * (RESOLUTION + 1)] = {
                    let kx = (2.0 * PI * x as f32 - PI * RESOLUTION as f32) / side_length;
                    let ky = (2.0 * PI * y as f32 - PI * RESOLUTION as f32) / side_length;
                    let k2 = kx * kx + ky * ky;
                    if (x as i64 - RESOLUTION as i64 / 2).abs() < RESOLUTION as i64 / 8
                        || (y as i64 - RESOLUTION as i64 / 2).abs() < RESOLUTION as i64 / 8
                    {
                        Zero::zero()
                    } else {
                        let power = 1.0 * f32::exp(-1.0 / (k2 * l * l))
                            * f32::exp(-k2 * min_wave_size * min_wave_size)
                            * f32::powi(kx * wx + ky * wy, 2)
                            / (k2 * k2);
                        Complex::new(
                            normal.ind_sample(&mut rng) as f32,
                            normal.ind_sample(&mut rng) as f32,
                        ) * (power / 2.0).sqrt()
                    }
                };
            }
        }

        let mut texture_data = Vec::new();
        for i in 0..MIPMAPS {
            texture_data.push(vec![[0; 4]; (RESOLUTION >> i) * (RESOLUTION >> i)]);
        }
        Self {
            spectrum,
            side_length,
            texture_data,
            heights: vec![0.0; RESOLUTION * RESOLUTION],
            texture_view,
            texture,
            time: 0.0,
        }
    }

    fn update_texture_data(&mut self) {
        let heights = &self.heights;
        let get = |x, y| {
            let x = ((x % RESOLUTION as i64) + RESOLUTION as i64) % RESOLUTION as i64;
            let y = ((y % RESOLUTION as i64) + RESOLUTION as i64) % RESOLUTION as i64;
            heights[x as usize + y as usize * RESOLUTION]
        };

        let min = heights.iter().cloned().fold(0. / 0., f32::min);
        let max = heights.iter().cloned().fold(0. / 0., f32::max);

        for i in 0..MIPMAPS {
            let resolution = RESOLUTION >> i;

            for y in 0..(resolution as i64) {
                for x in 0..(resolution as i64) {
                    self.texture_data[i as usize][x as usize + y as usize * resolution] = if i == 0
                    {
                        let sx = get(x + 1, y) - get(x - 1, y);
                        let sy = get(x, y + 1) - get(x, y - 1);
                        let n = Vector3::new(sx, 2000.0, sy).normalize();

                        [
                            (n.x * 127.5 + 127.5) as u8,
                            (n.z * 127.5 + 127.5) as u8,
                            (n.y * 127.5 + 127.5) as u8,
                            ((get(x, y) - min) / (max - min) * 255.0) as u8,
                        ]
                    } else if i < 3 {
                        let (px, py, presolution) =
                            (x as usize * 2, y as usize * 2, resolution * 2);
                        let p = [
                            self.texture_data[i as usize - 1][px + py * presolution],
                            self.texture_data[i as usize - 1][(px + 1) + py * presolution],
                            self.texture_data[i as usize - 1][px + (py + 1) * presolution],
                            self.texture_data[i as usize - 1][(px + 1) + (py + 1) * presolution],
                        ];

                        let mut sum = [0.0; 4];
                        for j in 0..4 {
                            for k in 0..4 {
                                sum[j] += p[k][j] as f32;
                            }
                        }

                        [
                            ((sum[0] / 4.0 - 127.5) / i as f32 + 127.0) as u8,
                            ((sum[1] / 4.0 - 127.5) / i as f32 + 127.0) as u8,
                            ((sum[2] / 4.0 - 127.5) / i as f32 + 255.0 - 127.5 / i as f32) as u8,
                            ((sum[3] / 4.0) / i as f32) as u8,
                        ]
                    } else {
                        [128, 128, 255, 0]
                    };
                }
            }
        }
    }

    fn update_simulation(&mut self, dt: f32) {
        self.time += dt;

        let mut input: Vec<Complex<f32>> = vec![Zero::zero(); RESOLUTION * RESOLUTION];
        let mut output: Vec<Complex<f32>> = vec![Zero::zero(); RESOLUTION * RESOLUTION];
        let mut input2: Vec<Complex<f32>> = vec![Zero::zero(); RESOLUTION * RESOLUTION];
        let mut output2: Vec<Complex<f32>> = vec![Zero::zero(); RESOLUTION * RESOLUTION];

        for y in 0..RESOLUTION {
            for x in 0..RESOLUTION {
                let x = if x == 0 {
                    RESOLUTION / 2
                } else if x <= RESOLUTION / 2 {
                    RESOLUTION / 2 - x
                } else {
                    RESOLUTION * 3 / 2 - x
                };
                let y = if y == 0 {
                    RESOLUTION / 2
                } else if y <= RESOLUTION / 2 {
                    RESOLUTION / 2 - y
                } else {
                    RESOLUTION * 3 / 2 - y
                };

                let exponent = {
                    let x = x as f32 - (RESOLUTION / 2) as f32;
                    let y = y as f32 - (RESOLUTION / 2) as f32;

                    let kx = (2.0 * PI * x as f32 - PI * RESOLUTION as f32) / self.side_length;
                    let ky = (2.0 * PI * y as f32 - PI * RESOLUTION as f32) / self.side_length;
                    let k2 = kx * kx + ky * ky;
                    Complex::new(
                        0.0,
                        self.time * f32::sqrt(9.81 * k2.sqrt() * (1.0 + k2 * 0.01)),
                    )
                };

                let h0_k = self.spectrum[x + y * (RESOLUTION + 1)];
                let h0_nk =
                    self.spectrum[(RESOLUTION - x) + (RESOLUTION - y) * (RESOLUTION + 1)].conj();

                input[x + y * RESOLUTION] =
                    h0_k * Complex::exp(&exponent) + h0_nk * Complex::exp(&-exponent);
            }
        }

        let fft = Radix4::new(RESOLUTION, true);
        fft.process_multi(&mut input, &mut output);
        for x in 0..RESOLUTION {
            for y in 0..RESOLUTION {
                input2[x + y * RESOLUTION] = output[y + x * RESOLUTION];
            }
        }
        fft.process_multi(&mut input2, &mut output2);

        for y in 0..RESOLUTION {
            for x in 0..RESOLUTION {
                self.heights[x + y * RESOLUTION] = output2[x + y * RESOLUTION].re;
            }
        }
    }

    pub fn update<C: gfx_core::command::Buffer<R>>(
        &mut self,
        encoder: &mut gfx::Encoder<R, C>,
        dt: f32,
    ) {
        self.update_simulation(dt);
        self.update_texture_data();

        for (i, ref mipmap_data) in self.texture_data.iter().enumerate() {
            encoder
                .update_texture::<R8_G8_B8_A8, gfx::format::Srgba8>(
                    &self.texture,
                    None,
                    gfx_core::texture::NewImageInfo {
                        xoffset: 0,
                        yoffset: 0,
                        zoffset: 0,
                        width: (RESOLUTION as u16) >> i,
                        height: (RESOLUTION as u16) >> i,
                        depth: 1,
                        format: (),
                        mipmap: i as u8,
                    },
                    &mipmap_data[..],
                )
                .unwrap();
        }
    }
}
