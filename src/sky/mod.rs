use std::collections::HashMap;
use std::io::{Cursor, Read};

use failure::Error;
use gfx;
use gfx::format::*;
use gfx::texture::Kind;
use gfx_core;
use image::{self, RgbaImage};
use zip::ZipArchive;

use cache::{AssetLoadContext, WebAsset};

const FACE_NAMES: [&'static str; 6] = ["east", "west", "up", "down", "north", "south"];

pub struct SkyboxAsset(String);
impl Default for SkyboxAsset {
    fn default() -> Self {
        SkyboxAsset("clouds1.zip".to_owned())
    }
}
impl WebAsset for SkyboxAsset {
    type Type = SkyboxRaw;

    fn url(&self) -> String {
        format!("https://opengameart.org/sites/default/files/{}", self.0)
    }
    fn filename(&self) -> String {
        format!("sky/{}", self.0)
    }
    fn parse(&self, context: &mut AssetLoadContext, data: Vec<u8>) -> Result<Self::Type, Error> {
        let mut zip = ZipArchive::new(Cursor::new(data))?;
        let mut faces = HashMap::new();

        for i in 0..zip.len() {
            context.set_progress(100 * i / zip.len());
            let mut file = zip.by_index(i)?;
            for face in FACE_NAMES.iter() {
                if file.name().contains(face) {
                    let mut image_data = Vec::new();
                    file.read_to_end(&mut image_data)?;
                    faces.insert(
                        face.to_owned(),
                        image::load_from_memory(&image_data)?.to_rgba(),
                    );
                    break;
                }
            }
        }
        context.set_progress(100);

        assert_eq!(faces.len(), 6);
        let width = faces.iter().next().unwrap().1.width();
        let height = faces.iter().next().unwrap().1.height();
        assert_eq!(width, height);
        for img in faces.values() {
            assert_eq!(img.width(), width);
            assert_eq!(img.height(), height);
        }

        Ok(SkyboxRaw {
            resolution: width as u16,
            faces,
        })
    }
}
pub struct SkyboxRaw {
    resolution: u16,
    faces: HashMap<&'static str, RgbaImage>,
}

pub struct Skybox<R: gfx::Resources> {
    pub(crate) texture_view: gfx_core::handle::ShaderResourceView<R, [f32; 4]>,
    pub(crate) _texture: gfx_core::handle::Texture<R, gfx_core::format::R8_G8_B8_A8>,
}
impl<R: gfx::Resources> Skybox<R> {
    pub fn new<F: gfx::Factory<R>, C: gfx_core::command::Buffer<R>>(
        factory: &mut F,
        encoder: &mut gfx::Encoder<R, C>,
    ) -> Self {
        let mut raw = SkyboxAsset::default()
            .load(&mut AssetLoadContext::new())
            .unwrap();
        let mut data = Vec::new();
        let mut data_slices = Vec::new();
        for face in FACE_NAMES.iter() {
            data.push(raw.faces.remove(face).unwrap().into_raw());
        }
        for face in data.iter() {
            data_slices.push(gfx::memory::cast_slice(&face[..]));
        }

        let mipmaps = 1 + (raw.resolution.next_power_of_two() as f32).log2() as u8;
        let texture = factory
            .create_texture::<R8_G8_B8_A8>(
                Kind::Cube(raw.resolution),
                mipmaps,
                gfx::memory::Bind::SHADER_RESOURCE,
                gfx::memory::Usage::Dynamic,
                Some(ChannelType::Srgb),
            )
            .unwrap();

        for (i, face) in gfx::texture::CUBE_FACES.iter().cloned().enumerate() {
            encoder
                .update_texture::<R8_G8_B8_A8, gfx::format::Srgba8>(
                    &texture,
                    Some(face),
                    gfx_core::texture::NewImageInfo {
                        xoffset: 0,
                        yoffset: 0,
                        zoffset: 0,
                        width: raw.resolution,
                        height: raw.resolution,
                        depth: 1,
                        format: (),
                        mipmap: 0,
                    },
                    &data_slices[i][..],
                )
                .unwrap();
        }

        let texture_view = factory
            .view_texture_as_shader_resource::<gfx::format::Srgba8>(
                &texture,
                (0, mipmaps),
                Swizzle::new(),
            )
            .unwrap();

        encoder.generate_mipmap(&texture_view);

        Self {
            texture_view,
            _texture: texture,
        }
    }
}
