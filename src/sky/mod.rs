use std::collections::HashMap;
use std::error::Error;
use std::io::{Cursor, Read};

use gfx;
use gfx::format::*;
use gfx::texture::Kind;
use gfx_core;

use image::{self, RgbaImage};
use zip::ZipArchive;

use cache::WebAsset;

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
    fn parse(&self, data: Vec<u8>) -> Result<Self::Type, Box<Error>> {
        let mut zip = ZipArchive::new(Cursor::new(data))?;
        let mut faces = HashMap::new();

        for i in 0..zip.len() {
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
    pub fn new<F: gfx::Factory<R>>(factory: &mut F) -> Self {
        let mut raw = SkyboxAsset::default().load().unwrap();
        let mut data = Vec::new();
        let mut data_slices = Vec::new();
        for face in FACE_NAMES.iter() {
            data.push(raw.faces.remove(face).unwrap().into_raw());
        }
        for face in data.iter() {
            data_slices.push(gfx::memory::cast_slice(&face[..]));
        }

        let (texture, texture_view) = factory
            .create_texture_immutable::<(R8_G8_B8_A8, gfx_core::format::Unorm)>(
                Kind::Cube(raw.resolution),
                &data_slices[..],
            )
            .unwrap();

        Self {
            texture_view,
            _texture: texture,
        }
    }
}
