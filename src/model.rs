use cache::{AssetLoadContext, GeneratedAsset, WebAsset};
use anyhow::Error;
use image::{self, PNG};
use lightbox::{Lightbox, Model};
use obj::{Mtl, Obj};
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::sync::{Arc, Mutex};
use zip::ZipArchive;

pub struct ModelFiles {
    obj_files: HashMap<String, Vec<u8>>,
    mtl_files: HashMap<String, Vec<u8>>,
    png_files: HashMap<String, Vec<u8>>,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum TreeType {
    Beech,
    Birch,
    Pine,
}

impl WebAsset for TreeType {
    type Type = ModelFiles;

    fn url(&self) -> String {
        match *self {
            TreeType::Beech => {
                "https://github.com/fintelia/terra/blob/assets/obj/beech.zip?raw=true"
            }
            TreeType::Birch => {
                "https://github.com/fintelia/terra/blob/assets/obj/birch.zip?raw=true"
            }
            TreeType::Pine => "https://github.com/fintelia/terra/blob/assets/obj/pine.zip?raw=true",
        }.to_owned()
    }

    fn filename(&self) -> String {
        match *self {
            TreeType::Beech => "models/beech.zip",
            TreeType::Birch => "models/birch.zip",
            TreeType::Pine => "models/pine.zip",
        }.to_owned()
    }

    fn parse(&self, _context: &mut AssetLoadContext, data: Vec<u8>) -> Result<Self::Type, Error> {
        let mut zip = ZipArchive::new(Cursor::new(data))?;

        let mut obj_files = HashMap::new();
        let mut mtl_files = HashMap::new();
        let mut png_files = HashMap::new();

        for i in 0..zip.len() {
            let mut file = zip.by_index(i)?;
            let file_name = file.name().to_owned();
            let mut contents = Vec::new();
            file.read_to_end(&mut contents)?;

            if file_name.ends_with(".obj") {
                obj_files.insert(file_name, contents);
            } else if file_name.ends_with(".mtl") {
                mtl_files.insert(file_name, contents);
            } else if file_name.ends_with(".png") {
                png_files.insert(file_name, contents);
            }
        }

        Ok(ModelFiles {
            obj_files,
            mtl_files,
            png_files,
        })
    }
}

pub struct TreeBillboardDef<'a, R, F, C, D: 'a>
where
    R: ::gfx::Resources,
    F: ::gfx::Factory<R> + 'static + Clone,
    C: ::gfx_core::command::Buffer<R>,
    D: ::gfx::Device<Resources = R, CommandBuffer = C>,
{
    pub ty: TreeType,
    pub lightbox: Arc<Mutex<Lightbox<R, F>>>,
    pub encoder: Arc<Mutex<&'a mut ::gfx::Encoder<R, C>>>,
    pub device: Arc<Mutex<&'a mut D>>,
}
impl<'a, R, F, C, D> GeneratedAsset for TreeBillboardDef<'a, R, F, C, D>
where
    R: ::gfx::Resources,
    F: ::gfx::Factory<R> + 'static + Clone,
    C: ::gfx_core::command::Buffer<R>,
    D: ::gfx::Device<Resources = R, CommandBuffer = C>,
{
    type Type = Vec<(u32, u32, Vec<u8>)>;

    fn filename(&self) -> String {
        match self.ty {
            TreeType::Beech => "models/beech.billboards",
            TreeType::Birch => "models/birch.billboards",
            TreeType::Pine => "models/pine.billboards",
        }.to_owned()
    }
    fn generate(&self, context: &mut AssetLoadContext) -> Result<Self::Type, Error> {
        let model_files = self.ty.load(context)?;

        let mut lightbox = self.lightbox.lock().unwrap();
        let mut encoder = self.encoder.lock().unwrap();
        let mut device = self.device.lock().unwrap();

        let encoder: &mut ::gfx::Encoder<R, C> = &mut *encoder;
        let device: &mut D = &mut *device;

        let textures: HashMap<String, Arc<image::RgbaImage>> = model_files
            .png_files
            .into_iter()
            .filter_map(|(name, contents)| {
                Some((
                    name.replace(".png", ""),
                    Arc::new(
                        image::load_from_memory_with_format(&contents[..], PNG)
                            .ok()?
                            .to_rgba(),
                    ),
                ))
            }).collect();

        let mut billboards = Vec::new();
        for (_filename, contents) in model_files.obj_files {
            let model = Obj::load_buf(&mut &contents[..])?;

            let mut materials = Vec::new();

            for mtl_filename in model.material_libs.iter() {
                let mtl_contents = match model_files.mtl_files.get(mtl_filename) {
                    Some(data) => data,
                    None => bail!("missing mtl file {}", mtl_filename),
                };

                let mut mtl = Mtl::load(&mut &mtl_contents[..]);
                materials.append(&mut mtl.materials);
            }

            let model = Model::from_obj(
                &mut lightbox,
                model,
                Some(Mtl { materials }),
                textures.clone(),
            )?;

            for image in lightbox.capture_billboards(&model, encoder, device)? {
                //image.save(&format!("{}.png", filename))?;
                let (width, height) = image.dimensions();
                billboards.push((width, height, image.into_raw()));
            }
        }

        Ok(billboards)
    }
}
