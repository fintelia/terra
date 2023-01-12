use crate::asset::TERRA_DIRECTORY;
use crate::cache::TextureFormat;
use anyhow::Error;
use atomicwrites::{AtomicFile, OverwriteBehavior};
use basis_universal::{TranscodeParameters, Transcoder, TranscoderTextureFormat};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::{fs, num::NonZeroU32};
use types::VNode;

const MAX_STREAMED_LEVEL: u8 = VNode::LEVEL_CELL_76M;

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum TileState {
    Base,
    GpuOnly,
    MissingBase,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(crate) struct TextureDescriptor {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: TextureFormat,
    #[serde(default)]
    pub array_texture: bool,
}

pub(crate) struct MapFile {
    _db: sled::Db,
    textures: sled::Tree,

    server: String,
    remote_tiles: Arc<Mutex<HashSet<VNode>>>,
    local_tiles: Arc<Mutex<HashSet<VNode>>>,
}
impl MapFile {
    pub(crate) async fn new(server: String) -> Result<Self, Error> {
        let directory = TERRA_DIRECTORY.join("tiles/meta");
        let db = sled::open(&directory).expect(&format!(
            "Failed to open/create sled database. Deleting the '{}' directory may fix this",
            directory.display()
        ));

        const CURRENT_VERSION: i32 = 2;
        let version = db.get("version").unwrap();
        let version = version
            .as_ref()
            .map(|v| std::str::from_utf8(v).unwrap_or("0"))
            .map(|s| s.parse())
            .unwrap_or(Ok(CURRENT_VERSION))
            .unwrap();
        if version < CURRENT_VERSION {
            db.drop_tree("tiles").unwrap();
            db.drop_tree("textures").unwrap();
        }
        db.insert("version", &*format!("{}", CURRENT_VERSION)).unwrap();

        let mapfile = Self {
            textures: db.open_tree("textures").unwrap(),
            _db: db,
            server,
            remote_tiles: Default::default(),
            local_tiles: Default::default(),
        };
        mapfile.reload_tile_states().await?;
        Ok(mapfile)
    }

    pub(crate) async fn read_tile(&self, node: VNode) -> Result<Option<Vec<u8>>, Error> {
        let filename = Self::tile_path(node);
        if filename.exists() {
            Ok(Some(tokio::fs::read(&filename).await?))
        } else {
            if !self.remote_tiles.lock().unwrap().contains(&node) {
                return Ok(None);
            }
            let contents = self.download(&format!("tiles/{}", Self::tile_name(node))).await?;
            if self.server.starts_with("http://") || self.server.starts_with("https://") {
                tokio::task::block_in_place(|| self.write_tile(node, &contents))?;
            }
            Ok(Some(contents))
        }
    }

    pub(crate) fn write_tile(&self, node: VNode, data: &[u8]) -> Result<(), Error> {
        let filename = Self::tile_path(node);
        if let Some(parent) = filename.parent() {
            fs::create_dir_all(parent)?;
        }

        AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
            .write(|f| f.write_all(data))?;

        self.local_tiles.lock().unwrap().insert(node);

        Ok(())
    }

    pub(crate) fn read_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: &str,
    ) -> Result<wgpu::Texture, Error> {
        let desc = self.lookup_texture(name)?.unwrap();

        let (width, height) = (desc.width as usize, desc.height as usize);
        assert_eq!(width % desc.format.block_size() as usize, 0);
        assert_eq!(height % desc.format.block_size() as usize, 0);
        let (width, height) =
            (width / desc.format.block_size() as usize, height / desc.format.block_size() as usize);

        let row_bytes = width * desc.format.bytes_per_block();

        let mut mip_level_count = 1;
        let mut data = if desc.format == TextureFormat::RGBA8 {
            image::open(TERRA_DIRECTORY.join(format!("{}.tiff", name)))?.to_rgba8().into_vec()
        } else if desc.format == TextureFormat::UASTC {
            let raw_data = fs::read(TERRA_DIRECTORY.join(format!("{}.basis", name)))?;
            let mut transcoder = Transcoder::new();
            transcoder.prepare_transcoding(&raw_data).unwrap();

            let mut data = Vec::new();
            'outer: for level in 0.. {
                if (width as u32 * desc.format.block_size()) >> level == 0
                    && (height as u32 * desc.format.block_size()) >> level == 0
                {
                    break;
                }

                for image in 0..desc.depth {
                    let transcoded = transcoder.transcode_image_level(
                        &raw_data,
                        if device.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
                            TranscoderTextureFormat::BC7_RGBA
                        } else {
                            TranscoderTextureFormat::ASTC_4x4_RGBA
                        },
                        TranscodeParameters {
                            image_index: image as u32,
                            level_index: level,
                            ..Default::default()
                        },
                    );

                    match transcoded {
                        Ok(bytes) => {
                            mip_level_count = level + 1;
                            data.extend_from_slice(&*bytes);
                        }
                        Err(_) => break 'outer,
                    }
                }
            }

            transcoder.end_transcoding();
            data
        } else {
            fs::read(TERRA_DIRECTORY.join(format!("{}.raw", name)))?
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: desc.width,
                height: desc.height,
                depth_or_array_layers: desc.depth,
            },
            format: desc.format.to_wgpu(device.features()),
            mip_level_count,
            sample_count: 1,
            dimension: if desc.depth == 1 || desc.array_texture {
                wgpu::TextureDimension::D2
            } else {
                wgpu::TextureDimension::D3
            },
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | if !desc.format.is_compressed() {
                    wgpu::TextureUsages::STORAGE_BINDING
                } else {
                    wgpu::TextureUsages::empty()
                },
            label: Some(&format!("texture.{}", name)),
        });

        if cfg!(feature = "small-trace") {
            let bytes_per_block = desc.format.bytes_per_block();
            for y in 0..(height * desc.depth as usize) {
                for x in 0..width {
                    if x % 16 == 0 && y % 16 == 0 {
                        continue;
                    }
                    let src = ((x & !15) + (y & !15) * width) * bytes_per_block;
                    let dst = (x + y * width) * bytes_per_block;
                    data.copy_within(src..src + bytes_per_block, dst);
                }
            }
        }

        let mut offset = row_bytes * height * desc.depth as usize;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &data[..offset],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(row_bytes as u32).unwrap()),
                rows_per_image: Some(NonZeroU32::new(height as u32).unwrap()),
            },
            wgpu::Extent3d {
                width: width as u32 * desc.format.block_size(),
                height: height as u32 * desc.format.block_size(),
                depth_or_array_layers: desc.depth,
            },
        );

        for mip in 1.. {
            let block_size = desc.format.block_size() as usize;
            let width = (width * block_size) >> mip;
            let height = (height * block_size) >> mip;
            let depth =
                if desc.array_texture { desc.depth as usize } else { desc.depth as usize >> mip };
            if width == 0 && height == 0 && depth == 0 {
                break;
            }

            let width = (width.max(1) - 1) / block_size + 1;
            let height = (height.max(1) - 1) / block_size + 1;
            let depth = depth.max(1);
            let bytes = width * height * depth * desc.format.bytes_per_block();
            if offset + bytes > data.len() {
                break;
            }

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &data[offset..],
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        NonZeroU32::new((width * desc.format.bytes_per_block()) as u32).unwrap(),
                    ),
                    rows_per_image: Some(NonZeroU32::new(height as u32).unwrap()),
                },
                wgpu::Extent3d {
                    width: width as u32 * desc.format.block_size(),
                    height: height as u32 * desc.format.block_size(),
                    depth_or_array_layers: depth as u32,
                },
            );
            offset += bytes;
        }

        Ok(texture)
    }

    pub(crate) fn write_texture(
        &self,
        name: &str,
        desc: TextureDescriptor,
        data: &[u8],
    ) -> Result<(), Error> {
        self.update_texture(name, desc)?;
        if desc.format == TextureFormat::RGBA8 {
            let filename = TERRA_DIRECTORY.join(format!("{}.tiff", name));
            let mut encoded = Vec::new();
            image::codecs::tiff::TiffEncoder::new(std::io::Cursor::new(&mut encoded)).encode(
                data,
                desc.width,
                desc.height * desc.depth,
                image::ColorType::Rgba8,
            )?;
            Ok(AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
                .write(|f| f.write_all(&encoded))?)
        } else if desc.format == TextureFormat::UASTC {
            let filename = TERRA_DIRECTORY.join(format!("{}.basis", name));
            Ok(AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
                .write(|f| f.write_all(data))?)
        } else {
            let filename = TERRA_DIRECTORY.join(format!("{}.raw", name));
            Ok(AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
                .write(|f| f.write_all(data))?)
        }
    }

    pub(crate) fn reload_texture(&self, name: &str) -> bool {
        let desc = self.lookup_texture(name);
        if let Ok(Some(desc)) = desc {
            if desc.format == TextureFormat::RGBA8 {
                TERRA_DIRECTORY.join(format!("{}.tiff", name)).exists()
            } else if desc.format == TextureFormat::UASTC {
                TERRA_DIRECTORY.join(format!("{}.basis", name)).exists()
            } else {
                TERRA_DIRECTORY.join(format!("{}.raw", name)).exists()
            }
        } else {
            false
        }
    }

    fn tile_name(node: VNode) -> String {
        let face = match node.face() {
            0 => "0E",
            1 => "180E",
            2 => "90E",
            3 => "90W",
            4 => "N",
            5 => "S",
            _ => unreachable!(),
        };
        format!("{}_{}_{}x{}.raw", node.level(), face, node.x(), node.y())
    }

    fn tile_path(node: VNode) -> PathBuf {
        TERRA_DIRECTORY.join("tiles").join(&Self::tile_name(node))
    }

    fn face_index(s: &str) -> Option<u8> {
        Some(match s {
            "0E" => 0,
            "180E" => 1,
            "90E" => 2,
            "90W" => 3,
            "N" => 4,
            "S" => 5,
            _ => return None,
        })
    }

    pub(crate) fn parse_tile_list(encoded: &[u8]) -> Result<HashSet<(u8, u8, u32, u32)>, Error> {
        let mut remote_files = String::new();
        lz4::Decoder::new(std::io::Cursor::new(&encoded))?.read_to_string(&mut remote_files)?;
        let mut all = HashSet::new();
        for filename in remote_files.split("\n") {
            if let Ok((level, face, x, y)) =
                sscanf::scanf!(filename, "{}_{}_{}x{}.raw", u8, String, u32, u32)
            {
                Self::face_index(&face).map(|face| all.insert((level, face, x, y)));
            }
        }
        Ok(all)
    }

    async fn reload_tile_states(&self) -> Result<(), Error> {
        let mut existing = HashSet::new();

        // Scan local files.
        let directory = TERRA_DIRECTORY.join("tiles");
        std::fs::create_dir_all(&directory)?;
        for file in fs::read_dir(directory)? {
            let filename = file?.file_name();
            let filename = filename.to_string_lossy();

            if let Ok((level, face, x, y)) =
                sscanf::scanf!(filename, "{}_{}_{}x{}.raw", u8, String, u32, u32)
            {
                Self::face_index(&face).map(|face| existing.insert((level, face, x, y)));
            }
        }

        // Download file list if necessary.
        let file_list_path = TERRA_DIRECTORY.join("tile_list.txt.lz4");
        let file_list_encoded = if !file_list_path.exists() {
            let contents = self.download("tile_list.txt.lz4").await?;
            if self.server.starts_with("http://") || self.server.starts_with("https://") {
                tokio::fs::write(&file_list_path, &contents).await?;
            }
            contents
        } else {
            tokio::fs::read(file_list_path).await?
        };

        // Parse file list to learn all files available from the remote.
        let all = Self::parse_tile_list(&file_list_encoded)?;

        let mut local_tiles = self.local_tiles.lock().unwrap();
        let mut remote_tiles = self.remote_tiles.lock().unwrap();

        VNode::breadth_first(|n| {
            if existing.contains(&(n.level(), n.face(), n.x(), n.y())) {
                local_tiles.insert(n);
            } else {
                local_tiles.remove(&n);
            }

            if all.contains(&(n.level(), n.face(), n.x(), n.y())) {
                remote_tiles.insert(n);
            }

            n.level() + 1 <= MAX_STREAMED_LEVEL
        });

        Ok(())
    }

    async fn download(&self, path: &str) -> Result<Vec<u8>, Error> {
        match self.server.split_once("//") {
            Some(("file:", base_path)) => {
                let full_path = PathBuf::from(base_path).join(path);
                Ok(tokio::fs::read(&full_path).await?)
            }
            Some(("http:", ..)) | Some(("https:", ..)) => {
                let url = format!("{}{}", self.server, path);
                let client = hyper::Client::builder()
                    .build::<_, hyper::Body>(hyper_tls::HttpsConnector::new());
                let resp = client.get(url.parse()?).await?;
                if resp.status().is_success() {
                    Ok(hyper::body::to_bytes(resp.into_body()).await?.to_vec())
                } else {
                    Err(anyhow::format_err!(
                        "Tile download failed with {:?} for URL '{}'",
                        resp.status(),
                        url
                    ))
                }
            }
            _ => Err(anyhow::format_err!("Invalid server URL {}", self.server)),
        }
    }

    // /// Return a list of the missing bases for a layer, as well as the total number bases in the layer.
    // pub(crate) fn get_missing_base(&self, layer: LayerType) -> (Vec<VNode>, usize) {
    //     let mut tiles_on_disk = self.local_tiles.lock().unwrap();
    //     let tiles_on_disk = tiles_on_disk.entry(layer.index()).or_insert_with(Default::default);

    //     let mut total = 0;
    //     let mut missing = Vec::new();
    //     VNode::breadth_first(|n| {
    //         total += 1;
    //         if !tiles_on_disk.contains(&n) {
    //             missing.push(n);
    //         }

    //         n.level() + 1 < layer.streamed_levels()
    //     });

    //     (missing, total)
    // }

    //
    // These functions use the database.
    //
    fn lookup_texture(&self, name: &str) -> Result<Option<TextureDescriptor>, Error> {
        Ok(self.textures.get(name)?.map(|value| serde_json::from_slice(&value).unwrap()))
    }
    fn update_texture(&self, name: &str, desc: TextureDescriptor) -> Result<(), Error> {
        let value = serde_json::to_vec(&desc).unwrap();
        self.textures.insert(name, value)?;
        Ok(())
    }
}
