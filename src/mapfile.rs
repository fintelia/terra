use crate::asset::TERRA_DIRECTORY;
use crate::cache::{LayerParams, LayerType, TextureFormat};
use anyhow::Error;
use atomicwrites::{AtomicFile, OverwriteBehavior};
use basis_universal::{TranscodeParameters, Transcoder, TranscoderTextureFormat};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::{fs, num::NonZeroU32};
use tokio::io::AsyncReadExt;
use types::VNode;
use vec_map::VecMap;

const TERRA_TILES_URL: &str = "https://terra.fintelia.io/file/terra-tiles/";

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

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ShaderDescriptor {
    hash: [u8; 32],
}

pub(crate) struct MapFile {
    layers: VecMap<LayerParams>,
    _db: sled::Db,
    textures: sled::Tree,

    remote_tiles: Arc<Mutex<VecMap<HashSet<VNode>>>>,
    local_tiles: Arc<Mutex<VecMap<HashSet<VNode>>>>,
}
impl MapFile {
    pub(crate) fn new(layers: VecMap<LayerParams>) -> Self {
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

        Self {
            layers,
            textures: db.open_tree("textures").unwrap(),
            _db: db,
            remote_tiles: Default::default(),
            local_tiles: Default::default(),
        }
    }

    pub(crate) fn tile_state(&self, layer: LayerType, node: VNode) -> Result<TileState, Error> {
        if node.level() >= self.layers[layer.index()].min_generated_level {
            return Ok(TileState::GpuOnly);
        }

        let exists = self
            .local_tiles
            .lock()
            .unwrap()
            .get(layer.index())
            .map(|m| m.contains(&node))
            .unwrap_or(false);

        if exists {
            Ok(TileState::Base)
        } else {
            Ok(TileState::MissingBase)
        }
    }
    pub(crate) async fn read_tile(&self, layer: LayerType, node: VNode) -> Result<Option<Vec<u8>>, Error> {
        assert!(layer.streamed());

        let filename = Self::tile_path(layer, node);
        if !filename.exists() {
            if !self.remote_tiles.lock().unwrap()[layer].contains(&node) {
                return Ok(None);
            }
            // unreachable!("{:?}", filename)
            let url = Self::tile_url(layer, node);
            let client = hyper::Client::builder()
                .build::<_, hyper::Body>(hyper_tls::HttpsConnector::new());
            let resp = client.get(url.parse()?).await?;
            if resp.status().is_success() {
                let data = hyper::body::to_bytes(resp.into_body()).await?.to_vec();
                // TODO: Fix lifetime issues so we can do this tile write asynchronously.
                tokio::task::block_in_place(|| self.write_tile(layer, node, &data))?;
                return Ok(Some(data));
            } else {
                panic!("Tile download failed with {:?} for URL '{}'", resp.status(), url);
            }
        }

        let mut contents = Vec::new();
        tokio::fs::File::open(filename).await?.read_to_end(&mut contents).await?;
        Ok(Some(contents))
    }

    pub(crate) fn write_tile(
        &self,
        layer: LayerType,
        node: VNode,
        data: &[u8],
    ) -> Result<(), Error> {
        let filename = Self::tile_path(layer, node);
        if let Some(parent) = filename.parent() {
            fs::create_dir_all(parent)?;
        }

        AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
            .write(|f| f.write_all(data))?;

        self.local_tiles
            .lock()
            .unwrap()
            .entry(layer.index())
            .or_insert_with(Default::default)
            .insert(node);

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

    pub(crate) fn layers(&self) -> &VecMap<LayerParams> {
        &self.layers
    }

    fn layer_name_ext_strs(layer: LayerType) -> (&'static str, &'static str) {
         match layer {
            LayerType::BaseAlbedo => ("albedo", "png"),
            LayerType::Heightmaps => ("heightmaps", "raw"),
            LayerType::TreeCover => ("treecover", "tiff"),
            LayerType::WaterMask => ("watermask", "tiff"),
            _ => unreachable!(),
        }
    }
    fn tile_name(layer: LayerType, node: VNode) -> String {
        let face = match node.face() {
            0 => "0E",
            1 => "180E",
            2 => "90E",
            3 => "90W",
            4 => "N",
            5 => "S",
            _ => unreachable!(),
        };
        let (layer, ext) = Self::layer_name_ext_strs(layer);
        format!("{}/{}_{}_{}_{}x{}.{}", layer, layer, node.level(), face, node.x(), node.y(), ext)
    }

    fn tile_path(layer: LayerType, node: VNode) -> PathBuf {
        TERRA_DIRECTORY.join("tiles").join(&Self::tile_name(layer, node))
    }

    fn tile_url(layer: LayerType, node: VNode) -> String {
        format!("{}{}", TERRA_TILES_URL, Self::tile_name(layer, node))
    }

    pub(crate) async fn reload_tile_states(&self, layer: LayerType) -> Result<(), Error> {
        let (target_layer, target_ext) = Self::layer_name_ext_strs(layer);

        fn face_index(s: &str) -> Option<u8>{
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

        let mut all = HashSet::new();
        let mut existing = HashSet::new();

        // Scan local files.
        let directory = TERRA_DIRECTORY.join("tiles").join(target_layer);
        std::fs::create_dir_all(&directory)?;
        for file in fs::read_dir(directory)? {
            let filename = file?.file_name();
            let filename = filename.to_string_lossy();

            if let Some((layer, level, face, x, y, ext)) =
                sscanf::scanf!(filename, "{}_{}_{}_{}x{}.{}", String, u8, String, u32, u32, String)
            {
                let face = face_index(&face);
                if layer == target_layer && ext == target_ext && face.is_some(){
                    existing.insert((level, face.unwrap(), x, y));
                }
            }
        }

        // Download file list if necessary.
        let file_list_path =
            TERRA_DIRECTORY.join(&format!("tiles/{}_tile_list.txt.lz4", target_layer));
        if !file_list_path.exists() {
            let url = format!("{}{}_tile_list.txt.lz4", TERRA_TILES_URL, target_layer);
            let client =
                hyper::Client::builder().build::<_, hyper::Body>(hyper_tls::HttpsConnector::new());
            let resp = client.get(url.parse()?).await?;
            if resp.status().is_success() {
                let contents = hyper::body::to_bytes(resp.into_body()).await?.to_vec();
                tokio::fs::write(&file_list_path, contents).await?;
            } else {
                anyhow::bail!("Failed to download '{}'", url);
            }
        }
        // Parse file list to learn all files available from the remote.
        let mut remote_files = String::new();
        let encoded = tokio::fs::read(file_list_path).await?;
        lz4::Decoder::new(std::io::Cursor::new(&encoded))?.read_to_string(&mut remote_files)?;
        for filename in remote_files.split("\n") {
            if let Some((layer, level, face, x, y, ext)) =
                sscanf::scanf!(filename, "{}_{}_{}_{}x{}.{}", String, u8, String, u32, u32, String)
            {
                let face = face_index(&face);
                if layer == target_layer && ext == target_ext && face.is_some(){
                    all.insert((level, face.unwrap(), x, y));
                }
            }
        }

        let mut local_tiles = self.local_tiles.lock().unwrap();
        let mut remote_tiles = self.remote_tiles.lock().unwrap();

        let local_tiles = local_tiles.entry(layer.index()).or_insert_with(Default::default);
        let remote_tiles = remote_tiles.entry(layer.index()).or_insert_with(Default::default);

        VNode::breadth_first(|n| {
            if existing.contains(&(n.level(), n.face(), n.x(), n.y())) {
                local_tiles.insert(n);
            } else {
                local_tiles.remove(&n);
            }

            if all.contains(&(n.level(), n.face(), n.x(), n.y())) {
                remote_tiles.insert(n);
            }

            n.level() + 1 < self.layers[layer.index()].min_generated_level
        });

        Ok(())
    }

    /// Return a list of the missing bases for a layer, as well as the total number bases in the layer.
    pub(crate) fn get_missing_base(&self, layer: LayerType) -> (Vec<VNode>, usize) {
        let mut tiles_on_disk = self.local_tiles.lock().unwrap();
        let tiles_on_disk = tiles_on_disk.entry(layer.index()).or_insert_with(Default::default);

        let mut total = 0;
        let mut missing = Vec::new();
        VNode::breadth_first(|n| {
            total += 1;
            if !tiles_on_disk.contains(&n) {
                missing.push(n);
            }

            n.level() + 1 < self.layers[layer.index()].min_generated_level
        });

        (missing, total)
    }

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
