use crate::asset::TERRA_DIRECTORY;
use crate::terrain::quadtree::node::VNode;
use crate::cache::{LayerParams, LayerType, TextureFormat};
use anyhow::Error;
use atomicwrites::{AtomicFile, OverwriteBehavior};
use image::bmp::BmpEncoder;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use tokio::io::AsyncReadExt;
use vec_map::VecMap;

const TERRA_TILES_URL: &str = "https://terra.fintelia.io/file/terra-tiles/";

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum TileState {
    Missing,
    Base,
    Generated,
    GpuOnly,
    MissingBase,
}

#[derive(PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum TileKind {
    Base,
    Generate,
    GpuOnly,
}

#[derive(PartialEq, Eq, Serialize, Deserialize)]
struct TileMeta {
    crc32: u32,
    state: TileState,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(crate) struct TextureDescriptor {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: TextureFormat,
    pub bytes: usize,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ShaderDescriptor {
    hash: [u8; 32],
}

pub(crate) struct MapFile {
    layers: VecMap<LayerParams>,
    _db: sled::Db,
    tiles: sled::Tree,
    textures: sled::Tree,
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
            tiles: db.open_tree("tiles").unwrap(),
            textures: db.open_tree("textures").unwrap(),
            _db: db,
        }
    }

    pub(crate) fn tile_state(&self, layer: LayerType, node: VNode) -> Result<TileState, Error> {
        Ok(match self.lookup_tile_meta(layer, node)? {
            Some(meta) => meta.state,
            None => TileState::GpuOnly,
        })
    }
    pub(crate) async fn read_tile(&self, layer: LayerType, node: VNode) -> Result<Vec<u8>, Error> {
        let filename = Self::tile_path(layer, node);
        if !filename.exists() {
            match layer {
                LayerType::Albedo | LayerType::Heightmaps | LayerType::Roughness => {
                    let url = Self::tile_url(layer, node);
                    let client = hyper::Client::builder()
                        .build::<_, hyper::Body>(hyper_tls::HttpsConnector::new());
                    let resp = client.get(url.parse()?).await?;
                    if resp.status().is_success() {
                        let data = hyper::body::to_bytes(resp.into_body()).await?.to_vec();
                        // TODO: Fix lifetime issues so we can do this tile write asynchronously.
                        tokio::task::block_in_place(|| self.write_tile(layer, node, &data, true))?;
                        return Ok(data);
                    } else {
                        panic!("Tile download failed with {:?} for URL '{}'", resp.status(), url);
                    }
                }
                _ => {}
            }
            anyhow::bail!("Tile missing: '{:?}'", filename);
        }

        let mut contents = Vec::new();
        tokio::fs::File::open(filename).await?.read_to_end(&mut contents).await?;
        Ok(contents)
    }

    pub(crate) fn write_tile(
        &self,
        layer: LayerType,
        node: VNode,
        data: &[u8],
        base: bool,
    ) -> Result<(), Error> {
        let filename = Self::tile_path(layer, node);
        if let Some(parent) = filename.parent() {
            fs::create_dir_all(parent)?;
        }

        AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
            .write(|f| f.write_all(data))?;

        self.update_tile_meta(
            layer,
            node,
            TileMeta { crc32: 0, state: if base { TileState::Base } else { TileState::Generated } },
        )
    }

    pub(crate) fn read_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: &str,
    ) -> Result<wgpu::Texture, Error> {
        let desc = self.lookup_texture(name)?.unwrap();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width: desc.width, height: desc.height, depth: desc.depth },
            format: desc.format.to_wgpu(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: if desc.depth == 1 {
                wgpu::TextureDimension::D2
            } else {
                wgpu::TextureDimension::D3
            },
            usage: wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::STORAGE,
            label: Some(&format!("texture.{}", name)),
        });

        let (width, height) = (desc.width as usize, (desc.height * desc.depth) as usize);
        assert_eq!(width % desc.format.block_size() as usize, 0);
        assert_eq!(height % desc.format.block_size() as usize, 0);
        let (width, height) =
            (width / desc.format.block_size() as usize, height / desc.format.block_size() as usize);

        let row_bytes = width * desc.format.bytes_per_block();

        let mut data = if desc.format == TextureFormat::RGBA8 {
            image::open(TERRA_DIRECTORY.join(format!("{}.bmp", name)))?.to_rgba8().into_vec()
        } else {
            fs::read(TERRA_DIRECTORY.join(format!("{}.raw", name)))?
        };

        if cfg!(feature = "small-trace") {
            let bytes_per_block = desc.format.bytes_per_block();
            for y in 0..height {
                for x in 0..width {
                    if x % 16 == 0 && y % 16 == 0 { continue; }
                    let src = ((x & !15) + (y & !15) * width) * bytes_per_block;
                    let dst = (x + y * width) * bytes_per_block;
                    data.copy_within(src..src+bytes_per_block, dst);
                }
            }
        }

        queue.write_texture(
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            },
            &data,
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: row_bytes as u32,
                rows_per_image: height as u32 / desc.depth,
            },
            wgpu::Extent3d {
                width: width as u32,
                height: height as u32 / desc.depth,
                depth: desc.depth,
            },
        );

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
            let filename = TERRA_DIRECTORY.join(format!("{}.bmp", name));
            let mut encoded = Vec::new();
            BmpEncoder::new(&mut encoded).encode(
                data,
                desc.width,
                desc.height * desc.depth,
                image::ColorType::Rgba8,
            )?;
            Ok(AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
                .write(|f| f.write_all(&encoded))?)
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
                TERRA_DIRECTORY.join(format!("{}.bmp", name)).exists()
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
        let (layer, ext) = match layer {
            LayerType::Displacements => ("displacements", "raw"),
            LayerType::Albedo => ("albedo", "png"),
            LayerType::Roughness => ("roughness", "raw.lz4"),
            LayerType::Normals => ("normals", "raw"),
            LayerType::Heightmaps => ("heightmaps", "raw"),
        };
        format!("{}/{}_{}_{}_{}x{}.{}", layer, layer, node.level(), face, node.x(), node.y(), ext)
    }

    fn tile_path(layer: LayerType, node: VNode) -> PathBuf {
        TERRA_DIRECTORY.join("tiles").join(&Self::tile_name(layer, node))
    }

    fn tile_url(layer: LayerType, node: VNode) -> String {
        format!("{}{}", TERRA_TILES_URL, Self::tile_name(layer, node))
    }

    pub(crate) fn reload_tile_state(
        &self,
        layer: LayerType,
        node: VNode,
        base: bool,
    ) -> Result<TileState, Error> {
        let filename = Self::tile_path(layer, node);
        let meta = self.lookup_tile_meta(layer, node);

        let exists = filename.exists();

        let target_state = if base && exists {
            TileState::Base
        } else if base {
            TileState::MissingBase
        } else if exists {
            TileState::Generated
        } else {
            TileState::Missing
        };

        if let Ok(Some(TileMeta { state, .. })) = meta {
            if state == target_state {
                return Ok(state);
            }
        }

        let new_meta = TileMeta { state: target_state, crc32: 0 };
        self.update_tile_meta(layer, node, new_meta)?;
        Ok(target_state)
    }
    #[allow(unused)]
    pub(crate) fn clear_generated(&self, layer: LayerType) -> Result<(), Error> {
        self.scan_tile_meta(layer, |node, meta| {
            if let TileState::Generated = meta.state {
                self.remove_tile_meta(layer, node)?;
            }
            Ok(())
        })
    }
    /// Return a list of the missing bases for a layer, as well as the total number bases in the layer.
    pub(crate) fn get_missing_base(&self, layer: LayerType) -> Result<(Vec<VNode>, usize), Error> {
        let mut total = 0;
        let mut missing = Vec::new();
        self.scan_tile_meta(layer, |node, meta| {
            total += 1;
            if let TileState::MissingBase = meta.state {
                missing.push(node);
            }
            Ok(())
        })?;
        Ok((missing, total))
    }

    //
    // These functions use the database.
    //
    fn lookup_tile_meta(&self, layer: LayerType, node: VNode) -> Result<Option<TileMeta>, Error> {
        let key = bincode::serialize(&(layer, node)).unwrap();
        Ok(self.tiles.get(key)?.map(|value| bincode::deserialize(&value).unwrap()))
    }
    fn update_tile_meta(&self, layer: LayerType, node: VNode, meta: TileMeta) -> Result<(), Error> {
        let key = bincode::serialize(&(layer, node)).unwrap();
        let value = bincode::serialize(&meta).unwrap();
        self.tiles.insert(key, value)?;
        Ok(())
    }
    fn remove_tile_meta(&self, layer: LayerType, node: VNode) -> Result<(), Error> {
        let key = bincode::serialize(&(layer, node)).unwrap();
        self.tiles.remove(key)?;
        Ok(())
    }
    fn scan_tile_meta<F: FnMut(VNode, TileMeta) -> Result<(), Error>>(
        &self,
        layer: LayerType,
        mut f: F,
    ) -> Result<(), Error> {
        let prefix = bincode::serialize(&layer).unwrap();
        for i in self.tiles.scan_prefix(&prefix) {
            let (k, v) = i?;
            let meta = bincode::deserialize::<TileMeta>(&v)?;
            let node = bincode::deserialize::<(LayerType, VNode)>(&k)?.1;
            f(node, meta)?;
        }
        Ok(())
    }

    fn lookup_texture(&self, name: &str) -> Result<Option<TextureDescriptor>, Error> {
        Ok(self.textures.get(name)?.map(|value| serde_json::from_slice(&value).unwrap()))
    }
    fn update_texture(&self, name: &str, desc: TextureDescriptor) -> Result<(), Error> {
        let value = serde_json::to_vec(&desc).unwrap();
        self.textures.insert(name, value)?;
        Ok(())
    }
}
