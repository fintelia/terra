use crate::cache::TERRA_DIRECTORY;
use crate::terrain::quadtree::node::VNode;
use crate::terrain::tile_cache::{
    LayerParams, LayerType, TextureDescriptor, TextureFormat,
};
use failure::Error;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use vec_map::VecMap;

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

#[derive(PartialEq, Eq, Serialize, Deserialize)]
enum KeyType {
    Tile,
    Texture,
}

pub struct MapFile {
    layers: VecMap<LayerParams>,
    db: sled::Db,
}
impl MapFile {
    pub(crate) fn new(layers: VecMap<LayerParams>) -> Self {
        let db = sled::open(TERRA_DIRECTORY.join("tiles/meta")).unwrap();
        Self { layers, db }
    }

    pub(crate) fn tile_state(&self, layer: LayerType, node: VNode) -> Result<TileState, Error> {
        Ok(match self.lookup_tile_meta(layer, node)? {
            Some(meta) => meta.state,
            None => TileState::GpuOnly,
        })
    }
    pub(crate) fn read_tile(&self, layer: LayerType, node: VNode) -> Option<Vec<u8>> {
        let filename = Self::tile_name(layer, node);
        match layer {
            LayerType::Albedo => {
                if !filename.exists() {
                    return None;
                }
                let image = image::open(filename).ok()?;
                Some(image.to_rgba().into_vec())
            }
            LayerType::Heightmaps | LayerType::Normals | LayerType::Displacements => {
                if !filename.exists() {
                    return None;
                }
                fs::read(filename).ok()
            }
        }
    }
    pub(crate) fn write_tile(
        &mut self,
        layer: LayerType,
        node: VNode,
        data: &[u8],
        base: bool,
    ) -> Result<(), Error> {
        let filename = Self::tile_name(layer, node);
        match layer {
            LayerType::Albedo => image::save_buffer_with_format(
                &filename,
                data,
                self.layers[layer].texture_resolution as u32,
                self.layers[layer].texture_resolution as u32,
                image::ColorType::Rgba8,
                image::ImageFormat::Bmp,
            )?,
            LayerType::Heightmaps | LayerType::Normals | LayerType::Displacements => {
                fs::write(filename, data)?;
            }
        }

        self.update_tile_meta(
            layer,
            node,
            TileMeta { crc32: 0, state: if base { TileState::Base } else { TileState::Generated } },
        )
    }

    pub(crate) fn read_texture(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
    ) -> Result<wgpu::Texture, Error> {
        let desc = self.lookup_texture(name)?.unwrap();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width: desc.resolution, height: desc.resolution, depth: 1 },
            format: desc.format.to_wgpu(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::STORAGE,
            label: None,
        });

        let resolution = desc.resolution as usize;
        let bytes_per_texel = desc.format.bytes_per_texel();
        let row_bytes = resolution * bytes_per_texel;
        let row_pitch = (row_bytes + 255) & !255;
        // let data = &self.file[desc.offset..][..desc.bytes];

        let filename = TERRA_DIRECTORY.join(format!("{}.bmp", name));
        let image = image::open(filename)?;
        let data = image.to_rgba().into_vec();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: (row_pitch * resolution) as u64,
            usage: wgpu::BufferUsage::MAP_WRITE | wgpu::BufferUsage::COPY_SRC,
            label: None,
            mapped_at_creation: true,
        });

        let mut buffer_view = buffer.slice(..).get_mapped_range_mut();
        for row in 0..resolution {
            buffer_view[row * row_pitch..][..row_bytes]
                .copy_from_slice(&data[row * row_bytes..][..row_bytes]);
        }

        drop(buffer_view);
        buffer.unmap();
        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: row_pitch as u32,
                    rows_per_image: resolution as u32,
                },
            },
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            },
            wgpu::Extent3d { width: resolution as u32, height: resolution as u32, depth: 1 },
        );

        Ok(texture)
    }

    pub(crate) fn write_texture(
        &self,
        name: &str,
        desc: TextureDescriptor,
        data: &[u8],
    ) -> Result<(), Error> {
        assert_eq!(desc.format, TextureFormat::RGBA8);
        let filename = TERRA_DIRECTORY.join(format!("{}.bmp", name));
        let resolution = desc.resolution as u32;
        self.update_texture(name, desc)?;
        Ok(image::save_buffer_with_format(
            &filename,
            data,
            resolution,
            resolution,
            image::ColorType::Rgba8,
            image::ImageFormat::Bmp,
        )?)
    }

    pub(crate) fn reload_texture(&self, name: &str) -> bool {
        let filename = TERRA_DIRECTORY.join(format!("{}.bmp", name));
        let desc = self.lookup_texture(name);
        if let Ok(Some(_)) = desc {
            filename.exists()
        } else {
            false
        }
    }

    pub(crate) fn layers(&self) -> &VecMap<LayerParams> {
        &self.layers
    }

    pub(crate) fn tile_name(layer: LayerType, node: VNode) -> PathBuf {
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
            LayerType::Albedo => ("albedo", "bmp"),
            LayerType::Normals => ("normals", "raw"),
            LayerType::Heightmaps => ("heightmaps", "raw"),
        };
        TERRA_DIRECTORY.join(&format!(
            "tiles/{}_{}_{}_{}x{}.{}",
            layer,
            node.level(),
            face,
            node.x(),
            node.y(),
            ext
        ))
    }

    pub(crate) fn reload_tile_state(
        &self,
        layer: LayerType,
        node: VNode,
        base: bool,
    ) -> Result<TileState, Error> {
        let filename = Self::tile_name(layer, node);
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
    // pub(crate) fn set_missing(
    //     &self,
    //     layer: LayerType,
    //     node: VNode,
    //     base: bool,
    // ) -> Result<(), Error> {
    //     let state = if base { TileState::MissingBase } else { TileState::Missing };
    //     self.update_tile_meta(layer, node, TileMeta { crc32: 0, state })
    // }
    pub(crate) fn clear_generated(&mut self, layer: LayerType) -> Result<(), Error> {
        self.scan_tile_meta(layer, |node, meta| {
            if let TileState::Generated = meta.state {
                self.remove_tile_meta(layer, node)?;
            }
            Ok(())
        })
    }
    pub(crate) fn get_missing_base(&self, layer: LayerType) -> Result<Vec<VNode>, Error> {
        let mut missing = Vec::new();
        self.scan_tile_meta(layer, |node, meta| {
            if let TileState::MissingBase = meta.state {
                missing.push(node);
            }
            Ok(())
        })?;
        Ok(missing)
    }

    //
    // These functions use the database.
    //
    fn lookup_tile_meta(&self, layer: LayerType, node: VNode) -> Result<Option<TileMeta>, Error> {
        let key = bincode::serialize(&(KeyType::Tile, layer, node)).unwrap();
        Ok(self.db.get(key)?.map(|value| bincode::deserialize(&value).unwrap()))
    }
    fn update_tile_meta(&self, layer: LayerType, node: VNode, meta: TileMeta) -> Result<(), Error> {
        let key = bincode::serialize(&(KeyType::Tile, layer, node)).unwrap();
        let value = bincode::serialize(&meta).unwrap();
        self.db.insert(key, value)?;
        Ok(())
    }
    fn remove_tile_meta(&self, layer: LayerType, node: VNode) -> Result<(), Error> {
        let key = bincode::serialize(&(KeyType::Tile, layer, node)).unwrap();
        self.db.remove(key)?;
        Ok(())
    }
    fn scan_tile_meta<F: FnMut(VNode, TileMeta) -> Result<(), Error>>(
        &self,
        layer: LayerType,
        mut f: F,
    ) -> Result<(), Error> {
        let prefix = bincode::serialize(&(KeyType::Tile, layer)).unwrap();
        for i in self.db.scan_prefix(&prefix) {
            let (k, v) = i?;
            let meta = bincode::deserialize::<TileMeta>(&v)?;
            let node = bincode::deserialize::<(KeyType, LayerType, VNode)>(&k)?.2;
            f(node, meta)?;
        }
        Ok(())
    }

    fn lookup_texture(&self, name: &str) -> Result<Option<TextureDescriptor>, Error> {
        let key = bincode::serialize(&(KeyType::Texture, name)).unwrap();
        Ok(self.db.get(key)?.map(|value| bincode::deserialize(&value).unwrap()))
    }
    fn update_texture(&self, name: &str, desc: TextureDescriptor) -> Result<(), Error> {
        let key = bincode::serialize(&(KeyType::Texture, name)).unwrap();
        let value = bincode::serialize(&desc).unwrap();
        self.db.insert(key, value)?;
        Ok(())
    }
}
