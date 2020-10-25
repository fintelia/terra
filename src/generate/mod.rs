use crate::cache::{AssetLoadContext, AssetLoadContextBuf, WebAsset};
use crate::coordinates::CoordinateSystem;
use crate::mapfile::{MapFile, TextureDescriptor};
use crate::srgb::SRGB_TO_LINEAR;
use crate::terrain::dem::DemSource;
use crate::terrain::dem::GlobalDem;
use crate::terrain::landcover::{BlueMarble, BlueMarbleTileSource};
use crate::terrain::quadtree::VNode;
use crate::terrain::raster::RasterCache;
use crate::terrain::tile_cache::{LayerParams, LayerType, TextureFormat};
use anyhow::Error;
use maplit::hashmap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use vec_map::VecMap;

mod gpu;
pub mod heightmap;

pub(crate) use gpu::*;

/// The radius of the earth in meters.
pub(crate) const EARTH_RADIUS: f64 = 6371000.0;
pub(crate) const EARTH_CIRCUMFERENCE: f64 = 2.0 * PI * EARTH_RADIUS;

// Mapping from side length to level number.
#[allow(unused)]
mod levels {
    pub const LEVEL_10000_KM: i32 = 0;
    pub const LEVEL_5000_KM: i32 = 1;
    pub const LEVEL_2500_KM: i32 = 2;
    pub const LEVEL_1250_KM: i32 = 3;
    pub const LEVEL_625_KM: i32 = 4;
    pub const LEVEL_300_KM: i32 = 5;
    pub const LEVEL_150_KM: i32 = 6;
    pub const LEVEL_75_KM: i32 = 7;
    pub const LEVEL_40_KM: i32 = 8;
    pub const LEVEL_20_KM: i32 = 9;
    pub const LEVEL_10_KM: i32 = 10;
    pub const LEVEL_5_KM: i32 = 11;
    pub const LEVEL_2_KM: i32 = 12;
    pub const LEVEL_1_KM: i32 = 13;
    pub const LEVEL_600_M: i32 = 14;
    pub const LEVEL_305_M: i32 = 15;
    pub const LEVEL_153_M: i32 = 16;
    pub const LEVEL_76_M: i32 = 17;
    pub const LEVEL_38_M: i32 = 18;
    pub const LEVEL_19_M: i32 = 19;
    pub const LEVEL_10_M: i32 = 20;
    pub const LEVEL_5_M: i32 = 21;
    pub const LEVEL_2_M: i32 = 22;
    pub const LEVEL_1_M: i32 = 23;
    pub const LEVEL_60_CM: i32 = 24;
    pub const LEVEL_30_CM: i32 = 25;
    pub const LEVEL_15_CM: i32 = 26;

    pub const TILE_CELL_20_KM: u8 = 0;
    pub const TILE_CELL_10_KM: u8 = 1;
    pub const TILE_CELL_5_KM: u8 = 2;
    pub const TILE_CELL_2_KM: u8 = 3;
    pub const TILE_CELL_1_KM: u8 = 4;
    pub const TILE_CELL_625M: u8 = 5;
    pub const TILE_CELL_305M: u8 = 6;
    pub const TILE_CELL_153M: u8 = 7;
    pub const TILE_CELL_76M: u8 = 8;
    pub const TILE_CELL_38M: u8 = 9;
    pub const TILE_CELL_19M: u8 = 10;
    pub const TILE_CELL_10M: u8 = 11;
    pub const TILE_CELL_5M: u8 = 12;
    pub const TILE_CELL_2M: u8 = 13;
    pub const TILE_CELL_1M: u8 = 14;
}
use levels::*;

/// Used to construct a `QuadTree`.
pub struct MapFileBuilder;
impl MapFileBuilder {
    /// Actually construct the `QuadTree`.
    ///
    /// This function will (the first time it is called) download many gigabytes of raw data,
    /// primarily datasets relating to real world land cover and elevation. These files will be
    /// stored in ~/.terra, so that they don't have to be fetched multiple times. This means that
    /// this function can largely resume from where it left off if interrupted.
    ///
    /// Even once all needed files have been downloaded, the generation process takes a large amount
    /// of CPU resources. You can expect it to run at full load continiously for several full
    /// minutes, even in release builds (you *really* don't want to wait for generation in debug
    /// mode...).
    pub async fn build() -> Result<MapFile, Error> {
        let layers: VecMap<LayerParams> = hashmap![
            LayerType::Heightmaps.index() => LayerParams {
                    layer_type: LayerType::Heightmaps,
                    texture_resolution: 521,
                    texture_border_size: 4,
                    texture_format: TextureFormat::R32F,
                },
            LayerType::Displacements.index() => LayerParams {
                    layer_type: LayerType::Displacements,
                    texture_resolution: 65,
                    texture_border_size: 0,
                    texture_format: TextureFormat::RGBA32F,
                },
            LayerType::Albedo.index() => LayerParams {
                    layer_type: LayerType::Albedo,
                    texture_resolution: 516,
                    texture_border_size: 2,
                    texture_format: TextureFormat::RGBA8,
                },
            LayerType::Roughness.index() => LayerParams {
                    layer_type: LayerType::Roughness,
                    texture_resolution: 516,
                    texture_border_size: 2,
                    texture_format: TextureFormat::BC4,
                },
            LayerType::Normals.index() => LayerParams {
                    layer_type: LayerType::Normals,
                    texture_resolution: 516,
                    texture_border_size: 2,
                    texture_format: TextureFormat::BC5,
                },
        ]
        .into_iter()
        .collect();

        let mut mapfile = MapFile::new(layers);
        VNode::breadth_first(|n| {
            mapfile.reload_tile_state(LayerType::Heightmaps, n, true).unwrap();
            n.level() < TILE_CELL_153M
        });
        VNode::breadth_first(|n| {
            mapfile.reload_tile_state(LayerType::Albedo, n, true).unwrap();
            n.level() < TILE_CELL_625M
        });
        VNode::breadth_first(|n| {
            mapfile.reload_tile_state(LayerType::Roughness, n, true).unwrap();
            false
        });

        VNode::breadth_first(|n| {
            mapfile.reload_tile_state(LayerType::Normals, n, false).unwrap();
            n.level() < TILE_CELL_625M
        });

        let mut context = AssetLoadContextBuf::new();
        let mut context = context.context("Generating mapfile...", 5);
        generate_heightmaps(&mut mapfile, &mut context).await?;
        context.set_progress(1);
        generate_albedo(&mut mapfile, &mut context)?;
        context.set_progress(2);
        generate_roughness(&mut mapfile, &mut context)?;
        context.set_progress(3);
        generate_noise(&mut mapfile)?;
        context.set_progress(4);
        generate_sky(&mut mapfile, &mut context)?;
        context.set_progress(5);

        Ok(mapfile)
    }
}

async fn generate_heightmaps<'a>(
    mapfile: &mut MapFile,
    context: &mut AssetLoadContext<'a>,
) -> Result<(), Error> {
    let mut missing = mapfile.get_missing_base(LayerType::Heightmaps)?;
    if missing.is_empty() {
        return Ok(());
    }

    missing.sort_by_key(|n| n.level());

    let mut gen = heightmap::HeightmapGen {
        tile_cache: heightmap::HeightmapCache::new(
            mapfile.layers()[LayerType::Heightmaps].clone(),
            32,
        ),
        dems: RasterCache::new(Box::new(DemSource::Srtm90m), 256),
        global_dem: GlobalDem.load(context)?,
    };

    let context = &mut context.increment_level("Writing heightmaps... ", missing.len());
    for (i, n) in missing.into_iter().enumerate() {
        context.set_progress(i as u64);
        gen.generate_heightmaps(context, mapfile, n).await?;
    }

    Ok(())
}

fn generate_albedo(mapfile: &mut MapFile, context: &mut AssetLoadContext) -> Result<(), Error> {
    let missing = mapfile.get_missing_base(LayerType::Albedo)?;
    if missing.is_empty() {
        return Ok(());
    }

    let layer = mapfile.layers()[LayerType::Albedo].clone();
    assert!(layer.texture_border_size >= 2);

    let bluemarble = BlueMarble.load(context)?;
    let bluemarble_spacing = bluemarble.spacing() as f32;

    let mut bluemarble_cache = RasterCache::new(Box::new(BlueMarbleTileSource), 8);

    let context = &mut context.increment_level("Generating colormaps... ", missing.len());
    for (i, n) in missing.into_iter().enumerate() {
        context.set_progress(i as u64);
        let mut colormap = Vec::with_capacity(
            layer.texture_resolution as usize * layer.texture_resolution as usize,
        );
        let spacing = n.aprox_side_length()
            / (layer.texture_resolution - 2 * layer.texture_border_size) as f32;

        let coordinates: Vec<_> = (0..(layer.texture_resolution * layer.texture_resolution))
            .into_par_iter()
            .map(|i| {
                let cspace = n.cell_position_cspace(
                    (i % layer.texture_resolution) as i32,
                    (i / layer.texture_resolution) as i32,
                    layer.texture_border_size as u16,
                    layer.texture_resolution as u16,
                );
                let sspace = CoordinateSystem::cspace_to_sspace(cspace);
                let polar = CoordinateSystem::sspace_to_polar(sspace);
                (polar.x.to_degrees(), polar.y.to_degrees())
            })
            .collect();

        if spacing < bluemarble_spacing {
            colormap.resize(coordinates.len() * 4, 255);

            let mut samples: HashMap<(i16, i16), Vec<(usize, f64, f64)>> = Default::default();
            for (i, (lat, long)) in coordinates.into_iter().enumerate() {
                samples
                    .entry((lat.floor() as i16, long.floor() as i16))
                    .or_default()
                    .push((i, lat, long));
            }

            for (tile, v) in samples {
                let tile = bluemarble_cache.get(context, tile.0, tile.1).unwrap();
                let v: Vec<_> = v
                    .into_par_iter()
                    .map(|(i, lat, long)| {
                        let [r, g, b] = tile.nearest3(lat, long).unwrap_or([255.0, 0.0, 0.0]);
                        (
                            i,
                            SRGB_TO_LINEAR[r as u8],
                            SRGB_TO_LINEAR[g as u8],
                            SRGB_TO_LINEAR[b as u8],
                        )
                    })
                    .collect();

                for (i, r, g, b) in v {
                    colormap[i * 4] = r;
                    colormap[i * 4 + 1] = g;
                    colormap[i * 4 + 2] = b;
                }
            }
        } else {
            for (lat, long) in coordinates {
                colormap.extend_from_slice(&[
                    SRGB_TO_LINEAR[bluemarble.interpolate(lat, long, 0) as u8],
                    SRGB_TO_LINEAR[bluemarble.interpolate(lat, long, 1) as u8],
                    SRGB_TO_LINEAR[bluemarble.interpolate(lat, long, 2) as u8],
                    255,
                ]);
            }
        }

        mapfile.write_tile(LayerType::Albedo, n, &colormap, true)?;
    }

    Ok(())
}

fn generate_roughness(mapfile: &mut MapFile, context: &mut AssetLoadContext) -> Result<(), Error> {
    let missing = mapfile.get_missing_base(LayerType::Roughness)?;
    if missing.is_empty() {
        return Ok(());
    }

    let layer = mapfile.layers()[LayerType::Roughness].clone();
    assert!(layer.texture_border_size >= 2);
    assert_eq!(layer.texture_resolution % 4, 0);

    let context = &mut context.increment_level("Generating roughness... ", missing.len());
    for (i, n) in missing.into_iter().enumerate() {
        context.set_progress(i as u64);

        let mut data = Vec::with_capacity(
            layer.texture_resolution as usize * layer.texture_resolution as usize / 2,
        );
        for _ in 0..(layer.texture_resolution / 4) {
            for _ in 0..(layer.texture_resolution / 4) {
                data.extend_from_slice(&[179, 180, 0, 0, 0, 0, 0, 0]);
            }
        }

        mapfile.write_tile(LayerType::Roughness, n, &data, true)?;
    }

    Ok(())
}

fn generate_noise(mapfile: &mut MapFile) -> Result<(), Error> {
    if !mapfile.reload_texture("noise") {
        // wavelength = 1.0 / 256.0;
        let noise_desc = TextureDescriptor {
            width: 2048,
            height: 2048,
            depth: 1,
            format: TextureFormat::RGBA8,
            bytes: 4 * 2048 * 2048,
        };

        let noise_heightmaps: Vec<_> =
            (0..4).map(|i| crate::terrain::heightmap::wavelet_noise(64 << i, 32 >> i)).collect();

        let len = noise_heightmaps[0].heights.len();
        let mut heights = vec![0u8; len * 4];
        for (i, heightmap) in noise_heightmaps.into_iter().enumerate() {
            let mut dist: Vec<(usize, f32)> = heightmap.heights.into_iter().enumerate().collect();
            dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for j in 0..len {
                heights[dist[j].0 * 4 + i] = (j * 256 / len) as u8;
            }
        }

        mapfile.write_texture("noise", noise_desc, &heights[..])?;
    }
    Ok(())
}

fn generate_sky(mapfile: &mut MapFile, context: &mut AssetLoadContext) -> Result<(), Error> {
    if !mapfile.reload_texture("sky") {
        let context = &mut context.increment_level("Generating sky texture... ", 1);
        let sky = WebTextureAsset {
            url: "https://www.eso.org/public/archives/images/original/eso0932a.tif".to_owned(),
            filename: "eso0932a.tif".to_owned(),
        }
        .load(context)?;
        mapfile.write_texture("sky", sky.0, &sky.1)?;
    }
    if !mapfile.reload_texture("transmittance") || !mapfile.reload_texture("inscattering") {
        let atmosphere = crate::sky::Atmosphere::new(context)?;
        mapfile.write_texture(
            "transmittance",
            TextureDescriptor {
                width: atmosphere.transmittance.size[0] as u32,
                height: atmosphere.transmittance.size[1] as u32,
                depth: 1,
                format: TextureFormat::RGBA32F,
                bytes: atmosphere.transmittance.data.len() * 4,
            },
            bytemuck::cast_slice(&atmosphere.transmittance.data),
        )?;
        mapfile.write_texture(
            "inscattering",
            TextureDescriptor {
                width: atmosphere.inscattering.size[0] as u32,
                height: atmosphere.inscattering.size[1] as u32,
                depth: atmosphere.inscattering.size[2] as u32,
                format: TextureFormat::RGBA32F,
                bytes: atmosphere.inscattering.data.len() * 4,
            },
            bytemuck::cast_slice(&atmosphere.inscattering.data),
        )?;
    }
    Ok(())
}

struct WebTextureAsset {
    url: String,
    filename: String,
}
impl WebAsset for WebTextureAsset {
    type Type = (TextureDescriptor, Vec<u8>);

    fn url(&self) -> String {
        self.url.clone()
    }
    fn filename(&self) -> String {
        self.filename.clone()
    }
    fn parse(&self, _context: &mut AssetLoadContext, data: Vec<u8>) -> Result<Self::Type, Error> {
        // TODO: handle other pixel formats
        let img = image::load_from_memory(&data)?.into_rgba();
        Ok((
            TextureDescriptor {
                format: TextureFormat::RGBA8,
                width: img.width(),
                height: img.height(),
                depth: 1,
                bytes: (*img).len(),
            },
            img.into_raw(),
        ))
    }
}
