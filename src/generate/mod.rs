use crate::cache::{AssetLoadContext, AssetLoadContextBuf, WebAsset};
use crate::coordinates::CoordinateSystem;
use crate::mapfile::MapFile;
use crate::srgb::SRGB_TO_LINEAR;
use crate::terrain::dem::GlobalDem;
use crate::terrain::heightmap;
use crate::terrain::landcover::{BlueMarble, BlueMarbleTileSource};
use crate::terrain::quadtree::VNode;
use crate::terrain::raster::RasterCache;
// use crate::terrain::reprojected_raster::{
//     DataType, RasterSource, ReprojectedDemDef, ReprojectedRaster, ReprojectedRasterDef,
// };
use crate::terrain::tile_cache::{
    LayerParams, LayerType, NoiseParams, TextureDescriptor, TextureFormat,
};
use anyhow::Error;
use byteorder::{LittleEndian, WriteBytesExt};
use maplit::hashmap;
// use rand;
// use rand::distributions::Distribution;
// use rand_distr::Normal;
// use std::cell::RefCell;
use std::f64::consts::PI;
// use std::io::Write;
// use std::rc::Rc;
use vec_map::VecMap;

mod gpu;
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
}
// use levels::*;

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
    pub fn build() -> Result<MapFile, Error> {
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
                    texture_format: TextureFormat::RG8,
                },
        ]
        .into_iter()
        .collect();

        let mut mapfile = MapFile::new(layers);
        VNode::breadth_first(|n| {
            mapfile.reload_tile_state(LayerType::Heightmaps, n, true).unwrap();
            n.level() < 3
        });
        VNode::breadth_first(|n| {
            mapfile.reload_tile_state(LayerType::Albedo, n, true).unwrap();
            n.level() < 5
        });
        VNode::breadth_first(|n| {
            mapfile.reload_tile_state(LayerType::Roughness, n, true).unwrap();
            false
        });

        let mut context = AssetLoadContextBuf::new();
        let mut context = context.context("Generating mapfile...", 5);
        generate_heightmaps(&mut mapfile, &mut context)?;
        context.set_progress(1);
        generate_albedo(&mut mapfile, &mut context)?;
        context.set_progress(2);
        generate_roughness(&mut mapfile, &mut context)?;
        context.set_progress(3);
        generate_noise(&mut mapfile)?;
        context.set_progress(4);

        if !mapfile.reload_texture("sky") {
            let sky = WebTextureAsset {
                url: "https://www.eso.org/public/archives/images/original/eso0932a.tif".to_owned(),
                filename: "eso0932a.tif".to_owned(),
            }
            .load(&mut context)?;
            mapfile.write_texture("sky", sky.0, &sky.1)?;
        }

        context.set_progress(5);
        Ok(mapfile)
    }
}

fn generate_heightmaps(mapfile: &mut MapFile, context: &mut AssetLoadContext) -> Result<(), Error> {
    let missing = mapfile.get_missing_base(LayerType::Heightmaps)?;
    if missing.is_empty() {
        return Ok(());
    }

    let layer = mapfile.layers()[LayerType::Heightmaps].clone();

    let global_dem = GlobalDem.load(context)?;

    let context = &mut context.increment_level("Writing heightmaps... ", missing.len());
    for (i, n) in missing.into_iter().enumerate() {
        context.set_progress(i as u64);
        let mut heightmap = Vec::new();
        for y in 0..layer.texture_resolution {
            for x in 0..layer.texture_resolution {
                let cspace = n.cell_position_cspace(
                    x as i32,
                    y as i32,
                    layer.texture_border_size as u16,
                    layer.texture_resolution as u16,
                );
                let sspace = CoordinateSystem::cspace_to_sspace(cspace);
                let polar = CoordinateSystem::sspace_to_polar(sspace);
                let (lat, long) = (polar.x.to_degrees(), polar.y.to_degrees());

                heightmap.write_f32::<LittleEndian>(global_dem.interpolate(lat, long, 0) as f32)?;
            }
        }
        mapfile.write_tile(LayerType::Heightmaps, n, &heightmap, true)?;
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

        for y in 0..layer.texture_resolution {
            for x in 0..layer.texture_resolution {
                let cspace = n.cell_position_cspace(
                    x as i32,
                    y as i32,
                    layer.texture_border_size as u16,
                    layer.texture_resolution as u16,
                );
                let sspace = CoordinateSystem::cspace_to_sspace(cspace);
                let polar = CoordinateSystem::sspace_to_polar(sspace);
                let (lat, long) = (polar.x.to_degrees(), polar.y.to_degrees());

                let color = if spacing < bluemarble_spacing {
                    let [r, g, b] =
                        bluemarble_cache.nearest3(context, lat, long).unwrap_or([255.0, 0.0, 0.0]);

                    [SRGB_TO_LINEAR[r as u8], SRGB_TO_LINEAR[g as u8], SRGB_TO_LINEAR[b as u8], 255]
                } else {
                    let r = bluemarble.interpolate(lat, long, 0) as u8;
                    let g = bluemarble.interpolate(lat, long, 1) as u8;
                    let b = bluemarble.interpolate(lat, long, 2) as u8;
                    [SRGB_TO_LINEAR[r], SRGB_TO_LINEAR[g], SRGB_TO_LINEAR[b], 255]
                };

                colormap.extend_from_slice(&color);
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

        let mut data =
            Vec::with_capacity(layer.texture_resolution as usize * layer.texture_resolution as usize / 2);
        for y in 0..(layer.texture_resolution / 4) {
            for x in 0..(layer.texture_resolution / 4) {
                data.extend_from_slice(&[179, 180, 0, 0, 0, 0, 0, 0]);
            }
        }

        mapfile.write_tile(LayerType::Roughness, n, &data, true)?;
    }

    Ok(())
}

fn generate_noise(mapfile: &mut MapFile) -> Result<(), Error> {
    if !mapfile.reload_texture("noise") {
        let noise = NoiseParams {
            texture: TextureDescriptor {
                width: 2048,
                height: 2048,
                format: TextureFormat::RGBA8,
                bytes: 4 * 2048 * 2048,
            },
            wavelength: 1.0 / 256.0,
        };

        let noise_heightmaps: Vec<_> =
            (0..4).map(|i| heightmap::wavelet_noise(64 << i, 32 >> i)).collect();

        let len = noise_heightmaps[0].heights.len();
        let mut heights = vec![0u8; len * 4];
        for (i, heightmap) in noise_heightmaps.into_iter().enumerate() {
            let mut dist: Vec<(usize, f32)> = heightmap.heights.into_iter().enumerate().collect();
            dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for j in 0..len {
                heights[dist[j].0 * 4 + i] = (j * 256 / len) as u8;
            }
        }

        mapfile.write_texture("noise", noise.texture, &heights[..])?;
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
                bytes: (*img).len(),
            },
            img.into_raw(),
        ))
    }
}
