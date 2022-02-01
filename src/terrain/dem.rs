use crate::terrain::raster::{GlobalRaster, Raster, RasterCache};
use anyhow::{ensure, Error};
use lazy_static::lazy_static;
use std::io::{Cursor, Read};
use std::str::FromStr;
use std::{collections::HashSet, path::Path};
use thiserror::Error;
use zip::ZipArchive;

#[derive(Debug, Error)]
#[error("failed to parse DEM file")]
pub struct DemParseError;

lazy_static! {
    static ref SRTM3_FILES: HashSet<&'static str> =
        include_str!("../../file_list_srtm3.txt").split('\n').collect();
}

lazy_static! {
    static ref NASADEM_FILES: HashSet<&'static str> =
        include_str!("../../file_list_nasadem.txt").split('\n').collect();
    static ref TREECOVER_FILES: HashSet<&'static str> =
        include_str!("../../file_list_treecover.txt").split('\n').collect();
}

// /// Which data source to use for digital elevation models.
// #[derive(Clone)]
// pub enum DemSource {
//     /// Use DEMs Shuttle Radar Topography Mission (SRTM) 3 Arc-Second Global data source. Data is
//     /// available globally between 60° north and 56° south latitude.
//     #[allow(unused)]
//     Srtm90m(PathBuf),
//     /// Use NASADEM
//     #[allow(unused)]
//     Nasadem(PathBuf),
// }
// impl DemSource {
//     #[allow(unused)]
//     pub(crate) fn url_str(&self) -> &str {
//         match *self {
//             DemSource::Srtm90m(_) => {
//                 "https://opentopography.s3.sdsc.edu/raster/SRTM_GL3/SRTM_GL3_srtm/"
//             }
//             DemSource::Nasadem(_) => {
//                 "https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11/NASADEM_HGT_"
//             }
//         }
//     }

//     /// Returns the approximate resolution of data from this source in meters.
//     #[allow(unused)]
//     pub(crate) fn resolution(&self) -> u32 {
//         match *self {
//             DemSource::Srtm90m(_) => 90,
//             DemSource::Nasadem(_) => 30,
//         }
//     }
//     /// Returns the size of cells from this data source in arcseconds.
//     #[allow(unused)]
//     pub(crate) fn cell_size(&self) -> f32 {
//         match *self {
//             DemSource::Srtm90m(_) => 3.0,
//             DemSource::Nasadem(_) => 1.0,
//         }
//     }

//     fn tile_name(&self, latitude: i16, longitude: i16) -> String {
//         let n_or_s = if latitude >= 0 { 'n' } else { 's' };
//         let e_or_w = if longitude >= 0 { 'e' } else { 'w' };
//         match *self {
//             DemSource::Srtm90m(_) => {
//                 format!("{}{:02}_{}{:03}.hgt.sz", n_or_s, latitude.abs(), e_or_w, longitude.abs())
//             }
//             DemSource::Nasadem(_) => {
//                 format!(
//                     "NASADEM_HGT_{}{:02}{}{:03}.zip",
//                     n_or_s,
//                     latitude.abs(),
//                     e_or_w,
//                     longitude.abs()
//                 )
//             }
//         }
//     }

//     pub(crate) fn tile_should_exist(&self, latitude: i16, longitude: i16) -> bool {
//         match *self {
//             DemSource::Srtm90m(_) => SRTM3_FILES.contains(&*self.tile_name(latitude, longitude)),
//             DemSource::Nasadem(_) => NASADEM_FILES.contains(&*self.tile_name(latitude, longitude)),
//         }
//     }
//     pub(crate) fn filename(&self, latitude: i16, longitude: i16) -> PathBuf {
//         match self {
//             DemSource::Srtm90m(p) | DemSource::Nasadem(p) => {
//                 p.join(self.tile_name(latitude, longitude))
//             }
//         }
//     }
// }

pub(crate) fn make_nasadem_raster_cache(
    base_directory: &Path,
    capacity: usize,
) -> RasterCache<f32> {
    let mut filenames = Vec::new();
    for latitude in -90..=89i16 {
        for longitude in -180..=179i16 {
            let n_or_s = if latitude >= 0 { 'n' } else { 's' };
            let e_or_w = if longitude >= 0 { 'e' } else { 'w' };
            let filename = format!(
                "NASADEM_HGT_{}{:02}{}{:03}.zip",
                n_or_s,
                latitude.abs(),
                e_or_w,
                longitude.abs()
            );
            if NASADEM_FILES.contains(&*filename) {
                filenames.push(Some(base_directory.join(filename)));
            } else {
                filenames.push(None);
            }
        }
    }

    RasterCache::new(filenames.into_boxed_slice(), 1, capacity, &parse_nasadem)
}

pub(crate) fn make_treecover_raster_cache(
    base_directory: &Path,
    capacity: usize,
) -> RasterCache<u8> {
    let mut filenames = Vec::new();
    for latitude in (-80..=90i16).step_by(10) {
        for longitude in (-180..=179i16).step_by(10) {
            let n_or_s = if latitude >= 0 { 'N' } else { 'S' };
            let e_or_w = if longitude >= 0 { 'E' } else { 'W' };
            let filename = format!(
                "Hansen_GFC-2020-v1.8_treecover2000_{:02}{}_{:03}{}.tif",
                latitude.abs(),
                n_or_s,
                longitude.abs(),
                e_or_w,
            );
            if TREECOVER_FILES.contains(&*filename) {
                filenames.push(Some(base_directory.join(filename)));
            } else {
                filenames.push(None);
            }
        }
    }
    assert_eq!(filenames.len(), 18 * 36);

    RasterCache::new(filenames.into_boxed_slice(), 10, capacity, &|lat, long, data| {
        let mut decoder = tiff::decoder::Decoder::new(Cursor::new(data))?
            .with_limits(tiff::decoder::Limits::unlimited());
        let (width, height) = decoder.dimensions()?;

        let values = match decoder.read_image() {
            Ok(tiff::decoder::DecodingResult::U8(data)) => data,
            e => {
                println!("Bad treecover file lat={} long={} ({:?})", lat, long, e);
                anyhow::bail!("bad treecover file");
                vec![0; width as usize * height as usize]
            }
        };

        Ok(Raster {
            width: width as usize,
            height: height as usize,
            bands: 1,
            cell_size: 0.00025,
            latitude_llcorner: f64::from(lat),
            longitude_llcorner: f64::from(long),
            values,
        })
    })
}

// #[async_trait::async_trait]
// impl RasterSource for DemSource {
//     type Type = f32;
//     type Container = Vec<f32>;
//     async fn load(&self, latitude: i16, longitude: i16) -> Result<Option<Raster<f32>>, Error> {
//         if !self.tile_should_exist(latitude, longitude) {
//             return Ok(None);
//         }

//         match self {
//             DemSource::Srtm90m(_) => {
//                 let filename = self.filename(latitude, longitude);
//                 let data = tokio::fs::read(filename).await?;
//                 let mut uncompressed = Vec::new();
//                 snap::read::FrameDecoder::new(Cursor::new(data)).read_to_end(&mut uncompressed)?;
//                 parse_srtm3_hgt(latitude, longitude, uncompressed).map(Some)
//             }
//             DemSource::Nasadem(_) => {
//                 let filename = self.filename(latitude, longitude);
//                 let data = tokio::fs::read(filename).await?;

//                 tokio::task::spawn_blocking(move || {
//                     parse_nasadem(latitude, longitude, data).map(Some)
//                 })
//                 .await?
//             }
//         }
//     }
//     fn bands(&self) -> usize {
//         1
//     }
// }

/// Load a zip file in the format for the USGS's National Elevation Dataset.
#[allow(unused)]
fn parse_ned_zip(data: Vec<u8>) -> Result<Raster<f32>, Error> {
    let mut hdr = String::new();
    let mut flt = Vec::new();

    let mut zip = ZipArchive::new(Cursor::new(data))?;
    for i in 0..zip.len() {
        let mut file = zip.by_index(i)?;
        if file.name().ends_with(".hdr") {
            assert_eq!(hdr.len(), 0);
            file.read_to_string(&mut hdr)?;
        } else if file.name().ends_with(".flt") {
            assert_eq!(flt.len(), 0);
            file.read_to_end(&mut flt)?;
        }
    }

    enum ByteOrder {
        LsbFirst,
        MsbFirst,
    }

    let mut width = None;
    let mut height = None;
    let mut xllcorner = None;
    let mut yllcorner = None;
    let mut cell_size = None;
    let mut byte_order = None;
    let mut nodata_value = None;
    for line in hdr.lines() {
        let mut parts = line.split_whitespace();
        let key = parts.next();
        let value = parts.next();
        if let (Some(key), Some(value)) = (key, value) {
            match key {
                "ncols" => width = usize::from_str(value).ok(),
                "nrows" => height = usize::from_str(value).ok(),
                "xllcorner" => xllcorner = f64::from_str(value).ok(),
                "yllcorner" => yllcorner = f64::from_str(value).ok(),
                "cellsize" => cell_size = f64::from_str(value).ok(),
                "NODATA_value" => nodata_value = f32::from_str(value).ok(),
                "byteorder" => {
                    byte_order = match value {
                        "LSBFIRST" => Some(ByteOrder::LsbFirst),
                        "MSBFIRST" => Some(ByteOrder::MsbFirst),
                        _ => panic!("unrecognized byte order: {}", value),
                    }
                }
                _ => {}
            }
        }
    }

    let width = width.ok_or(DemParseError)?;
    let height = height.ok_or(DemParseError)?;
    let xllcorner = xllcorner.ok_or(DemParseError)?;
    let yllcorner = yllcorner.ok_or(DemParseError)?;
    let cell_size = cell_size.ok_or(DemParseError)?;
    let byte_order = byte_order.ok_or(DemParseError)?;
    let nodata_value = nodata_value.ok_or(DemParseError)?;

    let size = width * height;
    if flt.len() != size * 4 {
        Err(DemParseError)?;
    }

    let flt: &[u32] = bytemuck::cast_slice(&flt[..]);
    let mut elevations: Vec<f32> = Vec::with_capacity(size);
    for f in flt {
        let e = bytemuck::cast(match byte_order {
            ByteOrder::LsbFirst => f.to_le(),
            ByteOrder::MsbFirst => f.to_be(),
        });
        elevations.push(if e == nodata_value { 0.0 } else { e });
    }

    Ok(Raster {
        width,
        height,
        bands: 1,
        latitude_llcorner: xllcorner,
        longitude_llcorner: yllcorner,
        cell_size,
        values: elevations,
    })
}

/// Load a HGT file in the format for the NASA's STRM 90m dataset.
fn parse_srtm3_hgt(latitude: i16, longitude: i16, hgt: Vec<u8>) -> Result<Raster<f32>, Error> {
    let resolution = 1201;
    let cell_size = 1.0 / 1200.0;

    if hgt.len() != resolution * resolution * 2 {
        Err(DemParseError)?;
    }

    let hgt = bytemuck::cast_slice(&hgt[..]);
    let mut elevations: Vec<f32> = Vec::with_capacity(resolution * resolution);

    for y in 0..resolution {
        for x in 0..resolution {
            let h = i16::from_be(hgt[x + y * resolution]);
            if h == -32768 {
                elevations.push(0.0);
            } else {
                elevations.push(h as f32);
            }
        }
    }

    Ok(Raster {
        width: resolution,
        height: resolution,
        bands: 1,
        latitude_llcorner: latitude as f64,
        longitude_llcorner: longitude as f64,
        cell_size,
        values: elevations,
    })
}

/// Load a ZIP file containing a HGT file in the format for the NASA's nasadem 30m dataset.
fn parse_nasadem(latitude: i16, longitude: i16, data: &[u8]) -> Result<Raster<f32>, Error> {
    let resolution = 3601;
    let cell_size = 1.0 / 3600.0;

    let mut zip = ZipArchive::new(Cursor::new(data))?;
    ensure!(zip.len() == 3, "Unexpected zip file contents");

    let filename = zip.file_names().find(|name| name.ends_with(".hgt")).map(str::to_owned);
    ensure!(filename.is_some(), "Zip doesn't contain .hgt");
    let mut file = zip.by_name(&filename.unwrap())?;

    let mut hgt = Vec::with_capacity(resolution * resolution * 2);
    file.read_to_end(&mut hgt)?;
    assert_eq!(hgt.len(), resolution * resolution * 2);
    if hgt.len() != resolution * resolution * 2 {
        Err(DemParseError)?;
    }

    let hgt = bytemuck::cast_slice(&hgt[..]);
    let mut elevations: Vec<f32> = Vec::with_capacity(resolution * resolution);

    for y in 0..resolution {
        for x in 0..resolution {
            let h = i16::from_be(hgt[x + y * resolution]);
            if h == -32768 {
                elevations.push(0.0);
            } else {
                elevations.push(h as f32);
            }
        }
    }

    Ok(Raster {
        width: resolution,
        height: resolution,
        bands: 1,
        latitude_llcorner: latitude as f64,
        longitude_llcorner: longitude as f64,
        cell_size,
        values: elevations,
    })
}

pub(crate) fn parse_etopo1(
    filename: impl AsRef<Path>,
    mut progress_callback: impl FnMut(&str, usize, usize) + Send,
) -> Result<GlobalRaster<i16>, Error> {
    let data = std::fs::read(filename)?;

    let mut zip = ZipArchive::new(Cursor::new(data))?;
    ensure!(zip.len() == 1, "Unexpected zip file contents");
    let mut file = zip.by_index(0)?;
    ensure!(file.name() == "ETOPO1_Ice_c_geotiff.tif", "Unexpected zip file contents");

    let mut contents = vec![0; file.size() as usize];
    for (i, chunk) in contents.chunks_mut(1024 * 1024).enumerate() {
        progress_callback(
            "Decompressing ETOPO1_Ice_c_geotiff.tif...",
            (i * 1024 * 1024) >> 10,
            file.size() as usize >> 10,
        );
        file.read_exact(chunk)?;
    }
    progress_callback(
        "Decompressing ETOPO1_Ice_c_geotiff.tif...",
        file.size() as usize >> 10,
        file.size() as usize >> 10,
    );

    let mut tiff_decoder = tiff::decoder::Decoder::new(Cursor::new(contents))?;
    let (width, height) = tiff_decoder.dimensions()?;

    let mut offset = 0;
    let mut values: Vec<i16> = vec![0; width as usize * height as usize];
    let strip_count = tiff_decoder.strip_count()?;

    for i in 0..strip_count {
        if i % 8 == 0 {
            progress_callback(
                "Decoding ETOPO1_Ice_c_geotiff.tif...",
                i as usize,
                strip_count as usize,
            );
        }
        if let tiff::decoder::DecodingResult::I16(v) = tiff_decoder.read_strip()? {
            values[offset..][..v.len()].copy_from_slice(&v);
            offset += v.len();
        } else {
            unreachable!();
        }
    }
    progress_callback(
        "Decoding ETOPO1_Ice_c_geotiff.tif...",
        strip_count as usize,
        strip_count as usize,
    );

    Ok(GlobalRaster { bands: 1, width: width as usize, height: height as usize, values })
}
