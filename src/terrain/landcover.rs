use crate::cache::{AssetLoadContext, CompressionType, WebAsset};
use crate::terrain::raster::{GlobalRaster, MMappedRasterHeader, Raster, RasterSource};
use anyhow::Error;
use image::png::PngDecoder;
use image::tiff::TiffDecoder;
use image::{self, ColorType, ImageDecoder};
use memmap::Mmap;
use std::collections::HashSet;
use std::io::{Cursor, Read};

lazy_static! {
    static ref GFC_DATAMASK_FILES: HashSet<&'static str> =
        include_str!("../../file_list_GFC2019_datamask.txt").split('\n').collect();
}

pub struct BlueMarble;
impl WebAsset for BlueMarble {
    type Type = GlobalRaster<u8>;

    fn url(&self) -> String {
        "https://eoimages.gsfc.nasa.gov/images/imagerecords/76000/76487/\
         world.200406.3x21600x10800.png"
            .to_owned()
    }
    fn filename(&self) -> String {
        "bluemarble/world.200406.3x21600x10800.png".to_owned()
    }
    fn parse(&self, context: &mut AssetLoadContext, data: Vec<u8>) -> Result<Self::Type, Error> {
        let decoder = PngDecoder::new(Cursor::new(data))?;
        let (width, height) = decoder.dimensions();
        let (width, height) = (width as usize, height as usize);
        assert_eq!(decoder.color_type(), ColorType::Rgb8);

        context.set_progress_and_total(0, height / 108);
        let row_len = width * 3;
        let mut values = vec![0; decoder.total_bytes() as usize];
        let mut reader = decoder.into_reader()?;
        for row in 0..height {
            reader.read_exact(&mut values[(row * row_len)..((row + 1) * row_len)])?;
            if (row + 1) % 108 == 0 {
                context.set_progress((row + 1) / 108);
            }
        }

        Ok(GlobalRaster { width, height, bands: 3, values })
    }
}

pub struct BlueMarbleTile {
    latitude_llcorner: i16,
    longitude_llcorner: i16,
}
impl BlueMarbleTile {
    fn name(&self) -> String {
        let x = match self.longitude_llcorner {
            -180 => "A",
            -90 => "B",
            0 => "C",
            90 => "D",
            _ => unreachable!(),
        };
        let y = match self.latitude_llcorner {
            0 => "1",
            -90 => "2",
            _ => unreachable!(),
        };
        format!("world.200406.3x21600x21600.{}{}.png", x, y)
    }
}
impl WebAsset for BlueMarbleTile {
    type Type = (MMappedRasterHeader, Vec<u8>);

    fn url(&self) -> String {
        format!("https://eoimages.gsfc.nasa.gov/images/imagerecords/76000/76487/{}", self.name())
    }
    fn filename(&self) -> String {
        format!("bluemarble/{}", self.name())
    }
    fn parse(&self, context: &mut AssetLoadContext, data: Vec<u8>) -> Result<Self::Type, Error> {
        let decoder = PngDecoder::new(Cursor::new(data))?;
        let (width, height) = decoder.dimensions();
        let (width, height) = (width as usize, height as usize);
        assert_eq!(decoder.color_type(), ColorType::Rgb8);

        context.set_progress_and_total(0, height / 108);
        let row_len = width * 3;
        let mut values = vec![0; decoder.total_bytes() as usize];
        let mut reader = decoder.into_reader()?;
        for row in 0..height {
            reader.read_exact(&mut values[(row * row_len)..][..row_len])?;
            if (row + 1) % 108 == 0 {
                context.set_progress((row + 1) / 108);
            }
        }

        Ok((
            MMappedRasterHeader {
                width,
                height,
                bands: 3,
                cell_size: 90.0 / 21600.0,
                latitude_llcorner: self.latitude_llcorner as f64,
                longitude_llcorner: self.longitude_llcorner as f64,
            },
            values,
        ))
    }
}

pub struct BlueMarbleTileSource;
impl RasterSource for BlueMarbleTileSource {
    type Type = u8;
    type Container = Mmap;
    fn bands(&self) -> usize {
        3
    }
    fn raster_size(&self) -> i16 {
        90
    }
    fn load(
        &self,
        context: &mut AssetLoadContext,
        latitude: i16,
        longitude: i16,
    ) -> Option<Raster<Self::Type, Self::Container>> {
        Some(
            Raster::from_mmapped_raster(
                BlueMarbleTile { latitude_llcorner: latitude, longitude_llcorner: longitude },
                context,
            )
            .unwrap(),
        )
    }
}

pub struct GfcWaterTile {
    latitude_llcorner: i16,
    longitude_llcorner: i16,
}
impl GfcWaterTile {
    fn name(&self) -> String {
        let n_or_s = if self.latitude_llcorner >= 0 { 'N' } else { 'S' };
        let e_or_w = if self.longitude_llcorner >= 0 { 'E' } else { 'W' };

        format!(
            "GFC-2019-v1.7_datamask_{:02}{}_{:03}{}.tif",
            self.latitude_llcorner.abs(),
            n_or_s,
            self.longitude_llcorner.abs(),
            e_or_w
        )
    }
}
impl WebAsset for GfcWaterTile {
    type Type = Vec<u8>;

    fn url(&self) -> String {
        format!(
            "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2019-v1.7/Hansen_{}",
            self.name()
        )
    }
    fn filename(&self) -> String {
        format!("landcover/datamask/{}.lz4", self.name())
    }
    fn compression(&self) -> CompressionType {
        CompressionType::Lz4
    }
    fn parse(&self, context: &mut AssetLoadContext, data: Vec<u8>) -> Result<Self::Type, Error> {
        let decoder = TiffDecoder::new(Cursor::new(data))?;
        let (width, height) = decoder.dimensions();
        let (width, height) = (width as usize, height as usize);
        assert_eq!(decoder.color_type(), ColorType::L8);

        context.set_progress_and_total(0, height / 100);
        let mut values = vec![0; decoder.total_bytes() as usize];
        let mut reader = decoder.into_reader()?;
        for row in 0..height {
            reader.read_exact(&mut values[(row * width)..][..width])?;
            if (row + 1) % 100 == 0 {
                context.set_progress((row + 1) / 108);
            }
        }

        Ok(values)
    }
}
pub struct GfcWaterTileSource;
impl RasterSource for GfcWaterTileSource {
    type Type = u8;
    type Container = Vec<u8>;
    fn bands(&self) -> usize {
        1
    }
    fn raster_size(&self) -> i16 {
        10
    }
    fn load(
        &self,
        context: &mut AssetLoadContext,
        latitude: i16,
        longitude: i16,
    ) -> Option<Raster<Self::Type, Self::Container>> {
        let tile = GfcWaterTile { latitude_llcorner: latitude, longitude_llcorner: longitude };
        if !GFC_DATAMASK_FILES.contains(&*tile.url()) {
            return None;
        }
        Some(Raster {
            width: 40000,
            height: 40000,
            bands: 1,
            cell_size: 1.0 / (60.0 * 60.0),
            latitude_llcorner: latitude as f64,
            longitude_llcorner: longitude as f64,
            values: tile.load(context).ok()?,
        })
    }
}

pub struct GfcTreeDensityTile {
    latitude_llcorner: i16,
    longitude_llcorner: i16,
}
impl GfcTreeDensityTile {
    fn name(&self) -> String {
        let n_or_s = if self.latitude_llcorner >= 0 { 'N' } else { 'S' };
        let e_or_w = if self.longitude_llcorner >= 0 { 'E' } else { 'W' };

        format!(
            "GFC-2019-v1.7_treecover2000_{:02}{}_{:03}{}.tif",
            self.latitude_llcorner.abs(),
            n_or_s,
            self.longitude_llcorner.abs(),
            e_or_w
        )
    }
}
impl WebAsset for GfcTreeDensityTile {
    type Type = Vec<u8>;

    fn url(&self) -> String {
        format!(
            "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2019-v1.7/Hansen_{}",
            self.name()
        )
    }
    fn filename(&self) -> String {
        format!("landcover/treedensity2000/{}.lz4", self.name())
    }
    fn compression(&self) -> CompressionType {
        CompressionType::Lz4
    }
    fn parse(&self, context: &mut AssetLoadContext, data: Vec<u8>) -> Result<Self::Type, Error> {
        let decoder = TiffDecoder::new(Cursor::new(data))?;
        let (width, height) = decoder.dimensions();
        let (width, height) = (width as usize, height as usize);
        assert_eq!(decoder.color_type(), ColorType::L8);

        context.set_progress_and_total(0, height / 100);
        let mut values = vec![0; decoder.total_bytes() as usize];
        let mut reader = decoder.into_reader()?;
        for row in 0..height {
            reader.read_exact(&mut values[(row * width)..][..width])?;
            if (row + 1) % 100 == 0 {
                context.set_progress((row + 1) / 108);
            }
        }

        Ok(values)
    }
}
pub struct GfcTreeDensityTileSource;
impl RasterSource for GfcTreeDensityTileSource {
    type Type = u8;
    type Container = Vec<u8>;
    fn bands(&self) -> usize {
        1
    }
    fn raster_size(&self) -> i16 {
        10
    }
    fn load(
        &self,
        context: &mut AssetLoadContext,
        latitude: i16,
        longitude: i16,
    ) -> Option<Raster<Self::Type, Self::Container>> {
        let tile = GfcTreeDensityTile { latitude_llcorner: latitude, longitude_llcorner: longitude };
        Some(Raster {
            width: 40000,
            height: 40000,
            bands: 1,
            cell_size: 1.0 / (60.0 * 60.0),
            latitude_llcorner: latitude as f64,
            longitude_llcorner: longitude as f64,
            values: tile.load(context).ok()?,
        })
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn to_raw_params() {
//         assert_eq!(
//             LandCoverParams {
//                 latitude: 165,
//                 longitude: 31,
//                 kind: LandCoverKind::TreeCover,
//                 raw: None,
//             }
//             .raw_params(),
//             RawLandCoverParams { latitude: 160, longitude: 30, kind: LandCoverKind::TreeCover }
//         );

//         assert_eq!(
//             LandCoverParams {
//                 latitude: 20,
//                 longitude: 20,
//                 kind: LandCoverKind::TreeCover,
//                 raw: None,
//             }
//             .raw_params(),
//             RawLandCoverParams { latitude: 20, longitude: 20, kind: LandCoverKind::TreeCover }
//         );

//         assert_eq!(
//             LandCoverParams {
//                 latitude: -18,
//                 longitude: -18,
//                 kind: LandCoverKind::TreeCover,
//                 raw: None,
//             }
//             .raw_params(),
//             RawLandCoverParams { latitude: -20, longitude: -20, kind: LandCoverKind::TreeCover }
//         );

//         assert_eq!(
//             LandCoverParams {
//                 latitude: -30,
//                 longitude: -30,
//                 kind: LandCoverKind::TreeCover,
//                 raw: None,
//             }
//             .raw_params(),
//             RawLandCoverParams { latitude: -30, longitude: -30, kind: LandCoverKind::TreeCover }
//         );
//     }
// }
