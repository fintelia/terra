use std::io::{Cursor, Read};

use anyhow::Error;
use image::png::PngDecoder;
use image::{self, ColorType, DynamicImage, GenericImageView, ImageDecoder, ImageFormat};
use memmap::Mmap;
use zip::ZipArchive;

use crate::cache::{AssetLoadContext, WebAsset};
use crate::terrain::raster::{
    BitContainer, GlobalRaster, MMappedRasterHeader, Raster, RasterSource,
};

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

pub struct GlobalWaterMask;
impl WebAsset for GlobalWaterMask {
    type Type = GlobalRaster<u8, BitContainer>;

    fn url(&self) -> String {
        "https://landcover.usgs.gov/documents/GlobalLandCover_tif.zip".to_owned()
    }
    fn filename(&self) -> String {
        "watermask/GlobalLandCover_tif.zip".to_owned()
    }
    fn parse(&self, context: &mut AssetLoadContext, data: Vec<u8>) -> Result<Self::Type, Error> {
        context.set_progress_and_total(0, 100);
        let mut zip = ZipArchive::new(Cursor::new(data))?;
        assert_eq!(zip.len(), 1);

        let mut data = Vec::new();
        zip.by_index(0)?.read_to_end(&mut data)?;

        let image = image::load_from_memory_with_format(&data[..], ImageFormat::Tiff)?;
        context.set_progress(100);
        let (width, height) = image.dimensions();
        let (width, height) = (width as usize, height as usize);
        if let DynamicImage::ImageLuma8(image) = image {
            Ok(GlobalRaster {
                width,
                height,
                bands: 1,
                values: BitContainer(image.into_raw().into_iter().map(|v| v == 0).collect()),
            })
        } else {
            unreachable!()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_raw_params() {
        assert_eq!(
            LandCoverParams {
                latitude: 165,
                longitude: 31,
                kind: LandCoverKind::TreeCover,
                raw: None,
            }
            .raw_params(),
            RawLandCoverParams { latitude: 160, longitude: 30, kind: LandCoverKind::TreeCover }
        );

        assert_eq!(
            LandCoverParams {
                latitude: 20,
                longitude: 20,
                kind: LandCoverKind::TreeCover,
                raw: None,
            }
            .raw_params(),
            RawLandCoverParams { latitude: 20, longitude: 20, kind: LandCoverKind::TreeCover }
        );

        assert_eq!(
            LandCoverParams {
                latitude: -18,
                longitude: -18,
                kind: LandCoverKind::TreeCover,
                raw: None,
            }
            .raw_params(),
            RawLandCoverParams { latitude: -20, longitude: -20, kind: LandCoverKind::TreeCover }
        );

        assert_eq!(
            LandCoverParams {
                latitude: -30,
                longitude: -30,
                kind: LandCoverKind::TreeCover,
                raw: None,
            }
            .raw_params(),
            RawLandCoverParams { latitude: -30, longitude: -30, kind: LandCoverKind::TreeCover }
        );
    }
}
