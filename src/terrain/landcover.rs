use std::error::Error;
use std::io::{Cursor, Read};
use std::sync::{Arc, Mutex};

use zip::ZipArchive;
use image::{self, ColorType, DynamicImage, GenericImage, ImageDecoder, ImageLuma8, ImageFormat};
use image::png::PNGDecoder;

use cache::{AssetLoadContext, GeneratedAsset, WebAsset};
use terrain::raster::{GlobalRaster, Raster, RasterSource};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LandCoverKind {
    TreeCover,
    WaterMask,
}
impl RasterSource for LandCoverKind {
    type Type = u8;
    fn load(
        &self,
        context: &mut AssetLoadContext,
        latitude: i16,
        longitude: i16,
    ) -> Option<Raster<u8>> {
        Some(LandCoverParams {
            latitude,
            longitude,
            kind: *self,
            raw: None,
        }.load(context)
            .unwrap())
    }
}

/// Coordinates are of the lower left corner of the 10x10 degree cell.
#[derive(Debug, Eq, PartialEq)]
struct RawLandCoverParams {
    pub latitude: i16,
    pub longitude: i16,
    pub kind: LandCoverKind,
}
impl WebAsset for RawLandCoverParams {
    type Type = Arc<Mutex<DynamicImage>>;

    fn url(&self) -> String {
        let (latitude, longitude) = (self.latitude + 10, self.longitude);
        assert_eq!(latitude % 10, 0);
        assert_eq!(longitude % 10, 0);
        let n_or_s = if latitude >= 0 { 'N' } else { 'S' };
        let e_or_w = if longitude >= 0 { 'E' } else { 'W' };

        match self.kind {
            LandCoverKind::TreeCover => {
                format!("https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/gtc/downloads/\
                         treecover2010_v3_individual/{:02}{}_{:03}{}_treecover2010_v3.tif.zip",
                latitude.abs(),
                n_or_s,
                longitude.abs(),
                e_or_w,
            )
            }
            LandCoverKind::WaterMask => {
                format!("https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/gtc/downloads/\
                         WaterMask2010_UMD_individual/Hansen_GFC2013_datamask_{:02}{}_{:03}{}\
                         .tif.zip",
                latitude.abs(),
                n_or_s,
                longitude.abs(),
                e_or_w,
            )
            }
        }
    }
    fn filename(&self) -> String {
        let (latitude, longitude) = (self.latitude + 10, self.longitude);
        assert_eq!(latitude % 10, 0);
        assert_eq!(longitude % 10, 0);
        let n_or_s = if latitude >= 0 { 'N' } else { 'S' };
        let e_or_w = if longitude >= 0 { 'E' } else { 'W' };

        match self.kind {
            LandCoverKind::TreeCover => {
                format!("treecover/raw/{:02}{}_{:03}{}_treecover2010_v3.tif.zip",
                        latitude.abs(),
                        n_or_s,
                        longitude.abs(),
                        e_or_w,
                )
            }
            LandCoverKind::WaterMask => {
                format!("watermask/raw/Hansen_GFC2013_datamask_{:02}{}_{:03}{}.tif.zip",
                        latitude.abs(),
                        n_or_s,
                        longitude.abs(),
                        e_or_w,
                )
            }
        }
    }
    fn parse(
        &self,
        context: &mut AssetLoadContext,
        data: Vec<u8>,
    ) -> Result<Self::Type, Box<::std::error::Error>> {
        let mut zip = ZipArchive::new(Cursor::new(data))?;
        assert_eq!(zip.len(), 1);

        let mut data = Vec::new();
        zip.by_index(0)?.read_to_end(&mut data)?;

        let image = image::load_from_memory_with_format(&data[..], ImageFormat::TIFF)?;
        let image = Arc::new(Mutex::new(image));

        for latitude in self.latitude..(self.latitude + 10) {
            for longitude in self.longitude..(self.longitude + 10) {
                let params = LandCoverParams {
                    latitude,
                    longitude,
                    kind: self.kind,
                    raw: Some(image.clone()),
                };
                assert_eq!(params.raw_params(), *self);
                params.load(context)?;
            }
        }

        Ok(image)
    }
}

pub struct LandCoverParams {
    pub latitude: i16,
    pub longitude: i16,
    pub kind: LandCoverKind,
    pub raw: Option<Arc<Mutex<DynamicImage>>>,
}
impl LandCoverParams {
    fn raw_params(&self) -> RawLandCoverParams {
        RawLandCoverParams {
            latitude: self.latitude - ((self.latitude % 10) + 10) % 10,
            longitude: self.longitude - ((self.longitude % 10) + 10) % 10,
            kind: self.kind,
        }
    }

    fn generate_from_raw(&self) -> Result<Raster<u8>, Box<Error>> {
        let (w, h, image) = {
            let mut image = self.raw.as_ref().unwrap().lock().unwrap();
            let (w, h) = (image.width(), image.height());
            assert_eq!(w, h);
            let image = image
                .crop(
                    (w / 10) * ((self.longitude % 10 + 10) % 10) as u32,
                    (h / 10) * (9 - ((self.latitude % 10 + 10) % 10) as u32),
                    w / 10 + 1,
                    h / 10 + 1,
                )
                .clone();
            (w, h, image)
        };
        let image = image.rotate90().flipv();
        let values = if let ImageLuma8(image) = image {
            image.into_raw().into_iter()
        } else {
            unreachable!()
        };

        let values = match self.kind {
            LandCoverKind::TreeCover => values.collect(),
            LandCoverKind::WaterMask => {
                values
                    .map(|v| if v == 1 {
                        0
                    } else if v == 0 || v == 2 {
                        255
                    } else {
                        unreachable!()
                    })
                    .collect()
            }
        };

        Ok(Raster {
            width: w as usize / 10 + 1,
            height: h as usize / 10 + 1,
            cell_size: 10.0 / (w - 1) as f64,
            xllcorner: self.latitude as f64,
            yllcorner: self.longitude as f64,
            values,
        })
    }
}
impl GeneratedAsset for LandCoverParams {
    type Type = Raster<u8>;

    fn filename(&self) -> String {
        let n_or_s = if self.latitude >= 0 { 'N' } else { 'S' };
        let e_or_w = if self.longitude >= 0 { 'E' } else { 'W' };

        let directory = match self.kind {
            LandCoverKind::TreeCover => "treecover/processed",
            LandCoverKind::WaterMask => "watermask/processed",
        };

        format!("{}/{:02}{}_{:03}{}.raster",
                directory,
                self.latitude.abs(),
                n_or_s,
                self.longitude.abs(),
                e_or_w,
        )
    }

    fn generate(&self, context: &mut AssetLoadContext) -> Result<Self::Type, Box<Error>> {
        if self.raw.is_some() {
            return self.generate_from_raw();
        }

        Self {
            latitude: self.latitude,
            longitude: self.longitude,
            kind: self.kind,
            raw: Some(self.raw_params().load(context)?),
        }.generate_from_raw()
    }
}

pub struct BlueMarble;
impl WebAsset for BlueMarble {
    type Type = GlobalRaster<u8>;

    fn url(&self) -> String {
        "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74218/\
         world.200412.3x21600x10800.png"
            .to_owned()
    }
    fn filename(&self) -> String {
        "bluemarble/world.200412.3x21600x10800.png".to_owned()
    }
    fn parse(
        &self,
        context: &mut AssetLoadContext,
        data: Vec<u8>,
    ) -> Result<Self::Type, Box<::std::error::Error>> {
        let mut decoder = PNGDecoder::new(Cursor::new(data));
        let (width, height) = decoder.dimensions()?;
        let (width, height) = (width as usize, height as usize);
        assert_eq!(decoder.colortype()?, ColorType::RGB(8));

        context.set_progress_and_total(0, height);
        let row_len = decoder.row_len()?;
        let mut values = vec![0; row_len * height];
        for row in 0..height {
            decoder.read_scanline(&mut values[(row * row_len)..((row+1) * row_len)])?;
            context.set_progress(row);
        }

        Ok(GlobalRaster {
            width,
            height,
            bands: 3,
            values,
        })
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
            }.raw_params(),
            RawLandCoverParams {
                latitude: 160,
                longitude: 30,
                kind: LandCoverKind::TreeCover,
            }
        );

        assert_eq!(
            LandCoverParams {
                latitude: 20,
                longitude: 20,
                kind: LandCoverKind::TreeCover,
                raw: None,
            }.raw_params(),
            RawLandCoverParams {
                latitude: 20,
                longitude: 20,
                kind: LandCoverKind::TreeCover,
            }
        );

        assert_eq!(
            LandCoverParams {
                latitude: -18,
                longitude: -18,
                kind: LandCoverKind::TreeCover,
                raw: None,
            }.raw_params(),
            RawLandCoverParams {
                latitude: -20,
                longitude: -20,
                kind: LandCoverKind::TreeCover,
            }
        );

        assert_eq!(
            LandCoverParams {
                latitude: -30,
                longitude: -30,
                kind: LandCoverKind::TreeCover,
                raw: None,
            }.raw_params(),
            RawLandCoverParams {
                latitude: -30,
                longitude: -30,
                kind: LandCoverKind::TreeCover,
            }
        );
    }
}
