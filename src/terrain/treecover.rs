use std::error::Error;
use std::io::{Cursor, Read};

use zip::ZipArchive;
use image::{self, GenericImage, ImageLuma8, ImageFormat};

use cache::{GeneratedAsset, WebAsset};
use terrain::raster::Raster;

/// Coordinates are of the lower left corner of the 10x10 degree cell.
#[derive(Debug, Eq, PartialEq)]
struct RawTreeCoverParams {
    pub latitude: i16,
    pub longitude: i16,
}
impl WebAsset for RawTreeCoverParams {
    type Type = Vec<u8>;

    fn url(&self) -> String {
        let (latitude, longitude) = (self.latitude + 10, self.longitude);
        assert_eq!(latitude % 10, 0);
        assert_eq!(longitude % 10, 0);
        let n_or_s = if latitude >= 0 { 'N' } else { 'S' };
        let e_or_w = if longitude >= 0 { 'E' } else { 'W' };

        let s = format!("https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/gtc/downloads/treecover2010_v3_individual/{:02}{}_{:03}{}_treecover2010_v3.tif.zip",
                latitude.abs(),
                n_or_s,
                longitude.abs(),
                e_or_w,
        );
        s
    }
    fn filename(&self) -> String {
        let (latitude, longitude) = (self.latitude + 10, self.longitude);
        assert_eq!(latitude % 10, 0);
        assert_eq!(longitude % 10, 0);
        let n_or_s = if latitude >= 0 { 'N' } else { 'S' };
        let e_or_w = if longitude >= 0 { 'E' } else { 'W' };

        format!("treecover/raw/{:02}{}_{:03}{}_treecover2010_v3.tif.zip",
                latitude.abs(),
                n_or_s,
                longitude.abs(),
                e_or_w,
        )
    }
    fn parse(&self, data: Vec<u8>) -> Result<Self::Type, Box<::std::error::Error>> {
        Ok(data)
    }
}

#[derive(Debug)]
pub struct TreeCoverParams {
    pub latitude: i16,
    pub longitude: i16,
}
impl TreeCoverParams {
    fn raw_params(&self) -> RawTreeCoverParams {
        RawTreeCoverParams {
            latitude: self.latitude - ((self.latitude % 10) + 10) % 10,
            longitude: self.longitude - ((self.longitude % 10) + 10) % 10,
        }
    }
}
impl GeneratedAsset for TreeCoverParams {
    type Type = Raster;

    fn filename(&self) -> String {
        let n_or_s = if self.latitude >= 0 { 'N' } else { 'S' };
        let e_or_w = if self.longitude >= 0 { 'E' } else { 'W' };

        format!("treecover/processed/{:02}{}_{:03}{}.raster",
                self.latitude.abs(),
                n_or_s,
                self.longitude.abs(),
                e_or_w,
        )
    }

    fn generate(&self) -> Result<Self::Type, Box<Error>> {
        let raw = self.raw_params().load().unwrap();
        let mut zip = ZipArchive::new(Cursor::new(raw))?;
        assert_eq!(zip.len(), 1);

        let mut data = Vec::new();
        zip.by_index(0)?.read_to_end(&mut data)?;

        let mut image = image::load_from_memory_with_format(&data[..], ImageFormat::TIFF).unwrap();
        let (w, h) = (image.width(), image.height());
        assert_eq!(w, h);

        let image = image
            .crop(
                (w / 10) * ((self.longitude % 10 + 10) % 10) as u32,
                (h / 10) * (9 - ((self.latitude % 10 + 10) % 10) as u32),
                w / 10 + 1,
                h / 10 + 1,
            )
            .rotate90()
            .flipv();
        let values = if let ImageLuma8(image) = image {
            image
                .into_raw()
                .into_iter()
                .map(|v| v as f32 / 255.0)
                .collect()
        } else {
            unreachable!()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_raw_params() {
        assert_eq!(
            TreeCoverParams {
                latitude: 165,
                longitude: 31,
            }.raw_params(),
            RawTreeCoverParams {
                latitude: 160,
                longitude: 30,
            }
        );

        assert_eq!(
            TreeCoverParams {
                latitude: 20,
                longitude: 20,
            }.raw_params(),
            RawTreeCoverParams {
                latitude: 20,
                longitude: 20,
            }
        );

        assert_eq!(
            TreeCoverParams {
                latitude: -18,
                longitude: -18,
            }.raw_params(),
            RawTreeCoverParams {
                latitude: -20,
                longitude: -20,
            }
        );

        assert_eq!(
            TreeCoverParams {
                latitude: -30,
                longitude: -30,
            }.raw_params(),
            RawTreeCoverParams {
                latitude: -30,
                longitude: -30,
            }
        );
    }
}
