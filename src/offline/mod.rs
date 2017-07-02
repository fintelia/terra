
use byteorder::{LittleEndian, BigEndian, ReadBytesExt};
use curl::easy::Easy;
use zip::ZipArchive;

use std::io::{Read, Cursor, Seek};
use std::str::FromStr;

pub enum DemSource {
    Usgs30m,
    Usgs10m,
}

pub struct Dem {
    pub width: usize,
    pub height: usize,
    pub cell_size: f64,

    pub xllcorner: f64,
    pub yllcorner: f64,

    pub elevations: Vec<f32>,
}

enum ByteOrder {
    LsbFirst,
    MsbFirst,
}

#[cfg(feature = "download")]
pub fn download_dem(latitude: i16, longitude: i16, source: DemSource) -> Cursor<Vec<u8>> {
    // Also see:
    // https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Elevation/1/ArcGrid/

    let resolution = match source {
        DemSource::Usgs30m => "1",
        DemSource::Usgs10m => "13",
    };
    let n_or_s = if latitude >= 0 { 'n' } else { 's' };
    let e_or_w = if longitude >= 0 { 'e' } else { 'w' };
    let url = format!("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/{}/GridFloat/\
                       USGS_NED_{}_{}{:02}{}{:03}_GridFloat.zip",
                      resolution,
                      resolution,
                      n_or_s,
                      latitude.abs(),
                      e_or_w,
                      longitude.abs());

    let mut data = Vec::<u8>::new();
    {
        let mut easy = Easy::new();
        easy.url(&url).unwrap();
        let mut easy = easy.transfer();
        easy.write_function(|d| {
                                let len = d.len();
                                data.extend(d);
                                Ok(len)
                            })
            .unwrap();
        easy.perform().unwrap();
    }

    Cursor::new(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let dem = parse_dem(download_dem(28, -81, DemSource::Usgs30m));
        assert_eq!(dem.width, 3612);
        assert_eq!(dem.height, 3612);
        assert!(dem.cell_size > 0.0002777);
        assert!(dem.cell_size < 0.0002778);
    }
}
