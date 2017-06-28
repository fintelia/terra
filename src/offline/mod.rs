
use byteorder::{LittleEndian, BigEndian, ReadBytesExt};
use curl::easy::Easy;
use zip::ZipArchive;

use std::io::{Read, Cursor};
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
pub fn download_dem(latitude: i16, longitude: i16, source: DemSource) -> Dem {
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

    let mut hdr = String::new();
    let mut flt = Vec::new();

    let reader = Cursor::new(&data[..]);
    let mut zip = ZipArchive::new(reader).unwrap();
    for i in 0..zip.len() {
        let mut file = zip.by_index(i).unwrap();
        if file.name().ends_with("_gridfloat.hdr") {
            assert_eq!(hdr.len(), 0);
            file.read_to_string(&mut hdr).unwrap();
        } else if file.name().ends_with("_gridfloat.flt") {
            assert_eq!(flt.len(), 0);
            file.read_to_end(&mut flt).unwrap();
        }
    }

    let mut width = None;
    let mut height = None;
    let mut xllcorner = None;
    let mut yllcorner = None;
    let mut cell_size = None;
    let mut byte_order = None;
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

    let size = width.unwrap() * height.unwrap();
    assert_eq!(flt.len(), size * 4);

    let mut rdr = Cursor::new(flt);
    let mut elevations: Vec<f32> = Vec::with_capacity(size);
    for _ in 0..size {
        let e = match byte_order {
            Some(ByteOrder::LsbFirst) => rdr.read_f32::<LittleEndian>().unwrap(),
            Some(ByteOrder::MsbFirst) => rdr.read_f32::<BigEndian>().unwrap(),
            None => unreachable!(),
        };
        elevations.push(e);
    }

    Dem {
        width: width.unwrap(),
        height: height.unwrap(),
        xllcorner: xllcorner.unwrap(),
        yllcorner: yllcorner.unwrap(),
        cell_size: cell_size.unwrap(),
        elevations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let dem = download_dem(28, -81, DemSource::Usgs30m);
        assert_eq!(dem.width, 3612);
        assert_eq!(dem.height, 3612);
        assert!(dem.cell_size > 0.0002777);
        assert!(dem.cell_size < 0.0002778);
    }
}
