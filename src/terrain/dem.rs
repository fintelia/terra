use zip::ZipArchive;
use safe_transmute;

use std::io::{Read, Cursor, Seek};
use std::str::FromStr;
use std::mem;

#[cfg(feature = "download")]
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

impl Dem {
    /// Create a Dem from a reader over the contents of a USGS GridFloat zip file.
    ///
    /// Such files can be found at:
    /// * https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Elevation/2/GridFloat
    /// * https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Elevation/1/GridFloat
    /// * https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Elevation/13/GridFloat
    pub fn from_gridfloat_zip<R: Read + Seek>(zip_file: R) -> Dem {
        let mut hdr = String::new();
        let mut flt = Vec::new();

        let mut zip = ZipArchive::new(zip_file).unwrap();
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

        let flt =
            unsafe { safe_transmute::guarded_transmute_many_pedantic::<u32>(&flt[..]).unwrap() };
        let mut elevations: Vec<f32> = Vec::with_capacity(size);
        for f in flt {
            let e = match byte_order {
                Some(ByteOrder::LsbFirst) => f.to_le(),
                Some(ByteOrder::MsbFirst) => f.to_be(),
                None => unreachable!(),
            };
            elevations.push(unsafe { mem::transmute::<u32, f32>(e) });
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

    /// Downloads a GridFloat zip for the indicated latitude and longitude sourced from the USGS.
    /// The output should be suitable to pass to Dem::from_gridfloat_zip().
    #[cfg(feature = "download")]
    pub fn download_gridfloat_zip(latitude: i16,
                                  longitude: i16,
                                  source: DemSource)
                                  -> Cursor<Vec<u8>> {
        use curl::easy::Easy;

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

    pub fn crop(&self, width: usize, height: usize) -> Self {
        assert!(width > 0 && width <= self.width);
        assert!(height > 0 && height <= self.height);

        let xoffset = (self.width - width) / 2;
        let yoffset = (self.height - height) / 2;

        let mut elevations = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                elevations.push(self.elevations[(x + xoffset) + (y + yoffset) * self.width]);
            }
        }

        Self {
            width,
            height,
            cell_size: self.cell_size,
            xllcorner: self.xllcorner + self.cell_size * (xoffset as f64),
            yllcorner: self.yllcorner + self.cell_size * (yoffset as f64),
            elevations,
        }
    }

    pub fn zero(mut self) -> Self {
        let mut max = None;
        let mut min = None;
        for h in self.elevations.iter().filter(|h| **h > 0.0) {
            if min.is_none() || h < min.as_ref().unwrap() {
                min = Some(*h);
            }
            if max.is_none() || h > max.as_ref().unwrap() {
                max = Some(*h);
            }
        }
        let min = min.unwrap();
        println!("min = {}, max = {}", min, max.unwrap());

        for h in &mut self.elevations {
            *h = (*h - min) * 0.1;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "download")]
    fn it_works() {
        use super::*;

        let zip = Dem::download_gridfloat_zip(28, -81, DemSource::Usgs30m);
        let dem = Dem::from_gridfloat_zip(zip);
        assert_eq!(dem.width, 3612);
        assert_eq!(dem.height, 3612);
        assert!(dem.cell_size > 0.0002777);
        assert!(dem.cell_size < 0.0002778);
    }
}
