use serde::{Deserialize, Serialize};
use super::description::{TextureFormat, DatasetFormat, Projection};
use super::SectorCache;
use std::{fs};
use std::io::{Cursor, Read};
use std::str::FromStr;
use std::path::PathBuf;
use zip::ZipArchive;
use failure::format_err;
use gfx_hal::Backend;

#[derive(Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub struct DatasetDesc {
    pub url: String,
    pub credentials: Option<(String, String)>,
    pub projection: Projection,
    pub resolution: u32,
    pub file_format: DatasetFormat,
    pub texture_format: TextureFormat,
}

pub struct Dataset<B: Backend> {
    pub desc: DatasetDesc,

    pub bib: Option<String>,
    pub license: Option<String>,

    pub directory: PathBuf,

    pub sector_cache: SectorCache<B>,
}
impl<B: Backend> Dataset<B> {
    fn parse(&mut self, data: Vec<u8>) -> Result<Vec<u8>, failure::Error> {
        match self.desc.file_format {
            DatasetFormat::ZippedGridFloat => parse_ned_zip(data),
        }
    }

    pub fn get_tile(&mut self, lat: i16, long: i16) -> Result<Option<Vec<u8>>, failure::Error> {
        let mut url = self.desc.url.clone();
        url = url.replace("{ns}", if lat >= 0 { "n" } else { "s" });
        url = url.replace("{ew}", if long >= 0 { "e" } else { "w" });
        url = url.replace("{lat}", &format!("{}", lat.abs()));
        url = url.replace("{long}", &format!("{}", long.abs()));
        url = url.replace("{lat02}", &format!("{:02}", lat.abs()));
        url = url.replace("{long03}", &format!("{:03}", long.abs()));

        let filename = self.directory.join(url.split_at(url.rfind("/").expect("URL without slash?")+1).1);


        let parsed = fs::read(&filename).ok().and_then(|contents| self.parse(contents).ok());
        if parsed.is_some() {
            return Ok(parsed);
        }

        let mut data = Vec::<u8>::new();
        {
            use curl::easy::Easy;
            let mut easy = Easy::new();
            easy.url(&url)?;
            easy.follow_location(true)?;
            if let Some((ref username, ref password)) = self.desc.credentials {
                easy.cookie_file("")?;
                easy.unrestricted_auth(true)?;
                easy.username(&username)?;
                easy.password(&password)?;
            }
            let mut easy = easy.transfer();
            easy.write_function(|d| {
                let len = d.len();
                data.extend(d);
                Ok(len)
            })?;
            if easy.perform().is_err() {
                return Ok(None);
            }
        }

        let _ = fs::write(filename, &data).unwrap();
        self.parse(data).map(Some)
    }
}


/// Load a zip file in the format for the USGS's National Elevation Dataset.
#[allow(unused)]
fn parse_ned_zip(data: Vec<u8>) -> Result<Vec<u8>, failure::Error> {
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
                "byteorder" => assert_eq!(value, "LSBFIRST"),
                _ => {}
            }
        }
    }

    let width = width.ok_or(format_err!("width field missing"))?;
    let height = height.ok_or(format_err!("height field missing"))?;
    // let xllcorner = xllcorner?;
    // let yllcorner = yllcorner?;
    // let cell_size = cell_size?;
    let nodata_value = nodata_value.ok_or(format_err!("nodata field missing"))?;

    let size = width * height;
    if flt.len() != size * 4 {
        return Err(format_err!("invalid file size"));
    }

    // {
    //     let flt: &mut [u32] = unsafe { safe_transmute::guarded_transmute_many_pedantic::<u32>(&mut flt[..]).unwrap() };
    //     let mut elevations: Vec<f32> = Vec::with_capacity(size);
    //     for f in &mut flt {
    //         *f = match byte_order {
    //             ByteOrder::LsbFirst => f.to_le(),
    //             ByteOrder::MsbFirst => f.to_be(),
    //         };
    //         let e = unsafe { mem::transmute::<u32, f32>(e) };
    //         elevations.push(if e == nodata_value { 0.0 } else { e });
    //     }
    // }

    Ok(flt)
    // Ok(Raster {
    //     width,
    //     height,
    //     bands: 1,
    //     latitude_llcorner: xllcorner,
    //     longitude_llcorner: yllcorner,
    //     cell_size,
    //     values: elevations,
    // })
}
