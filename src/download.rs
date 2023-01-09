use std::collections::btree_map::Entry;
use std::collections::{BTreeMap};
use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use atomicwrites::{AtomicFile, OverwriteBehavior};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use s3::bucket::Bucket;
use s3::creds::Credentials;

struct AtomicProgress<F: FnMut(String, usize, usize) + Send> {
    mutex: Mutex<(u64, F)>,
    message: String,
    total: u64,
}
impl<F: FnMut(String, usize, usize) + Send> AtomicProgress<F> {
    fn new(message: String, f: F, total: u64) -> Self {
        let s = Self { mutex: Mutex::new((0, f)), message, total };
        s.tick();
        s
    }
    fn tick(&self) {
        let lock = &mut self.mutex.lock().unwrap();
        lock.0 += 1;
        let v = lock.0;
        (lock.1)(self.message.clone(), v as usize, self.total as usize);
    }
}

fn check_etag_match(file: &Path, size: u64, etag: &str) -> bool {
    if let Ok(data) = std::fs::read(file) {
        if data.len() == size as usize {
            assert!(!etag.contains('-')); // TODO: handle multipart etags
            return etag == format!("\"{:x}\"", md5::compute(&data));
        }
    }
    false
}

fn s3_download(bucket: &Bucket, remote_path: &str, local_path: &Path) -> Result<(), anyhow::Error> {
    if local_path.exists() {
        let metadata = bucket.head_object_blocking(remote_path)?;
        if let (Some(size), Some(etag)) = (metadata.0.content_length, metadata.0.e_tag) {
            if check_etag_match(local_path, size as u64, &etag) {
                return Ok(());
            }
        }
    }

    let mut i = 0;
    let contents;
    loop {
        match bucket.get_object_blocking(&remote_path) {
            Ok(c) => {
                if c.status_code() != 200 {
                    println!("{}", remote_path);
                    return Ok(());
                }
                contents = c;
                break;
            }
            Err(e) => {
                if i < 20 {
                    i += 1;
                    continue;
                }
                return Err(e.into());
            }
        }
    }

    AtomicFile::new(local_path, OverwriteBehavior::AllowOverwrite)
        .write(|f| f.write_all(contents.bytes()))?;
    Ok(())
}

fn bulk_s3_download<F: FnMut(String, usize, usize) + Send>(
    message: String,
    bucket: &Bucket,
    mut paths: BTreeMap<String, PathBuf>,
    progress_callback: F,
) -> Result<(), anyhow::Error> {
    let progress = AtomicProgress::new(message, progress_callback, paths.len() as u64);

    let mut objects_listed = 0;
    let mut continuation_token = None;
    while objects_listed < paths.len() * 100 {
        let list =
            bucket.list_page_blocking("".to_string(), None, continuation_token, None, None)?.0;

        for object in &list.contents {
            if let Entry::Occupied(entry) = paths.entry(object.key.clone()) {
                if entry.get().exists()
                    && check_etag_match(entry.get(), object.size, object.e_tag.as_ref().unwrap())
                {
                    entry.remove();
                    progress.tick();
                }
            }
        }

        objects_listed += list.contents.len();
        continuation_token = list.next_continuation_token;
        if continuation_token.is_none() {
            break;
        }
    }

    paths.into_par_iter().try_for_each(
        |(remote_path, local_path)| -> Result<(), anyhow::Error> {
            s3_download(&bucket, &remote_path, &local_path)?;
            progress.tick();
            Ok(())
        },
    )?;
    Ok(())
}

fn bulk_http_download<F: FnMut(String, usize, usize) + Send>(
    message: String,
    downloads: BTreeMap<String, PathBuf>,
    progress_callback: F,
) -> Result<(), anyhow::Error> {
    let progress = AtomicProgress::new(message, progress_callback, downloads.len() as u64);

    let client = reqwest::blocking::ClientBuilder::new().timeout(None).build().unwrap();
    downloads.into_iter().try_for_each(|(url, path)| -> Result<(), anyhow::Error> {
        if path.exists() {
            let metadata = client.head(&url).send().unwrap();
            if let Some(size) = metadata.headers().get(reqwest::header::CONTENT_LENGTH) {
                let size = size.to_str().unwrap().parse::<u64>().unwrap();
                if size == std::fs::metadata(&path).unwrap().len() {
                    progress.tick();
                    return Ok(());
                }
            }
        }

        let contents = client.get(&url).send().unwrap().bytes().unwrap().to_vec();
        AtomicFile::new(path, OverwriteBehavior::AllowOverwrite)
            .write(|f| f.write_all(&contents)).unwrap();
        progress.tick();
        Ok(())
    }).unwrap();
    Ok(())
}

fn make_vrt(directory: &Path, extension: &OsStr) -> Result<(), anyhow::Error> {
    let files: Vec<OsString> = std::fs::read_dir(directory)?
        .filter_map(Result::ok)
        .filter(|f| f.path().extension() == Some(extension))
        .map(|f| f.file_name())
        .collect();

    let mut args = vec![OsString::from("merged.vrt")];
    args.extend(files);

    let output = std::process::Command::new("gdalbuildvrt")
        .current_dir(directory)
        .args(args)
        .output()
        .expect("Failed to run gdalbuildvrt. Is gdal installed?");

    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    Ok(())
}

// pub fn download_nasadem(path: &Path) -> Result<(), anyhow::Error> {
//     let directory = path.join("nasadem");
//     std::fs::create_dir_all(&directory)?;

//     // TODO: Actually do the download step

//     let files = std::fs::read_dir(directory)?.filter_map(Result::ok).filter(|f|f.path().extension() == ).collect();

//     Ok(())
// }

// Download global watermask from Copernicus dataset.
//
//  | Pixel | Value Meaning |
//  |-------|---------------|
//  |   0   | No water      |
//  |   1   | Ocean         |
//  |   2   | Lake          |
//  |   3   | River         |
//  |-------|---------------|
pub fn download_copernicus_wbm<F: FnMut(String, usize, usize) + Send>(
    path: &Path,
    mut progress_callback: F,
) -> Result<(), anyhow::Error> {
    let directory = path.join("download").join("copernicus-wbm");
    std::fs::create_dir_all(&directory)?;

    let bucket =
        Bucket::new("copernicus-dem-30m", "eu-central-1".parse()?, Credentials::anonymous()?)?;
    let bucket_fallback =
        Bucket::new("copernicus-dem-90m", "eu-central-1".parse()?, Credentials::anonymous()?)?;

    let tile_list = bucket.get_object_blocking("tileList.txt")?;
    assert_eq!(200, tile_list.status_code());
    let missing = bucket.get_object_blocking("blacklist.txt")?;
    assert_eq!(200, missing.status_code());

    let tile_list = String::from_utf8(tile_list.bytes().to_owned())?
        .split_ascii_whitespace()
        .map(|name| {
            let filename = format!("{}WBM.tif", &name[..name.len() - 3]);
            let local_path = directory.join(&filename);
            let remote_path = format!("{}/AUXFILES/{}", name, filename);
            (remote_path, local_path)
        })
        .collect();
    bulk_s3_download("Downloading WBM".to_string(), &bucket, tile_list, &mut progress_callback)?;

    let missing = String::from_utf8(missing.bytes().to_owned())?
        .split_ascii_whitespace()
        .map(|name| {
            let name = name.replace("DSM_10", "DSM_COG_30").replace(".tif", "");
            let filename = format!("{}WBM.tif", &name[..name.len() - 3]);
            let local_path = directory.join(&filename);
            let remote_path = format!("{}/AUXFILES/{}", name, filename);
            (remote_path, local_path)
        })
        .collect();
    bulk_s3_download(
        "Downloading WBM (fallbacks)".to_string(),
        &bucket_fallback,
        missing,
        &mut progress_callback,
    )?;

    if !directory.join("merged.vrt").exists() {
        make_vrt(&directory, OsStr::new("tif"))?;
    }

    Ok(())
}

// Download heights from Copernicus dataset.
//
// See https://registry.opendata.aws/copernicus-dem/
pub fn download_copernicus_hgt<F: FnMut(String, usize, usize) + Send>(
    path: &Path,
    mut progress_callback: F,
) -> Result<(), anyhow::Error> {
    let directory = path.join("download").join("copernicus-hgt");
    std::fs::create_dir_all(&directory)?;

    let bucket =
        Bucket::new("copernicus-dem-30m", "eu-central-1".parse()?, Credentials::anonymous()?)?;
    let bucket_fallback =
        Bucket::new("copernicus-dem-90m", "eu-central-1".parse()?, Credentials::anonymous()?)?;

    let tile_list = bucket.get_object_blocking("tileList.txt")?;
    assert_eq!(200, tile_list.status_code());
    let missing = bucket.get_object_blocking("blacklist.txt")?;
    assert_eq!(200, missing.status_code());

    let tile_list = String::from_utf8(tile_list.bytes().to_owned())?
        .split_ascii_whitespace()
        .map(|name| {
            let filename = format!("{}DEM.tif", &name[..name.len() - 3]);
            let local_path = directory.join(&filename);
            let remote_path = format!("{}/{}", name, filename);
            (remote_path, local_path)
        })
        .collect();
    bulk_s3_download("Downloading DEM".to_string(), &bucket, tile_list, &mut progress_callback)?;

    let missing = String::from_utf8(missing.bytes().to_owned())?
        .split_ascii_whitespace()
        .map(|name| {
            let name = name.replace("DSM_10", "DSM_COG_30").replace(".tif", "");
            let filename = format!("{}DEM.tif", &name[..name.len() - 3]);
            let local_path = directory.join(&filename);
            let remote_path = format!("{}/{}", name, filename);
            (remote_path, local_path)
        })
        .collect();
    bulk_s3_download(
        "Downloading DEM (fallbacks)".to_string(),
        &bucket_fallback,
        missing,
        &mut progress_callback,
    )?;

    if !directory.join("merged.vrt").exists() {
        make_vrt(&directory, OsStr::new("tif"))?;
    }

    Ok(())
}

pub fn download_bluemarble<F: FnMut(String, usize, usize) + Send>(
    path: &Path,
    mut progress_callback: F,
) -> Result<(), anyhow::Error> {
    let directory = path.join("download").join("bluemarble");
    std::fs::create_dir_all(&directory)?;

    pub const BLUE_MARBLE_URLS: [&str; 8] = [
        "https://eoimages.gsfc.nasa.gov/images/imagerecords/76000/76487/world.200406.3x21600x21600.A1.png",
        "https://eoimages.gsfc.nasa.gov/images/imagerecords/76000/76487/world.200406.3x21600x21600.A2.png",
        "https://eoimages.gsfc.nasa.gov/images/imagerecords/76000/76487/world.200406.3x21600x21600.B1.png",
        "https://eoimages.gsfc.nasa.gov/images/imagerecords/76000/76487/world.200406.3x21600x21600.B2.png",
        "https://eoimages.gsfc.nasa.gov/images/imagerecords/76000/76487/world.200406.3x21600x21600.C1.png",
        "https://eoimages.gsfc.nasa.gov/images/imagerecords/76000/76487/world.200406.3x21600x21600.C2.png",
        "https://eoimages.gsfc.nasa.gov/images/imagerecords/76000/76487/world.200406.3x21600x21600.D1.png",
        "https://eoimages.gsfc.nasa.gov/images/imagerecords/76000/76487/world.200406.3x21600x21600.D2.png",
    ];
    bulk_http_download(
        "Downloading bluemarble".to_string(),
        BLUE_MARBLE_URLS
            .iter()
            .map(|url| {
                let filename = url.split('/').last().unwrap();
                let local_path = directory.join(filename);
                let remote_path = url.to_string();
                (remote_path, local_path)
            })
            .collect(),
        &mut progress_callback,
    )?;

    // Write vrt file.
    let mut data = r#"<VRTDataset rasterXSize="86400" rasterYSize="43200">
<SRS dataAxisToSRSAxisMapping="2,1">GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]</SRS>
<GeoTransform> -180.0, 0.004166666666666667, 0.0, 90.0, 0.0, -0.004166666666666667</GeoTransform>
<VRTRasterBand dataType="Byte" band="1">
<ColorInterp>Red</ColorInterp>
"#.to_owned();
    for x in 0..4 {
        for y in 0..2 {
            let filename =
                format!("world.200406.3x21600x21600.{}{}.png", ['A', 'B', 'C', 'D'][x], y + 1);
            data += &format!(
                r#"  <SimpleSource>
    <SourceFilename relativeToVRT="1">{}</SourceFilename>
    <SourceBand>1</SourceBand>
    <SourceProperties RasterXSize="21600" RasterYSize="21600" DataType="Byte" />
    <SrcRect xOff="0" yOff="0" xSize="21600" ySize="21600"/>
    <DstRect xOff="{}" yOff="{}" xSize="21600" ySize="21600"/>
  </SimpleSource>
"#,
                filename,
                x * 21600,
                y * 21600,
            );
        }
    }
    data.push_str(
        "</VRTRasterBand>
</VRTDataset>",
    );
    std::fs::write(directory.join("merged.vrt"), data)?;

    Ok(())
}

pub fn download_treecover<F: FnMut(String, usize, usize) + Send>(
    path: &Path,
    mut progress_callback: F,
) -> Result<(), anyhow::Error> {
    let directory = path.join("download").join("treecover");
    std::fs::create_dir_all(&directory)?;

    bulk_http_download(
        "Downloading treecover".to_string(),
        include_str!("../file_list_treecover.txt")
            .lines()
            .map(|line| {
                let local_path = directory.join(line);
                let remote_path = format!(
                    "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2020-v1.8/{}",
                    line
                );
                (remote_path, local_path)
            })
            .collect(),
        &mut progress_callback,
    )?;

    if !directory.join("merged.vrt").exists() {
        make_vrt(&directory, OsStr::new("tif"))?;
    }

    Ok(())
}
