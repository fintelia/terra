use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::path::{Path, PathBuf};

use atomicwrites::{AtomicFile, OverwriteBehavior};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use s3::bucket::Bucket;
use s3::creds::Credentials;
//use s3::S3Error;

fn s3_download(bucket: &Bucket, remote_path: &str, local_path: &Path) -> Result<(), anyhow::Error> {
    if !local_path.exists() {
        let mut i = 0;
        let contents;
        loop {
            match bucket.get_object_blocking(&remote_path) {
                Ok((c, code)) => {
                    contents = c;
                    if code != 200 {
                        println!("{}", remote_path);
                        return Ok(());
                    }
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
            .write(|f| f.write_all(&contents))?;
    }
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
pub fn download_copernicus_wbm(path: &Path) -> Result<(), anyhow::Error> {
    let directory = path.join("copernicus-wbm");

    std::fs::create_dir_all(&directory)?;

    let bucket =
        Bucket::new("copernicus-dem-30m", "eu-central-1".parse()?, Credentials::anonymous()?)?;
    let bucket_fallback =
        Bucket::new("copernicus-dem-90m", "eu-central-1".parse()?, Credentials::anonymous()?)?;

    let (tile_list, code) = bucket.get_object_blocking("tileList.txt")?;
    assert_eq!(200, code);
    let (missing, code) = bucket.get_object_blocking("blacklist.txt")?;
    assert_eq!(200, code);

    let tile_list = String::from_utf8(tile_list)?;
    let tile_list: Vec<_> = tile_list.split_ascii_whitespace().collect();
    tile_list.par_iter().try_for_each(|name| -> Result<(), anyhow::Error> {
        let filename = format!("{}WBM.tif", &name[..name.len() - 3]);
        let local_path = directory.join(&filename);
        let remote_path = format!("{}/AUXFILES/{}", name, filename);
        s3_download(&bucket, &remote_path, &local_path)
    })?;

    let missing = String::from_utf8(missing)?;
    let missing: Vec<_> = missing.split_ascii_whitespace().collect();
    missing.par_iter().try_for_each(|name| -> Result<(), anyhow::Error> {
        let name = name.replace("DSM_10", "DSM_COG_30").replace(".tif", "");
        let filename = format!("{}WBM.tif", &name[..name.len() - 3]);
        let local_path = directory.join(&filename);
        let remote_path = format!("{}/AUXFILES/{}", name, filename);
        s3_download(&bucket_fallback, &remote_path, &local_path)
    })?;

    make_vrt(&directory, OsStr::new("tif"))?;

    Ok(())
}

// Download heights from Copernicus dataset.
//
// See https://registry.opendata.aws/copernicus-dem/
pub fn download_copernicus_hgt(path: &Path) -> Result<(), anyhow::Error> {
    let directory = path.join("copernicus-hgt");

    std::fs::create_dir_all(&directory)?;

    let bucket =
        Bucket::new("copernicus-dem-30m", "eu-central-1".parse()?, Credentials::anonymous()?)?;
    let bucket_fallback =
        Bucket::new("copernicus-dem-90m", "eu-central-1".parse()?, Credentials::anonymous()?)?;

    let (tile_list, code) = bucket.get_object_blocking("tileList.txt")?;
    assert_eq!(200, code);
    let (missing, code) = bucket.get_object_blocking("blacklist.txt")?;
    assert_eq!(200, code);

    let tile_list = String::from_utf8(tile_list)?;
    let tile_list: Vec<_> = tile_list.split_ascii_whitespace().collect();
    tile_list.into_par_iter().try_for_each(|name| -> Result<(), anyhow::Error> {
        let filename = format!("{}DEM.tif", &name[..name.len() - 3]);
        let local_path = directory.join(&filename);
        let remote_path = format!("{}/{}", name, filename);
        s3_download(&bucket, &remote_path, &local_path)
    })?;

    let missing = String::from_utf8(missing)?;
    let missing: Vec<_> = missing.split_ascii_whitespace().collect();
    missing.into_par_iter().try_for_each(|name| -> Result<(), anyhow::Error> {
        let name = name.replace("DSM_10", "DSM_COG_30").replace(".tif", "");
        let filename = format!("{}DEM.tif", &name[..name.len() - 3]);
        let local_path = directory.join(&filename);
        let remote_path = format!("{}/{}", name, filename);
        s3_download(&bucket_fallback, &remote_path, &local_path)
    })?;

    make_vrt(&directory, OsStr::new("tif"))?;

    Ok(())
}
