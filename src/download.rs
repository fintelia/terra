use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::path::Path;

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

    if !directory.join("merged.vrt").exists() {
        make_vrt(&directory, OsStr::new("tif"))?;
    }

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

    if !directory.join("merged.vrt").exists() {
        make_vrt(&directory, OsStr::new("tif"))?;
    }

    Ok(())
}

pub fn download_bluemarble(path: &Path) -> Result<(), anyhow::Error> {
    let directory = path.join("bluemarble");
    std::fs::create_dir_all(&directory)?;

    // TODO: actually download

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
