use std::{io::Write, path::Path};

use anyhow::Error;
use atomicwrites::{AtomicFile, OverwriteBehavior};
use image::GenericImageView;

use crate::generate::ktx2encode::encode_ktx2;
use crate::generate::sky::{InscatteringTable, TransmittanceTable};

use super::sky::LookupTableDefinition;

fn image_to_ktx2(img: image::DynamicImage) -> Result<Vec<u8>, Error> {
    let (width, height) = img.dimensions();
    let (img_data, format) = match img {
        image::DynamicImage::ImageRgb8(_) => {
            (img.to_rgba8().into_raw(), ktx2::Format::R8G8B8A8_UNORM)
        }
        image::DynamicImage::ImageRgba8(img) => (img.into_raw(), ktx2::Format::R8G8B8A8_UNORM),
        _ => todo!(),
    };

    encode_ktx2(&[img_data], width, height, 0, 0, format)
}

pub(crate) fn generate_textures<F: FnMut(String, usize, usize) + Send>(
    base_directory: &Path,
    mut progress_callback: F,
) -> Result<(), Error> {
    let assets_directory = base_directory.join("serve").join("assets");

    generate_materials(&base_directory, &mut progress_callback)?;
    generate_noise(&assets_directory, &mut progress_callback)?;
    generate_sky(&assets_directory, &mut progress_callback)?;

    let downloads =
        [("sky.ktx2", "https://www.eso.org/public/archives/images/original/eso0932a.tif")];
    for (i, (filename, url)) in downloads.into_iter().enumerate() {
        progress_callback("Downloading textures".to_string(), i, downloads.len());
        let filename = assets_directory.join(filename);
        if !filename.exists() {
            let response = reqwest::blocking::get(url)?;
            let contents = image_to_ktx2(image::load_from_memory(response.bytes()?.as_ref())?)?;
            AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
                .write(|f| f.write_all(&contents))?;
        }
    }

    let convert = [
        ("cloudcover.ktx2", "clouds_combined.png"),
        ("Oak_English_Sapling_Color.ktx2", "Oak_English_Sapling/Oak_English_Sapling_Color.png"),
        ("Oak_English_Sapling_Normal.ktx2", "Oak_English_Sapling/Oak_English_Sapling_Normal.png"),
        ("Oak_English_Sapling_SS.ktx2", "Oak_English_Sapling/Oak_English_Sapling_SS.png"),
    ];
    for (i, (filename, original)) in convert.into_iter().enumerate() {
        progress_callback("Converting textures".to_string(), i, convert.len());
        let filename = assets_directory.join(filename);
        if !filename.exists() {
            let contents =
                image_to_ktx2(image::open(base_directory.join("manual").join(original))?)?;
            AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
                .write(|f| f.write_all(&contents))?;
        }
    }

    let copy = [("Oak_English_Sapling.xml.zip", "Oak_English_Sapling/Oak_English_Sapling.xml.zip")];
    for (i, (filename, original)) in copy.into_iter().enumerate() {
        progress_callback("Copying assets".to_string(), i, copy.len());
        let filename = assets_directory.join(filename);
        if !filename.exists() {
            let contents = std::fs::read(base_directory.join("manual").join(original))?;
            AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
                .write(|f| f.write_all(&contents))?;
        }
    }

    Ok(())
}

fn generate_materials<F: FnMut(String, usize, usize) + Send>(
    base_directory: &Path,
    mut progress_callback: F,
) -> Result<(), Error> {
    let filename = base_directory.join("serve").join("assets").join("ground_albedo.ktx2");
    if filename.exists() {
        return Ok(());
    }

    let materials = [("ground", "leafy-grass2"), ("ground", "grass1"), ("rocks", "granite5")];

    let mut image_data = Vec::new();
    for (i, (group, name)) in materials.iter().enumerate() {
        progress_callback("Generating materials".to_string(), i, materials.len());

        let mut albedo_path = None;
        for file in std::fs::read_dir(
            &base_directory
                .join("manual")
                .join("free_pbr")
                .join(format!("Blender/{}-bl/{}-bl", group, name)),
        )? {
            let file = file?;
            let filename = file.file_name();
            let filename = filename.to_string_lossy();
            if filename.contains("albedo") {
                albedo_path = Some(file.path());
            }
        }

        let mut albedo = image::open(albedo_path.unwrap())?.to_rgba8();
        //material::high_pass_filter(&mut albedo);
        assert_eq!(albedo.width(), 2048);
        assert_eq!(albedo.height(), 2048);

        albedo =
            image::imageops::resize(&albedo, 1024, 1024, image::imageops::FilterType::Triangle);

        image_data.extend_from_slice(albedo.as_raw());
    }

    let contents = encode_ktx2(
        &[image_data],
        1024,
        1024,
        0,
        materials.len() as u32,
        ktx2::Format::R8G8B8A8_UNORM,
    )?;
    Ok(AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
        .write(|f| f.write_all(&contents))?)
}

fn generate_noise<F: FnMut(String, usize, usize) + Send>(
    assets_directory: &Path,
    mut progress_callback: F,
) -> Result<(), Error> {
    let filename = assets_directory.join("noise.ktx2");
    if filename.exists() {
        return Ok(());
    }

    // wavelength = 1.0 / 256.0;
    let noise_heightmaps: Vec<_> =
        (0..4).map(|i| crate::generate::noise::wavelet_noise(64 << i, 32 >> i)).collect();

    let len = noise_heightmaps[0].heights.len();
    let mut contents = vec![0u8; len * 4];
    for (i, heightmap) in noise_heightmaps.into_iter().enumerate() {
        progress_callback("Generating noise textures".to_string(), i, 4);
        let mut dist: Vec<(usize, f32)> = heightmap.heights.into_iter().enumerate().collect();
        dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for j in 0..len {
            contents[dist[j].0 * 4 + i] = (j * 256 / len) as u8;
        }
    }

    progress_callback("Saving noise textures".to_string(), 0, 1);
    let contents = encode_ktx2(&[contents], 2048, 2048, 0, 0, ktx2::Format::R8G8B8A8_UNORM)?;
    Ok(AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
        .write(|f| f.write_all(&contents))?)
}

fn generate_sky<F: FnMut(String, usize, usize) + Send>(
    assets_directory: &Path,
    mut progress_callback: F,
) -> Result<(), Error> {
    let filename = assets_directory.join("transmittance.ktx2");
    if !filename.exists() {
        let transmittance = TransmittanceTable { steps: 1000 }.generate(&mut progress_callback)?;

        let contents = encode_ktx2(
            &[bytemuck::cast_slice(&transmittance.data).to_vec()],
            transmittance.size[0] as u32,
            transmittance.size[1] as u32,
            0,
            0,
            ktx2::Format::R32G32B32A32_SFLOAT,
        )?;
        AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
            .write(|f| f.write_all(&contents))?;
    }

    let filename = assets_directory.join("inscattering.ktx2");
    if !filename.exists() {
        let transmittance = TransmittanceTable { steps: 1000 }.generate(&mut progress_callback)?;
        let inscattering = InscatteringTable { steps: 30, transmittance: &transmittance }
            .generate(&mut progress_callback)?;

        let contents = encode_ktx2(
            &[bytemuck::cast_slice(&inscattering.data).to_vec()],
            inscattering.size[0] as u32,
            inscattering.size[1] as u32,
            inscattering.size[2] as u32,
            0,
            ktx2::Format::R32G32B32A32_SFLOAT,
        )?;
        AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
            .write(|f| f.write_all(&contents))?;
    }

    Ok(())
}
