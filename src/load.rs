use crate::asset::{AssetLoadContext, AssetLoadContextBuf, WebAsset};
use crate::cache::{LayerParams, LayerType, TextureFormat};
use crate::mapfile::{MapFile, TextureDescriptor};
use anyhow::Error;
use basis_universal::Transcoder;
use types::VNode;
use vec_map::VecMap;

struct WebTextureAsset {
    url: String,
    filename: String,
    format: TextureFormat,
}
impl WebAsset for WebTextureAsset {
    type Type = (TextureDescriptor, Vec<u8>);

    fn url(&self) -> String {
        self.url.clone()
    }
    fn filename(&self) -> String {
        self.filename.clone()
    }
    fn parse(&self, _context: &mut AssetLoadContext, data: Vec<u8>) -> Result<Self::Type, Error> {
        match self.format {
            TextureFormat::UASTC => {
                let transcoder = Transcoder::new();
                let depth = transcoder.image_count(&data);
                let info = transcoder.image_info(&data, 0).unwrap();
                Ok((
                    TextureDescriptor {
                        format: self.format,
                        width: info.m_width,
                        height: info.m_height,
                        depth,
                        array_texture: true,
                    },
                    data,
                ))
            }
            TextureFormat::RGBA8 => {
                let img = image::load_from_memory(&data)?.into_rgba8();
                Ok((
                    TextureDescriptor {
                        format: TextureFormat::RGBA8,
                        width: img.width(),
                        height: img.height(),
                        depth: 1,
                        array_texture: false,
                    },
                    img.into_raw(),
                ))
            }
            _ => unimplemented!(),
        }
    }
}

struct WebModel {
    url: String,
    filename: String,
}
impl WebAsset for WebModel {
    type Type = ();

    fn url(&self) -> String {
        self.url.clone()
    }
    fn filename(&self) -> String {
        self.filename.clone()
    }
    fn parse(&self, _context: &mut AssetLoadContext, _data: Vec<u8>) -> Result<(), Error> {
        Ok(())
    }
}

pub(crate) async fn build_mapfile(server: String) -> Result<MapFile, Error> {
    let layers: VecMap<LayerParams> = LayerType::iter()
        .map(|layer_type| {
            let params = match layer_type {
                LayerType::Heightmaps => LayerParams {
                    texture_resolution: 521,
                    texture_border_size: 4,
                    texture_format: &[TextureFormat::R32],
                    grid_registration: true,
                    min_level: 0,
                    max_level: VNode::LEVEL_CELL_5MM,
                    layer_type,
                },
                LayerType::Displacements => LayerParams {
                    texture_resolution: 65,
                    texture_border_size: 0,
                    texture_format: &[TextureFormat::RGBA32F],
                    grid_registration: true,
                    min_level: 0,
                    max_level: VNode::LEVEL_CELL_5MM,
                    layer_type,
                },
                LayerType::AlbedoRoughness => LayerParams {
                    texture_resolution: 516,
                    texture_border_size: 2,
                    texture_format: &[TextureFormat::RGBA8],
                    grid_registration: false,
                    min_level: 0,
                    max_level: VNode::LEVEL_CELL_5MM,
                    layer_type,
                },
                LayerType::Normals => LayerParams {
                    texture_resolution: 516,
                    texture_border_size: 2,
                    texture_format: &[TextureFormat::RG8],
                    grid_registration: false,
                    min_level: 0,
                    max_level: VNode::LEVEL_CELL_5MM,
                    layer_type,
                },
                LayerType::GrassCanopy => LayerParams {
                    texture_resolution: 516,
                    texture_border_size: 2,
                    texture_format: &[TextureFormat::RGBA8],
                    grid_registration: false,
                    min_level: VNode::LEVEL_CELL_1M,
                    max_level: VNode::LEVEL_CELL_1M,
                    layer_type,
                },
                LayerType::AerialPerspective => LayerParams {
                    texture_resolution: 17,
                    texture_border_size: 0,
                    texture_format: &[TextureFormat::RGBA16F],
                    grid_registration: true,
                    min_level: 3,
                    max_level: VNode::LEVEL_SIDE_610M,
                    layer_type,
                },
                LayerType::BentNormals => LayerParams {
                    texture_resolution: 513,
                    texture_border_size: 0,
                    texture_format: &[TextureFormat::RGBA8],
                    grid_registration: true,
                    min_level: VNode::LEVEL_CELL_153M,
                    max_level: VNode::LEVEL_CELL_76M,
                    layer_type,
                },
                LayerType::TreeCover => LayerParams {
                    texture_resolution: 516,
                    texture_border_size: 2,
                    texture_format: &[TextureFormat::R8],
                    grid_registration: false,
                    min_level: 0,
                    max_level: VNode::LEVEL_CELL_76M,
                    layer_type,
                },
                LayerType::BaseAlbedo => LayerParams {
                    texture_resolution: 516,
                    texture_border_size: 2,
                    texture_format: &[TextureFormat::UASTC],
                    grid_registration: false,
                    min_level: 0,
                    max_level: VNode::LEVEL_CELL_610M,
                    layer_type,
                },
                LayerType::TreeAttributes => LayerParams {
                    texture_resolution: 516,
                    texture_border_size: 2,
                    texture_format: &[TextureFormat::RGBA8],
                    grid_registration: false,
                    min_level: VNode::LEVEL_CELL_10M,
                    max_level: VNode::LEVEL_CELL_10M,
                    layer_type,
                },
                LayerType::RootAerialPerspective => LayerParams {
                    texture_resolution: 65,
                    texture_border_size: 0,
                    texture_format: &[TextureFormat::RGBA16F],
                    grid_registration: true,
                    min_level: 0,
                    max_level: 0,
                    layer_type,
                },
                LayerType::LandFraction => LayerParams {
                    texture_resolution: 516,
                    texture_border_size: 2,
                    texture_format: &[TextureFormat::R8],
                    grid_registration: false,
                    min_level: 0,
                    max_level: VNode::LEVEL_CELL_76M,
                    layer_type,
                },
                LayerType::Slopes => LayerParams {
                    texture_resolution: 517,
                    texture_border_size: 2,
                    texture_format: &[TextureFormat::RG16F],
                    grid_registration: true,
                    min_level: VNode::LEVEL_CELL_10M,
                    max_level: VNode::LEVEL_CELL_10M,
                    layer_type,
                },
            };
            (layer_type.index(), params)
        })
        .collect();

    let mut mapfile = MapFile::new(layers, server).await?;
    let mut context = AssetLoadContextBuf::new();
    let mut context = context.context("Building Terrain...", 1);
    // generate_heightmaps(&mut mapfile, &mut context).await?;
    // generate_albedo(&mut mapfile, &mut context)?;
    // generate_roughness(&mut mapfile, &mut context)?;
    generate_noise(&mut mapfile, &mut context)?;
    generate_sky(&mut mapfile, &mut context)?;

    download_cloudcover(&mut mapfile, &mut context)?;
    download_ground_albedo(&mut mapfile, &mut context)?;
    download_models(&mut context)?;

    Ok(mapfile)
}

fn generate_noise(mapfile: &mut MapFile, context: &mut AssetLoadContext) -> Result<(), Error> {
    if !mapfile.reload_texture("noise") {
        // wavelength = 1.0 / 256.0;
        let noise_desc = TextureDescriptor {
            width: 2048,
            height: 2048,
            depth: 1,
            format: TextureFormat::RGBA8,
            array_texture: false,
        };

        let noise_heightmaps: Vec<_> =
            (0..4).map(|i| crate::noise::wavelet_noise(64 << i, 32 >> i)).collect();

        context.reset("Generating noise textures... ", noise_heightmaps.len() as u64);

        let len = noise_heightmaps[0].heights.len();
        let mut heights = vec![0u8; len * 4];
        for (i, heightmap) in noise_heightmaps.into_iter().enumerate() {
            context.set_progress(i as u64);
            let mut dist: Vec<(usize, f32)> = heightmap.heights.into_iter().enumerate().collect();
            dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for j in 0..len {
                heights[dist[j].0 * 4 + i] = (j * 256 / len) as u8;
            }
        }

        mapfile.write_texture("noise", noise_desc, &heights[..])?;
    }
    Ok(())
}

fn generate_sky(mapfile: &mut MapFile, context: &mut AssetLoadContext) -> Result<(), Error> {
    if !mapfile.reload_texture("sky") {
        context.reset("Generating sky texture... ", 1);
        let sky = WebTextureAsset {
            url: "https://www.eso.org/public/archives/images/original/eso0932a.tif".to_owned(),
            filename: "eso0932a.tif".to_owned(),
            format: TextureFormat::RGBA8,
        }
        .load(context)?;
        mapfile.write_texture("sky", sky.0, &sky.1)?;
    }
    if !mapfile.reload_texture("transmittance") || !mapfile.reload_texture("inscattering") {
        let atmosphere = crate::sky::Atmosphere::new(context)?;
        mapfile.write_texture(
            "transmittance",
            TextureDescriptor {
                width: atmosphere.transmittance.size[0] as u32,
                height: atmosphere.transmittance.size[1] as u32,
                depth: 1,
                format: TextureFormat::RGBA32F,
                array_texture: false,
            },
            bytemuck::cast_slice(&atmosphere.transmittance.data),
        )?;
        mapfile.write_texture(
            "inscattering",
            TextureDescriptor {
                width: atmosphere.inscattering.size[0] as u32,
                height: atmosphere.inscattering.size[1] as u32,
                depth: atmosphere.inscattering.size[2] as u32,
                format: TextureFormat::RGBA32F,
                array_texture: false,
            },
            bytemuck::cast_slice(&atmosphere.inscattering.data),
        )?;
    }
    Ok(())
}

fn download_cloudcover(mapfile: &mut MapFile, context: &mut AssetLoadContext) -> Result<(), Error> {
    if !mapfile.reload_texture("cloudcover") {
        let cloudcover = WebTextureAsset {
            url: "https://terra.fintelia.io/file/terra-tiles/clouds_combined.png".to_owned(),
            filename: "clouds_combined.png".to_owned(),
            format: TextureFormat::RGBA8,
        }
        .load(context)?;
        mapfile.write_texture("cloudcover", cloudcover.0, &cloudcover.1)?;
    }

    Ok(())
}

fn download_ground_albedo(
    mapfile: &mut MapFile,
    context: &mut AssetLoadContext,
) -> Result<(), Error> {
    if !mapfile.reload_texture("ground_albedo") {
        let texture = WebTextureAsset {
            url: "https://terra.fintelia.io/file/terra-tiles/ground_albedo.basis".to_owned(),
            filename: "ground_albedo.basis".to_owned(),
            format: TextureFormat::UASTC,
        }
        .load(context)?;
        mapfile.write_texture("ground_albedo", texture.0, &texture.1)?;
    }

    Ok(())
}

fn download_models(context: &mut AssetLoadContext) -> Result<(), Error> {
    WebModel {
        url: "https://terra.fintelia.io/file/terra-tiles/Oak_English_Sapling.zip".to_owned(),
        filename: "Oak_English_Sapling.zip".to_owned(),
    }
    .load(context)
}
