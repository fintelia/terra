use crate::asset::{AssetLoadContext, AssetLoadContextBuf, WebAsset};
use crate::cache::{LayerParams, LayerType, TextureFormat};
use crate::coordinates;
use crate::generate::heightmap::CogTileCache;
use crate::mapfile::{MapFile, TextureDescriptor};
use aligned_buf::AlignedBuf;
use anyhow::Error;
use atomicwrites::{AtomicFile, OverwriteBehavior};
use basis_universal::Transcoder;
use bytemuck::Pod;
use cogbuilder::CogBuilder;
use itertools::Itertools;
use num_traits::{WrappingAdd, WrappingSub, Zero};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashSet};
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{fs, mem};
use std::{io::Write, path::Path, sync::Mutex};
use types::{VFace, VNode};
use vec_map::VecMap;
use zip::ZipWriter;

mod gpu;
pub mod heightmap;
mod material;

pub(crate) use gpu::*;

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

pub(crate) async fn build_mapfile(server: String) -> Result<MapFile, Error> {
    let layers: VecMap<LayerParams> = LayerType::iter()
        .map(|layer_type| {
            let params = match layer_type {
                LayerType::Heightmaps => LayerParams {
                    texture_resolution: 517,
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

fn scan_directory(
    base: &Path,
    suffix: impl AsRef<Path>,
) -> Result<(PathBuf, HashSet<String>), Error> {
    let directory = base.join(suffix);
    fs::create_dir_all(&directory)?;

    let mut existing = HashSet::new();
    for entry in fs::read_dir(&directory)? {
        if let Ok(s) = entry?.file_name().into_string() {
            existing.insert(s);
        }
    }

    Ok((directory, existing))
}

pub(crate) struct Dataset<T> {
    pub base_directory: PathBuf,
    pub dataset_name: &'static str,
    pub max_level: u8,
    pub no_data_value: T,
    pub grid_registration: bool,
    pub bits_per_sample: Vec<u8>,
    pub signed: bool,
}
impl<T> Dataset<T>
where
    T: vrt_file::Scalar + Ord + Copy + bytemuck::Pod + ToString + Send + Sync + 'static,
{
    const BORDER_SIZE: u32 = 4;
    const TILE_INNER_RESOLUTION: u32 = 512;

    fn root_dimensions(&self) -> u32 {
        if self.grid_registration {
            1 + ((Self::TILE_INNER_RESOLUTION + 2 * Self::BORDER_SIZE) << self.max_level)
        } else {
            (Self::TILE_INNER_RESOLUTION + 2 * Self::BORDER_SIZE) << self.max_level
        }
    }

    fn cogs(&self) -> Result<Vec<(VNode, CogBuilder)>, anyhow::Error> {
        let root_dimensions = self.root_dimensions();

        let mut cogs = Vec::new();
        for root_node in VNode::roots() {
            let path = self
                .base_directory
                .join(format!("{}_reprojected", self.dataset_name))
                .join(format!("{}.tiff", VFace(root_node.face())));
            fs::create_dir_all(&path.parent().unwrap())?;
            cogs.push((
                root_node,
                cogbuilder::CogBuilder::new(
                    path,
                    root_dimensions,
                    root_dimensions,
                    self.bits_per_sample.clone(),
                    self.signed,
                    &self.no_data_value.to_string(),
                )?,
            ));
        }
        Ok(cogs)
    }

    pub(crate) fn reproject<F>(&self, progress_callback: F) -> Result<(), anyhow::Error>
    where
        F: FnMut(String, usize, usize) + Send,
    {
        let root_border_size = Self::BORDER_SIZE << self.max_level;
        let root_dimensions = self.root_dimensions();

        let vrt_file = vrt_file::VrtFile::new(
            &self.base_directory.join(self.dataset_name).join("merged.vrt"),
            self.bits_per_sample.len(),
        )?;

        let mut missing = Vec::new();
        let cogs = self
            .cogs()?
            .into_iter()
            .map(|(root_node, cog)| {
                for (tile, _) in cog.valid_mask(0)?.iter().enumerate().filter(|v| !v.1) {
                    missing.push((root_node, tile as u32));
                }
                Ok(Mutex::new(cog))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        let bands = self.bits_per_sample.len();
        let root_dimensions_tiles = (root_dimensions - 1) / cogbuilder::TILE_SIZE + 1;
        let total_sectors = (6 * root_dimensions_tiles * root_dimensions_tiles) as usize;
        let sectors_processed = AtomicUsize::new(total_sectors - missing.len());

        let progress_callback = Mutex::new(progress_callback);
        let geotransform = vrt_file.geotransform();

        vrt_file.alloc_user_bytes(
            u64::from(cogbuilder::TILE_SIZE * cogbuilder::TILE_SIZE)
                * (16 + mem::size_of::<T>() * bands) as u64
                * 128,
        );
        missing.into_iter().par_bridge().try_for_each(
            |(root, tile)| -> Result<(), anyhow::Error> {
                (progress_callback.lock().unwrap())(
                    format!("reprojecting {}...", self.dataset_name),
                    sectors_processed.load(std::sync::atomic::Ordering::SeqCst),
                    total_sectors,
                );

                let base_x = (tile % root_dimensions_tiles) * cogbuilder::TILE_SIZE;
                let base_y = (tile / root_dimensions_tiles) * cogbuilder::TILE_SIZE;

                let mut coordinates =
                    Vec::with_capacity((cogbuilder::TILE_SIZE * cogbuilder::TILE_SIZE) as usize);
                if self.grid_registration {
                    (0..(cogbuilder::TILE_SIZE * cogbuilder::TILE_SIZE))
                        .into_par_iter()
                        .map(|i| {
                            let cspace = root.grid_position_cspace(
                                (base_x + (i % cogbuilder::TILE_SIZE)) as i32,
                                (base_y + (i / cogbuilder::TILE_SIZE)) as i32,
                                root_border_size,
                                root_dimensions,
                            );
                            let polar = coordinates::cspace_to_polar(cspace);
                            let latitude = polar.x.to_degrees();
                            let longitude = polar.y.to_degrees();
                            let x = (longitude - geotransform[0]) / geotransform[1];
                            let y = (latitude - geotransform[3]) / geotransform[5];
                            (x, y)
                        })
                        .collect_into_vec(&mut coordinates);
                } else {
                    (0..(cogbuilder::TILE_SIZE * cogbuilder::TILE_SIZE))
                        .into_par_iter()
                        .map(|i| {
                            let cspace = root.cell_position_cspace(
                                (base_x + (i % cogbuilder::TILE_SIZE)) as i32,
                                (base_y + (i / cogbuilder::TILE_SIZE)) as i32,
                                root_border_size,
                                root_dimensions,
                            );
                            let polar = coordinates::cspace_to_polar(cspace);
                            let latitude = polar.x.to_degrees();
                            let longitude = polar.y.to_degrees();
                            let x = (longitude - geotransform[0]) / geotransform[1];
                            let y = (latitude - geotransform[3]) / geotransform[5];
                            (x, y)
                        })
                        .collect_into_vec(&mut coordinates);
                }

                let mut heightmap =
                    vec![
                        self.no_data_value;
                        cogbuilder::TILE_SIZE as usize * cogbuilder::TILE_SIZE as usize * bands
                    ];

                vrt_file.batch_lookup(&*coordinates, &mut heightmap);

                drop(coordinates);

                if heightmap.iter().any(|&v| v != self.no_data_value) {
                    let compressed = cogbuilder::compress_tile(bytemuck::cast_slice(&*heightmap));
                    let mut cog = cogs[root.face() as usize].lock().unwrap();
                    cog.write_tile(0, tile, &compressed)?;
                } else {
                    cogs[root.face() as usize].lock().unwrap().write_nodata_tile(0, tile)?;
                }

                sectors_processed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            },
        )?;
        vrt_file.free_user_bytes(
            u64::from(cogbuilder::TILE_SIZE * cogbuilder::TILE_SIZE)
                * (16 + mem::size_of::<T>() * bands) as u64
                * 128,
        );
        Ok(())
    }

    pub(crate) fn downsample_grid<F>(&self, progress_callback: F) -> Result<(), anyhow::Error>
    where
        F: FnMut(String, usize, usize) + Send,
    {
        assert!(self.grid_registration);
        self.downsample(progress_callback, None::<fn(T, T, T, T) -> T>)
    }

    pub(crate) fn downsample_average_int<F>(
        &self,
        progress_callback: F,
    ) -> Result<(), anyhow::Error>
    where
        T: Into<u64> + TryFrom<u64>,
        F: FnMut(String, usize, usize) + Send,
    {
        self.downsample(
            progress_callback,
            Some(|a: T, b: T, c: T, d: T| {
                T::try_from((a.into() + b.into() + c.into() + d.into()) / 4).ok().unwrap()
            }),
        )
    }

    pub(crate) fn downsample<F, Downsample>(
        &self,
        progress_callback: F,
        downsample_func: Option<Downsample>,
    ) -> Result<(), anyhow::Error>
    where
        F: FnMut(String, usize, usize) + Send,
        Downsample: Fn(T, T, T, T) -> T + Send + Sync,
    {
        let cogs = self
            .cogs()?
            .into_iter()
            .map(|(_, cog)| {
                let mut valid = Vec::new();
                for level in 1..cog.levels() {
                    valid.push(cog.valid_mask(level)?);
                }
                Ok((cog, valid))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        let progress_callback = Mutex::new(progress_callback);
        let total = cogs.iter().flat_map(|(_, v)| v.iter().map(|vv| vv.len())).sum();
        let completed = AtomicUsize::new(
            cogs.iter()
                .flat_map(|(_, v)| v.iter().flat_map(|vv| vv.iter().filter(|vvv| **vvv)))
                .count(),
        );

        let bands = self.bits_per_sample.len();
        let resolution = cogbuilder::TILE_SIZE as usize;
        cogs.into_par_iter().try_for_each(|(mut cog, valid_masks)| -> Result<(), anyhow::Error> {
            for level in 1..cog.levels() {
                let valid = &valid_masks[level as usize - 1];
                let tiles_across = cog.tiles_across(level);
                let parent_tiles_across = cog.tiles_across(level - 1);

                for tile in 0..valid.len() as u32 {
                    if valid[tile as usize] {
                        continue;
                    }

                    progress_callback.lock().unwrap()(
                        format!("downsampling {}...", self.dataset_name),
                        completed.load(std::sync::atomic::Ordering::SeqCst),
                        total,
                    );

                    let x = tile % tiles_across;
                    let y = tile / tiles_across;
                    let parents = [
                        cog.read_tile(level - 1, y * 2 * parent_tiles_across + x * 2)?,
                        cog.read_tile(level - 1, y * 2 * parent_tiles_across + x * 2 + 1)?,
                        cog.read_tile(level - 1, (y * 2 + 1) * parent_tiles_across + x * 2)?,
                        cog.read_tile(level - 1, (y * 2 + 1) * parent_tiles_across + x * 2 + 1)?,
                    ];

                    if parents.iter().all(Option::is_none) {
                        cog.write_nodata_tile(level, tile)?;
                        completed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        continue;
                    }

                    let mut parent_tiles = [None, None, None, None];
                    for (input, output) in parents.into_iter().zip(parent_tiles.iter_mut()) {
                        if let Some(input) = input {
                            let mut v = vec![self.no_data_value; resolution * resolution * bands];
                            bytemuck::cast_slice_mut(&mut v)
                                .copy_from_slice(&*cogbuilder::decompress_tile(&input)?);
                            *output = Some(v);
                        }
                    }

                    let mut downsampled = vec![self.no_data_value; resolution * resolution * bands];
                    for (i, parent) in
                        parent_tiles.into_iter().enumerate().filter_map(|(i, t)| t.map(|t| (i, t)))
                    {
                        let base_x = (i % 2) * (resolution / 2);
                        let base_y = (i / 2) * (resolution / 2);
                        for px in [0..resolution / 2, 0..resolution / 2, 0..bands]
                            .into_iter()
                            .multi_cartesian_product()
                        {
                            let (y, x, b) = (px[0], px[1], px[2]);
                            let (x2, y2) = (x * 2, y * 2);

                            if let Some(downsample_func) = &downsample_func {
                                let t00 = parent[(y2 * resolution + x2) * bands + b];
                                let t01 = parent[(y2 * resolution + x2 + 1) * bands + b];
                                let t10 = parent[((y2 + 1) * resolution + x2) * bands + b];
                                let t11 = parent[((y2 + 1) * resolution + x2 + 1) * bands + b];
                                let v = downsample_func(t00, t01, t10, t11);
                                downsampled[((base_y + y) * resolution + base_x + x) * bands + b] =
                                    v;
                            } else {
                                downsampled[((base_y + y) * resolution + base_x + x) * bands + b] =
                                    parent[(y2 * resolution + x2) * bands + b];
                            }
                        }
                    }

                    let compressed = cogbuilder::compress_tile(bytemuck::cast_slice(&*downsampled));
                    cog.write_tile(level, tile, &compressed)?;
                    completed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
            Ok(())
        })
    }
}

fn crop<T: Copy>(output: &mut [T], output_resolution: usize, input: &[T], input_resolution: usize) {
    assert!(input_resolution > output_resolution);
    assert!((input_resolution - output_resolution) % 2 == 0);
    let offset = (input_resolution - output_resolution) / 2;

    for y in 0..output_resolution {
        output[output_resolution * y..][..output_resolution].copy_from_slice(
            &input[input_resolution * (y + offset) + offset..][..output_resolution],
        )
    }
}

fn delta_encode<T: WrappingAdd + WrappingSub + Zero>(data: &mut [T]) {
    let mut last = T::zero();
    for v in data.iter_mut() {
        *v = v.wrapping_sub(&last);
        last = last.wrapping_add(v);
    }
}

fn compress<T: Pod + PartialEq + Zero>(data: &[T]) -> Vec<u8> {
    let mut bytes = Vec::<u8>::new();
    if data.iter().any(|&h| h != T::zero()) {
        let mut e = lz4::EncoderBuilder::new().level(1).build(&mut bytes).unwrap();
        e.write_all(&bytemuck::cast_slice(data)).unwrap();
        e.finish().0;
    }
    bytes
}

pub(crate) fn merge_datasets_to_tiles<F>(
    base_directory: PathBuf,
    copernicus_hgt: Dataset<i16>,
    copernicus_wbm: Dataset<u8>,
    treecover: Dataset<u8>,
    blue_marble: Dataset<u8>,
    progress_callback: F,
) -> Result<(), anyhow::Error>
where
    F: FnMut(String, usize, usize) + Send,
{
    let progress_callback = Mutex::new(progress_callback);
    let serve_directory = base_directory.join("serve");
    let (tiles_directory, existing_tiles) = scan_directory(&serve_directory, "tiles")?;

    // const INPUT_BORDER_SIZE: u32 = 4;
    const TILE_INNER_RESOLUTION: u32 = 512;
    let max_level = VNode::LEVEL_CELL_76M;
    // let min_level = VNode::LEVEL_CELL_1KM.min(max_level);

    let mut total_tiles = 0;
    let mut missing_tiles = Vec::new();
    VNode::breadth_first(|n| {
        let filename = format!("{}_{}_{}x{}.raw", n.level(), VFace(n.face()), n.x(), n.y());

        total_tiles += 1;
        if !existing_tiles.contains(&filename) {
            missing_tiles.push((tiles_directory.join(filename), n));
        }

        n.level() < max_level
    });

    // let mut cogs = vec![
    //     copernicus_hgt.cogs()?.into_iter(),
    //     copernicus_wbm.cogs()?.into_iter(),
    //     treecover.cogs()?.into_iter(),
    //     blue_marble.cogs()?.into_iter(),
    // ];
    let cogs: Vec<Vec<_>> = vec![
        copernicus_hgt.cogs()?.into_iter().map(|(_, c)| c).collect(),
        copernicus_wbm.cogs()?.into_iter().map(|(_, c)| c).collect(),
        treecover.cogs()?.into_iter().map(|(_, c)| c).collect(),
        blue_marble.cogs()?.into_iter().map(|(_, c)| c).collect(),
    ];
    let num_layers = cogs.len();
    let cog_levels: Vec<_> = cogs.iter().map(|c| c[0].levels()).collect();
    let cogs = CogTileCache::new(cogs);
    let grid_registration = vec![true, true, false, false];
    let bytes_per_element = vec![2, 1, 1, 3];
    let leaf_border_size = vec![2, 16, 2, 2];

    let no_data_values: Vec<Vec<u8>> = [
        bytemuck::bytes_of(&copernicus_hgt.no_data_value),
        bytemuck::bytes_of(&copernicus_wbm.no_data_value),
        bytemuck::bytes_of(&treecover.no_data_value),
        bytemuck::bytes_of(&blue_marble.no_data_value),
    ]
    .into_iter()
    .map(|slice| slice.into_iter().cycle().cloned().take(1024).collect())
    .collect();

    // let mut faces = Vec::new();
    // for tiles in missing_tiles {
    //     //let cogs: Vec<_> = cogs.iter_mut().map(|c| c.next().unwrap().1).collect();
    //     faces.push((tiles/* , cogs*/));
    // }

    let tiles_processed = AtomicUsize::new(total_tiles - missing_tiles.len());
    missing_tiles.into_par_iter().try_for_each(
        |(filename, node)| -> Result<(), anyhow::Error> {
            //assert!(node.face() as usize == i);
            progress_callback.lock().unwrap()(
                "Generating tiles...".to_string(),
                tiles_processed.load(Ordering::SeqCst),
                total_tiles,
            );

            let mut layers = (0..num_layers)
                .into_par_iter()
                .map(|layer| -> Result<Option<AlignedBuf>, anyhow::Error> {
                    if node.level() >= cog_levels[layer] as u8 {
                        return Ok(None);
                    }

                    let is_leaf = node.level() + 1 == cog_levels[layer] as u8;
                    let border = if is_leaf { leaf_border_size[layer] } else { 2 };
                    let cog_level = cog_levels[layer] - node.level() as u32 - 1;
                    let resolution =
                        if grid_registration[layer] { 513 + 2 * border } else { 512 + 2 * border };

                    let min_x = node.x() * TILE_INNER_RESOLUTION
                        + (Dataset::<u8>::BORDER_SIZE << node.level())
                        - border;
                    let min_y = node.y() * TILE_INNER_RESOLUTION
                        + (Dataset::<u8>::BORDER_SIZE << node.level())
                        - border;
                    let min_tile_x = min_x / cogbuilder::TILE_SIZE;
                    let min_tile_y = min_y / cogbuilder::TILE_SIZE;
                    let max_tile_x = (min_x + resolution - 1) / cogbuilder::TILE_SIZE;
                    let max_tile_y = (min_y + resolution - 1) / cogbuilder::TILE_SIZE;

                    let mut buf = AlignedBuf::new(
                        resolution as usize * resolution as usize * bytes_per_element[layer],
                    );
                    buf.as_slice_mut::<u8>().chunks_mut(1024).for_each(|c: &mut [u8]| {
                        let s = &no_data_values[layer][..c.len()];
                        c.copy_from_slice(&s)
                    });
                    for tile_y in min_tile_y..=max_tile_y {
                        for tile_x in min_tile_x..=max_tile_x {
                            let tile = tile_y * cogs.tiles_across(layer as u8, cog_level) + tile_x;

                            let contents =
                                cogs.get(layer as u8, node.face(), cog_level as u8, tile)?;
                            let contents = match contents {
                                ref c if c.is_some() => (**c).as_ref().unwrap(),
                                _ => {
                                    continue;
                                }
                            };

                            let min_rect_x = min_x.max(tile_x * cogbuilder::TILE_SIZE);
                            let min_rect_y = min_y.max(tile_y * cogbuilder::TILE_SIZE);
                            let max_rect_x =
                                (min_x + resolution).min((tile_x + 1) * cogbuilder::TILE_SIZE);
                            let max_rect_y =
                                (min_y + resolution).min((tile_y + 1) * cogbuilder::TILE_SIZE);

                            for y in min_rect_y..max_rect_y {
                                let src_offset = ((y - tile_y * cogbuilder::TILE_SIZE)
                                    * cogbuilder::TILE_SIZE
                                    + (min_rect_x - tile_x * cogbuilder::TILE_SIZE))
                                    as usize
                                    * bytes_per_element[layer];

                                let dst_offset = ((y - min_y) * resolution + min_rect_x - min_x)
                                    as usize
                                    * bytes_per_element[layer];

                                let bytes =
                                    (max_rect_x - min_rect_x) as usize * bytes_per_element[layer];

                                buf.as_slice_mut()[dst_offset..][..bytes]
                                    .copy_from_slice(&contents[src_offset..][..bytes]);
                            }
                        }
                    }

                    Ok(Some(buf))
                })
                .collect::<Result<Vec<_>, anyhow::Error>>()?;

            let zip_options = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            let mut zip = ZipWriter::new(Cursor::new(Vec::new()));

            let mut compressed_layers = BTreeMap::new();
            if node.level() + 1 == cog_levels[0] as u8 {
                let mut heights = layers[0].as_mut().unwrap().as_slice_mut::<i16>().to_vec();
                let mut raw_watermask = layers[1].as_mut().unwrap().as_slice_mut::<u8>().to_vec();

                let mut water_surface = heights.clone();

                if raw_watermask.iter().all(|&w| w != 0) {
                    heights.iter_mut().for_each(|h| *h = -1024);
                } else {
                    raw_watermask.iter_mut().for_each(|w| *w = if *w == 0 || *w == 3 { 1 } else { 0 });
                    assert_eq!(raw_watermask.len(), 545 * 545);
                    let watermask =
                        image::GrayImage::from_raw(513 + 32, 513 + 32, raw_watermask).unwrap();
                    let shore_distance =
                        imageproc::distance_transform::euclidean_squared_distance_transform(
                            &watermask,
                        );
                    for y in 0..517 {
                        for x in 0..517 {
                            let dist: f64 =
                                shore_distance.get_pixel(x as u32 + 14, y as u32 + 14)[0];
                            if dist > 0.0 {
                                heights[y * 517 + x] -= ((dist as f32).sqrt() * 4.0) as i16;
                            }
                        }
                    }
                    // for y in 0..517 {
                    //     for x in 0..517 {
                    //         if watermask.get_pixel(y as u32 + 14, x as u32 + 14)[0] == 1 {
                    //             water_surface[y * 517 + x] = -9999;
                    //         }
                    //     }
                    // }
                    // for _ in 0..3 {
                    //     for y in 0..517 {
                    //         for x in 0..517 {
                    //             if water_surface[y * 517 + x] == -9999 {
                    //                 let w0 = water_surface[y * 517 + x.saturating_sub(1)];
                    //                 let w1 = water_surface[y.saturating_sub(1) * 517 + x];
                    //                 let w2 = water_surface[y * 517 + (x + 1).min(516)];
                    //                 let w3 =
                    //                     water_surface[(y + 1).min(516) * 517 + (x + 1).min(516)];
                    //                 water_surface[y * 517 + x] = w0.max(w1).max(w2).max(w3);
                    //             }
                    //         }
                    //     }
                    // }
                    for y in 0..517 {
                        for x in 0..517 {
                            if watermask.get_pixel(x as u32 + 14, y as u32 + 14)[0] == 1 {
                                heights[y * 517 + x] = heights[y * 517 + x].max(1);//  .max(water_surface[y * 517 + x] + 1);
                            }
                        }
                    }
                }
                heights.iter_mut().filter(|&&mut h| h > 8 || h < -8).for_each(|h| *h = (*h / 4) * 4);
                delta_encode(&mut heights);
                compressed_layers.insert("heights.lz4", compress(&heights));

                // water_surface.iter_mut().for_each(|h| *h = (*h / 4) * 4);
                // delta_encode(&mut water_surface);
                // compressed_layers.insert("water_surface.lz4", compress(&water_surface));

                // let watermask = layers[1].as_mut().unwrap().as_slice_mut::<u8>();
                // watermask.iter_mut().for_each(|h| *h = h.wrapping_sub(1));
                // let mut cropped_watermask = vec![0; 517 * 517];
                // crop(&mut cropped_watermask, 517, watermask, 521);
                // delta_encode(&mut cropped_watermask);
                // compressed_layers.insert("watermask.lz4", compress(&cropped_watermask));

                let treecover = layers[2].as_mut().unwrap().as_slice_mut::<u8>();
                delta_encode(treecover);
                compressed_layers.insert("treecover.lz4", compress(treecover));
            } else {
                let heights = layers[0].as_mut().unwrap().as_slice_mut::<i16>();
                heights.iter_mut().for_each(|h| *h = (*h / 4) * 4);
                delta_encode(heights);
                compressed_layers.insert("heights.lz4", compress(heights));

                let watermask = layers[1].as_mut().unwrap().as_slice_mut::<u8>();
                watermask.iter_mut().for_each(|h| *h = h.wrapping_sub(1));
                delta_encode(watermask);
                compressed_layers.insert("watermask.lz4", compress(watermask));

                let treecover = layers[2].as_mut().unwrap().as_slice_mut::<u8>();
                delta_encode(treecover);
                compressed_layers.insert("treecover.lz4", compress(treecover));

                if let Some(layer) = layers[3].as_mut() {
                    if layer.as_slice::<u8>().iter().all(|v| *v == 0) {
                        compressed_layers.insert("albedo.basis", Vec::new());
                    } else {
                        let mut albedo_params = basis_universal::encoding::CompressorParams::new();
                        albedo_params.set_basis_format(basis_universal::BasisTextureFormat::ETC1S);
                        // albedo_params.set_etc1s_quality_level(basis_universal::ETC1S_QUALITY_MIN);
                        // albedo_params.set_generate_mipmaps(true);
                        albedo_params.source_image_mut(0).init(layer.as_slice(), 516, 516, 3);

                        let mut compressor = basis_universal::encoding::Compressor::new(1);
                        unsafe { compressor.init(&albedo_params) };
                        unsafe { compressor.process().unwrap() };

                        compressed_layers.insert("albedo.basis", compressor.basis_file().to_vec());
                    }
                }
            };

            let mut all_empty = true;
            for (name, data) in compressed_layers.iter() {
                if !data.is_empty() {
                    all_empty = false;
                    zip.start_file(name.to_string(), zip_options)?;
                    zip.write_all(&data)?;
                }
            }
            let bytes = if !all_empty { zip.finish().unwrap().into_inner() } else { Vec::new() };

            AtomicFile::new(filename, OverwriteBehavior::AllowOverwrite)
                .write(|f| f.write_all(&bytes))?;
            tiles_processed.fetch_add(1, Ordering::SeqCst);

            Ok(())
        },
    )?;

    // Write tile list
    let tile_list_path = serve_directory.join("tile_list.txt.lz4");
    if !tile_list_path.exists() {
        let mut list = Vec::new();
        for entry in fs::read_dir(tiles_directory)? {
            let entry = entry?;
            if entry.metadata()?.len() > 0 {
                if let Ok(s) = entry.file_name().into_string() {
                    list.push(s);
                }
            }
        }
        list.sort();
        let mut list_compressed = Vec::new();
        let mut e = lz4::EncoderBuilder::new().level(9).build(&mut list_compressed).unwrap();
        e.write_all(&list.join("\n").into_bytes()).unwrap();
        e.finish().0;
        AtomicFile::new(tile_list_path, OverwriteBehavior::AllowOverwrite)
            .write(|f| f.write_all(&list_compressed))?;
    }

    Ok(())
}

pub(crate) async fn generate_materials<F: FnMut(String, usize, usize) + Send>(
    mapfile: &MapFile,
    free_pbr_directory: PathBuf,
    mut progress_callback: F,
) -> Result<(), Error> {
    if mapfile.reload_texture("ground_albedo") {
        return Ok(());
    }

    let mut albedo_params = basis_universal::encoding::CompressorParams::new();
    albedo_params.set_basis_format(basis_universal::BasisTextureFormat::UASTC4x4);
    albedo_params.set_generate_mipmaps(true);

    let materials = [("ground", "leafy-grass2"), ("ground", "grass1"), ("rocks", "granite5")];

    for (i, (group, name)) in materials.iter().enumerate() {
        let path = free_pbr_directory.join(format!("Blender/{}-bl/{}-bl", group, name));

        let mut albedo_path = None;
        for file in std::fs::read_dir(&path)? {
            let file = file?;
            let filename = file.file_name();
            let filename = filename.to_string_lossy();
            if filename.contains("albedo") {
                albedo_path = Some(file.path());
            }
        }

        let mut albedo = image::open(albedo_path.unwrap())?.to_rgb8();
        //material::high_pass_filter(&mut albedo);
        assert_eq!(albedo.width(), 2048);
        assert_eq!(albedo.height(), 2048);

        albedo =
            image::imageops::resize(&albedo, 1024, 1024, image::imageops::FilterType::Triangle);

        albedo_params.source_image_mut(i as u32).init(&*albedo, 1024, 1024, 3);
    }

    progress_callback("Compressing ground albedo textures".to_owned(), 0, 1);
    let mut compressor = basis_universal::encoding::Compressor::new(8);
    unsafe { compressor.init(&albedo_params) };
    unsafe { compressor.process().unwrap() };
    progress_callback("Compressing ground albedo textures".to_owned(), 1, 1);

    let albedo_desc = TextureDescriptor {
        width: 1024,
        height: 1024,
        depth: materials.len() as u32,
        format: TextureFormat::UASTC,
        array_texture: true,
    };

    mapfile.write_texture("ground_albedo", albedo_desc, compressor.basis_file())?;

    Ok(())
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

        context.reset("Generating noise textures... ", noise_heightmaps.len());

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
