use crate::heightmap::CogTileCache;
use crate::ktx2encode::encode_ktx2;
use aligned_buf::AlignedBuf;
use anyhow::Error;
use atomicwrites::{AtomicFile, OverwriteBehavior};
use bytemuck::Pod;
use cgmath::{InnerSpace, Vector3};
use cogbuilder::CogBuilder;
use itertools::Itertools;
use num_traits::{WrappingAdd, WrappingSub, Zero};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashSet};
use std::io::{Cursor, Read};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{fs, mem};
use std::{io::Write, path::Path, sync::Mutex};
use types::{VFace, VNode};
use zip::ZipWriter;

pub mod download;
pub mod textures;

mod heightmap;
mod ktx2encode;
mod material;
mod noise;
mod sky;

fn cspace_to_polar(position: Vector3<f64>) -> Vector3<f64> {
    let p = Vector3::new(position.x, position.y, position.z).normalize();
    let latitude = f64::asin(p.z);
    let longitude = f64::atan2(p.y, p.x);
    Vector3::new(latitude, longitude, 0.0)
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

pub struct Dataset<T> {
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
                .join("derived")
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

    pub fn reproject<F>(&self, progress_callback: F) -> Result<(), anyhow::Error>
    where
        F: FnMut(String, usize, usize) + Send,
    {
        self.reproject_from(self.dataset_name, self.no_data_value, progress_callback, |_| {})
    }

    pub fn reproject_from<F, G>(
        &self,
        base_dataset_name: &str,
        base_no_data: T,
        progress_callback: F,
        postprocess: G,
    ) -> Result<(), anyhow::Error>
    where
        F: FnMut(String, usize, usize) + Send,
        G: Fn(&mut [T]) + Sync,
    {
        let root_border_size = Self::BORDER_SIZE << self.max_level;
        let root_dimensions = self.root_dimensions();

        let vrt_file = vrt_file::VrtFile::new(
            &self.base_directory.join("download").join(base_dataset_name).join("merged.vrt"),
            self.bits_per_sample.len(),
        )?;

        let mut missing = Vec::new();
        let cogs = self
            .cogs()?
            .into_iter()
            .map(|(root_node, cog)| {
                let mut tiles = Vec::new();
                let tiles_across = cog.tiles_across(0);
                for (tile, _) in cog.valid_mask(0)?.iter().enumerate().filter(|v| !v.1) {
                    tiles.push((
                        lindel::morton_encode([
                            tile as u32 % tiles_across,
                            tile as u32 / tiles_across,
                        ]),
                        tile as u32,
                    ));
                }
                tiles.sort_by_key(|v| v.0);
                for (_, tile) in tiles {
                    missing.push((root_node, tile));
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
                            let polar = cspace_to_polar(cspace);
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
                            let polar = cspace_to_polar(cspace);
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
                        base_no_data;
                        cogbuilder::TILE_SIZE as usize * cogbuilder::TILE_SIZE as usize * bands
                    ];

                vrt_file.batch_lookup(&*coordinates, &mut heightmap);

                drop(coordinates);

                postprocess(&mut *heightmap);

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

    fn derive_dataset_impl<F, G>(
        &self,
        progress_callback: F,
        no_data_values: &[&[u8]],
        input_row_bytes: &[usize],
        input_cogs: &[CogTileCache],
        g: G,
    ) -> Result<(), anyhow::Error>
    where
        F: FnMut(String, usize, usize) + Send,
        G: Fn(&[Vec<u8>], &mut [T]) + Sync,
    {
        let root_dimensions = self.root_dimensions();

        let mut missing = Vec::new();
        let out_cogs = self
            .cogs()?
            .into_iter()
            .map(|(root_node, cog)| {
                for (tile, _) in cog.valid_mask(0)?.iter().enumerate().filter(|v| !v.1) {
                    missing.push((root_node, tile as u32));
                }
                Ok(Mutex::new(cog))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        let out_bands = self.bits_per_sample.len();
        let root_dimensions_tiles = (root_dimensions - 1) / cogbuilder::TILE_SIZE + 1;
        let total_sectors = (6 * root_dimensions_tiles * root_dimensions_tiles) as usize;
        let sectors_processed = AtomicUsize::new(total_sectors - missing.len());

        let no_data = no_data_values
            .iter()
            .zip(input_row_bytes)
            .map(|(v, &n)| {
                v.iter()
                    .cycle()
                    .copied()
                    .take(n * cogbuilder::TILE_SIZE as usize * 9)
                    .collect::<Vec<u8>>()
            })
            .collect::<Vec<_>>();

        let progress_callback = Mutex::new(progress_callback);
        missing.into_iter().par_bridge().try_for_each(
            |(root, tile)| -> Result<(), anyhow::Error> {
                (progress_callback.lock().unwrap())(
                    format!("deriving {}...", self.dataset_name),
                    sectors_processed.load(std::sync::atomic::Ordering::SeqCst),
                    total_sectors,
                );
                let x = tile % root_dimensions_tiles;
                let y = tile / root_dimensions_tiles;

                let mut input_data: Vec<_> = no_data.clone();

                // Load data into input_data.
                for (i, (in_cog, &in_row_bytes)) in
                    input_cogs.iter().zip(input_row_bytes).enumerate()
                {
                    for dx in -1..=1 {
                        if dx == -1 && x == 0 || dx == 1 && x == root_dimensions_tiles - 1 {
                            continue;
                        }
                        for dy in -1..=1 {
                            if dy == -1 && y == 0 || dy == 1 && y == root_dimensions_tiles - 1 {
                                continue;
                            }
                            let tile = in_cog.get(
                                0,
                                root.face(),
                                0,
                                (y as i32 + dy) as u32 * root_dimensions_tiles
                                    + (x as i32 + dx) as u32,
                            )?;
                            if let Some(tile) = tile.as_ref() {
                                for row in 0..cogbuilder::TILE_SIZE as usize {
                                    let input_offset = (dy + 1) as usize
                                        * in_row_bytes
                                        * 3
                                        * cogbuilder::TILE_SIZE as usize
                                        + (dx + 1) as usize * in_row_bytes
                                        + row * in_row_bytes * 3;

                                    let input_row =
                                        &mut input_data[i][input_offset..][..in_row_bytes];
                                    let tile_row = &tile[row * in_row_bytes..][..in_row_bytes];
                                    input_row.copy_from_slice(tile_row);
                                }
                            }
                        }
                    }
                }

                let mut output_data =
                    vec![
                        self.no_data_value;
                        cogbuilder::TILE_SIZE as usize * cogbuilder::TILE_SIZE as usize * out_bands
                    ];

                g(&*input_data, &mut *output_data);

                if output_data.iter().any(|&v| v != self.no_data_value) {
                    let compressed = cogbuilder::compress_tile(bytemuck::cast_slice(&*output_data));
                    let mut cog = out_cogs[root.face() as usize].lock().unwrap();
                    cog.write_tile(0, tile, &compressed)?;
                } else {
                    out_cogs[root.face() as usize].lock().unwrap().write_nodata_tile(0, tile)?;
                }

                sectors_processed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            },
        )?;
        Ok(())
    }

    pub fn derive_dataset<F, G, U>(
        &self,
        input: &Dataset<U>,
        progress_callback: F,
        g: G,
    ) -> Result<(), anyhow::Error>
    where
        F: FnMut(String, usize, usize) + Send,
        G: Fn(&[U], &mut [T]) + Sync,
        U: vrt_file::Scalar + Ord + Copy + bytemuck::Pod + ToString + Send + Sync + 'static,
    {
        assert_eq!(input.max_level, self.max_level);
        assert_eq!(input.grid_registration, self.grid_registration);

        self.derive_dataset_impl(
            progress_callback,
            &[bytemuck::bytes_of(&input.no_data_value)],
            &[cogbuilder::TILE_SIZE as usize * input.bits_per_sample.len() * mem::size_of::<U>()],
            &[CogTileCache::new(vec![input.cogs()?.into_iter().map(|c| c.1).collect()])],
            |input, output| g(bytemuck::cast_slice(&*input[0]), output),
        )
    }

    pub fn derive_dataset2<F, G, U, V>(
        &self,
        input0: &Dataset<U>,
        input1: &Dataset<V>,
        progress_callback: F,
        g: G,
    ) -> Result<(), anyhow::Error>
    where
        F: FnMut(String, usize, usize) + Send,
        G: Fn(&[U], &[V], &mut [T]) + Sync,
        U: vrt_file::Scalar + Ord + Copy + bytemuck::Pod + ToString + Send + Sync + 'static,
        V: vrt_file::Scalar + Ord + Copy + bytemuck::Pod + ToString + Send + Sync + 'static,
    {
        assert_eq!(input0.max_level, self.max_level);
        assert_eq!(input1.max_level, self.max_level);
        assert_eq!(input0.grid_registration, self.grid_registration);
        assert_eq!(input1.grid_registration, self.grid_registration);

        self.derive_dataset_impl(
            progress_callback,
            &[bytemuck::bytes_of(&input0.no_data_value), bytemuck::bytes_of(&input1.no_data_value)],
            &[
                cogbuilder::TILE_SIZE as usize * input0.bits_per_sample.len() * mem::size_of::<U>(),
                cogbuilder::TILE_SIZE as usize * input1.bits_per_sample.len() * mem::size_of::<V>(),
            ],
            &[
                CogTileCache::new(vec![input0.cogs()?.into_iter().map(|c| c.1).collect()]),
                CogTileCache::new(vec![input1.cogs()?.into_iter().map(|c| c.1).collect()]),
            ],
            |input, output| {
                g(bytemuck::cast_slice(&*input[0]), bytemuck::cast_slice(&*input[1]), output)
            },
        )
    }

    pub fn compute_shore_distance<F>(
        &self,
        wbm: &Dataset<u8>,
        progress_callback: F,
    ) -> Result<(), anyhow::Error>
    where
        F: FnMut(String, usize, usize) + Send,
        T: From<i16>,
    {
        self.derive_dataset(wbm, progress_callback, |wbm, output| {
            let mut mask = vec![0; wbm.len()];
            for (i, &v) in wbm.iter().enumerate() {
                mask[i] = if v == 1 || v == 2 || v == 3 { 0 } else { 255 };
            }
            let mut mask_img = image::GrayImage::from_raw(
                cogbuilder::TILE_SIZE * 3,
                cogbuilder::TILE_SIZE * 3,
                mask,
            )
            .unwrap();

            const BORDER: u32 = 257;
            let mut mask_img = image::imageops::crop(
                &mut mask_img,
                cogbuilder::TILE_SIZE - BORDER,
                cogbuilder::TILE_SIZE - BORDER,
                cogbuilder::TILE_SIZE + 2 * BORDER,
                cogbuilder::TILE_SIZE + 2 * BORDER,
            )
            .to_image();

            if mask_img.pixels().all(|&v| v[0] == 0) {
                return;
            }

            let negative_distance =
                imageproc::distance_transform::euclidean_squared_distance_transform(&mask_img);
            image::imageops::invert(&mut mask_img);
            let positive_distance =
                imageproc::distance_transform::euclidean_squared_distance_transform(&mask_img);

            for y in 0..cogbuilder::TILE_SIZE {
                for x in 0..cogbuilder::TILE_SIZE {
                    let p = positive_distance.get_pixel(x + BORDER, y + BORDER)[0];
                    let n = negative_distance.get_pixel(x + BORDER, y + BORDER)[0];
                    output[y as usize * cogbuilder::TILE_SIZE as usize + x as usize] =
                        T::from(if p > n { p.sqrt() * 128.0 } else { n.sqrt() * -128.0 } as i16);
                }
            }
        })
    }

    pub fn compute_water_level<F>(
        &self,
        elevation: &Dataset<i16>,
        water_mask: &Dataset<u8>,
        progress_callback: F,
    ) -> Result<(), anyhow::Error>
    where
        F: FnMut(String, usize, usize) + Send,
        T: From<i16>,
    {
        self.derive_dataset2(
            elevation,
            water_mask,
            progress_callback,
            |elevation, water_mask, output| {
                let dim = cogbuilder::TILE_SIZE as usize;
                for y in 0..dim {
                    for x in 0..dim {
                        let px = dim + x;
                        let py = dim + y;
                        let elevation = elevation[py * 3 * dim + px];
                        let water_mask = water_mask[py * 3 * dim + px];

                        output[y * dim + x] =
                            T::from(if water_mask == 1 || water_mask == 2 || water_mask == 3 {
                                elevation
                            } else {
                                i16::MIN
                            });
                    }
                }

                // Ping-pong between the two buffers to compute a separable 3x3 max filter.
                let mut tmp = output.to_vec();
                for _ in 0..3 {
                    for y in 0..dim {
                        for x in 0..dim {
                            tmp[y * dim + x] = output[y * dim + x]
                                .max(output[y.saturating_sub(1) * dim + x])
                                .max(output[(y + 1).min(dim - 1) * dim + x]);
                        }
                    }
                    for y in 0..dim {
                        for x in 0..dim {
                            output[y * dim + x] = tmp[y * dim + x]
                                .max(tmp[y * dim + x.saturating_sub(1)])
                                .max(tmp[y * dim + (x + 1).min(dim - 1)]);
                        }
                    }
                }
            },
        )
    }

    pub fn downsample_grid<F>(&self, progress_callback: F) -> Result<(), anyhow::Error>
    where
        F: FnMut(String, usize, usize) + Send,
    {
        assert!(self.grid_registration);
        self.downsample(progress_callback, None::<fn(T, T, T, T) -> T>)
    }

    pub fn downsample_average_int<F>(
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

    pub fn downsample<F, Downsample>(
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

#[allow(unused)]
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

fn delta_encode<T: WrappingAdd + WrappingSub + Zero>(data: &mut [T]) -> &mut [T] {
    let mut last = T::zero();
    for v in data.iter_mut() {
        *v = v.wrapping_sub(&last);
        last = last.wrapping_add(v);
    }
    data
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

pub fn merge_datasets_to_tiles<F>(
    base_directory: PathBuf,
    heights_dataset: Dataset<i16>,
    water_level_dataset: Dataset<i16>,
    shore_distance_dataset: Dataset<i16>,
    albedo_dataset: Dataset<u8>,
    tree_cover_dataset: Dataset<u8>,
    land_fraction_dataset: Dataset<u8>,
    progress_callback: F,
) -> Result<(), anyhow::Error>
where
    F: FnMut(String, usize, usize) + Send,
{
    let progress_callback = Mutex::new(progress_callback);
    let serve_directory = base_directory.join("serve");
    let (tiles_directory, existing_tiles) = scan_directory(&serve_directory, "tiles")?;

    const TILE_INNER_RESOLUTION: u32 = 512;
    let max_level = VNode::LEVEL_CELL_76M;

    let tile_list_path = serve_directory.join("tile_list.txt.lz4");
    let all_tiles: Option<HashSet<VNode>> = if tile_list_path.exists() {
        let mut remote_files = String::new();
        lz4::Decoder::new(std::io::Cursor::new(&fs::read(&tile_list_path)?))?
            .read_to_string(&mut remote_files)?;
        Some(
            remote_files
                .split('\n')
                .filter_map(|f| f.strip_suffix(".raw"))
                .map(VNode::from_str)
                .collect::<Result<HashSet<VNode>, Error>>()?,
        )
    } else {
        None
    };

    let mut total_tiles = 0;
    let mut missing_tiles = Vec::new();
    VNode::breadth_first(|n| {
        if all_tiles.is_none() || all_tiles.as_ref().unwrap().contains(&n) {
            total_tiles += 1;
            let filename = format!("{}.raw", n);
            if !existing_tiles.contains(&filename) {
                missing_tiles.push((tiles_directory.join(filename), n));
            }
        }

        n.level() < max_level
    });

    // Layer indices
    const LAYER_HEIGHTS: usize = 0;
    const LAYER_WATER_LEVEL: usize = 1;
    const LAYER_SHORE_DIST: usize = 2;
    const LAYER_ALBEDO: usize = 3;
    const LAYER_TREECOVER: usize = 4;
    const LAYER_LAND_FRACT: usize = 5;

    // Per-layer parameters
    let cogs: Vec<Vec<_>> = vec![
        heights_dataset.cogs()?.into_iter().map(|(_, c)| c).collect(),
        water_level_dataset.cogs()?.into_iter().map(|(_, c)| c).collect(),
        shore_distance_dataset.cogs()?.into_iter().map(|(_, c)| c).collect(),
        albedo_dataset.cogs()?.into_iter().map(|(_, c)| c).collect(),
        tree_cover_dataset.cogs()?.into_iter().map(|(_, c)| c).collect(),
        land_fraction_dataset.cogs()?.into_iter().map(|(_, c)| c).collect(),
    ];
    let grid_registration = vec![true, true, true, false, false, false];
    let bytes_per_element = vec![2, 2, 2, 3, 1, 1];
    let no_data_values: Vec<Vec<u8>> = [
        bytemuck::bytes_of(&heights_dataset.no_data_value),
        bytemuck::bytes_of(&water_level_dataset.no_data_value),
        bytemuck::bytes_of(&shore_distance_dataset.no_data_value),
        bytemuck::bytes_of(&albedo_dataset.no_data_value),
        bytemuck::bytes_of(&tree_cover_dataset.no_data_value),
        bytemuck::bytes_of(&land_fraction_dataset.no_data_value),
    ]
    .into_iter()
    .map(|slice| slice.into_iter().cycle().cloned().take(1024).collect())
    .collect();

    let num_layers = cogs.len();
    let cog_levels: Vec<_> = cogs.iter().map(|c| c[0].levels()).collect();
    let cogs = CogTileCache::new(cogs);

    let tiles_processed = AtomicUsize::new(total_tiles - missing_tiles.len());
    missing_tiles.into_par_iter().try_for_each(
        |(filename, node)| -> Result<(), anyhow::Error> {
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

                    let cog_level = cog_levels[layer] - node.level() as u32 - 1;
                    let border = if grid_registration[layer] { 4 } else { 2 };
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
            let mut layers = layers.iter_mut().map(Option::as_mut).collect::<Vec<_>>();

            let zip_options = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            let mut zip = ZipWriter::new(Cursor::new(Vec::new()));

            let mut compressed_layers = BTreeMap::new();
            // if node.level() + 1 == cog_levels[0] as u8 {
            //     let mut heights = layers[0].as_mut().unwrap().as_slice_mut::<i16>().to_vec();
            //     let mut raw_watermask = layers[1].as_mut().unwrap().as_slice_mut::<u8>().to_vec();

            //     let mut water_surface = heights.clone();

            //     if raw_watermask.iter().all(|&w| w != 0) {
            //         heights.iter_mut().for_each(|h| *h = -1024);
            //     } else {
            //         raw_watermask
            //             .iter_mut()
            //             .for_each(|w| *w = if *w == 0 || *w == 3 { 1 } else { 0 });
            //         assert_eq!(raw_watermask.len(), 545 * 545);
            //         let watermask =
            //             image::GrayImage::from_raw(513 + 32, 513 + 32, raw_watermask).unwrap();
            //         let shore_distance =
            //             imageproc::distance_transform::euclidean_squared_distance_transform(
            //                 &watermask,
            //             );
            //         for y in 0..517 {
            //             for x in 0..517 {
            //                 let dist: f64 =
            //                     shore_distance.get_pixel(x as u32 + 14, y as u32 + 14)[0];
            //                 if dist > 0.0 {
            //                     heights[y * 517 + x] -= ((dist as f32).sqrt() * 4.0) as i16;
            //                 }
            //             }
            //         }
            //         // for y in 0..517 {
            //         //     for x in 0..517 {
            //         //         if watermask.get_pixel(y as u32 + 14, x as u32 + 14)[0] == 1 {
            //         //             water_surface[y * 517 + x] = -9999;
            //         //         }
            //         //     }
            //         // }
            //         // for _ in 0..3 {
            //         //     for y in 0..517 {
            //         //         for x in 0..517 {
            //         //             if water_surface[y * 517 + x] == -9999 {
            //         //                 let w0 = water_surface[y * 517 + x.saturating_sub(1)];
            //         //                 let w1 = water_surface[y.saturating_sub(1) * 517 + x];
            //         //                 let w2 = water_surface[y * 517 + (x + 1).min(516)];
            //         //                 let w3 =
            //         //                     water_surface[(y + 1).min(516) * 517 + (x + 1).min(516)];
            //         //                 water_surface[y * 517 + x] = w0.max(w1).max(w2).max(w3);
            //         //             }
            //         //         }
            //         //     }
            //         // }
            //         for y in 0..517 {
            //             for x in 0..517 {
            //                 if watermask.get_pixel(x as u32 + 14, y as u32 + 14)[0] == 1 {
            //                     heights[y * 517 + x] = heights[y * 517 + x].max(1);
            //                     //  .max(water_surface[y * 517 + x] + 1);
            //                 }
            //             }
            //         }
            //     }
            //     heights
            //         .iter_mut()
            //         .filter(|&&mut h| h > 8 || h < -8)
            //         .for_each(|h| *h = (*h / 4) * 4);
            //     delta_encode(&mut heights);
            //     compressed_layers.insert("heights.lz4", compress(&heights));

            //     // water_surface.iter_mut().for_each(|h| *h = (*h / 4) * 4);
            //     // delta_encode(&mut water_surface);
            //     // compressed_layers.insert("water_surface.lz4", compress(&water_surface));

            //     // let watermask = layers[1].as_mut().unwrap().as_slice_mut::<u8>();
            //     // watermask.iter_mut().for_each(|h| *h = h.wrapping_sub(1));
            //     // let mut cropped_watermask = vec![0; 517 * 517];
            //     // crop(&mut cropped_watermask, 517, watermask, 521);
            //     // delta_encode(&mut cropped_watermask);
            //     // compressed_layers.insert("watermask.lz4", compress(&cropped_watermask));

            //     let treecover = layers[2].as_mut().unwrap().as_slice_mut::<u8>();
            //     delta_encode(treecover);
            //     compressed_layers.insert("treecover.lz4", compress(treecover));

            //     let landfraction = layers[4].as_mut().unwrap().as_slice_mut::<u8>();
            //     delta_encode(landfraction);
            //     compressed_layers.insert("landfraction.lz4", compress(landfraction));
            // } else {

            let heights = layers[LAYER_HEIGHTS].take().unwrap().as_slice_mut::<i16>();
            let water_level = layers[LAYER_WATER_LEVEL].take().unwrap().as_slice_mut::<i16>();
            let shore_distance = layers[LAYER_SHORE_DIST].take().unwrap().as_slice_mut::<i16>();
            let tree_cover = layers[LAYER_TREECOVER].take().unwrap().as_slice_mut::<u8>();
            let land_fraction = layers[LAYER_LAND_FRACT].take().unwrap().as_slice_mut::<u8>();

            heights.iter_mut().for_each(|h| *h = h.saturating_mul(4));
            water_level.iter_mut().for_each(|w| *w = w.saturating_mul(4));

            if node.level() < VNode::LEVEL_CELL_76M {
                heights
                    .iter_mut()
                    .zip(water_level.iter())
                    .for_each(|(h, &w)| *h = w.max((*h / 16) * 16));
            } else {
                heights.iter_mut().zip(water_level.iter()).zip(shore_distance.iter()).for_each(
                    |((h, &w), &d)| {
                        if d > 0 {
                            *h = w.saturating_add(1).max((*h / 16) * 16);
                            assert!(*h > w);
                        } else {
                            *h = (w.saturating_sub(1)).min((h.saturating_add(d / 4) / 16) * 16);
                            assert!(*h < w);
                        }
                    },
                );
                compressed_layers.insert("waterlevel.lz4", compress(delta_encode(water_level)));
            }
            compressed_layers.insert("heights.lz4", compress(delta_encode(heights)));
            compressed_layers.insert("treecover.lz4", compress(delta_encode(tree_cover)));
            compressed_layers.insert("landfraction.lz4", compress(delta_encode(land_fraction)));

            if let Some(ref layer) = layers[LAYER_ALBEDO] {
                if layer.as_slice::<u8>().iter().all(|v| *v == 0) {
                    compressed_layers.insert("albedo.ktx2", Vec::new());
                } else {
                    let layer = layer.as_slice::<u8>();
                    assert_eq!(layer.len(), 516 * 516 * 3);
                    let mut albedo = vec![0; 516 * 516 * 4];
                    for i in 0..516 * 516 {
                        albedo[i * 4] = layer[i * 3];
                        albedo[i * 4 + 1] = layer[i * 3 + 1];
                        albedo[i * 4 + 2] = layer[i * 3 + 2];
                        albedo[i * 4 + 3] = 255;
                    }

                    compressed_layers.insert(
                        "albedo.ktx2",
                        encode_ktx2(&[albedo], 516, 516, 0, 0, ktx2::Format::R8G8B8A8_UNORM)?,
                    );
                }
            }

            let mut all_empty = true;
            for (name, data) in compressed_layers.iter() {
                zip.start_file(name.to_string(), zip_options)?;
                zip.write_all(&data)?;

                if !data.is_empty() {
                    all_empty = false;
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
