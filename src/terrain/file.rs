use std::error::Error;
use std::f64::consts::PI;
use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::*;
use gfx;
use progress::Bar;
use rand;
use rand::distributions::{Normal, IndependentSample};

use cache::{GeneratedAsset, MMappedAsset, WebAsset};
use sky::Skybox;
use terrain::dem::{DemSource, DigitalElevationModelParams};
use terrain::heightmap::{self, Heightmap};
use terrain::material::MaterialSet;
use terrain::quadtree::{node, Node, NodeId, QuadTree};
use terrain::raster::RasterCache;
use terrain::tile_cache::{TileHeader, LayerParams, LayerType, NoiseParams};
use terrain::treecover::TreeCoverParams;
use runtime_texture::TextureFormat;
use utils::math::BoundingBox;

// This file assumes that all coordinates are provided relative to the earth represented as a
// perfect sphere. This isn't quite accurate: The coordinate system of the input datasets are
// actually WGS84 or NAD83. However, for our purposes the difference should not be noticable.

pub struct TerrainFileParams<R: gfx::Resources> {
    pub latitude: i16,
    pub longitude: i16,
    pub source: DemSource,
    pub materials: MaterialSet<R>,
    pub sky: Skybox<R>,
}
impl<R: gfx::Resources> TerrainFileParams<R> {
    pub fn build_quadtree<F: gfx::Factory<R>>(
        self,
        factory: F,
        color_buffer: &gfx::handle::RenderTargetView<R, gfx::format::Srgba8>,
        depth_buffer: &gfx::handle::DepthStencilView<R, gfx::format::DepthStencil>,
    ) -> Result<QuadTree<R, F>, Box<Error>> {
        let (header, data) = self.load()?;

        Ok(QuadTree::new(
            header,
            data,
            self.materials,
            self.sky,
            factory,
            color_buffer,
            depth_buffer,
        ))
    }
}

impl<R: gfx::Resources> MMappedAsset for TerrainFileParams<R> {
    type Header = TileHeader;

    fn filename(&self) -> String {
        let n_or_s = if self.latitude >= 0 { 'n' } else { 's' };
        let e_or_w = if self.longitude >= 0 { 'e' } else { 'w' };
        format!(
            "maps/{}{:02}_{}{:03}_{}m",
            n_or_s,
            self.latitude.abs(),
            e_or_w,
            self.longitude.abs(),
            self.source.resolution(),
        )
    }

    fn generate<W: Write>(&self, mut writer: W) -> Result<Self::Header, Box<Error>> {
        let mut layers = Vec::new();
        let mut bytes_written = 0;

        let mut dem_cache = RasterCache::new(
            |latitude, longitude| {
                DigitalElevationModelParams {
                    latitude,
                    longitude,
                    source: self.source,
                }.load()
                    .ok()
            },
            32,
        );

        let mut treecover_cache = RasterCache::new(
            |latitude, longitude| {
                Some(
                    TreeCoverParams {
                        latitude,
                        longitude,
                    }.load()
                        .unwrap(),
                )
            },
            32,
        );

        let world_center =
            Vector2::<f32>::new(self.longitude as f32 + 0.5, self.latitude as f32 + 0.5);
        let sun_direction = Vector3::new(0.0, 1.0, 1.0).normalize();

        let scale_x = (1.0 / 360.0) * (EARTH_CIRCUMFERENCE as f32) *
            world_center.y.to_radians().cos();
        let scale_y = (-1.0 / 360.0) * EARTH_CIRCUMFERENCE as f32;

        // Cell size in the y (latitude) direction, in meters. The x (longitude) direction will have
        // smaller cell sizes due to the projection.
        let dem_cell_size_y = self.source.cell_size() / (360.0 * 60.0 * 60.0) *
            EARTH_CIRCUMFERENCE as f32;

        const HEIGHTS_RESOLUTION: u16 = 65;
        const TEXTURE_RESOLUTION: u16 = 1025;

        let resolution_ratio = ((TEXTURE_RESOLUTION - 1) / (HEIGHTS_RESOLUTION - 1)) as u16;

        let world_size = 1048576.0 / 2.0;
        let max_level = 12i32;
        let max_texture_level = max_level - (resolution_ratio as f32).log2() as i32;

        let cell_size = world_size / ((HEIGHTS_RESOLUTION - 1) as f32) * (0.5f32).powi(max_level);
        let num_fractal_levels = (dem_cell_size_y / cell_size).log2().ceil().max(0.0) as i32;
        let max_dem_level = max_texture_level - num_fractal_levels.max(0).min(max_texture_level);

        // Amount of space outside of tile that is included in heightmap. Used for computing
        // normals and such. Must be even.
        let skirt = 4;
        assert_eq!(skirt % 2, 0);

        // Heightmaps for all nodes in layers 0...max_texture_level.
        let mut heightmaps: Vec<Heightmap<f32>> = Vec::new();
        // Resolution of each heightmap stored in heightmaps. They are at higher resolution that
        // HEIGHTS_RESOLUTION so that the more detailed textures can be derived from them.
        let heightmap_resolution = TEXTURE_RESOLUTION as u16 + 2 * skirt;

        let in_skirt = |x, y| -> bool {
            x < skirt && y < skirt || x >= heightmap_resolution - skirt ||
                y >= heightmap_resolution - skirt
        };

        let world_position = |x, y, bounds: BoundingBox| -> Vector2<f32> {
            let fx = (x - skirt as i32) as f32 / (heightmap_resolution - 1 - 2 * skirt) as f32;
            let fy = (y - skirt as i32) as f32 / (heightmap_resolution - 1 - 2 * skirt) as f32;

            let world_position = Vector2::<f32>::new(
                (bounds.min.x + (bounds.max.x - bounds.min.x) * fx) / scale_x,
                (bounds.min.z + (bounds.max.z - bounds.min.z) * fy) / scale_y,
            );

            world_center + world_position
        };

        let random = {
            let normal = Normal::new(0.0, 1.0);
            let v = (0..(15 * 15))
                .map(|_| normal.ind_sample(&mut rand::thread_rng()) as f32)
                .collect();
            Heightmap::new(v, 15, 15)
        };

        let mut nodes = Node::make_nodes(world_size, 3000.0, max_level as u8);
        layers.push(LayerParams {
            layer_type: LayerType::Heights,
            offset: 0,
            tile_count: nodes.len(),
            tile_resolution: HEIGHTS_RESOLUTION as u32,
            border_size: 0,
            format: TextureFormat::F32,
            tile_bytes: 4 * HEIGHTS_RESOLUTION as usize * HEIGHTS_RESOLUTION as usize,
        });
        let mut progress_bar = Bar::new();
        progress_bar.set_job_title("Generating heightmaps...");
        for i in 0..nodes.len() {
            progress_bar.reach_percent((100 * i / nodes.len()) as i32);
            nodes[i].tile_indices[LayerType::Heights.index()] = Some(i as u32);

            if nodes[i].level as i32 > max_texture_level {
                let (ancestor, generations, mut offset) =
                    Node::find_ancestor(&nodes, NodeId::new(i as u32), |id| {
                        nodes[id].level as i32 <= max_texture_level
                    }).unwrap();

                let ancestor = ancestor.index();
                let offset_scale = 1 << generations;
                let step = resolution_ratio >> generations;
                let ancestor_heightmap = &heightmaps[ancestor];
                offset *= (heightmap_resolution - 2 * skirt) as i32 / offset_scale;
                let offset = Vector2::new(offset.x as u16, offset.y as u16);

                for y in 0..HEIGHTS_RESOLUTION {
                    for x in 0..HEIGHTS_RESOLUTION {
                        let height = ancestor_heightmap
                            .get(x * step + offset.x + skirt, y * step + offset.y + skirt)
                            .unwrap();

                        writer.write_f32::<LittleEndian>(height)?;
                        bytes_written += 4;
                    }
                }

            } else if nodes[i].level as i32 > max_dem_level {
                let mut heights = Vec::with_capacity(
                    heightmap_resolution as usize * heightmap_resolution as usize,
                );
                let offset = node::OFFSETS[nodes[i].parent.as_ref().unwrap().1 as usize];
                let offset = Point2::new(
                    skirt / 2 + offset.x as u16 * (heightmap_resolution / 2 - skirt),
                    skirt / 2 + offset.y as u16 * (heightmap_resolution / 2 - skirt),
                );

                let layer_scale = nodes[i].size / (heightmap_resolution - 2 * skirt - 1) as i32;
                let layer_origin = Vector2::new(
                    (nodes[i].center.x - nodes[i].size / 2) / layer_scale,
                    (nodes[i].center.y - nodes[i].size / 2) / layer_scale,
                );

                // Extra scope needed due to lack of support for non-lexical lifetimes.
                {
                    let ph = &heightmaps[nodes[i].parent.as_ref().unwrap().0.index()];
                    for y in 0..heightmap_resolution {
                        for x in 0..heightmap_resolution {
                            let height = if x % 2 == 0 && y % 2 == 0 {
                                ph.at(x / 2 + offset.x, y / 2 + offset.y)
                            } else if x % 2 == 0 {
                                let h0 = ph.at(x / 2 + offset.x, y / 2 + offset.y - 1);
                                let h1 = ph.at(x / 2 + offset.x, y / 2 + offset.y);
                                let h2 = ph.at(x / 2 + offset.x, y / 2 + offset.y + 1);
                                let h3 = ph.at(x / 2 + offset.x, y / 2 + offset.y + 2);
                                -0.0625 * h0 + 0.5625 * h1 + 0.5625 * h2 - 0.0625 * h3
                            } else if y % 2 == 0 {
                                let h0 = ph.at(x / 2 + offset.x - 1, y / 2 + offset.y);
                                let h1 = ph.at(x / 2 + offset.x, y / 2 + offset.y);
                                let h2 = ph.at(x / 2 + offset.x + 1, y / 2 + offset.y);
                                let h3 = ph.at(x / 2 + offset.x + 2, y / 2 + offset.y);
                                -0.0625 * h0 + 0.5625 * h1 + 0.5625 * h2 - 0.0625 * h3
                            } else {
                                let h0 = //rustfmt
                                    ph.at(x / 2 + offset.x - 1, y / 2 + offset.y - 1) * -0.0625 +
                                    ph.at(x / 2 + offset.x - 1, y / 2 + offset.y + 0) * 0.5625 +
                                    ph.at(x / 2 + offset.x - 1, y / 2 + offset.y + 1) * 0.5625 +
                                    ph.at(x / 2 + offset.x - 1, y / 2 + offset.y + 2) * -0.0625;
                                let h1 = //rustfmt
                                    ph.at(x / 2 + offset.x , y / 2 + offset.y - 1) * -0.0625 +
                                    ph.at(x / 2 + offset.x, y / 2 + offset.y + 0) * 0.5625 +
                                    ph.at(x / 2 + offset.x, y / 2 + offset.y + 1) * 0.5625 +
                                    ph.at(x / 2 + offset.x, y / 2 + offset.y + 2) * -0.0625;
                                let h2 = //rustfmt
                                    ph.at(x / 2 + offset.x + 1, y / 2 + offset.y - 1) * -0.0625 +
                                    ph.at(x / 2 + offset.x + 1, y / 2 + offset.y + 0) * 0.5625 +
                                    ph.at(x / 2 + offset.x + 1, y / 2 + offset.y + 1) * 0.5625 +
                                    ph.at(x / 2 + offset.x + 1, y / 2 + offset.y + 2) * -0.0625;
                                let h3 = //rustfmt
                                    ph.at(x / 2 + offset.x + 2, y / 2 + offset.y - 1) * -0.0625 +
                                    ph.at(x / 2 + offset.x + 2, y / 2 + offset.y + 0) * 0.5625 +
                                    ph.at(x / 2 + offset.x + 2, y / 2 + offset.y + 1) * 0.5625 +
                                    ph.at(x / 2 + offset.x + 2, y / 2 + offset.y + 2) * -0.0625;
                                -0.0625 * h0 + 0.5625 * h1 + 0.5625 * h2 - 0.0625 * h3
                            };
                            heights.push(height);
                        }
                    }
                }
                let mut heightmap =
                    Heightmap::new(heights, heightmap_resolution, heightmap_resolution);

                // Compute noise.
                let mut noise = Vec::with_capacity(
                    heightmap_resolution as usize * heightmap_resolution as usize,
                );
                let noise_scale = nodes[i].side_length /
                    (heightmap_resolution - 1 - 2 * skirt) as f32;
                let slope_scale = 0.5 * (heightmap_resolution - 1) as f32 / nodes[i].side_length;
                for y in 0..heightmap_resolution {
                    for x in 0..heightmap_resolution {
                        if (x % 2 != 0 || y % 2 != 0) && x > 0 && y > 0 &&
                            x < heightmap_resolution - 1 &&
                            y < heightmap_resolution - 1 &&
                            heightmap.at(x, y) > 0.0
                        {
                            let slope_x = heightmap.at(x + 1, y) - heightmap.at(x - 1, y);
                            let slope_y = heightmap.at(x, y + 1) - heightmap.at(x, y - 1);
                            let slope = (slope_x * slope_x + slope_y * slope_y).sqrt() *
                                slope_scale;

                            let bias = -noise_scale * 0.3 * (slope - 0.5).max(0.0);

                            let noise_strength = ((slope - 0.2).max(0.0) + 0.05).min(1.0);
                            let wx = layer_origin.x + (x as i32 - skirt as i32);
                            let wy = layer_origin.y + (y as i32 - skirt as i32);
                            noise.push(
                                0.15 * random.get_wrapping(wx as i64, wy as i64) * noise_scale *
                                    noise_strength + bias,
                            );
                        } else {
                            noise.push(0.0);
                        }
                    }
                }

                // Apply noise.
                for y in 0..heightmap_resolution {
                    for x in 0..heightmap_resolution {
                        heightmap.raise(
                            x,
                            y,
                            noise[x as usize + y as usize * heightmap_resolution as usize],
                        );
                    }
                }

                // Write tile.
                let step = (heightmap_resolution - 2 * skirt - 1) / (HEIGHTS_RESOLUTION - 1);
                for y in 0..HEIGHTS_RESOLUTION {
                    for x in 0..HEIGHTS_RESOLUTION {
                        let height = heightmap.get(x * step + skirt, y * step + skirt).unwrap();
                        writer.write_f32::<LittleEndian>(height)?;
                        bytes_written += 4;
                    }
                }

                heightmaps.push(heightmap);
            } else {
                assert_eq!(heightmaps.len(), i);
                let node = &nodes[i];
                let mut heights = Vec::with_capacity(
                    heightmap_resolution as usize * heightmap_resolution as usize,
                );
                for y in 0..(heightmap_resolution as i32) {
                    for x in 0..(heightmap_resolution as i32) {
                        let p = world_position(x, y, node.bounds);
                        let height = dem_cache.interpolate(p.y as f64, p.x as f64).unwrap_or(0.0);
                        heights.push(height);

                        if (x - skirt as i32) % resolution_ratio as i32 == 0 &&
                            (y - skirt as i32) % resolution_ratio as i32 == 0 &&
                            !in_skirt(x as u16, y as u16)
                        {
                            writer.write_f32::<LittleEndian>(height)?;
                            bytes_written += 4;
                        }
                    }
                }

                heightmaps.push(Heightmap::new(
                    heights,
                    heightmap_resolution,
                    heightmap_resolution,
                ));
            }
        }
        progress_bar.reach_percent(100);
        progress_bar.jobs_done();

        // Colors
        assert!(skirt >= 2);
        let colormap_resolution = heightmap_resolution - 5;
        layers.push(LayerParams {
            layer_type: LayerType::Colors,
            offset: bytes_written,
            tile_count: heightmaps.len(),
            tile_resolution: colormap_resolution as u32,
            border_size: skirt as u32 - 2,
            format: TextureFormat::SRGBA,
            tile_bytes: 4 * colormap_resolution as usize * colormap_resolution as usize,
        });
        let rock = self.materials.get_average_albedo(0);
        let grass = self.materials.get_average_albedo(1);
        progress_bar.set_job_title("Generating colormaps...");
        for i in 0..heightmaps.len() {
            progress_bar.reach_percent((100 * i / heightmaps.len()) as i32);
            nodes[i].tile_indices[LayerType::Colors.index()] = Some(i as u32);

            let heights = &heightmaps[i];
            let spacing = nodes[i].side_length / (heightmap_resolution - 2 * skirt) as f32;
            for y in 2..(2 + colormap_resolution) {
                for x in 2..(2 + colormap_resolution) {
                    let p = world_position(x as i32, y as i32, nodes[i].bounds);
                    let treecover = treecover_cache.interpolate(p.y as f64, p.x as f64);

                    let h00 = heights.get(x, y).unwrap();
                    let h01 = heights.get(x, y + 1).unwrap();
                    let h10 = heights.get(x + 1, y).unwrap();
                    let h11 = heights.get(x + 1, y + 1).unwrap();

                    let normal =
                        Vector3::new(h10 + h11 - h00 - h01, 2.0 * spacing, h01 + h11 - h00 - h10)
                            .normalize();
                    let light = (normal.dot(sun_direction).max(0.0) * 255.0) as u8;

                    if normal.y > 0.9 {
                        let t = 1.0 - treecover.unwrap_or(0.0);
                        writer.write_u8((grass[0] as f32 * t) as u8)?;
                        writer.write_u8((grass[1] as f32 * t) as u8)?;
                        writer.write_u8((grass[2] as f32 * t) as u8)?;

                    } else {
                        writer.write_u8(rock[0])?;
                        writer.write_u8(rock[1])?;
                        writer.write_u8(rock[2])?;
                    }
                    writer.write_u8(light)?;
                    bytes_written += 4;
                }
            }
        }
        progress_bar.reach_percent(100);
        progress_bar.jobs_done();

        // Normals
        assert!(skirt >= 2);
        let normalmap_resolution = heightmap_resolution - 5;
        let normalmap_offset = bytes_written;
        let mut normalmap_count = 0;
        progress_bar.set_job_title("Generating normalmaps...");
        for i in 0..heightmaps.len() {
            progress_bar.reach_percent((100 * i / heightmaps.len()) as i32);
            if nodes[i].level as i32 != max_texture_level {
                continue;
            }

            nodes[i].tile_indices[LayerType::Normals.index()] = Some(normalmap_count as u32);
            normalmap_count += 1;

            let heights = &heightmaps[i];
            let spacing = nodes[i].side_length / (heightmap_resolution - 2 * skirt) as f32;
            for y in 2..(2 + normalmap_resolution) {
                for x in 2..(2 + normalmap_resolution) {
                    let h00 = heights.get(x, y).unwrap();
                    let h01 = heights.get(x, y + 1).unwrap();
                    let h10 = heights.get(x + 1, y).unwrap();
                    let h11 = heights.get(x + 1, y + 1).unwrap();

                    let normal =
                        Vector3::new(h10 + h11 - h00 - h01, 2.0 * spacing, h01 + h11 - h00 - h10)
                            .normalize();

                    let splat = if normal.y > 0.9 { 1 } else { 0 };

                    writer.write_u8((normal.x * 127.5 + 127.5) as u8)?;
                    writer.write_u8((normal.y * 127.5 + 127.5) as u8)?;
                    writer.write_u8((normal.z * 127.5 + 127.5) as u8)?;
                    writer.write_u8(splat)?;
                    bytes_written += 4;
                }
            }
        }
        layers.push(LayerParams {
            layer_type: LayerType::Normals,
            offset: normalmap_offset,
            tile_count: normalmap_count,
            tile_resolution: normalmap_resolution as u32,
            border_size: skirt as u32 - 2,
            format: TextureFormat::RGBA8,
            tile_bytes: 4 * normalmap_resolution as usize * normalmap_resolution as usize,
        });
        progress_bar.reach_percent(100);
        progress_bar.jobs_done();

        // Water
        assert!(skirt >= 2);
        let watermap_resolution = heightmap_resolution - 5;
        layers.push(LayerParams {
            layer_type: LayerType::Water,
            offset: bytes_written,
            tile_count: heightmaps.len(),
            tile_resolution: watermap_resolution as u32,
            border_size: skirt as u32 - 2,
            format: TextureFormat::RGBA8,
            tile_bytes: 4 * watermap_resolution as usize * watermap_resolution as usize,
        });
        progress_bar.set_job_title("Generating water masks...");
        for i in 0..heightmaps.len() {
            progress_bar.reach_percent((100 * i / heightmaps.len()) as i32);
            nodes[i].tile_indices[LayerType::Water.index()] = Some(i as u32);

            let heights = &heightmaps[i];
            for y in 2..(2 + watermap_resolution) {
                for x in 2..(2 + watermap_resolution) {
                    let mut w = 0.0f32;
                    if heights.at(x, y) <= 0.0 {
                        w += 0.25;
                    }
                    if heights.at(x + 1, y) <= 0.0 {
                        w += 0.25;
                    }
                    if heights.at(x, y + 1) <= 0.0 {
                        w += 0.25;
                    }
                    if heights.at(x + 1, y + 1) <= 0.0 {
                        w += 0.25;
                    }
                    writer.write_u8(((w * 255.0) as u8))?;
                    writer.write_u8(0)?;
                    writer.write_u8(255)?;
                    writer.write_u8(0)?;
                    bytes_written += 4;
                }
            }
        }
        progress_bar.reach_percent(100);
        progress_bar.jobs_done();

        let noise = NoiseParams {
            offset: bytes_written,
            resolution: 512,
            format: TextureFormat::RGBA8,
            bytes: 4 * 512 * 512,
            wavelength: 1.0 / 64.0,
        };
        let noise_heightmaps = [
            heightmap::wavelet_noise(64, 8),
            heightmap::wavelet_noise(64, 8),
            heightmap::wavelet_noise(64, 8),
            heightmap::wavelet_noise(64, 8),
        ];
        for i in 0..noise_heightmaps[0].heights.len() {
            for j in 0..4 {
                let v = noise_heightmaps[j].heights[i].max(-3.0).min(3.0);
                writer.write_u8((v * 127.5 / 3.0 + 127.5) as u8)?;
                bytes_written += 1;
            }
        }
        assert_eq!(bytes_written, noise.offset + noise.bytes);

        Ok(TileHeader {
            layers,
            noise,
            nodes,
        })
    }
}

/// The radius of the earth in meters.
const EARTH_RADIUS: f64 = 6371000.0;
const EARTH_CIRCUMFERENCE: f64 = 2.0 * PI * EARTH_RADIUS;
