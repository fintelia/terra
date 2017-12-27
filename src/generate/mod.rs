use std::error::Error;
use std::f64::consts::PI;
use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::*;
use gfx;
use rand;
use rand::distributions::{IndependentSample, Normal};

use cache::{AssetLoadContext, MMappedAsset, WebAsset};
use coordinates::CoordinateSystem;
use sky::Skybox;
use terrain::dem::DemSource;
use terrain::heightmap::{self, Heightmap};
use terrain::material::MaterialSet;
use terrain::quadtree::{node, Node, NodeId, QuadTree};
use terrain::raster::{GlobalRaster, RasterCache};
use terrain::tile_cache::{LayerParams, LayerType, MeshDescriptor, NoiseParams, TileHeader};
use terrain::landcover::{BlueMarble, LandCoverKind};
use runtime_texture::TextureFormat;
use utils::math::BoundingBox;

/// The radius of the earth in meters.
const EARTH_RADIUS: f64 = 6371000.0;
const EARTH_CIRCUMFERENCE: f64 = 2.0 * PI * EARTH_RADIUS;

const HEIGHTS_RESOLUTION: u16 = 65;
const TEXTURE_RESOLUTION: u16 = 257;

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
        let (header, data) = self.load(&mut AssetLoadContext::new())?;

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

    fn generate<W: Write>(
        &self,
        context: &mut AssetLoadContext,
        writer: W,
    ) -> Result<Self::Header, Box<Error>> {
        let world_center =
            Vector2::<f32>::new(self.longitude as f32 + 0.5, self.latitude as f32 + 0.5);
        let sun_direction = Vector3::new(0.0, 1.0, 1.0).normalize();

        // Cell size in the y (latitude) direction, in meters. The x (longitude) direction will have
        // smaller cell sizes due to the projection.
        let dem_cell_size_y =
            self.source.cell_size() / (360.0 * 60.0 * 60.0) * EARTH_CIRCUMFERENCE as f32;

        let resolution_ratio = ((TEXTURE_RESOLUTION - 1) / (HEIGHTS_RESOLUTION - 1)) as u16;

        let world_size = 1048576.0 / 2.0 * 8.0;
        let max_level = 10i32 - 1 + 3;
        let max_texture_level = max_level - (resolution_ratio as f32).log2() as i32;

        let cell_size = world_size / ((HEIGHTS_RESOLUTION - 1) as f32) * (0.5f32).powi(max_level);
        let num_fractal_levels = (dem_cell_size_y / cell_size).log2().ceil().max(0.0) as i32;
        let max_dem_level = max_texture_level - num_fractal_levels.max(0).min(max_texture_level);

        // Amount of space outside of tile that is included in heightmap. Used for computing
        // normals and such. Must be even.
        let skirt = 4;
        assert_eq!(skirt % 2, 0);

        // Heightmaps for all nodes in layers 0...max_texture_level.
        // let heightmaps: Vec<Heightmap<f32>> = Vec::new();
        // Resolution of each heightmap stored in heightmaps. They are at higher resolution that
        // HEIGHTS_RESOLUTION so that the more detailed textures can be derived from them.
        let heightmap_resolution = TEXTURE_RESOLUTION as u16 + 2 * skirt;

        let mut state = State {
            dem_cache: RasterCache::new(Box::new(self.source), 32),
            bluemarble: BlueMarble.load(context)?,
            random: {
                let normal = Normal::new(0.0, 1.0);
                let v = (0..(15 * 15))
                    .map(|_| normal.ind_sample(&mut rand::thread_rng()) as f32)
                    .collect();
                Heightmap::new(v, 15, 15)
            },
            heightmap_resolution,
            max_texture_level,
            resolution_ratio,
            writer,
            heightmaps: Vec::new(),
            max_dem_level,
            materials: &self.materials,
            skirt,
            sun_direction,
            system: CoordinateSystem::from_lla(Vector3::new(
                world_center.y.to_radians() as f64,
                world_center.x.to_radians() as f64,
                0.0,
            )),
            nodes: Node::make_nodes(world_size, 3000.0, max_level as u8),
            layers: Vec::new(),
            bytes_written: 0,
        };

        context.set_progress_and_total(0, 4);
        state.generate_heightmaps(context)?;
        context.set_progress(1);
        state.generate_normalmaps(context)?;
        context.set_progress(2);
        state.generate_colormaps(context)?;
        context.set_progress(3);
        state.generate_watermasks(context)?;
        context.set_progress(4);

        let planet_mesh = state.generate_planet_mesh(context)?;
        let noise = state.generate_noise(context)?;
        let State { layers, nodes, .. } = state;

        Ok(TileHeader {
            planet_mesh,
            layers,
            noise,
            nodes,
        })
    }
}

struct State<'a, W: Write, R: gfx::Resources> {
    dem_cache: RasterCache<f32>,
    bluemarble: GlobalRaster<u8>,

    random: Heightmap<f32>,
    heightmaps: Vec<Heightmap<f32>>,

    skirt: u16,
    heightmap_resolution: u16,
    max_texture_level: i32,
    max_dem_level: i32,
    resolution_ratio: u16,
    writer: W,
    materials: &'a MaterialSet<R>,
    sun_direction: Vector3<f32>,

    system: CoordinateSystem,

    layers: Vec<LayerParams>,
    nodes: Vec<Node>,
    bytes_written: usize,
}

impl<'a, W: Write, R: gfx::Resources> State<'a, W, R> {
    fn world_position(&self, x: i32, y: i32, bounds: BoundingBox) -> Vector2<f64> {
        let fx = (x - self.skirt as i32) as f32
            / (self.heightmap_resolution - 1 - 2 * self.skirt) as f32;
        let fy = (y - self.skirt as i32) as f32
            / (self.heightmap_resolution - 1 - 2 * self.skirt) as f32;

        Vector2::new(
            (bounds.min.x + (bounds.max.x - bounds.min.x) * fx) as f64,
            (bounds.min.z + (bounds.max.z - bounds.min.z) * fy) as f64,
        )
    }

    pub fn write_detail_heightmap(&mut self, i: usize) -> Result<(), Box<Error>> {
        let (ancestor, generations, mut offset) =
            Node::find_ancestor(&self.nodes, NodeId::new(i as u32), |id| {
                self.nodes[id].level as i32 <= self.max_texture_level
            }).unwrap();

        let ancestor = ancestor.index();
        let offset_scale = 1 << generations;
        let step = self.resolution_ratio >> generations;
        let ancestor_heightmap = &self.heightmaps[ancestor];
        offset *= (self.heightmap_resolution - 2 * self.skirt) as i32 / offset_scale;
        let offset = Vector2::new(offset.x as u16, offset.y as u16);

        for y in 0..HEIGHTS_RESOLUTION {
            for x in 0..HEIGHTS_RESOLUTION {
                let height = ancestor_heightmap
                    .get(
                        x * step + offset.x + self.skirt,
                        y * step + offset.y + self.skirt,
                    )
                    .unwrap();

                self.writer.write_f32::<LittleEndian>(height)?;
                self.bytes_written += 4;
            }
        }
        Ok(())
    }

    pub fn generate_interpolated_heightmap(
        &mut self,
        i: usize,
    ) -> Result<Heightmap<f32>, Box<Error>> {
        let mut heights = Vec::with_capacity(
            self.heightmap_resolution as usize * self.heightmap_resolution as usize,
        );
        let offset = node::OFFSETS[self.nodes[i].parent.as_ref().unwrap().1 as usize];
        let offset = Point2::new(
            self.skirt / 2 + offset.x as u16 * (self.heightmap_resolution / 2 - self.skirt),
            self.skirt / 2 + offset.y as u16 * (self.heightmap_resolution / 2 - self.skirt),
        );

        let layer_scale =
            self.nodes[i].size / (self.heightmap_resolution - 2 * self.skirt - 1) as i32;
        let layer_origin = Vector2::new(
            (self.nodes[i].center.x - self.nodes[i].size / 2) / layer_scale,
            (self.nodes[i].center.y - self.nodes[i].size / 2) / layer_scale,
        );

        // Extra scope needed due to lack of support for non-lexical lifetimes.
        {
            let ph = &self.heightmaps[self.nodes[i].parent.as_ref().unwrap().0.index()];
            for y in 0..self.heightmap_resolution {
                for x in 0..self.heightmap_resolution {
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
        let mut heightmap = Heightmap::new(
            heights,
            self.heightmap_resolution,
            self.heightmap_resolution,
        );

        // Compute noise.
        let mut noise = Vec::with_capacity(
            self.heightmap_resolution as usize * self.heightmap_resolution as usize,
        );
        let noise_scale =
            self.nodes[i].side_length / (self.heightmap_resolution - 1 - 2 * self.skirt) as f32;
        let slope_scale = 0.5 * (self.heightmap_resolution - 1) as f32 / self.nodes[i].side_length;
        for y in 0..self.heightmap_resolution {
            for x in 0..self.heightmap_resolution {
                if (x % 2 != 0 || y % 2 != 0) && x > 0 && y > 0 && x < self.heightmap_resolution - 1
                    && y < self.heightmap_resolution - 1
                    && heightmap.at(x, y) > 0.0
                {
                    let slope_x = heightmap.at(x + 1, y) - heightmap.at(x - 1, y);
                    let slope_y = heightmap.at(x, y + 1) - heightmap.at(x, y - 1);
                    let slope = (slope_x * slope_x + slope_y * slope_y).sqrt() * slope_scale;

                    let bias = -noise_scale * 0.3 * (slope - 0.5).max(0.0);

                    let noise_strength = ((slope - 0.2).max(0.0) + 0.05).min(1.0);
                    let wx = layer_origin.x + (x as i32 - self.skirt as i32);
                    let wy = layer_origin.y + (y as i32 - self.skirt as i32);
                    noise.push(
                        0.15 * self.random.get_wrapping(wx as i64, wy as i64) * noise_scale
                            * noise_strength + bias,
                    );
                } else {
                    noise.push(0.0);
                }
            }
        }

        // Apply noise.
        for y in 0..self.heightmap_resolution {
            for x in 0..self.heightmap_resolution {
                heightmap.raise(
                    x,
                    y,
                    noise[x as usize + y as usize * self.heightmap_resolution as usize],
                );
            }
        }

        // Write tile.
        let step = (self.heightmap_resolution - 2 * self.skirt - 1) / (HEIGHTS_RESOLUTION - 1);
        for y in 0..HEIGHTS_RESOLUTION {
            for x in 0..HEIGHTS_RESOLUTION {
                let height = heightmap
                    .get(x * step + self.skirt, y * step + self.skirt)
                    .unwrap();
                self.writer.write_f32::<LittleEndian>(height)?;
                self.bytes_written += 4;
            }
        }
        Ok(heightmap)
    }

    fn generate_heightmaps(&mut self, context: &mut AssetLoadContext) -> Result<(), Box<Error>> {
        self.layers.push(LayerParams {
            layer_type: LayerType::Heights,
            offset: 0,
            tile_count: self.nodes.len(),
            tile_resolution: HEIGHTS_RESOLUTION as u32,
            border_size: 0,
            format: TextureFormat::F32,
            tile_bytes: 4 * HEIGHTS_RESOLUTION as usize * HEIGHTS_RESOLUTION as usize,
        });

        context.increment_level("Generating heightmaps... ", self.nodes.len());
        for i in 0..self.nodes.len() {
            context.set_progress(i as u64);
            self.nodes[i].tile_indices[LayerType::Heights.index()] = Some(i as u32);

            if self.nodes[i].level as i32 > self.max_texture_level {
                self.write_detail_heightmap(i)?;
            } else if self.nodes[i].level as i32 > self.max_dem_level {
                let heightmap = self.generate_interpolated_heightmap(i)?;
                self.heightmaps.push(heightmap);
            } else {
                assert_eq!(self.heightmaps.len(), i);
                let node = &self.nodes[i];
                let mut heights = Vec::with_capacity(
                    self.heightmap_resolution as usize * self.heightmap_resolution as usize,
                );
                for y in 0..(self.heightmap_resolution as i32) {
                    for x in 0..(self.heightmap_resolution as i32) {
                        let world = self.world_position(x, y, node.bounds);
                        let mut world3 = Vector3::new(
                            world.x,
                            EARTH_RADIUS
                                * ((1.0 - world.magnitude2() / EARTH_RADIUS).max(0.25).sqrt()
                                    - 1.0),
                            world.y,
                        );
                        for i in 0..5 {
                            world3.x = world.x;
                            world3.z = world.y;
                            let mut lla = self.system.world_to_lla(world3);
                            lla.z = if i >= 3 && world.magnitude2() < 250000.0 * 250000.0 {
                                self.dem_cache
                                    .interpolate(context, lla.x.to_degrees(), lla.y.to_degrees())
                                    .unwrap_or(0.0) as f64
                            } else {
                                0.0
                            };
                            world3 = self.system.lla_to_world(lla);
                        }
                        heights.push(world3.y as f32);
                    }
                }

                let heightmap = Heightmap::new(
                    heights,
                    self.heightmap_resolution,
                    self.heightmap_resolution,
                );

                // Write tile.
                let step =
                    (self.heightmap_resolution - 2 * self.skirt - 1) / (HEIGHTS_RESOLUTION - 1);
                for y in 0..HEIGHTS_RESOLUTION {
                    for x in 0..HEIGHTS_RESOLUTION {
                        let height = heightmap
                            .get(x * step + self.skirt, y * step + self.skirt)
                            .unwrap();
                        self.writer.write_f32::<LittleEndian>(height)?;
                        self.bytes_written += 4;
                    }
                }

                self.heightmaps.push(heightmap);
            }
        }
        context.decrement_level();
        Ok(())
    }
    fn generate_colormaps(&mut self, context: &mut AssetLoadContext) -> Result<(), Box<Error>> {
        assert!(self.skirt >= 2);
        let colormap_resolution = self.heightmap_resolution - 5;
        self.layers.push(LayerParams {
            layer_type: LayerType::Colors,
            offset: self.bytes_written,
            tile_count: self.heightmaps.len(),
            tile_resolution: colormap_resolution as u32,
            border_size: self.skirt as u32 - 2,
            format: TextureFormat::SRGBA,
            tile_bytes: 4 * colormap_resolution as usize * colormap_resolution as usize,
        });
        let rock = self.materials.get_average_albedo(0);
        let grass = self.materials.get_average_albedo(1);
        context.increment_level("Generating colormaps... ", self.heightmaps.len());
        let mut treecover_cache = RasterCache::new(Box::new(LandCoverKind::TreeCover), 256);
        for i in 0..self.heightmaps.len() {
            context.set_progress(i as u64);
            self.nodes[i].tile_indices[LayerType::Colors.index()] = Some(i as u32);

            let heights = &self.heightmaps[i];
            let spacing =
                self.nodes[i].side_length / (self.heightmap_resolution - 2 * self.skirt) as f32;
            let use_blue_marble = spacing >= self.bluemarble.spacing() as f32;
            for y in 2..(2 + colormap_resolution) {
                for x in 2..(2 + colormap_resolution) {
                    let world = self.world_position(x as i32, y as i32, self.nodes[i].bounds);
                    let h00 = heights.get(x, y).unwrap();
                    let h01 = heights.get(x, y + 1).unwrap();
                    let h10 = heights.get(x + 1, y).unwrap();
                    let h11 = heights.get(x + 1, y + 1).unwrap();
                    let h = (h00 + h01 + h10 + h11) as f64 * 0.25;
                    let lla = self.system.world_to_lla(Vector3::new(world.x, h, world.y));

                    if use_blue_marble {
                        self.writer.write_u8(self.bluemarble.interpolate(
                            lla.x.to_degrees(),
                            lla.y.to_degrees(),
                            0,
                        ) as u8)?;
                        self.writer.write_u8(self.bluemarble.interpolate(
                            lla.x.to_degrees(),
                            lla.y.to_degrees(),
                            1,
                        ) as u8)?;
                        self.writer.write_u8(self.bluemarble.interpolate(
                            lla.x.to_degrees(),
                            lla.y.to_degrees(),
                            2,
                        ) as u8)?;
                        self.writer.write_u8(255)?;
                        self.bytes_written += 4;
                        continue;
                    };

                    let normal =
                        Vector3::new(h10 + h11 - h00 - h01, 2.0 * spacing, h01 + h11 - h00 - h10)
                            .normalize();
                    let light = (normal.dot(self.sun_direction).max(0.0) * 255.0) as u8;

                    if normal.y > 0.9 {
                        let treecover = treecover_cache.interpolate(
                            context,
                            lla.x.to_degrees(),
                            lla.y.to_degrees(),
                        );
                        let t = 1.0 - 0.4 * treecover.unwrap_or(0.0) / 100.0;
                        self.writer.write_u8((grass[0] as f64 * t) as u8)?;
                        self.writer.write_u8((grass[1] as f64 * t) as u8)?;
                        self.writer.write_u8((grass[2] as f64 * t) as u8)?;
                    } else {
                        self.writer.write_u8(rock[0])?;
                        self.writer.write_u8(rock[1])?;
                        self.writer.write_u8(rock[2])?;
                    }
                    self.writer.write_u8(light)?;
                    self.bytes_written += 4;
                }
            }
        }
        context.decrement_level();
        Ok(())
    }
    fn generate_normalmaps(&mut self, context: &mut AssetLoadContext) -> Result<(), Box<Error>> {
        assert!(self.skirt >= 2);
        let normalmap_resolution = self.heightmap_resolution - 5;
        let normalmap_nodes: Vec<_> = (0..self.heightmaps.len())
            .filter(|&i| self.nodes[i].level as i32 == self.max_texture_level)
            .collect();
        self.layers.push(LayerParams {
            layer_type: LayerType::Normals,
            offset: self.bytes_written,
            tile_count: normalmap_nodes.len(),
            tile_resolution: normalmap_resolution as u32,
            border_size: self.skirt as u32 - 2,
            format: TextureFormat::RGBA8,
            tile_bytes: 4 * normalmap_resolution as usize * normalmap_resolution as usize,
        });
        context.increment_level("Generating normalmaps... ", normalmap_nodes.len());
        for (i, id) in normalmap_nodes.into_iter().enumerate() {
            context.set_progress(i as u64);
            self.nodes[id].tile_indices[LayerType::Normals.index()] = Some(i as u32);

            let heights = &self.heightmaps[id];
            let spacing =
                self.nodes[id].side_length / (self.heightmap_resolution - 2 * self.skirt) as f32;
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

                    self.writer.write_u8((normal.x * 127.5 + 127.5) as u8)?;
                    self.writer.write_u8((normal.y * 127.5 + 127.5) as u8)?;
                    self.writer.write_u8((normal.z * 127.5 + 127.5) as u8)?;
                    self.writer.write_u8(splat)?;
                    self.bytes_written += 4;
                }
            }
        }
        context.decrement_level();
        Ok(())
    }
    fn generate_watermasks(&mut self, context: &mut AssetLoadContext) -> Result<(), Box<Error>> {
        assert!(self.skirt >= 2);
        let watermap_resolution = self.heightmap_resolution - 5;
        self.layers.push(LayerParams {
            layer_type: LayerType::Water,
            offset: self.bytes_written,
            tile_count: self.heightmaps.len(),
            tile_resolution: watermap_resolution as u32,
            border_size: self.skirt as u32 - 2,
            format: TextureFormat::RGBA8,
            tile_bytes: 4 * watermap_resolution as usize * watermap_resolution as usize,
        });
        context.increment_level("Generating water masks... ", self.heightmaps.len());
        let mut watermask_cache = RasterCache::new(Box::new(LandCoverKind::WaterMask), 256);
        for i in 0..self.heightmaps.len() {
            context.set_progress(i as u64);
            self.nodes[i].tile_indices[LayerType::Water.index()] = Some(i as u32);

            let heights = &self.heightmaps[i];
            for y in 2..(2 + watermap_resolution) {
                for x in 2..(2 + watermap_resolution) {
                    let world = self.world_position(x as i32, y as i32, self.nodes[i].bounds);
                    let h00 = heights.get(x, y).unwrap();
                    let h01 = heights.get(x, y + 1).unwrap();
                    let h10 = heights.get(x + 1, y).unwrap();
                    let h11 = heights.get(x + 1, y + 1).unwrap();
                    let h = (h00 + h01 + h10 + h11) as f64 * 0.25;
                    let lla = self.system.world_to_lla(Vector3::new(world.x, h, world.y));
                    let w = if world.magnitude2() < 1000000.0 * 1000000.0 {
                        let mut w = 0.0;
                        if h00 <= 0.0 {
                            w += 0.25;
                        }
                        if h01 <= 0.0 {
                            w += 0.25;
                        }
                        if h10 <= 0.0 {
                            w += 0.25;
                        }
                        if h11 <= 0.0 {
                            w += 0.25;
                        }
                        watermask_cache
                            .interpolate(context, lla.x.to_degrees(), lla.y.to_degrees())
                            .unwrap_or(w * 255.0) as u8
                    } else {
                        0
                    };

                    self.writer.write_u8(w)?;
                    self.writer.write_u8(0)?;
                    self.writer.write_u8(255)?;
                    self.writer.write_u8(0)?;
                    self.bytes_written += 4;
                }
            }
        }
        context.decrement_level();
        Ok(())
    }
    fn generate_noise(
        &mut self,
        _context: &mut AssetLoadContext,
    ) -> Result<NoiseParams, Box<Error>> {
        let noise = NoiseParams {
            offset: self.bytes_written,
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
                self.writer.write_u8((v * 127.5 / 3.0 + 127.5) as u8)?;
                self.bytes_written += 1;
            }
        }
        assert_eq!(self.bytes_written, noise.offset + noise.bytes);
        Ok(noise)
    }

    fn generate_planet_mesh(
        &mut self,
        _context: &mut AssetLoadContext,
    ) -> Result<MeshDescriptor, Box<Error>> {
        let root_bounds = self.nodes[0].bounds;
        let offset = self.bytes_written;
        let mut num_vertices = 0;

        {
            let mut vertices = Vec::new();
            for y in 0..HEIGHTS_RESOLUTION {
                for x in 0..HEIGHTS_RESOLUTION {
                    let world = self.world_position(x as i32, y as i32, root_bounds);
                    let mut world3 = Vector3::new(world.x, -2.0 * EARTH_RADIUS, world.y);
                    for _ in 0..5 {
                        world3.x = world.x;
                        world3.z = world.y;
                        let mut lla = self.system.world_to_lla(world3);
                        lla.z = 0.0;
                        world3 = self.system.lla_to_world(lla);
                    }

                    vertices.push(Vector3::<f32>::new(
                        world3.x as f32,
                        world3.y as f32,
                        world3.z as f32,
                    ));
                }
            }

            let mut write_vertex = |v: Vector3<f32>| -> Result<(), Box<Error>> {
                self.writer.write_f32::<LittleEndian>(v.x)?;
                self.writer.write_f32::<LittleEndian>(v.y)?;
                self.writer.write_f32::<LittleEndian>(v.z)?;
                self.bytes_written += 12;
                num_vertices += 1;
                Ok(())
            };
            for y in 0..(HEIGHTS_RESOLUTION - 1) as usize {
                for x in 0..(HEIGHTS_RESOLUTION - 1) as usize {
                    let v00 = vertices[x + y * HEIGHTS_RESOLUTION as usize];
                    let v10 = vertices[x + 1 + y * HEIGHTS_RESOLUTION as usize];
                    let v01 = vertices[x + (y + 1) * HEIGHTS_RESOLUTION as usize];
                    let v11 = vertices[x + 1 + (y + 1) * HEIGHTS_RESOLUTION as usize];

                    write_vertex(v00)?;
                    write_vertex(v10)?;
                    write_vertex(v01)?;

                    write_vertex(v11)?;
                    write_vertex(v01)?;
                    write_vertex(v10)?;
                }
            }
        }
        Ok(MeshDescriptor {
            bytes: self.bytes_written - offset,
            offset,
            num_vertices,
        })
    }
}
