use std::error::Error;
use std::f64::consts::PI;
use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::*;
use gfx;

use cache::{MMappedAsset, WebAsset};
use terrain::dem::{self, DemSource, DigitalElevationModelParams};
use terrain::heightmap::Heightmap;
use terrain::tile_cache::{TileHeader, LayerParams, LayerFormat, LayerType};
use terrain::quadtree::{node, Node, QuadTree};

// This file assumes that all coordinates are provided relative to the earth represented as a
// perfect sphere. This isn't quite accurate: The coordinate system of the input datasets are
// actually WGS84 or NAD83. However, for our purposes the difference should not be noticable.

pub struct TerrainFileParams {
    pub latitude: i16,
    pub longitude: i16,
    pub source: DemSource,
}
impl TerrainFileParams {
    pub fn build_quadtree<R: gfx::Resources, F: gfx::Factory<R>>(
        self,
        factory: F,
        color_buffer: &gfx::handle::RenderTargetView<R, gfx::format::Srgba8>,
        depth_buffer: &gfx::handle::DepthStencilView<R, gfx::format::DepthStencil>,
    ) -> Result<QuadTree<R, F>, Box<Error>> {
        let (header, data) = self.load()?;

        Ok(QuadTree::new(
            header,
            data,
            factory,
            color_buffer,
            depth_buffer,
        ))
    }
}

impl MMappedAsset for TerrainFileParams {
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

        let dem = DigitalElevationModelParams {
            latitude: self.latitude,
            longitude: self.longitude,
            source: self.source,
        }.load()?;

        let world_center =
            Vector2::<f32>::new(dem.xllcorner as f32 + 0.5, dem.yllcorner as f32 + 0.5);

        let ycenter = dem.yllcorner + 0.5 * dem.cell_size * dem.height as f64;
        let scale_x = ((1.0 / 360.0) * EARTH_CIRCUMFERENCE * ycenter.to_radians().cos()) as f32;
        let scale_y = (1.0 / 360.0) * EARTH_CIRCUMFERENCE as f32;

        // Cell size in the y (latitude) direction, in meters. The x (longitude) direction will have
        // smaller cell sizes due to the projection.
        let dem_cell_size_y = scale_y * dem.cell_size as f32;

        const HEIGHTS_RESOLUTION: u16 = 33;
        const TEXTURE_RESOLUTION: u16 = 513;

        let resolution_ratio = ((TEXTURE_RESOLUTION - 1) / (HEIGHTS_RESOLUTION - 1)) as u16;

        let world_size = 524288.0 / 8.0;
        let max_level = 10i32;
        let max_texture_level = max_level - (resolution_ratio as f32).log2() as i32;

        let cell_size = world_size / ((HEIGHTS_RESOLUTION - 1) as f32) * (0.5f32).powi(max_level);
        let num_fractal_levels = (dem_cell_size_y / cell_size).log2().ceil().max(0.0) as i32;
        let max_dem_level = max_level - num_fractal_levels.max(0).min(max_level);

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

        let mut nodes = Node::make_nodes(world_size, 3000.0, max_level as u8);
        layers.push(LayerParams {
            layer_type: LayerType::Heights,
            offset: 0,
            tile_count: nodes.len(),
            tile_resolution: HEIGHTS_RESOLUTION as u32,
            border_size: 0,
            format: LayerFormat::F32,
            tile_bytes: 4 * HEIGHTS_RESOLUTION as usize * HEIGHTS_RESOLUTION as usize,
        });
        for i in 0..nodes.len() {
            nodes[i].tile_indices[LayerType::Heights.index()] = Some(i as u32);

            if nodes[i].level as i32 > max_texture_level {
                let mut ancestor = i;
                let mut offset = Vector2::new(0, 0);
                let mut offset_scale = 1;
                let mut step: u16 = resolution_ratio;

                while nodes[ancestor].level as i32 > max_texture_level {
                    let &(parent_id, child_index) = nodes[ancestor].parent.as_ref().unwrap();
                    offset += node::OFFSETS[child_index as usize] * offset_scale;
                    ancestor = parent_id.index();
                    offset_scale *= 2;
                    step /= 2;
                }
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

                // Extra scope needed due to lack of support for non-lexical lifetimes.
                {
                    let parent_heightmap = &heightmaps[nodes[i].parent.as_ref().unwrap().0.index()];
                    for y in 0..heightmap_resolution {
                        for x in 0..heightmap_resolution {
                            let height = if x % 2 == 0 && y % 2 == 0 {
                                parent_heightmap
                                    .get(x / 2 + offset.x, y / 2 + offset.y)
                                    .unwrap()
                            } else if x % 2 == 0 {
                                let h0 = parent_heightmap
                                    .get(x / 2 + offset.x, y / 2 + offset.y)
                                    .unwrap();
                                let h1 = parent_heightmap
                                    .get(x / 2 + offset.x, y / 2 + offset.y + 1)
                                    .unwrap();

                                (h0 + h1) * 0.5
                            } else if y % 2 == 0 {
                                let h0 = parent_heightmap
                                    .get(x / 2 + offset.x, y / 2 + offset.y)
                                    .unwrap();
                                let h1 = parent_heightmap
                                    .get(x / 2 + offset.x + 1, y / 2 + offset.y)
                                    .unwrap();

                                (h0 + h1) * 0.5
                            } else {
                                let h0 = parent_heightmap
                                    .get(x / 2 + offset.x, y / 2 + offset.y)
                                    .unwrap();
                                let h1 = parent_heightmap
                                    .get(x / 2 + offset.x, y / 2 + offset.y + 1)
                                    .unwrap();
                                let h2 = parent_heightmap
                                    .get(x / 2 + offset.x + 1, y / 2 + offset.y)
                                    .unwrap();
                                let h3 = parent_heightmap
                                    .get(x / 2 + offset.x + 1, y / 2 + offset.y + 1)
                                    .unwrap();

                                (h0 + h1 + h2 + h3) * 0.25
                            };

                            heights.push(height);

                            if (x as i32 - skirt as i32) % resolution_ratio as i32 == 0 &&
                                (y as i32 - skirt as i32) % resolution_ratio as i32 == 0 &&
                                !in_skirt(x, y)
                            {
                                writer.write_f32::<LittleEndian>(height)?;
                                bytes_written += 4;
                            }
                        }
                    }
                }

                heightmaps.push(Heightmap::new(
                    heights,
                    heightmap_resolution,
                    heightmap_resolution,
                ));
            } else {
                assert_eq!(heightmaps.len(), i);
                let node = &nodes[i];
                let mut heights = Vec::with_capacity(
                    heightmap_resolution as usize * heightmap_resolution as usize,
                );
                for y in 0..(heightmap_resolution as i32) {
                    for x in 0..(heightmap_resolution as i32) {
                        let fx = (x - skirt as i32) as f32 /
                            (heightmap_resolution - 1 - 2 * skirt) as f32;
                        let fy = (y - skirt as i32) as f32 /
                            (heightmap_resolution - 1 - 2 * skirt) as f32;

                        let world_position = Vector2::<f32>::new(
                            (node.bounds.min.x + (node.bounds.max.x - node.bounds.min.x) * fx) /
                                scale_x,
                            (node.bounds.min.z + (node.bounds.max.z - node.bounds.min.z) * fy) /
                                scale_y,
                        );

                        let p = world_center + world_position;
                        let height = dem.get_elevation(p.x as f64, p.y as f64).unwrap_or(0.0);
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

        // Normals
        assert!(skirt >= 2);
        let normalmap_resolution = heightmap_resolution - 4;
        layers.push(LayerParams {
            layer_type: LayerType::Normals,
            offset: bytes_written,
            tile_count: heightmaps.len(),
            tile_resolution: normalmap_resolution as u32,
            border_size: skirt as u32 - 2,
            format: LayerFormat::RGBA8,
            tile_bytes: 4 * normalmap_resolution as usize * normalmap_resolution as usize,
        });
        for i in 0..heightmaps.len() {
            nodes[i].tile_indices[LayerType::Normals.index()] = Some(i as u32);

            let heights = &heightmaps[i];
            let spacing = nodes[i].side_length / (heightmap_resolution - 2 * skirt) as f32;
            for y in 2..(2 + normalmap_resolution) {
                for x in 2..(2 + normalmap_resolution) {
                    let h00 = heights.get(x, y).unwrap();
                    let h01 = heights.get(x, y + 1).unwrap();
                    let h10 = heights.get(x + 1, y).unwrap();
                    let h11 = heights.get(x + 1, y + 1).unwrap();

                    let nx = h10 + h11 - h00 - h01;
                    let ny = h01 + h11 - h00 - h10;
                    let nz = 2.0 * spacing;
                    let len = (nx * nx + ny * ny + nz * nz).sqrt();

                    writer.write_u8(((nx / len) * 127.5 + 127.5) as u8)?;
                    writer.write_u8(((ny / len) * 127.5 + 127.5) as u8)?;
                    writer.write_u8(((nz / len) * 127.5 + 127.5) as u8)?;
                    writer.write_u8(0)?;
                    bytes_written += 4;
                }
            }
        }

        Ok(TileHeader { layers, nodes })
    }
}

/// The radius of the earth in meters.
const EARTH_RADIUS: f64 = 6371000.0;
const EARTH_CIRCUMFERENCE: f64 = 2.0 * PI * EARTH_RADIUS;

#[derive(Serialize, Deserialize)]
pub struct TerrainFile {
    width: usize,
    height: usize,
    cell_size: f32,

    elevations: Vec<f32>,
    slopes: Vec<(f32, f32)>,
    shadows: Vec<f32>,
}

impl TerrainFile {
    fn compute_slopes(
        width: usize,
        height: usize,
        cell_size: f32,
        elevations: &[f32],
    ) -> Vec<(f32, f32)> {
        let get_elevation = |x: usize, y: usize| {
            assert!(x < width);
            assert!(y < height);
            elevations[x + y * width]
        };

        let mut slopes = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let xslope = if x == 0 {
                    (get_elevation(x + 1, y) - get_elevation(x, y)) / cell_size
                } else if x == width - 1 {
                    (get_elevation(x, y) - get_elevation(x - 1, y)) / cell_size
                } else {
                    (get_elevation(x + 1, y) - get_elevation(x - 1, y)) / (2.0 * cell_size)
                };

                let yslope = if y == 0 {
                    (get_elevation(x, y + 1) - get_elevation(x, y)) / cell_size
                } else if y == height - 1 {
                    (get_elevation(x, y) - get_elevation(x, y - 1)) / cell_size
                } else {
                    (get_elevation(x, y + 1) - get_elevation(x, y - 1)) / (2.0 * cell_size)
                };

                slopes.push((xslope, yslope));
            }
        }

        slopes
    }

    fn compute_shadows(
        width: usize,
        height: usize,
        cell_size: f32,
        elevations: &[f32],
    ) -> Vec<f32> {
        let get_elevation = |x: usize, y: usize| {
            assert!(x < width);
            assert!(y < height);
            elevations[x + y * width]
        };

        let ray_slope = 0.4 * cell_size;

        let mut shadows = Vec::with_capacity(width * height);
        for y in 0..height {
            let mut highest = None;
            for x in 0..width {
                let h = get_elevation(x, y);
                let shadow_height = highest
                    .as_ref()
                    .map(|&(sx, sh)| sh - ((x - sx) as f32) * ray_slope)
                    .unwrap_or(h - 1.0);

                if shadow_height < h {
                    highest = Some((x, h));
                }

                shadows.push(shadow_height);
            }
        }

        shadows
    }

    /// Construct a `TerrainFile` from a `DigitalElevationModel`.
    pub fn from_digital_elevation_model(dem: dem::DigitalElevationModel) -> Self {
        // Compute approximate cell size in meters.
        let ycenter = dem.yllcorner + 0.5 * dem.cell_size * dem.height as f64;
        let cell_size_x = (dem.cell_size / 360.0) * EARTH_CIRCUMFERENCE *
            ycenter.to_radians().cos();
        let cell_size_y = (dem.cell_size / 360.0) * EARTH_CIRCUMFERENCE;
        let cell_size = cell_size_y as f32;

        let cell_size_ratio = cell_size_y / cell_size_x;
        let width = (dem.width as f64 / cell_size_ratio).floor() as usize;
        let height = dem.height;
        let mut elevations = Vec::with_capacity(width * height);

        for y in 0..height {
            for x in 0..width {
                let fx = x as f64 * cell_size_ratio;
                let floor_fx = fx.floor() as usize;
                let ceil_fx = fx.ceil() as usize;
                assert!(floor_fx < dem.width);
                assert!(ceil_fx < dem.width);
                let t = fx.fract() as f32;
                let h = dem.elevations[floor_fx + y * dem.width] * (1.0 - t) +
                    dem.elevations[ceil_fx + y * dem.width] * t;
                elevations.push(h)
            }
        }

        let slopes = Self::compute_slopes(width, height, cell_size, &elevations[..]);
        let shadows = Self::compute_shadows(width, height, cell_size, &elevations[..]);
        TerrainFile {
            width,
            height,
            cell_size,
            elevations,
            slopes,
            shadows,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn elevation(&self, x: usize, y: usize) -> f32 {
        assert!(x < self.width);
        assert!(y < self.height);
        self.elevations[x + y * self.width]
    }

    pub fn slope(&self, x: usize, y: usize) -> (f32, f32) {
        assert!(x < self.width);
        assert!(y < self.height);
        self.slopes[x + y * self.width]
    }

    pub fn elevations(&self) -> &[f32] {
        &self.elevations[..]
    }

    pub fn slopes(&self) -> &[(f32, f32)] {
        &self.slopes[..]
    }

    pub fn shadows(&self) -> &[f32] {
        &self.shadows[..]
    }

    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }
}
