use std::error::Error;
use std::f64::consts::PI;
use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::*;
use gfx;

use cache::{MMappedAsset, WebAsset};
use terrain::dem::{self, DemSource, DigitalElevationModelParams};
use terrain::quadtree::{Node, QuadTree};
use terrain::tile_cache::{TileHeader, LayerParams, HEIGHTS_LAYER, NUM_LAYERS};

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
        let mut _bytes_written = 0;

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

        // let cell_size_ratio = cell_size_y / cell_size_x;
        // let width = (dem.width as f64 / cell_size_ratio).floor() as usize;
        // let height = dem.height;

        const HEIGHTS_RESOLUTION: usize = 33;

        let mut nodes = Node::make_nodes(524288.0 / 8.0, 3000.0, 13 - 3);
        for (i, node) in nodes.iter_mut().enumerate() {
            node.tile_indices[HEIGHTS_LAYER] = Some(i as u32);

            for y in 0..HEIGHTS_RESOLUTION {
                for x in 0..HEIGHTS_RESOLUTION {
                    let fx = x as f32 / (HEIGHTS_RESOLUTION - 1) as f32;
                    let fy = y as f32 / (HEIGHTS_RESOLUTION - 1) as f32;

                    let world_position = Vector2::<f32>::new(
                        (node.bounds.min.x + (node.bounds.max.x - node.bounds.min.x) * fx) /
                            scale_x,
                        (node.bounds.min.z + (node.bounds.max.z - node.bounds.min.z) * fy) /
                            scale_y,
                    );

                    let p = world_center + world_position;
                    let height = dem.get_elevation(p.x as f64, p.y as f64).unwrap_or(0.0);
                    // let height = 1000.0 * (0.001 * world_position.distance(Point2::origin())).sin();

                    writer.write_f32::<LittleEndian>(height)?;
                    _bytes_written += 4;
                }
            }
        }

        let mut layers: [LayerParams; NUM_LAYERS] = Default::default();
        layers[HEIGHTS_LAYER] = LayerParams {
            offset: 0,
            tile_count: nodes.len(),
            tile_resolution: HEIGHTS_RESOLUTION as u32,
            sample_bytes: 4,
            tile_bytes: 4 * HEIGHTS_RESOLUTION * HEIGHTS_RESOLUTION,
        };

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
