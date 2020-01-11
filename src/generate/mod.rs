use crate::cache::{AssetLoadContext, MMappedAsset, WebAsset};
use crate::coordinates::CoordinateSystem;
use crate::mapfile::MapFile;
use crate::terrain::dem::DemSource;
use crate::terrain::dem::GlobalDem;
use crate::terrain::heightmap::{self, Heightmap};
use crate::terrain::landcover::{BlueMarble, BlueMarbleTileSource};
use crate::terrain::quadtree::VNode;
use crate::terrain::raster::RasterCache;
use crate::terrain::reprojected_raster::{
    DataType, RasterSource, ReprojectedDemDef, ReprojectedRaster, ReprojectedRasterDef,
};
use crate::terrain::tile_cache::{
    ByteRange, LayerParams, LayerType, MeshDescriptor, NoiseParams, TextureDescriptor,
    TextureFormat, TileHeader,
};
use crate::utils::math::BoundingBox;
use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::*;
use failure::Error;
use rand;
use rand::distributions::Distribution;
use rand_distr::Normal;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::io::Write;
use std::rc::Rc;
use vec_map::VecMap;

mod gpu;
pub(crate) use gpu::*;

/// The radius of the earth in meters.
const EARTH_RADIUS: f64 = 6371000.0;
const EARTH_CIRCUMFERENCE: f64 = 2.0 * PI * EARTH_RADIUS;

// Mapping from side length to level number.
#[allow(unused)]
mod levels {
    pub const LEVEL_4194_KM: i32 = 0;
    pub const LEVEL_2097_KM: i32 = 1;
    pub const LEVEL_1049_KM: i32 = 2;
    pub const LEVEL_524_KM: i32 = 3;
    pub const LEVEL_262_KM: i32 = 4;
    pub const LEVEL_131_KM: i32 = 5;
    pub const LEVEL_66_KM: i32 = 6;
    pub const LEVEL_33_KM: i32 = 7;
    pub const LEVEL_16_KM: i32 = 8;
    pub const LEVEL_8_KM: i32 = 9;
    pub const LEVEL_4_KM: i32 = 10;
    pub const LEVEL_2_KM: i32 = 11;
    pub const LEVEL_1_KM: i32 = 12;
    pub const LEVEL_256_M: i32 = 13;
    pub const LEVEL_128_M: i32 = 14;
    pub const LEVEL_64_M: i32 = 15;
    pub const LEVEL_32_M: i32 = 16;
    pub const LEVEL_16_M: i32 = 17;
    pub const LEVEL_8_M: i32 = 18;
    pub const LEVEL_4_M: i32 = 19;
    pub const LEVEL_2_M: i32 = 20;
    pub const LEVEL_1_M: i32 = 21;
    pub const LEVEL_50_CM: i32 = 22;
    pub const LEVEL_25_CM: i32 = 23;
    pub const LEVEL_13_CM: i32 = 24;
    pub const LEVEL_6_CM: i32 = 25;
}
use levels::*;

/// How much detail the terrain mesh should have. Higher values require more resources to render,
/// but produce nicer results.
pub enum VertexQuality {
    /// Use up to about 4M triangles per frame.
    High,
    /// About 1M triangles per frame.
    Medium,
    /// A couple hundred thousand triangles per frame.
    Low,
}
impl VertexQuality {
    fn resolution(&self) -> u16 {
        match *self {
            VertexQuality::Low => 33,
            VertexQuality::Medium => 65,
            VertexQuality::High => 129,
        }
    }
    fn resolution_log2(&self) -> u32 {
        let r = self.resolution() - 1;
        assert!(r.is_power_of_two());
        r.trailing_zeros()
    }
    fn as_str(&self) -> &str {
        match *self {
            VertexQuality::Low => "vl",
            VertexQuality::Medium => "vm",
            VertexQuality::High => "vh",
        }
    }
}

/// What resolution to use for terrain texture mapping. Higher values consume much more GPU memory
/// and increase the file size.
pub enum TextureQuality {
    /// Quality suitable for a 4K display.
    Ultra,
    /// Good for resolutions up to 1080p.
    High,
    /// About half the quality `High`.
    Low,
}
impl TextureQuality {
    fn resolution(&self) -> u16 {
        match *self {
            TextureQuality::Low => 256,
            TextureQuality::High => 512,
            TextureQuality::Ultra => 1024,
        }
    }
    fn as_str(&self) -> &str {
        match *self {
            TextureQuality::Low => "tl",
            TextureQuality::High => "th",
            TextureQuality::Ultra => "tu",
        }
    }
}

/// Spacing between adjacent mesh grid points for the most detailed quadtree level.
pub enum GridSpacing {
    // 4000 cm between mesh grid points.
    FourMeters,
    // 2000 cm between mesh grid points.
    TwoMeters,
    // 1000 cm between mesh grid points.
    OneMeter,
    // 500 cm between mesh grid points.
    HalfMeter,
    // 250 cm between mesh grid points.
    QuarterMeter,
}
impl GridSpacing {
    fn log2_spacing(&self) -> i32 {
        match *self {
            GridSpacing::FourMeters => 2,
            GridSpacing::TwoMeters => 1,
            GridSpacing::OneMeter => 0,
            GridSpacing::HalfMeter => -1,
            GridSpacing::QuarterMeter => -2,
        }
    }
    fn as_str(&self) -> &str {
        match *self {
            GridSpacing::FourMeters => "4m",
            GridSpacing::TwoMeters => "2m",
            GridSpacing::OneMeter => "1m",
            GridSpacing::HalfMeter => "500cm",
            GridSpacing::QuarterMeter => "250cm",
        }
    }
}

/// Used to construct a `QuadTree`.
pub struct MapFileBuilder {
    latitude: i16,
    longitude: i16,
    source: DemSource,
    vertex_quality: VertexQuality,
    texture_quality: TextureQuality,
    grid_spacing: GridSpacing,
    // materials: MaterialSet<R>,
    // sky: Skybox<R>,
    context: Option<AssetLoadContext>,
}
impl MapFileBuilder {
    /// Create a new `QuadTreeBuilder` with default arguments.
    ///
    /// At very least, the latitude and longitude should probably be set to their desired values
    /// before calling `build()`.
    pub fn new() -> Self {
        let context = AssetLoadContext::new();
        Self {
            latitude: 38,
            longitude: -122,
            source: DemSource::Srtm30m,
            vertex_quality: VertexQuality::High,
            texture_quality: TextureQuality::High,
            grid_spacing: GridSpacing::TwoMeters,
            // materials: MaterialSet::load(&mut factory, encoder, &mut context).unwrap(),
            // sky: Skybox::new(&mut factory, encoder, &mut context),
            context: Some(context),
            // factory,
            // encoder,
        }
    }

    /// The latitude the generated map should be centered at, in degrees.
    pub fn latitude(mut self, latitude: i16) -> Self {
        assert!(latitude >= -90 && latitude <= 90);
        self.latitude = latitude;
        self
    }

    /// The longitude the generated map should be centered at, in degrees.
    pub fn longitude(mut self, longitude: i16) -> Self {
        assert!(longitude >= -180 && longitude <= 180);
        self.longitude = longitude;
        self
    }

    /// How detailed the resulting terrain mesh should be.
    pub fn vertex_quality(mut self, quality: VertexQuality) -> Self {
        self.vertex_quality = quality;
        self
    }

    /// How high resolution the terrain's textures should be.
    pub fn texture_quality(mut self, quality: TextureQuality) -> Self {
        self.texture_quality = quality;
        self
    }

    pub fn grid_spacing(mut self, spacing: GridSpacing) -> Self {
        self.grid_spacing = spacing;
        self
    }

    /// Actually construct the `QuadTree`.
    ///
    /// This function will (the first time it is called) download many gigabytes of raw data,
    /// primarily datasets relating to real world land cover and elevation. These files will be
    /// stored in ~/.terra, so that they don't have to be fetched multiple times. This means that
    /// this function can largely resume from where it left off if interrupted.
    ///
    /// Even once all needed files have been downloaded, the generation process takes a large amount
    /// of CPU resources. You can expect it to run at full load continiously for several full
    /// minutes, even in release builds (you *really* don't want to wait for generation in debug
    /// mode...).
    pub fn build(mut self) -> Result<MapFile, Error> {
        let mut context = self.context.take().unwrap();
        let (header, data) = self.load(&mut context)?;

        Ok(MapFile::new(header, data))
    }

    fn name(&self) -> String {
        let n_or_s = if self.latitude >= 0 { 'n' } else { 's' };
        let e_or_w = if self.longitude >= 0 { 'e' } else { 'w' };
        format!(
            "{}{:02}_{}{:03}_{}m_{}_{}_{}",
            n_or_s,
            self.latitude.abs(),
            e_or_w,
            self.longitude.abs(),
            self.source.resolution(),
            self.grid_spacing.as_str(),
            self.vertex_quality.as_str(),
            self.texture_quality.as_str(),
        )
    }
}

impl MMappedAsset for MapFileBuilder {
    type Header = TileHeader;

    fn filename(&self) -> String {
        format!("maps/{}", self.name())
    }

    fn generate<W: Write>(
        &self,
        context: &mut AssetLoadContext,
        writer: W,
    ) -> Result<Self::Header, Error> {
        let world_center =
            Vector2::<f32>::new(self.longitude as f32 + 0.5, self.latitude as f32 + 0.5);

        // Cell size in the y (latitude) direction, in meters. The x (longitude) direction will have
        // smaller cell sizes due to the projection.
        let dem_cell_size_y =
            self.source.cell_size() / (360.0 * 60.0 * 60.0) * EARTH_CIRCUMFERENCE as f32;

        let resolution_ratio =
            self.texture_quality.resolution() / (self.vertex_quality.resolution() - 1);
        assert!(resolution_ratio > 0);

        let world_size = 4194304.0;
        let mut max_heights_level = LEVEL_1_M
            - self.vertex_quality.resolution_log2() as i32
            - (self.grid_spacing.log2_spacing() - 1);
        let max_heights_present_level =
            LEVEL_32_M - self.vertex_quality.resolution_log2() as i32 + 1;
        let mut max_texture_level = max_heights_level - (resolution_ratio as f32).log2() as i32;
        let max_texture_present_level =
            max_heights_present_level - (resolution_ratio as f32).log2() as i32;

        max_heights_level = max_heights_present_level;
        max_texture_level = max_heights_present_level;

        let cell_size = world_size / ((self.vertex_quality.resolution() - 1) as f32)
            * (0.5f32).powi(max_heights_level);
        let num_fractal_levels = (dem_cell_size_y / cell_size).log2().ceil().max(0.0) as i32;
        let max_dem_level = max_texture_level - num_fractal_levels.max(0).min(max_texture_level);

        // Amount of space outside of tile that is included in heightmap. Used for computing
        // normals and such. Must be even.
        let skirt = 4;
        assert_eq!(skirt % 2, 0);

        // Resolution of each heightmap stored in heightmaps. They are at higher resolution than
        // self.vertex_quality.resolution() so that the more detailed textures can be derived from
        // them.
        let heightmap_resolution = self.texture_quality.resolution() + 1 + 2 * skirt;

        let mut state = State {
            random: {
                let normal = Normal::new(0.0, 1.0).unwrap();
                let v =
                    (0..(15 * 15)).map(|_| normal.sample(&mut rand::thread_rng()) as f32).collect();
                Heightmap::new(v, 15, 15)
            },
            dem_source: self.source,
            heightmap_resolution,
            heights_resolution: self.vertex_quality.resolution(),
            max_heights_level: max_heights_level as u8,
            max_heights_present_level: max_heights_present_level as u8,
            max_texture_level: max_texture_level as u8,
            max_texture_present_level: max_texture_present_level as u8,
            max_dem_level: max_dem_level as u8,
            resolution_ratio,
            writer,
            heightmaps: None,
            skirt,
            system: CoordinateSystem::from_lla(Vector3::new(
                world_center.y.to_radians() as f64,
                world_center.x.to_radians() as f64,
                0.0,
            )),
            nodes: VNode::make_nodes(30000.0, max_heights_level as u8),
            layers: VecMap::new(),
            bytes_written: 0,
            directory_name: format!("maps/t.{}/", self.name()),
        };

        context.set_progress_and_total(0, 5);
        let base_heights = state.generate_heightmaps(context)?;
        context.set_progress(1);
        state.generate_normalmaps(context)?;
        context.set_progress(2);
        state.generate_colormaps(context)?;
        context.set_progress(3);
        let planet_mesh = state.generate_planet_mesh(context)?;
        let planet_mesh_texture = state.generate_planet_mesh_texture(context)?;
        context.set_progress(4);
        let noise = state.generate_noise(context)?;
        let State { layers, nodes, system, .. } = state;

        context.set_progress(5);

        Ok(TileHeader {
            system,
            base_heights,
            planet_mesh,
            planet_mesh_texture,
            layers,
            noise,
            nodes,
        })
    }
}

struct State<W: Write> {
    dem_source: DemSource,

    random: Heightmap<f32>,
    heightmaps: Option<ReprojectedRaster>,

    /// Resolution of the heightmap for each quadtree node.
    heights_resolution: u16,
    /// Resolution of the intermediate heightmaps which are used to generate normalmaps and
    /// colormaps. Derived from the target texture resolution.
    heightmap_resolution: u16,

    skirt: u16,

    max_heights_level: u8,
    max_heights_present_level: u8,
    max_texture_level: u8,
    max_texture_present_level: u8,
    max_dem_level: u8,

    resolution_ratio: u16,
    writer: W,
    // materials: &'a MaterialSet<R>,
    system: CoordinateSystem,

    layers: VecMap<LayerParams>,
    nodes: Vec<VNode>,
    bytes_written: usize,

    directory_name: String,
}

impl<W: Write> State<W> {
    #[allow(unused)]
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
    #[allow(unused)]
    fn world_positionf(&self, x: f32, y: f32, bounds: BoundingBox) -> Vector2<f64> {
        let fx = (x - self.skirt as f32) / (self.heightmap_resolution - 1 - 2 * self.skirt) as f32;
        let fy = (y - self.skirt as f32) / (self.heightmap_resolution - 1 - 2 * self.skirt) as f32;

        Vector2::new(
            (bounds.min.x + (bounds.max.x - bounds.min.x) * fx) as f64,
            (bounds.min.z + (bounds.max.z - bounds.min.z) * fy) as f64,
        )
    }

    fn page_pad(&mut self) -> Result<(), Error> {
        if self.bytes_written % 4096 != 0 {
            let data = vec![0; 4096 - (self.bytes_written % 4096)];
            self.writer.write_all(&data)?;
            self.bytes_written += data.len();
        }
        return Ok(());
    }

    fn generate_heightmaps(
        &mut self,
        context: &mut AssetLoadContext,
    ) -> Result<TextureDescriptor, Error> {
        let present_tile_count =
            self.nodes.iter().filter(|n| n.level() <= self.max_heights_present_level as u8).count();
        let vacant_tile_count = self
            .nodes
            .iter()
            .filter(|n| {
                n.level() > self.max_heights_present_level as u8
                    && n.level() <= self.max_heights_level as u8
            })
            .count();

        let tile_valid_bitmap = ByteRange {
            offset: self.bytes_written,
            length: present_tile_count + vacant_tile_count,
        };
        self.writer.write_all(&vec![1u8; present_tile_count])?;
        self.writer.write_all(&vec![0u8; vacant_tile_count])?;
        self.bytes_written += present_tile_count + vacant_tile_count;
        self.page_pad()?;

        let dem_cache = Rc::new(RefCell::new(RasterCache::new(Box::new(self.dem_source), 128)));

        let global_dem = GlobalDem.load(context)?;
        let base_heights = TextureDescriptor {
            offset: self.bytes_written,
            resolution: 4097,
            format: TextureFormat::RGBA32F,
            bytes: 4097 * 4097 * 16,
        };
        context.increment_level("Generating base_heights... ", base_heights.resolution);
        let bh_resolution = 4097;
        let mut bh_data = vec![0.0; bh_resolution * bh_resolution];
        let mut be_data = vec![0.0; bh_resolution * bh_resolution];
        for y in 0..bh_resolution {
            context.set_progress(y as u64);
            for x in 0..bh_resolution {
                let world = Vector2::new(
                    32.0 * (x as f64 - (base_heights.resolution / 2) as f64),
                    32.0 * (y as f64 - (base_heights.resolution / 2) as f64),
                );
                let mut world3 = Vector3::new(
                    world.x,
                    EARTH_RADIUS
                        * ((1.0 - world.magnitude2() / EARTH_RADIUS).max(0.25).sqrt() - 1.0),
                    world.y,
                );
                let mut lla = Vector3::new(0.0, 0.0, 0.0);
                for i in 0..5 {
                    world3.x = world.x;
                    world3.z = world.y;
                    lla = self.system.world_to_lla(world3);
                    lla.z = if i >= 3 && world.magnitude2() < 2000000.0 * 2000000.0 {
                        if world.magnitude2() < 250000.0 * 250000.0 {
                            dem_cache
                                .borrow_mut()
                                .interpolate(context, lla.x.to_degrees(), lla.y.to_degrees(), 0)
                                .unwrap_or(0.0) as f64
                        } else {
                            global_dem
                                .interpolate(lla.x.to_degrees(), lla.y.to_degrees(), 0)
                                .max(0.0)
                        }
                    } else {
                        0.0
                    };
                    world3 = self.system.lla_to_world(lla);
                }
                bh_data[x + y * bh_resolution] = world3.y as f32;
                be_data[x + y * bh_resolution] = lla.z as f32;
            }
        }
        context.decrement_level();

        context.increment_level("Writing base_heights... ", 100);
        for y in 0..bh_resolution {
            for x in 0..bh_resolution {
                let dx = if x == 0 {
                    bh_data[x + 1 + bh_resolution * y] - bh_data[x + bh_resolution * y]
                } else if x == bh_resolution - 1 {
                    bh_data[x + bh_resolution * y] - bh_data[x - 1 + bh_resolution * y]
                } else {
                    0.5 * (bh_data[x + 1 + bh_resolution * y] - bh_data[x - 1 + bh_resolution * y])
                };
                let dy = if y == 0 {
                    bh_data[x + bh_resolution * (y + 1)] - bh_data[x + bh_resolution * y]
                } else if y == bh_resolution - 1 {
                    bh_data[x + bh_resolution * y] - bh_data[x + bh_resolution * (y - 1)]
                } else {
                    0.5 * (bh_data[x + bh_resolution * (y + 1)]
                        - bh_data[x + bh_resolution * (y - 1)])
                };
                self.writer.write_f32::<LittleEndian>(bh_data[x + y * bh_resolution])?;
                self.writer.write_f32::<LittleEndian>(dx / 32.0)?;
                self.writer.write_f32::<LittleEndian>(dy / 32.0)?;
                self.writer.write_f32::<LittleEndian>(be_data[x + y * bh_resolution])?;
                self.bytes_written += 16;
            }
        }
        context.decrement_level();

        let tile_bytes = 16 * self.heights_resolution as usize * self.heights_resolution as usize;
        let tile_locations = (0..(present_tile_count + vacant_tile_count))
            .map(|i| ByteRange { offset: self.bytes_written + i * tile_bytes, length: tile_bytes })
            .collect();
        self.layers.insert(
            LayerType::Heights.index(),
            LayerParams {
                layer_type: LayerType::Heights,
                tile_indices: (0..(present_tile_count + vacant_tile_count))
                    .map(|i| (self.nodes[i], i as u32))
                    .collect(),
                tile_valid_bitmap,
                tile_locations,
                texture_resolution: self.heights_resolution as u32,
                texture_border_size: 0,
                texture_format: TextureFormat::RGBA32F,
            },
        );
        let reproject = ReprojectedDemDef {
            name: format!("{}dem", self.directory_name),
            dem_cache,
            system: &self.system,
            nodes: &self.nodes,
            random: &self.random,
            skirt: self.skirt,
            max_dem_level: self.max_dem_level as u8,
            max_texture_present_level: self.max_texture_present_level as u8,
            resolution: self.heightmap_resolution,
            global_dem,
        };
        self.heightmaps = Some(ReprojectedRaster::from_dem(reproject, context)?);

        let tile_from_node: HashMap<VNode, usize> =
            self.nodes.iter().cloned().enumerate().map(|(i, n)| (n, i)).collect();

        context.increment_level("Writing heightmaps... ", present_tile_count);
        for i in 0..present_tile_count {
            context.set_progress(i as u64);

            let (heightmap, offset, step) =
                if self.nodes[i].level() > self.max_texture_present_level {
                    let (ancestor, generations, mut offset) = self.nodes[i]
                        .find_ancestor(|node| node.level() <= self.max_texture_present_level)
                        .unwrap();

                    let ancestor = tile_from_node[&ancestor];
                    let offset_scale = 1 << generations;
                    let step = self.resolution_ratio >> generations;
                    offset *= (self.heightmap_resolution - 2 * self.skirt) as u32 / offset_scale;
                    let offset = Vector2::new(offset.x as u16, offset.y as u16);

                    (ancestor, offset, step)
                } else {
                    let step = (self.heightmap_resolution - 2 * self.skirt - 1)
                        / (self.heights_resolution - 1);
                    (i, Vector2::new(0, 0), step)
                };

            let mut miny = None;
            let mut maxy = None;
            for y in 0..self.heights_resolution {
                for x in 0..self.heights_resolution {
                    let position = Vector2::new(
                        x * step + offset.x + self.skirt,
                        y * step + offset.y + self.skirt,
                    );
                    let height =
                        self.heightmaps.as_ref().unwrap().get(heightmap, position.x, position.y, 0);

                    miny = Some(height.min(miny.unwrap_or(height)));
                    maxy = Some(match maxy {
                        Some(y) if y > height => y,
                        _ => height,
                    });

                    let world2 = self.world_position(
                        position.x as i32,
                        position.y as i32,
                        self.nodes[i].bounds(),
                    );
                    let altitude =
                        self.system.world_to_lla(Vector3::new(world2.x, height as f64, world2.y)).z;

                    self.writer.write_f32::<LittleEndian>(0.0)?;
                    self.writer.write_f32::<LittleEndian>(height)?;
                    self.writer.write_f32::<LittleEndian>(0.0)?;
                    self.writer.write_f32::<LittleEndian>(altitude as f32)?;
                    self.bytes_written += 16;
                }
            }
        }

        let vacant_bytes = vacant_tile_count
            * self.heights_resolution as usize
            * self.heights_resolution as usize
            * 16;
        self.writer.write_all(&vec![0; vacant_bytes])?;
        self.bytes_written += vacant_bytes;

        context.decrement_level();

        Ok(base_heights)
    }
    fn generate_colormaps(&mut self, context: &mut AssetLoadContext) -> Result<(), Error> {
        assert!(self.skirt >= 2);
        let colormap_skirt = self.skirt - 2;
        let colormap_resolution = self.heightmap_resolution - 5;

        let tile_count =
            self.nodes.iter().filter(|n| n.level() <= self.max_texture_level as u8).count();

        context.increment_level("Generating colormaps... ", tile_count);

        let reproject_bluemarble = ReprojectedRasterDef {
            name: format!("{}bluemarble", self.directory_name),
            heights: self.heightmaps.as_ref().unwrap(),
            system: &self.system,
            nodes: &self.nodes,
            skirt: self.skirt,
            datatype: DataType::U8,
            raster: RasterSource::Hybrid {
                global: Box::new(BlueMarble),
                cache: Rc::new(RefCell::new(RasterCache::new(Box::new(BlueMarbleTileSource), 8))),
            },
        };
        let bluemarble = ReprojectedRaster::from_raster(reproject_bluemarble, context)?;

        // let reproject = ReprojectedRasterDef {
        //     name: format!("{}watermasks", self.directory_name),
        //     heights: self.heightmaps.as_ref().unwrap(),
        //     system: &self.system,
        //     nodes: &self.nodes,
        //     skirt: self.skirt,
        //     datatype: DataType::U8,
        //     raster: RasterSource::GlobalRaster::<_, BitContainer> {
        //         global: Box::new(GlobalWaterMask),
        //     },
        // };
        // let watermasks = ReprojectedRaster::from_raster(reproject, context)?;

        let _mix = |a: u8, b: u8, t: f32| (f32::from(a) * (1.0 - t) + f32::from(b) * t) as u8;

        let mut colormaps: Vec<Vec<u8>> = Vec::new();
        for i in 0..tile_count {
            context.set_progress(i as u64);

            let mut colormap =
                Vec::with_capacity(colormap_resolution as usize * colormap_resolution as usize);
            let _heights = self.heightmaps.as_ref().unwrap();
            let spacing =
                self.nodes[i].side_length() / (self.heightmap_resolution - 2 * self.skirt) as f32;

            if spacing <= bluemarble.spacing().unwrap() as f32 {
                continue;
            }

            for y in 2..(2 + colormap_resolution) {
                for x in 2..(2 + colormap_resolution) {
                    let color = if false
                    /*watermasks.get(i, x, y, 0) > 0.01*/
                    {
                        [0, 6, 13, 77]
                    } else {
                        let r = bluemarble.get(i, x, y, 0) as u8;
                        let g = bluemarble.get(i, x, y, 1) as u8;
                        let b = bluemarble.get(i, x, y, 2) as u8;
                        let roughness = (0.7 * 255.0) as u8;
                        [r, g, b, roughness]
                    };

                    colormap.extend_from_slice(&color);
                }
            }

            colormaps.push(colormap);
        }

        let tile_valid_bitmap = ByteRange { offset: self.bytes_written, length: tile_count };
        self.writer.write_all(&vec![1u8; colormaps.len()])?;
        self.writer.write_all(&vec![0u8; tile_count - colormaps.len()])?;
        self.bytes_written += tile_count;
        self.page_pad()?;

        let tile_bytes = 4 * colormap_resolution as usize * colormap_resolution as usize;
        let tile_locations = (0..colormaps.len())
            .map(|i| ByteRange { offset: self.bytes_written + i * tile_bytes, length: tile_bytes })
            .collect();

        self.layers.insert(
            LayerType::Colors.index(),
            LayerParams {
                layer_type: LayerType::Colors,
                tile_indices: (0..tile_count).map(|i| (self.nodes[i], i as u32)).collect(),
                tile_valid_bitmap,
                tile_locations,
                texture_resolution: colormap_resolution as u32,
                texture_border_size: colormap_skirt as u32,
                texture_format: TextureFormat::SRGBA,
            },
        );

        for colormap in colormaps {
            self.writer.write_all(&colormap[..])?;
            self.bytes_written += colormap.len();
        }

        context.decrement_level();
        Ok(())
    }
    fn generate_normalmaps(&mut self, context: &mut AssetLoadContext) -> Result<(), Error> {
        assert!(self.skirt >= 2);
        let normalmap_resolution = self.heightmap_resolution - 5;
        let tile_count =
            self.nodes.iter().filter(|n| n.level() <= self.max_texture_level as u8).count();

        let tile_bytes = 2 * normalmap_resolution as usize * normalmap_resolution as usize;
        let tile_locations = (0..tile_count)
            .map(|i| ByteRange { offset: self.bytes_written + i * tile_bytes, length: tile_bytes })
            .collect();

        let zero_tile =
            vec![0u8; normalmap_resolution as usize * normalmap_resolution as usize * 2];

        let mut bitmap = vec![1u8; tile_count];
        context.increment_level("Generating normalmaps... ", tile_count);
        for i in 0..tile_count {
            context.set_progress(i as u64);

            let heights = self.heightmaps.as_ref().unwrap();
            let spacing =
                self.nodes[i].side_length() / (self.heightmap_resolution - 2 * self.skirt) as f32;

            if self.nodes[i].level() > self.max_texture_present_level {
                bitmap[i] = 0;
                self.writer.write_all(&zero_tile)?;
                self.bytes_written += zero_tile.len();
            } else {
                for y in 2..(2 + normalmap_resolution) {
                    for x in 2..(2 + normalmap_resolution) {
                        let h00 = heights.get(i, x, y, 0);
                        let h01 = heights.get(i, x, y + 1, 0);
                        let h10 = heights.get(i, x + 1, y, 0);
                        let h11 = heights.get(i, x + 1, y + 1, 0);

                        let normal = Vector3::new(
                            h10 + h11 - h00 - h01,
                            2.0 * spacing,
                            -1.0 * (h01 + h11 - h00 - h10),
                        )
                        .normalize();

                        self.writer.write_u8((normal.x * 127.5 + 127.5) as u8)?;
                        self.writer.write_u8((normal.z * 127.5 + 127.5) as u8)?;
                        self.bytes_written += 2;
                    }
                }
            }
        }
        context.decrement_level();

        self.page_pad()?;
        let tile_valid_bitmap = ByteRange { offset: self.bytes_written, length: tile_count };
        self.writer.write_all(&bitmap)?;
        self.bytes_written += tile_count;
        self.page_pad()?;

        self.layers.insert(
            LayerType::Normals.index(),
            LayerParams {
                layer_type: LayerType::Normals,
                tile_indices: (0..tile_count).map(|i| (self.nodes[i], i as u32)).collect(),
                tile_valid_bitmap,
                tile_locations,
                texture_resolution: normalmap_resolution as u32,
                texture_border_size: self.skirt as u32 - 2,
                texture_format: TextureFormat::RG8,
            },
        );

        Ok(())
    }

    fn generate_noise(&mut self, _context: &mut AssetLoadContext) -> Result<NoiseParams, Error> {
        let noise = NoiseParams {
            texture: TextureDescriptor {
                offset: self.bytes_written,
                resolution: 2048,
                format: TextureFormat::RGBA8,
                bytes: 4 * 2048 * 2048,
            },
            wavelength: 1.0 / 256.0,
        };

        let noise_heightmaps: Vec<_> = (0..4)
            .map(|_| {
                let mut octaves = vec![
                    heightmap::wavelet_noise(512, 4),
                    heightmap::wavelet_noise(256, 8),
                    heightmap::wavelet_noise(128, 16),
                    heightmap::wavelet_noise(64, 32),
                    heightmap::wavelet_noise(32, 64),
                ];
                let mut heightmap = octaves.remove(0);
                assert_eq!(octaves.len(), 4);
                for octave in octaves {
                    assert_eq!(heightmap.heights.len(), octave.heights.len());
                    for i in 0..heightmap.heights.len() {
                        heightmap.heights[i] += octave.heights[i];
                    }
                }
                heightmap
            })
            .collect();

        for i in 0..noise_heightmaps[0].heights.len() {
            for j in 0..4 {
                let v = (noise_heightmaps[j].heights[i] * 0.2).max(-3.0).min(3.0);
                self.writer.write_u8((v * 127.5 / 3.0 + 127.5) as u8)?;
                self.bytes_written += 1;
            }
        }
        assert_eq!(self.bytes_written, noise.texture.offset + noise.texture.bytes);
        Ok(noise)
    }

    fn generate_planet_mesh(
        &mut self,
        _context: &mut AssetLoadContext,
    ) -> Result<MeshDescriptor, Error> {
        fn write_vertex<W: Write>(writer: &mut W, v: Vector3<f32>) -> Result<(), Error> {
            writer.write_f32::<LittleEndian>(v.x)?;
            writer.write_f32::<LittleEndian>(v.y)?;
            writer.write_f32::<LittleEndian>(v.z)?;
            Ok(())
        };

        let root_side_length = self.nodes[0].side_length();
        let offset = self.bytes_written;
        let mut num_vertices = 0;

        let resolution =
            Vector2::new(self.heights_resolution / 8, (self.heights_resolution - 1) / 2 + 1);
        for i in 0..4 {
            let mut vertices = Vec::new();
            for y in 0..resolution.y {
                for x in 0..resolution.x {
                    // grid coordinates
                    let fx = x as f64 / (resolution.x - 1) as f64;
                    let fy = y as f64 / (resolution.y - 1) as f64;

                    // circle coordinates
                    let theta = PI * (0.25 + 0.5 * fy);
                    let cx = (theta.sin() - (PI * 0.25).sin()) * 0.5 * 2f64.sqrt();
                    let cy = ((PI * 0.25).cos() - theta.cos()) / 2f64.sqrt();

                    // Interpolate between the two points.
                    let x = fx * cx;
                    let y = fy * (1.0 - fx) + cy * fx;

                    // Compute location in world space.
                    let world = Vector2::new(
                        (x + 0.5) * root_side_length as f64,
                        (y - 0.5) * root_side_length as f64,
                    );
                    let world = match i {
                        0 => world,
                        1 => Vector2::new(world.y, world.x),
                        2 => Vector2::new(-world.x, world.y),
                        3 => Vector2::new(world.y, -world.x),
                        _ => unreachable!(),
                    };

                    // Project onto ellipsoid.
                    let mut world3 = Vector3::new(
                        world.x,
                        EARTH_RADIUS
                            * ((1.0 - world.magnitude2() / EARTH_RADIUS).max(0.25).sqrt() - 1.0),
                        world.y,
                    );
                    for _ in 0..5 {
                        world3.x = world.x;
                        world3.z = world.y;
                        let mut lla = self.system.world_to_lla(world3);
                        lla.z = 0.0;
                        world3 = self.system.lla_to_world(lla);
                    }

                    vertices.push(Vector3::new(world.x as f32, world3.y as f32, world.y as f32));
                }
            }

            for y in 0..(resolution.y - 1) as usize {
                for x in 0..(resolution.x - 1) as usize {
                    let v00 = vertices[x + y * resolution.x as usize];
                    let v10 = vertices[x + 1 + y * resolution.x as usize];
                    let v01 = vertices[x + (y + 1) * resolution.x as usize];
                    let v11 = vertices[x + 1 + (y + 1) * resolution.x as usize];

                    // To support back face culling, we must invert draw order if the vertices were
                    // flipped above.
                    if i == 0 || i == 3 {
                        write_vertex(&mut self.writer, v00)?;
                        write_vertex(&mut self.writer, v10)?;
                        write_vertex(&mut self.writer, v01)?;

                        write_vertex(&mut self.writer, v11)?;
                        write_vertex(&mut self.writer, v01)?;
                        write_vertex(&mut self.writer, v10)?;
                    } else {
                        write_vertex(&mut self.writer, v00)?;
                        write_vertex(&mut self.writer, v01)?;
                        write_vertex(&mut self.writer, v10)?;

                        write_vertex(&mut self.writer, v11)?;
                        write_vertex(&mut self.writer, v10)?;
                        write_vertex(&mut self.writer, v01)?;
                    }

                    self.bytes_written += 72;
                    num_vertices += 6;
                }
            }
        }

        let mut vertices = Vec::new();
        let radius = root_side_length as f64 * 0.5 * 2f64.sqrt();
        let resolution =
            Vector2::new(self.heights_resolution / 4, ((self.heights_resolution - 1) / 2) * 4);

        for y in 0..resolution.y {
            let fy = y as f64 / resolution.y as f64;
            let theta = 2.0 * PI * fy;

            let tworld = Vector2::new(theta.cos() * radius, theta.sin() * radius);
            let mut tworld3 = Vector3::new(tworld.x, 0.0, tworld.y);
            for _ in 0..5 {
                tworld3.x = tworld.x;
                tworld3.z = tworld.y;
                let mut lla = self.system.world_to_lla(tworld3);
                lla.z = 0.0;
                tworld3 = self.system.lla_to_world(lla);
            }

            let phi_min = f64::acos((EARTH_RADIUS + tworld3.y) / EARTH_RADIUS);

            for x in 0..resolution.x {
                let fx = x as f64 / (resolution.x - 1) as f64;
                let phi = phi_min + fx * (100f64.to_radians() - phi_min);

                let world = Vector3::new(tworld3.x, (phi.cos() - 1.0) * EARTH_RADIUS, tworld3.z);
                let lla = self.system.world_to_lla(world);
                let surface_point = self.system.lla_to_world(Vector3::new(lla.x, lla.y, 0.0));

                vertices.push(Vector3::new(
                    surface_point.x as f32,
                    surface_point.y as f32,
                    surface_point.z as f32,
                ));
            }
        }
        for y in 0..resolution.y as usize {
            for x in 0..(resolution.x - 1) as usize {
                let v00 = vertices[x + y * resolution.x as usize];
                let v10 = vertices[x + 1 + y * resolution.x as usize];
                let v01 = vertices[x + ((y + 1) % resolution.y as usize) * resolution.x as usize];
                let v11 =
                    vertices[x + 1 + ((y + 1) % resolution.y as usize) * resolution.x as usize];

                write_vertex(&mut self.writer, v00)?;
                write_vertex(&mut self.writer, v10)?;
                write_vertex(&mut self.writer, v01)?;

                write_vertex(&mut self.writer, v11)?;
                write_vertex(&mut self.writer, v01)?;
                write_vertex(&mut self.writer, v10)?;

                self.bytes_written += 72;
                num_vertices += 6;
            }
        }

        Ok(MeshDescriptor { bytes: self.bytes_written - offset, offset, num_vertices })
    }

    fn generate_planet_mesh_texture(
        &mut self,
        context: &mut AssetLoadContext,
    ) -> Result<TextureDescriptor, Error> {
        let resolution = 8 * (self.heightmap_resolution - 1 - 2 * self.skirt) as usize;
        let descriptor = TextureDescriptor {
            offset: self.bytes_written,
            resolution: resolution as u32,
            format: TextureFormat::SRGBA,
            bytes: resolution * resolution * 4,
        };

        struct PlanetMesh<'a> {
            name: String,
            system: &'a CoordinateSystem,
            resolution: usize,
        };
        impl<'a> MMappedAsset for PlanetMesh<'a> {
            type Header = usize;
            fn filename(&self) -> String {
                self.name.clone()
            }
            fn generate<W: Write>(
                &self,
                context: &mut AssetLoadContext,
                mut writer: W,
            ) -> Result<Self::Header, Error> {
                let bluemarble = BlueMarble.load(context)?;

                let mut bytes_written = 0;
                for y in 0..self.resolution {
                    for x in 0..self.resolution {
                        let fx = 2.0 * (x as f64 + 0.5) / self.resolution as f64 - 1.0;
                        let fy = 2.0 * (y as f64 + 0.5) / self.resolution as f64 - 1.0;
                        let r = (fx * fx + fy * fy).sqrt().min(1.0);

                        let phi = r * PI;
                        let theta = f64::atan2(fy, fx);

                        let world3 = Vector3::new(
                            EARTH_RADIUS * theta.cos() * phi.sin(),
                            EARTH_RADIUS * (phi.cos() - 1.0),
                            EARTH_RADIUS * theta.sin() * phi.sin(),
                        );
                        let lla = self.system.world_to_lla(world3);

                        let brighten = |x: f64| (255.0 * (x / 255.0).powf(0.6)) as u8;

                        let (lat, long) = (lla.x.to_degrees(), lla.y.to_degrees());
                        let r = brighten(bluemarble.interpolate(lat, long, 0));
                        let g = brighten(bluemarble.interpolate(lat, long, 1));
                        let b = brighten(bluemarble.interpolate(lat, long, 2));
                        let a = 0; //watermask.interpolate(lat, long, 0) as u8;

                        writer.write_u8(r)?;
                        writer.write_u8(g)?;
                        writer.write_u8(b)?;
                        writer.write_u8(a)?;
                        bytes_written += 4;
                    }
                }
                Ok(bytes_written)
            }
        }

        let (bytes, mmap) = PlanetMesh {
            name: format!("{}planetmesh-texture", self.directory_name),
            system: &self.system,
            resolution,
        }
        .load(context)?;
        self.writer.write_all(&mmap[..bytes])?;
        self.bytes_written += bytes;

        Ok(descriptor)
    }
}
