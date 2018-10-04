use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::io::Write;
use std::rc::Rc;

use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::*;
use failure::Error;
use gfx;
use gfx_core;
use rand::distributions::{Distribution, Normal, Uniform};
use rand::{self, Rng};

use cache::{AssetLoadContext, MMappedAsset, WebAsset};
use coordinates::CoordinateSystem;
use runtime_texture::TextureFormat;
use sky::Skybox;
use srgb::{LINEAR_TO_SRGB, SRGB_TO_LINEAR};
use terrain::dem::DemSource;
use terrain::heightmap::{self, Heightmap};
use terrain::landcover::{BlueMarble, BlueMarbleTileSource, GlobalWaterMask, LandCoverKind};
use terrain::material::{MaterialSet, MaterialType};
use terrain::quadtree::{node, Node, NodeId, QuadTree};
use terrain::raster::{BlurredSource, RasterCache};
use terrain::reprojected_raster::{
    DataType, RasterSource, ReprojectedDemDef, ReprojectedRaster, ReprojectedRasterDef,
};
use terrain::tile_cache::{
    ByteRange, LayerParams, LayerType, MeshDescriptor, MeshInstance, NoiseParams, PayloadType,
    TextureDescriptor, TileHeader,
};
use utils::math::BoundingBox;

/// The radius of the earth in meters.
const EARTH_RADIUS: f64 = 6371000.0;
const EARTH_CIRCUMFERENCE: f64 = 2.0 * PI * EARTH_RADIUS;

// // Mapping from side length to level number.
// const LEVEL_4194_KM: i32 = 0;
// const LEVEL_2097_KM: i32 = 1;
// const LEVEL_1049_KM: i32 = 2;
// const LEVEL_524_KM: i32 = 3;
// const LEVEL_262_KM: i32 = 4;
// const LEVEL_131_KM: i32 = 5;
// const LEVEL_66_KM: i32 = 6;
// const LEVEL_33_KM: i32 = 7;
// const LEVEL_16_KM: i32 = 8;
const LEVEL_8_KM: i32 = 9;
// const LEVEL_4_KM: i32 = 10;
// const LEVEL_2_KM: i32 = 11;
// const LEVEL_1_KM: i32 = 12;
// const LEVEL_256_M: i32 = 13;
// const LEVEL_128_M: i32 = 14;
// const LEVEL_64_M: i32 = 15;
// const LEVEL_32_M: i32 = 16;
// const LEVEL_16_M: i32 = 17;
// const LEVEL_8_M: i32 = 18;
// const LEVEL_4_M: i32 = 19;
// const LEVEL_2_M: i32 = 20;
// const LEVEL_1_M: i32 = 21;

/// How much detail the terrain mesh should have. Higher values require more resources to render,
/// but produce nicer results.
pub enum VertexQuality {
    /// Probably overkill. Uses 4x as many triangles as high does.
    Ultra,
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
            VertexQuality::Ultra => 257,
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
            VertexQuality::Ultra => "vu",
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
    /// Bad looking at virtually any resolution, but very fast. Requires vertex quality of medium or
    /// lower.
    VeryLow,
}
impl TextureQuality {
    fn resolution(&self) -> u16 {
        match *self {
            TextureQuality::VeryLow => 64,
            TextureQuality::Low => 256,
            TextureQuality::High => 512,
            TextureQuality::Ultra => 1024,
        }
    }
    fn as_str(&self) -> &str {
        match *self {
            TextureQuality::VeryLow => "tvl",
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
pub struct QuadTreeBuilder<
    'a,
    R: gfx::Resources,
    F: gfx::Factory<R>,
    C: gfx_core::command::Buffer<R>,
> {
    latitude: i16,
    longitude: i16,
    source: DemSource,
    vertex_quality: VertexQuality,
    texture_quality: TextureQuality,
    grid_spacing: GridSpacing,
    materials: MaterialSet<R>,
    sky: Skybox<R>,
    factory: F,
    encoder: &'a mut gfx::Encoder<R, C>,
    context: Option<AssetLoadContext>,
}
impl<'a, R: gfx::Resources, F: gfx::Factory<R>, C: gfx_core::command::Buffer<R>>
    QuadTreeBuilder<'a, R, F, C>
{
    /// Create a new `QuadTreeBuilder` with default arguments.
    ///
    /// At very least, the latitude and longitude should probably be set to their desired values
    /// before calling `build()`.
    pub fn new(mut factory: F, encoder: &'a mut gfx::Encoder<R, C>) -> Self {
        let mut context = AssetLoadContext::new();
        Self {
            latitude: 38,
            longitude: -122,
            source: DemSource::Srtm30m,
            vertex_quality: VertexQuality::High,
            texture_quality: TextureQuality::High,
            grid_spacing: GridSpacing::TwoMeters,
            materials: MaterialSet::load(&mut factory, encoder, &mut context).unwrap(),
            sky: Skybox::new(&mut factory, encoder, &mut context),
            context: Some(context),
            factory,
            encoder,
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
    pub fn build(
        mut self,
        color_buffer: &gfx::handle::RenderTargetView<R, gfx::format::Rgba16F>,
        depth_buffer: &gfx::handle::DepthStencilView<R, gfx::format::Depth32F>,
    ) -> Result<QuadTree<R, F>, Error> {
        let mut context = self.context.take().unwrap();
        let (header, data) = self.load(&mut context)?;

        QuadTree::new(
            header,
            data,
            self.materials,
            self.sky,
            self.factory,
            self.encoder,
            color_buffer,
            depth_buffer,
            context,
        )
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

impl<'a, R: gfx::Resources, F: gfx::Factory<R>, C: gfx_core::command::Buffer<R>> MMappedAsset
    for QuadTreeBuilder<'a, R, F, C>
{
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
        let max_level =
            22i32 - self.vertex_quality.resolution_log2() as i32 - self.grid_spacing.log2_spacing();
        let max_texture_level = max_level - (resolution_ratio as f32).log2() as i32;
        let max_tree_density_level = LEVEL_8_KM;

        let cell_size =
            world_size / ((self.vertex_quality.resolution() - 1) as f32) * (0.5f32).powi(max_level);
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
                let normal = Normal::new(0.0, 1.0);
                let v = (0..(15 * 15))
                    .map(|_| normal.sample(&mut rand::thread_rng()) as f32)
                    .collect();
                Heightmap::new(v, 15, 15)
            },
            dem_source: self.source,
            heightmap_resolution,
            heights_resolution: self.vertex_quality.resolution(),
            max_texture_level,
            max_tree_density_level,
            resolution_ratio,
            writer,
            heightmaps: None,
            treecover: None,
            tree_placements: HashMap::new(),
            max_dem_level,
            materials: &self.materials,
            skirt,
            system: CoordinateSystem::from_lla(Vector3::new(
                world_center.y.to_radians() as f64,
                world_center.x.to_radians() as f64,
                0.0,
            )),
            nodes: Node::make_nodes(world_size, 3000.0, max_level as u8),
            layers: Vec::new(),
            bytes_written: 0,
            directory_name: format!("maps/t.{}/", self.name()),
        };

        context.set_progress_and_total(0, 8);
        state.generate_heightmaps(context)?;
        context.set_progress(1);
        state.generate_normalmaps(context)?;
        context.set_progress(2);
        state.generate_splats(context)?;
        context.set_progress(3);
        state.place_trees(context)?;
        state.generate_colormaps(context)?;
        context.set_progress(4);

        state.write_trees(context)?;
        context.set_progress(5);

        let planet_mesh = state.generate_planet_mesh(context)?;
        context.set_progress(6);
        let planet_mesh_texture = state.generate_planet_mesh_texture(context)?;
        context.set_progress(7);
        let noise = state.generate_noise(context)?;
        let State {
            layers,
            nodes,
            system,
            ..
        } = state;

        context.set_progress(8);

        Ok(TileHeader {
            system,
            planet_mesh,
            planet_mesh_texture,
            layers,
            noise,
            nodes,
        })
    }
}

struct State<'a, W: Write, R: gfx::Resources> {
    dem_source: DemSource,

    random: Heightmap<f32>,
    heightmaps: Option<ReprojectedRaster>,

    treecover: Option<ReprojectedRaster>,
    tree_placements: HashMap<usize, Vec<MeshInstance>>,

    /// Resolution of the heightmap for each quadtree node.
    heights_resolution: u16,
    /// Resolution of the intermediate heightmaps which are used to generate normalmaps and
    /// colormaps. Derived from the target texture resolution.
    heightmap_resolution: u16,

    skirt: u16,

    max_texture_level: i32,
    max_dem_level: i32,
    max_tree_density_level: i32,

    resolution_ratio: u16,
    writer: W,
    materials: &'a MaterialSet<R>,

    system: CoordinateSystem,

    layers: Vec<LayerParams>,
    nodes: Vec<Node>,
    bytes_written: usize,

    directory_name: String,
}

impl<'a, W: Write, R: gfx::Resources> State<'a, W, R> {
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

    fn compute_splat(cos_slope: f32) -> MaterialType {
        if cos_slope > 0.9999 {
            MaterialType::Dirt
        } else if cos_slope > 0.965 {
            MaterialType::Grass
        // } else if cos_slope > 0.955 {
        //     MaterialType::GrassRocky
        } else if cos_slope > 0.95 {
            MaterialType::Rock
        } else {
            MaterialType::RockSteep
        }
    }

    fn generate_heightmaps(&mut self, context: &mut AssetLoadContext) -> Result<(), Error> {
        let tile_bytes = 4 * self.heights_resolution as usize * self.heights_resolution as usize;
        let tile_locations = (0..self.nodes.len())
            .map(|i| ByteRange {
                offset: i * tile_bytes,
                length: tile_bytes,
            }).collect();
        self.layers.push(LayerParams {
            layer_type: LayerType::Heights,
            tile_locations,
            payload_type: PayloadType::Texture {
                resolution: self.heights_resolution as u32,
                border_size: 0,
                format: TextureFormat::F32,
            },
        });

        let reproject = ReprojectedDemDef {
            name: format!("{}dem", self.directory_name),
            dem_cache: Rc::new(RefCell::new(RasterCache::new(
                Box::new(self.dem_source),
                128,
            ))),
            system: &self.system,
            nodes: &self.nodes,
            random: &self.random,
            skirt: self.skirt,
            max_dem_level: self.max_dem_level as u8,
            max_texture_level: self.max_texture_level as u8,
            resolution: self.heightmap_resolution,
        };
        self.heightmaps = Some(ReprojectedRaster::from_dem(reproject, context)?);

        context.increment_level("Writing heightmaps... ", self.nodes.len());
        for i in 0..self.nodes.len() {
            context.set_progress(i as u64);
            self.nodes[i].tile_indices[LayerType::Heights.index()] = Some(i as u32);

            let (heightmap, offset, step) = if self.nodes[i].level as i32 > self.max_texture_level {
                let (ancestor, generations, mut offset) =
                    Node::find_ancestor(&self.nodes, NodeId::new(i as u32), |id| {
                        self.nodes[id].level as i32 <= self.max_texture_level
                    }).unwrap();

                let ancestor = ancestor.index();
                let offset_scale = 1 << generations;
                let step = self.resolution_ratio >> generations;
                offset *= (self.heightmap_resolution - 2 * self.skirt) as i32 / offset_scale;
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
                    let height = self.heightmaps.as_ref().unwrap().get(
                        heightmap,
                        x * step + offset.x + self.skirt,
                        y * step + offset.y + self.skirt,
                        0,
                    );

                    miny = Some(match miny {
                        Some(y) if y < height => y,
                        _ => height,
                    });
                    maxy = Some(match maxy {
                        Some(y) if y > height => y,
                        _ => height,
                    });

                    self.writer.write_f32::<LittleEndian>(height)?;
                    self.bytes_written += 4;
                }
            }
            self.nodes[i].bounds.min.y = miny.unwrap();
            self.nodes[i].bounds.max.y = maxy.unwrap();
        }

        context.decrement_level();
        Ok(())
    }
    fn generate_colormaps(&mut self, context: &mut AssetLoadContext) -> Result<(), Error> {
        assert!(self.skirt >= 2);
        let colormap_skirt = self.skirt - 2;
        let colormap_resolution = self.heightmap_resolution - 5;
        let tile_count = self.heightmaps.as_ref().unwrap().len();
        let tile_bytes = 4 * colormap_resolution as usize * colormap_resolution as usize;
        let tile_locations = (0..tile_count)
            .map(|i| ByteRange {
                offset: self.bytes_written + i * tile_bytes,
                length: tile_bytes,
            }).collect();

        self.layers.push(LayerParams {
            layer_type: LayerType::Colors,
            tile_locations,
            payload_type: PayloadType::Texture {
                resolution: colormap_resolution as u32,
                border_size: colormap_skirt as u32,
                format: TextureFormat::SRGBA,
            },
        });
        context.increment_level(
            "Generating colormaps... ",
            self.heightmaps.as_ref().unwrap().len(),
        );

        let reproject_bluemarble = ReprojectedRasterDef {
            name: format!("{}bluemarble", self.directory_name),
            heights: self.heightmaps.as_ref().unwrap(),
            system: &self.system,
            nodes: &self.nodes,
            skirt: self.skirt,
            datatype: DataType::U8,
            raster: RasterSource::Hybrid {
                global: Box::new(BlueMarble),
                cache: Rc::new(RefCell::new(RasterCache::new(
                    Box::new(BlueMarbleTileSource),
                    8,
                ))),
            },
        };
        let bluemarble = ReprojectedRaster::from_raster(reproject_bluemarble, context)?;

        let reproject = ReprojectedRasterDef {
            name: format!("{}watermasks", self.directory_name),
            heights: self.heightmaps.as_ref().unwrap(),
            system: &self.system,
            nodes: &self.nodes,
            skirt: self.skirt,
            datatype: DataType::U8,
            raster: RasterSource::Hybrid {
                global: Box::new(GlobalWaterMask),
                cache: Rc::new(RefCell::new(RasterCache::new(
                    Box::new(BlurredSource::new(
                        Rc::new(RefCell::new(RasterCache::new(
                            Box::new(LandCoverKind::WaterMask),
                            128,
                        ))),
                        0.0,
                    )),
                    64,
                ))),
            },
        };
        let watermasks = ReprojectedRaster::from_raster(reproject, context)?;

        let mix = |a: u8, b: u8, t: f32| (f32::from(a) * (1.0 - t) + f32::from(b) * t) as u8;

        let mut colormaps: Vec<Vec<u8>> = Vec::new();
        for i in 0..self.heightmaps.as_ref().unwrap().len() {
            context.set_progress(i as u64);
            self.nodes[i].tile_indices[LayerType::Colors.index()] = Some(i as u32);

            let mut colormap =
                Vec::with_capacity(colormap_resolution as usize * colormap_resolution as usize);
            let heights = self.heightmaps.as_ref().unwrap();
            let spacing =
                self.nodes[i].side_length / (self.heightmap_resolution - 2 * self.skirt) as f32;
            let use_blue_marble = spacing * 2.0 >= bluemarble.spacing().unwrap() as f32;
            for y in 2..(2 + colormap_resolution) {
                for x in 2..(2 + colormap_resolution) {
                    let h00 = heights.get(i, x, y, 0);
                    let h01 = heights.get(i, x, y + 1, 0);
                    let h10 = heights.get(i, x + 1, y, 0);
                    let h11 = heights.get(i, x + 1, y + 1, 0);

                    let normal = Vector3::new(
                        h10 + h11 - h00 - h01,
                        2.0 * spacing,
                        -1.0 * (h01 + h11 - h00 - h10),
                    ).normalize();

                    let water = watermasks.get(i, x, y, 0) as u8;
                    let color = if use_blue_marble {
                        let brighten = |x: f32| (255.0 * (x / 255.0).powf(0.6)) as u8;
                        let r = brighten(bluemarble.get(i, x, y, 0));
                        let g = brighten(bluemarble.get(i, x, y, 1));
                        let b = brighten(bluemarble.get(i, x, y, 2));
                        [r, g, b, water]
                    } else {
                        let splat = Self::compute_splat(normal.y);
                        let albedo = self.materials.get_average_albedo(splat);
                        if self.nodes[i].level <= self.max_tree_density_level as u8 {
                            let tree_density =
                                (self.treecover.as_ref().unwrap().get(i, x, y, 0) / 100.0).min(1.0);
                            [
                                mix(albedo[0], LINEAR_TO_SRGB[13], tree_density),
                                mix(albedo[1], LINEAR_TO_SRGB[31], tree_density),
                                mix(albedo[2], 0, tree_density),
                                water,
                            ]
                        } else {
                            [albedo[0], albedo[1], albedo[2], water]
                        }
                    };
                    colormap.extend_from_slice(&color);
                }
            }

            if let (Some(parent), false) = (self.nodes[i].parent.as_ref(), use_blue_marble) {
                let resolution = colormap_resolution as usize;
                let skirt = self.skirt as usize - 2;
                let offset = node::OFFSETS[parent.1 as usize];
                let offset = Point2::new(
                    skirt / 2 + offset.x as usize * (resolution / 2 - skirt),
                    skirt / 2 + offset.y as usize * (resolution / 2 - skirt),
                );

                let stl = |srgb| SRGB_TO_LINEAR[srgb] as i16;
                let lts = |linear: i16| LINEAR_TO_SRGB[linear.max(0).min(255) as u8];

                let pc = &colormaps[parent.0.index()];
                for y in (0..resolution).step_by(2) {
                    for x in (0..resolution).step_by(2) {
                        for i in 0..3 {
                            let p = stl(
                                pc[i + ((x / 2 + offset.x) + (y / 2 + offset.y) * resolution) * 4]
                            );
                            let c00 = stl(colormap[i + (x + y * resolution) * 4]);
                            let c10 = stl(colormap[i + ((x + 1) + y * resolution) * 4]);
                            let c01 = stl(colormap[i + (x + (y + 1) * resolution) * 4]);
                            let c11 = stl(colormap[i + ((x + 1) + (y + 1) * resolution) * 4]);

                            let shift = (p - (c00 + c01 + c10 + c11) / 4) / 2;

                            colormap[i + (x + y * resolution) * 4] = lts(c00 + shift);
                            colormap[i + ((x + 1) + y * resolution) * 4] = lts(c10 + shift);
                            colormap[i + (x + (y + 1) * resolution) * 4] = lts(c01 + shift);
                            colormap[i + ((x + 1) + (y + 1) * resolution) * 4] = lts(c11 + shift);
                        }
                    }
                }
            }

            let mut tree_placements = None;
            if self.tree_placements.contains_key(&i) {
                tree_placements = Some(i);
            } else if let Some((parent, _)) = self.nodes[i].parent {
                if self.tree_placements.contains_key(&parent.index()) {
                    tree_placements = Some(parent.index());
                }
            }
            if let Some(ancestor) = tree_placements {
                let bounds = &self.nodes[i].bounds;
                let side_length = self.nodes[i].side_length;
                let resolution = self.heightmap_resolution - self.skirt * 2 - 1;

                let cell_size = side_length / resolution as f32;
                let radius = (7.5 / cell_size).round() as isize;
                // let border = cell_size * colormap_skirt as f32 + radius as f32;

                for placement in self.tree_placements[&ancestor].iter() {
                    if placement.position[0] > bounds.min.x
                        && placement.position[2] > bounds.min.z
                        && placement.position[0] < bounds.max.x
                        && placement.position[2] < bounds.max.z
                    {
                        let x = (resolution as f32 * (placement.position[0] - bounds.min.x)
                            / side_length)
                            .round()
                            .max(0.0) as isize
                            + colormap_skirt as isize;
                        let z = (resolution as f32 * (placement.position[2] - bounds.min.z)
                            / side_length)
                            .round()
                            .max(0.0) as isize
                            + colormap_skirt as isize;

                        for h in -radius..=radius {
                            for k in -radius..=radius {
                                let x =
                                    (x + h).max(0).min(colormap_resolution as isize - 1) as usize;
                                let z =
                                    (z + k).max(0).min(colormap_resolution as isize - 1) as usize;
                                let color = &mut colormap
                                    [4 * (x + z * colormap_resolution as usize)..][..4];

                                for i in 0..3 {
                                    color[i] =
                                        LINEAR_TO_SRGB[(255.0 * placement.color[i]).round() as u8];
                                }
                            }
                        }
                    }
                }
            }

            colormaps.push(colormap);
        }

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
        let tile_count = self.heightmaps.as_ref().unwrap().len();
        let tile_bytes = 4 * normalmap_resolution as usize * normalmap_resolution as usize;
        let tile_locations = (0..tile_count)
            .map(|i| ByteRange {
                offset: self.bytes_written + i * tile_bytes,
                length: tile_bytes,
            }).collect();
        self.layers.push(LayerParams {
            layer_type: LayerType::Normals,
            tile_locations,
            payload_type: PayloadType::Texture {
                resolution: normalmap_resolution as u32,
                border_size: self.skirt as u32 - 2,
                format: TextureFormat::RGBA8,
            },
        });
        context.increment_level("Generating normalmaps... ", tile_count);
        for i in 0..tile_count {
            context.set_progress(i as u64);
            self.nodes[i].tile_indices[LayerType::Normals.index()] = Some(i as u32);

            let heights = self.heightmaps.as_ref().unwrap();
            let spacing =
                self.nodes[i].side_length / (self.heightmap_resolution - 2 * self.skirt) as f32;
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
                    ).normalize();

                    self.writer.write_u8((normal.x * 127.5 + 127.5) as u8)?;
                    self.writer.write_u8((normal.y * 127.5 + 127.5) as u8)?;
                    self.writer.write_u8((normal.z * 127.5 + 127.5) as u8)?;
                    self.writer.write_u8(0)?;
                    self.bytes_written += 4;
                }
            }
        }
        context.decrement_level();
        Ok(())
    }
    fn generate_splats(&mut self, context: &mut AssetLoadContext) -> Result<(), Error> {
        assert!(self.skirt >= 2);
        let splatmap_resolution = self.heightmap_resolution - 5;
        let splatmap_nodes: Vec<_> = (0..self.heightmaps.as_ref().unwrap().len())
            .filter(|&i| self.nodes[i].level as i32 == self.max_texture_level)
            .collect();
        let tile_count = splatmap_nodes.len();
        let tile_bytes = splatmap_resolution as usize * splatmap_resolution as usize;
        let tile_locations = (0..tile_count)
            .map(|i| ByteRange {
                offset: self.bytes_written + i * tile_bytes,
                length: tile_bytes,
            }).collect();
        self.layers.push(LayerParams {
            layer_type: LayerType::Splats,
            tile_locations,
            payload_type: PayloadType::Texture {
                resolution: splatmap_resolution as u32,
                border_size: self.skirt as u32 - 2,
                format: TextureFormat::R8,
            },
        });
        context.increment_level("Generating splats... ", splatmap_nodes.len());
        for (i, id) in splatmap_nodes.into_iter().enumerate() {
            context.set_progress(i as u64);
            self.nodes[id].tile_indices[LayerType::Splats.index()] = Some(i as u32);

            let heights = self.heightmaps.as_ref().unwrap();
            let spacing =
                self.nodes[id].side_length / (self.heightmap_resolution - 2 * self.skirt) as f32;
            for y in 2..(2 + splatmap_resolution) {
                for x in 2..(2 + splatmap_resolution) {
                    let h00 = heights.get(id, x, y, 0);
                    let h01 = heights.get(id, x, y + 1, 0);
                    let h10 = heights.get(id, x + 1, y, 0);
                    let h11 = heights.get(id, x + 1, y + 1, 0);

                    let normal = Vector3::new(
                        h10 + h11 - h00 - h01,
                        2.0 * spacing,
                        -1.0 * (h01 + h11 - h00 - h10),
                    ).normalize();

                    self.writer.write_u8(Self::compute_splat(normal.y) as u8)?;
                    self.bytes_written += 1;
                }
            }
        }
        context.decrement_level();
        Ok(())
    }

    fn generate_noise(&mut self, _context: &mut AssetLoadContext) -> Result<NoiseParams, Error> {
        let noise = NoiseParams {
            offset: self.bytes_written,
            resolution: 2048,
            format: TextureFormat::RGBA8,
            bytes: 4 * 2048 * 2048,
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
            }).collect();

        for i in 0..noise_heightmaps[0].heights.len() {
            for j in 0..4 {
                let v = (noise_heightmaps[j].heights[i] * 0.2).max(-3.0).min(3.0);
                self.writer.write_u8((v * 127.5 / 3.0 + 127.5) as u8)?;
                self.bytes_written += 1;
            }
        }
        assert_eq!(self.bytes_written, noise.offset + noise.bytes);
        Ok(noise)
    }
    fn place_trees(&mut self, context: &mut AssetLoadContext) -> Result<(), Error> {
        let step = 1;
        let resolution = (self.heightmap_resolution - 2 * self.skirt - 1) / step;
        let nodes: Vec<_> = (0..self.heightmaps.as_ref().unwrap().len())
            .filter(|&i| self.nodes[i].level as i32 == self.max_tree_density_level + 1)
            .collect();

        let reproject_treecover = ReprojectedRasterDef::<u8, Vec<u8>, Vec<u8>> {
            name: format!("{}treecover", self.directory_name),
            heights: self.heightmaps.as_ref().unwrap(),
            system: &self.system,
            nodes: &self.nodes,
            skirt: self.skirt,
            datatype: DataType::U8,
            raster: RasterSource::RasterCache {
                cache: Rc::new(RefCell::new(RasterCache::new(
                    Box::new(LandCoverKind::TreeCover),
                    256,
                ))),
                default: 0.0,
                radius2: Some(1_000_000.0 * 1_000_000.0),
            },
        };
        self.treecover = Some(ReprojectedRaster::from_raster(
            reproject_treecover,
            context,
        )?);

        let mut rng = rand::thread_rng();
        let color_dist = Normal::new(1.0, 0.08);
        let rotation_dist = Uniform::new(0.0, 2.0 * PI);

        context.increment_level("Placing trees... ", nodes.len());
        for (i, id) in nodes.into_iter().enumerate() {
            context.set_progress(i as u64);
            let heights = self.heightmaps.as_ref().unwrap();
            let spacing =
                self.nodes[id].side_length / (self.heightmap_resolution - 2 * self.skirt) as f32;

            let side_length = self.nodes[id].side_length;

            let _cell_size = side_length / resolution as f32; // = 8m
            let rmin = 0.0;
            let rmax = 1.0;

            self.tree_placements.insert(id, Vec::new());
            for y in 0..resolution {
                for x in 0..resolution {
                    let t = self.treecover.as_ref().unwrap().get(
                        id,
                        self.skirt + step * x,
                        self.skirt + step * y,
                        0,
                    );
                    if rng.gen_range(0.0, 255.0) > t {
                        continue;
                    }

                    let h00 = heights.get(id, self.skirt + step * x, self.skirt + step * y, 0);
                    let h01 = heights.get(id, self.skirt + step * x, self.skirt + step * y + 1, 0);
                    let h10 = heights.get(id, self.skirt + step * x + 1, self.skirt + step * y, 0);
                    let h11 =
                        heights.get(id, self.skirt + step * x + 1, self.skirt + step * y + 1, 0);

                    let normal = Vector3::new(
                        h10 + h11 - h00 - h01,
                        2.0 * spacing,
                        -1.0 * (h01 + h11 - h00 - h10),
                    ).normalize();

                    if normal.y > 0.965 {
                        let position = Vector3::new(
                            self.nodes[id].bounds.min.x
                                + (x as f32 + rng.gen_range(rmin, rmax)) / resolution as f32
                                    * side_length,
                            (h00 + h01 + h10 + h11) * 0.25,
                            self.nodes[id].bounds.min.z
                                + (y as f32 + rng.gen_range(rmin, rmax)) / resolution as f32
                                    * side_length,
                        );
                        let color = Vector3::new(
                            (color_dist.sample(&mut rng) as f32 * 0.1 - 0.1 + 13.0 / 255.0)
                                .max(0.0)
                                .min(1.0),
                            (color_dist.sample(&mut rng) as f32 * 0.1 - 0.1 + 31.0 / 255.0)
                                .max(0.0)
                                .min(1.0),
                            (color_dist.sample(&mut rng) as f32 * 0.1 - 0.1)
                                .max(0.0)
                                .min(1.0),
                        );

                        self.tree_placements
                            .get_mut(&id)
                            .unwrap()
                            .push(MeshInstance {
                                position: [position.x, position.y, position.z],
                                color: [color.x, color.y, color.z],
                                rotation: rotation_dist.sample(&mut rng) as f32,
                                scale: 1.5,
                                normal: [normal.x, normal.y, normal.z],
                                padding1: 0.0,
                                padding2: [0.0, 0.0, 0.0, 0.0],
                            });
                    }
                }
            }
        }
        context.decrement_level();
        Ok(())
    }
    fn write_trees(&mut self, context: &mut AssetLoadContext) -> Result<(), Error> {
        let nodes: Vec<_> = (0..self.heightmaps.as_ref().unwrap().len())
            .filter(|&i| self.nodes[i].level as i32 == self.max_tree_density_level + 2)
            .collect();
        let mut tile_locations = Vec::new();

        let mut max_instances = 0;
        context.increment_level("Placing trees... ", nodes.len());
        for (i, id) in nodes.into_iter().enumerate() {
            context.set_progress(i as u64);

            let offset = self.bytes_written;
            let bounds = &self.nodes[id].bounds;

            let mut node = id;
            while !self.tree_placements.contains_key(&node) {
                node = self.nodes[node]
                    .parent
                    .expect("No tree placement found for node")
                    .0
                    .index();
            }

            let mut instances = 0;
            for placement in &self.tree_placements[&node] {
                if placement.position[0] > bounds.min.x
                    && placement.position[2] > bounds.min.z
                    && placement.position[0] < bounds.max.x
                    && placement.position[2] < bounds.max.z
                {
                    for i in 0..3 {
                        self.writer
                            .write_f32::<LittleEndian>(placement.position[i])?;
                    }
                    for i in 0..3 {
                        self.writer.write_f32::<LittleEndian>(placement.color[i])?;
                    }
                    self.writer.write_f32::<LittleEndian>(placement.rotation)?;
                    self.writer.write_f32::<LittleEndian>(placement.scale)?;
                    for i in 0..3 {
                        self.writer.write_f32::<LittleEndian>(placement.normal[i])?;
                    }
                    for _ in 0..5 {
                        self.writer.write_f32::<LittleEndian>(0.0)?;
                    }
                    self.bytes_written += 64;
                    instances += 1;
                }
            }

            max_instances = instances.max(max_instances);
            self.nodes[id].tile_indices[LayerType::Foliage.index()] = Some(i as u32);
            tile_locations.push(ByteRange {
                offset,
                length: self.bytes_written - offset,
            });
        }

        #[rustfmt::skip]
        let mesh_data: Vec<f32> = vec![
            -5.0,  0.0,  0.0,    0.0, 0.0,    0.0, 0.0, 0.0,
             5.0,  0.0,  0.0,    0.5, 0.0,    0.0, 0.0, 0.0,
            -5.0, 10.0,  0.0,    0.0, 0.5,    0.0, 0.0, 0.0,
            -5.0, 10.0,  0.0,    0.0, 0.5,    0.0, 0.0, 0.0,
             5.0, 10.0,  0.0,    0.5, 0.5,    0.0, 0.0, 0.0,
             5.0,  0.0,  0.0,    0.5, 0.0,    0.0, 0.0, 0.0,

             0.0,  0.0, -5.0,    0.5, 0.0,    0.0, 0.0, 0.0,
             0.0,  0.0,  5.0,    1.0, 0.0,    0.0, 0.0, 0.0,
             0.0, 10.0, -5.0,    0.5, 0.5,    0.0, 0.0, 0.0,
             0.0, 10.0, -5.0,    0.5, 0.5,    0.0, 0.0, 0.0,
             0.0, 10.0,  5.0,    1.0, 0.5,    0.0, 0.0, 0.0,
             0.0,  0.0,  5.0,    1.0, 0.0,    0.0, 0.0, 0.0,

            -5.0,  5.0, -5.0,    0.5, 0.5,    0.0, 0.0, 0.0,
             5.0,  5.0, -5.0,    1.0, 0.5,    0.0, 0.0, 0.0,
             5.0,  5.0,  5.0,    1.0, 1.0,    0.0, 0.0, 0.0,
            -5.0,  5.0, -5.0,    0.5, 0.5,    0.0, 0.0, 0.0,
            -5.0,  5.0,  5.0,    0.5, 1.0,    0.0, 0.0, 0.0,
             5.0,  5.0,  5.0,    1.0, 1.0,    0.0, 0.0, 0.0,
        ];
        let mesh = MeshDescriptor {
            offset: self.bytes_written,
            bytes: mesh_data.len() * 4,
            num_vertices: mesh_data.len() / 8,
        };
        for f in mesh_data {
            self.writer.write_f32::<LittleEndian>(f)?;
        }
        self.bytes_written += mesh.bytes;

        let mut texture_data = vec![255u8; 256 * 256 * 4];
        for x in 0..16 {
            for h in 0..16 {
                for y in 0..16 {
                    for k in 0..16 {
                        for i in 0..4 {
                            texture_data[((x * 16 + h) + (y * 16 + k) * 256) * 4 + i] =
                                if x % 2 == y % 2 { 64 } else { 192 };
                        }
                    }
                }
            }
        }
        let texture = TextureDescriptor {
            offset: self.bytes_written,
            resolution: 256,
            format: TextureFormat::SRGBA,
            bytes: texture_data.len(),
        };
        self.writer.write_all(&texture_data[..])?;
        self.bytes_written += texture.bytes;

        self.layers.push(LayerParams {
            layer_type: LayerType::Foliage,
            tile_locations,
            payload_type: PayloadType::InstancedMesh {
                mesh,
                texture,
                max_instances,
            },
        });
        context.decrement_level();
        Ok(())
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

        let root_side_length = self.nodes[0].side_length;
        let offset = self.bytes_written;
        let mut num_vertices = 0;

        let resolution = Vector2::new(
            self.heights_resolution / 8,
            (self.heights_resolution - 1) / 2 + 1,
        );
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

                    vertices.push(Vector3::new(
                        world.x as f32,
                        world3.y as f32,
                        world.y as f32,
                    ));
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
        let resolution = Vector2::new(
            self.heights_resolution / 4,
            ((self.heights_resolution - 1) / 2) * 4,
        );

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

        Ok(MeshDescriptor {
            bytes: self.bytes_written - offset,
            offset,
            num_vertices,
        })
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
                let watermask = GlobalWaterMask.load(context)?;

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
                        let a = watermask.interpolate(lat, long, 0) as u8;

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
        }.load(context)?;
        self.writer.write_all(&mmap[..bytes])?;
        self.bytes_written += bytes;

        Ok(descriptor)
    }
}
