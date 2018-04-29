use std::cell::RefCell;
use std::f64::consts::PI;
use std::io::Write;
use std::rc::Rc;

use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::*;
use failure::Error;
use gfx;
use gfx_core;
use rand::{self, Rng};
use rand::distributions::{IndependentSample, Normal};

use cache::{AssetLoadContext, MMappedAsset, WebAsset};
use coordinates::CoordinateSystem;
use sky::Skybox;
use srgb::{LINEAR_TO_SRGB, SRGB_TO_LINEAR};
use terrain::dem::DemSource;
use terrain::heightmap::{self, Heightmap};
use terrain::material::{MaterialSet, MaterialType};
use terrain::quadtree::{node, Node, NodeId, QuadTree};
use terrain::raster::{BlurredSource, RasterCache};
use terrain::reprojected_raster::{DataType, RasterSource, ReprojectedDemDef, ReprojectedRaster,
                                  ReprojectedRasterDef};
use terrain::tile_cache::{ByteRange, LayerParams, LayerType, MeshDescriptor, NoiseParams,
                          PayloadType, TextureDescriptor, TileHeader};
use terrain::landcover::{BlueMarble, GlobalWaterMask, LandCoverKind};
use runtime_texture::TextureFormat;
use utils::math::BoundingBox;

/// The radius of the earth in meters.
const EARTH_RADIUS: f64 = 6371000.0;
const EARTH_CIRCUMFERENCE: f64 = 2.0 * PI * EARTH_RADIUS;

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
        Self {
            latitude: 38,
            longitude: -122,
            source: DemSource::Srtm30m,
            vertex_quality: VertexQuality::High,
            texture_quality: TextureQuality::High,
            materials: MaterialSet::load(&mut factory, encoder).unwrap(),
            sky: Skybox::new(&mut factory, encoder),
            context: Some(AssetLoadContext::new()),
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
        color_buffer: &gfx::handle::RenderTargetView<R, gfx::format::Srgba8>,
        depth_buffer: &gfx::handle::DepthStencilView<R, gfx::format::DepthStencil>,
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
        )
    }

    fn name(&self) -> String {
        let n_or_s = if self.latitude >= 0 { 'n' } else { 's' };
        let e_or_w = if self.longitude >= 0 { 'e' } else { 'w' };
        format!(
            "{}{:02}_{}{:03}_{}m_{}_{}",
            n_or_s,
            self.latitude.abs(),
            e_or_w,
            self.longitude.abs(),
            self.source.resolution(),
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
        let sun_direction = Vector3::new(0.0, 1.0, 1.0).normalize();

        // Cell size in the y (latitude) direction, in meters. The x (longitude) direction will have
        // smaller cell sizes due to the projection.
        let dem_cell_size_y =
            self.source.cell_size() / (360.0 * 60.0 * 60.0) * EARTH_CIRCUMFERENCE as f32;

        let resolution_ratio =
            self.texture_quality.resolution() / (self.vertex_quality.resolution() - 1);
        assert!(resolution_ratio > 0);

        let world_size = 4194304.0;
        let max_level = 22i32 - self.vertex_quality.resolution_log2() as i32 - 1;
        let max_texture_level = max_level - (resolution_ratio as f32).log2() as i32;

        let cell_size =
            world_size / ((self.vertex_quality.resolution() - 1) as f32) * (0.5f32).powi(max_level);
        assert_eq!(cell_size, 2.0);
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
                    .map(|_| normal.ind_sample(&mut rand::thread_rng()) as f32)
                    .collect();
                Heightmap::new(v, 15, 15)
            },
            dem_source: self.source,
            heightmap_resolution,
            heights_resolution: self.vertex_quality.resolution(),
            max_texture_level,
            resolution_ratio,
            writer,
            heightmaps: None,
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
            directory_name: format!("maps/t.{}/", self.name()),
        };

        context.set_progress_and_total(0, 4);
        state.generate_heightmaps(context)?;
        context.set_progress(1);
        state.generate_normalmaps(context)?;
        context.set_progress(2);
        state.generate_watermasks(context)?;
        context.set_progress(3);
        state.generate_colormaps(context)?;
        context.set_progress(4);

        state.generate_trees(context)?;

        let planet_mesh = state.generate_planet_mesh(context)?;
        let planet_mesh_texture = state.generate_planet_mesh_texture(context)?;
        let noise = state.generate_noise(context)?;
        let State { layers, nodes, .. } = state;

        Ok(TileHeader {
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

    /// Resolution of the heightmap for each quadtree node.
    heights_resolution: u16,
    /// Resolution of the intermediate heightmaps which are used to generate normalmaps and
    /// colormaps. Derived from the target texture resolution.
    heightmap_resolution: u16,

    skirt: u16,
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

    directory_name: String,
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
            })
            .collect();
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
        let colormap_resolution = self.heightmap_resolution - 5;
        let tile_count = self.heightmaps.as_ref().unwrap().len();
        let tile_bytes = 4 * colormap_resolution as usize * colormap_resolution as usize;
        let tile_locations = (0..tile_count)
            .map(|i| ByteRange {
                offset: self.bytes_written + i * tile_bytes,
                length: tile_bytes,
            })
            .collect();

        self.layers.push(LayerParams {
            layer_type: LayerType::Colors,
            tile_locations,
            payload_type: PayloadType::Texture {
                resolution: colormap_resolution as u32,
                border_size: self.skirt as u32 - 2,
                format: TextureFormat::SRGBA,
            },
        });
        context.increment_level(
            "Generating colormaps... ",
            self.heightmaps.as_ref().unwrap().len(),
        );

        let reproject_treecover = ReprojectedRasterDef::<u8> {
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
        // let treecover = ReprojectedRaster::from_raster(reproject_treecover, context)?;

        let reproject_bluemarble = ReprojectedRasterDef {
            name: format!("{}bluemarble", self.directory_name),
            heights: self.heightmaps.as_ref().unwrap(),
            system: &self.system,
            nodes: &self.nodes,
            skirt: self.skirt,
            datatype: DataType::U8,
            raster: RasterSource::GlobalRaster {
                global: Box::new(BlueMarble),
            },
        };
        let bluemarble = ReprojectedRaster::from_raster(reproject_bluemarble, context)?;

        let mut colormaps: Vec<Vec<u8>> = Vec::new();
        for i in 0..self.heightmaps.as_ref().unwrap().len() {
            context.set_progress(i as u64);
            self.nodes[i].tile_indices[LayerType::Colors.index()] = Some(i as u32);

            let mut colormap =
                Vec::with_capacity(colormap_resolution as usize * colormap_resolution as usize);
            let heights = self.heightmaps.as_ref().unwrap();
            let spacing =
                self.nodes[i].side_length / (self.heightmap_resolution - 2 * self.skirt) as f32;
            let use_blue_marble = spacing >= bluemarble.spacing().unwrap() as f32;
            for y in 2..(2 + colormap_resolution) {
                for x in 2..(2 + colormap_resolution) {
                    // let world = self.world_position(x as i32, y as i32, self.nodes[i].bounds);
                    let h00 = heights.get(i, x, y, 0);
                    let h01 = heights.get(i, x, y + 1, 0);
                    let h10 = heights.get(i, x + 1, y, 0);
                    let h11 = heights.get(i, x + 1, y + 1, 0);
                    // let h = (h00 + h01 + h10 + h11) as f64 * 0.25;
                    // let lla = self.system.world_to_lla(Vector3::new(world.x, h, world.y));
                    // let (lat, long) = (lla.x.to_degrees(), lla.y.to_degrees());

                    let normal =
                        Vector3::new(h10 + h11 - h00 - h01, 2.0 * spacing, h01 + h11 - h00 - h10)
                            .normalize();
                    let light = (normal.dot(self.sun_direction).max(0.0) * 255.0) as u8;

                    let color = if use_blue_marble {
                        let r = bluemarble.get(i, x, y, 0) as u8;
                        let g = bluemarble.get(i, x, y, 1) as u8;
                        let b = bluemarble.get(i, x, y, 2) as u8;
                        [r, g, b, light]
                    } else {
                        let splat = Self::compute_splat(normal.y);
                        let albedo = self.materials.get_average_albedo(splat);
                        [albedo[0], albedo[1], albedo[2], light]
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
                            let p = stl(pc[i + ((x / 2 + offset.x) + (y / 2 + offset.y) * resolution) * 4]);
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
        let normalmap_nodes: Vec<_> = (0..self.heightmaps.as_ref().unwrap().len())
            .filter(|&i| self.nodes[i].level as i32 == self.max_texture_level)
            .collect();
        let tile_count = normalmap_nodes.len();
        let tile_bytes = 4 * normalmap_resolution as usize * normalmap_resolution as usize;
        let tile_locations = (0..tile_count)
            .map(|i| ByteRange {
                offset: self.bytes_written + i * tile_bytes,
                length: tile_bytes,
            })
            .collect();
        self.layers.push(LayerParams {
            layer_type: LayerType::Normals,
            tile_locations,
            payload_type: PayloadType::Texture {
                resolution: normalmap_resolution as u32,
                border_size: self.skirt as u32 - 2,
                format: TextureFormat::RGBA8,
            },
        });
        context.increment_level("Generating normalmaps... ", normalmap_nodes.len());
        for (i, id) in normalmap_nodes.into_iter().enumerate() {
            context.set_progress(i as u64);
            self.nodes[id].tile_indices[LayerType::Normals.index()] = Some(i as u32);

            let heights = self.heightmaps.as_ref().unwrap();
            let spacing =
                self.nodes[id].side_length / (self.heightmap_resolution - 2 * self.skirt) as f32;
            for y in 2..(2 + normalmap_resolution) {
                for x in 2..(2 + normalmap_resolution) {
                    let h00 = heights.get(id, x, y, 0);
                    let h01 = heights.get(id, x, y + 1, 0);
                    let h10 = heights.get(id, x + 1, y, 0);
                    let h11 = heights.get(id, x + 1, y + 1, 0);

                    let normal =
                        Vector3::new(h10 + h11 - h00 - h01, 2.0 * spacing, h01 + h11 - h00 - h10)
                            .normalize();

                    self.writer.write_u8((normal.x * 127.5 + 127.5) as u8)?;
                    self.writer.write_u8((normal.y * 127.5 + 127.5) as u8)?;
                    self.writer.write_u8((normal.z * 127.5 + 127.5) as u8)?;
                    self.writer.write_u8(Self::compute_splat(normal.y) as u8)?;
                    self.bytes_written += 4;
                }
            }
        }
        context.decrement_level();
        Ok(())
    }
    fn generate_watermasks(&mut self, context: &mut AssetLoadContext) -> Result<(), Error> {
        assert!(self.skirt >= 2);
        let watermap_resolution = self.heightmap_resolution - 5;
        let tile_count = self.heightmaps.as_ref().unwrap().len();
        let tile_bytes = 4 * watermap_resolution as usize * watermap_resolution as usize;
        let tile_locations = (0..tile_count)
            .map(|i| ByteRange {
                offset: self.bytes_written + i * tile_bytes,
                length: tile_bytes,
            })
            .collect();
        self.layers.push(LayerParams {
            layer_type: LayerType::Water,
            tile_locations,
            payload_type: PayloadType::Texture {
                resolution: watermap_resolution as u32,
                border_size: self.skirt as u32 - 2,
                format: TextureFormat::RGBA8,
            },
        });
        context.increment_level(
            "Generating water masks... ",
            self.heightmaps.as_ref().unwrap().len(),
        );

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
                            256,
                        ))),
                        30.0,
                    )),
                    128,
                ))),
            },
        };
        let watermasks = ReprojectedRaster::from_raster(reproject, context)?;

        for i in 0..self.heightmaps.as_ref().unwrap().len() {
            context.set_progress(i as u64);
            self.nodes[i].tile_indices[LayerType::Water.index()] = Some(i as u32);

            for y in 2..(2 + watermap_resolution) {
                for x in 2..(2 + watermap_resolution) {
                    self.writer.write_u8(watermasks.get(i, x, y, 0) as u8)?;
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
            })
            .collect();

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
    fn generate_trees(&mut self, context: &mut AssetLoadContext) -> Result<(), Error> {
        let step = 4;
        let resolution = (self.heightmap_resolution - 2 * self.skirt - 1) / step;
        let nodes: Vec<_> = (0..self.heightmaps.as_ref().unwrap().len())
            .filter(|&i| self.nodes[i].level as i32 == self.max_texture_level)
            .collect();
        let mut tile_locations = Vec::new();

        let mut rng = rand::thread_rng();

        context.increment_level("Generating trees... ", nodes.len());
        for (i, id) in nodes.into_iter().enumerate() {
            context.set_progress(i as u64);
            let offset = self.bytes_written;
            let heights = self.heightmaps.as_ref().unwrap();
            let spacing =
                self.nodes[id].side_length / (self.heightmap_resolution - 2 * self.skirt) as f32;
            for y in 0..resolution {
                for x in 0..resolution {
                    let h00 = heights.get(id, self.skirt + step * x, self.skirt + step * y, 0);
                    let h01 = heights.get(id, self.skirt + step * x, self.skirt + step * y + 1, 0);
                    let h10 = heights.get(id, self.skirt + step * x + 1, self.skirt + step * y, 0);
                    let h11 =
                        heights.get(id, self.skirt + step * x + 1, self.skirt + step * y + 1, 0);

                    let normal =
                        Vector3::new(h10 + h11 - h00 - h01, 2.0 * spacing, h01 + h11 - h00 - h10)
                            .normalize();

                    if normal.y > 0.965 {
                        let position = self.world_positionf(
                            (step as f32) * (x as f32 + rng.gen::<f32>()),
                            (step as f32) * (y as f32 + rng.gen::<f32>()),
                            self.nodes[id].bounds,
                        );
                        let position = Vector3::new(
                            position.x as f32,
                            (h00 + h01 + h10 + h11) * 0.25,
                            position.y as f32,
                        );
                        let color = Vector3::new(0.0, 0.0, 0.0);
                        let rotation = 0.0;
                        let texture_layer = 0.0;

                        self.writer.write_f32::<LittleEndian>(position.x)?;
                        self.writer.write_f32::<LittleEndian>(position.y)?;
                        self.writer.write_f32::<LittleEndian>(position.z)?;
                        self.writer.write_f32::<LittleEndian>(color.x)?;
                        self.writer.write_f32::<LittleEndian>(color.y)?;
                        self.writer.write_f32::<LittleEndian>(color.z)?;
                        self.writer.write_f32::<LittleEndian>(rotation)?;
                        self.writer.write_f32::<LittleEndian>(texture_layer)?;
                        self.bytes_written += 32;
                    }
                }
            }

            self.nodes[id].tile_indices[LayerType::Foliage.index()] = Some(i as u32);
            tile_locations.push(ByteRange {
                offset,
                length: self.bytes_written - offset,
            });
        }
        self.layers.push(LayerParams {
            layer_type: LayerType::Foliage,
            tile_locations,
            payload_type: PayloadType::InstancedMesh {
                max_instances: resolution as usize * resolution as usize,
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
            self.heights_resolution,
            ((self.heights_resolution - 1) / 2) * 4,
        );
        for y in 0..resolution.y {
            for x in 0..resolution.x {
                let fx = x as f64 / (resolution.x - 1) as f64;
                let fy = y as f64 / resolution.y as f64;
                let theta = 2.0 * PI * fy;
                let world = Vector2::new(theta.cos() * radius, theta.sin() * radius);
                let mut world3 = Vector3::new(
                    world.x,
                    EARTH_RADIUS * (1.0 - radius * radius / EARTH_RADIUS).max(0.25).sqrt()
                        - EARTH_RADIUS,
                    world.y,
                );
                for _ in 0..5 {
                    world3.x = world.x;
                    world3.z = world.y;
                    let mut lla = self.system.world_to_lla(world3);
                    lla.z = 0.0;
                    world3 = self.system.lla_to_world(lla);
                }

                if x == 0 {
                    world3 = Vector3::new(world.x, world3.y, world.y);
                } else {
                    world3 = Vector3::new(world.x, world3.y - EARTH_RADIUS * 2.0 * fx, world.y);
                    let mut lla = self.system.world_to_lla(world3);
                    lla.z = 0.0;
                    world3 = self.system.lla_to_world(lla);
                }

                vertices.push(Vector3::new(
                    world3.x as f32,
                    world3.y as f32,
                    world3.z as f32,
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
        for y in 0..(resolution.y - 2) as usize {
            for i in [0, y + 2, y + 1].iter() {
                let v = vertices[(resolution.x as usize - 1) + i * resolution.x as usize];
                write_vertex(&mut self.writer, v)?;
            }
            self.bytes_written += 36;
            num_vertices += 3;
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

                        let (lat, long) = (lla.x.to_degrees(), lla.y.to_degrees());
                        let r = bluemarble.interpolate(lat, long, 0) as u8;
                        let g = bluemarble.interpolate(lat, long, 1) as u8;
                        let b = bluemarble.interpolate(lat, long, 2) as u8;
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
        self.writer.write_all(&unsafe { mmap.as_slice() }[..bytes])?;
        self.bytes_written += bytes;

        Ok(descriptor)
    }
}
