use std::cell::RefCell;
use std::f64::consts::PI;
use std::io::Write;
use std::rc::Rc;

use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::*;
use failure::Error;
use gfx;
use gfx_core;
use rand;
use rand::distributions::{IndependentSample, Normal};

use cache::{AssetLoadContext, MMappedAsset, WebAsset};
use coordinates::CoordinateSystem;
use sky::Skybox;
use terrain::dem::DemSource;
use terrain::heightmap::{self, Heightmap};
use terrain::material::MaterialSet;
use terrain::quadtree::{node, Node, NodeId, QuadTree};
use terrain::raster::{BitContainer, GlobalRaster, RasterCache};
use terrain::reprojected_raster::{RasterSource, ReprojectedDemDef, ReprojectedRaster,
                                  ReprojectedRasterDef};
use terrain::tile_cache::{LayerParams, LayerType, MeshDescriptor, NoiseParams, TextureDescriptor,
                          TileHeader};
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
pub struct QuadTreeBuilder<R: gfx::Resources, F: gfx::Factory<R>> {
    latitude: i16,
    longitude: i16,
    source: DemSource,
    vertex_quality: VertexQuality,
    texture_quality: TextureQuality,
    materials: MaterialSet<R>,
    sky: Skybox<R>,
    factory: F,
    context: Option<AssetLoadContext>,
}
impl<R: gfx::Resources, F: gfx::Factory<R>> QuadTreeBuilder<R, F> {
    /// Create a new `QuadTreeBuilder` with default arguments.
    ///
    /// At very least, the latitude and longitude should probably be set to their desired values
    /// before calling `build()`.
    pub fn new<C: gfx_core::command::Buffer<R>>(
        mut factory: F,
        encoder: &mut gfx::Encoder<R, C>,
    ) -> Self {
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

impl<R: gfx::Resources, F: gfx::Factory<R>> MMappedAsset for QuadTreeBuilder<R, F> {
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
            bluemarble: BlueMarble.load(context)?,
            global_watermask: GlobalWaterMask.load(context)?,
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
        state.generate_colormaps(context)?;
        context.set_progress(3);
        state.generate_watermasks(context)?;
        context.set_progress(4);

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
    bluemarble: GlobalRaster<u8>,
    global_watermask: GlobalRaster<f64, BitContainer>,
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

    fn generate_heightmaps(&mut self, context: &mut AssetLoadContext) -> Result<(), Error> {
        self.layers.push(LayerParams {
            layer_type: LayerType::Heights,
            offset: 0,
            tile_count: self.nodes.len(),
            tile_resolution: self.heights_resolution as u32,
            border_size: 0,
            format: TextureFormat::F32,
            tile_bytes: 4 * self.heights_resolution as usize * self.heights_resolution as usize,
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
        /// Takes a f64 in the range [0, 1] an converts it to a u8 in the sRGB color space.
        fn linear_to_srgb(linear: f64) -> u8 {
            let srgb = if linear <= 0.00313066844250063 {
                linear * 12.92
            } else {
                1.055 * linear.powf(0.41666666666) - 0.055
            };
            (srgb * 255.0) as u8
        }
        fn srgb_to_linear(srgb: u8) -> f64 {
            let srgb = (srgb as f64) * (1.0 / 255.0);
            if srgb <= 0.0404482362771082 {
                srgb / 12.92
            } else {
                ((srgb + 0.055) / 1.055).powf(2.4)
            }
        }

        assert!(self.skirt >= 2);
        let colormap_resolution = self.heightmap_resolution - 5;
        self.layers.push(LayerParams {
            layer_type: LayerType::Colors,
            offset: self.bytes_written,
            tile_count: self.heightmaps.as_ref().unwrap().len(),
            tile_resolution: colormap_resolution as u32,
            border_size: self.skirt as u32 - 2,
            format: TextureFormat::SRGBA,
            tile_bytes: 4 * colormap_resolution as usize * colormap_resolution as usize,
        });
        let rock = self.materials.get_average_albedo(0);
        let grass = self.materials.get_average_albedo(1);
        context.increment_level(
            "Generating colormaps... ",
            self.heightmaps.as_ref().unwrap().len(),
        );

        let reproject = ReprojectedRasterDef {
            name: format!("{}treecover", self.directory_name),
            heights: self.heightmaps.as_ref().unwrap(),
            system: &self.system,
            nodes: &self.nodes,
            skirt: self.skirt,
            raster: RasterSource::RasterCacheU8 {
                cache: Rc::new(RefCell::new(RasterCache::new(
                    Box::new(LandCoverKind::TreeCover),
                    256,
                ))),
                default: 0.0,
                radius2: Some(500_000.0 * 500_000.0),
            },
        };
        let treecover = ReprojectedRaster::from_raster(reproject, context)?;

        let mut colormaps: Vec<Vec<u8>> = Vec::new();
        for i in 0..self.heightmaps.as_ref().unwrap().len() {
            context.set_progress(i as u64);
            self.nodes[i].tile_indices[LayerType::Colors.index()] = Some(i as u32);

            let mut colormap = Vec::new();
            let heights = self.heightmaps.as_ref().unwrap();
            let spacing =
                self.nodes[i].side_length / (self.heightmap_resolution - 2 * self.skirt) as f32;
            let use_blue_marble = spacing >= self.bluemarble.spacing() as f32;
            for y in 2..(2 + colormap_resolution) {
                for x in 2..(2 + colormap_resolution) {
                    let world = self.world_position(x as i32, y as i32, self.nodes[i].bounds);
                    let h00 = heights.get(i, x, y, 0);
                    let h01 = heights.get(i, x, y + 1, 0);
                    let h10 = heights.get(i, x + 1, y, 0);
                    let h11 = heights.get(i, x + 1, y + 1, 0);
                    let h = (h00 + h01 + h10 + h11) as f64 * 0.25;
                    let lla = self.system.world_to_lla(Vector3::new(world.x, h, world.y));
                    let (lat, long) = (lla.x.to_degrees(), lla.y.to_degrees());

                    let normal =
                        Vector3::new(h10 + h11 - h00 - h01, 2.0 * spacing, h01 + h11 - h00 - h10)
                            .normalize();
                    let light = (normal.dot(self.sun_direction).max(0.0) * 255.0) as u8;

                    let color = if use_blue_marble {
                        let r = self.bluemarble.interpolate(lat, long, 0) as u8;
                        let g = self.bluemarble.interpolate(lat, long, 1) as u8;
                        let b = self.bluemarble.interpolate(lat, long, 2) as u8;
                        [r, g, b, light]
                    } else {
                        if normal.y > 0.9 {
                            let t = 1.0 - 0.4 * treecover.get(i, x, y, 0) / 100.0;
                            let r = (grass[0] as f32 * t) as u8;
                            let g = (grass[1] as f32 * t) as u8;
                            let b = (grass[2] as f32 * t) as u8;
                            [r, g, b, light]
                        } else {
                            [rock[0], rock[1], rock[2], light]
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
                let pc = &colormaps[parent.0.index()];
                for y in (0..resolution).step_by(2) {
                    for x in (0..resolution).step_by(2) {
                        for i in 0..3 {
                            let p =
                                pc[i + ((x / 2 + offset.x) + (y / 2 + offset.y) * resolution) * 4];
                            let p = srgb_to_linear(p);
                            let c00 = srgb_to_linear(colormap[i + (x + y * resolution) * 4]);
                            let c10 = srgb_to_linear(colormap[i + ((x + 1) + y * resolution) * 4]);
                            let c01 = srgb_to_linear(colormap[i + (x + (y + 1) * resolution) * 4]);
                            let c11 =
                                srgb_to_linear(colormap[i + ((x + 1) + (y + 1) * resolution) * 4]);

                            let shift = (p - (c00 + c01 + c10 + c11) * 0.25) * 0.5;

                            colormap[i + (x + y * resolution) * 4] =
                                linear_to_srgb((c00 + shift).max(0.0).min(1.0));
                            colormap[i + ((x + 1) + y * resolution) * 4] =
                                linear_to_srgb((c10 + shift).max(0.0).min(1.0));
                            colormap[i + (x + (y + 1) * resolution) * 4] =
                                linear_to_srgb((c01 + shift).max(0.0).min(1.0));
                            colormap[i + ((x + 1) + (y + 1) * resolution) * 4] =
                                linear_to_srgb((c11 + shift).max(0.0).min(1.0));
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
    fn generate_watermasks(&mut self, context: &mut AssetLoadContext) -> Result<(), Error> {
        assert!(self.skirt >= 2);
        let watermap_resolution = self.heightmap_resolution - 5;
        self.layers.push(LayerParams {
            layer_type: LayerType::Water,
            offset: self.bytes_written,
            tile_count: self.heightmaps.as_ref().unwrap().len(),
            tile_resolution: watermap_resolution as u32,
            border_size: self.skirt as u32 - 2,
            format: TextureFormat::RGBA8,
            tile_bytes: 4 * watermap_resolution as usize * watermap_resolution as usize,
        });
        context.increment_level(
            "Generating water masks... ",
            self.heightmaps.as_ref().unwrap().len(),
        );
        let mut watermask_cache = RasterCache::new(Box::new(LandCoverKind::WaterMask), 256);
        for i in 0..self.heightmaps.as_ref().unwrap().len() {
            context.set_progress(i as u64);
            self.nodes[i].tile_indices[LayerType::Water.index()] = Some(i as u32);

            let heights = &self.heightmaps.as_ref().unwrap();
            for y in 2..(2 + watermap_resolution) {
                for x in 2..(2 + watermap_resolution) {
                    let world = self.world_position(x as i32, y as i32, self.nodes[i].bounds);
                    let h00 = heights.get(i, x, y, 0);
                    let h01 = heights.get(i, x, y + 1, 0);
                    let h10 = heights.get(i, x + 1, y, 0);
                    let h11 = heights.get(i, x + 1, y + 1, 0);
                    let h = (h00 + h01 + h10 + h11) as f64 * 0.25;
                    let lla = self.system.world_to_lla(Vector3::new(world.x, h, world.y));

                    let spacing = self.nodes[i].side_length
                        / (self.heightmap_resolution - 2 * self.skirt) as f32;
                    let use_global_watermask = spacing >= self.global_watermask.spacing() as f32;
                    let w = if use_global_watermask {
                        self.global_watermask
                            .interpolate(lla.x.to_degrees(), lla.y.to_degrees(), 0)
                    } else {
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
                            .unwrap_or(w * 255.0)
                    };

                    self.writer.write_u8(w as u8)?;
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
        _context: &mut AssetLoadContext,
    ) -> Result<TextureDescriptor, Error> {
        let resolution = 8 * (self.heightmap_resolution - 1 - 2 * self.skirt) as usize;
        let descriptor = TextureDescriptor {
            offset: self.bytes_written,
            resolution: resolution as u32,
            format: TextureFormat::SRGBA,
            bytes: resolution * resolution * 4,
        };

        for y in 0..resolution {
            for x in 0..resolution {
                let fx = 2.0 * (x as f64 + 0.5) / resolution as f64 - 1.0;
                let fy = 2.0 * (y as f64 + 0.5) / resolution as f64 - 1.0;
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
                let r = self.bluemarble.interpolate(lat, long, 0) as u8;
                let g = self.bluemarble.interpolate(lat, long, 1) as u8;
                let b = self.bluemarble.interpolate(lat, long, 2) as u8;
                let aa = self.global_watermask.interpolate(lat, long, 0) as u8;

                self.writer.write_u8(r)?;
                self.writer.write_u8(g)?;
                self.writer.write_u8(b)?;
                self.writer.write_u8(aa)?;
                self.bytes_written += 4;
            }
        }
        Ok(descriptor)
    }
}
