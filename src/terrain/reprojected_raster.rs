use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use cache::{AssetLoadContext, MMappedAsset};
use cgmath::*;
use coordinates::{CoordinateSystem, PLANET_RADIUS};
use failure::Error;
use memmap::Mmap;
use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;
use terrain::heightmap::Heightmap;
use terrain::quadtree::Node;
use terrain::quadtree::node;
use terrain::raster::{GlobalRaster, RasterCache};
use utils::math::BoundingBox;

#[derive(Serialize, Deserialize)]
enum DataType {
    F32,
    U8,
}

fn world_position(
    x: i32,
    y: i32,
    bounds: BoundingBox,
    skirt: u16,
    resolution: u16,
) -> Vector2<f64> {
    let fx = (x - skirt as i32) as f32 / (resolution - 1 - 2 * skirt) as f32;
    let fy = (y - skirt as i32) as f32 / (resolution - 1 - 2 * skirt) as f32;

    Vector2::new(
        (bounds.min.x + (bounds.max.x - bounds.min.x) * fx) as f64,
        (bounds.min.z + (bounds.max.z - bounds.min.z) * fy) as f64,
    )
}

pub(crate) struct ReprojectedDemDef<'a> {
    pub name: String,
    pub dem_cache: Rc<RefCell<RasterCache<f32>>>,
    pub system: &'a CoordinateSystem,
    pub nodes: &'a Vec<Node>,
    pub random: &'a Heightmap<f32>,

    pub skirt: u16,
    pub max_dem_level: u8,
    pub max_texture_level: u8,
    pub resolution: u16,
}
impl<'a> MMappedAsset for ReprojectedDemDef<'a> {
    type Header = ReprojectedRasterHeader;

    fn filename(&self) -> String {
        self.name.clone()
    }
    fn generate<W: Write>(
        &self,
        context: &mut AssetLoadContext,
        mut writer: W,
    ) -> Result<Self::Header, Error> {
        let tiles = self.nodes
            .iter()
            .filter(|n| n.level <= self.max_texture_level)
            .count();

        let mut heightmaps: Vec<Heightmap<f32>> = Vec::with_capacity(tiles);

        context.increment_level("Reprojecting DEMs... ", tiles);
        for i in 0..tiles {
            context.set_progress(i as u64);

            assert!(self.nodes[i].level <= self.max_texture_level);
            if self.nodes[i].level > self.max_dem_level {
                let mut heights =
                    Vec::with_capacity(self.resolution as usize * self.resolution as usize);
                let offset = node::OFFSETS[self.nodes[i].parent.as_ref().unwrap().1 as usize];
                let offset = Point2::new(
                    self.skirt / 2 + offset.x as u16 * (self.resolution / 2 - self.skirt),
                    self.skirt / 2 + offset.y as u16 * (self.resolution / 2 - self.skirt),
                );

                let layer_scale =
                    self.nodes[i].size / (self.resolution - 2 * self.skirt - 1) as i32;
                let layer_origin = Vector2::new(
                    (self.nodes[i].center.x - self.nodes[i].size / 2) / layer_scale,
                    (self.nodes[i].center.y - self.nodes[i].size / 2) / layer_scale,
                );

                let ph = &heightmaps[self.nodes[i].parent.as_ref().unwrap().0.index()];
                for y in 0..self.resolution {
                    for x in 0..self.resolution {
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

                let mut heightmap = Heightmap::new(heights, self.resolution, self.resolution);

                // Compute noise.
                let mut noise =
                    Vec::with_capacity(self.resolution as usize * self.resolution as usize);
                let noise_scale =
                    self.nodes[i].side_length / (self.resolution - 1 - 2 * self.skirt) as f32;
                let slope_scale = 0.5 * (self.resolution - 1) as f32 / self.nodes[i].side_length;
                for y in 0..self.resolution {
                    for x in 0..self.resolution {
                        if (x % 2 != 0 || y % 2 != 0) && x > 0 && y > 0 && x < self.resolution - 1
                            && y < self.resolution - 1
                            && heightmap.at(x, y) > 0.0
                        {
                            let slope_x = heightmap.at(x + 1, y) - heightmap.at(x - 1, y);
                            let slope_y = heightmap.at(x, y + 1) - heightmap.at(x, y - 1);
                            let slope =
                                (slope_x * slope_x + slope_y * slope_y).sqrt() * slope_scale;

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
                for y in 0..self.resolution {
                    for x in 0..self.resolution {
                        heightmap.raise(
                            x,
                            y,
                            noise[x as usize + y as usize * self.resolution as usize],
                        );
                    }
                }
                heightmaps.push(heightmap);
            } else {
                let bounds = self.nodes[i].bounds;
                let mut heights =
                    Vec::with_capacity(self.resolution as usize * self.resolution as usize);
                for y in 0..(self.resolution as i32) {
                    for x in 0..(self.resolution as i32) {
                        let world = world_position(x, y, bounds, self.skirt, self.resolution);
                        let mut world3 = Vector3::new(
                            world.x,
                            PLANET_RADIUS
                                * ((1.0 - world.magnitude2() / PLANET_RADIUS).max(0.25).sqrt()
                                    - 1.0),
                            world.y,
                        );
                        for i in 0..5 {
                            world3.x = world.x;
                            world3.z = world.y;
                            let mut lla = self.system.world_to_lla(world3);
                            lla.z = if i >= 3 && world.magnitude2() < 250000.0 * 250000.0 {
                                self.dem_cache
                                    .borrow_mut()
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
                heightmaps.push(Heightmap::new(heights, self.resolution, self.resolution));
            }
        }
        context.reset("Saving reprojecting DEMs... ", heightmaps.len());
        for heightmap in heightmaps {
            for height in heightmap.heights {
                writer.write_f32::<LittleEndian>(height)?;
            }
        }
        context.decrement_level();

        Ok(ReprojectedRasterHeader {
            resolution: self.resolution,
            bands: 1,
            tiles,
            datatype: DataType::F32,
        })
    }
}

pub(crate) enum RasterSource<'a> {
    GlobalRasterU8(&'a GlobalRaster<u8>),
    RasterCacheU8 {
        cache: Rc<RefCell<RasterCache<u8>>>,
        default: f64,
        radius2: Option<f64>,
    },
}
pub(crate) struct ReprojectedRasterDef<'a> {
    pub name: String,
    pub heights: &'a ReprojectedRaster,

    pub system: &'a CoordinateSystem,
    pub nodes: &'a Vec<Node>,
    pub skirt: u16,

    pub raster: RasterSource<'a>,
}
impl<'a> MMappedAsset for ReprojectedRasterDef<'a> {
    type Header = ReprojectedRasterHeader;

    fn filename(&self) -> String {
        self.name.clone()
    }
    fn generate<W: Write>(
        &self,
        context: &mut AssetLoadContext,
        mut writer: W,
    ) -> Result<Self::Header, Error> {
        let (datatype, bands) = match self.raster {
            RasterSource::GlobalRasterU8(gr) => (DataType::U8, gr.bands as u16),
            RasterSource::RasterCacheU8 { .. } => (DataType::U8, 1),
        };

        assert_eq!(self.heights.header.bands, 1);
        context.set_progress_and_total(0, self.heights.header.tiles);
        for i in 0..self.heights.header.tiles {
            for y in 0..(self.heights.header.resolution - 1) {
                for x in 0..(self.heights.header.resolution - 1) {
                    let world = world_position(
                        x as i32,
                        y as i32,
                        self.nodes[i].bounds,
                        self.skirt,
                        self.heights.header.resolution,
                    );
                    let h00 = self.heights.get(i, x, y, 0);
                    let h01 = self.heights.get(i, x, y + 1, 0);
                    let h10 = self.heights.get(i, x + 1, y, 0);
                    let h11 = self.heights.get(i, x + 1, y + 1, 0);
                    let h = (h00 + h01 + h10 + h11) as f64 * 0.25;
                    let lla = self.system.world_to_lla(Vector3::new(world.x, h, world.y));
                    let (lat, long) = (lla.x.to_degrees(), lla.y.to_degrees());

                    for band in 0..bands {
                        let v = match self.raster {
                            RasterSource::GlobalRasterU8(gr) => {
                                gr.interpolate(lat, long, band as usize)
                            }
                            RasterSource::RasterCacheU8 {
                                ref cache,
                                ref radius2,
                                default,
                            } => {
                                if radius2.is_none() || world.magnitude2() < radius2.unwrap() {
                                    cache
                                        .borrow_mut()
                                        .interpolate(context, lat, long)
                                        .unwrap_or(default)
                                } else {
                                    default
                                }
                            }
                        };

                        match datatype {
                            DataType::F32 => writer.write_f32::<LittleEndian>(v as f32),
                            DataType::U8 => writer.write_u8(v as u8),
                        }?;
                    }
                }
            }
            context.set_progress(i + 1);
        }

        Ok(ReprojectedRasterHeader {
            resolution: self.heights.header.resolution - 1,
            tiles: self.heights.header.tiles,
            datatype,
            bands,
        })
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ReprojectedRasterHeader {
    resolution: u16,
    bands: u16,
    tiles: usize,
    datatype: DataType,
}

pub(crate) struct ReprojectedRaster {
    header: ReprojectedRasterHeader,
    data: Mmap,
}
impl ReprojectedRaster {
    pub fn from_dem<'a>(
        def: ReprojectedDemDef<'a>,
        context: &mut AssetLoadContext,
    ) -> Result<Self, Error> {
        let (header, data) = def.load(context)?;
        Ok(Self { data, header })
    }

    pub fn from_raster<'a>(
        def: ReprojectedRasterDef<'a>,
        context: &mut AssetLoadContext,
    ) -> Result<Self, Error> {
        let (header, data) = def.load(context)?;
        Ok(Self { data, header })
    }

    pub fn get(&self, tile: usize, x: u16, y: u16, band: u16) -> f32 {
        assert!(band < self.header.bands);
        assert!(x < self.header.resolution);
        assert!(y < self.header.resolution);

        let resolution = self.header.resolution as usize;
        let index = band as usize
            + (x as usize + (y as usize + tile * resolution) * resolution)
                * self.header.bands as usize;
        match self.header.datatype {
            DataType::F32 => LittleEndian::read_f32(&unsafe { self.data.as_slice() }[index * 4..]),
            DataType::U8 => unsafe { f32::from(self.data.as_slice()[index]) },
        }
    }

    pub fn len(&self) -> usize {
        self.header.tiles
    }
}
