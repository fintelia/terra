use crate::cache::{AssetLoadContext, MMappedAsset, WebAsset};
use crate::coordinates::CoordinateSystem;
use crate::terrain::dem::GlobalDem;
use crate::terrain::heightmap::Heightmap;
use crate::terrain::quadtree::VNode;
use crate::terrain::raster::{GlobalRaster, RasterCache};
use crate::utils::math::BoundingBox;
use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use cgmath::*;
use failure::Error;
use memmap::Mmap;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::convert::TryInto;
use std::io::Write;
use std::ops::{Deref, Index};
use std::rc::Rc;

#[derive(Copy, Clone, Serialize, Deserialize)]
pub(crate) enum DataType {
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
    pub dem_cache: Rc<RefCell<RasterCache<f32, Vec<f32>>>>,
    pub nodes: &'a Vec<VNode>,
    pub random: &'a Heightmap<f32>,
    pub global_dem: GlobalRaster<i16>,

    pub skirt: u16,
    pub max_dem_level: u8,
    pub max_texture_present_level: u8,
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
        let tiles =
            self.nodes.iter().filter(|n| n.level() <= self.max_texture_present_level).count();

        let global_dem = GlobalDem.load(context)?;
        let mut heightmaps: Vec<Heightmap<f32>> = Vec::with_capacity(tiles);

        context.increment_level("Reprojecting DEMs... ", tiles);
        for i in 0..tiles {
            context.set_progress(i as u64);

            assert!(self.nodes[i].level() <= self.max_texture_present_level);
            let mut heights =
                Vec::with_capacity(self.resolution as usize * self.resolution as usize);
            for y in 0..(self.resolution as i32) {
                for x in 0..(self.resolution as i32) {
                    let cspace = self.nodes[i].grid_position_cspace(x, y, self.skirt, self.resolution);
                    let sspace = CoordinateSystem::cspace_to_sspace(cspace);

                    // let world = world_position(x, y, bounds, self.skirt, self.resolution);
                    // let mut world3 = Vector3::new(
                    //     world.x,
                    //     PLANET_RADIUS
                    //         * ((1.0 - world.magnitude2() / PLANET_RADIUS).max(0.25).sqrt() - 1.0),
                    //     world.y,
                    // );
                    // for i in 0..5 {
                    //     world3.x = world.x;
                    //     world3.z = world.y;
                    //     let mut lla = self.system.world_to_lla(world3);
                    //     lla.z = if i >= 3 && world.magnitude2() < 2000000.0 * 2000000.0 {
                    //         if world.magnitude2() < 250000.0 * 250000.0 {
                    //             self.dem_cache
                    //                 .borrow_mut()
                    //                 .interpolate(context, lla.x.to_degrees(), lla.y.to_degrees(), 0)
                    //                 .unwrap_or(0.0) as f64
                    //         } else {
                    //             global_dem
                    //                 .interpolate(lla.x.to_degrees(), lla.y.to_degrees(), 0)
                    //                 .max(0.0)
                    //         }
                    //     } else {
                    //         0.0
                    //     };
                    //     world3 = self.system.lla_to_world(lla);
                    // }
                    // heights.push(world3.y as f32);
                    heights.push(0.0);
                }
            }
            heightmaps.push(Heightmap::new(heights, self.resolution, self.resolution));
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
            spacing: None,
        })
    }
}

pub(crate) enum RasterSource<T, C = Vec<T>, C2: Deref<Target = [T]> = Vec<T>>
where
    T: Into<f64> + Copy,
    C: Index<usize, Output = T>,
{
    #[allow(unused)]
    GlobalRaster { global: Box<dyn WebAsset<Type = GlobalRaster<T, C>>> },
    Hybrid {
        global: Box<dyn WebAsset<Type = GlobalRaster<T, C>>>,
        cache: Rc<RefCell<RasterCache<T, C2>>>,
    },
}
pub(crate) struct ReprojectedRasterDef<'a, T, C, C2>
where
    T: Into<f64> + Copy,
    C: Index<usize, Output = T>,
    C2: Deref<Target = [T]>,
{
    pub name: String,

    pub nodes: &'a [VNode],
    pub resolution: u16,
    pub skirt: u16,

    pub datatype: DataType,
    pub raster: RasterSource<T, C, C2>,
}
impl<'a, T, C, C2> MMappedAsset for ReprojectedRasterDef<'a, T, C, C2>
where
    T: Into<f64> + Copy,
    C: Index<usize, Output = T>,
    C2: Deref<Target = [T]>,
{
    type Header = ReprojectedRasterHeader;

    fn filename(&self) -> String {
        self.name.clone()
    }
    fn generate<W: Write>(
        &self,
        context: &mut AssetLoadContext,
        mut writer: W,
    ) -> Result<Self::Header, Error> {
        let (global_raster, global_spacing) = match self.raster {
            RasterSource::GlobalRaster { ref global } | RasterSource::Hybrid { ref global, .. } => {
                let global_raster = global.load(context)?;
                let spacing = global_raster.spacing() as f32;
                (Some(global_raster), Some(spacing))
            }
        };

        let bands = match self.raster {
            RasterSource::GlobalRaster { .. } => global_raster.as_ref().unwrap().bands,
            RasterSource::Hybrid { ref cache, .. } => {
                assert_eq!(global_raster.as_ref().unwrap().bands, cache.borrow().bands());
                cache.borrow().bands()
            }
        };

        context.set_progress_and_total(0, self.nodes.len());
        for (i, node) in self.nodes.iter().enumerate() {
            let spacing = node.aprox_side_length() / (self.resolution - 2 * self.skirt) as f32;

            for y in 0..(self.resolution as i32) {
                for x in 0..(self.resolution as i32) {
                    let cspace = node.cell_position_cspace(x, y, self.skirt, self.resolution);
                    let sspace = CoordinateSystem::cspace_to_sspace(cspace);
                    let polar = CoordinateSystem::sspace_to_polar(sspace);
                    let (lat, long) = (polar.x.to_degrees(), polar.y.to_degrees());

                    for band in 0..bands {
                        let v = match self.raster {
                            RasterSource::GlobalRaster { .. } => global_raster
                                .as_ref()
                                .unwrap()
                                .interpolate(lat, long, band as usize),
                            RasterSource::Hybrid { ref cache, .. } => {
                                if spacing >= *global_spacing.as_ref().unwrap() {
                                    global_raster.as_ref().unwrap().interpolate(lat, long, band)
                                } else {
                                    cache
                                        .borrow_mut()
                                        .interpolate(context, lat, long, band)
                                        .unwrap_or_else(|| {
                                            global_raster
                                                .as_ref()
                                                .unwrap()
                                                .interpolate(lat, long, band)
                                        })
                                }
                            }
                        };

                        match self.datatype {
                            DataType::F32 => writer.write_f32::<LittleEndian>(v as f32),
                            DataType::U8 => writer.write_u8(v as u8),
                        }?;
                    }
                }
            }
            context.set_progress(i + 1);
        }

        let spacing = match self.raster {
            RasterSource::GlobalRaster { .. } => Some(global_raster.unwrap().spacing()),
            RasterSource::Hybrid { ref cache, .. } => {
                cache.borrow().spacing().or(Some(global_raster.unwrap().spacing()))
            }
        };

        Ok(ReprojectedRasterHeader {
            resolution: self.resolution,
            tiles: self.nodes.len(),
            datatype: self.datatype,
            bands: bands.try_into().unwrap(),
            spacing,
        })
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ReprojectedRasterHeader {
    resolution: u16,
    bands: u16,
    tiles: usize,
    datatype: DataType,
    spacing: Option<f64>,
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
        Ok(Self { data: data.make_read_only()?, header })
    }

    pub fn from_raster<'a, T, C, C2>(
        def: ReprojectedRasterDef<'a, T, C, C2>,
        context: &mut AssetLoadContext,
    ) -> Result<Self, Error>
    where
        T: Into<f64> + Copy,
        C: Index<usize, Output = T>,
        C2: Deref<Target = [T]>,
    {
        let (header, data) = def.load(context)?;
        Ok(Self { data: data.make_read_only()?, header })
    }

    pub fn get(&self, tile: usize, x: u16, y: u16, band: u16) -> f32 {
        assert!(band < self.header.bands);
        assert!(x < self.header.resolution);
        assert!(y < self.header.resolution);

        let resolution = self.header.resolution as usize;
        let index = band as usize
            + (x as usize + (y as usize + tile * resolution) * resolution)
            * self.header.bands as usize;
        if index >= self.data.len() {
            eprintln!("ERROR: tile = {}/{}, x = {}, y = {}, band = {}, resolution = {}", tile, self.header.tiles, x, y, band, resolution);
        }
        match self.header.datatype {
            DataType::F32 => LittleEndian::read_f32(&self.data[index * 4..]),
            DataType::U8 => f32::from(self.data[index]),
        }
    }

    /// Returns the spacing of the source dataset, if known.
    pub fn spacing(&self) -> Option<f64> {
        self.header.spacing
    }

    pub fn tiles(&self) -> usize {
        self.header.tiles
    }
}
