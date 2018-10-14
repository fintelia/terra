//! Terra is a large scale terrain rendering library built on top of gfx.

#![feature(custom_attribute)]
#![feature(nll)]
#![feature(non_ascii_idents)]
#![feature(slice_patterns)]
#![feature(stmt_expr_attributes)]
#![feature(test)]
#![feature(try_from)]
#![feature(unboxed_closures)]

#[cfg(test)]
#[macro_use]
extern crate approx;
extern crate astro;
extern crate bincode;
extern crate bit_vec;
extern crate byteorder;
extern crate cgmath;
extern crate collision;
extern crate coord_transforms;
extern crate curl;
extern crate dirs;
#[macro_use]
extern crate failure;
#[macro_use]
extern crate gfx;
extern crate gfx_core;
extern crate image;
#[macro_use]
extern crate lazy_static;
extern crate lightbox;
extern crate lru_cache;
extern crate memmap;
extern crate nalgebra;
extern crate notify;
extern crate num;
extern crate obj;
extern crate pbr;
extern crate rand;
#[macro_use]
extern crate rshader;
extern crate rustfft;
extern crate safe_transmute;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate shader_version;
extern crate test;
extern crate vec_map;
extern crate vecmath;
extern crate zip;

mod cache;
mod coordinates;
mod generate;
mod model;
mod ocean;
mod runtime_texture;
mod sky;
mod srgb;
mod terrain;
mod utils;

pub use generate::{GridSpacing, QuadTreeBuilder, TextureQuality, VertexQuality};
pub use terrain::dem::DemSource;
pub use terrain::quadtree::QuadTree;

use cache::{AssetLoadContext, GeneratedAsset};
use model::{TreeBillboardDef, TreeType};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

lazy_static! {
    static ref TREE_BILLBOARDS: Arc<Mutex<HashMap<TreeType, Vec<(u32, u32, Vec<u8>)>>>> =
        { Arc::new(Mutex::new(HashMap::new())) };
}

pub fn prepare_assets<R, F, C, D>(
    factory: F,
    encoder: &mut gfx::Encoder<R, C>,
    device: &mut D,
) -> Result<(), failure::Error>
where
    R: gfx::Resources,
    F: gfx::Factory<R> + 'static + Clone,
    C: gfx_core::command::Buffer<R>,
    D: gfx::Device<Resources = R, CommandBuffer = C>,
{
    let lightbox = lightbox::Lightbox::new(2048, 2048, factory).unwrap();

    let lightbox = Arc::new(Mutex::new(lightbox));
    let encoder = Arc::new(Mutex::new(encoder));
    let device = Arc::new(Mutex::new(device));

    TREE_BILLBOARDS.lock().unwrap().insert(
        TreeType::Birch,
        TreeBillboardDef::<R, F, C, D> {
            ty: TreeType::Birch,
            lightbox: lightbox.clone(),
            encoder: encoder.clone(),
            device: device.clone(),
        }.load(&mut AssetLoadContext::new())?,
    );

    TREE_BILLBOARDS.lock().unwrap().insert(
        TreeType::Beech,
        TreeBillboardDef::<R, F, C, D> {
            ty: TreeType::Beech,
            lightbox: lightbox.clone(),
            encoder: encoder.clone(),
            device: device.clone(),
        }.load(&mut AssetLoadContext::new())?,
    );

    TREE_BILLBOARDS.lock().unwrap().insert(
        TreeType::Pine,
        TreeBillboardDef::<R, F, C, D> {
            ty: TreeType::Pine,
            lightbox: lightbox.clone(),
            encoder: encoder.clone(),
            device: device.clone(),
        }.load(&mut AssetLoadContext::new())?,
    );

    Ok(())
}
