//! Terra is a large scale terrain rendering library built on top of gfx.

#![feature(custom_attribute)]
#![feature(iterator_step_by)]
#![feature(nll)]
#![feature(non_ascii_idents)]
#![feature(ord_max_min)]
#![feature(slice_patterns)]
#![feature(test)]
#![feature(try_from)]
#![feature(unboxed_closures)]

#[cfg(test)]
#[macro_use]
extern crate approx;
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
extern crate lru_cache;
extern crate memmap;
extern crate nalgebra;
extern crate notify;
extern crate num;
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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
