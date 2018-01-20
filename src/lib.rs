//! Terra is a large scale terrain rendering library built on top of gfx.

#![feature(iterator_step_by)]
#![feature(ord_max_min)]
#![feature(test)]
#![feature(unboxed_closures)]
#![feature(use_nested_groups)]
#![feature(try_from)]

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

mod utils;
mod runtime_texture;
mod coordinates;
mod generate;

mod cache;
mod ocean;
mod sky;
mod terrain;

pub use terrain::dem::DemSource;
pub use generate::{QuadTreeBuilder, TextureQuality, VertexQuality};
pub use terrain::quadtree::QuadTree;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
