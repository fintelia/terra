extern crate bincode;
extern crate cgmath;
extern crate curl;
#[macro_use]
extern crate gfx;
extern crate gfx_core;
extern crate image;
#[macro_use]
extern crate lazy_static;
extern crate notify;
extern crate rand;
#[macro_use]
extern crate rshader;
extern crate safe_transmute;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate shader_version;
extern crate vecmath;
extern crate vec_map;
extern crate zip;

mod utils;

pub mod cache;
pub mod terrain;

pub use terrain::clipmap::Clipmap;
pub use terrain::dem::{DigitalElevationModel, DigitalElevationModelParams, DemSource};
pub use terrain::file::{TerrainFile, TerrainFileParams};
pub use terrain::material::MaterialSet;
pub use terrain::quadtree::QuadTree;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
