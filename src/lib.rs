
#[macro_use]
extern crate gfx;
extern crate gfx_core;
extern crate rand;
extern crate safe_transmute;
extern crate shader_version;
extern crate vecmath;
extern crate zip;

#[cfg(feature = "download")]
extern crate curl;

pub mod terrain;

pub use terrain::dem::DigitalElevationModel;
pub use terrain::clipmap::Clipmap;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
