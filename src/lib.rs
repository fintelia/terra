
extern crate vecmath;
#[macro_use]
extern crate gfx;
extern crate gfx_core;
extern crate shader_version;
extern crate rand;

#[cfg(feature = "download")]
extern crate byteorder;
#[cfg(feature = "download")]
extern crate curl;
#[cfg(feature = "download")]
extern crate zip;

pub mod terrain;
#[cfg(feature = "download")]
pub mod offline;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
