
extern crate vecmath;
#[macro_use]
extern crate gfx;
extern crate gfx_core;
extern crate shader_version;

mod clipmap;

pub use clipmap::Terrain;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
