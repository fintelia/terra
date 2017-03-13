
extern crate vecmath;
#[macro_use]
extern crate gfx;
extern crate gfx_core;
extern crate shader_version;

mod vertex_buffer;
mod clipmap;
mod heightmap;

pub use clipmap::Terrain;
pub use heightmap::Heightmap;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
