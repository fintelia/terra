
extern crate piston_window;
extern crate vecmath;
extern crate camera_controllers;
#[macro_use]
extern crate gfx;
extern crate gfx_core;
extern crate gfx_device_gl;
extern crate sdl2_window;
extern crate shader_version;

mod clipmap;

pub use clipmap::Vertex;
pub use clipmap::pipe;
pub use clipmap::Terrain;

//----------------------------------------
// Cube associated data


//----------------------------------------


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
