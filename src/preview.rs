
extern crate camera_controllers;
extern crate gfx;
extern crate gfx_terrain;
extern crate piston_window;
extern crate sdl2_window;
extern crate vecmath;
extern crate imagefmt;

use piston_window::*;
use sdl2_window::Sdl2Window;
use camera_controllers::{
    FirstPersonSettings,
    FirstPerson,
    CameraPerspective,
    model_view_projection
};

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use gfx_terrain::Terrain;
use gfx_terrain::Heightmap;

fn load_heightmap<P: AsRef<Path>>(path: P) -> (u16, u16, Vec<u16>) {
    let file = File::open(path).unwrap();
    let reader = &mut BufReader::new(file);
    let pic = imagefmt::png::read(reader, imagefmt::ColFmt::Y).unwrap();
    let (w, h) = (pic.w, pic.h);
    let mut heights = pic.buf;

    let min:u16 = {
        *heights.iter().filter(|h| **h > 0).min().unwrap()
    };
    for h in &mut heights {
        *h = h.saturating_sub(min);
    }
    (w as u16, h as u16, heights)
}

fn main() {
    let opengl = OpenGL::V3_2;

    let mut window: PistonWindow<Sdl2Window> =
        WindowSettings::new("gfx_terrain preview", [640, 480])
        .exit_on_esc(true)
        .samples(4)
        .opengl(opengl)
        .build()
        .unwrap();
    window.set_capture_cursor(true);

    let (w, h, heights) =
        load_heightmap("../assets/mtsthelens_before.png");
    let heightmap = Heightmap::new(heights, w as u16, h as u16);
    let mut terrain = Terrain::new(heightmap,
                                   window.factory.clone(),
                                   window.output_color.clone(),
                                   window.output_stencil.clone());

    terrain.generate_textures(&mut window.encoder);

    let get_projection = |w: &PistonWindow<Sdl2Window>| {
        let draw_size = w.window.draw_size();
        CameraPerspective {
            fov: 90.0, near_clip: 0.1, far_clip: 1000.0,
            aspect_ratio: (draw_size.width as f32) / (draw_size.height as f32)
        }.projection()
    };

    let model = vecmath::mat4_id();
    let mut projection = get_projection(&window);
    let mut first_person = FirstPerson::new(
        [0.5, 0.5, 4.0],
        FirstPersonSettings::keyboard_wasd()
    );

    while let Some(e) = window.next() {
        first_person.event(&e);

        window.draw_3d(&e, |window| {
            let args = e.render_args().unwrap();

            window.encoder.clear(&window.output_color, [0.3, 0.3, 0.3, 1.0]);
            window.encoder.clear_depth(&window.output_stencil, 1.0);

            terrain.update(model_view_projection(
                model,
                first_person.camera(args.ext_dt).orthogonal(),
                projection
            ));
            terrain.render(&mut window.encoder);
        });

        if let Some(_) = e.resize_args() {
            projection = get_projection(&window);
        }
    }
}
