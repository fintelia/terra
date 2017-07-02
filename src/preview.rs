
extern crate camera_controllers;
extern crate gfx;
extern crate terra;
extern crate piston_window;
extern crate sdl2_window;
extern crate vecmath;
extern crate imagefmt;

use piston_window::*;
use sdl2_window::Sdl2Window;
use camera_controllers::{FirstPersonSettings, FirstPerson, CameraPerspective,
                         model_view_projection};

use std::fs::File;
use std::io::BufReader;

use terra::terrain::Terrain;
use terra::terrain::dem::Dem;

fn main() {
    let opengl = OpenGL::V3_2;

    let mut window: PistonWindow<Sdl2Window> = WindowSettings::new("terra preview", [640, 480])
        .exit_on_esc(true)
        .samples(4)
        .opengl(opengl)
        .build()
        .unwrap();
    window.set_capture_cursor(true);

    let file = File::open("../assets/USGS_NED_1_n62w144_GridFloat.zip").unwrap();
    let reader = &mut BufReader::new(file);
    let mut terrain = Terrain::new(Dem::from_gridfloat_zip(reader),
                                   window.factory.clone(),
                                   window.output_color.clone(),
                                   window.output_stencil.clone());

    terrain.generate_textures(&mut window.encoder);

    let get_projection = |w: &PistonWindow<Sdl2Window>| {
        let draw_size = w.window.draw_size();
        CameraPerspective {
                fov: 90.0,
                near_clip: 10.0,
                far_clip: 100000.0,
                aspect_ratio: (draw_size.width as f32) / (draw_size.height as f32),
            }
            .projection()
    };

    let model = vecmath::mat4_id();
    let mut projection = get_projection(&window);
    let mut first_person = FirstPerson::new([10000.0, 5500.0, 10000.0],
                                            FirstPersonSettings::keyboard_wasd());
    first_person.settings.speed_vertical = 500.0;
    first_person.settings.speed_horizontal = 5000.0;

    while let Some(e) = window.next() {
        first_person.event(&e);

        window.draw_3d(&e, |window| {
            let args = e.render_args().unwrap();

            window
                .encoder
                .clear(&window.output_color, [0.3, 0.3, 0.3, 1.0]);
            window.encoder.clear_depth(&window.output_stencil, 1.0);

            let camera = first_person.camera(args.ext_dt);
            terrain.update(model_view_projection(model, camera.orthogonal(), projection),
                           [camera.position[0], camera.position[2]]);
            terrain.render(&mut window.encoder);
        });

        if let Some(_) = e.resize_args() {
            projection = get_projection(&window);
        }
    }
}
