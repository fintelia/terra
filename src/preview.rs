extern crate camera_controllers;
extern crate gfx;
extern crate piston_window;
extern crate terra;
extern crate vecmath;

use piston_window::*;
use camera_controllers::{FirstPersonSettings, FirstPerson, CameraPerspective,
                         model_view_projection};

use std::fs::File;
use std::io::BufReader;

use terra::{Clipmap, DigitalElevationModel, TerrainFile};

fn main() {
    let file = File::open("../assets/USGS_NED_1_n62w144_GridFloat.zip").unwrap_or_else(|_| {
        println!("Failed to open '../assets/USGS_NED_1_n62w144_GridFloat.zip'");
        println!("If necessary, it can be downloaded from: \
                  https://prd-tnm.s3.amazonaws.com/index.html? \
                  prefix=StagedProducts/Elevation/1/GridFloat/");
        panic!()
    });
    let dem = DigitalElevationModel::from_gridfloat_zip(&mut BufReader::new(file));
    let terrain_file = TerrainFile::from_digital_elevation_model(dem);

    let mut window: PistonWindow = WindowSettings::new("terra preview", [640, 480])
        .exit_on_esc(true)
        .opengl(OpenGL::V4_5)
        .samples(2)
        .build()
        .unwrap();
    window.set_capture_cursor(true);

    let mut terrain = Clipmap::new(terrain_file,
                                   window.factory.clone(),
                                   &mut window.encoder,
                                   &window.output_color,
                                   &window.output_stencil);

    let get_projection = |w: &PistonWindow| {
        let draw_size = w.window.draw_size();
        CameraPerspective {
                fov: 90.0,
                near_clip: 0.5,
                far_clip: 100000.0,
                aspect_ratio: (draw_size.width as f32) / (draw_size.height as f32),
            }
            .projection()
    };

    let mut projection = get_projection(&window);
    let mut first_person = FirstPerson::new([0.0, 0.0, 0.0], FirstPersonSettings::keyboard_wasd());
    first_person.settings.speed_vertical = 500.0;
    first_person.settings.speed_horizontal = 500.0;

    while let Some(e) = window.next() {
        first_person.event(&e);

        if let Some(_) = e.resize_args() {
            projection = get_projection(&window);
        }

        window.draw_3d(&e, |window| {
            let args = e.render_args().unwrap();

            window
                .encoder
                .clear(&window.output_color, [0.3, 0.3, 0.3, 1.0]);
            window.encoder.clear_depth(&window.output_stencil, 1.0);

            let mut camera = first_person.camera(args.ext_dt);
            if let Some(h) = terrain.get_approximate_height([camera.position[0],
                                                             camera.position[2]]) {
                camera.position[1] += h + 2.0;
            }
            terrain.update(model_view_projection(vecmath::mat4_id(),
                                                 camera.orthogonal(),
                                                 projection),
                           [camera.position[0], camera.position[2]]);
            terrain.render(&mut window.encoder);
        });
    }
}
