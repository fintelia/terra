extern crate camera_controllers;
extern crate gfx;
extern crate piston_window;
extern crate terra;
extern crate vecmath;

use piston_window::*;
use camera_controllers::{FirstPersonSettings, FirstPerson, CameraPerspective,
                         model_view_projection};

use terra::{Clipmap, DemSource, TerrainFileParams, MaterialSet};
use terra::cache::GeneratedAsset;

fn main() {
    let terrain_file = TerrainFileParams {
        latitude: 44,
        longitude: -74,
        source: DemSource::Usgs30m,
    }.load()
        .unwrap();

    let mut window: PistonWindow = WindowSettings::new("terra preview", [640, 480])
        .exit_on_esc(true)
        .opengl(OpenGL::V3_3)
        .samples(1)
        .build()
        .unwrap();
    window.set_capture_cursor(true);

    let materials = MaterialSet::load(&mut window.factory, &mut window.encoder).unwrap();
    window.encoder.flush(&mut window.device);

    let mut terrain = Clipmap::new(
        terrain_file,
        materials,
        window.factory.clone(),
        &mut window.encoder,
        &window.output_color,
        &window.output_stencil,
    );

    let get_projection = |w: &PistonWindow| {
        let draw_size = w.window.draw_size();
        CameraPerspective {
            fov: 90.0,
            near_clip: 0.5,
            far_clip: 100000.0,
            aspect_ratio: (draw_size.width as f32) / (draw_size.height as f32),
        }.projection()
    };

    let mut projection = get_projection(&window);
    let mut first_person = FirstPerson::new([0.0, 0.0, 0.0], FirstPersonSettings::keyboard_wasd());
    first_person.settings.speed_vertical = 50.0;
    first_person.settings.speed_horizontal = 2.0;

    while let Some(e) = window.next() {
        first_person.event(&e);

        if let Some(_) = e.resize_args() {
            projection = get_projection(&window);
        }

        window.draw_3d(&e, |window| {
            let args = e.render_args().unwrap();

            window.encoder.clear(
                &window.output_color,
                [0.3, 0.3, 0.3, 1.0],
            );
            window.encoder.clear_depth(&window.output_stencil, 1.0);

            let mut camera = first_person.camera(args.ext_dt);
            if let Some(h) = terrain.get_approximate_height(
                [camera.position[0], camera.position[2]],
            )
            {
                camera.position[1] += h + 2.0;
            }
            terrain.update(
                model_view_projection(vecmath::mat4_id(), camera.orthogonal(), projection),
                camera.position,
                &mut window.encoder,
            );
            terrain.render(&mut window.encoder);
        });
    }
}
