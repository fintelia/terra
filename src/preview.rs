
extern crate camera_controllers;
extern crate gfx;
extern crate gfx_device_gl;
extern crate gfx_terrain;
extern crate piston_window;
extern crate sdl2_window;
extern crate vecmath;

use gfx_terrain::Terrain;

fn main() {
    use piston_window::*;
    use sdl2_window::Sdl2Window;
    use camera_controllers::{
        FirstPersonSettings,
        FirstPerson,
        CameraPerspective,
        model_view_projection
    };

    let opengl = OpenGL::V3_2;

    let mut window: PistonWindow<Sdl2Window> =
        WindowSettings::new("gfx_terrain preview", [640, 480])
        .exit_on_esc(true)
        .samples(4)
        .opengl(opengl)
        .build()
        .unwrap();
    window.set_capture_cursor(true);

    let mut terrain = Terrain::new(&mut window);
    
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
