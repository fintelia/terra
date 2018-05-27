extern crate camera_controllers;
extern crate cgmath;
extern crate collision;
extern crate fps_counter;
extern crate gfx;
extern crate gfx_smaa;
extern crate gfx_text;
extern crate piston_window;
extern crate terra;
extern crate vecmath;

extern crate lightbox;

use std::time::Instant;

use camera_controllers::{Camera, FirstPerson, FirstPersonSettings};
use cgmath::*;
use collision::Frustum;
use fps_counter::FPSCounter;
use piston_window::*;
use vecmath::vec3_dot;

use terra::{GridSpacing, QuadTreeBuilder, TextureQuality, VertexQuality};

fn compute_projection_matrix(w: &PistonWindow) -> Matrix4<f32> {
    let draw_size = w.window.draw_size();
    PerspectiveFov {
        fovy: Rad((90.0f32 * 9.0 / 16.0).to_radians()),
        near: 10.0,
        far: 50000000.0,
        aspect: (draw_size.width as f32) / (draw_size.height as f32),
    }.into()
}

fn compute_view_matrix(c: Camera) -> Matrix4<f32> {
    let p = c.position;
    let r = c.right;
    let u = c.up;
    let f = c.forward;

    #[cfg_attr(rustfmt, rustfmt_skip)]
    Matrix4::new(
        r[0], u[0], f[0], 0.0,
        r[1], u[1], f[1], 0.0,
        r[2], u[2], f[2], 0.0,
        -vec3_dot(r, p), -vec3_dot(u, p), -vec3_dot(f, p), 1.0,
    )
}

fn main() {
    let mut window: PistonWindow = PistonWindow::new(
        OpenGL::V3_3,
        0,
        WindowSettings::new("terra preview", [1920, 1080])
            .exit_on_esc(true)
            .opengl(OpenGL::V3_3)
            .vsync(false)
            // .srgb(false)
            .fullscreen(true)
            .build()
            .unwrap(),
    );
    window.set_capture_cursor(true);
    window.set_max_fps(240);
    window.set_ups(240);

    // let mut lightbox = lightbox::Lightbox::new(1024, 1024, window.factory.clone()).unwrap();

    let mut smaa_target =
        gfx_smaa::SmaaTarget::new(&mut window.factory, window.output_color.clone(), 1920, 1080)
            .unwrap();

    let mut terrain = QuadTreeBuilder::new(window.factory.clone(), &mut window.encoder)
        .latitude(42)
        .longitude(-73)
        .vertex_quality(VertexQuality::High)
        .texture_quality(TextureQuality::High)
        .grid_spacing(GridSpacing::TwoMeters)
        .build(&smaa_target.output_color(), &smaa_target.output_stencil())
        .unwrap();

    let mut first_person =
        FirstPerson::new([0.0, 1000.0, 0.0], FirstPersonSettings::keyboard_wasd());
    first_person.settings.speed_vertical = 5000.0;
    first_person.settings.speed_horizontal = 5000.0;

    let mut projection_matrix = compute_projection_matrix(&window);
    let mut detached_camera = false;
    let mut camera_position = Point3::new(0.0, 0.0, 0.0);
    let mut camera_frustum = None;

    let mut text = gfx_text::new(window.factory.clone())
        .with_size(12)
        .build()
        .unwrap();

    let mut fps_counter = FPSCounter::new();
    let mut last_frame = Instant::now();
    while let Some(e) = window.next() {
        if let Some(_) = e.resize_args() {
            projection_matrix = compute_projection_matrix(&window);
        }
        if let Some(Button::Keyboard(key)) = e.press_args() {
            if key == Key::Tab {
                detached_camera = !detached_camera;
            }
        }

        first_person.event(&e);
        e.update(|_args| {
            if !detached_camera {
                let center_distance: f32 = first_person.position[0] * first_person.position[0]
                    + first_person.position[2] * first_person.position[2];
                let center_distance = center_distance.sqrt();

                if center_distance > 3000.0 {
                    first_person.position[0] =
                        first_person.position[0] / (center_distance / 3000.0);
                    first_person.position[2] =
                        first_person.position[2] / (center_distance / 3000.0);
                }
                camera_position = Point3::new(
                    first_person.position[0],
                    first_person.position[1],
                    first_person.position[2],
                );
                camera_frustum = Frustum::from_matrix4(
                    projection_matrix * compute_view_matrix(first_person.camera(0.0)),
                );
                assert!(camera_frustum.is_some());
            }
            first_person.settings.speed_vertical =
                (5.0 * first_person.position[1] as f32).max(100.0f32);
        });

        window.draw_3d(&e, |window| {
            let now = Instant::now();
            let dt = (now - last_frame).as_secs() as f32
                + (now - last_frame).subsec_nanos() as f32 / 1000_000_000.0;
            last_frame = now;

            window
                .encoder
                .clear_depth(&smaa_target.output_stencil(), 1.0);
            window
                .encoder
                .clear(&smaa_target.output_color(), [0.3, 0.3, 0.3, 1.0]);

            let view_matrix = compute_view_matrix(first_person.camera(0.0));
            terrain.update(
                projection_matrix * view_matrix,
                camera_position,
                camera_frustum,
                &mut window.encoder,
                dt,
            );
            terrain.render(&mut window.encoder).unwrap();
            terrain.render_sky(&mut window.encoder);

            let text_color = [0.0, 1.0, 1.0, 1.0];
            let fps = fps_counter.tick();
            text.add(&format!("FPS: {}", fps), [5, 5], text_color);
            text.add(
                &format!("Frame time: {:.1}", 1000.0 / fps as f32),
                [5, 17],
                text_color,
            );
            text.draw(&mut window.encoder, &smaa_target.output_color())
                .unwrap();

            smaa_target.resolve(&mut window.encoder);
        });
    }
}
