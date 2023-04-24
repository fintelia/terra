use clap::{Parser, Subcommand};
use gilrs::{Axis, Button, Gilrs};
use planetcam::DualPlanetCam;
use std::time::Instant;
use winit::{
    dpi::PhysicalPosition,
    event::{self, ElementState, MouseButton},
    event_loop::{ControlFlow, EventLoop},
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, global = true, default_value = "8FH495PF+29")]
    plus: String,
    #[arg(long, global = true, default_value = "0")]
    heading: f64,
    #[arg(short, long, global = true, default_value = "200000")]
    elevation: f64,
    #[arg(long, global = true)]
    time: Option<String>,
    #[arg(long, global = true, default_value = "0.0")]
    timescale: f64,
    #[arg(long, global = true)]
    server: Option<String>,

    #[command(subcommand)]
    subcommand: Option<SubcommandArgs>,
}

#[derive(Subcommand, Debug)]
enum SubcommandArgs {
    #[cfg(feature = "generate")]
    Generate {
        path: std::path::PathBuf,
        #[arg(long)]
        download: bool,
    },
}

fn compute_projection_matrix(width: f32, height: f32) -> cgmath::Matrix4<f32> {
    let aspect = width / height;
    let f = 1.0 / (45.0f32.to_radians() / aspect).tan();
    let near = 0.1;

    #[cfg_attr(rustfmt, rustfmt_skip)]
    cgmath::Matrix4::new(
        f/aspect,  0.0,  0.0,   0.0,
        0.0,       f,    0.0,   0.0,
        0.0,       0.0,  0.0,  -1.0,
        0.0,       0.0,  near,  0.0)
}

fn make_depth_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
            label: None,
        })
        .create_view(&Default::default())
}

fn configure_surface(
    device: &wgpu::Device,
    surface: &wgpu::Surface,
    swapchain_format: wgpu::TextureFormat,
    size: winit::dpi::PhysicalSize<u32>,
) {
    surface.configure(
        &device,
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo, // disable vsync by switching to Mailbox,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: Vec::new(),
        },
    );
}

fn main() {
    env_logger::init();

    let runtime = tokio::runtime::Runtime::new().unwrap();

    let opt = Args::parse();
    let epoch = opt
        .time
        .map(|s| {
            let t = time::Time::parse(
                &s,
                time::macros::format_description!("[hour]:[minute]:[second]"),
            )
            .unwrap();
            t.hour() as f64 / 24.0 + t.minute() as f64 / 1440.0 + t.second() as f64 / 86400.0 - 0.5
        })
        .unwrap_or(0.0);

    let trace_path: Option<&std::path::Path> = if cfg!(feature = "trace") {
        std::fs::create_dir_all("trace").unwrap();
        Some(std::path::Path::new("trace"))
    } else {
        None
    };

    let event_loop = EventLoop::new();
    let monitor = event_loop
        .available_monitors()
        .find(|monitor| monitor.video_modes().any(|mode| mode.size().width == 1920));
    let window = winit::window::WindowBuilder::new()
        .with_visible(false)
        .with_fullscreen(Some(winit::window::Fullscreen::Borderless(monitor)))
        .build(&event_loop)
        .unwrap();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let surface = unsafe { instance.create_surface(&window).unwrap() };
    let adapter = runtime
        .block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Unable to create compatible wgpu adapter");
    let swapchain_format = surface.get_capabilities(&adapter).formats[0];

    // Terra requires support for BC texture compression.
    assert!(adapter.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC));
    assert!(adapter.features().contains(wgpu::Features::PUSH_CONSTANTS));
    let features = wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        | wgpu::Features::PUSH_CONSTANTS
        | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
        | adapter.features() & wgpu::Features::MULTI_DRAW_INDIRECT;

    let (device, queue) = runtime
        .block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                features,
                limits: wgpu::Limits {
                    max_texture_array_layers: 1024,
                    max_compute_invocations_per_workgroup: 512,
                    max_push_constant_size: 128,
                    max_storage_buffer_binding_size: 256 * 1024 * 1024,
                    ..wgpu::Limits::default()
                },
                label: None,
            },
            trace_path,
        ))
        .expect("Unable to create compatible wgpu device");

    let mut size = window.inner_size();
    let mut depth_buffer = make_depth_buffer(&device, size.width, size.height);

    configure_surface(&device, &surface, swapchain_format, size);

    #[cfg(feature = "smaa")]
    let mut smaa_target = smaa::SmaaTarget::new(
        &device,
        &queue,
        size.width,
        size.height,
        swapchain_format,
        smaa::SmaaMode::Smaa1X,
    );

    let mut gilrs = Gilrs::new().unwrap();
    let mut current_gamepad = None;
    for (_id, gamepad) in gilrs.gamepads() {
        current_gamepad = Some(gamepad.id());
    }

    let plus_center =
        open_location_code::decode(&opt.plus).expect("Failed to parse plus code").center;

    let mut camera =
        DualPlanetCam::new(plus_center.y(), plus_center.x(), opt.heading, -10.0, opt.elevation);

    let mut mouse_state = false;
    let mut last_mouse_position: Option<PhysicalPosition<f64>> = None;

    let mut up_key = false;
    let mut down_key = false;
    let mut right_key = false;
    let mut left_key = false;
    let mut space_key = false;
    let mut z_key = false;

    if let Some(opt2) = opt.subcommand {
        match opt2 {
            #[cfg(feature = "generate")]
            SubcommandArgs::Generate { path, download } => {
                let pb = indicatif::ProgressBar::new(100);
                pb.set_style(
                    indicatif::ProgressStyle::default_bar()
                        .template(
                            "{msg} {pos}/{len} [{wide_bar}] {percent}% {per_sec} {eta_precise}",
                        )
                        .unwrap()
                        .progress_chars("=> "),
                );
                let mut last_message: Option<String> = None;
                let progress_callback = |l: String, i: usize, total: usize| {
                    pb.set_length(total as u64);
                    pb.set_position(i as u64);
                    if last_message.is_none() || &*l != last_message.as_ref().unwrap() {
                        last_message = Some(l.clone());
                        pb.set_message(l);
                        pb.reset_eta();
                    }
                };
                runtime
                    .block_on(terra_generate::generate(&path, download, progress_callback))
                    .unwrap()
            }
        }
    };

    let server = opt.server.unwrap_or_else(|| terra::DEFAULT_TILE_SERVER_URL.to_string());
    let mut terrain = runtime.block_on(terra::Terrain::new(&device, &queue, server)).unwrap();

    {
        let pb = indicatif::ProgressBar::new(100);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{msg} {pos}/{len} [{wide_bar}] {percent}% {per_sec} {eta_precise}")
                .unwrap()
                .progress_chars("=> "),
        );
        pb.set_length(100);
        pb.set_message("Streaming tiles");
        terrain.poll_loading_status(
            &device,
            &queue,
            camera.anchored_position_view(0.0).0.into(),
            |n| pb.set_position(n as u64),
        );
        pb.finish_and_clear();
    }

    let mut last_time = None;
    let start_time = std::time::Instant::now();
    window.set_visible(true);
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            event::Event::WindowEvent { event, .. } => match event {
                event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                event::WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                    mouse_state = state == ElementState::Pressed;
                    if !mouse_state {
                        last_mouse_position = None;
                    }
                }
                event::WindowEvent::CursorMoved { position, .. } => {
                    if let Some(last_position) = last_mouse_position {
                        camera.increase_bearing((position.x - last_position.x) * -0.2);
                        camera.increase_pitch((position.y - last_position.y) * 0.1);
                    }
                    if mouse_state {
                        last_mouse_position = Some(position);
                    }
                }
                #[allow(deprecated)]
                event::WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(keycode), state, modifiers, ..
                        },
                    ..
                } => {
                    let pressed = state == event::ElementState::Pressed;
                    match keycode {
                        event::VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                        event::VirtualKeyCode::Left => left_key = pressed,
                        event::VirtualKeyCode::Right => right_key = pressed,
                        event::VirtualKeyCode::Up => up_key = pressed,
                        event::VirtualKeyCode::Down => down_key = pressed,
                        event::VirtualKeyCode::Space => space_key = pressed,
                        event::VirtualKeyCode::Z | event::VirtualKeyCode::Semicolon => {
                            z_key = pressed
                        }
                        event::VirtualKeyCode::Tab => {
                            if pressed && modifiers.ctrl() {
                                if camera.is_detached() {
                                    camera.attach();
                                } else {
                                    camera.detach();
                                }
                            }
                        }
                        _ => {}
                    }
                }
                event::WindowEvent::Resized(new_size) => {
                    size = new_size;

                    #[cfg(feature = "smaa")]
                    smaa_target.resize(&device, new_size.width, new_size.height);

                    configure_surface(&device, &surface, swapchain_format, size);
                    depth_buffer = make_depth_buffer(&device, size.width, size.height);
                }
                _ => {}
            },
            event::Event::MainEventsCleared => {
                window.request_redraw();
            }
            event::Event::RedrawRequested(_) => {
                let frame_texture = surface.get_current_texture();
                let frame_texture = match frame_texture {
                    Ok(f) => f,
                    Err(_) => return,
                };
                let frame_texture_view = frame_texture.texture.create_view(&Default::default());

                #[cfg(not(feature = "smaa"))]
                let frame = &frame_texture_view;

                #[cfg(feature = "smaa")]
                let frame = smaa_target.start_frame(&device, &queue, &frame_texture_view);

                let time = Instant::now();
                let dt = (time - last_time.unwrap_or(time)).as_secs_f64();
                last_time = Some(time);

                // Compute motion from keyboard.
                let mut up_factor = space_key as i32 as f64 - z_key as i32 as f64;
                let mut right_factor = right_key as i32 as f64 - left_key as i32 as f64;
                let mut forward_factor = up_key as i32 as f64 - down_key as i32 as f64;

                // Incorporate gamepad input.
                while let Some(gilrs::Event { id, event: _event, time: _ }) = gilrs.next_event() {
                    current_gamepad = Some(id);
                }
                if let Some(gamepad) = current_gamepad.map(|id| gilrs.gamepad(id)) {
                    forward_factor += gamepad.value(Axis::LeftStickY) as f64;
                    right_factor += gamepad.value(Axis::LeftStickX) as f64;
                    if gamepad.is_pressed(Button::DPadUp) {
                        up_factor += 1.0;
                    }
                    if gamepad.is_pressed(Button::DPadDown) {
                        up_factor += -1.0;
                    }
                    camera.increase_bearing(120.0 * gamepad.value(Axis::RightZ) as f64 * dt);
                    camera.increase_bearing(120.0 * gamepad.value(Axis::RightStickX) as f64 * dt);
                    camera.increase_pitch(120.0 * gamepad.value(Axis::RightStickY) as f64 * dt);
                }

                // Use control inputs to update camera location.
                let vertical_speed = 3.0 * camera.height();
                let horizontal_speed = 12.0 * camera.height().clamp(2.0, 100000.0);
                camera.move_up(up_factor * vertical_speed * dt);
                camera.move_forward(forward_factor * horizontal_speed * dt);
                camera.move_right(right_factor * horizontal_speed * dt);

                // Compute position and camera matrices.
                let (lat, long) = camera.latitude_longitude();
                let surface_height = terrain.get_height(lat.to_radians(), long.to_radians()) as f64;
                let (position, view) = camera.anchored_position_view(surface_height + 2.0);
                let proj = compute_projection_matrix(size.width as f32, size.height as f32);
                let view: cgmath::Matrix4<f32> = cgmath::Matrix3::from(view).into();
                let view_proj = proj * view;
                let view_proj = mint::ColumnMatrix4 {
                    x: view_proj.x.into(),
                    y: view_proj.y.into(),
                    z: view_proj.z.into(),
                    w: view_proj.w.into(),
                };

                let render_view = camera.free_position_view(surface_height + 2.0);
                let render_view_proj = proj * cgmath::Matrix4::from(render_view);
                let render_view_proj = mint::ColumnMatrix4 {
                    x: render_view_proj.x.into(),
                    y: render_view_proj.y.into(),
                    z: render_view_proj.z.into(),
                    w: render_view_proj.w.into(),
                };

                terrain.update(
                    &device,
                    &queue,
                    view_proj,
                    position.into(),
                    2451545.0
                        + epoch
                        + start_time.elapsed().as_secs_f64() * opt.timescale / 86400.0,
                );
                terrain.render_shadows(&device, &queue);
                terrain.render(
                    &device,
                    &queue,
                    &frame,
                    &depth_buffer,
                    (size.width, size.height),
                    render_view_proj,
                );

                drop(frame);
                frame_texture.present();
            }
            _ => (),
        }
    });
}
