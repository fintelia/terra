use cgmath::EuclideanSpace;
use gilrs::{Axis, Button, Gilrs};
use std::f64::consts::PI;
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

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

fn make_swapchain(
    device: &wgpu::Device,
    surface: &wgpu::Surface,
    width: u32,
    height: u32,
) -> wgpu::SwapChain {
    device.create_swap_chain(
        &surface,
        &wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width,
            height,
            present_mode: wgpu::PresentMode::Mailbox,
        },
    )
}
fn make_depth_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width, height, depth: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            label: None,
        })
        .create_default_view()
}

fn main() {
    env_logger::init();

    let mut gilrs = Gilrs::new().unwrap();
    let mut current_gamepad = None;
    for (_id, gamepad) in gilrs.gamepads() {
        current_gamepad = Some(gamepad.id());
    }

    let mapfile = terra::MapFileBuilder::build().unwrap();

    let event_loop = EventLoop::new();
    let instance = wgpu::Instance::new(wgpu::BackendBit::VULKAN);
    let window = winit::window::Window::new(&event_loop).unwrap();
    for monitor in window.available_monitors() {
        if monitor.video_modes().any(|mode| mode.size().width == 1920) {
            window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(monitor)));
            break;
        }
    }
    let mut size = window.inner_size();
    let surface = unsafe { instance.create_surface(&window) };
    let adapter =
        futures::executor::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: Some(&surface),
        }))
        .unwrap();
    let (device, mut queue) = futures::executor::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
            shader_validation: true,
        },
        None,
    )).unwrap();
    let mut swap_chain = make_swapchain(&device, &surface, size.width, size.height);
    let mut depth_buffer = make_depth_buffer(&device, size.width, size.height);
    let mut proj = compute_projection_matrix(size.width as f32, size.height as f32);

    let planet_radius = 6371000.0;
    let mut angle = 0.0f64;
    let mut lat = 0.0f64;
    let mut long = 0.0f64;
    let mut altitude = 100.0f64;

    let mut terrain = terra::Terrain::new(&device, &mut queue, mapfile).unwrap();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };
        match event {
            event::Event::WindowEvent { event, .. } => match event {
                event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                event::WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => match keycode {
                    event::VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                    event::VirtualKeyCode::Space => altitude += 0.1 * altitude,
                    event::VirtualKeyCode::Semicolon => altitude -= 0.1 * altitude,
                    // event::VirtualKeyCode::Left => eye.x -= 50.0,
                    // event::VirtualKeyCode::Right => eye.x += 50.0,
                    // event::VirtualKeyCode::Up => eye.z -= 50.0,
                    // event::VirtualKeyCode::Down => eye.z += 50.0,
                    _ => {}
                },
                event::WindowEvent::Resized(new_size) => {
                    size = new_size;
                    swap_chain = make_swapchain(&device, &surface, size.width, size.height);
                    depth_buffer = make_depth_buffer(&device, size.width, size.height);
                    proj = compute_projection_matrix(size.width as f32, size.height as f32);
                }
                _ => {}
            },
            event::Event::MainEventsCleared => {
                let frame = &swap_chain
                    .get_next_frame()
                    .unwrap().output.view;

                while let Some(gilrs::Event { id, event: _event, time: _ }) = gilrs.next_event() {
                    current_gamepad = Some(id);
                }
                if let Some(gamepad) = current_gamepad.map(|id| gilrs.gamepad(id)) {
                    lat += angle.cos() * gamepad.value(Axis::LeftStickY) as f64 * 0.01
                        - angle.sin() * gamepad.value(Axis::LeftStickX) as f64 * 0.01;

                    long += -angle.sin() * gamepad.value(Axis::LeftStickY) as f64 * 0.01
                        + angle.cos() * gamepad.value(Axis::LeftStickX) as f64 * 0.01;

                    angle -= gamepad.value(Axis::RightZ) as f64 * 0.01;

                    if gamepad.is_pressed(Button::DPadUp) {
                        altitude += 0.01 * altitude;
                    }
                    if gamepad.is_pressed(Button::DPadDown) {
                        altitude -= 0.01 * altitude;
                    }
                }

                lat = lat.max(-PI).min(PI);
                if long < -PI {
                    long += PI * 2.0;
                }
                if long > PI {
                    long -= PI * 2.0;
                }

                let r = altitude + planet_radius;
                let eye = cgmath::Point3::new(
                    r * lat.cos() * long.cos(),
                    r * lat.cos() * long.sin(),
                    r * lat.sin(),
                );

                let latc = lat + angle.cos() * 0.001;
                let longc = long - angle.sin() * 0.001;

                let center = cgmath::Point3::new(
                    planet_radius * latc.cos() * longc.cos() - eye.x,
                    planet_radius * latc.cos() * longc.sin() - eye.y,
                    planet_radius * latc.sin() - eye.z,
                );
                let up = cgmath::Vector3::new(eye.x as f32, eye.y as f32, eye.z as f32);

                let view = cgmath::Matrix4::look_at(
                    cgmath::Point3::origin(),
                    cgmath::Point3::new(center.x as f32, center.y as f32, center.z as f32),
                    up,
                );

                let view_proj = proj * view;
                let view_proj = mint::ColumnMatrix4 {
                    x: view_proj.x.into(),
                    y: view_proj.y.into(),
                    z: view_proj.z.into(),
                    w: view_proj.w.into(),
                };

                terrain.render(
                    &device,
                    &mut queue,
                    &frame,
                    &depth_buffer,
                    (size.width, size.height),
                    view_proj,
                    eye.into(),
                );
            }
            _ => (),
        }
    });
}
