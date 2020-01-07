use gilrs::{Axis, Button, Gilrs};
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

fn compute_projection_matrix(width: f32, height: f32) -> cgmath::Matrix4<f32> {
    let aspect = width as f32 / height as f32;
    let f = 1.0 / (55.0f32.to_radians() / aspect).tan();
    let near = 0.1;

    #[cfg_attr(rustfmt, rustfmt_skip)]
    cgmath::Matrix4::new(
        f / aspect, 0.0,  0.0,  0.0,
        0.0,          f,  0.0,  0.0,
        0.0,        0.0,  0.0, -1.0,
        0.0,        0.0, near,  0.0)
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
            present_mode: wgpu::PresentMode::Vsync,
        },
    )
}
fn make_depth_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width, height, depth: 1 },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        })
        .create_default_view()
}

fn main() {
    env_logger::init();

    let mut gilrs = Gilrs::new().unwrap();
    let mut current_gamepad = None;
    for (_id, gamepad) in gilrs.gamepads() {
        println!("{} is {:?}", gamepad.name(), gamepad.power_info());
        current_gamepad = Some(gamepad.id());
    }

    let event_loop = EventLoop::new();

    let (window, mut size, surface) = {
        let window = winit::window::Window::new(&event_loop).unwrap();

        for monitor in window.available_monitors() {
            if monitor.video_modes().any(|mode| mode.size().width == 1920.0) {
                window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(monitor)));
                break;
            }
        }

        let size = window.inner_size().to_physical(window.hidpi_factor());

        let surface = wgpu::Surface::create(&window);
        (window, size, surface)
    };

    let adapter = wgpu::Adapter::request(
        &wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::Default },
        wgpu::BackendBit::PRIMARY,
    )
    .unwrap();

    let (device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions { anisotropic_filtering: false },
        limits: wgpu::Limits::default(),
    });

    let mapfile = terra::MapFileBuilder::new()
        .latitude(42)
        .longitude(-73)
        .vertex_quality(terra::VertexQuality::Medium)
        .texture_quality(terra::TextureQuality::High)
        .grid_spacing(terra::GridSpacing::OneMeter)
        .build()
        .unwrap();

    let mut terrain = terra::Terrain::new(&device, &mut queue, mapfile);

    let mut swap_chain =
        make_swapchain(&device, &surface, size.width.round() as u32, size.height.round() as u32);
    let mut depth_buffer =
        make_depth_buffer(&device, size.width.round() as u32, size.height.round() as u32);

    let proj = compute_projection_matrix(size.width as f32, size.height as f32);

    let mut angle = 3.14159f32;
    let mut eye = cgmath::Point3::new(0.0, 300.0, 0.0);

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
                    event::VirtualKeyCode::Space => eye.y += 0.01 * eye.y,
                    event::VirtualKeyCode::Semicolon => eye.y -= 0.01 * eye.y,
                    event::VirtualKeyCode::Left => eye.x -= 50.0,
                    event::VirtualKeyCode::Right => eye.x += 50.0,
                    event::VirtualKeyCode::Up => eye.z -= 50.0,
                    event::VirtualKeyCode::Down => eye.z += 50.0,
                    _ => {}
                },
                event::WindowEvent::Resized(new_size) => {
                    size = new_size.to_physical(window.hidpi_factor());
                    swap_chain = make_swapchain(
                        &device,
                        &surface,
                        size.width.round() as u32,
                        size.height.round() as u32,
                    );
                    depth_buffer = make_depth_buffer(
                        &device,
                        size.width.round() as u32,
                        size.height.round() as u32,
                    );
                }
                _ => {}
            },
            event::Event::EventsCleared => {
                let frame = swap_chain
                    .get_next_texture()
                    .expect("Timeout when acquiring next swap chain texture");

                let forward = cgmath::Vector3::new(angle.sin(), 0.0, angle.cos());
                let right = cgmath::Vector3::new(angle.cos(), 0.0, -angle.sin());

                while let Some(gilrs::Event { id, .. }) = gilrs.next_event() {
                    current_gamepad = Some(id);
                }
                if let Some(gamepad) = current_gamepad.map(|id| gilrs.gamepad(id)) {
                    if gamepad.value(Axis::LeftStickY).abs() > 0.1 {
                        eye += forward * gamepad.value(Axis::LeftStickY) * 50.0;
                    }
                    if gamepad.value(Axis::LeftStickX).abs() > 0.1 {
                        eye -= right * gamepad.value(Axis::LeftStickX) * 50.0;
                    }
                    if gamepad.is_pressed(Button::DPadLeft) {
                        angle += 0.02;
                    }
                    if gamepad.is_pressed(Button::DPadRight) {
                        angle -= 0.02;
                    }
                }

                let view = cgmath::Matrix4::look_at(
                    cgmath::Point3::new(eye.x, eye.y, eye.z),
                    cgmath::Point3::new(
                        eye.x + 20000.0 * forward.x,
                        0.0,
                        eye.z + 20000.0 * forward.z,
                    ),
                    cgmath::Vector3::new(0.0, -1.0, 0.0),
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
                    (size.width.round() as u32, size.height.round() as u32),
                    view_proj,
                    eye.into(),
                );
            }
            _ => (),
        }
    });
}
