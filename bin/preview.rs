use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

fn compute_projection_matrix(width: f32, height: f32) -> cgmath::Matrix4<f32> {
    let aspect = width as f32 / height as f32;
    let f = 1.0 / (45.0f32.to_radians() / aspect).tan();
    let near = 0.1;

    #[cfg_attr(rustfmt, rustfmt_skip)]
    cgmath::Matrix4::new(
        f / aspect, 0.0,  0.0,  0.0,
        0.0,          f,  0.0,  0.0,
        0.0,        0.0,  0.0, -1.0,
        0.0,        0.0, near,  0.0)
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new();

    let (_window, size, surface) = {
        let window = winit::window::Window::new(&event_loop).unwrap();
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

    let quadtree = terra::QuadTreeBuilder::new()
        .latitude(42)
        .longitude(-73)
        .vertex_quality(terra::VertexQuality::High)
        .texture_quality(terra::TextureQuality::High)
        .grid_spacing(terra::GridSpacing::TwoMeters)
        .build()
        .unwrap();

    let mut swap_chain = device.create_swap_chain(
        &surface,
        &wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width.round() as u32,
            height: size.height.round() as u32,
            present_mode: wgpu::PresentMode::Vsync,
        },
    );

    let proj = compute_projection_matrix(size.width as f32, size.height as f32);

    let mut terrain = terra::Terrain::new(&device, &mut queue, quadtree);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };
        match event {
            event::Event::WindowEvent { event, .. } => match event {
                event::WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::Escape),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                }
                | event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            event::Event::EventsCleared => {
                let frame = swap_chain
                    .get_next_texture()
                    .expect("Timeout when acquiring next swap chain texture");

                let eye = mint::Point3::from_slice(&[0.0, 200.0, -500.0]);
                let view = cgmath::Matrix4::look_at(
                    cgmath::Point3::new(eye.x, eye.y, eye.z),
                    cgmath::Point3::new(0.0, 0.0, 0.0),
                    cgmath::Vector3::new(0.0, 1.0, 0.0),
                );

                let view_proj = proj * view;
                let view_proj = mint::ColumnMatrix4 {
                    x: view_proj.x.into(),
                    y: view_proj.y.into(),
                    z: view_proj.z.into(),
                    w: view_proj.w.into(),
                };

                terrain.render(&device, &mut queue, &frame, view_proj, eye);
            }
            _ => (),
        }
    });
}
