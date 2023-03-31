use clap::Parser;
use num::Integer;
use planetcam::DualPlanetCam;
use std::num::NonZeroU32;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = "8FH495PF+29")]
    plus: String,
    #[arg(long, default_value = "0")]
    heading: f64,
    #[arg(short, long, default_value = "200000")]
    elevation: f64,
    #[arg(long)]
    time: Option<String>,
    #[arg(long, default_value = "1920")]
    width: u32,
    #[arg(long, default_value = "1080")]
    height: u32,
    #[arg(short, long, default_value = "screenshot.png")]
    output: String,
}

#[tokio::main]
async fn main() {
    env_logger::init();

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

    let row_pitch: u32 =
        Integer::next_multiple_of(&(opt.width * 4), &wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

    let projection_matrix = {
        let aspect = opt.width as f32 / opt.height as f32;
        let f = 1.0 / (45.0f32.to_radians() / aspect).tan();
        let near = 0.1;
        #[cfg_attr(rustfmt, rustfmt_skip)]
        cgmath::Matrix4::new(
            f/aspect,  0.0,  0.0,   0.0,
            0.0,       f,    0.0,   0.0,
            0.0,       0.0,  0.0,  -1.0,
            0.0,       0.0,  near,  0.0)
    };

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        dx12_shader_compiler: wgpu::Dx12Compiler::default(),
    });
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::TEXTURE_COMPRESSION_BC
                    | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                    | wgpu::Features::PUSH_CONSTANTS
                    | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
                limits: wgpu::Limits {
                    max_texture_array_layers: 1024,
                    max_compute_invocations_per_workgroup: 512,
                    max_push_constant_size: 128,
                    ..wgpu::Limits::default()
                },
                label: None,
            },
            None,
        )
        .await
        .unwrap();

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (row_pitch * opt.height) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let size = wgpu::Extent3d { width: opt.width, height: opt.height, depth_or_array_layers: 1 };
    let color_buffer = device.create_texture(&wgpu::TextureDescriptor {
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        label: None,
        view_formats: &[],
    });
    let depth_buffer = device.create_texture(&wgpu::TextureDescriptor {
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
        label: None,
    });

    let mut terrain =
        terra::Terrain::new(&device, &queue, terra::DEFAULT_TILE_SERVER_URL.to_string())
            .await
            .unwrap();

    let plus_center =
        open_location_code::decode(&opt.plus).expect("Failed to parse plus code").center;
    let camera =
        DualPlanetCam::new(plus_center.y(), plus_center.x(), opt.heading, -10.0, opt.elevation);
    let (lat, long) = camera.latitude_longitude();
    let surface_height = terrain.get_height(lat.to_radians(), long.to_radians()) as f64;

    let position = camera.anchored_position_view(surface_height).0;
    let render_view = camera.free_position_view(surface_height);
    let render_view_proj = projection_matrix * cgmath::Matrix4::from(render_view);
    let render_view_proj = mint::ColumnMatrix4 {
        x: render_view_proj.x.into(),
        y: render_view_proj.y.into(),
        z: render_view_proj.z.into(),
        w: render_view_proj.w.into(),
    };

    terrain.update(&device, &queue, render_view_proj, position.into(), 2451545.0 + epoch);
    terrain.render_shadows(&device, &queue);
    terrain.render(
        &device,
        &queue,
        &color_buffer.create_view(&Default::default()),
        &depth_buffer.create_view(&Default::default()),
        (opt.width, opt.height),
        render_view_proj,
    );

    let command_buffer = {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            color_buffer.as_image_copy(),
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(row_pitch),
                    rows_per_image: None,
                },
            },
            size,
        );
        encoder.finish()
    };
    queue.submit(Some(command_buffer));

    wgpu::util::DownloadBuffer::read_buffer(
        &device,
        &queue,
        &output_buffer.slice(..),
        move |download| {
            let mut buffer = download
                .unwrap()
                .chunks(row_pitch as usize)
                .flat_map(|row| &row[..opt.width as usize * 4])
                .copied()
                .collect::<Vec<_>>();
            for pixel in buffer.chunks_exact_mut(4) {
                pixel.swap(0, 2);
                pixel[3] = 255;
            }
            image::save_buffer(
                opt.output,
                &buffer,
                opt.width as u32,
                opt.height as u32,
                image::ColorType::Rgba8,
            )
            .unwrap();
        },
    );
    device.poll(wgpu::Maintain::Wait);
}
