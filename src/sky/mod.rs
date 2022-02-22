use crate::asset::AssetLoadContext;
use crate::sky::lut::{LookupTable, LookupTableDefinition};
use crate::sky::precompute::{InscatteringTable, TransmittanceTable};
use anyhow::Error;
use wgpu::util::DeviceExt;

mod lut;
mod precompute;

pub(crate) struct Atmosphere {
    pub transmittance: LookupTable,
    pub inscattering: LookupTable,
}
impl Atmosphere {
    pub fn new(context: &mut AssetLoadContext) -> Result<Self, Error> {
        let transmittance = TransmittanceTable { steps: 1000 }.generate(context)?;
        let inscattering =
            InscatteringTable { steps: 30, transmittance: &transmittance }.generate(context)?;

        Ok(Self { transmittance, inscattering })
    }
}

pub(crate) fn create_starfield(device: &wgpu::Device) -> wgpu::Buffer {
    let mut stars = vec![0.0f32; 4 * 9096];
    bytemuck::cast_slice_mut(&mut stars).copy_from_slice(include_bytes!("../../assets/stars.bin"));

    for star in stars.chunks_mut(4) {
        let (gal_lat, gal_long) = (star[0] as f64, star[1] as f64);
        star[0] = astro::coords::dec_frm_gal(gal_long, gal_lat) as f32;
        star[1] = astro::coords::asc_frm_gal(gal_long, gal_lat) as f32;
    }

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("buffer.starfield"),
        contents: bytemuck::cast_slice(&stars),
        usage: wgpu::BufferUsages::STORAGE,
    })
}
