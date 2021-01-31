use crate::asset::AssetLoadContext;
use anyhow::Error;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub(crate) trait LookupTableDefinition: Sync {
    fn name(&self) -> String;
    fn size(&self) -> [u16; 3];
    fn compute(&self, _: [u16; 3]) -> [f32; 4];

    fn inv_size(&self) -> [f64; 3] {
        let s = self.size();
        [1.0 / f64::from(s[0]), 1.0 / f64::from(s[1]), 1.0 / f64::from(s[2])]
    }

    fn generate(&self, context: &mut AssetLoadContext) -> Result<LookupTable, Error> {
        let size = self.size();
        let total = size[0] as u64 * size[1] as u64 * size[2] as u64;
        context.reset(&format!("Generating {}... ", &self.name()), total / 1000);

        let data = (0..total)
            .into_iter()
            .collect::<Vec<_>>()
            .chunks(1000)
            .enumerate()
            .flat_map(|(i, chunk)| {
                context.set_progress(i);
                chunk
                    .into_par_iter()
                    .map(|i| {
                        let x = i % size[0] as u64;
                        let y = (i / size[0] as u64) % size[1] as u64;
                        let z = i / (size[0] as u64 * size[1] as u64) % size[2] as u64;
                        let value = self.compute([x as u16, y as u16, z as u16]);
                        for c in &value {
                            assert!(!c.is_nan())
                        }
                        value
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        context.set_progress(total / 1000);
        Ok(LookupTable { size, data })
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct LookupTable {
    pub size: [u16; 3],
    pub data: Vec<[f32; 4]>,
}
impl LookupTable {
    pub fn get2(&self, x: f64, y: f64) -> [f32; 4] {
        assert_eq!(self.size[2], 1);
        assert!(x >= 0.0);
        assert!(y >= 0.0);
        assert!(x <= 1.0);
        assert!(y <= 1.0);

        let x = (x * (self.size[0] - 1) as f64).round() as usize;
        let y = (y * (self.size[1] - 1) as f64).round() as usize;
        self.data[x + y * self.size[0] as usize]
    }
}
// pub struct GpuLookupTable<R: gfx::Resources> {
//     pub(crate) texture_view: handle::ShaderResourceView<R, [f32; 4]>,

//     #[allow(unused)]
//     pub(crate) texture: handle::Texture<R, format::R32_G32_B32_A32>,
// }
// impl<R: gfx::Resources> GpuLookupTable<R> {
//     pub fn new<F: gfx::Factory<R>>(factory: &mut F, table: &LookupTable) -> Result<Self, Error> {
//         let kind = match table.size {
//             [x, 1, 1] => Kind::D1(x),
//             [x, y, 1] => Kind::D2(x, y, AaMode::Single),
//             [x, y, z] => Kind::D3(x, y, z),
//         };

//         let (texture, texture_view) = factory.create_texture_immutable::<format::Rgba32F>(
//             kind,
//             Mipmap::Provided,
//             &[gfx::memory::cast_slice(&table.data[..])],
//         )?;

//         Ok(Self {
//             texture_view,
//             texture,
//         })
//     }
// }
