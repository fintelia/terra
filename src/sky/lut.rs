use failure::Error;
use gfx;
use gfx::texture::{AaMode, Kind, Mipmap};
use gfx_core::{format, handle};

use cache::{AssetLoadContext, GeneratedAsset};

pub trait LookupTableDefinition {
    fn filename(&self) -> String;
    fn size(&self) -> [u16; 4];
    fn compute(&self, [u16; 4]) -> [f32; 4];

    fn inv_size(&self) -> [f64; 4] {
        let s = self.size();
        [
            1.0 / f64::from(s[0]),
            1.0 / f64::from(s[1]),
            1.0 / f64::from(s[2]),
            1.0 / f64::from(s[3]),
        ]
    }
}
impl<T: LookupTableDefinition> GeneratedAsset for T {
    type Type = LookupTable;

    fn filename(&self) -> String {
        LookupTableDefinition::filename(self)
    }
    fn generate(&self, context: &mut AssetLoadContext) -> Result<Self::Type, Error> {
        let size = self.size();
        context.increment_level(
            &format!("Generating {}", self.filename()),
            size[1] * size[2] * size[3],
        );

        let mut data = Vec::new();
        for w in 0..size[3] {
            for z in 0..size[2] {
                for y in 0..size[1] {
                    context.set_progress(w * size[2] * size[1] + z * size[1] + y);
                    for x in 0..size[0] {
                        data.push(self.compute([x, y, z, w]));
                    }
                }
            }
        }
        context.set_progress(size[1] * size[2] * size[3]);
        context.decrement_level();

        Ok(LookupTable { size, data })
    }
}

#[derive(Serialize, Deserialize)]
pub struct LookupTable {
    pub size: [u16; 4],
    pub data: Vec<[f32; 4]>,
}
impl LookupTable {
    pub fn get2(&self, x: f32, y: f32) -> [f32; 4] {
        assert_eq!(self.size[2], 1);
        assert_eq!(self.size[3], 1);
        assert!(x >= 0.0);
        assert!(y >= 0.0);
        assert!(x <= 1.0);
        assert!(y <= 1.0);

        let x = (x * (self.size[0] - 1) as f32).round() as usize;
        let y = (y * (self.size[1] - 1) as f32).round() as usize;
        self.data[x + y * self.size[0] as usize]
    }
}
pub struct GpuLookupTable<R: gfx::Resources> {
    pub(crate) texture_view: handle::ShaderResourceView<R, [f32; 4]>,

    #[allow(unused)]
    pub(crate) texture: handle::Texture<R, format::R32_G32_B32_A32>,
}
impl<R: gfx::Resources> GpuLookupTable<R> {
    pub fn new<F: gfx::Factory<R>>(factory: &mut F, table: &LookupTable) -> Result<Self, Error> {
        let kind = match table.size {
            [x, 1, 1, 1] => Kind::D1(x),
            [x, y, 1, 1] => Kind::D2(x, y, AaMode::Single),
            [x, y, z, 1] => Kind::D3(x, y, z),
            [x, y, z, w] => Kind::D3(x, y, z * w),
        };

        let (texture, texture_view) = factory.create_texture_immutable::<format::Rgba32F>(
            kind,
            Mipmap::Provided,
            &[gfx::memory::cast_slice(&table.data[..])],
        )?;

        Ok(Self {
            texture_view,
            texture,
        })
    }
}
