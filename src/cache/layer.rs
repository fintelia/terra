use std::{ops::{RangeInclusive, Index, IndexMut}, num::NonZeroU32, collections::HashMap};

use serde::{Serialize, Deserialize};
use types::VNode;
use vec_map::VecMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TextureFormat {
    R8,
    RG8,
    RGBA8,
    R16,
    RG16F,
    RGBA16F,
    R32F,
    RG32F,
    RGBA32F,
    SRGBA,
    BC4,
    BC5,
    UASTC,
}
impl TextureFormat {
    /// Returns the number of bytes in a single texel of the format. Actually reports bytes per
    /// block for compressed formats.
    pub fn bytes_per_block(&self) -> usize {
        match *self {
            TextureFormat::R8 => 1,
            TextureFormat::RG8 => 2,
            TextureFormat::RGBA8 => 4,
            TextureFormat::R16 => 2,
            TextureFormat::RG16F => 4,
            TextureFormat::RGBA16F => 8,
            TextureFormat::R32F => 4,
            TextureFormat::RG32F => 8,
            TextureFormat::RGBA32F => 16,
            TextureFormat::SRGBA => 4,
            TextureFormat::BC4 => 8,
            TextureFormat::BC5 => 16,
            TextureFormat::UASTC => 16,
        }
    }
    pub fn to_wgpu(&self, wgpu_features: wgpu::Features) -> wgpu::TextureFormat {
        match *self {
            TextureFormat::R8 => wgpu::TextureFormat::R8Unorm,
            TextureFormat::RG8 => wgpu::TextureFormat::Rg8Unorm,
            TextureFormat::RGBA8 => wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::R16 => wgpu::TextureFormat::R16Unorm,
            TextureFormat::RG16F => wgpu::TextureFormat::Rg16Float,
            TextureFormat::RGBA16F => wgpu::TextureFormat::Rgba16Float,
            TextureFormat::R32F => wgpu::TextureFormat::R32Float,
            TextureFormat::RG32F => wgpu::TextureFormat::Rg32Float,
            TextureFormat::RGBA32F => wgpu::TextureFormat::Rgba32Float,
            TextureFormat::SRGBA => wgpu::TextureFormat::Rgba8UnormSrgb,
            TextureFormat::BC4 => wgpu::TextureFormat::Bc4RUnorm,
            TextureFormat::BC5 => wgpu::TextureFormat::Bc5RgUnorm,
            TextureFormat::UASTC => {
                if wgpu_features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
                    wgpu::TextureFormat::Bc7RgbaUnorm
                } else if wgpu_features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC_LDR) {
                    wgpu::TextureFormat::Astc {
                        block: wgpu::AstcBlock::B4x4,
                        channel: wgpu::AstcChannel::Unorm,
                    }
                } else {
                    unreachable!("Wgpu reports no texture compression support?")
                }
            }
        }
    }
    pub fn block_size(&self) -> u32 {
        match *self {
            TextureFormat::BC4 | TextureFormat::BC5 | TextureFormat::UASTC => 4,
            TextureFormat::R8
            | TextureFormat::RG8
            | TextureFormat::RGBA8
            | TextureFormat::R16
            | TextureFormat::RG16F
            | TextureFormat::RGBA16F
            | TextureFormat::R32F
            | TextureFormat::RG32F
            | TextureFormat::RGBA32F
            | TextureFormat::SRGBA => 1,
        }
    }
    pub fn is_compressed(&self) -> bool {
        match *self {
            TextureFormat::BC4 | TextureFormat::BC5 | TextureFormat::UASTC => true,
            TextureFormat::R8
            | TextureFormat::RG8
            | TextureFormat::RGBA8
            | TextureFormat::R16
            | TextureFormat::RG16F
            | TextureFormat::RGBA16F
            | TextureFormat::R32F
            | TextureFormat::RG32F
            | TextureFormat::RGBA32F
            | TextureFormat::SRGBA => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum LayerType {
    Heightmaps = 0,
    Displacements = 1,
    AlbedoRoughness = 2,
    Normals = 3,
    GrassCanopy = 4,
    TreeAttributes = 5,
    AerialPerspective = 6,
    BentNormals = 7,
    TreeCover = 8,
    BaseAlbedo = 9,
    RootAerialPerspective = 10,
    LandFraction = 11,
    Slopes = 12,
}
impl LayerType {
    pub fn index(&self) -> usize {
        *self as usize
    }
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => LayerType::Heightmaps,
            1 => LayerType::Displacements,
            2 => LayerType::AlbedoRoughness,
            3 => LayerType::Normals,
            4 => LayerType::GrassCanopy,
            5 => LayerType::TreeAttributes,
            6 => LayerType::AerialPerspective,
            7 => LayerType::BentNormals,
            8 => LayerType::TreeCover,
            9 => LayerType::BaseAlbedo,
            10 => LayerType::RootAerialPerspective,
            11 => LayerType::LandFraction,
            12 => LayerType::Slopes,
            _ => unreachable!(),
        }
    }
    pub fn bit_mask(&self) -> LayerMask {
        (*self).into()
    }
    pub fn name(&self) -> &'static str {
        match *self {
            LayerType::Heightmaps => "heightmaps",
            LayerType::Displacements => "displacements",
            LayerType::AlbedoRoughness => "albedo",
            LayerType::Normals => "normals",
            LayerType::GrassCanopy => "grass_canopy",
            LayerType::TreeAttributes => "tree_attributes",
            LayerType::AerialPerspective => "aerial_perspective",
            LayerType::BentNormals => "bent_normals",
            LayerType::TreeCover => "treecover",
            LayerType::BaseAlbedo => "base_albedo",
            LayerType::RootAerialPerspective => "root_aerial_perspective",
            LayerType::LandFraction => "land_fraction",
            LayerType::Slopes => "slopes",
        }
    }
    pub fn streamed_levels(&self) -> u8 {
        match *self {
            LayerType::Heightmaps => VNode::LEVEL_CELL_76M + 1,
            LayerType::BaseAlbedo => VNode::LEVEL_CELL_610M + 1,
            LayerType::TreeCover => VNode::LEVEL_CELL_76M + 1,
            LayerType::LandFraction => VNode::LEVEL_CELL_76M + 1,
            _ => 0,
        }
    }
    pub fn dynamic(&self) -> bool {
        match *self {
            LayerType::AerialPerspective | LayerType::RootAerialPerspective => true,
            _ => false,
        }
    }
    pub fn grid_registration(&self) -> bool {
        match *self {
            LayerType::Heightmaps => true,
            LayerType::Displacements => true,
            LayerType::AlbedoRoughness => false,
            LayerType::Normals => false,
            LayerType::GrassCanopy => false,
            LayerType::TreeAttributes => false,
            LayerType::AerialPerspective => true,
            LayerType::BentNormals => true,
            LayerType::TreeCover => false,
            LayerType::BaseAlbedo => false,
            LayerType::RootAerialPerspective => true,
            LayerType::LandFraction => false,
            LayerType::Slopes => true,
        }
    }
    /// Number of samples in each dimension, per tile.
    pub fn texture_resolution(&self) -> u32 {
        match *self {
            LayerType::Heightmaps => 521,
            LayerType::Displacements => 65,
            LayerType::AlbedoRoughness => 516,
            LayerType::Normals => 516,
            LayerType::GrassCanopy => 516,
            LayerType::TreeAttributes => 516,
            LayerType::AerialPerspective => 17,
            LayerType::BentNormals => 513,
            LayerType::TreeCover => 516,
            LayerType::BaseAlbedo => 516,
            LayerType::RootAerialPerspective => 65,
            LayerType::LandFraction => 516,
            LayerType::Slopes => 517,
        }
    }
    /// Number of samples outside the tile on each side.
    pub fn texture_border_size(&self) -> u32 {
        match *self {
            LayerType::Heightmaps => 4,
            LayerType::Displacements => 0,
            LayerType::AlbedoRoughness => 2,
            LayerType::Normals => 2,
            LayerType::GrassCanopy => 2,
            LayerType::TreeAttributes => 2,
            LayerType::AerialPerspective => 0,
            LayerType::BentNormals => 0,
            LayerType::TreeCover => 2,
            LayerType::BaseAlbedo => 2,
            LayerType::RootAerialPerspective => 0,
            LayerType::LandFraction => 2,
            LayerType::Slopes => 2,
        }
    }
    pub fn texture_formats(&self) -> &'static [TextureFormat] {
        match *self {
            LayerType::Heightmaps => &[TextureFormat::R16],
            LayerType::Displacements => &[TextureFormat::RGBA32F],
            LayerType::AlbedoRoughness => &[TextureFormat::RGBA8],
            LayerType::Normals => &[TextureFormat::RG8],
            LayerType::GrassCanopy => &[TextureFormat::RGBA8],
            LayerType::TreeAttributes => &[TextureFormat::RGBA8],
            LayerType::AerialPerspective => &[TextureFormat::RGBA16F],
            LayerType::BentNormals => &[TextureFormat::RGBA8],
            LayerType::TreeCover => &[TextureFormat::R8],
            LayerType::BaseAlbedo => &[TextureFormat::RGBA8],
            LayerType::RootAerialPerspective => &[TextureFormat::RGBA16F],
            LayerType::LandFraction => &[TextureFormat::R8],
            LayerType::Slopes => &[TextureFormat::RG16F],
        }
    }
    pub fn level_range(&self) -> RangeInclusive<u8> {
        match *self {
            LayerType::Heightmaps => 0..=VNode::LEVEL_CELL_10M,
            LayerType::Displacements => 0..=VNode::LEVEL_CELL_5MM,
            LayerType::AlbedoRoughness => 0..=VNode::LEVEL_CELL_5MM,
            LayerType::Normals => 0..=VNode::LEVEL_CELL_5MM,
            LayerType::GrassCanopy => VNode::LEVEL_CELL_1M..=VNode::LEVEL_CELL_1M,
            LayerType::TreeAttributes => VNode::LEVEL_CELL_10M..=VNode::LEVEL_CELL_10M,
            LayerType::AerialPerspective => 3..=VNode::LEVEL_SIDE_610M,
            LayerType::BentNormals => VNode::LEVEL_CELL_153M..=VNode::LEVEL_CELL_76M,
            LayerType::TreeCover => 0..=VNode::LEVEL_CELL_76M,
            LayerType::BaseAlbedo => 0..=VNode::LEVEL_CELL_610M,
            LayerType::RootAerialPerspective => 0..=0,
            LayerType::LandFraction => 0..=VNode::LEVEL_CELL_76M,
            LayerType::Slopes => VNode::LEVEL_CELL_10M..=VNode::LEVEL_CELL_10M,
        }
    }
    pub fn min_level(&self) -> u8 {
        *self.level_range().start()
    }
    pub fn max_level(&self) -> u8 {
        *self.level_range().end()
    }
    pub fn iter() -> impl Iterator<Item = Self> {
        (0..=12).map(Self::from_index)
    }
}
impl<T> Index<LayerType> for VecMap<T> {
    type Output = T;
    fn index(&self, i: LayerType) -> &Self::Output {
        &self[i as usize]
    }
}
impl<T> IndexMut<LayerType> for VecMap<T> {
    fn index_mut(&mut self, i: LayerType) -> &mut Self::Output {
        &mut self[i as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum MeshType {
    Terrain = 0,
    Grass = 1,
    TreeBillboards = 2,
}
impl MeshType {
    pub fn bit_mask(&self) -> LayerMask {
        (*self).into()
    }
    pub fn name(&self) -> &'static str {
        match *self {
            MeshType::Terrain => "terrain",
            MeshType::Grass => "grass",
            MeshType::TreeBillboards => "tree_billboards",
        }
    }
    fn from_index(i: usize) -> Self {
        match i {
            0 => MeshType::Terrain,
            1 => MeshType::Grass,
            2 => MeshType::TreeBillboards,
            _ => unreachable!(),
        }
    }
    pub fn iter() -> impl Iterator<Item = Self> {
        (0..=2).map(Self::from_index)
    }
}
impl<T> Index<MeshType> for VecMap<T> {
    type Output = T;
    fn index(&self, i: MeshType) -> &Self::Output {
        &self[i as usize]
    }
}
impl<T> IndexMut<MeshType> for VecMap<T> {
    fn index_mut(&mut self, i: MeshType) -> &mut Self::Output {
        &mut self[i as usize]
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) struct LayerMask(NonZeroU32);
impl LayerMask {
    const VALID: u32 = 0x80000000;

    pub fn empty() -> Self {
        Self(NonZeroU32::new(Self::VALID).unwrap())
    }
    pub fn contains_layer(&self, t: LayerType) -> bool {
        assert!((t as usize) < 16);
        self.0.get() & (1 << (t as usize)) != 0
    }
    pub fn contains_mesh(&self, t: MeshType) -> bool {
        assert!((t as usize) < 8);
        self.0.get() & (1 << (t as usize + 16)) != 0
    }
}
impl From<LayerType> for LayerMask {
    fn from(t: LayerType) -> Self {
        assert!((t as usize) < 16);
        Self(NonZeroU32::new(Self::VALID | (1 << (t as usize))).unwrap())
    }
}
impl From<MeshType> for LayerMask {
    fn from(t: MeshType) -> Self {
        assert!((t as usize) < 8);
        Self(NonZeroU32::new(Self::VALID | (1 << (t as usize + 16))).unwrap())
    }
}
impl std::ops::BitOr for LayerMask {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for LayerMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl std::ops::BitAnd for LayerMask {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(NonZeroU32::new(Self::VALID | (self.0.get() & rhs.0.get())).unwrap())
    }
}
impl std::ops::BitAndAssign for LayerMask {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 = NonZeroU32::new(Self::VALID | (self.0.get() & rhs.0.get())).unwrap();
    }
}
impl std::ops::Not for LayerMask {
    type Output = Self;
    fn not(self) -> Self {
        Self(NonZeroU32::new(Self::VALID | !self.0.get()).unwrap())
    }
}

lazy_static! {
    pub(crate) static ref LAYERS_BY_NAME: HashMap<&'static str, LayerType> =
        LayerType::iter().map(|t| (t.name(), t)).collect();
}
