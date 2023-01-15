use anyhow::Error;
use ktx2::Format;

pub(crate) fn encode_ktx2(
    image_slices: &[Vec<u8>],
    width: u32,
    height: u32,
    depth: u32,
    layers: u32,
    format: ktx2::Format,
) -> Result<Vec<u8>, Error> {
    let samples: u16 = match format {
        Format::R8_UNORM | Format::R32_SFLOAT => 1,
        Format::R8G8_UNORM | Format::R32G32_SFLOAT => 2,
        Format::R8G8B8A8_UNORM | Format::R32G32B32A32_SFLOAT => 4,
        Format::BC1_RGB_UNORM_BLOCK
        | Format::BC4_UNORM_BLOCK
        | Format::BC7_UNORM_BLOCK
        | Format::ASTC_4x4_UNORM_BLOCK => 1,
        Format::BC5_UNORM_BLOCK => 2,
        _ => unimplemented!("{:?}", format),
    };
    let (compressed, bytes_per_block) = match format {
        Format::R8_UNORM | Format::R8G8_UNORM | Format::R8G8B8A8_UNORM => (false, samples),
        Format::R32_SFLOAT | Format::R32G32_SFLOAT | Format::R32G32B32A32_SFLOAT => {
            (false, 4 * samples)
        }
        Format::BC1_RGB_UNORM_BLOCK | Format::BC4_UNORM_BLOCK => (true, 8),
        Format::BC7_UNORM_BLOCK | Format::ASTC_4x4_UNORM_BLOCK | Format::BC5_UNORM_BLOCK => {
            (true, 16)
        }
        _ => unimplemented!(),
    };
    let levels = image_slices.len() as u32;
    let dfd_size = 28u32 + 16 * samples as u32;

    let mut contents = Vec::new();
    contents.extend_from_slice(&[
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
    ]);
    contents.extend_from_slice(&format.0.get().to_le_bytes());
    contents.extend_from_slice(&1u32.to_le_bytes());
    contents.extend_from_slice(&width.to_le_bytes());
    contents.extend_from_slice(&height.to_le_bytes());
    contents.extend_from_slice(&depth.to_le_bytes());
    contents.extend_from_slice(&layers.to_le_bytes());
    contents.extend_from_slice(&1u32.to_le_bytes()); // faces
    contents.extend_from_slice(&levels.to_le_bytes()); // levels
    contents.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme

    contents.extend_from_slice(&(80 + 24 * levels).to_le_bytes());
    contents.extend_from_slice(&dfd_size.to_le_bytes());
    contents.extend_from_slice(&0u32.to_le_bytes()); // kvdByteOffset
    contents.extend_from_slice(&0u32.to_le_bytes()); // kvdByteLength
    contents.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
    contents.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength

    let mut offset = (80 + 24 * levels + dfd_size) as u64;

    assert_eq!(contents.len(), 80);
    for image_slice in image_slices {
        contents.extend_from_slice(&offset.to_le_bytes());
        contents.extend_from_slice(&(image_slice.len() as u64).to_le_bytes());
        contents.extend_from_slice(&(image_slice.len() as u64).to_le_bytes());
        offset += image_slice.len() as u64;
    }

    assert_eq!(contents.len(), 80 + 24 * levels as usize);
    contents.extend_from_slice(&dfd_size.to_le_bytes());
    contents.extend_from_slice(&0u32.to_le_bytes()); // vendor ID + descriptor type
    contents.extend_from_slice(&2u16.to_le_bytes()); // version number
    contents.extend_from_slice(&(24u16 + 16 * samples).to_le_bytes()); // descriptor block size
    contents.push(1); // model
    contents.push(1); // color primaries
    contents.push(1); // transfer function (1 = linear, 2 = sRGB)
    contents.push(0); // flags (1 = premultiplied alpha)
    contents.push(if compressed { 3 } else { 0 }); // texel block dimension0 (0 = 1x1 block, 3 = 4x4 block)
    contents.push(if compressed { 3 } else { 0 }); // texel block dimension1 (0 = 1x1 block, 3 = 4x4 block)
    contents.push(0); // texel block dimension2
    contents.push(0); // texel block dimension3
    contents.push(bytes_per_block as u8); // bytes plane0
    contents.push(0); // bytes plane1
    contents.push(0); // bytes plane2
    contents.push(0); // bytes plane3
    contents.push(0); // bytes plane4
    contents.push(0); // bytes plane5
    contents.push(0); // bytes plane6
    contents.push(0); // bytes plane7

    match format {
        Format::R8_UNORM | Format::R8G8_UNORM | Format::R8G8B8A8_UNORM => {
            for i in 0..samples {
                contents.extend_from_slice(&(i as u16 * 8).to_le_bytes()); // bitOffset
                contents.push(7); // bitLength
                contents.push(if i == 3 { 0x1F } else { i as u8 }); // channelType + F[loat] + S[igned] + E[ponent] + L[inear]
                contents.extend_from_slice(&[0; 4]); // samplePosition[0..3]
                contents.extend_from_slice(&0u32.to_le_bytes()); // sampleLower
                contents.extend_from_slice(&255u32.to_le_bytes()); // sampleUpper
            }
        }
        Format::R32_SFLOAT | Format::R32G32_SFLOAT | Format::R32G32B32A32_SFLOAT => {
            for i in 0..samples {
                contents.extend_from_slice(&(i as u16 * 32).to_le_bytes()); // bitOffset
                contents.push(31); // bitLength
                contents.push(if i == 3 { 0b1101_1111 } else { 0b1100_0000 | i as u8 }); // channelType + F[loat] + S[igned] + E[ponent] + L[inear]
                contents.extend_from_slice(&[0; 4]); // samplePosition[0..3]
                contents.extend_from_slice(&0u32.to_le_bytes()); // sampleLower
                contents.extend_from_slice(&u32::MAX.to_le_bytes()); // sampleUpper
            }
        }
        Format::BC1_RGB_UNORM_BLOCK
        | Format::BC4_UNORM_BLOCK
        | Format::BC7_UNORM_BLOCK
        | Format::ASTC_4x4_UNORM_BLOCK => {
            contents.extend_from_slice(&0u16.to_le_bytes()); // bitOffset
            contents.push(bytes_per_block as u8 * 8 - 1); // bitLength
            contents.push(0); // channelType + F[loat] + S[igned] + E[ponent] + L[inear]
            contents.extend_from_slice(&[0; 4]); // samplePosition[0..3]
            contents.extend_from_slice(&0u32.to_le_bytes()); // sampleLower
            contents.extend_from_slice(&u32::MAX.to_le_bytes()); // sampleUpper
        }
        Format::BC5_UNORM_BLOCK => {
            for i in 0..2 {
                contents.extend_from_slice(&(i as u16 * 64).to_le_bytes()); // bitOffset
                contents.push(63); // bitLength
                contents.push(i); // channelType + F[loat] + S[igned] + E[ponent] + L[inear]
                contents.extend_from_slice(&[0; 4]); // samplePosition[0..3]
                contents.extend_from_slice(&0u32.to_le_bytes()); // sampleLower
                contents.extend_from_slice(&u32::MAX.to_le_bytes()); // sampleUpper
            }
        }
        _ => unimplemented!(),
    }

    if contents.len() % 4 != 0 {
        contents.resize((contents.len() & !3) + 4, 0);
    }

    assert_eq!(contents.len(), 80 + 24 * levels as usize + dfd_size as usize);
    for image_slice in image_slices {
        contents.extend_from_slice(image_slice);
    }

    Ok(contents)
}
