#![cfg_attr(test, feature(test))]

#[cfg(test)]
extern crate test;

use cgmath::Vector2;
use std::collections::VecDeque;
use std::io::{Cursor, Read, Write};

pub fn compress_heightmap_tile(
    resolution: usize,
    log2_scale_factor: i8,
    heights: &[i16],
    parent: Option<(Vector2<i32>, usize, &[i16])>, // (parent_offset, skirt, parent_heights)
    compression_level: u32,
) -> Vec<u8> {
    assert_eq!(resolution % 2, 1);
    let half_resolution = resolution / 2 + 1;
    let mut output = Vec::new();
    let mut grid = Vec::new();

    assert!(log2_scale_factor >= 0 && log2_scale_factor <= 12);
    let scale_factor = 1i16 << log2_scale_factor;
    let inv_scale_factor = 1.0 / scale_factor as f32;

    match parent {
        None => {
            for y in 0..half_resolution {
                for x in 0..half_resolution {
                    output.push(
                        (heights[(x * 2) + (y * 2) * resolution] as f32 * inv_scale_factor).round()
                            as i16,
                    );
                }
            }
            grid = output.clone();
        }
        Some((parent_offset, skirt, parent_heights)) => {
            let offset: Vector2<usize> = Vector2::new(skirt / 2, skirt / 2)
                + parent_offset.cast().unwrap() * ((resolution - 2 * skirt) / 2);
            for y in 0..half_resolution {
                for x in 0..half_resolution {
                    let height = heights[(x * 2) + (y * 2) * resolution];
                    let pheight = parent_heights[(offset.x + x) + (offset.y + y) * resolution];
                    let delta =
                        (height.wrapping_sub(pheight) as f32 * inv_scale_factor).round() as i16;
                    output.push(delta);
                    grid.push(pheight.wrapping_add(delta * scale_factor));
                }
            }
        }
    }

    for y in (0..resolution).step_by(2) {
        for x in (1..resolution).step_by(2) {
            let interpolated = (grid[(x - 1) / 2 + y * half_resolution / 2] as i32
                + grid[(x + 1) / 2 + y * half_resolution / 2] as i32)
                / 2;
            output.push(
                (heights[x + y * resolution].wrapping_sub(interpolated as i16) as f32
                    * inv_scale_factor)
                    .round() as i16,
            );
        }
    }

    for y in (1..resolution).step_by(2) {
        for x in (0..resolution).step_by(2) {
            let interpolated = (grid[x / 2 + (y - 1) * half_resolution / 2] as i32
                + grid[x / 2 + (y + 1) * half_resolution / 2] as i32)
                / 2;
            output.push(
                (heights[x + y * resolution].wrapping_sub(interpolated as i16) as f32
                    * inv_scale_factor)
                    .round() as i16,
            );
        }
    }

    for y in (1..resolution).step_by(2) {
        for x in (1..resolution).step_by(2) {
            let interpolated = (grid[(x - 1) / 2 + (y - 1) * half_resolution / 2] as i32
                + grid[(x + 1) / 2 + (y - 1) * half_resolution / 2] as i32
                + grid[(x - 1) / 2 + (y + 1) * half_resolution / 2] as i32
                + grid[(x + 1) / 2 + (y + 1) * half_resolution / 2] as i32)
                / 4;
            output.push(
                (heights[x + y * resolution].wrapping_sub(interpolated as i16) as f32
                    * inv_scale_factor)
                    .round() as i16,
            );
        }
    }

    // let mut v = vec![0; output.len() * 2+2];
    // v[0] = 1;
    // v[1] = 8;
    // v[2..].copy_from_slice(bytemuck::cast_slice(&output));
    // v
    // let mut e = flate2::write::ZlibEncoder::new(vec![0, 8], flate2::Compression::default());
    // e.write_all(bytemuck::cast_slice(&output)).unwrap();
    // e.finish().unwrap()

    let mut header = vec![3, log2_scale_factor as u8, b'T', b'E', b'R', b'R', b'A', b'!'];
    header.extend_from_slice(&(resolution as u32).to_le_bytes());

    let mut e = lz4::EncoderBuilder::new().level(compression_level).build(header).unwrap();
    e.write_all(bytemuck::cast_slice(&output)).unwrap();
    e.finish().0
}

pub fn uncompress_heightmap_tile(
    parent: Option<(Vector2<i32>, usize, usize, &[i16])>,
    bytes: &[u8],
) -> (usize, Vec<i16>) {
    let scale_factor;
    let header_end;
    let resolution;

    if bytes[0] == 1 {
        scale_factor = bytes[1] as i16;
        resolution = 521;
        header_end = 2;
    } else if bytes[0] == 2 {
        assert!(bytes[1] as i8 >= 0 && bytes[1] <= 12);
        scale_factor = 1i16 << bytes[1];
        resolution = 521;
        header_end = 2;
    } else if bytes[0] == 3 {
        assert!(bytes[1] as i8 >= 0 && bytes[1] <= 12);
        assert_eq!(&bytes[2..8], b"TERRA!");
        scale_factor = 1i16 << bytes[1];
        resolution = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        header_end = 12;
    } else {
        panic!("unknown heightmap tile version.");
    };

    let mut encoded = vec![0i16; resolution * resolution];
    // flate2::read::ZlibDecoder::new(Cursor::new(&bytes[2..]))
    //     .read_exact(bytemuck::cast_slice_mut(&mut encoded))
    //     .unwrap();
    lz4::Decoder::new(Cursor::new(&bytes[header_end..]))
        .unwrap()
        .read_exact(bytemuck::cast_slice_mut(&mut encoded))
        .unwrap();
    // encoded.copy_from_slice(bytemuck::cast_slice(&bytes[2..]));

    let encoded: VecDeque<i16> = encoded.into();
    let mut encoded = encoded.into_iter();

    assert_eq!(resolution % 2, 1);
    let half_resolution = resolution / 2 + 1;
    let mut heights = vec![0; resolution * resolution];

    let mut q_0 = vec![1234; (resolution / 2 + 1) * (resolution / 2 + 1)];
    let mut q_1 = vec![1234; (resolution / 2 + 1) * (resolution / 2)];
    let mut q_2 = vec![1234; (resolution / 2) * (resolution / 2 + 1)];
    let mut q_3 = vec![1234; (resolution / 2) * (resolution / 2)];

    if let Some((parent_offset, skirt, parent_resolution, parent_heights)) = parent {
        let offset: Vector2<usize> = Vector2::new(skirt / 2, skirt / 2)
            + parent_offset.cast().unwrap() * ((parent_resolution - 2 * skirt) / 2);
        for (y, row) in q_0.chunks_exact_mut(half_resolution).enumerate() {
            let parent_row = &parent_heights[(offset.x + (offset.y + y) * parent_resolution)..]
                [..half_resolution];
            for ((h, r), p) in row.iter_mut().zip(&mut encoded).zip(parent_row) {
                *h = p.wrapping_add(r.wrapping_mul(scale_factor));
            }
        }
    } else {
        for (h, r) in q_0.iter_mut().zip(&mut encoded) {
            *h = r.wrapping_mul(scale_factor);
        }
    }

    for (y, row) in q_1.chunks_exact_mut(resolution / 2).enumerate() {
        for (x, (h, r)) in row.iter_mut().zip(&mut encoded).enumerate() {
            let e = x + y * half_resolution;
            let interpolated = (q_0[e] as i32 + q_0[e + 1] as i32) / 2;
            *h = r.wrapping_mul(scale_factor).wrapping_add(interpolated as i16);
        }
    }

    for (y, row) in q_2.chunks_exact_mut(resolution / 2 + 1).enumerate() {
        for (x, (h, r)) in row.iter_mut().zip(&mut encoded).enumerate() {
            let e = x + y * half_resolution;
            let interpolated = (q_0[e] as i32 + q_0[e + half_resolution] as i32) / 2;
            *h = r.wrapping_mul(scale_factor).wrapping_add(interpolated as i16);
        }
    }

    for (y, row) in q_3.chunks_exact_mut(resolution / 2).enumerate() {
        for (x, (h, r)) in row.iter_mut().zip(&mut encoded).enumerate() {
            let e = x + y * half_resolution;
            let interpolated = (q_0[e] as i32
                + q_0[e + 1] as i32
                + q_0[e + half_resolution] as i32
                + q_0[e + half_resolution + 1] as i32)
                / 4;
            *h = r.wrapping_mul(scale_factor).wrapping_add(interpolated as i16);
        }
    }

    assert!(encoded.next().is_none());

    for (y, row) in heights.chunks_exact_mut(resolution).enumerate() {
        if y % 2 == 0 {
            let q0_row = &q_0[(y / 2) * (resolution / 2 + 1)..][..(resolution / 2 + 1)];
            let q1_row = &q_1[(y / 2) * (resolution / 2)..][..(resolution / 2)];
            for (h, (&q0, &q1)) in row.chunks_exact_mut(2).zip(q0_row.iter().zip(q1_row)) {
                h[0] = q0;
                h[1] = q1;
            }
            row[resolution - 1] = q0_row[(resolution - 1) / 2];
        } else {
            let q2_row = &q_2[(y / 2) * (resolution / 2 + 1)..][..(resolution / 2 + 1)];
            let q3_row = &q_3[(y / 2) * (resolution / 2)..][..(resolution / 2)];
            for (h, (&q2, &q3)) in row.chunks_exact_mut(2).zip(q2_row.iter().zip(q3_row)) {
                h[0] = q2;
                h[1] = q3;
            }
            row[resolution - 1] = q2_row[(resolution - 1) / 2];
        }
    }

    assert_eq!(heights[0], q_0[0]);
    assert_eq!(heights[1], q_1[0]);
    assert_eq!(heights[2], q_0[1]);
    assert_eq!(heights[resolution], q_2[0]);
    assert_eq!(heights[resolution + 1], q_3[0]);
    assert_eq!(heights[resolution + 2], q_2[1]);

    (resolution, heights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::{Distribution, Uniform};
    use test::Bencher;

    #[test]
    fn compress_roundtrip() {
        let skirt = 8;
        let resolution = 513 + skirt * 2;

        let mut rng = rand::thread_rng();
        let dist = Uniform::from(-1000..8000);

        let parent: Vec<i16> =
            (0..(resolution * resolution)).map(|_| dist.sample(&mut rng)).collect();
        let child: Vec<i16> =
            (0..(resolution * resolution)).map(|_| dist.sample(&mut rng)).collect();

        let bytes = compress_heightmap_tile(
            resolution,
            3,
            &*child,
            Some((Vector2::new(0, 0), skirt, &*parent)),
            9,
        );
        let roundtrip = uncompress_heightmap_tile(
            Some((Vector2::new(0, 0), skirt, resolution, &*parent)),
            &*bytes,
        )
        .1;

        for i in 0..(resolution * resolution) {
            assert!(
                (roundtrip[i] - child[i]).abs() <= 4,
                "i={}: roundtrip={} child={}",
                i,
                roundtrip[i],
                child[i]
            );
        }
    }

    #[bench]
    fn bench_compress(b: &mut Bencher) {
        let skirt = 8;
        let resolution = 513 + skirt * 2;

        let mut rng = rand::thread_rng();
        let dist = Uniform::from(-1000..8000);

        let parent: Vec<i16> =
            (0..(resolution * resolution)).map(|_| dist.sample(&mut rng)).collect();
        let child: Vec<i16> =
            (0..(resolution * resolution)).map(|_| dist.sample(&mut rng)).collect();

        b.iter(|| {
            compress_heightmap_tile(
                resolution,
                3,
                &*child,
                Some((Vector2::new(0, 0), skirt, &*parent)),
                9,
            )
        });
    }

    #[bench]
    fn bench_uncompress(b: &mut Bencher) {
        let skirt = 8;
        let resolution = 513 + skirt * 2;

        let mut rng = rand::thread_rng();
        let dist = Uniform::from(-1000..8000);

        let parent: Vec<i16> =
            (0..(resolution * resolution)).map(|_| dist.sample(&mut rng)).collect();
        let child: Vec<i16> =
            (0..(resolution * resolution)).map(|_| dist.sample(&mut rng)).collect();

        let bytes = compress_heightmap_tile(
            resolution,
            3,
            &*child,
            Some((Vector2::new(0, 0), skirt, &*parent)),
            9,
        );
        b.iter(|| {
            uncompress_heightmap_tile(
                Some((Vector2::new(0, 0), skirt, resolution, &*parent)),
                &*bytes,
            )
        });
    }
}
