use rand::distributions::Distribution;
use rand_distr::Normal;

pub struct Heightmap<T> {
    pub heights: Vec<T>,
    pub width: u16,
    pub height: u16,
}

/// Evaluate wavelet noise on a grid with the given resolution and grid spacing. ///
/// The output heightmap will have a width and height of `grid_resolution` * `grid_spacing`. Values
/// will have a mean of approximately zero, and a variance of 1.
pub fn wavelet_noise(grid_resolution: usize, grid_spacing: usize) -> Heightmap<f32> {
    // See: https://graphics.pixar.com/library/WaveletNoise/paper.pdf

    fn modulo(x: i32, n: usize) -> usize {
        let m = x % n as i32;
        if m < 0 {
            (m + n as i32) as usize
        } else {
            m as usize
        }
    }
    fn downsample(from: &[f32], to: &mut [f32], n: usize, stride: usize) {
        const ARAD: i32 = 16;
        #[cfg_attr(rustfmt, rustfmt_skip)]
        const COEFFS: [f32; 32] = [
            0.000334,-0.001528, 0.000410, 0.003545,-0.000938,-0.008233, 0.002172, 0.019120,
            -0.005040,-0.044412, 0.011655, 0.103311,-0.025936,-0.243780, 0.033979, 0.655340,
            0.655340, 0.033979,-0.243780,-0.025936, 0.103311, 0.011655,-0.044412,-0.005040,
            0.019120, 0.002172,-0.008233,-0.000938, 0.003546, 0.000410,-0.001528, 0.000334
        ];
        for i in 0..(n / 2) {
            to[i * stride] = 0.0;
            let min_k = 2 * (i as i32) - ARAD;
            let max_k = 2 * (i as i32) + ARAD;
            for k in min_k..(max_k) {
                let index = (ARAD + k - 2 * i as i32) as usize;
                to[i * stride] += COEFFS[index] * from[modulo(k, n) * stride];
            }
        }
    }
    fn upsample(from: &[f32], to: &mut [f32], n: usize, stride: usize) {
        let p_coeffs = [0.25, 0.75, 0.75, 0.25];
        for i in 0..n {
            to[i * stride] = 0.0;
            for k in (i / 2)..(i / 2 + 2) {
                to[i * stride] += p_coeffs[2 + i - 2 * k] * from[(k % (n / 2)) * stride];
            }
        }
    }
    fn generate_noise_tile(n: usize) -> Vec<f32> {
        assert!(n % 2 == 0); // size must be even!

        let mut temp1 = vec![0.0; n * n];
        let mut temp2 = vec![0.0; n * n];
        let mut noise = Vec::new();

        // Step 1. Fill the tile with random numbers in the range -1 to 1.
        let normal = Normal::new(0.0, 1.0).unwrap();
        for _ in 0..(n * n) {
            noise.push(normal.sample(&mut rand::thread_rng()) as f32);
        }

        // Steps 2 and 3. Downsample and upsample the tile
        for iy in 0..n {
            // each x row
            let i = iy * n;
            downsample(&noise[i..], &mut temp1[i..], n, 1);
            upsample(&temp1[i..], &mut temp2[i..], n, 1);
        }
        for ix in 0..n {
            // each y row
            let i = ix;
            downsample(&temp2[i..], &mut temp1[i..], n, n);
            upsample(&temp1[i..], &mut temp2[i..], n, n);
        }

        // Step 4. Subtract out the coarse-scale contribution
        for i in 0..(n * n) {
            noise[i] -= temp2[i];
        }
        // Avoid even/odd variance difference by adding odd-offset version of noise to itself.
        let mut offset = n / 2;
        if offset % 2 == 0 {
            offset += 1;
        }
        let mut i = 0;
        for ix in 0..n {
            for iy in 0..n {
                temp1[i] = noise[((ix + offset) % n) + ((iy + offset) % n) * n];
                i += 1;
            }
        }
        for i in 0..(n * n) {
            noise[i] += temp1[i];
        }

        noise
    }

    /* Non-projected 2D noise */
    fn noise(noise_tile: &[f32], n: usize, p: [f32; 2]) -> f32 {
        let mut mid = [0; 2];
        let mut w = [[0.0; 3]; 2];

        /* Evaluate quadratic B-spline basis functions */
        for i in 0..2 {
            let center = p[i] - 0.5;
            let t = center.ceil() - center;

            mid[i] = center.ceil() as i32;
            w[i][0] = 0.5 * t * t;
            w[i][2] = 0.5 * (1.0 - t) * (1.0 - t);
            w[i][1] = 1.0 - w[i][0] - w[i][2];
        }
        /* Evaluate noise by weighting noise coefficients by basis function values */
        let mut result = 0.0;
        for fx in 0..3 {
            for fy in 0..3 {
                let cx = modulo(mid[0] + (fx as i32 - 1), n);
                let cy = modulo(mid[1] + (fy as i32 - 1), n);
                let weight = w[0][fx] * w[1][fy];
                result += weight * noise_tile[cx + cy * n];
            }
        }
        result
    }

    let noise_tile = generate_noise_tile(grid_resolution);

    let mut heights = Vec::new();
    for x in 0..(grid_resolution * grid_spacing) {
        for y in 0..(grid_resolution * grid_spacing) {
            let p = [x as f32 / grid_spacing as f32, y as f32 / grid_spacing as f32];
            heights.push(noise(&noise_tile, grid_resolution, p))
        }
    }

    // Force mean = 0.
    let mean = heights.iter().sum::<f32>() / heights.len() as f32;
    for h in &mut heights {
        *h -= mean;
    }

    // Force variance = 1.
    let variance = heights.iter().map(|n| n * n).sum::<f32>() / heights.len() as f32;
    let inv_variance = 1.0 / variance;
    for h in &mut heights {
        *h *= inv_variance;
    }

    Heightmap {
        heights,
        width: (grid_resolution * grid_spacing) as u16,
        height: (grid_resolution * grid_spacing) as u16,
    }
}
