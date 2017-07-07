use rand;
use rand::Rng;
use std::f32::consts::PI;

#[allow(unused)]
fn modulo(a: i64, b: i64) -> usize {
    (((a % b) + b) % b) as usize
}

pub struct Heightmap<T> {
    pub heights: Vec<T>,
    pub width: u16,
    pub height: u16,
}

impl<T> Heightmap<T> {
    #[allow(unused)]
    pub fn new(heights: Vec<T>, width: u16, height: u16) -> Self {
        assert_eq!(heights.len(), (width as usize) * (height as usize));
        Heightmap {
            heights,
            width,
            height,
        }
    }

    pub fn get(&self, x: u16, y: u16) -> Option<T>
        where T: Clone
    {
        if x >= self.width || y >= self.height {
            return None;
        }

        self.heights
            .get(x as usize + y as usize * self.width as usize)
            .cloned()
    }

    #[allow(unused)]
    pub fn get_wrapping(&self, x: i64, y: i64) -> T
        where T: Clone
    {
        let x = modulo(x, self.width as i64);
        let y = modulo(y, self.height as i64);
        self.heights[x + y * self.width as usize].clone()
    }

    /// Produces a Vec of arrays where each array consists of the height at that point followed by
    /// the slope in the x and y directions respectively.
    pub fn as_height_and_slopes(&self, spacing: f32) -> Vec<[f32; 3]>
        where f32: From<T>,
              T: Clone
    {
        let mut result = Vec::with_capacity(self.width as usize * self.height as usize);
        let scale_factor = 0.5 / spacing;

        for y in 0..self.height {
            for x in 0..self.width {
                let mx: f32 = if x > 0 {
                    self.get(x - 1, y).unwrap().into()
                } else {
                    self.get(self.width - 1, y).unwrap().into()
                };
                let px: f32 = if x < self.width - 1 {
                    self.get(x + 1, y).unwrap().into()
                } else {
                    self.get(0, y).unwrap().into()
                };
                let my: f32 = if y > 0 {
                    self.get(x, y - 1).unwrap().into()
                } else {
                    self.get(x, self.height - 1).unwrap().into()
                };
                let py: f32 = if y < self.height - 1 {
                    self.get(x, y + 1).unwrap().into()
                } else {
                    self.get(x, 0).unwrap().into()
                };
                let v = [self.heights[x as usize + y as usize * self.width as usize]
                             .clone()
                             .into(),
                         (mx - px) * scale_factor,
                         (my - py) * scale_factor];
                result.push(v);
            }
        }
        result
    }
}

pub fn perlin_noise(grid_resolution: usize, grid_spacing: usize) -> Heightmap<f32> {
    fn dot(a: (f32, f32), b: (f32, f32)) -> f32 {
        a.0 * b.0 + a.1 * b.1
    }

    fn fade(t: f32) -> f32 {
        t * t * t * (6.0 * t * t - 15.0 * t + 10.0)
    }

    fn interp(a: f32, b: f32, t: f32) -> f32 {
        let t = fade(t);
        a * (1.0 - t) + b * t
    }

    let mut rng = rand::thread_rng();
    let gradients: Vec<(f32, f32)> = (0..(grid_resolution * grid_resolution))
        .map(|_| rng.gen_range(0.0, 2.0 * PI).sin_cos())
        .collect();

    let mut heights =
        vec![-9999.0; grid_resolution * grid_resolution * grid_spacing * grid_spacing];
    for y in 0..grid_resolution {
        for x in 0..grid_resolution {
            let xp = (x + 1) % grid_resolution;
            let yp = (y + 1) % grid_resolution;
            let a = gradients[x + y * grid_resolution];
            let b = gradients[x + yp * grid_resolution];
            let c = gradients[xp + y * grid_resolution];
            let d = gradients[xp + yp * grid_resolution];

            for k in 0..grid_spacing {
                for h in 0..grid_spacing {
                    let v = (h as f32 / grid_spacing as f32, k as f32 / grid_spacing as f32);

                    let ad = dot(a, v);
                    let bd = dot(b, (v.0, v.1 - 1.0));
                    let cd = dot(c, (v.0 - 1.0, v.1));
                    let dd = dot(d, (v.0 - 1.0, v.1 - 1.0));

                    let height = interp(interp(ad, bd, v.1), interp(cd, dd, v.1), v.0);

                    let i = (h + x * grid_spacing) +
                            (k + y * grid_spacing) * grid_resolution * grid_spacing;
                    heights[i] = height;
                }
            }
        }
    }

    Heightmap {
        heights,
        width: (grid_resolution * grid_spacing) as u16,
        height: (grid_resolution * grid_spacing) as u16,
    }
}
