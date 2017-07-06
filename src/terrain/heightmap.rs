use rand;
use rand::Rng;

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

pub fn detail_heightmap(width: u16, height: u16) -> Heightmap<f32> {
    let mut rng = rand::thread_rng();
    let mut heights = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            if x % 2 == 0 && y % 2 == 0 {
                heights.push(0f32)
            } else {
                heights.push(rng.gen_range(-1f32, 1f32))
            }
        }
    }

    Heightmap {
        heights,
        width,
        height,
    }
}
