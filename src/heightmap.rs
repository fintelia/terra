
pub struct Heightmap<T> {
    pub heights: Vec<T>,
    pub width: u16,
    pub height: u16,
}

impl<T> Heightmap<T> {
    pub fn new(heights: Vec<T>, width: u16, height: u16) -> Self {
        assert_eq!(heights.len(), (width as usize) * (height as usize));
        Heightmap {
            heights: heights,
            width: width,
            height: height,
        }
    }
}
