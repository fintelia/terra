pub struct VFace(pub u8);
pub struct VSector(pub VFace, pub u8, pub u8);

impl std::fmt::Display for VFace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "{}",
            match self.0 {
                0 => "0E",
                1 => "180E",
                2 => "90E",
                3 => "90W",
                4 => "N",
                5 => "S",
                _ => unreachable!(),
            }
        )
    }
}

impl std::fmt::Display for VSector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "S-{}-x{:03}-y{:03}", self.0, self.1, self.2)
    }
}
