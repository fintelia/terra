
use clipmap::*;

pub fn generate(resolution: i8, include_center: bool) -> Vec<Vertex> {
    let mut vertices = Vec::new();

    let cstart = (resolution + 1) / 4;
    let cend = cstart + (resolution + 1) / 2 - 2;

    let edge = resolution - 2;
    let even = |v| v % 2 == 0;

    for x in 0..(resolution - 1) {
        for y in 0..(resolution - 1) {
            if !include_center && x >= cstart && x <= cend && y >= cstart && y <= cend {
                continue;
            }

            if x == 0 && even(y) {
                if y != 0 {
                    vertices.push(Vertex { pos: [x, y] });
                    vertices.push(Vertex { pos: [x + 1, y] });
                    vertices.push(Vertex { pos: [x + 1, y + 1] });
                }
                if y != edge - 1 {
                    vertices.push(Vertex { pos: [x, y + 2] });
                    vertices.push(Vertex { pos: [x + 1, y + 2] });
                    vertices.push(Vertex { pos: [x + 1, y + 1] });
                }
                vertices.push(Vertex { pos: [x, y] });
                vertices.push(Vertex { pos: [x + 1, y + 1] });
                vertices.push(Vertex { pos: [x, y + 2] });
            }

            if y == 0 && even(x) {
                if x != 0 {
                    vertices.push(Vertex { pos: [x, y] });
                    vertices.push(Vertex { pos: [x, y + 1] });
                    vertices.push(Vertex { pos: [x + 1, y + 1] });
                }
                if x != edge - 1 {
                    vertices.push(Vertex { pos: [x + 2, y] });
                    vertices.push(Vertex { pos: [x + 2, y + 1] });
                    vertices.push(Vertex { pos: [x + 1, y + 1] });
                }
                vertices.push(Vertex { pos: [x, y] });
                vertices.push(Vertex { pos: [x + 1, y + 1] });
                vertices.push(Vertex { pos: [x + 2, y] });
            }

            if x == edge && even(y) {
                if y != 0 {
                    vertices.push(Vertex { pos: [x, y] });
                    vertices.push(Vertex { pos: [x, y + 1] });
                    vertices.push(Vertex { pos: [x + 1, y] });
                }
                if y != edge - 1 {
                    vertices.push(Vertex { pos: [x, y + 1] });
                    vertices.push(Vertex { pos: [x, y + 2] });
                    vertices.push(Vertex { pos: [x + 1, y + 2] });
                }
                vertices.push(Vertex { pos: [x + 1, y] });
                vertices.push(Vertex { pos: [x, y + 1] });
                vertices.push(Vertex { pos: [x + 1, y + 2] });
            }

            if y == edge && even(x) {
                if x != 0 {
                    vertices.push(Vertex { pos: [x, y] });
                    vertices.push(Vertex { pos: [x + 1, y] });
                    vertices.push(Vertex { pos: [x, y + 1] });
                }
                if x != edge - 1 {
                    vertices.push(Vertex { pos: [x + 1, y] });
                    vertices.push(Vertex { pos: [x + 2, y] });
                    vertices.push(Vertex { pos: [x + 2, y + 1] });
                }
                vertices.push(Vertex { pos: [x, y + 1] });
                vertices.push(Vertex { pos: [x + 1, y] });
                vertices.push(Vertex { pos: [x + 2, y + 1] });
            }

            if x != 0 && y != 0 && x != edge && y != edge {
                vertices.push(Vertex { pos: [x, y] });
                vertices.push(Vertex { pos: [x, y + 1] });
                vertices.push(Vertex { pos: [x + 1, y] });

                vertices.push(Vertex { pos: [x + 1, y] });
                vertices.push(Vertex { pos: [x, y + 1] });
                vertices.push(Vertex { pos: [x + 1, y + 1] });
            }
        }
    }

    vertices
}
