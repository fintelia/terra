# Terra [![crates.io](https://img.shields.io/crates/v/terra.svg)](https://crates.io/crates/terra) [![docs.rs](https://docs.rs/terra/badge.svg)](https://docs.rs/terra)
Terra is a terrain rendering library built on top of [gfx](https://github.com/gfx-rs/gfx). It uses a quadtree representation that incorporates continuous level of detail to render highly detailed large scale terrains. Data is sourced from either the STRM 30m dataset or USGS's National Elevation Dataset and then enhanced using fractal refinement.

![Screenshot](/screenshot.png?raw=true)

# Status
Terrain:
- [x] CDLOD quadtree implementation
- [x] Oceans

Foliage:
- [ ] Grass
- [ ] Trees

Atmosphere:
- [x] Fog rendering
- [ ] Aerial perspective
- [ ] Multiple scattering
