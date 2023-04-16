# Terra [![crates.io](https://img.shields.io/crates/v/terra.svg)](https://crates.io/crates/terra) [![docs.rs](https://docs.rs/terra/badge.svg)](https://docs.rs/terra) [![Github CI](https://img.shields.io/github/workflow/status/fintelia/terra/Rust)](https://github.com/fintelia/terra/actions?query=workflow%3ARust)

Terra is work in progress large scale terrain rendering library built on top of
[wgpu](https://github.com/gfx-rs/wgpu).

<img src="https://terra.fintelia.io/file/terra-tiles/screenshots/merged.png" />

# Overview

Terra supports rendering an entire planet with details ranging in scale from
thousands of kilometers down to centimeters. In Terra, terrain is treated as a
[heightmap](https://en.wikipedia.org/wiki/Heightmap) along with a collection of
texture maps storing the surface normal, albedo, etc.

All of this information can take quite a bit of space, so it isn't included in
this repository. Instead, the necessary files are streamed from the internet at
runtime and cached locally in a subdirectory with the current user's [cache
directory](https://docs.rs/dirs/3.0.1/dirs/fn.cache_dir.html) (which for
instance defaults to `~/.cache/terra` on Linux).

### Level of detail (LOD)

To ensure smooth frame rates and avoid noticable "LOD popping", Terra internally
uses sphere mapped version of the [Continuous Distance-Dependent Level of
Detail](https://github.com/fstrugar/CDLOD/blob/master/cdlod_paper_latest.pdf)
algorithm.

### Incremental Generation

Terra works by streaming coarse grained tiles containing terrain attributes and then
adding fractal details using wgpu compute shaders.

# Getting Started

[Install Rust](https://rustup.rs/) and then the other needed dependencies:

```bash
$ sudo apt-get install libegl-dev libudev-dev build-essential libssl-dev cmake
```

Running should be as simple as:

```bash
git clone git@github.com:fintelia/terra && cd terra
cargo run --release
```

The first time you run Terra, it may take a minute or two to stream the necessary
files. Don't worry if you have to kill the process part way through, on subsequent
runs it will resume where it left off.

Once that step is done, you should see the main Terra window. You can navigate
with the arrow keeps, and increase/decrease your altitude via the Space and Z
keys respectively. Joystick controls are also supported if one is detected. To
exit, press Escape.

You can also pass `--help` to see some other command line options.

### System Requirements

* Windows or Linux operating system (Terra may work on MacOS but this hasn't been tested)
* A fast internet connection
* GPU with 2+ GB of VRAM

# Data Sources / Credits

During operation, this library downloads and merges datasets from a variety of sources. If you integrate
it into your own project, please be sure to give proper credit to all of the following as applicable.

| Kind | Source |
| --- | --- |
| Elevation | [ETOPO1 Global Relief Model](https://www.ngdc.noaa.gov/mgg/global)
| Elevation | [NASA Digital Elevation Model](https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.032021.4326.2)
| Orthoimagery | [Blue Marble Next Generation](https://visibleearth.nasa.gov/view.php?id=76487)
| Stars | [Yale Bright Star Catalog](http://tdc-www.harvard.edu/catalogs/bsc5.html)
| Treecover | [Global Forest Change](https://data.globalforestwatch.org/documents/134f92e59f344549947a3eade9d80783/explore)
| Trees | [SpeedTree](https://store.speedtree.com/)
| Ground Textures | [FreePBR](https://freepbr.com/)
