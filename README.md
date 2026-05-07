# Geometrize

[GitHub](https://github.com/jac0-b/geometrize), [Docs.rs](https://docs.rs/geometrize), [crates.io](https://crates.io/crates/geometrize)

Transform images into geometric art.

This crate provides two rendering styles:
- **Low-poly**: Decomposes an image into colored triangles via Delaunay triangulation.
- **Pointillist**: Renders the same triangulation as overlapping circles.

## Installation

```
cargo add geometrize
```

## Quick Start
Images are transformed using the `geometrize` function.
```rust
use geometrize::{geometrize, Style, SamplingParams};
use image::{open, RgbaImage};

let image = open("launch.jpg").unwrap();

let lowpoly: RgbaImage = geometrize(
    &image,
    Style::Lowpoly,
    70_000,
    SamplingParams::default()
).unwrap();
lowpoly.save("lowpoly_70k.png").unwrap();

let pointillist: RgbaImage = geometrize(
    &image,
    Style::Pointillist {noise: 0.0},
    20_000,
    SamplingParams::default()
).unwrap();
pointillist.save("pointillist_20k.png").unwrap();
```

View images at [Docs.rs](https://docs.rs/geometrize#quick-start)

## Transparent images

This also works with transparent images.

```rust
let image = open("dice.png").unwrap();

let lowpoly: RgbaImage = geometrize(
    &image,
    Style::Lowpoly,
    50_000,
    SamplingParams::default()
).unwrap();
lowpoly.save("lowpoly_50k_dice.png").unwrap();

let pointillist: RgbaImage = geometrize(
    &image,
    Style::Pointillist {noise: 0.0},
    10_000,
    SamplingParams::default()
).unwrap();
pointillist.save("pointillist_10k_dice.png").unwrap();
```

View images at [Docs.rs](https://docs.rs/geometrize#transparent-images)
