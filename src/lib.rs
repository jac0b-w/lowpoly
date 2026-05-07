//! # Geometrize
//!
//! Transform images into geometric art.
//!
//! This crate provides two rendering styles:
//! - **Low-poly**: Decomposes an image into colored triangles via Delaunay triangulation.
//! - **Pointillist**: Renders the same triangulation as overlapping circles.
//!
//! # Quick Start
//! Images are transformed using the [`geometrize`] function.
//! ```
//! use geometrize::{geometrize, Style, SamplingParams};
//! use image::{open, RgbaImage};
//!
//! let image = open("launch.jpg").unwrap();
//! # let image = open("./tests/fixtures/launch.jpg").unwrap();
//! 
//! let lowpoly: RgbaImage = geometrize(
//!     &image,
//!     Style::Lowpoly,
//!     70_000,
//!     SamplingParams::default()
//! ).unwrap();
//! lowpoly.save("lowpoly_70k.jpg").unwrap();
//!
//! let pointillist: RgbaImage = geometrize(
//!     &image,
//!     Style::Pointillist {noise: 0.0},
//!     20_000,
//!     SamplingParams::default()
//! ).unwrap();
//! pointillist.save("pointillist_20k.jpg").unwrap();
//! ```
//!
//! <div style="display:flex; gap:0.5%;">
//!   <figure style="width:33%; margin:0;">
//!     <img src="https://github.com/jac0-b/geometrize/blob/990b60fbee240eb5f8bf885198ad6d8f0ef60931/tests/fixtures/launch.jpg?raw=true" style="width:100%;">
//!     <figcaption>launch.jpg</figcaption>
//!   </figure>
//!
//!   <figure style="width:33%; margin:0;">
//!     <img src="https://github.com/jac0-b/geometrize/blob/0140865afdb167904d2feb58f82eef07478130d7/images/lowpoly_70k_launch.jpg?raw=true" style="width:100%;">
//!     <figcaption>lowpoly_70k.jpg</figcaption>
//!   </figure>
//!   <figure style="width:33%; margin:0;">
//!     <img src="https://github.com/jac0-b/geometrize/blob/0140865afdb167904d2feb58f82eef07478130d7/images/pointillist__20k_launch.jpg?raw=true" style="width:100%;">
//!     <figcaption>pointillist_20k.jpg</figcaption>
//!   </figure>
//! </div>
//!
//! # Transparent images
//!
//! This also works with transparent images.
//!
//! ```
//! let image = open("dice.png").unwrap();
//! # let image = open("tests/fixtures/dice.png").unwrap();
//!
//! let lowpoly: RgbaImage = geometrize(
//!     &image,
//!     Style::Lowpoly,
//!     50_000,
//!     SamplingParams::default()
//! ).unwrap();
//! lowpoly.save("lowpoly_50k_dice.png").unwrap();
//!
//! let pointillist: RgbaImage = geometrize(
//!     &image,
//!     Style::Pointillist {noise: 0.0},
//!     10_000,
//!     SamplingParams::default()
//! ).unwrap();
//! pointillist.save("pointillist_10k_dice.png").unwrap();
//! ```
//!
//! <div style="display:flex; gap:0.5%;">
//!   <figure style="width:33%; margin:0;">
//!     <img src="https://github.com/jac0-b/geometrize/blob/990b60fbee240eb5f8bf885198ad6d8f0ef60931/tests/fixtures/dice.png?raw=true" style="width:100%;">
//!     <figcaption>dice.png</figcaption>
//!   </figure>
//!
//!   <figure style="width:33%; margin:0;">
//!     <img src="https://private-user-images.githubusercontent.com/51512690/588902290-de99517f-9471-4d33-b925-a12b4cf54f7a.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzgxNTkwMzUsIm5iZiI6MTc3ODE1ODczNSwicGF0aCI6Ii81MTUxMjY5MC81ODg5MDIyOTAtZGU5OTUxN2YtOTQ3MS00ZDMzLWI5MjUtYTEyYjRjZjU0ZjdhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA1MDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNTA3VDEyNTg1NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTFkYzBkMzQ2MWRjZjc3ZTUwZTRkYmRhNWYyZGVhMTczM2Y2NGQ3ZmM2MGU2MmZiYmU2NjYzZjFiMzJmYmMzOWYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.Uk5sZGIWFGzGW-p7clBJA-w5Wd5kf9eEdB3QgmC_o7E" style="width:100%;">
//!     <figcaption>lowpoly_50k_dice.png</figcaption>
//!   </figure>
//!   <figure style="width:33%; margin:0;">
//!     <img src="https://private-user-images.githubusercontent.com/51512690/588902770-ab1f754e-66e6-4689-90ca-3ea91e13a0e5.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzgxNTkwMzUsIm5iZiI6MTc3ODE1ODczNSwicGF0aCI6Ii81MTUxMjY5MC81ODg5MDI3NzAtYWIxZjc1NGUtNjZlNi00Njg5LTkwY2EtM2VhOTFlMTNhMGU1LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA1MDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNTA3VDEyNTg1NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTYwNTdjMjRjNmNmMTczM2E5MmRlYjQxOTEyODZjMTZhODg2NDdkMWM4N2ZhMDJmN2QwNTI2ZWIxMjBmZWM2NjImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.ehe9W9nrT6V5fqBNgOeuqbpDN9NSxKwdSfYLt0R6eC8" style="width:100%;">
//!     <figcaption>pointillist_10k_dice.png</figcaption>
//!   </figure>
//! </div>
//!

use image::{DynamicImage, GenericImageView, GrayImage, Rgba, RgbaImage};
use rand::{RngExt, SeedableRng, rngs::SmallRng, seq::index};
use rayon::prelude::*;
use spade::{DelaunayTriangulation, Point2, Triangulation};
use thiserror::Error;

/// Error type for [`geometrize`].
#[derive(Error, Debug)]
pub enum GeometrizeError {
    /// The requested sample count `n` was outside the valid range `[10, num_pixels]`.
    #[error("Expected n in range [10, {num_pixels}], got {n}")]
    NSamplesError { num_pixels: u32, n: u32 },

    /// The internal Gaussian blur step failed.
    #[error("Error blurring image for edge detection")]
    BlurError,

    /// The `noise` parameter for [`Style::Pointillist`] was outside `[0.0, 1.0]`.
    #[error("Expected noise in range [0.0, 1.0] got {0}")]
    NoiseError(f32),
}

/// The visual style used to render the output image.
#[derive(Debug, Clone)]
pub enum Style {
    /// Renders the image as a mosaic of colored triangles based on this [tutorial by Samuel Hinton](https://cosmiccoding.com.au/tutorials/lowpoly/).
    Lowpoly,
    /// Renders the image as overlapping colored circles.
    Pointillist {
        /// `noise` controls the draw order of the circles. A value of `0.0` will draw smaller circles in the foreground, while
        /// a value of `1.0` will randomly draw circles regardless of their size.
        /// Must be in the range `[0.0, 1.0]`.
        ///
        /// ```
        /// let image = open("aurora.jpg").unwrap();
        /// # let image = open("./tests/fixtures/aurora.jpg").unwrap();
        ///
        /// geometrize(
        ///     &image,
        ///     Style::Pointillist {noise: 0.0},
        ///     100_000,
        ///     SamplingParams::default()
        /// ).unwrap().save("pointillist_0_noise.jpg").unwrap();
        ///
        /// geometrize(
        ///     &image,
        ///     Style::Pointillist {noise: 1.0},
        ///     20_000,
        ///     SamplingParams::default()
        /// ).unwrap().save("pointillist_100_noise.jpg").unwrap();
        /// ```
        ///
        /// <div style="display:flex; gap:0.5%;">
        ///   <figure style="width:33%; margin:0;">
        ///     <img src="https://github.com/jac0-b/geometrize/blob/0140865afdb167904d2feb58f82eef07478130d7/images/aurora.jpg?raw=true" style="width:100%;">
        ///     <figcaption>bubbles.jpg</figcaption>
        ///   </figure>
        ///   <figure style="width:33%; margin:0;">
        ///     <img src="https://github.com/jac0-b/geometrize/blob/0140865afdb167904d2feb58f82eef07478130d7/images/pointillist_noise0_20k_aurora.jpg?raw=true" style="width:100%;">
        ///     <figcaption>pointillist_0_noise.jpg</figcaption>
        ///   </figure>
        ///   <figure style="width:33%; margin:0;">
        ///     <img src="https://github.com/jac0-b/geometrize/blob/0140865afdb167904d2feb58f82eef07478130d7/images/pointillist_noise1_20k_aurora.jpg?raw=true" style="width:100%;">
        ///     <figcaption>pointillist_100_noise.jpg</figcaption>
        ///   </figure>
        /// </div>
        noise: f32,
    },
}

/// Parameters controlling how sample points are chosen from the image.
pub struct SamplingParams {
    /// Determines the random seed used for point sampling. Defaults to [`SampleSeed::Image`].
    pub seed: SampleSeed,
    /// Controls how many extra points are placed along the image border to prevent
    /// distorted triangles at the edge of the image. Defaults to [`EdgePoints::Auto`].
    pub edge_mode: EdgePoints,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            seed: SampleSeed::Image,
            edge_mode: EdgePoints::Auto,
        }
    }
}

/// Source of randomness for point sampling.
#[derive(Debug, Copy, Clone)]
pub enum SampleSeed {
    /// A fresh random seed each run -- output will differ every time.
    Random,
    /// A custom seed.
    Custom(u64),
    /// A seed derived from the image content. This is the equivalent to
    /// using [`seed_from_image`].
    /// ```ignore
    /// SampleSeed::Custom(seed_from_image(&image))
    /// ```
    Image,
}

/// Controls how many sample points are placed along the image border.
///
/// This ensures the entire image frame is filled and avoids distortions at the edges.
#[derive(Debug, Copy, Clone)]
pub enum EdgePoints {
    /// Automatically calculate the number of points to place around the border.
    /// This is the default.
    Auto,
    /// Do not place additional points around the border. This may result in gaps
    /// around the edges of the image.
    Disabled,
    /// Add exactly `count` border points, distributed evenly around the perimeter.
    Custom { count: u32 },
}

type Color = [u8; 4];

#[derive(Debug, Copy, Clone)]
struct Point<T> {
    x: T,
    y: T,
}

impl<T: Copy + 'static> Point<T> {
    fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
    fn cast<U: Copy + 'static>(&self) -> Point<U>
    where
        T: num_traits::AsPrimitive<U>,
    {
        Point {
            x: self.x.as_(),
            y: self.y.as_(),
        }
    }
}

impl<T: Copy + 'static> From<Point<T>> for [T; 2] {
    fn from(p: Point<T>) -> Self {
        [p.x, p.y]
    }
}

impl<T: Copy + 'static> From<Point<T>> for (T, T) {
    fn from(p: Point<T>) -> Self {
        (p.x, p.y)
    }
}
#[derive(Debug, Clone)]
struct ColoredTriangle<T> {
    vertices: [Point<T>; 3],
    color: Color,
}

impl<T: Copy> ColoredTriangle<T> {
    fn new(vertices: [Point<T>; 3], color: Color) -> Self {
        Self { vertices, color }
    }
    fn draw(&self, canvas: &mut RgbaImage)
    where
        T: num_traits::AsPrimitive<i32>,
    {
        imageproc::drawing::draw_antialiased_polygon_mut(
            canvas,
            &self
                .vertices
                .map(|v| imageproc::point::Point::new(v.x.as_(), v.y.as_())),
            Rgba(self.color),
            imageproc::pixelops::interpolate,
        );
    }
}
#[derive(Debug, Clone)]
struct ColoredCircle<T> {
    center: Point<T>,
    radius: T,
    color: Color,
}

impl<T: Copy> ColoredCircle<T> {
    fn draw(&self, canvas: &mut RgbaImage)
    where
        T: num_traits::AsPrimitive<i32>,
    {
        use imageproc::drawing::draw_filled_circle_mut;
        draw_filled_circle_mut(
            canvas,
            self.center.cast::<i32>().into(),
            self.radius.as_(),
            Rgba(self.color),
        );
    }
}

impl<T> From<ColoredTriangle<T>> for ColoredCircle<T>
where
    T: Copy + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive,
{
    fn from(ctriangle: ColoredTriangle<T>) -> Self {
        let [v1, v2, v3] = ctriangle.vertices;

        // calculate centroid of triangle
        let cx = (v1.x.as_() + v2.x.as_() + v3.x.as_()) / 3.0;
        let cy = (v1.y.as_() + v2.y.as_() + v3.y.as_()) / 3.0;

        // calculate distances from centroid to vertices
        let [a, b, c] = [v1, v2, v3].map(|Point { x, y }| f32::hypot(cx - x.as_(), cy - y.as_()));

        // +0.5 to account for any rounding down i.e. casts
        // this ensures that the circles always cover the full image.
        let radius = a.max(b).max(c) + 0.5;

        ColoredCircle {
            center: Point {
                x: T::from_f32(cx).unwrap(),
                y: T::from_f32(cy).unwrap(),
            },
            radius: T::from_f32(radius).unwrap(),
            color: ctriangle.color,
        }
    }
}

/// Derive a seed from the content of an image.
///
/// Identical images will produce the same seed.
pub fn seed_from_image(image: &DynamicImage) -> u64 {
    use rustc_hash::FxHasher;
    use std::hash::Hasher;

    let mut hasher = FxHasher::with_seed(1);
    hasher.write(image.as_bytes());
    hasher.finish()
}

/// Convert an image into geometric art.
///
/// Samples `n` points from the image biased toward areas of high contrast.
/// This is effectively the level of detail of the output image.
/// Note that `n` is just approximate and some samples will be rejected as triangle vertices.
///
/// Returns a stylized image built from shapes depending on the selected [`Style`]
///
/// # Arguments
///
/// - `image` — The source [`DynamicImage`] to geometrize.
/// - `style` — [`Style::Lowpoly`] for triangles, [`Style::Pointillist`] for circles.
/// - `n` — Number of sample points. The more points the more detail. Must be in `[10, width × height]`.
/// - `sampling` — Seed and edge-point configuration; use [`SamplingParams::default()`] to start.
///
/// # Errors
///
/// Returns [`GeometrizeError::NSamplesError`] if `n` is out of range `[10, width × height]`,
/// [`GeometrizeError::NoiseError`] if `noise` is outside `[0.0, 1.0]`,
/// or [`GeometrizeError::BlurError`] if edge detection fails.
///
/// # Example
///
/// ```ignore
/// use geometrize::{geometrize, Style, SamplingParams};
/// use image::{open, RgbaImage};
///
/// let image = open("image.png").unwrap();
///
/// let lowpoly: RgbaImage = geometrize(
///     &image.clone(),
///     Style::Lowpoly,
///     100_000,
///     SamplingParams::default()
/// ).unwrap();
/// lowpoly.save("lowpoly_100k.png").unwrap();
///
/// let pointillist: RgbaImage = geometrize(
///     &image,
///     Style::Pointillist {noise: 0.0},
///     20_000,
///     SamplingParams::default()
/// ).unwrap();
/// pointillist.save("pointillist_20k.png").unwrap();
/// ```
///
///
///
///
pub fn geometrize(
    image: &DynamicImage,
    style: Style,
    n: u32,
    sampling: SamplingParams,
) -> Result<RgbaImage, GeometrizeError> {
    let (w, h) = image.dimensions();
    let pixels = w * h;

    if !(10..=(w * h)).contains(&n) {
        return Err(GeometrizeError::NSamplesError {
            num_pixels: pixels,
            n,
        });
    }

    let rng = match sampling.seed {
        SampleSeed::Random => rand::make_rng(),
        SampleSeed::Custom(state) => SmallRng::seed_from_u64(state),
        SampleSeed::Image => SmallRng::seed_from_u64(seed_from_image(&image)),
    };
    match style {
        Style::Lowpoly => lowpoly(image, n, rng, sampling.edge_mode),
        Style::Pointillist { noise } => pointillist(image, n, noise, rng, sampling.edge_mode),
    }
}

fn lowpoly(
    image: &DynamicImage,
    n: u32,
    mut rng: SmallRng,
    edge_mode: EdgePoints,
) -> Result<RgbaImage, GeometrizeError> {
    let grayscale = DynamicImage::ImageLuma8(image.to_luma8());
    let diff_image = diff_of_gaussians(grayscale)?;

    let points: Vec<[f32; 2]> = sample(&diff_image, n, &mut rng, edge_mode)?
        .iter()
        .map(|[x, y]| [*x as f32, *y as f32])
        .collect();

    let triangulation = delaunay(&points[..]);
    let colored_triangles = get_color_of_tri(&image, &triangulation);

    let (w, h) = image.dimensions();
    let mut background = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));
    for triangle in colored_triangles {
        triangle.draw(&mut background);
    }

    Ok(background)
}

fn pointillist(
    image: &DynamicImage,
    n: u32,
    noise: f32,
    mut rng: SmallRng,
    edge_mode: EdgePoints,
) -> Result<RgbaImage, GeometrizeError> {
    if !(0.0..=1.0).contains(&noise) {
        return Err(GeometrizeError::NoiseError(noise));
    }

    let grayscale = DynamicImage::ImageLuma8(image.to_luma8());
    let diff_image = diff_of_gaussians(grayscale)?;

    let points: Vec<[f32; 2]> = sample(&diff_image, n, &mut rng, edge_mode)?
        .iter()
        .map(|[x, y]| [*x as f32, *y as f32])
        .collect();

    let triangulation = delaunay(&points[..]);
    let vertices_colors = get_color_of_tri(&image, &triangulation);

    let (w, h) = image.dimensions();
    let mut background = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));

    let mut circles: Vec<ColoredCircle<_>> = vertices_colors
        .into_iter()
        .map(|colored_triangle| colored_triangle.into())
        .collect();

    circles.sort_by(|a, b| b.radius.total_cmp(&a.radius));

    add_noise(&mut circles, noise, &mut rng);

    for circle in circles {
        circle.draw(&mut background);
    }

    Ok(background)
}

fn add_noise<T>(v: &mut Vec<T>, displacement_fraction: f32, rng: &mut SmallRng) {
    let len = v.len();
    if len == 0 {
        return;
    }

    let max_displacement = (len as f32 * displacement_fraction) as usize;

    for i in 0..len {
        let j = rng.random_range(i..=(i + max_displacement).min(len - 1));
        v.swap(i, j);
    }
}

fn diff_of_gaussians(gray_image: DynamicImage) -> Result<GrayImage, GeometrizeError> {
    let (width, height) = gray_image.dimensions();

    let gauss1 = add_blur(gray_image.clone(), 2.0)?.to_luma32f();
    let gauss2 = add_blur(gray_image, 30.0)?.to_luma32f();

    let diff: Vec<f32> = gauss1
        .pixels()
        .zip(gauss2.pixels())
        .map(|(p1, p2)| {
            let d = p1.0[0] - p2.0[0];
            if d < 0.0 { d * 0.1 } else { d }
        })
        .collect();
    let max_diff = diff.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Ok(GrayImage::from_vec(
        width,
        height,
        diff.iter()
            .map(|e| ((e.abs() / max_diff).sqrt() * 255.0) as u8)
            .collect::<Vec<u8>>(),
    )
    .expect("Should always be able to rebuild image from gray_image dimensions"))
}

fn sample(
    diff_image: &GrayImage,
    n: u32,
    mut rng: &mut SmallRng,
    edge_mode: EdgePoints,
) -> Result<Vec<[u32; 2]>, GeometrizeError> {
    let (w, h) = diff_image.dimensions();
    let num_pixels = w as u32 * h as u32;

    let mut points: Vec<[u32; 2]> = index::sample(&mut rng, num_pixels as usize, n as usize - 4)
        .into_iter()
        .filter_map(|rand_index| {
            let (rand_x, rand_y) = (rand_index as u32 % w, rand_index as u32 / w);
            let rand_luma = rng.random_range(0..255);
            let value = diff_image.get_pixel(rand_x, rand_y).to_owned().0[0];

            if rand_luma < value {
                Some([rand_x, rand_y])
            } else {
                None
            }
        })
        .collect();

    match edge_mode {
        EdgePoints::Auto => {
            let pixel_ratio = w * h / (2 * w + 2 * h);
            let edge_n = (n as u32 / pixel_ratio).max(20);
            add_samples_to_edge(&mut points, edge_n, w, h);
        }
        EdgePoints::Custom { count } => add_samples_to_edge(&mut points, count, w, h),
        EdgePoints::Disabled => (),
    }

    Ok(points)
}

fn add_samples_to_edge(points: &mut Vec<[u32; 2]>, n: u32, w: u32, h: u32) {
    let half_perimeter = w + h;
    let points_along_width = w * n / half_perimeter;
    let points_along_height = h * n / half_perimeter;

    // corners
    points.extend([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]);

    // top and bottom edges
    points.extend((1..points_along_width).flat_map(|x| {
        let px = (w * x) / points_along_width;
        [[px, 0], [px, h - 1]]
    }));

    // left and right edges
    points.extend((1..points_along_height).flat_map(|y| {
        let py = (h * y) / points_along_height;
        [[0, py], [w - 1, py]]
    }));
}

fn delaunay(samples: &[[f32; 2]]) -> DelaunayTriangulation<Point2<f32>> {
    let points: Vec<Point2<f32>> = samples.iter().map(|&[x, y]| Point2::new(x, y)).collect();
    DelaunayTriangulation::bulk_load(points).unwrap()
}

struct BoundingBox {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
}

impl BoundingBox {
    fn from_positions(positions: &[Point2<f32>; 3]) -> Self {
        Self {
            min_x: positions.iter().map(|p| p.x as u32).min().unwrap(),
            min_y: positions.iter().map(|p| p.y as u32).min().unwrap(),
            max_x: positions.iter().map(|p| p.x.ceil() as u32).max().unwrap(),
            max_y: positions.iter().map(|p| p.y.ceil() as u32).max().unwrap(),
        }
    }
    fn pixel_coords(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        let xs = self.min_x..=self.max_x;
        let ys = self.min_y..=self.max_y;
        ys.flat_map(move |y| xs.clone().map(move |x| (x, y)))
    }
}

struct Triangle([Point2<f32>; 3]);

impl Triangle {
    fn bounding_box(&self) -> BoundingBox {
        BoundingBox::from_positions(&self.0)
    }
    // Is the point inside of the triangle?
    fn contains(&self, x: f32, y: f32) -> bool {
        let [a, b, c] = &self.0;
        let p = (x, y);

        let cross = |(ax, ay): (f32, f32), (bx, by): (f32, f32), (cx, cy): (f32, f32)| {
            (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
        };

        let d1 = cross(p, (a.x, a.y), (b.x, b.y));
        let d2 = cross(p, (b.x, b.y), (c.x, c.y));
        let d3 = cross(p, (c.x, c.y), (a.x, a.y));

        // Point is inside iff it's on the same side of all three edges
        let all_non_negative = d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0;
        let all_non_positive = d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0;
        all_non_negative || all_non_positive
    }
    fn avg_pixel_color(&self, image: &RgbaImage) -> Color {
        let bbox = self.bounding_box();

        let (sum, count) = bbox
            .pixel_coords()
            .filter(|&(x, y)| self.contains(x as f32, y as f32))
            .map(|(x, y)| *image.get_pixel(x, y))
            .fold(([0u64; 4], 0u64), |(mut acc, count), px| {
                acc.iter_mut().zip(px.0).for_each(|(a, b)| *a += b as u64);
                (acc, count + 1)
            });

        if count == 0 {
            return [0, 0, 0, 0];
        }
        sum.map(|s| (s / count) as u8)
    }
}

fn get_color_of_tri(
    image: &DynamicImage,
    tri: &DelaunayTriangulation<Point2<f32>>,
) -> Vec<ColoredTriangle<f32>> {
    let rgba = image.to_rgba8();
    tri.inner_faces()
        .collect::<Vec<_>>()
        .par_iter()
        .map(|face| {
            let positions = face.vertices().map(|v| v.position());
            let triangle = Triangle(positions);
            let color = triangle.avg_pixel_color(&rgba);
            ColoredTriangle::new(positions.map(|v| Point::new(v.x, v.y)), color)
        })
        .collect()
}

fn add_blur(image: DynamicImage, sigma: f64) -> Result<DynamicImage, GeometrizeError> {
    use libblur::{self, AnisotropicRadius};

    libblur::fast_gaussian_blur_image(
        image,
        AnisotropicRadius::new(sigma as u32),
        libblur::EdgeMode2D::new(libblur::EdgeMode::Clamp),
        libblur::ThreadingPolicy::Adaptive,
    )
    .ok_or(GeometrizeError::BlurError)
}
