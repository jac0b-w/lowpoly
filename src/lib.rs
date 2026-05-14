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
//! ``` no_run
//! use geometrize::{geometrize, Style, SamplingParams};
//! use image::{open, RgbaImage};
//!
//! let image = open("launch.jpg").unwrap();
//!
//! let lowpoly: RgbaImage = geometrize(
//!     &image,
//!     Style::Lowpoly,
//!     70_000,
//!     SamplingParams::default()
//! ).unwrap();
//! lowpoly.save("lowpoly_70k.png").unwrap();
//!
//! let pointillist: RgbaImage = geometrize(
//!     &image,
//!     Style::Pointillist {noise: 0.0},
//!     20_000,
//!     SamplingParams::default()
//! ).unwrap();
//! pointillist.save("pointillist_20k.png").unwrap();
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
//!     <figcaption>lowpoly_70k.png</figcaption>
//!   </figure>
//!   <figure style="width:33%; margin:0;">
//!     <img src="https://github.com/jac0-b/geometrize/blob/0140865afdb167904d2feb58f82eef07478130d7/images/pointillist__20k_launch.jpg?raw=true" style="width:100%;">
//!     <figcaption>pointillist_20k.png</figcaption>
//!   </figure>
//! </div>
//!
//! # Transparent images
//!
//! This also works with transparent images.
//!
//! ```no_run
//! # use geometrize::{geometrize, Style, SamplingParams};
//! # use image::{open, RgbaImage};
//!
//! let image = open("dice.png").unwrap();
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
//!     <img src="https://github.com/jac0-b/geometrize/blob/2594d40a15b8726bccafa647243d1b1627a021a0/images/lowpoly_50k_dice.png?raw=true" style="width:100%;">
//!     <figcaption>lowpoly_50k_dice.png</figcaption>
//!   </figure>
//!   <figure style="width:33%; margin:0;">
//!     <img src="https://github.com/jac0-b/geometrize/blob/2594d40a15b8726bccafa647243d1b1627a021a0/images/pointillist_10k_dice.png?raw=true" style="width:100%;">
//!     <figcaption>pointillist_10k_dice.png</figcaption>
//!   </figure>
//! </div>
//!

mod point;
mod sampling;
mod shape;
mod style_builder;

use crate::shape::{DrawableShape, Shape, ShapeKind};
pub use crate::{
    point::Point,
    sampling::{ColorSampler, EdgePoints, PointSampler, SampleDistribution},
    style_builder::CustomStyleBuilder,
};

use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use rand::{Rng, RngExt, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use spade::{DelaunayTriangulation, Triangulation};
use thiserror::Error;

type DelaunayFace<'a> =
    spade::handles::FaceHandle<'a, spade::handles::InnerTag, spade::Point2<f32>, (), (), ()>;
type VoronoiEdge<'a> = spade::handles::DirectedVoronoiEdge<'a, spade::Point2<f32>, (), (), ()>;

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

    #[error("Cannot construct polygon with {0} sides")]
    PolgonConstructionError(usize),
}

/// The visual style used to render the output image.
#[derive(Debug, Clone)]
pub enum Style {
    /// Renders the image as a mosaic of colored triangles based on this [tutorial by Samuel Hinton](https://cosmiccoding.com.au/tutorials/lowpoly/).
    Lowpoly,
    Voronoi,
    Custom {
        shapes: Vec<ShapeKind>,
        noise: f32,
    },
}

impl Style {
    /// /// Renders the image as overlapping colored circles.
    ///
    /// `noise` controls the draw order of the circles. A value of `0.0` will draw smaller circles in the foreground, while
    /// a value of `1.0` will randomly draw circles regardless of their size.
    /// Must be in the range `[0.0, 1.0]`.
    ///
    /// ```no_run
    /// # use geometrize::{geometrize, Style, SamplingParams};
    /// # use image::{open, RgbaImage};
    ///
    /// let image = open("aurora.jpg").unwrap();
    ///
    /// geometrize(
    ///     &image,
    ///     Style::Pointillist {noise: 0.0},
    ///     100_000,
    ///     SamplingParams::default()
    /// ).unwrap().save("pointillist_0_noise.png").unwrap();
    ///
    /// geometrize(
    ///     &image,
    ///     Style::Pointillist {noise: 1.0},
    ///     20_000,
    ///     SamplingParams::default()
    /// ).unwrap().save("pointillist_100_noise.png").unwrap();
    /// ```
    ///
    /// <div style="display:flex; gap:0.5%;">
    ///   <figure style="width:33%; margin:0;">
    ///     <img src="https://github.com/jac0-b/geometrize/blob/0140865afdb167904d2feb58f82eef07478130d7/images/aurora.jpg?raw=true" style="width:100%;">
    ///     <figcaption>aurora.jpg</figcaption>
    ///   </figure>
    ///   <figure style="width:33%; margin:0;">
    ///     <img src="https://github.com/jac0-b/geometrize/blob/0140865afdb167904d2feb58f82eef07478130d7/images/pointillist_noise0_20k_aurora.jpg?raw=true" style="width:100%;">
    ///     <figcaption>pointillist_0_noise.png</figcaption>
    ///   </figure>
    ///   <figure style="width:33%; margin:0;">
    ///     <img src="https://github.com/jac0-b/geometrize/blob/0140865afdb167904d2feb58f82eef07478130d7/images/pointillist_noise1_20k_aurora.jpg?raw=true" style="width:100%;">
    ///     <figcaption>pointillist_100_noise.png</figcaption>
    ///   </figure>
    /// </div>
    pub fn pointillist(noise: f32) -> Self {
        Self::Custom {
            shapes: vec![ShapeKind::circle()],
            noise,
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

use core::f32;
use std::marker::PhantomData;

pub struct Untriangulated;
pub struct Triangulated;

pub struct Geometrize<'a, State, R: Rng> {
    _state: PhantomData<State>,
    image: &'a DynamicImage,
    rng: R,
    triangulation: Option<DelaunayTriangulation<spade::Point2<f32>>>,
}

impl<'a> Geometrize<'a, Untriangulated, SmallRng> {
    pub fn new(image: &'a DynamicImage) -> Self {
        let seed = seed_from_image(&image);
        Self::new_with_seed(image, seed)
    }
    pub fn new_with_seed(image: &'a DynamicImage, seed: u64) -> Self {
        let rng = SmallRng::seed_from_u64(seed);
        Self {
            _state: PhantomData,
            image,
            rng: rng,
            triangulation: None,
        }
    }
}

impl<'a, R: Rng> Geometrize<'a, Untriangulated, R> {
    pub fn new_with_rng(image: &'a DynamicImage, rng: R) -> Self {
        Self {
            _state: PhantomData,
            image,
            rng,
            triangulation: None,
        }
    }
    pub fn sample(
        mut self,
        point_sampler: PointSampler,
    ) -> Result<Geometrize<'a, Triangulated, R>, GeometrizeError> {
        let points: Vec<spade::Point2<f32>> = point_sampler.generate(self.image, &mut self.rng)?;
        let triangulation = DelaunayTriangulation::bulk_load(points).unwrap();

        Ok(Geometrize {
            _state: PhantomData,
            image: self.image,
            rng: self.rng,
            triangulation: Some(triangulation),
        })
    }
    pub fn with_num_samples(
        self,
        samples: u32,
    ) -> Result<Geometrize<'a, Triangulated, R>, GeometrizeError> {
        self.sample(PointSampler::from_num_samples(samples))
    }
}

impl<'a, R: Rng + Clone> Geometrize<'a, Triangulated, R> {
    pub fn render_with_color_sampler(
        &self,
        style: &Style,
        color_sampler: &ColorSampler,
    ) -> Result<RgbaImage, GeometrizeError> {
        match style {
            Style::Custom { noise, .. } => {
                if !(0.0..=1.0).contains(noise) {
                    return Err(GeometrizeError::NoiseError(*noise));
                }
            }
            _ => (),
        }

        let rgba_image = self.image.to_rgba8();
        let triangulation = self.triangulation.as_ref().unwrap();

        let drawable_shapes = match style {
            Style::Lowpoly => lowpoly(&rgba_image, triangulation, color_sampler),
            Style::Voronoi => voronoi(&rgba_image, triangulation, &color_sampler),
            Style::Custom { shapes, noise } => custom_shapes(
                &rgba_image,
                triangulation,
                shapes,
                *noise,
                &color_sampler,
                &mut self.rng.clone(),
            ),
        }?;

        let (w, h) = self.image.dimensions();

        // draw shapes onto image
        let mut canvas = RgbaImage::from_pixel(w, h, Rgba([0; 4]));
        for drawable_shape in drawable_shapes {
            drawable_shape.draw(&mut canvas)
        }

        Ok(canvas)
    }

    pub fn render(&self, style: &Style) -> Result<RgbaImage, GeometrizeError> {
        let color_sampler = ColorSampler::default();
        self.render_with_color_sampler(style, &color_sampler)
    }
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
/// ```no_run
/// # use geometrize::{geometrize, Style, SamplingParams};
/// # use image::{open, RgbaImage};
///
/// let image = open("launch.jpg").unwrap();
///
/// let lowpoly: RgbaImage = geometrize(
///     &image,
///     Style::Lowpoly,
///     70_000,
///     SamplingParams::default()
/// ).unwrap();
/// lowpoly.save("lowpoly_70k.png").unwrap();
///
/// let pointillist: RgbaImage = geometrize(
///     &image,
///     Style::Pointillist {noise: 0.0},
///     20_000,
///     SamplingParams::default()
/// ).unwrap();
/// pointillist.save("pointillist_20k.png").unwrap();
/// ```

fn lowpoly(
    image: &RgbaImage,
    triangulation: &DelaunayTriangulation<spade::Point2<f32>>,
    color_sampling: &ColorSampler,
) -> Result<Vec<DrawableShape>, GeometrizeError> {
    let faces = triangulation.inner_faces().collect();
    let colored_triangles = faces_to_triangle_par_iter(faces)
        .map(|triangle| {
            shape::DrawableShape::from_shape_image(
                &ShapeKind::Polygon(triangle),
                &image,
                &color_sampling,
            )
        })
        .collect();
    Ok(colored_triangles)
}

fn voronoi(
    image: &RgbaImage,
    triangulation: &DelaunayTriangulation<spade::Point2<f32>>,
    color_sampler: &ColorSampler,
) -> Result<Vec<DrawableShape>, GeometrizeError> {
    let dimensions = image.dimensions();

    let colored_polygons: Vec<_> = triangulation
        .voronoi_faces()
        .collect::<Vec<_>>()
        .into_par_iter()
        .filter_map(|face| {
            let vertices: Vec<Point> = face
                .adjacent_edges()
                .flat_map(|edge| voronoi_edge_to_vertex_pair(edge, &dimensions))
                .flatten()
                .collect();

            let vertices = add_vertices_at_corners(vertices, &dimensions);

            let polygon = shape::Polygon::try_from(vertices)
                .expect("Polygon should have at least 3 vertices.");
            Some(shape::DrawableShape::from_shape_image(
                &shape::ShapeKind::Polygon(polygon),
                &image,
                color_sampler,
            ))
        })
        .collect();

    Ok(colored_polygons)
}

fn custom_shapes<R: Rng>(
    image: &RgbaImage,
    triangulation: &DelaunayTriangulation<spade::Point2<f32>>,
    shapes: &Vec<shape::ShapeKind>,
    noise: f32,
    color_sampler: &ColorSampler,
    mut rng: &mut R,
) -> Result<Vec<DrawableShape>, GeometrizeError> {
    let faces: Vec<_> = triangulation.inner_faces().collect();

    let rotations: Vec<f32> = rng.random_iter().take(faces.len()).collect();

    let dist = rand::distr::Uniform::new(0, shapes.len()).unwrap();
    let shape_indices: Vec<usize> = rng.sample_iter(dist).take(faces.len()).collect();

    let mut drawable_shapes: Vec<_> = faces_to_triangle_par_iter(faces)
        .zip(rotations.into_par_iter())
        .zip(shape_indices.into_par_iter())
        .map(|((tri, rot), index)| {
            // + 1.3 to ensure the image is covered
            let scale = 1.3 + tri.circumradius().expect("tri should be a triangle");
            let circumcemter = tri.circumcenter().expect("tri should be a triangle");

            let shape = &shapes[index].transformed(scale, circumcemter.into(), rot);

            (
                scale,
                DrawableShape::from_shape_image(shape, image, color_sampler),
            )
        })
        .collect();

    drawable_shapes.sort_by(|(a, _), (b, _)| b.total_cmp(a));

    let mut drawable_shapes: Vec<_> = drawable_shapes.into_iter().map(|(_, p)| p).collect();

    add_noise(&mut drawable_shapes, noise, &mut rng);

    Ok(drawable_shapes)
}

fn faces_to_triangle_par_iter(
    faces: Vec<DelaunayFace>,
) -> impl IndexedParallelIterator<Item = shape::Polygon> {
    faces.into_par_iter().map(|face| {
        let vertices: Vec<Point> = face
            .vertices()
            .iter()
            .map(|v| v.position().into())
            .collect();
        shape::Polygon::try_from(vertices).unwrap()
    })
}

fn voronoi_edge_to_vertex_pair(edge: VoronoiEdge, dimensions: &(u32, u32)) -> [Option<Point>; 2] {
    use spade::handles::VoronoiVertex::Inner;

    match edge.as_undirected().vertices() {
        [Inner(from), Inner(to)] => [
            Some(Point::from(from.circumcenter()).clamped_to_frame(*dimensions)),
            Some(Point::from(to.circumcenter()).clamped_to_frame(*dimensions)),
        ],
        [Inner(inner), _] | [_, Inner(inner)] => {
            let inner_point = Point::from(inner.circumcenter()).clamped_to_frame(*dimensions);
            [
                Some(inner_point),
                Some(
                    inner_point
                        .clamped_to_frame_direction(&edge.direction_vector().into(), dimensions),
                ),
            ]
        }
        _ => {
            // https://docs.rs/spade/latest/spade/struct.DelaunayTriangulation.html#extracting-the-voronoi-diagram-example
            [None, None]
        }
    }
}

fn add_vertices_at_corners(mut vertices: Vec<Point>, dimensions: &(u32, u32)) -> Vec<Point> {
    let w = dimensions.0 as f32;
    let h = dimensions.1 as f32;
    let corners = [
        Point::from([0.0, 0.0]),
        Point::from([w, 0.0]),
        Point::from([w, h]),
        Point::from([0.0, h]),
    ];

    let boundary_count = vertices
        .iter()
        .filter(|p| p.x_near(0.0) || p.y_near(0.0) || p.x_near(w) || p.y_near(h))
        .count();

    let touches_left: Vec<_> = vertices.iter().filter(|p| p.x_near(0.0)).copied().collect();
    let touches_right: Vec<_> = vertices.iter().filter(|p| p.x_near(w)).copied().collect();
    let touches_top: Vec<_> = vertices.iter().filter(|p| p.y_near(0.0)).copied().collect();
    let touches_bottom: Vec<_> = vertices.iter().filter(|p| p.y_near(h)).copied().collect();

    if boundary_count >= 2 {
        for corner in corners {
            if vertices.iter().any(|p| p.near(corner)) {
                continue;
            }
            let needs_corner = match corner {
                c if c.near([0.0, 0.0].into()) => {
                    !touches_left.is_empty() && !touches_top.is_empty()
                }
                c if c.near([w, 0.0].into()) => {
                    !touches_right.is_empty() && !touches_top.is_empty()
                }
                c if c.near([w, h].into()) => {
                    !touches_right.is_empty() && !touches_bottom.is_empty()
                }
                c if c.near([0.0, h].into()) => {
                    !touches_left.is_empty() && !touches_bottom.is_empty()
                }
                _ => panic!("Points should be corners."),
            };
            if needs_corner {
                vertices.push(corner);
            }
        }
    }

    vertices
}

fn add_noise<T, R: Rng>(v: &mut [T], displacement_fraction: f32, rng: &mut R) {
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
