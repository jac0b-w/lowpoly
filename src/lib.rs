use image::{DynamicImage, GenericImageView, GrayImage, Rgba, RgbaImage};
use rand::{RngExt, SeedableRng, rngs::SmallRng, seq::index};
use rayon::prelude::*;
use spade::{DelaunayTriangulation, Point2, Triangulation};
use thiserror::Error;

// https://cosmiccoding.com.au/tutorials/lowpoly/

#[derive(Error, Debug)]
pub enum GeometrizeError {
    #[error("Expected n in range [10, {num_pixels}], got {n}")]
    NSamplesError { num_pixels: u32, n: u32 },

    #[error("Error blurring image for edge detection")]
    BlurError,

    #[error("Expected noise in range [0.0, 1.0] got {0}")]
    NoiseError(f32),
}

#[derive(Debug, Clone)]
pub enum Style {
    Lowpoly,
    Pointillist { noise: f32 },
}

pub struct SamplingParams {
    pub seed: SampleSeed,
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

#[derive(Debug, Copy, Clone)]
pub enum SampleSeed {
    Random,
    Custom(u64),
    Image,
}

#[derive(Debug, Copy, Clone)]
pub enum EdgePoints {
    Auto,
    Disabled,
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

        // calculate distances from centriod to vertices
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

pub fn seed_from_image(image: &DynamicImage) -> u64 {
    use rustc_hash::FxHasher;
    use std::hash::Hasher;

    let mut hasher = FxHasher::with_seed(1);
    hasher.write(image.as_bytes());
    hasher.finish()
}

pub fn geometrize(
    image: DynamicImage,
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
    image: DynamicImage,
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
    image: DynamicImage,
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
    // Is the point inside of the triangle
    fn contains(&self, x: f32, y: f32) -> bool {
        let [a, b, c] = &self.0;
        let sign = |p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)| {
            (p1.0 - p3.0) * (p2.1 - p3.1) - (p2.0 - p3.0) * (p1.1 - p3.1)
        };
        let d1 = sign((x, y), (a.x, a.y), (b.x, b.y));
        let d2 = sign((x, y), (b.x, b.y), (c.x, c.y));
        let d3 = sign((x, y), (c.x, c.y), (a.x, a.y));
        let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
        let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
        !(has_neg && has_pos)
    }

    fn avg_pixel_color(&self, image: &DynamicImage) -> Color {
        let bbox = self.bounding_box();

        let pixels: Vec<Rgba<u8>> = bbox
            .pixel_coords()
            .filter(|&(x, y)| self.contains(x as f32, y as f32))
            .map(|(x, y)| image.get_pixel(x, y))
            .collect();

        let count = pixels.len() as u64;
        if count == 0 {
            return [0, 0, 0, 0];
        }
        pixels
            .iter()
            .fold([0u64; 4], |mut acc, px| {
                acc.iter_mut().zip(px.0).for_each(|(a, b)| *a += b as u64);
                acc
            })
            .map(|sum| (sum / count) as u8)
    }
}

fn get_color_of_tri(
    image: &DynamicImage,
    tri: &DelaunayTriangulation<Point2<f32>>,
) -> Vec<ColoredTriangle<f32>> {
    tri.inner_faces()
        .collect::<Vec<_>>()
        .par_iter()
        .map(|face| {
            let positions = face.vertices().map(|v| v.position());
            let triangle = Triangle(positions);
            let color = triangle.avg_pixel_color(image);
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
