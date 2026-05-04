use image::{DynamicImage, GenericImageView, GrayImage, Rgba, RgbaImage};
use rand::{RngExt, SeedableRng, rngs::SmallRng, seq::index};
use rayon::prelude::*;
use spade::{DelaunayTriangulation, Point2, Triangulation};
use thiserror::Error;
use tiny_skia::{Paint, PathBuilder, Pixmap, Transform};

// https://cosmiccoding.com.au/tutorials/lowpoly/

#[derive(Error, Debug)]
pub enum LowpolyError {
    #[error("Expected n in range [10, {num_pixels}], got {n}")]
    NSamplesError { num_pixels: u32, n: u32 },

    #[error("Error blurring image for edge detection")]
    BlurError,

    #[error("Expected noise in range [0.0, 1.0] got {0}")]
    NoiseError(f32),
}

#[derive(Debug, Clone)]
pub enum Style {
    Lowpoly { anti_alias_pass: bool },
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
struct Point {
    x: f32,
    y: f32,
}

impl Point {
    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone)]
struct ColoredTriangle {
    vertices: [Point; 3],
    color: Color,
}

impl ColoredTriangle {
    fn new(vertices: [Point; 3], color: Color) -> Self {
        Self { vertices, color }
    }
}

impl ColoredTriangle {
    fn draw(&self, canvas: &mut Pixmap, anti_alias: bool) {
        let [r, g, b, a] = self.color;
        let mut paint = Paint {
            anti_alias,
            ..Paint::default()
        };
        paint.set_color_rgba8(r, g, b, a);
        let mut path_builder = PathBuilder::new();

        let [a, b, c] = self.vertices;

        // build triangle
        path_builder.move_to(a.x, a.y);
        path_builder.line_to(b.x, b.y);
        path_builder.line_to(c.x, c.y);
        path_builder.close();

        let path = path_builder.finish().unwrap();

        canvas.fill_path(
            &path,
            &paint,
            tiny_skia::FillRule::Winding,
            Transform::default(),
            None,
        );
    }
}

#[derive(Debug, Clone)]
struct ColoredCircle {
    center: Point,
    radius: f32,
    color: Color,
}

impl ColoredCircle {
    fn draw(&self, canvas: &mut Pixmap) {
        let [r, g, b, a] = self.color;
        let mut paint = Paint::default();
        paint.set_color_rgba8(r, g, b, a);
        let path = PathBuilder::from_circle(self.center.x, self.center.y, self.radius).unwrap();
        canvas.fill_path(
            &path,
            &paint,
            tiny_skia::FillRule::Winding,
            Transform::default(),
            None,
        )
    }
}

impl From<ColoredTriangle> for ColoredCircle {
    fn from(ctriangle: ColoredTriangle) -> Self {
        let [v1, v2, v3] = ctriangle.vertices;

        // calculate centroid of triangle
        let cx = (v1.x + v2.x + v3.x) / 3.0;
        let cy = (v1.y + v2.y + v3.y) / 3.0;

        // calculate distances from centriod to vertices
        let [a, b, c] = [v1, v2, v3].map(|Point { x, y }| f32::hypot(cx - x, cy - y));

        // +0.5 to account for any rounding down i.e. casts
        // this ensures that the circles always cover the full image.
        let radius = a.max(b).max(c) + 0.5;

        ColoredCircle {
            center: Point { x: cx, y: cy },
            radius: radius,
            color: ctriangle.color,
        }
    }
}

pub fn seed_from_image(image: &DynamicImage) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    image.as_bytes().hash(&mut hasher);
    hasher.finish()
}

pub fn geometrize(
    image: DynamicImage,
    style: Style,
    n: u32,
    sampling: SamplingParams,
) -> Result<RgbaImage, LowpolyError> {
    let (w, h) = image.dimensions();
    let pixels = w * h;

    if !(10..=(w * h)).contains(&n) {
        return Err(LowpolyError::NSamplesError {
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
        Style::Lowpoly { anti_alias_pass } => {
            lowpoly(image, n, anti_alias_pass, rng, sampling.edge_mode)
        }
        Style::Pointillist { noise } => pointillist(image, n, noise, rng, sampling.edge_mode),
    }
}

fn lowpoly(
    image: DynamicImage,
    n: u32,
    anti_alias_pass: bool,
    mut rng: SmallRng,
    edge_mode: EdgePoints,
) -> Result<RgbaImage, LowpolyError> {
    let grayscale = DynamicImage::ImageLuma8(image.to_luma8());
    let diff_image = diff_of_gaussians(grayscale)?;

    let points: Vec<[f32; 2]> = sample(&diff_image, n, &mut rng, edge_mode)?
        .iter()
        .map(|[x, y]| [*x as f32, *y as f32])
        .collect();

    let triangulation = delaunay(&points[..]);
    let colored_triangles = get_color_of_tri(&image, &triangulation);

    let (w, h) = image.dimensions();
    let mut canvas = Pixmap::new(w, h).unwrap();
    for triangle in &colored_triangles {
        triangle.draw(&mut canvas, false);
    }
    // add an antialiasing pass
    if anti_alias_pass {
        for triangle in &colored_triangles {
            triangle.draw(&mut canvas, true);
        }
    }

    let output =
        RgbaImage::from_raw(canvas.width(), canvas.height(), canvas.data().to_vec()).unwrap();

    Ok(output)
}

fn pointillist(
    image: DynamicImage,
    n: u32,
    noise: f32,
    mut rng: SmallRng,
    edge_mode: EdgePoints,
) -> Result<RgbaImage, LowpolyError> {
    if !(0.0..=1.0).contains(&noise) {
        return Err(LowpolyError::NoiseError(noise));
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

    let mut circles: Vec<ColoredCircle> = vertices_colors
        .into_iter()
        .map(|colored_triangle| colored_triangle.into())
        .collect();

    circles.sort_by(|a, b| b.radius.total_cmp(&a.radius));

    add_noise(&mut circles, noise, &mut rng);

    let mut canvas = Pixmap::new(w, h).unwrap();
    for circle in circles {
        circle.draw(&mut canvas);
    }

    let output =
        RgbaImage::from_raw(canvas.width(), canvas.height(), canvas.data().to_vec()).unwrap();

    Ok(output)
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

fn diff_of_gaussians(gray_image: DynamicImage) -> Result<GrayImage, LowpolyError> {
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
) -> Result<Vec<[u32; 2]>, LowpolyError> {
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

fn get_color_of_tri(
    image: &DynamicImage,
    tri: &DelaunayTriangulation<Point2<f32>>,
) -> Vec<ColoredTriangle> {
    let (width, height) = image.dimensions();
    tri.inner_faces()
        .par_bridge()
        .map(|face| {
            let verts = face.vertices();
            let positions = verts.map(|v| v.position());

            // Bounding box clipped to image
            let min_x = positions.iter().map(|p| p.x as u32).min().unwrap().max(0);
            let min_y = positions.iter().map(|p| p.y as u32).min().unwrap().max(0);
            let max_x = positions
                .iter()
                .map(|p| p.x.ceil() as u32)
                .max()
                .unwrap()
                .min(width - 1);
            let max_y = positions
                .iter()
                .map(|p| p.y.ceil() as u32)
                .max()
                .unwrap()
                .min(height - 1);

            let pixels: Vec<Rgba<u8>> = (min_y..=max_y)
                .flat_map(|y| (min_x..=max_x).map(move |x| (x, y)))
                .filter(|&(x, y)| {
                    point_in_triangle(
                        x as f32,
                        y as f32,
                        &positions[0],
                        &positions[1],
                        &positions[2],
                    )
                })
                .map(|(x, y)| image.get_pixel(x, y))
                .collect();

            ColoredTriangle::new(
                positions.map(|v| Point::new(v.x, v.y)),
                avg_color(&pixels[..]),
            )
        })
        .collect()
}

fn add_blur(image: DynamicImage, sigma: f64) -> Result<DynamicImage, LowpolyError> {
    use libblur::{self, AnisotropicRadius};

    libblur::fast_gaussian_blur_image(
        image,
        AnisotropicRadius::new(sigma as u32),
        libblur::EdgeMode2D::new(libblur::EdgeMode::Clamp),
        libblur::ThreadingPolicy::Adaptive,
    )
    .ok_or(LowpolyError::BlurError)
}

fn avg_color(pixels: &[Rgba<u8>]) -> Color {
    let count = pixels.len() as u64;
    if count == 0 {
        return [0, 0, 0, 255];
    }
    pixels
        .iter()
        .fold([0u64; 4], |mut acc, px| {
            acc[0] += px[0] as u64;
            acc[1] += px[1] as u64;
            acc[2] += px[2] as u64;
            acc[3] += px[3] as u64;
            acc
        })
        .map(|sum| (sum / count) as u8)
}

fn point_in_triangle(px: f32, py: f32, a: &Point2<f32>, b: &Point2<f32>, c: &Point2<f32>) -> bool {
    let sign = |p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)| {
        (p1.0 - p3.0) * (p2.1 - p3.1) - (p2.0 - p3.0) * (p1.1 - p3.1)
    };
    let d1 = sign((px, py), (a.x, a.y), (b.x, b.y));
    let d2 = sign((px, py), (b.x, b.y), (c.x, c.y));
    let d3 = sign((px, py), (c.x, c.y), (a.x, a.y));
    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
    !(has_neg && has_pos)
}
