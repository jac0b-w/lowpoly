use image::{DynamicImage, GenericImageView, GrayImage, Rgba, RgbaImage};
use rayon::prelude::*;
use spade::{DelaunayTriangulation, Point2, Triangulation};
use thiserror::Error;

// https://cosmiccoding.com.au/tutorials/lowpoly/

#[derive(Error, Debug)]
pub enum LowpolyError {
    #[error("Expected n in range [4, {1}], got {0}")]
    NSamplesError(u64, u64),

    #[error("Error blurring image for edge detection")]
    BlurError,
}

#[derive(Debug, Copy, Clone)]
pub enum SampleSeed {
    Random,
    Custom(u64),
}

#[derive(Debug, Copy, Clone)]
pub enum EdgePoints {
    Auto,
    Disabled,
    Custom { count: u32 },
}

type TriangleVertices = [[f32; 2]; 3];
type Color = [u8; 4];

pub fn seed_from_image(image: &DynamicImage) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    image.as_bytes().hash(&mut hasher);
    hasher.finish()
}

pub fn lowpoly(image: DynamicImage, n: u64) -> Result<RgbaImage, LowpolyError> {
    let seed = SampleSeed::Custom(seed_from_image(&image));
    lowpoly_with_seed(image, n, seed, EdgePoints::Auto)
}

pub fn lowpoly_with_seed(
    image: DynamicImage,
    n: u64,
    seed: SampleSeed,
    edge_mode: EdgePoints,
) -> Result<RgbaImage, LowpolyError> {
    let grayscale = DynamicImage::ImageLuma8(image.to_luma8());
    let diff_image = diff_of_gaussians(grayscale)?;

    let points: Vec<[f32; 2]> = sample(&diff_image, n, seed, edge_mode)?
        .iter()
        .map(|[x, y]| [*x as f32, *y as f32])
        .collect();

    let triangulation = delaunay(&points[..]);
    let vertices_colors = get_color_of_tri(&image, &triangulation);

    let (w, h) = image.dimensions();
    let lowpoly_image = draw_triangles(vertices_colors, w, h);

    Ok(lowpoly_image)
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
    n: u64,
    seed: SampleSeed,
    edge_mode: EdgePoints,
) -> Result<Vec<[u32; 2]>, LowpolyError> {
    use rand::{RngExt, SeedableRng, rngs::SmallRng, seq::index};

    let (w, h) = diff_image.dimensions();

    let num_pixels = w as u64 * h as u64;
    if !(4..num_pixels).contains(&n) {
        return Err(LowpolyError::NSamplesError(n, num_pixels));
    }

    let mut rng: SmallRng = match seed {
        SampleSeed::Random => rand::make_rng(),
        SampleSeed::Custom(state) => SmallRng::seed_from_u64(state),
    };

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
) -> Vec<(TriangleVertices, Color)> {
    let (width, height) = image.dimensions();
    tri.inner_faces()
        .collect::<Vec<_>>()
        .par_iter()
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

            (
                verts.map(|v| [v.position().x, v.position().y]),
                avg_color(&pixels),
            )
        })
        .collect()
}

fn draw_triangles(vertices_colors: Vec<(TriangleVertices, Color)>, w: u32, h: u32) -> RgbaImage {
    use imageproc::drawing::draw_antialiased_polygon_mut;
    use imageproc::point::Point;

    let mut output = RgbaImage::new(w, h);

    for (poly, color) in vertices_colors {
        draw_antialiased_polygon_mut(
            &mut output,
            &poly.map(|e| Point::new((e[0] as f32) as i32, (e[1] as f32) as i32))[..],
            Rgba(color),
            imageproc::pixelops::interpolate,
        );
    }
    output
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
