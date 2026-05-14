use image::{DynamicImage, GenericImageView};
use rand::{Rng, seq::index};

use crate::GeometrizeError;

#[derive(Debug, Clone, Default)]
pub enum ColorSampler {
    Single,
    #[default]
    All,
}

pub enum PointSampler {
    Generated {
        samples: u32,
        distribution: SampleDistribution,
        /// Controls how many extra points are placed along the image border to prevent
        /// distorted triangles at the edge of the image. Defaults to [`EdgePoints::Auto`].
        edge_mode: EdgePoints,
    },
    Custom(Vec<spade::Point2<f32>>),
}

/// Parameters controlling how sample points are chosen from the image.
impl PointSampler {
    pub fn from_num_samples(samples: u32) -> Self {
        Self::Generated {
            samples,
            distribution: SampleDistribution::ContrastBias { radii: (2, 30) },
            edge_mode: EdgePoints::Auto,
        }
    }
    pub fn from_points<I>(points: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<spade::Point2<f32>>,
    {
        let points: Vec<spade::Point2<f32>> = points.into_iter().map(|p| p.into()).collect();

        Self::Custom(points)
    }
    pub fn generate<R: Rng>(
        self,
        image: &DynamicImage,
        mut rng: &mut R,
    ) -> Result<Vec<spade::Point2<f32>>, GeometrizeError> {
        match self {
            Self::Custom(points) => Ok(points),
            Self::Generated {
                samples,
                distribution,
                edge_mode,
            } => {
                let (w, h) = image.dimensions();
                let mut points =
                    sample_points(image, &mut rng, w, h, samples as usize, distribution)?;

                // Add edge points
                match edge_mode {
                    EdgePoints::Auto => {
                        let pixel_ratio = w * h / (2 * w + 2 * h);
                        let edge_n = (samples / pixel_ratio).max(20);
                        add_samples_to_edge(&mut points, edge_n, w, h);
                    }
                    EdgePoints::Custom { count } => add_samples_to_edge(&mut points, count, w, h),
                    EdgePoints::Disabled => (),
                };
                Ok(points
                    .into_iter()
                    .map(|[x, y]| [x as f32, y as f32].into())
                    .collect())
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum SampleDistribution {
    UniformRandom,
    ContrastBias {
        /// Blur radii used for difference of Gaussians.
        radii: (u32, u32),
    },
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
    /// Add `count` border points, distributed evenly around the perimeter.
    Custom { count: u32 },
}

fn sample_points<R: Rng>(
    image: &DynamicImage,
    rng: &mut R,
    w: u32,
    h: u32,
    samples: usize,
    distribution: SampleDistribution,
) -> Result<Vec<[u32; 2]>, GeometrizeError> {
    let num_pixels = (w * h) as usize;
    let index_to_coord = |index| [index as u32 % w, index as u32 / w];

    let points = match distribution {
        SampleDistribution::ContrastBias { radii } => {
            let diff = diff_of_gaussians(image, radii)?;

            index::sample_weighted(rng, num_pixels, |index| diff[index] + 1.0, samples)
                .unwrap()
                .into_iter()
                .map(index_to_coord)
                .collect()
        }
        SampleDistribution::UniformRandom => index::sample(rng, num_pixels, samples)
            .into_iter()
            .map(index_to_coord)
            .collect(),
    };
    Ok(points)
}

fn diff_of_gaussians(image: &DynamicImage, radii: (u32, u32)) -> Result<Vec<f32>, GeometrizeError> {
    let image_luma8 = DynamicImage::from(image.clone().into_luma8());

    let gauss1 = add_blur(image_luma8.clone(), radii.0)?.to_luma32f();
    let gauss2 = add_blur(image_luma8, radii.1)?.to_luma32f();

    let diff: Vec<f32> = gauss1
        .pixels()
        .zip(gauss2.pixels())
        .map(|(p1, p2)| {
            let d = p1.0[0] - p2.0[0];
            if d < 0.0 { d * 0.1 } else { d }
        })
        .collect();
    let max_diff = diff.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Ok(diff.iter().map(|e| (e.abs() / max_diff).sqrt()).collect())
}

fn add_blur(image: DynamicImage, radius: u32) -> Result<DynamicImage, GeometrizeError> {
    use libblur::{self, AnisotropicRadius};

    libblur::fast_gaussian_blur_image(
        image,
        AnisotropicRadius::new(radius),
        libblur::EdgeMode2D::new(libblur::EdgeMode::Clamp),
        libblur::ThreadingPolicy::Adaptive,
    )
    .ok_or(GeometrizeError::BlurError)
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
