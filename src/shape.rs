use core::f32;

use crate::{ColorSampler, GeometrizeError, point::Point};
use image::{Rgba, RgbaImage};

#[derive(Debug, Clone)]
pub(crate) struct DrawableShape {
    shape: ShapeKind,
    color: Rgba<u8>,
}

impl DrawableShape {
    pub fn draw(self, image: &mut RgbaImage) {
        match self.shape {
            ShapeKind::Polygon(polygon) => {
                let vertices: Vec<_> = polygon
                    .vertices
                    .iter()
                    .map(|v| imageproc::point::Point::new(v.x as i32, v.y as i32))
                    .collect();
                imageproc::drawing::draw_antialiased_polygon_mut(
                    &mut *image,
                    &vertices,
                    self.color,
                    imageproc::pixelops::interpolate,
                );
            }
            ShapeKind::Circle(circle) => imageproc::drawing::draw_filled_circle_mut(
                &mut *image,
                (circle.center.x as i32, circle.center.y as i32),
                circle.radius as i32,
                self.color,
            ),
        }
    }
    pub fn from_shape_image(
        shape: &ShapeKind,
        image: &RgbaImage,
        color_sampler: &ColorSampler,
    ) -> Self {
        let color = match color_sampler {
            ColorSampler::All => avg_color_of_shape(shape, image),
            ColorSampler::Single => {
                let centroid = shape.centroid().clamped_to_frame(image.dimensions());
                *image.get_pixel(centroid.x as u32, centroid.y as u32)
            }
        };
        Self {
            shape: shape.clone(),
            color,
        }
    }
}

pub(crate) struct BoundingBox {
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
}

impl BoundingBox {
    fn from_vertices(vertices: &[Point], frame: (u32, u32)) -> Self {
        let (min_x, max_x, min_y, max_y) =
            vertices.iter().fold((u32::MAX, 0, u32::MAX, 0), |acc, p| {
                let p = p.clamped_to_frame(frame);
                (
                    acc.0.min(p.x.floor().max(0.0) as u32),
                    acc.1.max(p.x.ceil().min(frame.0 as f32 - 1.0) as u32),
                    acc.2.min(p.y.floor().max(0.0) as u32),
                    acc.3.max(p.y.ceil().min(frame.1 as f32 - 1.0) as u32),
                )
            });
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }
    fn from_corners(top_left: Point, bottom_right: Point, frame: (u32, u32)) -> Self {
        Self {
            min_x: top_left.x.floor().max(0.0) as u32,
            max_x: bottom_right.x.ceil().min(frame.0 as f32 - 1.0) as u32,
            min_y: top_left.y.floor().max(0.0) as u32,
            max_y: bottom_right.y.ceil().min(frame.1 as f32 - 1.0) as u32,
        }
    }
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = (u32, u32)> + 'a> {
        Box::new(
            (self.min_y..=self.max_y)
                .flat_map(move |y| (self.min_x..=self.max_x).map(move |x| (x, y))),
        )
    }
    // fn sample_iter<'a>(&'a self) -> Box<dyn Iterator<Item = (u32, u32)> + 'a> {
    //     let mut rng: SmallRng = rand::make_rng();
    //     Box::new(std::iter::repeat_with(move || {
    //         let rand_x = rng.random_range(self.min_x..self.max_x);
    //         let rand_y = rng.random_range(self.min_y..self.max_y);
    //         (rand_x, rand_y)
    //     }))
    // }
}

pub(crate) trait Shape {
    fn contains(&self, p: Point) -> bool;
    fn transformed(&self, scale: f32, translation: [f32; 2], rotation: f32) -> Self;
    fn centroid(&self) -> Point;
    fn bbox(&self, frame: (u32, u32)) -> BoundingBox;
}

#[derive(Debug, Clone)]
pub enum ShapeKind {
    Polygon(Polygon),
    Circle(Circle),
}

impl ShapeKind {
    pub fn polygon(sides: u8) -> Self {
        ShapeKind::Polygon(Polygon::try_from_num_sides(sides).unwrap())
    }
    pub fn circle() -> Self {
        ShapeKind::Circle(Circle::default())
    }
}

impl Shape for ShapeKind {
    fn contains(&self, p: Point) -> bool {
        match self {
            ShapeKind::Polygon(polygon) => polygon.contains(p),
            ShapeKind::Circle(circle) => circle.contains(p),
        }
    }
    fn transformed(&self, scale: f32, translation: [f32; 2], rotation: f32) -> Self {
        match self {
            ShapeKind::Polygon(polygon) => {
                ShapeKind::Polygon(polygon.transformed(scale, translation, rotation))
            }
            ShapeKind::Circle(circle) => {
                ShapeKind::Circle(circle.transformed(scale, translation, rotation))
            }
        }
    }
    fn centroid(&self) -> Point {
        match self {
            ShapeKind::Polygon(polygon) => polygon.centroid(),
            ShapeKind::Circle(circle) => circle.centroid(),
        }
    }
    fn bbox(&self, frame: (u32, u32)) -> BoundingBox {
        match self {
            ShapeKind::Polygon(polygon) => polygon.bbox(frame),
            ShapeKind::Circle(circle) => circle.bbox(frame),
        }
    }
}

fn centroid(vertices: &[Point]) -> Point {
    let n = vertices.len() as f32;
    let (sum_x, sum_y) = vertices
        .iter()
        .fold((0.0, 0.0), |acc, p| (acc.0 + p.x, acc.1 + p.y));
    Point::new(sum_x / n, sum_y / n)
}

pub fn avg_color_of_shape(shape: &ShapeKind, image: &RgbaImage) -> Rgba<u8> {
    let (sum, count) = shape
        .bbox(image.dimensions())
        .iter()
        .filter(|&(x, y)| shape.contains([x as f32, y as f32].into()))
        .map(|(x, y)| *image.get_pixel(x, y))
        .fold(([0u64; 4], 0u64), |(mut acc, count), px| {
            acc.iter_mut().zip(px.0).for_each(|(a, b)| *a += b as u64);
            (acc, count + 1)
        });

    if count == 0 {
        return Rgba([0; 4]);
    }
    Rgba(sum.map(|s| (s / count) as u8))
}

#[derive(Debug, Clone)]
pub struct Polygon {
    vertices: Vec<Point>,
}

impl Polygon {
    pub fn try_from_num_sides(n: u8) -> Result<Self, GeometrizeError> {
        if n < 3 {
            return Err(GeometrizeError::PolgonConstructionError(n as usize));
        }

        let apothem = |n| (f32::consts::PI / n as f32).cos();
        let r = 1.0 / apothem(n);

        let vertices: Vec<Point> = (0..n)
            .map(|i| {
                let (y_cmp, x_cmp) = (2.0 * f32::consts::PI * i as f32 / n as f32).sin_cos();

                Point {
                    x: r * x_cmp,
                    y: r * y_cmp,
                }
            })
            .collect();

        Ok(Self { vertices })
    }
    fn sort_clockwise(mut vertices: Vec<Point>) -> Vec<Point> {
        let centroid = centroid(&vertices);

        vertices.sort_by(|a, b| {
            let angle_a = (a.y - centroid.y).atan2(a.x - centroid.x);
            let angle_b = (b.y - centroid.y).atan2(b.x - centroid.x);
            angle_a.total_cmp(&angle_b)
        });

        vertices
    }
    // pub fn furthest_vertex(&self) -> f32 {
    //     let c = self.centroid();
    //     self.vertices
    //         .iter()
    //         .map(|v| f32::hypot(v.x - c.x, v.y - c.y))
    //         .max_by(f32::total_cmp)
    //         .expect("Polygon should not have 0 vertices.")
    // }
    /// The circumcenter if the polygon is a triangle
    pub fn circumcenter(&self) -> Option<Point> {
        let [a, b, c] = self.vertices.as_slice() else {
            return None;
        };

        let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));

        let norm_squared = |vertex: &Point| vertex.x.powi(2) + vertex.y.powi(2);

        let x = norm_squared(a) * (b.y - c.y)
            + norm_squared(b) * (c.y - a.y)
            + norm_squared(c) * (a.y - b.y);
        let y = norm_squared(a) * (c.x - b.x)
            + norm_squared(b) * (a.x - c.x)
            + norm_squared(c) * (b.x - a.x);

        Some([x / d, y / d].into())
    }
    pub fn circumradius(&self) -> Option<f32> {
        let [a, b, c] = self.vertices.as_slice() else {
            return None;
        };

        let area = (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)).abs() / 2.0;

        let circumradius = (a.dist(*b) * b.dist(*c) * c.dist(*a)) / (4.0 * area);

        Some(circumradius)
    }
}

impl Shape for Polygon {
    fn contains(&self, p: Point) -> bool {
        let vertices = &self.vertices;
        let Point { x, y } = p;

        let mut pos = false;
        let mut neg = false;

        let edge_points = vertices
            .iter()
            .zip(vertices[1..].iter().chain(std::iter::once(&vertices[0])));

        for (a, b) in edge_points {
            let (dx, dy) = (b.x - a.x, b.y - a.y);
            // Cross product z-component: (b-a) × (p-a)
            let z = dx * (y - a.y) - dy * (x - a.x);
            if z >= 0.0 {
                pos = true;
            } else if z < 0.0 {
                neg = true;
            }
            if pos && neg {
                return false;
            } // early exit
        }
        true
    }
    fn transformed(&self, scale: f32, translation: [f32; 2], rotation: f32) -> Self {
        let angle = 2.0 * f32::consts::PI * rotation;
        let [dx, dy] = translation;

        let vertices = self
            .vertices
            .iter()
            .map(|v| v.scaled(scale).rotated(angle).translated(dx, dy))
            .collect();

        Self { vertices }
    }
    fn centroid(&self) -> Point {
        centroid(&self.vertices)
    }
    fn bbox(&self, frame: (u32, u32)) -> BoundingBox {
        BoundingBox::from_vertices(&self.vertices, frame)
    }
}

impl TryFrom<Vec<Point>> for Polygon {
    type Error = GeometrizeError;

    fn try_from(vertices: Vec<Point>) -> Result<Self, Self::Error> {
        let mut vertices = Self::sort_clockwise(vertices);
        vertices
            .dedup_by(|a, b| (a.x - b.x).abs() < f32::EPSILON && (a.y - b.y).abs() < f32::EPSILON);

        let n = vertices.len();
        if n < 3 {
            return Err(GeometrizeError::PolgonConstructionError(n));
        }

        Ok(Self { vertices })
    }
}

#[derive(Debug, Clone)]
pub struct Circle {
    center: Point,
    radius: f32,
}

impl Default for Circle {
    fn default() -> Self {
        Self {
            center: Point::new(0.0, 0.0),
            radius: 1.0,
        }
    }
}

impl Shape for Circle {
    fn contains(&self, p: Point) -> bool {
        f32::hypot(self.center.x - p.x, self.center.y - p.y) <= self.radius
    }
    fn transformed(&self, scale: f32, translation: [f32; 2], _rotation: f32) -> Self {
        let [dx, dy] = translation;
        Self {
            center: self.center.translated(dx, dy),
            radius: self.radius * scale,
        }
    }
    fn centroid(&self) -> Point {
        self.center
    }
    fn bbox(&self, frame: (u32, u32)) -> BoundingBox {
        let top_left = self.center.translated(-self.radius, -self.radius);
        let bottom_right = self.center.translated(self.radius, self.radius);
        BoundingBox::from_corners(top_left, bottom_right, frame)
    }
}
