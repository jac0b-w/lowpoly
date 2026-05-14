use std::ops::{Add, Div};

#[derive(Debug, Copy, Clone)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    pub fn scaled(&self, scale: f32) -> Self {
        Self {
            x: self.x * scale,
            y: self.y * scale,
        }
    }
    pub fn rotated(&self, angle: f32) -> Self {
        let (sin_theta, cos_theta) = angle.sin_cos();
        Self {
            x: self.x * cos_theta - self.y * sin_theta,
            y: self.x * sin_theta + self.y * cos_theta,
        }
    }
    pub fn translated(&self, dx: f32, dy: f32) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
        }
    }
    pub fn near(&self, other: Point) -> bool {
        self.x_near(other.x) && self.y_near(other.y)
    }
    pub fn x_near(&self, x: f32) -> bool {
        near(self.x, x)
    }
    pub fn y_near(&self, y: f32) -> bool {
        near(self.y, y)
    }
    pub fn dist(&self, other: Point) -> f32 {
        f32::hypot(self.x - other.x, self.y - other.y)
    }
    pub fn clamped_to_frame(&self, frame: (u32, u32)) -> Self {
        Self {
            x: self.x.clamp(0.0, frame.0 as f32 - 0.01),
            y: self.y.clamp(0.0, frame.1 as f32 - 0.01),
        }
    }
    pub fn clamped_to_frame_direction(&self, direction: &[f32; 2], frame: &(u32, u32)) -> Point {
        let [dir_x, dir_y] = *direction;
        let (w, h) = (frame.0 as f32, frame.1 as f32);

        let ray = |dir_x: f32, dir_y: f32| {
            let mut t = f32::INFINITY;
            if dir_x > 0.0 {
                t = t.min((w - self.x) / dir_x);
            }
            if dir_x < 0.0 {
                t = t.min(-self.x / dir_x);
            }
            if dir_y > 0.0 {
                t = t.min((h - self.y) / dir_y);
            }
            if dir_y < 0.0 {
                t = t.min(-self.y / dir_y);
            }
            (t, Point::from([self.x + dir_x * t, self.y + dir_y * t]))
        };

        let (t1, p1) = ray(dir_x, dir_y);
        let (t2, p2) = ray(-dir_x, -dir_y);
        if t1 < t2 { p1 } else { p2 }
    }
}

impl Add for Point {
    type Output = Point;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Div<f32> for Point {
    type Output = Point;
    fn div(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl From<Point> for [f32; 2] {
    fn from(p: Point) -> Self {
        [p.x, p.y]
    }
}

impl From<[f32; 2]> for Point {
    fn from(array: [f32; 2]) -> Self {
        Self {
            x: array[0],
            y: array[1],
        }
    }
}

impl From<Point> for (f32, f32) {
    fn from(p: Point) -> Self {
        (p.x, p.y)
    }
}

impl From<Point> for imageproc::point::Point<i32> {
    fn from(p: Point) -> Self {
        imageproc::point::Point::new(p.x as i32, p.y as i32)
    }
}

impl From<Point> for (i32, i32) {
    fn from(p: Point) -> Self {
        (p.x as i32, p.y as i32)
    }
}

impl From<spade::Point2<f32>> for Point {
    fn from(point: spade::Point2<f32>) -> Self {
        Self {
            x: point.x,
            y: point.y,
        }
    }
}

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-4
}
