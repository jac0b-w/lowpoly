use std::marker::PhantomData;

use crate::{ShapeKind, Style};

#[derive(Debug, Clone)]
pub struct Empty;
#[derive(Debug, Clone)]
pub struct HasShape;

#[derive(Debug, Clone)]
pub struct CustomStyleBuilder<State> {
    _state: PhantomData<State>,
    shapes: Vec<ShapeKind>,
    noise: f32,
}

impl<State> CustomStyleBuilder<State> {
    pub fn with_noise(self, noise: f32) -> Self {
        Self {
            noise,
            ..self
        }
    }
}

impl CustomStyleBuilder<Empty> {
    pub fn new() -> Self {
        Self {
            _state: PhantomData,
            shapes: Vec::new(),
            noise: 0.0
        }
    }
    fn with_shape(self, shape: ShapeKind) -> CustomStyleBuilder<HasShape> {
        CustomStyleBuilder {
            _state: PhantomData,
            shapes: vec![shape],
            noise: 0.0,
        }
    }
    pub fn circle(self) -> CustomStyleBuilder<HasShape> {
        self.with_shape(ShapeKind::circle())
    }
    pub fn polygon(self, sides: u8) -> CustomStyleBuilder<HasShape> {
        self.with_shape(ShapeKind::polygon(sides))
    }
    pub fn triangle(self) -> CustomStyleBuilder<HasShape> {
        self.with_shape(ShapeKind::polygon(3))
    }
    pub fn square(self) -> CustomStyleBuilder<HasShape> {
        self.with_shape(ShapeKind::polygon(4))
    }
    pub fn pentagon(self) -> CustomStyleBuilder<HasShape> {
        self.with_shape(ShapeKind::polygon(5))
    }
    pub fn hexagon(self) -> CustomStyleBuilder<HasShape> {
        self.with_shape(ShapeKind::polygon(6))
    }
    pub fn heptagon(self) -> CustomStyleBuilder<HasShape> {
        self.with_shape(ShapeKind::polygon(7))
    }
    pub fn octagon(self) -> CustomStyleBuilder<HasShape> {
        self.with_shape(ShapeKind::polygon(8))
    }
    pub fn nonagon(self) -> CustomStyleBuilder<HasShape> {
        self.with_shape(ShapeKind::polygon(9))
    }
}

impl CustomStyleBuilder<HasShape> {
    fn with_shape(self, shape: ShapeKind) -> CustomStyleBuilder<HasShape> {
        let mut shapes = self.shapes;
        shapes.push(shape);
        CustomStyleBuilder {
            shapes,
            noise: self.noise,
            ..self
        }
    }
    pub fn build(&self) -> Style {
        Style::Custom {
            shapes: self.shapes.clone(),
            noise: self.noise,
        }
    }
    pub fn circle(self) -> Self {
        self.with_shape(ShapeKind::circle())
    }
    pub fn polygon(self, sides: u8) -> Self {
        self.with_shape(ShapeKind::polygon(sides))
    }
    pub fn triangle(self) -> Self {
        self.with_shape(ShapeKind::polygon(3))
    }
    pub fn square(self) -> Self {
        self.with_shape(ShapeKind::polygon(4))
    }
    pub fn pentagon(self) -> Self {
        self.with_shape(ShapeKind::polygon(5))
    }
    pub fn hexagon(self) -> Self {
        self.with_shape(ShapeKind::polygon(6))
    }
    pub fn heptagon(self) -> Self {
        self.with_shape(ShapeKind::polygon(7))
    }
    pub fn octagon(self) -> Self {
        self.with_shape(ShapeKind::polygon(8))
    }
    pub fn nonagon(self) -> Self {
        self.with_shape(ShapeKind::polygon(9))
    }
}
