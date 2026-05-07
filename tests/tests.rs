use geometrize::*;

use image::{DynamicImage, RgbImage};
use std::path::{Path, PathBuf};

const SAMPLE_SIZES: &[u32] = &[
    50, 100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
];

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn output_dir(subfolder: Option<&str>) -> PathBuf {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test-output");
    let dir = match subfolder {
        Some(sub) => base.join(sub),
        None => base,
    };
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn setup(infile: &str, outfile: &str, subfolder: Option<&str>) -> (DynamicImage, PathBuf) {
    let infile = sanitize_filename::sanitize(infile);
    let outfile = sanitize_filename::sanitize(outfile);

    let image = image::ImageReader::open(fixtures_dir().join(&infile))
        .unwrap()
        .decode()
        .unwrap();

    (image, output_dir(subfolder).join(format!("{outfile}.png")))
}

fn fetch_rgb_image(url: &str) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let bytes = reqwest::blocking::get(url)?.error_for_status()?.bytes()?;
    Ok(image::load_from_memory(&bytes)?.into_rgb8())
}

pub fn fetch_images(base: &str, filenames: &[&str]) -> (Vec<RgbImage>, Vec<String>) {
    let images = filenames
        .iter()
        .map(|name| fetch_rgb_image(&format!("{base}{name}")).unwrap())
        .collect();

    let names = filenames
        .iter()
        .map(|f| Path::new(f).file_stem().unwrap().to_string_lossy().into())
        .collect();

    (images, names)
}

fn run_geometrize_test(image_name: &str, style: Style, n: u32, subfolder: Option<&str>) {
    let out_name = format!("{:?}_{}k_{}", style, n / 1000, image_name);
    let (image, out_path) = setup(image_name, &out_name, subfolder);
    let output = geometrize(&image, style, n, SamplingParams::default()).unwrap();

    output.save(out_path).unwrap();
}

#[test]
fn test_lowpoly_mountains() {
    for sample_size in SAMPLE_SIZES {
        run_geometrize_test(
            "mountains.jpg",
            Style::Lowpoly,
            *sample_size,
            Some("mountains"),
        );
    }
}

#[test]
fn test_pointillist_mountains() {
    for sample_size in SAMPLE_SIZES {
        run_geometrize_test(
            "mountains.jpg",
            Style::Pointillist { noise: 0.0 },
            *sample_size,
            Some("mountains"),
        );
    }
}

#[test]
fn test_lowpoly_bubbles() {
    run_geometrize_test("bubbles.jpg", Style::Lowpoly, 100_000, Some("bubbles"));
}

#[test]
fn test_lowpoly_launch() {
    run_geometrize_test("launch.jpg", Style::Lowpoly, 70_000, Some("launch"));
}

#[test]
fn test_pointillist_launch() {
    run_geometrize_test("launch.jpg", Style::Pointillist { noise: 0.0 }, 20_000, Some("launch"));
}

#[test]
fn test_lowpoly_aurora() {
    run_geometrize_test("aurora.png", Style::Lowpoly, 100_000, Some("aurora"))
}

#[test]
fn test_pointillist_bubbles() {
    run_geometrize_test(
        "bubbles.jpg",
        Style::Pointillist { noise: 0.0 },
        40_000,
        Some("bubbles"),
    );
    run_geometrize_test(
        "bubbles.jpg",
        Style::Pointillist { noise: 0.5 },
        40_000,
        Some("bubbles"),
    );
    run_geometrize_test(
        "bubbles.jpg",
        Style::Pointillist { noise: 1.0 },
        40_000,
        Some("bubbles"),
    )
}

#[test]
fn test_lowpoly_dice() {
    for sample_size in SAMPLE_SIZES.iter().take(8) {
        run_geometrize_test("dice.png", Style::Lowpoly, *sample_size, Some("dice"));
    }
}

#[test]
fn test_pointillist_dice() {
    for sample_size in SAMPLE_SIZES.iter().take(8) {
        run_geometrize_test(
            "dice.png",
            Style::Pointillist { noise: 0.0 },
            *sample_size,
            Some("dice"),
        );
    }
}

#[test]
fn test_pointillist_noise() {
    for i in 0..=10 {
        run_geometrize_test(
            "bubbles.jpg",
            Style::Pointillist {
                noise: i as f32 * 0.1,
            },
            100_000,
            Some("noise"),
        );
    }
}

const STANDARD_TEST_IMAGE_BASE: &str = "https://raw.githubusercontent.com/mohammadimtiazz/\
    standard-test-images-for-Image-Processing/master/standard_test_images/";

const STANDARD_TEST_FILENAMES: &[&str] = &[
    "HappyFish.jpg",
    "boat.png",
    "cameraman.tif",
    "fruits.png",
    "lena.bmp",
    "lena_color_512.tif",
    "mandril_color.tif",
    "mandril_gray.tif",
    "peppers.png",
    "tulips.png",
];

fn run_standard_images_test(style: Style, subfolder: &str) {
    let (images, names) = fetch_images(STANDARD_TEST_IMAGE_BASE, STANDARD_TEST_FILENAMES);

    for (image, name) in images.into_iter().zip(names) {
        geometrize(
            &DynamicImage::ImageRgb8(image),
            style.clone(),
            10_000,
            SamplingParams::default(),
        )
        .unwrap()
        .save(output_dir(Some(subfolder)).join(format!("{name}.png")))
        .unwrap();
    }
}

#[test]
fn test_standard_lowpoly() {
    run_standard_images_test(Style::Lowpoly, "standard-images-lowpoly");
}

#[test]
fn test_standard_pointillist() {
    run_standard_images_test(
        Style::Pointillist { noise: 1.0 },
        "standard-images-pointillist",
    );
}
