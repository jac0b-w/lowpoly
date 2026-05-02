use criterion::{BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main};
use image::{GenericImageView, ImageReader};
use lowpoly::{SampleSeed, lowpoly, lowpoly_with_seed};
use std::hint::black_box;
use std::time::Duration;

const LARGE_IMAGE_PATH: &str = "benches/inputs/large_image.png";

pub fn large_image(c: &mut Criterion) {
    let large_image = ImageReader::open(LARGE_IMAGE_PATH)
        .unwrap()
        .decode()
        .unwrap();
    let (w, h) = large_image.dimensions();

    let mut group = c.benchmark_group(format!("Sample Sizes - Large Image ({w}x{h})"));
    group.measurement_time(Duration::from_secs(30));

    let plot_config = PlotConfiguration::default().summary_scale(criterion::AxisScale::Linear);
    group.plot_config(plot_config);

    for sample_sizes in vec![
        50, 100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000,
        10_000_000,
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(sample_sizes),
            &sample_sizes,
            |b, &n| b.iter(|| lowpoly(black_box(large_image.clone()), black_box(n))),
        );
    }
    group.finish();
}

pub fn compare_seeds(c: &mut Criterion) {
    let large_image = ImageReader::open(LARGE_IMAGE_PATH)
        .unwrap()
        .decode()
        .unwrap();
    let (w, h) = large_image.dimensions();

    let n = 100_000;

    let mut group = c.benchmark_group(format!("Seed Type - Large Image ({w}x{h})"));
    group.measurement_time(Duration::from_secs(120));

    for seeding in vec![SampleSeed::Random, SampleSeed::Custom(100)] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", seeding)),
            &seeding,
            |b, &s| {
                b.iter(|| {
                    lowpoly_with_seed(
                        black_box(large_image.clone()),
                        black_box(n),
                        black_box(s),
                        black_box(lowpoly::EdgePoints::Auto),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, large_image);
criterion_main!(benches);
