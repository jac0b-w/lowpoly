use criterion::{
    BatchSize, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use geometrize::*;
use image::{GenericImageView, ImageReader};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Duration;
use thousands::Separable;

const SAMPLE_SIZES: &[u32] = &[
    50, 100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
];

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn bench_large_image(c: &mut Criterion, style: Style) {
    let large_image = ImageReader::open(fixtures_dir().join("mountains.jpg"))
        .unwrap()
        .decode()
        .unwrap();
    let (w, h) = large_image.dimensions();

    let mut group = c.benchmark_group(format!("{style:?} Large Image ({w}x{h}) - Sample Sizes"));
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));

    for &sample_sizes in SAMPLE_SIZES {
        group.bench_with_input(
            BenchmarkId::from_parameter(sample_sizes.separate_with_commas()),
            &sample_sizes,
            |b, &n| {
                b.iter_batched(
                    || large_image.clone(),
                    |image| {
                        geometrize(
                            black_box(&image),
                            black_box(style.clone()),
                            black_box(n),
                            black_box(SamplingParams::default()),
                        )
                    },
                    BatchSize::LargeInput,
                )
            },
        );
    }
    group.finish();
}

pub fn large_image_lowpoly(c: &mut Criterion) {
    bench_large_image(c, Style::Lowpoly);
}

pub fn large_image_pointillist(c: &mut Criterion) {
    bench_large_image(c, Style::Pointillist { noise: 0.5 });
}

criterion_group!(benches, large_image_lowpoly, large_image_pointillist);
criterion_main!(benches);
