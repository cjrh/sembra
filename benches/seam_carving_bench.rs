use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use image::{DynamicImage, RgbImage, Rgb};
use sembra::{resize, ResizeConfig, EnergyMode, ResizeOrder};

/// Create a gradient test image
fn create_test_image(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = ((x + y) as f32 / (width + height) as f32 * 255.0) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// Benchmark shrinking width (most common operation)
fn bench_shrink_width(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrink_width");

    // Test various image sizes
    for &(width, height) in &[(100, 100), (200, 200), (400, 400), (800, 600)] {
        let img = create_test_image(width, height);
        let target_width = (width as f32 * 0.7) as usize;  // Shrink by 30%

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", width, height)),
            &img,
            |b, img| {
                b.iter(|| {
                    let config = ResizeConfig {
                        width: Some(target_width),
                        height: None,
                        energy_mode: EnergyMode::Backward,
                        ..Default::default()
                    };
                    resize(black_box(img.clone()), config).unwrap()
                });
            },
        );
    }
    group.finish();
}

/// Benchmark expanding width
fn bench_expand_width(c: &mut Criterion) {
    let mut group = c.benchmark_group("expand_width");

    // Test smaller images (expansion is more expensive)
    for &(width, height) in &[(100, 100), (200, 200), (400, 400)] {
        let img = create_test_image(width, height);
        let target_width = (width as f32 * 1.3) as usize;  // Expand by 30%

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", width, height)),
            &img,
            |b, img| {
                b.iter(|| {
                    let config = ResizeConfig {
                        width: Some(target_width),
                        height: None,
                        energy_mode: EnergyMode::Backward,
                        step_ratio: 0.5,
                        ..Default::default()
                    };
                    resize(black_box(img.clone()), config).unwrap()
                });
            },
        );
    }
    group.finish();
}

/// Benchmark shrinking both dimensions
fn bench_shrink_both(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrink_both");

    for &(width, height) in &[(100, 100), (200, 200), (400, 400)] {
        let img = create_test_image(width, height);
        let target_width = (width as f32 * 0.7) as usize;
        let target_height = (height as f32 * 0.7) as usize;

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", width, height)),
            &img,
            |b, img| {
                b.iter(|| {
                    let config = ResizeConfig {
                        width: Some(target_width),
                        height: Some(target_height),
                        energy_mode: EnergyMode::Backward,
                        order: ResizeOrder::WidthFirst,
                        ..Default::default()
                    };
                    resize(black_box(img.clone()), config).unwrap()
                });
            },
        );
    }
    group.finish();
}

/// Benchmark forward vs backward energy
fn bench_energy_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_modes");
    let img = create_test_image(200, 200);
    let target_width = 140;

    group.bench_function("backward_energy", |b| {
        b.iter(|| {
            let config = ResizeConfig {
                width: Some(target_width),
                height: None,
                energy_mode: EnergyMode::Backward,
                ..Default::default()
            };
            resize(black_box(img.clone()), config).unwrap()
        });
    });

    group.bench_function("forward_energy", |b| {
        b.iter(|| {
            let config = ResizeConfig {
                width: Some(target_width),
                height: None,
                energy_mode: EnergyMode::Forward,
                ..Default::default()
            };
            resize(black_box(img.clone()), config).unwrap()
        });
    });

    group.finish();
}

/// Benchmark resize order (width-first vs height-first)
fn bench_resize_order(c: &mut Criterion) {
    let mut group = c.benchmark_group("resize_order");
    let img = create_test_image(200, 200);
    let target_width = 140;
    let target_height = 140;

    group.bench_function("width_first", |b| {
        b.iter(|| {
            let config = ResizeConfig {
                width: Some(target_width),
                height: Some(target_height),
                energy_mode: EnergyMode::Backward,
                order: ResizeOrder::WidthFirst,
                ..Default::default()
            };
            resize(black_box(img.clone()), config).unwrap()
        });
    });

    group.bench_function("height_first", |b| {
        b.iter(|| {
            let config = ResizeConfig {
                width: Some(target_width),
                height: Some(target_height),
                energy_mode: EnergyMode::Backward,
                order: ResizeOrder::HeightFirst,
                ..Default::default()
            };
            resize(black_box(img.clone()), config).unwrap()
        });
    });

    group.finish();
}

/// Benchmark different step ratios for expansion
fn bench_step_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_ratios");
    let img = create_test_image(100, 100);
    let target_width = 150;  // 50% expansion

    for &ratio in &[0.25, 0.5, 0.75, 1.0] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("ratio_{}", ratio)),
            &ratio,
            |b, &ratio| {
                b.iter(|| {
                    let config = ResizeConfig {
                        width: Some(target_width),
                        height: None,
                        energy_mode: EnergyMode::Backward,
                        step_ratio: ratio,
                        ..Default::default()
                    };
                    resize(black_box(img.clone()), config).unwrap()
                });
            },
        );
    }
    group.finish();
}

/// Benchmark large reduction (stress test)
fn bench_large_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_reduction");
    group.sample_size(10);  // Reduce sample size for expensive operations

    let img = create_test_image(400, 400);

    group.bench_function("50_percent", |b| {
        b.iter(|| {
            let config = ResizeConfig {
                width: Some(200),
                height: Some(200),
                energy_mode: EnergyMode::Backward,
                ..Default::default()
            };
            resize(black_box(img.clone()), config).unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_shrink_width,
    bench_expand_width,
    bench_shrink_both,
    bench_energy_modes,
    bench_resize_order,
    bench_step_ratios,
    bench_large_reduction,
);
criterion_main!(benches);
