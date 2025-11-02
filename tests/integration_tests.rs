use image::{DynamicImage, RgbImage, Rgb, GenericImageView};
use sembra::{resize, ResizeConfig, EnergyMode, ResizeOrder};

/// Create a test image with a gradient pattern
fn create_gradient_image(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = 128;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// Create a test image with a vertical stripe pattern
fn create_stripe_image(width: u32, height: u32, stripe_width: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let is_white = (x / stripe_width).is_multiple_of(2);
            let color = if is_white { 255 } else { 0 };
            img.put_pixel(x, y, Rgb([color, color, color]));
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// Create a test image with a central square object
fn create_object_image(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    let square_size = width / 4;
    let square_start = width / 2 - square_size / 2;
    let square_end = square_start + square_size;

    for y in 0..height {
        for x in 0..width {
            let is_square = x >= square_start && x < square_end
                         && y >= square_start && y < square_end;
            let color = if is_square {
                Rgb([255, 0, 0])  // Red square
            } else {
                Rgb([200, 200, 200])  // Gray background
            };
            img.put_pixel(x, y, color);
        }
    }
    DynamicImage::ImageRgb8(img)
}

#[test]
fn test_shrink_width_backward() {
    let img = create_gradient_image(100, 80);
    let config = ResizeConfig {
        width: Some(60),
        height: None,
        energy_mode: EnergyMode::Backward,
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (60, 80));
}

#[test]
fn test_shrink_height_backward() {
    let img = create_gradient_image(100, 80);
    let config = ResizeConfig {
        width: None,
        height: Some(50),
        energy_mode: EnergyMode::Backward,
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (100, 50));
}

#[test]
fn test_shrink_both_dimensions_backward() {
    let img = create_gradient_image(100, 80);
    let config = ResizeConfig {
        width: Some(60),
        height: Some(50),
        energy_mode: EnergyMode::Backward,
        order: ResizeOrder::WidthFirst,
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (60, 50));
}

#[test]
fn test_shrink_both_dimensions_height_first() {
    let img = create_gradient_image(100, 80);
    let config = ResizeConfig {
        width: Some(60),
        height: Some(50),
        energy_mode: EnergyMode::Backward,
        order: ResizeOrder::HeightFirst,
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (60, 50));
}

#[test]
fn test_expand_width_backward() {
    let img = create_gradient_image(50, 40);
    let config = ResizeConfig {
        width: Some(80),
        height: None,
        energy_mode: EnergyMode::Backward,
        step_ratio: 0.5,
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (80, 40));
}

#[test]
fn test_expand_height_backward() {
    let img = create_gradient_image(50, 40);
    let config = ResizeConfig {
        width: None,
        height: Some(70),
        energy_mode: EnergyMode::Backward,
        step_ratio: 0.5,
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (50, 70));
}

#[test]
fn test_expand_both_dimensions() {
    let img = create_gradient_image(50, 40);
    let config = ResizeConfig {
        width: Some(80),
        height: Some(70),
        energy_mode: EnergyMode::Backward,
        step_ratio: 0.5,
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (80, 70));
}

#[test]
fn test_shrink_width_forward_energy() {
    let img = create_gradient_image(100, 80);
    let config = ResizeConfig {
        width: Some(60),
        height: None,
        energy_mode: EnergyMode::Forward,
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (60, 80));
}

#[test]
fn test_no_change_dimensions() {
    let img = create_gradient_image(100, 80);
    let config = ResizeConfig {
        width: Some(100),
        height: Some(80),
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (100, 80));
}

#[test]
fn test_stripe_pattern_preservation() {
    // Vertical stripes should be better preserved than horizontal seams
    let img = create_stripe_image(100, 80, 10);
    let config = ResizeConfig {
        width: Some(60),
        height: None,
        energy_mode: EnergyMode::Backward,
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (60, 80));

    // Basic sanity check: result should have 3 channels (RGB)
    let rgb = result.to_rgb8();
    assert_eq!(rgb.dimensions(), (60, 80));
}

#[test]
fn test_mixed_shrink_expand() {
    // Shrink width, expand height
    let img = create_gradient_image(100, 50);
    let config = ResizeConfig {
        width: Some(60),
        height: Some(80),
        energy_mode: EnergyMode::Backward,
        step_ratio: 0.5,
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (60, 80));
}

#[test]
fn test_output_is_valid_rgb() {
    let img = create_gradient_image(50, 40);
    let config = ResizeConfig {
        width: Some(30),
        height: Some(25),
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    let rgb = result.to_rgb8();

    // Check that result is valid RGB (dimensions match and can be converted)
    assert_eq!(rgb.dimensions(), (30, 25));
    // u8 channels are always valid, just verify we got pixels
    assert_eq!(rgb.pixels().count(), 30 * 25);
}

#[test]
fn test_large_reduction() {
    // Test reducing to 50% of original size
    let img = create_gradient_image(200, 160);
    let config = ResizeConfig {
        width: Some(100),
        height: Some(80),
        energy_mode: EnergyMode::Backward,
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (100, 80));
}

#[test]
fn test_large_expansion() {
    // Test expanding to 2x original size
    let img = create_gradient_image(50, 40);
    let config = ResizeConfig {
        width: Some(100),
        height: Some(80),
        energy_mode: EnergyMode::Backward,
        step_ratio: 0.5,
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (100, 80));
}

#[test]
fn test_object_centered_image() {
    // Test with an image that has a clear central object
    let img = create_object_image(100, 100);
    let config = ResizeConfig {
        width: Some(60),
        height: None,
        energy_mode: EnergyMode::Backward,
        ..Default::default()
    };

    let result = resize(img, config).expect("Resize failed");
    assert_eq!(result.dimensions(), (60, 100));

    // The red square should still be present (basic check)
    let rgb = result.to_rgb8();
    let has_red = rgb.pixels().any(|p| p[0] > 200 && p[1] < 50 && p[2] < 50);
    assert!(has_red, "Red object should be preserved after resize");
}

#[test]
fn test_consistency_across_orders() {
    // Both orders should produce valid results with same dimensions
    let img1 = create_gradient_image(100, 80);
    let img2 = create_gradient_image(100, 80);

    let config1 = ResizeConfig {
        width: Some(60),
        height: Some(50),
        energy_mode: EnergyMode::Backward,
        order: ResizeOrder::WidthFirst,
        ..Default::default()
    };

    let config2 = ResizeConfig {
        width: Some(60),
        height: Some(50),
        energy_mode: EnergyMode::Backward,
        order: ResizeOrder::HeightFirst,
        ..Default::default()
    };

    let result1 = resize(img1, config1).expect("Resize failed");
    let result2 = resize(img2, config2).expect("Resize failed");

    assert_eq!(result1.dimensions(), (60, 50));
    assert_eq!(result2.dimensions(), (60, 50));
}
