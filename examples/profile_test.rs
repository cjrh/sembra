use image::{DynamicImage, RgbImage, Rgb};
use sembra::{resize, ResizeConfig};

/// Create a test image with realistic content (gradient pattern)
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

fn main() {
    println!("Creating test image...");
    let img = create_test_image(400, 400);

    println!("Running seam carving resize (shrink to 280x280)...");
    let config = ResizeConfig {
        width: Some(280),
        height: Some(280),
        // Use default energy mode (Forward - 2x faster than Backward)
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let _result = resize(img, config).expect("Resize failed");
    let duration = start.elapsed();

    println!("Resize completed in {:?}", duration);
    println!("Profile complete!");
}
