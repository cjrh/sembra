//! # Sembra - Content-Aware Image Resizing
//!
//! Sembra implements seam carving, an algorithm for content-aware image resizing.
//! Instead of uniformly scaling, seam carving removes or adds "seams" (paths of pixels)
//! that have low importance, preserving the more important features of the image.
//!
//! ## How It Works
//!
//! 1. **Energy calculation**: Determine the importance of each pixel using gradient or forward energy
//! 2. **Seam finding**: Use dynamic programming to find minimum-energy vertical paths
//! 3. **Seam removal/insertion**: Remove low-energy seams to shrink, or duplicate them to expand
//! 4. **Masking**: Optionally protect or target specific regions for removal
//!
//! ## Example
//!
//! ```no_run
//! use sembra::{resize, ResizeConfig, EnergyMode, ResizeOrder};
//! use image;
//!
//! // Load an image
//! let img = image::open("input.jpg").unwrap();
//!
//! // Configure resize
//! let config = ResizeConfig {
//!     width: Some(400),
//!     height: Some(300),
//!     energy_mode: EnergyMode::Backward,
//!     order: ResizeOrder::WidthFirst,
//!     keep_mask: None,
//!     drop_mask: None,
//!     step_ratio: 0.5,
//! };
//!
//! // Perform seam carving
//! let resized = resize(img, config).unwrap();
//!
//! // Save the result
//! resized.save("output.jpg").unwrap();
//! ```

use image::{DynamicImage, RgbImage, Rgb};
use ndarray::{Array2, Array3, s, Axis, Zip};
use std::fmt;

// Constants for mask energy values
const DROP_MASK_ENERGY: f32 = 1e5;
const KEEP_MASK_ENERGY: f32 = 1e3;

/// Errors that can occur during seam carving operations
#[derive(Debug, Clone)]
pub enum SeamCarvingError {
    /// Target dimension is invalid (e.g., zero or larger than reasonable limits)
    InvalidDimensions { message: String },
    /// Mask dimensions don't match image dimensions
    MaskSizeMismatch { expected: (usize, usize), got: (usize, usize) },
    /// Step ratio is invalid (must be > 0 and <= 1)
    InvalidStepRatio { value: f32 },
    /// Image has invalid dimensions for processing
    InvalidImageDimensions { message: String },
    /// General processing error
    ProcessingError { message: String },
}

impl fmt::Display for SeamCarvingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeamCarvingError::InvalidDimensions { message } => {
                write!(f, "Invalid dimensions: {}", message)
            }
            SeamCarvingError::MaskSizeMismatch { expected, got } => {
                write!(f, "Mask size mismatch: expected {:?}, got {:?}", expected, got)
            }
            SeamCarvingError::InvalidStepRatio { value } => {
                write!(f, "Invalid step ratio: {} (must be > 0 and <= 1)", value)
            }
            SeamCarvingError::InvalidImageDimensions { message } => {
                write!(f, "Invalid image dimensions: {}", message)
            }
            SeamCarvingError::ProcessingError { message } => {
                write!(f, "Processing error: {}", message)
            }
        }
    }
}

impl std::error::Error for SeamCarvingError {}

/// Energy calculation mode for seam finding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyMode {
    /// Backward energy: gradient-based (edge detection using Sobel-like filter)
    Backward,
    /// Forward energy: removal cost-based (estimates discontinuity after removal)
    Forward,
}

/// Order of operations when resizing both width and height
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeOrder {
    /// Resize width first, then height
    WidthFirst,
    /// Resize height first, then width
    HeightFirst,
}

/// Configuration for seam carving resize operation
#[derive(Debug, Clone)]
pub struct ResizeConfig {
    /// Target width (None to keep original width)
    pub width: Option<usize>,
    /// Target height (None to keep original height)
    pub height: Option<usize>,
    /// Energy calculation mode
    pub energy_mode: EnergyMode,
    /// Resize order when changing both dimensions
    pub order: ResizeOrder,
    /// Optional mask to protect regions (true = keep)
    pub keep_mask: Option<Array2<bool>>,
    /// Optional mask to remove regions (true = remove)
    pub drop_mask: Option<Array2<bool>>,
    /// Step ratio for expansion (0 < ratio <= 1)
    pub step_ratio: f32,
}

impl Default for ResizeConfig {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            energy_mode: EnergyMode::Backward,
            order: ResizeOrder::WidthFirst,
            keep_mask: None,
            drop_mask: None,
            step_ratio: 0.5,
        }
    }
}

/// Convert an image from the `image` crate to an ndarray (f32, range 0-255).
/// Internal function for converting DynamicImage to Array3<f32>.
fn image_to_ndarray(img: &DynamicImage) -> Array3<f32> {
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    let mut arr = Array3::<f32>::zeros((height as usize, width as usize, 3));
    for (x, y, pixel) in rgb_img.enumerate_pixels() {
        let (r, g, b) = (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
        arr[[y as usize, x as usize, 0]] = r;
        arr[[y as usize, x as usize, 1]] = g;
        arr[[y as usize, x as usize, 2]] = b;
    }
    arr
}

/// Convert an ndarray back into an RgbImage.
/// Internal function for converting Array3<f32> to RgbImage.
fn ndarray_to_image(arr: &Array3<f32>) -> RgbImage {
    let (h, w, c) = arr.dim();
    assert_eq!(c, 3, "Expect 3 channels for RGB image");
    let mut img_buf = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let r = arr[[y, x, 0]].clamp(0.0, 255.0) as u8;
            let g = arr[[y, x, 1]].clamp(0.0, 255.0) as u8;
            let b = arr[[y, x, 2]].clamp(0.0, 255.0) as u8;
            img_buf.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    img_buf
}

/// Convert an image into a 2D boolean mask (true/false).
///
/// Non-zero luminance pixels are treated as true. The resulting mask has shape (height, width).
pub fn image_to_bool_mask(img: &DynamicImage) -> Array2<bool> {
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    let mut mask = Array2::<bool>::default((h as usize, w as usize));
    for (x, y, pixel) in gray.enumerate_pixels() {
        mask[[y as usize, x as usize]] = pixel[0] > 0;
    }
    mask
}

/// Main seam carving resize function
///
/// Resizes an image using the seam carving algorithm with the provided configuration.
///
/// # Arguments
///
/// * `image` - Input image (takes ownership)
/// * `config` - Configuration specifying target dimensions, energy mode, masks, etc.
///
/// # Returns
///
/// Result containing the resized image or an error
///
/// # Errors
///
/// Returns an error if:
/// - Image dimensions are invalid (zero size)
/// - Step ratio is not in range (0, 1]
/// - Mask dimensions don't match image dimensions
/// - Target dimensions are invalid
///
/// # Example
///
/// ```no_run
/// use sembra::{resize, ResizeConfig, EnergyMode, ResizeOrder};
/// use image;
///
/// let img = image::open("input.jpg").unwrap();
/// let config = ResizeConfig {
///     width: Some(400),
///     height: Some(300),
///     ..Default::default()
/// };
///
/// let resized = resize(img, config).unwrap();
/// resized.save("output.jpg").unwrap();
/// ```
pub fn resize(image: DynamicImage, config: ResizeConfig) -> Result<DynamicImage, SeamCarvingError> {
    // Convert to ndarray for processing
    let image_array = image_to_ndarray(&image);

    // Validate inputs
    let (h, w, c) = image_array.dim();

    if h == 0 || w == 0 || c == 0 {
        return Err(SeamCarvingError::InvalidImageDimensions {
            message: format!("Image has zero dimension: {}x{}x{}", h, w, c),
        });
    }

    if config.step_ratio <= 0.0 || config.step_ratio > 1.0 {
        return Err(SeamCarvingError::InvalidStepRatio {
            value: config.step_ratio,
        });
    }

    // Validate masks if provided
    if let Some(ref mask) = config.keep_mask {
        let (mh, mw) = mask.dim();
        if mh != h || mw != w {
            return Err(SeamCarvingError::MaskSizeMismatch {
                expected: (h, w),
                got: (mh, mw),
            });
        }
    }

    if let Some(ref mask) = config.drop_mask {
        let (mh, mw) = mask.dim();
        if mh != h || mw != w {
            return Err(SeamCarvingError::MaskSizeMismatch {
                expected: (h, w),
                got: (mh, mw),
            });
        }
    }

    // Validate target dimensions
    if let Some(target_w) = config.width {
        if target_w == 0 {
            return Err(SeamCarvingError::InvalidDimensions {
                message: "Target width cannot be zero".to_string(),
            });
        }
    }

    if let Some(target_h) = config.height {
        if target_h == 0 {
            return Err(SeamCarvingError::InvalidDimensions {
                message: "Target height cannot be zero".to_string(),
            });
        }
    }

    // Convert energy mode to string for internal functions
    let energy_mode_str = match config.energy_mode {
        EnergyMode::Backward => "backward",
        EnergyMode::Forward => "forward",
    };

    let order_str = match config.order {
        ResizeOrder::WidthFirst => "width-first",
        ResizeOrder::HeightFirst => "height-first",
    };

    let resized_array = seamcarve_resize(
        &image_array,
        config.width,
        config.height,
        energy_mode_str,
        order_str,
        config.keep_mask,
        config.drop_mask,
        config.step_ratio,
    );

    // Convert back to DynamicImage
    let result_image = ndarray_to_image(&resized_array);
    Ok(DynamicImage::ImageRgb8(result_image))
}

// ============================================================================
// Internal implementation (private functions)
// ============================================================================

/// Convert a 3D ndarray (HWC) to 2D grayscale by weighted coefficients.
fn rgb_to_gray(arr: &Array3<f32>) -> Array2<f32> {
    let (h, w, c) = arr.dim();
    if c == 1 {
        arr.index_axis(Axis(2), 0).to_owned()
    } else {
        let mut gray = Array2::<f32>::zeros((h, w));
        // Weighted sum: 0.2125R + 0.7154G + 0.0721B
        for y in 0..h {
            for x in 0..w {
                let r = arr[[y, x, 0]];
                let g = arr[[y, x, 1]];
                let b = arr[[y, x, 2]];
                let val = 0.2125 * r + 0.7154 * g + 0.0721 * b;
                gray[[y, x]] = val;
            }
        }
        gray
    }
}

/// Apply a Sobel-like filter to compute gradient magnitude as "backward" energy.
fn get_energy_backward(gray: &Array2<f32>) -> Array2<f32> {
    let (h, w) = gray.dim();
    let mut energy = Array2::<f32>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let left   = if x == 0 { gray[[y, x]] } else { gray[[y, x-1]] };
            let right  = if x == w-1 { gray[[y, x]] } else { gray[[y, x+1]] };
            let up     = if y == 0 { gray[[y, x]] } else { gray[[y-1, x]] };
            let down   = if y == h-1 { gray[[y, x]] } else { gray[[y+1, x]] };
            let grad_x = right - left;
            let grad_y = down - up;
            energy[[y, x]] = grad_x.abs() + grad_y.abs();
        }
    }

    energy
}

/// Remove one vertical seam from a 2D array according to a given seam path.
fn remove_seam_2d(arr: &Array2<f32>, seam: &[usize]) -> Array2<f32> {
    let (h, w) = arr.dim();
    let mut out = Array2::<f32>::zeros((h, w-1));
    for (r, &c) in seam.iter().enumerate().take(h) {
        out.slice_mut(s![r, 0..c]).assign(&arr.slice(s![r, 0..c]));
        out.slice_mut(s![r, c..]).assign(&arr.slice(s![r, c+1..]));
    }
    out
}

/// Remove one vertical seam from a 3D image (HWC).
#[allow(dead_code)]
fn remove_seam_3d(arr: &Array3<f32>, seam: &[usize]) -> Array3<f32> {
    let (h, w, c) = arr.dim();
    let mut out = Array3::<f32>::zeros((h, w-1, c));
    for (r, &cidx) in seam.iter().enumerate().take(h) {
        out.slice_mut(s![r, 0..cidx, ..])
            .assign(&arr.slice(s![r, 0..cidx, ..]));
        out.slice_mut(s![r, cidx.., ..])
            .assign(&arr.slice(s![r, cidx+1.., ..]));
    }
    out
}

/// Get the minimum vertical seam using backward energy with dynamic programming.
fn get_min_seam_backward(energy: &Array2<f32>) -> Vec<usize> {
    let (h, w) = energy.dim();
    let mut dp = energy.clone();
    let mut parent = Array2::<i32>::zeros((h, w));

    // Forward accumulate
    for r in 1..h {
        for c in 0..w {
            let mut min_cost = dp[[r-1, c]];
            let mut min_idx = c as i32;
            if c > 0 && dp[[r-1, c-1]] < min_cost {
                min_cost = dp[[r-1, c-1]];
                min_idx = (c - 1) as i32;
            }
            if c < w-1 && dp[[r-1, c+1]] < min_cost {
                min_cost = dp[[r-1, c+1]];
                min_idx = (c + 1) as i32;
            }
            dp[[r, c]] += min_cost;
            parent[[r, c]] = min_idx;
        }
    }

    // Find global min in bottom row
    let last_row_min = dp.slice(s![-1, ..]).indexed_iter()
        .fold((0, f32::MAX), |acc, x| {
            if *x.1 < acc.1 { (x.0, *x.1) } else { acc }
        });

    let mut seam = vec![0usize; h];
    seam[h-1] = last_row_min.0;
    // Trace upwards
    for r in (0..(h-1)).rev() {
        let c = seam[r+1];
        let pc = parent[[r+1, c]] as usize;
        seam[r] = pc;
    }
    seam
}

/// Get the minimum vertical seam using forward energy.
fn get_min_seam_forward(gray: &Array2<f32>) -> Vec<usize> {
    let (h, w) = gray.dim();
    let mut dp = Array2::<f32>::zeros((h, w));
    let mut parent = Array2::<i32>::zeros((h, w));

    for r in 1..h {
        for c in 0..w {
            let c_left = if c == 0 { c } else { c - 1 };
            let c_right = if c == w-1 { c } else { c + 1 };

            let left_val = gray[[r, c_left]];
            let right_val = gray[[r, c_right]];
            let mid_cost = (left_val - right_val).abs();

            let mut best_cost = dp[[r-1, c]];
            let mut best_parent = c as i32;

            if c > 0 {
                let cost = dp[[r-1, c-1]] + mid_cost;
                if cost < best_cost {
                    best_cost = cost;
                    best_parent = (c - 1) as i32;
                }
            }
            if c < w-1 {
                let cost = dp[[r-1, c+1]] + mid_cost;
                if cost < best_cost {
                    best_cost = cost;
                    best_parent = (c + 1) as i32;
                }
            }
            dp[[r, c]] = best_cost + mid_cost;
            parent[[r, c]] = best_parent;
        }
    }

    // Bottom row min
    let mut min_idx = 0usize;
    let mut min_val = f32::MAX;
    for c in 0..w {
        if dp[[h-1, c]] < min_val {
            min_val = dp[[h-1, c]];
            min_idx = c;
        }
    }

    // Trace up
    let mut seam = vec![0usize; h];
    seam[h-1] = min_idx;
    for r in (0..(h-1)).rev() {
        let c = seam[r+1];
        let pc = parent[[r+1, c]] as usize;
        seam[r] = pc;
    }

    seam
}

/// Find and mark N seams for removal, returning a boolean mask.
fn get_seams(
    gray: &Array2<f32>,
    num_seams: usize,
    energy_mode: &str,
    aux_energy: &mut Option<Array2<f32>>,
) -> Array2<bool> {
    let (h, w) = gray.dim();
    let mut removed = Array2::<bool>::from_elem((h, w), false);
    let mut working_gray = gray.clone();
    let mut idx_map = Array2::<usize>::from_shape_fn((h, w), |(_r, c)| c);

    if let Some(aux) = aux_energy {
        Zip::from(&mut working_gray)
            .and(aux)
            .for_each(|g, aux_val| {
                *g += *aux_val;
            });
    }

    let mut cur_w = w;
    for _ in 0..num_seams {
        let seam = match energy_mode {
            "backward" => get_min_seam_backward(&working_gray),
            "forward" => get_min_seam_forward(&working_gray),
            _ => panic!("Unsupported energy mode"),
        };

        for r in 0..h {
            let c = idx_map[[r, seam[r]]];
            removed[[r, c]] = true;
        }

        let _seam_mask = seam_to_mask(&working_gray, &seam);
        working_gray = remove_seam_2d(&working_gray, &seam);
        idx_map = remove_seam_2d_usize(&idx_map, &seam);

        if let Some(ref mut aux) = aux_energy {
            *aux = remove_seam_2d(aux, &seam);
        }

        cur_w -= 1;

        if cur_w > 1 {
            match energy_mode {
                "backward" => {
                    working_gray = get_energy_backward(&working_gray);
                    if let Some(ref aux) = aux_energy {
                        Zip::from(&mut working_gray).and(aux).for_each(|g, &x| *g += x);
                    }
                },
                "forward" => {},
                _ => {}
            }
        }
    }

    removed
}

/// Convert a seam path to a boolean mask.
fn seam_to_mask(arr: &Array2<f32>, seam: &[usize]) -> Array2<bool> {
    let (h, w) = arr.dim();
    let mut mask = Array2::<bool>::from_elem((h, w), false);
    for r in 0..h {
        let c = seam[r];
        mask[[r, c]] = true;
    }
    mask
}

/// Remove a seam from an Array2<usize>.
fn remove_seam_2d_usize(arr: &Array2<usize>, seam: &[usize]) -> Array2<usize> {
    let (h, w) = arr.dim();
    let mut out = Array2::<usize>::zeros((h, w-1));
    for (r, &c) in seam.iter().enumerate().take(h) {
        out.slice_mut(s![r, 0..c]).assign(&arr.slice(s![r, 0..c]));
        out.slice_mut(s![r, c..]).assign(&arr.slice(s![r, c+1..]));
    }
    out
}

/// Reduce width by removing seams.
fn reduce_width(
    src: &Array3<f32>,
    delta_width: usize,
    energy_mode: &str,
    aux_energy: &mut Option<Array2<f32>>,
) -> Array3<f32> {
    let (h, w, c) = src.dim();
    assert!(delta_width <= w, "Cannot reduce more than current width!");
    let gray = rgb_to_gray(src);

    let removed_mask = get_seams(&gray, delta_width, energy_mode, aux_energy);

    let new_w = w - delta_width;
    let mut out = Array3::<f32>::zeros((h, new_w, c));
    for r in 0..h {
        let mut dst_col = 0;
        for col in 0..w {
            if !removed_mask[[r, col]] {
                for channel in 0..c {
                    out[[r, dst_col, channel]] = src[[r, col, channel]];
                }
                dst_col += 1;
            }
        }
    }
    out
}

/// Transpose a 3D array (H, W, C) -> (W, H, C).
fn transpose_3d(arr: &Array3<f32>) -> Array3<f32> {
    let (h, w, c) = arr.dim();
    let mut out = Array3::<f32>::zeros((w, h, c));
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                out[[x, y, ch]] = arr[[y, x, ch]];
            }
        }
    }
    out
}

/// Insert seams into an image by duplicating them.
fn insert_seams(
    src: &Array3<f32>,
    seams: &Array2<bool>,
    delta_width: usize
) -> Array3<f32> {
    let (h, w, c) = src.dim();
    let new_w = w + delta_width;
    let mut out = Array3::<f32>::zeros((h, new_w, c));

    for row in 0..h {
        let mut dst_col = 0;
        for col in 0..w {
            if seams[[row, col]] {
                // Insert an average pixel first
                let left = if col > 0 { src.slice(s![row, col-1, ..]) }
                           else { src.slice(s![row, col, ..]) };
                let right = src.slice(s![row, col, ..]);
                for ch in 0..c {
                    let val = (left[ch] + right[ch]) * 0.5;
                    out[[row, dst_col, ch]] = val;
                }
                dst_col += 1;
            }
            // Copy the original
            for ch in 0..c {
                out[[row, dst_col, ch]] = src[[row, col, ch]];
            }
            dst_col += 1;
        }
    }

    out
}

/// Expand width by repeatedly inserting seams.
fn expand_width(
    src: &Array3<f32>,
    delta_width: usize,
    energy_mode: &str,
    aux_energy: &mut Option<Array2<f32>>,
    step_ratio: f32
) -> Array3<f32> {
    let mut out_img = src.clone();
    let mut to_expand = delta_width;
    while to_expand > 0 {
        let w = out_img.dim().1;
        let max_step = ((w as f32) * step_ratio).round().max(1.0) as usize;
        let step_size = max_step.min(to_expand);

        let gray = rgb_to_gray(&out_img);
        let removed_mask = get_seams(&gray, step_size, energy_mode, aux_energy);

        let inserted = insert_seams(&out_img, &removed_mask, step_size);
        out_img = inserted;

        if let Some(ref mut aux) = aux_energy {
            let new_aux = insert_seams_2d(aux, &removed_mask, step_size);
            *aux = new_aux;
        }
        to_expand -= step_size;
    }
    out_img
}

/// Insert seams in a 2D array.
fn insert_seams_2d(
    arr2d: &Array2<f32>,
    seams: &Array2<bool>,
    delta_width: usize
) -> Array2<f32> {
    let (h, w) = arr2d.dim();
    let new_w = w + delta_width;
    let mut out = Array2::<f32>::zeros((h, new_w));
    for row in 0..h {
        let mut dst_col = 0;
        for col in 0..w {
            if seams[[row, col]] {
                let left = if col > 0 { arr2d[[row, col-1]] }
                           else { arr2d[[row, col]] };
                let right = arr2d[[row, col]];
                out[[row, dst_col]] = 0.5*(left + right);
                dst_col += 1;
            }
            out[[row, dst_col]] = arr2d[[row, col]];
            dst_col += 1;
        }
    }
    out
}

/// Resize image width to target.
fn resize_width(
    src: &Array3<f32>,
    new_width: usize,
    energy_mode: &str,
    aux_energy: &mut Option<Array2<f32>>,
    step_ratio: f32
) -> Array3<f32> {
    let (_, w, _) = src.dim();
    if new_width == w {
        src.clone()
    } else if new_width < w {
        let delta = w - new_width;
        reduce_width(src, delta, energy_mode, aux_energy)
    } else {
        let delta = new_width - w;
        expand_width(src, delta, energy_mode, aux_energy, step_ratio)
    }
}

/// Resize image height by transposing -> resizing width -> transposing back.
fn resize_height(
    src: &Array3<f32>,
    new_height: usize,
    energy_mode: &str,
    aux_energy: &mut Option<Array2<f32>>,
    step_ratio: f32
) -> Array3<f32> {
    let t = transpose_3d(src);
    let resized = resize_width(&t, new_height, energy_mode, aux_energy, step_ratio);
    transpose_3d(&resized)
}

/// Top-level seam carving resize implementation.
#[allow(clippy::too_many_arguments)]
fn seamcarve_resize(
    src: &Array3<f32>,
    width: Option<usize>,
    height: Option<usize>,
    energy_mode: &str,
    order: &str,
    keep_mask: Option<Array2<bool>>,
    drop_mask: Option<Array2<bool>>,
    step_ratio: f32
) -> Array3<f32> {
    let (h, w, _) = src.dim();
    let mut aux_energy: Option<Array2<f32>> = None;

    if keep_mask.is_some() || drop_mask.is_some() {
        let mut aux = Array2::<f32>::zeros((h, w));
        if let Some(ref km) = keep_mask {
            Zip::from(&mut aux).and(km).for_each(|a, &m| {
                if m { *a += KEEP_MASK_ENERGY; }
            });
        }
        if let Some(ref dm) = drop_mask {
            Zip::from(&mut aux).and(dm).for_each(|a, &m| {
                if m { *a -= DROP_MASK_ENERGY; }
            });
        }
        aux_energy = Some(aux);
    }

    let mut out = src.clone();

    // Object removal with drop mask
    if let Some(ref mut aux) = aux_energy {
        fn max_negative_seam_per_row(aux: &Array2<f32>) -> usize {
            let (h, w) = aux.dim();
            let mut max = 0usize;
            for r in 0..h {
                let mut row_neg = 0usize;
                for c in 0..w {
                    if aux[[r, c]] < 0.0 {
                        row_neg += 1;
                    }
                }
                if row_neg > max {
                    max = row_neg;
                }
            }
            max
        }

        let is_object = |aux: &Array2<f32>| {
            aux.iter().any(|&v| v < 0.0)
        };

        if order == "height-first" && is_object(aux) {
            out = transpose_3d(&out);
            *aux = transpose_2d(aux);
        }

        let mut neg_count = max_negative_seam_per_row(aux);
        while neg_count > 0 {
            out = reduce_width(&out, neg_count, energy_mode, &mut Some(aux.clone()));
            let new_aux = rgb_to_gray_for_aux(&out);
            *aux = new_aux;
            neg_count = max_negative_seam_per_row(aux);
        }

        if order == "height-first" {
            out = transpose_3d(&out);
            *aux = transpose_2d(aux);
        }
    }

    // Resize to target dimensions
    if let (Some(dw), Some(dh)) = (width, height) {
        if order == "width-first" {
            out = resize_width(&out, dw, energy_mode, &mut aux_energy, step_ratio);
            out = resize_height(&out, dh, energy_mode, &mut aux_energy, step_ratio);
        } else {
            out = resize_height(&out, dh, energy_mode, &mut aux_energy, step_ratio);
            out = resize_width(&out, dw, energy_mode, &mut aux_energy, step_ratio);
        }
    }

    out
}

/// Transpose a 2D array.
fn transpose_2d(arr: &Array2<f32>) -> Array2<f32> {
    let (h, w) = arr.dim();
    let mut out = Array2::<f32>::zeros((w, h));
    for r in 0..h {
        for c in 0..w {
            out[[c, r]] = arr[[r, c]];
        }
    }
    out
}

/// Placeholder for auxiliary energy tracking after object removal.
fn rgb_to_gray_for_aux(_img: &Array3<f32>) -> Array2<f32> {
    Array2::<f32>::zeros((0, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    #[test]
    fn test_resize_same_dimensions() {
        use image::RgbImage;
        let img = DynamicImage::ImageRgb8(RgbImage::new(10, 10));
        let config = ResizeConfig {
            width: Some(10),
            height: Some(10),
            ..Default::default()
        };
        let result = resize(img, config).unwrap();
        assert_eq!(result.dimensions(), (10, 10));
    }

    #[test]
    fn test_invalid_target_dimensions() {
        use image::RgbImage;
        let img = DynamicImage::ImageRgb8(RgbImage::new(10, 10));
        let config = ResizeConfig {
            width: Some(0),
            ..Default::default()
        };
        assert!(resize(img, config).is_err());
    }

    #[test]
    fn test_invalid_step_ratio() {
        use image::RgbImage;
        let img = DynamicImage::ImageRgb8(RgbImage::new(10, 10));
        let config = ResizeConfig {
            step_ratio: 0.0,
            ..Default::default()
        };
        assert!(resize(img.clone(), config).is_err());

        let img2 = DynamicImage::ImageRgb8(RgbImage::new(10, 10));
        let config2 = ResizeConfig {
            step_ratio: 1.5,
            ..Default::default()
        };
        assert!(resize(img2, config2).is_err());
    }

    #[test]
    fn test_mask_size_mismatch() {
        use image::RgbImage;
        let img = DynamicImage::ImageRgb8(RgbImage::new(10, 10));
        let wrong_mask = Array2::<bool>::default((5, 5));
        let config = ResizeConfig {
            keep_mask: Some(wrong_mask),
            ..Default::default()
        };
        assert!(resize(img, config).is_err());
    }

    #[test]
    fn test_rgb_to_gray() {
        let mut img = Array3::<f32>::zeros((2, 2, 3));
        img[[0, 0, 0]] = 100.0; // R
        img[[0, 0, 1]] = 100.0; // G
        img[[0, 0, 2]] = 100.0; // B

        let gray = rgb_to_gray(&img);
        assert_eq!(gray.dim(), (2, 2));
        // Weighted sum: 0.2125*100 + 0.7154*100 + 0.0721*100 = 100
        assert!((gray[[0, 0]] - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_transpose_3d() {
        let img = Array3::<f32>::from_shape_fn((3, 4, 2), |(y, x, c)| {
            (y * 100 + x * 10 + c) as f32
        });
        let transposed = transpose_3d(&img);
        assert_eq!(transposed.dim(), (4, 3, 2));
        assert_eq!(transposed[[0, 0, 0]], img[[0, 0, 0]]);
        assert_eq!(transposed[[1, 2, 1]], img[[2, 1, 1]]);
    }

    #[test]
    fn test_energy_backward() {
        let gray = Array2::<f32>::from_shape_fn((3, 3), |(_y, x)| {
            if x == 1 { 255.0 } else { 0.0 }
        });
        let energy = get_energy_backward(&gray);
        // Edges between columns should have high energy
        // The leftmost and rightmost columns neighbor the bright middle column
        assert!(energy[[1, 0]] > energy[[1, 1]]);
        assert!(energy[[1, 2]] > energy[[1, 1]]);
    }
}
