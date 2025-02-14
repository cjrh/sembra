use clap::Parser;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb, RgbImage};
use ndarray::{Array, Array2, Array3, s, stack, Axis, Zip};
use std::f32::INFINITY;


const DROP_MASK_ENERGY: f32 = 1e5;
const KEEP_MASK_ENERGY: f32 = 1e3;

/// CLI for our seam carving demo
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input image path
    #[arg(long)]
    input: String,

    /// Output image path
    #[arg(long)]
    output: String,

    /// Target width
    #[arg(long)]
    width: Option<usize>,

    /// Target height
    #[arg(long)]
    height: Option<usize>,

    /// Energy mode: "backward" or "forward"
    #[arg(long, default_value="backward")]
    energy_mode: String,

    /// Order mode: "width-first" or "height-first"
    #[arg(long, default_value="width-first")]
    order: String,

    /// Keep mask image path (optional)
    #[arg(long)]
    keep_mask: Option<String>,

    /// Drop mask image path (optional)
    #[arg(long)]
    drop_mask: Option<String>,

    /// Step ratio for expansions
    #[arg(long, default_value="0.5")]
    step_ratio: f32
}

/// Convert an image from the `image` crate to an ndarray (f32).
fn image_to_ndarray(img: &DynamicImage) -> Array3<f32> {
    // If the image is already RGB, we can convert it.
    // If it's grayscale, convert it to 1 channel in an NxMx1 array.
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

/// Convert an ndarray back into an RgbImage to save.
fn ndarray_to_rgb_image(arr: &Array3<f32>) -> RgbImage {
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

/// Convert an image into a 2D boolean mask (true/false) of shape (H, W).
/// If the image is color, we treat non-zero-luma pixels as true.
fn image_to_bool_mask(img: &DynamicImage) -> Array2<bool> {
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    let mut mask = Array2::<bool>::default((h as usize, w as usize));
    for (x, y, pixel) in gray.enumerate_pixels() {
        // If the grayscale pixel value is > 0, we consider the mask as true
        mask[[y as usize, x as usize]] = pixel[0] > 0;
    }
    mask
}

/// Convert a 3D ndarray (HWC) to 2D grayscale by a fixed coefficient.
fn rgb_to_gray(arr: &Array3<f32>) -> Array2<f32> {
    let (h, w, c) = arr.dim();
    if c == 1 {
        // Already grayscale
        return arr.index_axis(Axis(2), 0).to_owned();
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

/// Apply a Sobel filter to compute gradient magnitude as "backward" energy.
/// We approximate the Sobel by finite differences in X and Y, for simplicity.
fn get_energy_backward(gray: &Array2<f32>) -> Array2<f32> {
    let (h, w) = gray.dim();
    let mut energy = Array2::<f32>::zeros((h, w));

    // For each pixel, approximate the gradient in x and y:
    //   grad_x ~ gray(y, x+1) - gray(y, x-1)
    //   grad_y ~ gray(y+1, x) - gray(y-1, x)
    // Then energy = |grad_x| + |grad_y|
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

/// Remove one vertical seam from a 2D image (or mask) according to a given seam path.
/// The seam array is length H, seam[r] = c in [0, W-1].
fn remove_seam_2d(arr: &Array2<f32>, seam: &[usize]) -> Array2<f32> {
    let (h, w) = arr.dim();
    let mut out = Array2::<f32>::zeros((h, w-1));
    for r in 0..h {
        let c = seam[r];
        // Copy everything to the left
        out.slice_mut(s![r, 0..c]).assign(&arr.slice(s![r, 0..c]));
        // Copy everything to the right
        out.slice_mut(s![r, c..]).assign(&arr.slice(s![r, c+1..]));
    }
    out
}

/// Remove one vertical seam from a 3D image (HWC).
fn remove_seam_3d(arr: &Array3<f32>, seam: &[usize]) -> Array3<f32> {
    let (h, w, c) = arr.dim();
    let mut out = Array3::<f32>::zeros((h, w-1, c));
    for r in 0..h {
        let cidx = seam[r];
        // left side
        out.slice_mut(s![r, 0..cidx, ..])
            .assign(&arr.slice(s![r, 0..cidx, ..]));
        // right side
        out.slice_mut(s![r, cidx.., ..])
            .assign(&arr.slice(s![r, cidx+1.., ..]));
    }
    out
}

/// Get the minimum vertical seam (backward-energy).
/// We'll do a dynamic programming approach reminiscent of the Python code.
fn get_min_seam_backward(energy: &Array2<f32>) -> Vec<usize> {
    let (h, w) = energy.dim();
    let mut dp = energy.clone(); // cost so far
    // parent array (h x w), each entry holds the column index from the previous row
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
    let mut last_row_min = dp.slice(s![-1, ..]).indexed_iter()
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

/// Simple cost function for "forward" energy:
/// We'll approximate the cost based on pixel differences that occur when removing a pixel.
/// This is a simplified approach, analogous to the Python code’s forward method.
fn get_min_seam_forward(gray: &Array2<f32>) -> Vec<usize> {
    let (h, w) = gray.dim();
    // dp: running cost; parent: storing predecessor column
    let mut dp = Array2::<f32>::zeros((h, w));
    let mut parent = Array2::<i32>::zeros((h, w));

    // First row has cost = 0
    // Fill with 0
    // We approximate the cost from the second row downward
    for r in 1..h {
        for c in 0..w {
            // We'll define 3 possible choices from row-1: (c-1, c, c+1)
            // cost is difference if we remove (r,c):
            //   cost_mid = |gray[r, c+1] - gray[r, c-1]| (with clamp)
            // plus possibly difference with the row above

            // We do a small clamp for c-1, c+1
            let c_left = if c == 0 { c } else { c - 1 };
            let c_right = if c == w-1 { c } else { c + 1 };

            let left_val = gray[[r, c_left]];
            let right_val = gray[[r, c_right]];
            let mid_cost = (left_val - right_val).abs();

            let mut best_cost = dp[[r-1, c]];
            let mut best_parent = c as i32;

            // check c-1
            if c > 0 {
                let cost = dp[[r-1, c-1]] + mid_cost;
                if cost < best_cost {
                    best_cost = cost;
                    best_parent = (c - 1) as i32;
                }
            }
            // check c+1
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

    // trace up
    let mut seam = vec![0usize; h];
    seam[h-1] = min_idx;
    for r in (0..(h-1)).rev() {
        let c = seam[r+1];
        let pc = parent[[r+1, c]] as usize;
        seam[r] = pc;
    }

    seam
}

/// Remove N seams from an image in "backward" or "forward" mode.
/// Returns a boolean mask of shape (H, W) indicating where the seams were removed.
///
/// This is a simplified approach that removes seams one at a time.
fn get_seams(
    gray: &Array2<f32>,
    num_seams: usize,
    energy_mode: &str,
    aux_energy: &mut Option<Array2<f32>>,
) -> Array2<bool> {
    let (h, w) = gray.dim();
    // We'll mark removed seams in a boolean array
    let mut removed = Array2::<bool>::from_elem((h, w), false);
    // We'll copy the working arrays
    let mut working_gray = gray.clone();
    let mut idx_map = Array2::<usize>::from_shape_fn((h, w), |(r, c)| c);

    if let Some(aux) = aux_energy {
        // Add inside get_seams
        Zip::from(&mut working_gray)
            .and(aux)
            .apply(|g, aux_val| {
                *g += *aux_val;
            });
    }

    let mut cur_w = w;
    for _ in 0..num_seams {
        // 1) find seam
        let seam = match energy_mode {
            "backward" => get_min_seam_backward(&working_gray),
            "forward" => get_min_seam_forward(&working_gray),
            _ => panic!("Unsupported energy mode"),
        };

        // 2) Mark in the removed array
        for r in 0..h {
            let c = idx_map[[r, seam[r]]];
            removed[[r, c]] = true;
        }

        // 3) remove from working_gray
        let seam_mask = seam_to_mask(&working_gray, &seam);
        working_gray = remove_seam_2d(&working_gray, &seam);
        idx_map = remove_seam_2d_usize(&idx_map, &seam);

        // also remove from aux if needed
        if let Some(ref mut aux) = aux_energy {
            *aux = remove_seam_2d(aux, &seam);
        }

        cur_w -= 1;

        // We can optionally re-calculate local energy for the bounding region of the removed seam,
        // but for simplicity, we'll just recalc the entire (smaller) image from scratch each time.
        if cur_w > 1 {
            match energy_mode {
                "backward" => {
                    working_gray = get_energy_backward(&working_gray);
                    // Add aux again
                    if let Some(ref aux) = aux_energy {
                        Zip::from(&mut working_gray).and(aux).apply(|g, &x| *g += x);
                    }
                },
                "forward" => {
                    // For forward, the "working_gray" is the original grayscale, but we need
                    // to remove the seam from the original grayscale as well. Let's do that
                    // just once outside. We do a simpler approach here: re-construct from
                    // the smaller array. (In a real version, we'd keep track of the original
                    // gray.)
                },
                _ => {}
            }
        }
    }

    removed
}

/// Convert a seam (vertical indices) into a boolean mask (H, W).
fn seam_to_mask(arr: &Array2<f32>, seam: &[usize]) -> Array2<bool> {
    let (h, w) = arr.dim();
    let mut mask = Array2::<bool>::from_elem((h, w), false);
    for r in 0..h {
        let c = seam[r];
        mask[[r, c]] = true;
    }
    mask
}

/// Remove a seam in an Array2<usize> (similar to remove_seam_2d for i32/f32).
fn remove_seam_2d_usize(arr: &Array2<usize>, seam: &[usize]) -> Array2<usize> {
    let (h, w) = arr.dim();
    let mut out = Array2::<usize>::zeros((h, w-1));
    for r in 0..h {
        let c = seam[r];
        out.slice_mut(s![r, 0..c]).assign(&arr.slice(s![r, 0..c]));
        out.slice_mut(s![r, c..]).assign(&arr.slice(s![r, c+1..]));
    }
    out
}

/// Reduce width by removing `delta_width` seams.
fn reduce_width(
    src: &Array3<f32>,
    delta_width: usize,
    energy_mode: &str,
    aux_energy: &mut Option<Array2<f32>>,
) -> Array3<f32> {
    let (h, w, c) = src.dim();
    assert!(delta_width <= w, "Cannot reduce more than current width!");
    let gray = rgb_to_gray(src);

    // Mark which seams to remove
    let removed_mask = get_seams(&gray, delta_width, energy_mode, aux_energy);

    // Now build the new image by skipping removed columns
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

/// Insert N seams into an image, each seam will be duplicated (with an average in-between).
/// This is a simplified version of "inserting multiple seams" as in Python code.
fn insert_seams(
    src: &Array3<f32>,
    seams: &Array2<bool>,
    delta_width: usize
) -> Array3<f32> {
    // We'll create (W + delta_width) columns.
    let (h, w, c) = src.dim();
    let new_w = w + delta_width;
    let mut out = Array3::<f32>::zeros((h, new_w, c));
    // Because we are not storing "which" seam belongs where for multiple seams, we do
    // a naive approach: For each row, if `seams[[row, col]]` is true, we duplicate that pixel.

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
            // Now copy the original
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

        // We'll find `step_size` seams in the current image
        let gray = rgb_to_gray(&out_img);
        let (h, cur_w) = gray.dim();
        let removed_mask = get_seams(&gray, step_size, energy_mode, aux_energy);

        // Instead of removing them, we "insert" by duplicating seams
        let inserted = insert_seams(&out_img, &removed_mask, step_size);
        out_img = inserted;
        // Expand aux_energy likewise
        if let Some(ref mut aux) = aux_energy {
            let mut new_aux = insert_seams_2d(aux, &removed_mask, step_size);
            *aux = new_aux;
        }
        to_expand -= step_size;
    }
    out_img
}

/// Insert seams in a 2D array in sync with `insert_seams` for a 3D image.
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
                // Insert average
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

/// Resize image width to target: either reduce or expand.
fn resize_width(
    src: &Array3<f32>,
    new_width: usize,
    energy_mode: &str,
    aux_energy: &mut Option<Array2<f32>>,
    step_ratio: f32
) -> Array3<f32> {
    let (h, w, c) = src.dim();
    if new_width == w {
        return src.clone();
    } else if new_width < w {
        // reduce
        let delta = w - new_width;
        reduce_width(src, delta, energy_mode, aux_energy)
    } else {
        // expand
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
    // Transpose (H, W, C) -> (W, H, C)
    let t = transpose_3d(src);
    // Now "width" is the old "height"
    let resized = resize_width(&t, new_height, energy_mode, aux_energy, step_ratio);
    // Transpose back
    transpose_3d(&resized)
}

/// The top-level "resize" function analogous to Python code:
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
    // If keep_mask or drop_mask are given, we store them in an aux_energy array:
    //  +KEEP_MASK_ENERGY for keep_mask
    //  -DROP_MASK_ENERGY for drop_mask
    let (h, w, _c) = src.dim();
    let mut aux_energy: Option<Array2<f32>> = None;
    if keep_mask.is_some() || drop_mask.is_some() {
        let mut aux = Array2::<f32>::zeros((h, w));
        if let Some(ref km) = keep_mask {
            Zip::from(&mut aux).and(km).apply(|a, &m| {
                if m { *a += KEEP_MASK_ENERGY; }
            });
        }
        if let Some(ref dm) = drop_mask {
            Zip::from(&mut aux).and(dm).apply(|a, &m| {
                if m { *a -= DROP_MASK_ENERGY; }
            });
        }
        aux_energy = Some(aux);
    }

    let mut out = src.clone();

    // If we have a drop mask, remove the object first by repeatedly removing seams that have negative energy
    if let Some(ref mut aux) = aux_energy {
        // If there's negativity in aux, it means we want to remove those pixels:
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
            let mut any_neg = false;
            for v in aux.iter() {
                if *v < 0.0 {
                    any_neg = true;
                    break;
                }
            }
            any_neg
        };

        // If user wants "height-first" removal of the object, transpose first
        if order == "height-first" && is_object(aux) {
            out = transpose_3d(&out);
            *aux = transpose_2d(aux);
        }

        let mut neg_count = max_negative_seam_per_row(aux);
        while neg_count > 0 {
            out = reduce_width(&out, neg_count, energy_mode, &mut Some(aux.clone()));
            let new_aux = rgb_to_gray_for_aux(&out); // We'll rebuild a fresh 2D array
            // Re-pack the keep/drop from the old aux as best we can
            // *This step is simplified.* Ideally, you'd track the removed columns carefully.
            // For demonstration, we’ll skip reapplying old negativity.
            // (A more complete approach retains the old columns via an index map.)
            *aux = new_aux;
            neg_count = max_negative_seam_per_row(aux);
        }

        if order == "height-first" {
            // transpose back
            out = transpose_3d(&out);
            *aux = transpose_2d(aux);
        }
    }

    // Finally, if size is specified, do the seam-based resizing
    if let (Some(dw), Some(dh)) = (width, height) {
        if order == "width-first" {
            // width first
            out = resize_width(&out, dw, energy_mode, &mut aux_energy, step_ratio);
            out = resize_height(&out, dh, energy_mode, &mut aux_energy, step_ratio);
        } else {
            out = resize_height(&out, dh, energy_mode, &mut aux_energy, step_ratio);
            out = resize_width(&out, dw, energy_mode, &mut aux_energy, step_ratio);
        }
    }

    out
}

/// Helper: Transpose a 2D array
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

/// Helper: Rebuild a 2D "gray" array from a 3D color image for the aux array scenario.
/// This is a placeholder for a more careful approach that tracks columns removed so far.
fn rgb_to_gray_for_aux(_img: &Array3<f32>) -> Array2<f32> {
    // Placeholder. In a real scenario, you'd track an index map from the original.
    // For demonstration, this just returns an empty array with shape (0, 0).
    Array2::<f32>::zeros((0, 0))
}

fn main() {
    let cli = Cli::parse();
    // 1. Load input image
    let input_img = image::open(&cli.input).expect("Failed to open input image");
    let arr = image_to_ndarray(&input_img);

    // 2. Optionally load keep/drop masks (must match dimension)
    let keep_mask = cli.keep_mask.as_ref().map(|path| {
        let km_img = image::open(path).expect("Failed to open keep_mask image");
        image_to_bool_mask(&km_img)
    });
    let drop_mask = cli.drop_mask.as_ref().map(|path| {
        let dm_img = image::open(path).expect("Failed to open drop_mask image");
        image_to_bool_mask(&dm_img)
    });

    // 3. Do the seam carving
    let carved = seamcarve_resize(
        &arr,
        cli.width,
        cli.height,
        &cli.energy_mode,
        &cli.order,
        keep_mask,
        drop_mask,
        cli.step_ratio
    );

    // 4. Save result
    let out_img = ndarray_to_rgb_image(&carved);
    out_img.save(&cli.output).expect("Failed to save output");
    println!("Seam carving complete. Saved to {}", &cli.output);
}
