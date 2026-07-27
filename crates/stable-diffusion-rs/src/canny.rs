//! Canny edge detection, for producing a ControlNet hint from a photograph.
//!
//! A ControlNet does not take an image — it takes a *control map*, and for the
//! canny models that map is a one-pixel-wide white edge skeleton on black. This
//! module makes one, so `sdrs controlnet --init-image photo.png` works without
//! a second tool.
//!
//! Canny is four stages, and the last two are what separate it from "run a
//! Sobel filter and threshold":
//!
//! 1. blur, so single-pixel noise does not become an edge
//! 2. Sobel gradients, giving magnitude and direction
//! 3. **non-maximum suppression** — keep a pixel only if it is the local peak
//!    *along the gradient*, which is what makes edges one pixel wide instead of
//!    thick bands
//! 4. **hysteresis** — keep strong pixels, and weak ones only where they
//!    connect to a strong one, which is what stops a single threshold from
//!    either shattering long edges or admitting noise
//!
//! Both thresholds are in the same units as the gradient magnitude, which this
//! module normalises to `[0, 1]`. The defaults (0.1 / 0.2) match the ratio the
//! ControlNet demos use.

/// A greyscale image in `[0, 1]`, row-major.
pub struct Gray<'a> {
    pub data: &'a [f32],
    pub width: usize,
    pub height: usize,
}

/// 5x5 Gaussian, sigma ~1.4 — the kernel Canny's paper uses, and what OpenCV
/// applies by default before its Sobel pass.
const GAUSSIAN: [f32; 25] = [
    2.0, 4.0, 5.0, 4.0, 2.0, //
    4.0, 9.0, 12.0, 9.0, 4.0, //
    5.0, 12.0, 15.0, 12.0, 5.0, //
    4.0, 9.0, 12.0, 9.0, 4.0, //
    2.0, 4.0, 5.0, 4.0, 2.0,
];
const GAUSSIAN_SUM: f32 = 159.0;

fn at(data: &[f32], w: usize, h: usize, x: isize, y: isize) -> f32 {
    // Clamp at the border rather than treating outside as black: a black
    // border is itself a strong edge, and every image would come back framed.
    let x = x.clamp(0, w as isize - 1) as usize;
    let y = y.clamp(0, h as isize - 1) as usize;
    data[y * w + x]
}

fn blur(img: &Gray<'_>) -> Vec<f32> {
    let (w, h) = (img.width, img.height);
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for ky in 0..5usize {
                for kx in 0..5usize {
                    let v = at(
                        img.data,
                        w,
                        h,
                        x as isize + kx as isize - 2,
                        y as isize + ky as isize - 2,
                    );
                    acc += v * GAUSSIAN[ky * 5 + kx];
                }
            }
            out[y * w + x] = acc / GAUSSIAN_SUM;
        }
    }
    out
}

/// Detect edges. Returns a mask in `[0, 1]`, 1 on an edge.
///
/// `low` and `high` are the hysteresis thresholds on normalised gradient
/// magnitude; `low` must not exceed `high`, and both are clamped into `[0, 1]`.
pub fn canny(img: &Gray<'_>, low: f32, high: f32) -> Vec<f32> {
    let (w, h) = (img.width, img.height);
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let high = high.clamp(0.0, 1.0);
    let low = low.clamp(0.0, high);

    let smooth = blur(img);

    // -- Sobel ------------------------------------------------------------
    let mut mag = vec![0f32; w * h];
    let mut dir = vec![0u8; w * h];
    let mut peak = 0f32;
    for y in 0..h {
        for x in 0..w {
            let (xi, yi) = (x as isize, y as isize);
            let p = |dx: isize, dy: isize| at(&smooth, w, h, xi + dx, yi + dy);
            let gx = (p(1, -1) + 2.0 * p(1, 0) + p(1, 1)) - (p(-1, -1) + 2.0 * p(-1, 0) + p(-1, 1));
            let gy = (p(-1, 1) + 2.0 * p(0, 1) + p(1, 1)) - (p(-1, -1) + 2.0 * p(0, -1) + p(1, -1));
            let m = (gx * gx + gy * gy).sqrt();
            mag[y * w + x] = m;
            peak = peak.max(m);
            // Quantise the direction to one of four neighbour pairs. The
            // comparison in the next stage is between *neighbouring pixels*,
            // so a continuous angle has nowhere finer to point.
            let angle = gy.atan2(gx).to_degrees().rem_euclid(180.0);
            dir[y * w + x] = match angle {
                a if !(22.5..157.5).contains(&a) => 0, // horizontal: compare left/right
                a if a < 67.5 => 1,                    // diagonal /
                a if a < 112.5 => 2,                   // vertical: compare up/down
                _ => 3,                                // diagonal \
            };
        }
    }
    // Normalise so the thresholds mean the same thing on any image. A flat
    // image has no gradient at all and no edges, so return early rather than
    // dividing by zero.
    if peak <= f32::EPSILON {
        return vec![0f32; w * h];
    }
    for m in &mut mag {
        *m /= peak;
    }

    // -- non-maximum suppression ------------------------------------------
    let mut thin = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (dx, dy) = match dir[i] {
                0 => (1isize, 0isize),
                1 => (1, -1),
                2 => (0, 1),
                _ => (1, 1),
            };
            let m = mag[i];
            let a = at(&mag, w, h, x as isize + dx, y as isize + dy);
            let b = at(&mag, w, h, x as isize - dx, y as isize - dy);
            // `>=` on one side and `>` on the other: a plateau of equal
            // magnitudes would otherwise suppress every pixel in it and erase
            // the edge entirely.
            if m >= a && m > b {
                thin[i] = m;
            }
        }
    }

    // -- hysteresis --------------------------------------------------------
    //
    // Flood from the strong pixels rather than sweeping the image repeatedly:
    // a weak pixel survives exactly when a path of weak pixels connects it to
    // a strong one, which is reachability, and a single pass in raster order
    // would miss any chain that runs backwards.
    let mut out = vec![0f32; w * h];
    let mut stack: Vec<usize> = Vec::new();
    for (i, &m) in thin.iter().enumerate() {
        if m >= high {
            out[i] = 1.0;
            stack.push(i);
        }
    }
    while let Some(i) = stack.pop() {
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if out[j] == 0.0 && thin[j] >= low {
                    out[j] = 1.0;
                    stack.push(j);
                }
            }
        }
    }
    out
}

/// Read an image and return a ControlNet hint: `[1, 3, height, width]` in
/// `[0, 1]`, white edges on black.
///
/// The edges are detected *at the target resolution* rather than detected and
/// then resized — resampling a one-pixel skeleton is what turns crisp edges
/// into grey smears, and the ControlNet reads a smear as a different shape.
pub fn hint_from_image<P: AsRef<std::path::Path>>(
    path: P,
    width: u32,
    height: u32,
    low: f32,
    high: f32,
    device: &sd_tensor::Device,
) -> sd_tensor::Result<sd_tensor::Tensor> {
    let img = image::open(path.as_ref())
        .map_err(|e| sd_tensor::Error::Msg(format!("failed to read image: {e}")))?;
    let img = img
        .resize_exact(width, height, image::imageops::FilterType::Lanczos3)
        .to_luma8();
    let data: Vec<f32> = img.as_raw().iter().map(|&b| b as f32 / 255.0).collect();

    let edges = canny(
        &Gray {
            data: &data,
            width: width as usize,
            height: height as usize,
        },
        low,
        high,
    );
    // Replicated across three channels: the canny ControlNets take an RGB
    // control map, and the edge map is greyscale by construction.
    let mut rgb = Vec::with_capacity(edges.len() * 3);
    for _ in 0..3 {
        rgb.extend_from_slice(&edges);
    }
    sd_tensor::Tensor::from_vec(rgb, (1, 3, height as usize, width as usize), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A white square on black, 64x64 with the square from 16..48.
    fn square() -> Vec<f32> {
        let mut v = vec![0f32; 64 * 64];
        for y in 16..48 {
            for x in 16..48 {
                v[y * 64 + x] = 1.0;
            }
        }
        v
    }

    #[test]
    fn a_square_gives_edges_on_its_border_and_nowhere_else() {
        let data = square();
        let edges = canny(
            &Gray {
                data: &data,
                width: 64,
                height: 64,
            },
            0.1,
            0.2,
        );
        // The interior and the far background must be empty: a filled region
        // has no gradient, and an edge detector that marks it is thresholding
        // brightness rather than finding edges.
        assert_eq!(edges[32 * 64 + 32], 0.0, "centre of the square");
        assert_eq!(edges[2 * 64 + 2], 0.0, "far background");
        // And the border must be found. The blur spreads the step over a few
        // pixels, so look in a small band around the true edge.
        let near_edge = (14..19).any(|y| edges[y * 64 + 32] > 0.0);
        assert!(near_edge, "no edge found on the square's top border");
    }

    #[test]
    fn edges_are_thin() {
        // The point of non-maximum suppression. Without it the 5x5 blur turns
        // each border into a band several pixels wide, and the ControlNet
        // reads a thick smear as a different shape.
        let data = square();
        let edges = canny(
            &Gray {
                data: &data,
                width: 64,
                height: 64,
            },
            0.1,
            0.2,
        );
        let column: usize = (0..32).filter(|y| edges[y * 64 + 32] > 0.0).count();
        assert!(
            (1..=2).contains(&column),
            "top border is {column} pixels thick, want 1-2"
        );
    }

    #[test]
    fn a_flat_image_has_no_edges() {
        let data = vec![0.5f32; 32 * 32];
        let edges = canny(
            &Gray {
                data: &data,
                width: 32,
                height: 32,
            },
            0.1,
            0.2,
        );
        assert!(edges.iter().all(|&e| e == 0.0));
    }

    #[test]
    fn the_border_is_not_itself_an_edge() {
        // Sampling outside the image as black would frame every result. The
        // clamp in `at` is what prevents it, and this is the test that would
        // catch its removal.
        let data = vec![1.0f32; 32 * 32];
        let edges = canny(
            &Gray {
                data: &data,
                width: 32,
                height: 32,
            },
            0.1,
            0.2,
        );
        assert!(edges.iter().all(|&e| e == 0.0), "white image framed itself");
    }
}
