//! Tiled VAE decoding on MLX.
//!
//! A decode allocates a convolution im2col of `cin * 9` values per position, so
//! the peak scales with the area being decoded — and the failure lands at the
//! *end* of a run, after every denoise step has been paid for. Tiling turns
//! that into a slightly different image instead of a dead run.
//!
//! The failure tiling itself introduces is a hard line at a seam, which no
//! comparison against a whole-image decode catches well: the mean difference
//! stays small while one column is visibly wrong. It gets its own test.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_vae_tiled -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::vae::{self, VaeConfig, TILE_LATENT_EDGE};
use sd_tensor::mlx::{load_safetensors, Array, Stream};
use sd_tensor::rng::SeededRng;
use sd_tensor::{Device, Tensor};

fn weights() -> Option<sd_models::mlx::Weights> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/vae_decoder/vae.safetensors");
    p.exists().then(|| load_safetensors(&p).expect("vae"))
}

/// A seeded latent, NHWC.
fn latent(seed: u64, edge: usize) -> Array {
    let t: Tensor = SeededRng::new(seed)
        .randn((1, 4, edge, edge), &Device::Cpu)
        .unwrap();
    let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // NCHW -> NHWC.
    let mut out = vec![0.0f32; v.len()];
    for c in 0..4 {
        for y in 0..edge {
            for x in 0..edge {
                out[(y * edge + x) * 4 + c] = v[c * edge * edge + y * edge + x];
            }
        }
    }
    Array::from_slice_f32(&out, &[1, edge, edge, 4]).unwrap()
}

/// Below the tile edge, tiling must be exactly the untiled decode — not merely
/// close to it, since it is meant to be the same call.
#[test]
fn a_latent_that_already_fits_is_not_tiled() {
    let Some(w) = weights() else {
        sd_tensor::skip_missing_fixture!("SKIP: no vae fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = VaeConfig::sd15();
    let z = latent(1, 32);

    let whole = vae::decode_with(&z, &cfg, &w, &s).expect("whole");
    let tiled = vae::decode_tiled(&z, &cfg, TILE_LATENT_EDGE, &w, &s).expect("tiled");
    assert_eq!(
        whole.to_vec_f32(&s).unwrap(),
        tiled.to_vec_f32(&s).unwrap(),
        "a latent inside one tile must take the untiled path exactly"
    );
}

/// Tiled and whole-image decodes must agree closely, but not exactly: the
/// edge-padding difference is real and tiling is an approximation by
/// construction.
#[test]
fn tiling_agrees_closely_with_a_whole_image_decode() {
    let Some(w) = weights() else {
        sd_tensor::skip_missing_fixture!("SKIP: no vae fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = VaeConfig::sd15();
    // Comfortably over one tile at the edge used here, so tiling engages.
    let tile = 32;
    let z = latent(1, 48);

    let whole = vae::decode_with(&z, &cfg, &w, &s).expect("whole");
    let tiled = vae::decode_tiled(&z, &cfg, tile, &w, &s).expect("tiled");
    assert_eq!(whole.shape(), tiled.shape(), "tiling must not change shape");

    let (a, b) = (whole.to_vec_f32(&s).unwrap(), tiled.to_vec_f32(&s).unwrap());
    let mean_abs = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs() as f64)
        .sum::<f64>()
        / a.len() as f64;
    let max_abs = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("tiled vs whole: mean_abs {mean_abs:.5}, max_abs {max_abs:.5}");

    // `golden_vae_tiled.rs`'s bound and its reasoning: not 1e-4, because on a
    // [-1, 1] image a mean of a few thousandths is imperceptible and the max is
    // dominated by the few pixels nearest a seam.
    assert!(
        mean_abs < 0.02,
        "the tiled decode drifts too far from the whole-image one: mean_abs {mean_abs:.5}"
    );
}

/// **The seams must not be visible.**
///
/// This is the failure tiling introduces, and the agreement test above does not
/// catch it: a hard line at one column leaves the mean difference small. So the
/// gradient across each column is compared against the gradient elsewhere — if
/// the blending works, no column stands out.
#[test]
fn tile_seams_are_not_visible_as_discontinuities() {
    let Some(w) = weights() else {
        sd_tensor::skip_missing_fixture!("SKIP: no vae fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = VaeConfig::sd15();
    let z = latent(2, 64);
    let img = vae::decode_tiled(&z, &cfg, 32, &w, &s).expect("tiled");

    let [_, h, wd, c] = img.shape()[..] else {
        panic!("NHWC")
    };
    let v = img.to_vec_f32(&s).unwrap();
    let at = |ch: usize, y: usize, x: usize| v[(y * wd + x) * c + ch];

    // Horizontal step between adjacent columns, averaged over rows and
    // channels.
    let col_step = |x: usize| -> f32 {
        let mut acc = 0.0;
        for ch in 0..c {
            for y in 0..h {
                acc += (at(ch, y, x) - at(ch, y, x - 1)).abs();
            }
        }
        acc / (c * h) as f32
    };
    let steps: Vec<f32> = (1..wd).map(col_step).collect();
    let mean = steps.iter().sum::<f32>() / steps.len() as f32;
    let worst = steps.iter().cloned().fold(0f32, f32::max);
    eprintln!(
        "column step: mean {mean:.5}, worst {worst:.5}, ratio {:.2}",
        worst / mean
    );

    // A hard seam shows up as one column an order of magnitude above the rest.
    assert!(
        worst < mean * 6.0,
        "a column step of {worst:.4} against a mean of {mean:.4} looks like a visible seam"
    );
}

/// **On a flat field, tiling must not introduce a step the whole decode lacks.**
///
/// A uniform latent has no true structure, so every step in the output is the
/// decoder's or the tiling's — and the whole-image decode gives the former on
/// its own. Comparing the two isolates what tiling added.
///
/// This deliberately does *not* assert that the two images are close. They are
/// not: each tile's border is computed against zero padding rather than its
/// true neighbours, and through three upsamples that reaches far enough inward
/// to measure 0.228 on a `[-1, 1]` image. That difference is smooth, which is
/// why it is invisible, and asserting on its magnitude would be asserting the
/// wrong property.
#[test]
fn tiling_a_flat_field_adds_no_step_the_whole_decode_lacks() {
    let Some(w) = weights() else {
        sd_tensor::skip_missing_fixture!("SKIP: no vae fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = VaeConfig::sd15();
    let edge = 64;
    let z = Array::from_slice_f32(&vec![0.35f32; edge * edge * 4], &[1, edge, edge, 4]).unwrap();

    let worst_step = |img: &Array| -> f32 {
        let [_, h, wd, c] = img.shape()[..] else {
            panic!("NHWC")
        };
        let v = img.to_vec_f32(&s).unwrap();
        let at = |ch: usize, y: usize, x: usize| v[(y * wd + x) * c + ch];
        (1..wd)
            .map(|x| {
                let mut acc = 0.0;
                for ch in 0..c {
                    for y in 0..h {
                        acc += (at(ch, y, x) - at(ch, y, x - 1)).abs();
                    }
                }
                acc / (c * h) as f32
            })
            .fold(0.0f32, f32::max)
    };

    let whole = worst_step(&vae::decode_with(&z, &cfg, &w, &s).expect("whole"));
    let tiled = worst_step(&vae::decode_tiled(&z, &cfg, 32, &w, &s).expect("tiled"));
    eprintln!("flat field, worst column step: whole {whole:.5}, tiled {tiled:.5}");
    assert!(
        tiled < whole * 3.0 + 1e-3,
        "tiling raised the worst column step from {whole:.5} to {tiled:.5}; the blend is \
         leaving a discontinuity a flat field cannot explain"
    );
}

/// A tile edge of zero is a caller error, not a mode.
#[test]
fn a_zero_tile_edge_is_refused() {
    let Some(w) = weights() else {
        sd_tensor::skip_missing_fixture!("SKIP: no vae fixture.");
        return;
    };
    let s = Stream::gpu();
    assert!(vae::decode_tiled(&latent(3, 16), &VaeConfig::sd15(), 0, &w, &s).is_err());
}
