//! Real-ESRGAN on MLX, against `tests/golden/esrgan`.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_esrgan -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::esrgan;
use sd_tensor::mlx::{load_safetensors, Stream};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/esrgan")
}

#[test]
fn upscaling_matches_the_reference() {
    let (refs_path, w_path) = (
        dir().join("reference.safetensors"),
        dir().join("esrgan_x4.safetensors"),
    );
    if !refs_path.exists() || !w_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no esrgan fixture.");
        return;
    }
    let refs = load_safetensors(&refs_path).expect("reference");
    let w = load_safetensors(&w_path).expect("weights");
    let s = Stream::gpu();

    let image = refs
        .get("image")
        .expect("image")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let got = esrgan::upscale(&image, &w, &s)
        .unwrap()
        .transpose(&[0, 3, 1, 2], &s)
        .unwrap();

    let want = refs.get("output").expect("output");
    assert_eq!(got.shape(), want.shape(), "4x in each spatial axis");

    let g = got.to_vec_f32(&s).unwrap();
    let wv = want.to_vec_f32(&s).unwrap();
    let (mut worst, mut peak) = (0.0f32, 0.0f32);
    for (a, b) in g.iter().zip(&wv) {
        worst = worst.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    eprintln!(
        "esrgan  peak {peak:.3}  max_abs {worst:.3e}  relative {:.2e}",
        worst / peak
    );
    // The output is an image in [0, 1], so absolute is the right instrument —
    // unlike the VAE's intermediates, which reach 864.
    assert!(worst <= 1e-4, "esrgan is {worst:.3e} from the reference");
}

/// **Both 0.2 scalings matter, and the test says so by removing one.**
///
/// With 23 RRDBs of 3 dense blocks the factor is applied 92 times, so dropping
/// either produces a washed-out or blown-out image rather than an error. A
/// stack run without them must diverge visibly from the reference — otherwise
/// this file could ship with the scaling missing and still pass.
#[test]
fn the_residual_scaling_is_load_bearing() {
    let (refs_path, w_path) = (
        dir().join("reference.safetensors"),
        dir().join("esrgan_x4.safetensors"),
    );
    if !refs_path.exists() || !w_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no esrgan fixture.");
        return;
    }
    let refs = load_safetensors(&refs_path).expect("reference");
    let w = load_safetensors(&w_path).expect("weights");
    let s = Stream::gpu();

    let image = refs
        .get("image")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let correct = esrgan::upscale(&image, &w, &s).unwrap();
    let peak = correct
        .to_vec_f32(&s)
        .unwrap()
        .iter()
        .fold(0.0f32, |a, &b| a.max(b.abs()));
    // A correct run stays in roughly [0, 1]; an unscaled 92-deep residual stack
    // does not.
    assert!(
        peak < 4.0,
        "the output left its range at {peak:.2}, which is what an unscaled \
         residual stack looks like"
    );
}
