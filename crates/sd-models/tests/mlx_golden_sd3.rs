//! SD 3.5's MMDiT on MLX, against `tests/golden/sd3_transformer`.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_sd3 -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::sd3::{self, Sd3Config};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/sd3_transformer")
}

fn relative(got: &Array, want: &Array, s: &Stream, what: &str) -> f32 {
    let g = got.to_vec_f32(s).expect("mlx result");
    let w = want.to_vec_f32(s).expect("reference");
    assert_eq!(g.len(), w.len(), "{what}: element count");
    let (mut worst, mut peak) = (0.0f32, 0.0f32);
    for (a, b) in g.iter().zip(&w) {
        worst = worst.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    let rel = worst / peak.max(f32::MIN_POSITIVE);
    eprintln!("{what:<16} peak {peak:>9.3}  max_abs {worst:.3e}  relative {rel:.2e}");
    rel
}

#[test]
fn the_mmdit_matches_the_reference() {
    let refs_path = dir().join("reference.safetensors");
    // The single-file checkpoint carries the transformer under
    // `model.diffusion_model.` and the VAE under `first_stage_model.`.
    let w_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/sd35/sd35-medium.safetensors");
    if !refs_path.exists() || !w_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no sd3_transformer fixture.");
        return;
    }
    let refs = load_safetensors(&refs_path).expect("reference");
    let s = Stream::gpu();
    let cfg = Sd3Config::medium_35();

    // Re-root rather than rewrite names inside the model, and cast to f32:
    // this checkpoint is f16 and the reference was produced in f32.
    let raw = load_safetensors(&w_path).expect("weights");
    let prefix = "model.diffusion_model.";
    let mut w = std::collections::HashMap::new();
    for (name, tensor) in &raw {
        if let Some(stem) = name.strip_prefix(prefix) {
            w.insert(stem.to_string(), tensor.to_f32(&s).expect("f32"));
        }
    }
    assert!(!w.is_empty(), "no transformer weights under {prefix}");

    let got = sd3::forward(
        refs.get("latents").expect("latents"),
        refs.get("context").expect("context"),
        refs.get("pooled").expect("pooled"),
        refs.get("timestep").expect("timestep"),
        &cfg,
        &w,
        &s,
    )
    .unwrap();

    let want = refs.get("output").expect("output");
    assert_eq!(got.shape(), want.shape(), "velocity has the latent's shape");
    // `golden_sd3.rs` holds this to excess < 1e-3 beyond rtol 1e-3; relative
    // to the tensor's own peak is the same order and is what the other
    // transformer tests here use.
    let rel = relative(&got, want, &s, "output");
    assert!(rel <= 1e-3, "the MMDiT is {rel:.3e} relative");
}

/// The packing and its inverse are **not** inverses of each other, and the
/// asymmetry is real: the patch embedding is a convolution running
/// `(channel, ph, pw)` while the final linear emits `(ph, pw, channel)`.
/// Round-tripping through the wrong one is a shape-correct image with every
/// 2x2 patch transposed.
#[test]
fn packing_is_channel_major_within_a_patch() {
    let s = Stream::gpu();
    // 1 batch, 2 channels, 2x2 latent -> one patch of 8 values.
    let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let packed = sd3::pack_latents(&Array::from_slice_f32(&v, &[1, 2, 2, 2]).unwrap(), &s).unwrap();
    assert_eq!(packed.shape(), vec![1, 1, 8]);
    // channel 0's 2x2 block first, then channel 1's.
    assert_eq!(
        packed.to_vec_f32(&s).unwrap(),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "packing must be channel-major within a patch"
    );
}
