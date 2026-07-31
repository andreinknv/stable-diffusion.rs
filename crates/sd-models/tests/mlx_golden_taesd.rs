//! TAESD on MLX, against `tests/golden/taesd`.
//!
//! `golden_taesd.rs` compares against `AutoencoderTiny` end to end so the
//! latent convention is covered rather than assumed — TAESD's `scaling_factor`
//! is 1.0, and applying the SD VAE's `/ 0.18215` here multiplies the input by
//! 5.5 and yields a washed-out image with no error anywhere.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_taesd -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::taesd;
use sd_tensor::mlx::{load_safetensors, Array, Stream};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/taesd")
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let refs = golden_dir().join("reference.safetensors");
    let weights = golden_dir().join("taesd.safetensors");
    if !refs.exists() || !weights.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no taesd fixture.");
        return None;
    }
    Some((
        load_safetensors(&refs).expect("reference"),
        load_safetensors(&weights).expect("weights"),
    ))
}

/// Relative to the tensor's own peak, the instrument the VAE tests settled on.
fn relative(got_nhwc: &Array, want_nchw: &Array, s: &Stream, what: &str) -> f32 {
    let got = got_nhwc
        .transpose(&[0, 3, 1, 2], s)
        .expect("NHWC -> NCHW")
        .to_vec_f32(s)
        .expect("mlx result");
    let want = want_nchw.to_vec_f32(s).expect("reference");
    assert_eq!(got.len(), want.len(), "{what}: element count");
    let (mut worst, mut peak) = (0.0f32, 0.0f32);
    for (g, w) in got.iter().zip(&want) {
        worst = worst.max((g - w).abs());
        peak = peak.max(w.abs());
    }
    let rel = worst / peak.max(f32::MIN_POSITIVE);
    eprintln!("{what:<16} peak {peak:>8.3}  max_abs {worst:.3e}  relative {rel:.2e}");
    rel
}

/// 1e-4 relative, the same bound the VAE decoder's stages use and four orders
/// under a real porting bug.
const REL: f32 = 1e-4;

#[test]
fn decode_matches_autoencoder_tiny() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();

    // `decoder_raw` is the decoder's own output, before any latent scaling the
    // caller might apply — which for TAESD is none.
    let latent = refs
        .get("latent")
        .expect("latent")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let got = taesd::decode(&latent, &w, &s).unwrap();
    assert_eq!(got.shape(), vec![1, 256, 256, 3]);
    let rel = relative(
        &got,
        refs.get("decoder_raw").expect("decoder_raw"),
        &s,
        "decoded",
    );
    assert!(rel <= REL, "TAESD decode is {rel:.3e} relative");
}

#[test]
fn encode_matches_autoencoder_tiny() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();

    let image = refs
        .get("image")
        .expect("image")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let got = taesd::encode(&image, &w, &s).unwrap();
    assert_eq!(got.shape(), vec![1, 32, 32, 4]);
    let rel = relative(
        &got,
        refs.get("encoder_raw").expect("encoder_raw"),
        &s,
        "encoded",
    );
    assert!(rel <= REL, "TAESD encode is {rel:.3e} relative");
}

/// The soft clamp is not decoration: `tanh(x/3)*3` is not the identity
/// anywhere, so removing it changes ordinary output too. A latent well outside
/// [-3, 3] must come back bounded rather than as a bright artefact.
#[test]
fn the_decoder_soft_clamps_its_input() {
    let Some((_, w)) = fixtures() else { return };
    let s = Stream::gpu();

    let wild = Array::from_slice_f32(&vec![50.0; 4 * 8 * 8], &[1, 8, 8, 4]).unwrap();
    let out = taesd::decode(&wild, &w, &s).unwrap();
    let pixels = out.to_vec_f32(&s).unwrap();
    let peak = pixels.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(
        peak.is_finite() && peak < 50.0,
        "a latent at 50 should be squashed, not amplified: peak {peak}"
    );
}
