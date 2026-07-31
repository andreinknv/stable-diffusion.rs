//! Flux's VAE on MLX, against `tests/golden/flux_vae`.
//!
//! The same convolutional geometry as SD's, so this is not a new port — it is
//! a check that the MLX one is genuinely parameterised rather than accidentally
//! specialised to SD. Three things differ and each has a silent failure mode:
//!
//! - **16 latent channels** instead of 4. A hardcoded 4 gives a shape error,
//!   which is the harmless case.
//! - **No `quant_conv` / `post_quant_conv`.** Building them anyway looks for
//!   weights that do not exist; not building them when they do exist silently
//!   drops a 1x1 convolution.
//! - **A latent shift** as well as a scale: `(x - shift) * scale`. Applying
//!   these in the wrong order leaves a recognisable image with wrong contrast —
//!   the failure that survives eyeballing.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_flux_vae -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::vae::{self, VaeConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

/// The decoder's bound — `mlx_golden_vae`'s, and `golden_flux_vae.rs`'s.
const ATOL: f32 = 1e-4;

/// The **encoder's** bound, and it is 20x looser for a measured reason rather
/// than a convenient one.
///
/// `golden_flux_vae.rs` records the measurement: running diffusers' own Flux
/// encoder in f32 and f64 and comparing gives max_abs **9.605e-4** — the
/// reference's own f32 noise floor. candle's deviation from diffusers f32 is
/// 9.606e-4, i.e. exactly at the floor; this port's is 1.515e-3, which is 1.6x
/// the floor and is what a different reduction order costs at that floor. The
/// same measurement on SD's encoder gives 1.226e-4, which is why *that* one
/// holds to 1e-4: the Flux VAE is genuinely ~8x worse conditioned, and its
/// config says so by setting `force_upcast`.
///
/// This is not a licence to be sloppy. A structural fault here is orders of
/// magnitude larger, not marginally: the symmetric-padding bug in this same
/// encoder measured 17.32.
const ENCODER_NOISE_FLOOR: f32 = 2e-3;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/flux_vae")
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let (refs, w) = (
        golden().join("reference.safetensors"),
        golden().join("vae.safetensors"),
    );
    if !refs.exists() || !w.exists() {
        return None;
    }
    Some((
        load_safetensors(&refs).expect("reference"),
        load_safetensors(&w).expect("weights"),
    ))
}

/// `got` is NHWC, `want` NCHW.
fn compare(got: &Array, want: &Array, s: &Stream, what: &str) -> f32 {
    let g = got
        .transpose(&[0, 3, 1, 2], s)
        .expect("NHWC -> NCHW")
        .to_vec_f32(s)
        .expect("got");
    let w = want.to_vec_f32(s).expect("want");
    assert_eq!(g.len(), w.len(), "{what}: element count");
    let (mut worst, mut peak) = (0.0f32, 0.0f32);
    for (a, b) in g.iter().zip(&w) {
        worst = worst.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    eprintln!("{what:<22} peak {peak:>8.3}  max_abs {worst:.3e}  atol {ATOL:.0e}");
    worst
}

#[test]
fn the_flux_encoder_matches_diffusers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no flux_vae fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = VaeConfig::flux();
    let image = refs
        .get("encoder_input")
        .expect("encoder_input")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();

    let moments = vae::encode_moments_with(&image, &cfg, &w, &s).expect("encode");
    // Twice the latent width: mean, then log-variance.
    assert_eq!(moments.shape(), vec![1, 32, 32, 32]);

    // Compared per half, as `golden_flux_vae.rs` does: the two have very
    // different magnitudes — the log-variance peaks at 29 against the mean's
    // 5.5 — so one combined figure hides which of them moved.
    let got = moments
        .transpose(&[0, 3, 1, 2], &s)
        .unwrap()
        .to_vec_f32(&s)
        .unwrap();
    let want = refs
        .get("encoder_moments")
        .expect("encoder_moments")
        .to_vec_f32(&s)
        .unwrap();
    let per = 32 * 32;
    for (name, range) in [("mean", 0..16usize), ("logvar", 16..32usize)] {
        let (mut worst, mut peak) = (0.0f32, 0.0f32);
        for c in range {
            for i in 0..per {
                let k = c * per + i;
                worst = worst.max((got[k] - want[k]).abs());
                peak = peak.max(want[k].abs());
            }
        }
        eprintln!(
            "encoder {name:<7} peak {peak:>8.3}  max_abs {worst:.3e}  floor              {ENCODER_NOISE_FLOOR:.0e}"
        );
        assert!(
            worst <= ENCODER_NOISE_FLOOR,
            "the Flux encoder's {name} is {worst:.3e} out, past the reference's own              f32 noise floor with room; a structural fault here measures in whole units"
        );
    }
}

/// The scaled mean — where the shift enters.
#[test]
fn the_latent_shift_is_applied_before_the_scale() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no flux_vae fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = VaeConfig::flux();
    let image = refs
        .get("encoder_input")
        .expect("encoder_input")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();

    let scaled = vae::encode_scaled(&image, &cfg, &w, &s).expect("encode_scaled");
    assert_eq!(scaled.shape(), vec![1, 32, 32, 16]);
    let worst = compare(
        &scaled,
        refs.get("encoder_scaled_mean")
            .expect("encoder_scaled_mean"),
        &s,
        "encoder_scaled_mean",
    );
    // This *is* the encoder's mean, multiplied by 0.3611 — so its error is the
    // encoder's error scaled by the same factor, and the floor scales with it.
    // A bound that ignored that would be tighter here than on the tensor this
    // is computed from, which would be an accident rather than a check.
    let floor = ENCODER_NOISE_FLOOR * cfg.scaling_factor;
    assert!(
        worst <= floor,
        "the scaled latent is {worst:.3e} out, past {floor:.3e} — check the shift \
         precedes the scale, which costs whole units rather than fractions of one"
    );
}

/// The decoder, from the raw latent.
#[test]
fn the_flux_decoder_matches_diffusers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no flux_vae fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = VaeConfig::flux();
    let latent = refs
        .get("latent")
        .expect("latent")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();

    let image = vae::decode_with(&latent, &cfg, &w, &s).expect("decode");
    assert_eq!(image.shape(), vec![1, 256, 256, 3]);
    let worst = compare(&image, refs.get("image").expect("image"), &s, "image");
    assert!(worst <= ATOL, "the Flux decoder is {worst:.3e} out");
}

/// Decoding a *scaled* latent: `unscale` then decode, which is what a pipeline
/// does at the end of a run.
#[test]
fn unscaling_inverts_the_shift_and_scale() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no flux_vae fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = VaeConfig::flux();
    // The fixture unscales **`latent`** — the random latent the decoder test
    // uses — not the encoder's mean. `x / scale + shift` applied to a tensor
    // that was never scaled is still a well-defined round trip, and it is the
    // one the reference recorded.
    let raw = refs
        .get("latent")
        .expect("latent")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();

    let latent = cfg.unscale(&raw, &s).unwrap();
    let image = vae::decode_with(&latent, &cfg, &w, &s).expect("decode");
    let worst = compare(
        &image,
        refs.get("decoded_from_scaled")
            .expect("decoded_from_scaled"),
        &s,
        "decoded_from_scaled",
    );
    assert!(worst <= ATOL, "the round trip is {worst:.3e} out");
}

/// **`scale` and `unscale` must be inverses, in opposite orders.**
///
/// Pinned as arithmetic because the failure is a recognisable image with wrong
/// contrast, which no eye check rejects.
#[test]
fn scale_and_unscale_round_trip() {
    let s = Stream::gpu();
    let cfg = VaeConfig::flux();
    let x = Array::from_slice_f32(&[-3.0, -0.5, 0.0, 1.25, 7.0], &[5]).unwrap();

    let round = cfg.unscale(&cfg.scale(&x, &s).unwrap(), &s).unwrap();
    for (a, b) in round
        .to_vec_f32(&s)
        .unwrap()
        .iter()
        .zip(x.to_vec_f32(&s).unwrap())
    {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }

    // And the order is observable: with a nonzero shift, scale-then-shift is a
    // different function from shift-then-scale.
    let scaled = cfg.scale(&x, &s).unwrap().to_vec_f32(&s).unwrap();
    let wrong: Vec<f32> = x
        .to_vec_f32(&s)
        .unwrap()
        .iter()
        .map(|v| v * cfg.scaling_factor - cfg.shift_factor)
        .collect();
    let spread = scaled
        .iter()
        .zip(&wrong)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        spread > 1e-3,
        "the two orders agree to {spread:.3e}; this test cannot see the difference \
         it exists to catch"
    );
}

/// The configurations differ in the ways that matter, and are not each other.
#[test]
fn the_vae_configs_say_what_they_are() {
    let (flux, sd, sdxl, s35) = (
        VaeConfig::flux(),
        VaeConfig::sd15(),
        VaeConfig::sdxl(),
        VaeConfig::sd35(),
    );
    assert_eq!(flux.latent_channels, 16);
    assert_eq!(sd.latent_channels, 4);
    assert_ne!(flux.shift_factor, 0.0, "Flux latents are shifted");
    assert_eq!(sd.shift_factor, 0.0, "SD latents are not");
    assert!(!flux.use_quant_conv, "Flux has no quant convolutions");
    assert!(sd.use_quant_conv, "SD does");
    // SDXL differs from SD 1.5 in exactly one field.
    assert_eq!(sdxl.latent_channels, sd.latent_channels);
    assert_ne!(sdxl.scaling_factor, sd.scaling_factor);
    // SD 3.5 shares Flux's geometry and not its scaling.
    assert_eq!(s35.latent_channels, flux.latent_channels);
    assert_eq!(s35.use_quant_conv, flux.use_quant_conv);
    assert_ne!(s35.scaling_factor, flux.scaling_factor);
}
