//! The VAE decoder on MLX, against `tests/golden/vae_decoder`.
//!
//! Same fixture and same reference as `golden_vae.rs`. Compared stage by stage
//! — post_quant_conv, conv_in, mid, each up block, conv_out — for the reason
//! the UNet test gives: one number at the end says only that something is
//! wrong, and the VAE's own history here is a padding bug that showed 17.32.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_vae -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::vae;
use sd_tensor::mlx::{load_safetensors, Array, Stream};

/// `sd_tensor::testing::DEFAULT_ATOL`, the bound `golden_vae.rs` uses for the
/// decoded image.
const ATOL: f32 = 1e-4;

/// Intermediates are judged against their own scale, for the reason on
/// [`relative`]. 1e-4 is six times the worst stage measured.
const STAGE_REL: f32 = 1e-4;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/vae_decoder")
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let refs = golden_dir().join("reference.safetensors");
    let vae = golden_dir().join("vae.safetensors");
    if !refs.exists() || !vae.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no vae_decoder fixture.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py vae_decoder --output tests/golden\n"
        );
        return None;
    }
    Some((
        load_safetensors(&refs).expect("loading reference"),
        load_safetensors(&vae).expect("loading VAE weights"),
    ))
}

/// Max absolute difference as a fraction of the tensor's own peak.
///
/// **Absolute bounds are meaningless on this decoder's intermediates**, which
/// peak at 22, 84, 195 and 864 as the image is upsampled. `DEFAULT_ATOL` on the
/// last of those asks for 1.2e-7 relative agreement — far tighter than float32
/// delivers, and a test of summation order rather than of this port. That is
/// `UNET_RTOL`'s argument, and it bites harder here than in the UNet, where the
/// activations only reached 26.6.
///
/// Elementwise `atol + rtol*|want|` is not the right instrument either: the
/// worst absolute error lands on a mid-magnitude element, so `up_block_3`
/// shows 1.5e-3 locally while the tensor as a whole agrees to 4.05e-6. What is
/// being asked here is whether a *block* is right, and error against the
/// block's own scale answers that.
///
/// Measured across the five stages: 2.3e-7, 1.1e-6, 1.6e-5, 5.9e-6, 7.7e-6,
/// 4.1e-6. The bound below is 1e-4 — six times the worst of them, and four
/// orders under a real porting bug, which shows up as O(1): the VAE's own
/// asymmetric-padding bug measured 17.32 against a peak of the same order.
///
/// `golden_vae.rs` sidesteps all of this by comparing only the decoded image,
/// which peaks below 1 and is where an absolute bound is the right tool. That
/// remains the gate; this is localisation for when it fails.
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
    eprintln!("{what:<16} peak {peak:>9.3}  max_abs {worst:.3e}  relative {rel:.2e}");
    rel
}

fn max_abs(got_nhwc: &Array, want_nchw: &Array, s: &Stream, what: &str) -> f32 {
    let got = got_nhwc
        .transpose(&[0, 3, 1, 2], s)
        .expect("NHWC -> NCHW")
        .to_vec_f32(s)
        .expect("mlx result");
    let want = want_nchw.to_vec_f32(s).expect("reference");
    assert_eq!(got.len(), want.len(), "{what}: element count");
    let worst = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{what:<16} max_abs {worst:.3e}   atol {ATOL:.0e}");
    worst
}

/// The decoder end to end: latent in, image out.
#[test]
fn decode_matches_diffusers() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();

    let latent = refs
        .get("latent")
        .expect("latent")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let got = vae::decode(&latent, &w, &s).unwrap();

    assert_eq!(got.shape(), vec![1, 256, 256, 3], "decoded to NHWC");
    let worst = max_abs(&got, refs.get("image").expect("image"), &s, "image");
    assert!(
        worst <= ATOL,
        "the VAE decoder is {worst:.3e} from diffusers, past atol {ATOL:.0e}"
    );
}

/// Stage by stage, so a failure names the block rather than the decoder.
#[test]
fn every_decoder_stage_matches_diffusers() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();

    let latent = refs
        .get("latent")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();

    let h = sd_models::mlx::conv(
        &latent,
        w.get("post_quant_conv.weight").unwrap(),
        w.get("post_quant_conv.bias"),
        0,
        &s,
    )
    .unwrap();
    assert!(
        relative(
            &h,
            refs.get("post_quant_conv").unwrap(),
            &s,
            "post_quant_conv"
        ) <= STAGE_REL,
        "post_quant_conv"
    );

    let h = sd_models::mlx::conv(
        &h,
        w.get("decoder.conv_in.weight").unwrap(),
        w.get("decoder.conv_in.bias"),
        1,
        &s,
    )
    .unwrap();
    assert!(
        relative(&h, refs.get("conv_in").unwrap(), &s, "conv_in") <= STAGE_REL,
        "conv_in"
    );

    let h = vae::mid_block(&h, &w, &s).unwrap();
    assert!(
        relative(&h, refs.get("mid_block").unwrap(), &s, "mid_block") <= STAGE_REL,
        "mid_block"
    );

    let mut h = h;
    let mut first_bad = None;
    for (i, has_up) in [true, true, true, false].into_iter().enumerate() {
        h = vae::up_block(&h, &w, i, has_up, &s).unwrap();
        let name = format!("up_block_{i}");
        let rel = relative(&h, refs.get(&name).unwrap(), &s, &name);
        if rel > STAGE_REL && first_bad.is_none() {
            first_bad = Some((i, rel));
        }
    }
    if let Some((i, rel)) = first_bad {
        panic!("first bad up block is {i} at {rel:.3e} relative, past {STAGE_REL:.0e}");
    }
}

/// The encoder, against the moments diffusers produced from the same image.
///
/// `golden_vae.rs` bounds this with `ENCODER_RTOL`/`ENCODER_ATOL` at 1e-3,
/// relative for the same reason the decoder's stages are — and this is the
/// tensor whose asymmetric downsample padding is the trap: a symmetric pad
/// produces the right shape and shifts the image half a pixel per level.
#[test]
fn encode_matches_diffusers() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();
    const ENCODER_REL: f32 = 1e-4;

    let image = refs
        .get("encoder_input")
        .expect("encoder_input")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let got = vae::encode_moments(&image, &w, &s).unwrap();

    assert_eq!(got.shape(), vec![1, 32, 32, 8], "mean and log-variance");
    let rel = relative(
        &got,
        refs.get("encoder_moments").expect("encoder_moments"),
        &s,
        "encoder_moments",
    );
    assert!(
        rel <= ENCODER_REL,
        "the encoder is {rel:.3e} relative from diffusers; a half-pixel shift \
         from symmetric downsample padding shows here first"
    );
}
