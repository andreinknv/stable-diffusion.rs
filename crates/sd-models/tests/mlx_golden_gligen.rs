//! GLIGEN on MLX, against `tests/golden/gligen`.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_gligen -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::{gligen, unet_forward_with, UNetConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

const ATOL: f32 = 1e-4;

fn max_abs(got_nhwc: &Array, want_nchw: &Array, s: &Stream, what: &str) -> f32 {
    let got = got_nhwc
        .transpose(&[0, 3, 1, 2], s)
        .expect("NHWC -> NCHW")
        .to_vec_f32(s)
        .expect("mlx");
    let want = want_nchw.to_vec_f32(s).expect("reference");
    assert_eq!(got.len(), want.len(), "{what}: element count");
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{what:<16} max_abs {worst:.3e}   atol {ATOL:.0e}");
    worst
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let g = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/gligen");
    let (refs, unet) = (
        g.join("reference.safetensors"),
        g.join("gligen_unet.safetensors"),
    );
    if !refs.exists() || !unet.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no gligen fixture.");
        return None;
    }
    Some((
        load_safetensors(&refs).expect("reference"),
        load_safetensors(&unet).expect("unet"),
    ))
}

#[test]
fn grounded_generation_matches_diffusers() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();
    let cfg = UNetConfig::sd15();

    let x = refs
        .get("unet_sample")
        .expect("unet_sample")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let t = refs.get("unet_timestep").expect("unet_timestep");
    let text = refs.get("unet_text").expect("unet_text");
    let objs = refs.get("objs").expect("objs");

    let got = unet_forward_with(&x, t, text, None, None, None, Some(objs), &cfg, &w, &s).unwrap();
    let worst = max_abs(&got, refs.get("unet_grounded").unwrap(), &s, "grounded");
    assert!(worst <= ATOL, "grounded UNet is {worst:.3e}");
}

/// Without grounding tokens the same checkpoint must reproduce the ungrounded
/// image — the fuser is skipped entirely rather than run with zeros.
#[test]
fn no_grounding_reproduces_the_plain_run() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();
    let cfg = UNetConfig::sd15();

    let x = refs
        .get("unet_sample")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let got = unet_forward_with(
        &x,
        refs.get("unet_timestep").unwrap(),
        refs.get("unet_text").unwrap(),
        None,
        None,
        None,
        None,
        &cfg,
        &w,
        &s,
    )
    .unwrap();
    let worst = max_abs(&got, refs.get("unet_plain").unwrap(), &s, "plain");
    assert!(worst <= ATOL, "ungrounded UNet is {worst:.3e}");

    // And the two must differ, or the grounding never reached the model.
    let a = got.to_vec_f32(&s).unwrap();
    let b = refs.get("unet_grounded").unwrap().to_vec_f32(&s).unwrap();
    let spread = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        spread > 1e-3,
        "grounded and plain are the same image ({spread:.3e}); the boxes did nothing"
    );
}

/// `position_net` turns boxes and phrases into the grounding tokens the fuser
/// attends over. **The Fourier axis order is the whole subtlety**: the axes are
/// `(coordinate, frequency, sin/cos)` and flatten as
/// `(frequency, sin/cos, coordinate)`. Any ordering yields 64 numbers and loads
/// against the same weights.
#[test]
fn position_net_matches_diffusers() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();

    let got = gligen::position_net(
        refs.get("boxes").expect("boxes"),
        refs.get("masks").expect("masks"),
        refs.get("phrases").expect("phrases"),
        &w,
        &s,
    )
    .unwrap();
    let want = refs.get("objs").expect("objs");
    assert_eq!(got.shape(), want.shape(), "grounding token shape");

    let g = got.to_vec_f32(&s).unwrap();
    let wv = want.to_vec_f32(&s).unwrap();
    let (mut worst, mut peak) = (0.0f32, 0.0f32);
    for (a, b) in g.iter().zip(&wv) {
        worst = worst.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    eprintln!("position_net    peak {peak:.3}  max_abs {worst:.3e}");
    assert!(worst <= 1e-3, "position_net is {worst:.3e} from diffusers");
}
