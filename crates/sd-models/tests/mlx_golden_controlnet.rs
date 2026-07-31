//! ControlNet on MLX, against `tests/golden/controlnet`.
//!
//! Twelve down corrections and one mid correction, compared entry by entry for
//! the reason the UNet's skip stack is: one number at the end says only that
//! something is wrong.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_controlnet -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::{controlnet, UNetConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

/// `golden_controlnet.rs` holds this to `DEFAULT_ATOL`, noting the agreement is
/// "tighter than float32 delivers" only where the tensors are order 1. These
/// corrections are, so absolute is the right instrument — the UNet's own skips
/// pass at 1e-4 as well.
const ATOL: f32 = 1e-4;

fn max_abs(got_nhwc: &Array, want_nchw: &Array, s: &Stream, what: &str) -> f32 {
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
    eprintln!("{what:<10} peak {peak:>8.3}  max_abs {worst:.3e}   atol {ATOL:.0e}");
    worst
}

#[test]
fn the_controlnet_matches_diffusers() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/controlnet");
    let (refs_path, cn_path) = (
        dir.join("reference.safetensors"),
        dir.join("controlnet.safetensors"),
    );
    if !refs_path.exists() || !cn_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no controlnet fixture.");
        return;
    }
    let refs = load_safetensors(&refs_path).expect("reference");
    let w = load_safetensors(&cn_path).expect("weights");
    let s = Stream::gpu();
    let cfg = UNetConfig::sd15();

    let sample = refs
        .get("sample")
        .expect("sample")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let hint = refs
        .get("hint")
        .expect("hint")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();

    let control = controlnet::forward(
        &sample,
        refs.get("timestep").unwrap(),
        refs.get("context").unwrap(),
        &hint,
        1.0,
        &cfg,
        &w,
        &s,
    )
    .unwrap();

    assert_eq!(control.down.len(), 12, "one correction per UNet skip");

    let mut first_bad = None;
    for (i, got) in control.down.iter().enumerate() {
        let name = format!("down_{i:02}");
        let worst = max_abs(got, refs.get(&name).unwrap(), &s, &name);
        if worst > ATOL && first_bad.is_none() {
            first_bad = Some((i, worst));
        }
    }
    let mid = max_abs(&control.mid, refs.get("mid").unwrap(), &s, "mid");
    if let Some((i, worst)) = first_bad {
        panic!("first bad correction is {i} at {worst:.3e}, past atol {ATOL:.0e}");
    }
    assert!(mid <= ATOL, "the mid correction is {mid:.3e}");
}

/// `scale = 0` must contribute *exactly* nothing, not merely almost nothing —
/// a caller disabling a ControlNet expects the uncontrolled image bit for bit.
#[test]
fn a_zero_scale_contributes_exactly_nothing() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/controlnet");
    let (refs_path, cn_path) = (
        dir.join("reference.safetensors"),
        dir.join("controlnet.safetensors"),
    );
    if !refs_path.exists() || !cn_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no controlnet fixture.");
        return;
    }
    let refs = load_safetensors(&refs_path).expect("reference");
    let w = load_safetensors(&cn_path).expect("weights");
    let s = Stream::gpu();

    let sample = refs
        .get("sample")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let hint = refs
        .get("hint")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();

    let control = controlnet::forward(
        &sample,
        refs.get("timestep").unwrap(),
        refs.get("context").unwrap(),
        &hint,
        0.0,
        &UNetConfig::sd15(),
        &w,
        &s,
    )
    .unwrap();

    for (i, c) in control.down.iter().enumerate() {
        for v in c.to_vec_f32(&s).unwrap() {
            assert_eq!(v, 0.0, "down correction {i} is not exactly zero at scale 0");
        }
    }
    for v in control.mid.to_vec_f32(&s).unwrap() {
        assert_eq!(v, 0.0, "the mid correction is not exactly zero at scale 0");
    }
}
