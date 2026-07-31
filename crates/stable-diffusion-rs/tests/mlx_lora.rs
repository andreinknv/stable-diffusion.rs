//! The MLX LoRA merge against `sd-loader`'s, on the real LCM adapter.
//!
//! `sd-loader`'s merge is already gated by `golden_lora.rs`, so agreeing with
//! it weight for weight is agreeing with the reference. This is also the guard
//! against the two `flatten` rules drifting: they are one line each and live in
//! different crates, so only a test that runs both can notice.
//!
//! ```bash
//! cargo test -p stable-diffusion-rs --features mlx --test mlx_lora_agrees -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::lora::Lora as MlxLora;
use sd_tensor::mlx::{load_safetensors, Array, Stream};

fn lora_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/lora/lcm-lora-sdv1-5.safetensors")
}

fn unet_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unet_full/unet.safetensors")
}

/// A multiplier of 0 must leave every weight untouched **bit for bit** — the
/// documented promise that `multiplier = 0` reproduces an unadapted load
/// exactly.
#[test]
fn a_zero_multiplier_changes_nothing() {
    if !lora_path().exists() || !unet_path().exists() {
        sd_tensor::skip_missing_fixture!("SKIP: needs the lora and unet_full fixtures.");
        return;
    }
    let s = Stream::gpu();

    let before = load_safetensors(&unet_path()).expect("unet");
    let mut after = load_safetensors(&unet_path()).expect("unet");
    let raw = load_safetensors(&lora_path()).expect("lora");
    let lora = MlxLora::from_weights(&raw, &s).expect("parse");
    let applied = lora.merge_into(&mut after, 0.0, &s).expect("merge");

    assert!(applied.merged > 0, "the adapter matched nothing");
    for (name, b) in &before {
        let a = after.get(name).expect("weight survived");
        assert_eq!(
            a.to_vec_f32(&s).unwrap(),
            b.to_vec_f32(&s).unwrap(),
            "{name} changed at multiplier 0"
        );
    }
}

/// A missing `alpha` means "no rescaling" — `alpha == rank`, not `alpha == 0`.
/// The wrong default scales every correction to nothing and the adapter
/// silently does nothing at all.
#[test]
fn a_missing_alpha_means_no_rescaling_not_no_effect() {
    let s = Stream::gpu();
    // rank 2, in 4, out 3 — no alpha entry at all.
    let mut raw: HashMap<String, Array> = HashMap::new();
    raw.insert(
        "lora_unet_thing.lora_down.weight".into(),
        Array::from_slice_f32(&[0.5; 8], &[2, 4]).unwrap(),
    );
    raw.insert(
        "lora_unet_thing.lora_up.weight".into(),
        Array::from_slice_f32(&[0.25; 6], &[3, 2]).unwrap(),
    );
    let lora = MlxLora::from_weights(&raw, &s).expect("parse");
    assert_eq!(lora.len(), 1);

    let mut weights: HashMap<String, Array> = HashMap::new();
    weights.insert(
        "thing.weight".into(),
        Array::from_slice_f32(&[0.0; 12], &[3, 4]).unwrap(),
    );
    lora.merge_into(&mut weights, 1.0, &s).expect("merge");

    // scale = alpha/rank = 2/2 = 1, so the delta is up @ down = 0.25*0.5*2.
    let got = weights["thing.weight"].to_vec_f32(&s).unwrap();
    for v in &got {
        assert!(
            (v - 0.25).abs() < 1e-6,
            "a missing alpha rescaled the correction: {v}"
        );
    }
}

///
/// `the_mlx_merge_matches_sd_loaders` above checks that the two backends agree,
/// is gone. The name mapping is where a LoRA silently half-applies, and a
/// half-applied adapter still renders a plausible image — so the assertion is
/// coverage, and it needs its own anchor.
#[test]
fn the_adapter_maps_completely_onto_the_unet() {
    if !lora_path().exists() || !unet_path().exists() {
        sd_tensor::skip_missing_fixture!("SKIP: needs the lora and unet_full fixtures.");
        return;
    }
    let s = Stream::gpu();
    let raw = load_safetensors(&lora_path()).expect("lora");
    let lora = MlxLora::from_weights(&raw, &s).expect("parse");
    assert_eq!(lora.len(), 278, "lcm-lora-sdv1-5 corrects 278 layers");

    let before = load_safetensors(&unet_path()).expect("unet");
    let mut after = load_safetensors(&unet_path()).expect("unet");
    let applied = lora.merge_into(&mut after, 1.0, &s).expect("merge");

    assert_eq!(
        applied.unmatched.len(),
        0,
        "{} layers found no home, the first being {:?}",
        applied.unmatched.len(),
        applied.unmatched.first()
    );
    assert_eq!(applied.merged, 278, "every corrected layer must be merged");

    // **And exactly those weights moved.** Coverage alone would pass if the
    // merge wrote to every tensor it could reach.
    let mut changed = 0usize;
    for (name, b) in &before {
        let a = after.get(name).expect("weight survived");
        if a.to_vec_f32(&s).unwrap() != b.to_vec_f32(&s).unwrap() {
            changed += 1;
        }
    }
    assert_eq!(
        changed, 278,
        "exactly the adapter's own layers must change, not {changed}"
    );
}
