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
use sd_tensor::{DType, Device, Tensor};

const MULTIPLIER: f64 = 0.8;
/// f32 in a different order on a different device.
const TOL: f32 = 2e-6;

fn lora_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/lora/lcm-lora-sdv1-5.safetensors")
}

fn unet_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unet_full/unet.safetensors")
}

#[test]
fn the_mlx_merge_matches_sd_loaders() {
    if !lora_path().exists() || !unet_path().exists() {
        sd_tensor::skip_missing_fixture!("SKIP: needs the lora and unet_full fixtures.");
        return;
    }
    let s = Stream::gpu();
    let dev = Device::Cpu;

    // candle side.
    let mut candle_weights: HashMap<String, Tensor> =
        sd_tensor::safetensors::load(unet_path(), &dev).expect("unet");
    let candle_lora = sd_loader::Lora::load(lora_path(), &dev).expect("lora");
    let candle_applied = candle_lora
        .merge_into(&mut candle_weights, MULTIPLIER)
        .expect("candle merge");

    // MLX side.
    let mut mlx_weights = load_safetensors(&unet_path()).expect("unet");
    let mlx_raw = load_safetensors(&lora_path()).expect("lora");
    let mlx_lora = MlxLora::from_weights(&mlx_raw, &s).expect("parse");
    let mlx_applied = mlx_lora
        .merge_into(&mut mlx_weights, MULTIPLIER as f32, &s)
        .expect("mlx merge");

    eprintln!(
        "candle merged {} unmatched {} | mlx merged {} unmatched {}",
        candle_applied.merged,
        candle_applied.unmatched.len(),
        mlx_applied.merged,
        mlx_applied.unmatched.len()
    );
    assert!(
        mlx_applied.merged > 0,
        "nothing merged; the mapping missed entirely"
    );
    assert_eq!(
        mlx_applied.merged, candle_applied.merged,
        "the two merges touched different numbers of weights, so the flatten rules have drifted"
    );
    assert_eq!(
        mlx_applied.unmatched.len(),
        candle_applied.unmatched.len(),
        "different unmatched counts"
    );

    // Every merged weight must agree, not just the count.
    let mut checked = 0usize;
    let mut worst = 0.0f32;
    for (name, want) in &candle_weights {
        let Some(got) = mlx_weights.get(name) else {
            panic!("mlx lost the weight {name}");
        };
        let w: Vec<f32> = want
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let g = got.to_vec_f32(&s).unwrap();
        assert_eq!(g.len(), w.len(), "{name}: element count");
        for (a, b) in g.iter().zip(&w) {
            worst = worst.max((a - b).abs());
        }
        checked += 1;
    }
    eprintln!("compared {checked} weights, worst {worst:.3e}");
    assert!(worst <= TOL, "merged weights differ by {worst:.3e}");
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
