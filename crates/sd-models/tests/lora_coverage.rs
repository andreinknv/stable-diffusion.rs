//! Does a published LoRA map onto our SD 1.5 UNet, completely?
//!
//! This is the test the feature lives or dies by. The merge arithmetic is
//! three lines and hard to get subtly wrong; the *name mapping* is where a
//! LoRA silently half-applies, and a half-applied adapter still renders a
//! plausible image. So the assertion is coverage, not plausibility.

use std::collections::HashMap;
use std::path::PathBuf;

use sd_tensor::{DType, Device, Tensor};

fn lora_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/lora/lcm-lora-sdv1-5.safetensors")
}

fn unet_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/sd15/unet/diffusion_pytorch_model.safetensors")
}

fn weights() -> Option<HashMap<String, Tensor>> {
    let p = unet_path();
    if !p.exists() {
        eprintln!("SKIP: no SD 1.5 UNet at {}", p.display());
        return None;
    }
    Some(sd_tensor::safetensors::load(&p, &Device::Cpu).expect("loading UNet"))
}

#[test]
fn every_lora_entry_finds_a_weight_in_the_unet() {
    if !lora_path().exists() {
        eprintln!("SKIP: no LoRA fixture at {}", lora_path().display());
        return;
    }
    let Some(mut w) = weights() else { return };

    let lora = sd_loader::Lora::load(lora_path(), &Device::Cpu).expect("loading LoRA");
    assert_eq!(lora.len(), 278, "lcm-lora-sdv1-5 corrects 278 layers");

    let applied = lora.merge_into(&mut w, 1.0).expect("merging");
    assert_eq!(
        applied.unmatched.len(),
        0,
        "every entry must find a weight; {} did not, first: {:?}",
        applied.unmatched.len(),
        applied.unmatched.first()
    );
    assert_eq!(
        applied.merged, 278,
        "every entry must be merged exactly once"
    );
}

#[test]
fn a_zero_multiplier_leaves_every_weight_untouched() {
    // The identity that makes the feature safe to expose: --lora-scale 0 must
    // be indistinguishable from not passing --lora at all, bit for bit. If it
    // is not, the merge is doing something other than adding a scaled delta.
    if !lora_path().exists() {
        eprintln!("SKIP: no LoRA fixture.");
        return;
    }
    let Some(before) = weights() else { return };
    let mut after = before.clone();

    let lora = sd_loader::Lora::load(lora_path(), &Device::Cpu).expect("loading LoRA");
    let applied = lora.merge_into(&mut after, 0.0).expect("merging");
    assert!(applied.merged > 0, "the test is vacuous if nothing matched");

    for (name, a) in &after {
        let b = before.get(name).expect("same keys");
        let diff = (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0, "{name} changed at multiplier 0");
    }
}

#[test]
fn a_nonzero_multiplier_changes_exactly_the_targeted_weights() {
    // The complement of the test above: the adapter must actually do
    // something, and only to the layers it names. A merge that touched
    // everything would pass the coverage test and still be wrong.
    if !lora_path().exists() {
        eprintln!("SKIP: no LoRA fixture.");
        return;
    }
    let Some(before) = weights() else { return };
    let mut after = before.clone();

    let lora = sd_loader::Lora::load(lora_path(), &Device::Cpu).expect("loading LoRA");
    lora.merge_into(&mut after, 1.0).expect("merging");

    let mut changed = 0usize;
    for (name, a) in &after {
        let b = before.get(name).expect("same keys");
        let diff = (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        if diff > 0.0 {
            changed += 1;
            assert!(
                name.ends_with(".weight"),
                "{name} is not a weight tensor and must not have moved"
            );
        }
    }
    assert_eq!(changed, 278, "exactly the adapter's own layers must change");
}
