//! Golden verification for GLIGEN's grounding projection.
//!
//! The Fourier embedding's axis order is what this exists for. Boxes expand to
//! 64 numbers under any permutation of `(coordinate, frequency, sin/cos)`, and
//! all of them load against the same weights — only one lines up with what the
//! MLP was trained on, and the others produce grounding tokens that are wrong
//! without being malformed.
//!
//! The reference masks one of three slots off, so the learned null features
//! are exercised rather than assumed. A reference with every mask set would
//! pass with them ignored.

use std::path::PathBuf;

use sd_models::gligen::PositionNet;
use sd_tensor::{testing, DType, Device};

/// Three small linear layers on inputs of order 1. A plain absolute bound is
/// meaningful here, unlike deep in a UNet.
const TOL: f64 = 1e-4;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/gligen")
}

#[test]
fn matches_the_reference_projection() {
    let dev = Device::Cpu;
    let refs_path = golden_dir().join("reference.safetensors");
    let weights = golden_dir().join("gligen_unet.safetensors");
    if !refs_path.exists() || !weights.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no reference data. Generate it with:\n\n    \
             python3 xtask/golden/dump_reference.py gligen --output tests/golden\n"
        );
        return;
    }
    let refs = sd_tensor::safetensors::load(&refs_path, &dev).expect("reference");
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev).expect("weights");
    let net = PositionNet::new(768, 768, vb.pp("position_net")).expect("builds");

    let objs = net
        .forward(&refs["boxes"], &refs["masks"], &refs["phrases"])
        .expect("forward");
    assert_eq!(objs.dims(), refs["objs"].dims());
    let excess = testing::allclose_excess(&objs, &refs["objs"], 0.0).expect("compare");
    assert!(excess <= TOL, "grounding tokens: max diff {excess:.3e}");
    println!("gligen position_net max diff {excess:.3e}");
}

#[test]
fn a_masked_slot_uses_the_learned_null_not_zeros() {
    // The reference's third slot is masked off, so if the nulls were ignored
    // — padding with zeros instead — the comparison above would already fail.
    // This pins the *reason*: a masked slot must not depend on what was in it.
    let dev = Device::Cpu;
    let refs_path = golden_dir().join("reference.safetensors");
    let weights = golden_dir().join("gligen_unet.safetensors");
    if !refs_path.exists() || !weights.exists() {
        sd_tensor::skip_missing_fixture!("SKIP a_masked_slot_uses_the_learned_null_not_zeros");
        return;
    }
    let refs = sd_tensor::safetensors::load(&refs_path, &dev).expect("reference");
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev).expect("weights");
    let net = PositionNet::new(768, 768, vb.pp("position_net")).expect("builds");

    let base = net
        .forward(&refs["boxes"], &refs["masks"], &refs["phrases"])
        .expect("forward");

    // Change the masked slot's box and phrase entirely; its token must not move.
    let boxes = (&refs["boxes"] * 0.37).unwrap();
    let phrases = (&refs["phrases"] * -2.0).unwrap();
    let altered = net
        .forward(&boxes, &refs["masks"], &phrases)
        .expect("forward");

    let slot = |t: &sd_tensor::Tensor, i: usize| {
        t.narrow(1, i, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    assert_eq!(
        slot(&base, 2),
        slot(&altered, 2),
        "a masked slot depended on its contents"
    );
    assert_ne!(
        slot(&base, 0),
        slot(&altered, 0),
        "an unmasked slot did not move"
    );
}

// -- the fusers, not just the projection ----------------------------------

fn gligen_unet(dev: &Device) -> Option<sd_models::unet::UNet2DConditionModel> {
    let weights = golden_dir().join("gligen_unet.safetensors");
    if !weights.exists() {
        return None;
    }
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, dev).expect("weights");
    Some(
        sd_models::unet::UNet2DConditionModel::new(&sd_models::unet::UNetConfig::sd15(), vb)
            .expect("building a GLIGEN UNet"),
    )
}

#[test]
fn a_grounded_unet_matches_diffusers() {
    // What the projection comparison cannot cover: *where* each fuser sits.
    // Sixteen of them go between attn1 and attn2, and putting them after attn2
    // instead keeps every shape valid.
    let dev = Device::Cpu;
    let refs_path = golden_dir().join("reference.safetensors");
    let (Some(refs), Some(unet)) = (
        refs_path
            .exists()
            .then(|| sd_tensor::safetensors::load(&refs_path, &dev).expect("reference")),
        gligen_unet(&dev),
    ) else {
        sd_tensor::skip_missing_fixture!(
            "SKIP a_grounded_unet_matches_diffusers: missing fixtures"
        );
        return;
    };
    if !refs.contains_key("unet_grounded") {
        sd_tensor::skip_missing_fixture!(
            "SKIP: reference predates the grounded dump; regenerate it"
        );
        return;
    }

    let _guard = sd_models::unet::gligen::with_objs(refs["objs"].clone());
    let out = unet
        .forward(
            &refs["unet_sample"],
            &refs["unet_timestep"],
            &refs["unet_text"],
        )
        .expect("forward");
    let excess = testing::allclose_excess(&out, &refs["unet_grounded"], 1e-3).expect("compare");
    assert!(excess <= 1e-3, "grounded UNet: excess {excess:.3e}");
    println!("grounded unet excess {excess:.3e}");
}

#[test]
fn without_grounding_tokens_the_fusers_do_nothing() {
    // A GLIGEN checkpoint with no tokens supplied must behave as an ordinary
    // SD 1.5 UNet — that is what makes scheduled sampling expressible by
    // simply dropping the guard part way through a run.
    let dev = Device::Cpu;
    let refs_path = golden_dir().join("reference.safetensors");
    let (Some(refs), Some(unet)) = (
        refs_path
            .exists()
            .then(|| sd_tensor::safetensors::load(&refs_path, &dev).expect("reference")),
        gligen_unet(&dev),
    ) else {
        sd_tensor::skip_missing_fixture!("SKIP without_grounding_tokens_the_fusers_do_nothing");
        return;
    };
    if !refs.contains_key("unet_plain") {
        sd_tensor::skip_missing_fixture!("SKIP: reference predates the grounded dump");
        return;
    }

    // No guard: `objs()` is None, so every fuser is skipped.
    let out = unet
        .forward(
            &refs["unet_sample"],
            &refs["unet_timestep"],
            &refs["unet_text"],
        )
        .expect("forward");
    let excess = testing::allclose_excess(&out, &refs["unet_plain"], 1e-3).expect("compare");
    assert!(excess <= 1e-3, "ungrounded UNet: excess {excess:.3e}");

    // And grounding must actually change something, or the test above would
    // pass with the fusers never running at all.
    let _guard = sd_models::unet::gligen::with_objs(refs["objs"].clone());
    let grounded = unet
        .forward(
            &refs["unet_sample"],
            &refs["unet_timestep"],
            &refs["unet_text"],
        )
        .expect("forward");
    let a = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = grounded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_ne!(a, b, "grounding changed nothing");
}
