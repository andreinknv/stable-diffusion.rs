//! Golden verification for unCLIP's image-embedding conditioning.
//!
//! Two things are checked, and they fail for different reasons.
//!
//! The **augmentation** is arithmetic on a 1024-vector: whiten, mix in noise
//! from a cosine schedule, un-whiten, append the level's sinusoid. Nothing
//! about it has a shape that could catch a mistake — the two halves of the
//! output are the same width, so a reversed concatenation is well-formed, and
//! the schedule's `t` divides by `n` where SD's own schedules divide by
//! `n - 1`, which moves every alpha by a part in a thousand and nothing else.
//!
//! The **UNet** is checked end to end because the projected embedding is
//! *added into the timestep embedding* before any block runs. Dropping it,
//! projecting it with the wrong weights, or adding it after the wrong thing
//! all return a tensor of exactly the right shape. `unet_out_zero` is the
//! guidance batch's unconditional row and doubles as the control: if it
//! matched `unet_out`, the class path would not be reaching the model at all.
//!
//! # Tolerances, measured rather than chosen
//!
//! `python3 xtask/golden/reference_precision.py unclip` runs diffusers against
//! itself in f64:
//!
//! ```text
//!   augmented_0    peak  4.101   max_abs 6.391e-06   max_rel 1.558e-06
//!   augmented_250  peak  4.479   max_abs 3.290e-07   max_rel 7.345e-08
//!   level_sinusoid peak  1.000   max_abs 1.020e-05   max_rel 1.020e-05
//!   unet_output    peak  1.901   max_abs 2.757e-04   max_rel 1.450e-04
//! ```
//!
//! The UNet's own f32 misses its f64 by **2.757e-04**, so a 1e-4 absolute
//! bound on it would be below the reference's noise floor and would be testing
//! summation order rather than this port.
//!
//! The augmentation looks quiet and is not, for two reasons that took
//! measuring. **`1 - alpha`** cancels: near the top of the ladder `alpha` is
//! within 4e-5 of one, so an absolute 2e-8 difference in the schedule is a
//! *relative* 5e-4 one in what the noise gets multiplied by — which is why
//! level 0 has a floor twenty times level 250's. And the **noise level's
//! sinusoid** is evaluated at arguments up to the level itself; rounding the
//! frequency to f32 costs `250 * 6e-8` in the argument, and `cos` carries that
//! straight through. Both are float32, not this port: an f64 recomputation of
//! either moves it by more than the port and the reference differ by.
//!
//! A first attempt at these numbers held the schedule at diffusers' f32
//! constants and routed the sinusoid through `get_timestep_embedding`, which
//! hardcodes f32 internally. That reported a floor of 2.652e-07 — 40x too low,
//! and it would have condemned a correct implementation.

use std::path::PathBuf;

use sd_models::clip::{ClipVisionConfig, ClipVisionEncoder};
use sd_models::unclip::NoiseAugmentor;
use sd_models::unet::{UNet2DConditionModel, UNetConfig};
use sd_tensor::{testing, DType, Device, Tensor};

/// The augmentation, against a reference noise floor of 1.020e-05 — see the
/// module docs for where that comes from. Five times the floor: tight enough
/// that a wrong schedule (1.1e-3 at this level) or a reversed concatenation
/// (order 1) is caught, and above f32's own trig noise.
const AUGMENT_TOL: f64 = 5e-5;
/// The whole UNet, against a reference noise floor of 2.757e-04.
const UNET_RTOL: f64 = 1e-3;
const UNET_TOL: f64 = 1e-3;
/// The vision tower, matching the bound the ViT-H reference already uses.
const VISION_TOL: f64 = 1e-3;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unclip")
}

const REGENERATE: &str = "SKIP: no unCLIP reference data. Generate it with:\n\n    \
     python3 xtask/golden/dump_reference.py unclip --output tests/golden\n";

/// The reference tensors and one weight file, or `None` if the fixtures are
/// absent — golden data is generated locally, so CI has none of it.
fn fixtures(
    weights: &str,
    dev: &Device,
) -> Option<(std::collections::HashMap<String, Tensor>, PathBuf)> {
    let refs_path = golden_dir().join("reference.safetensors");
    let weights = golden_dir().join(weights);
    if !refs_path.exists() || !weights.exists() {
        sd_tensor::skip_missing_fixture!("{REGENERATE}");
        return None;
    }
    Some((
        sd_tensor::safetensors::load(&refs_path, dev).expect("reference"),
        weights,
    ))
}

#[test]
fn the_augmentation_matches_diffusers() {
    let dev = Device::Cpu;
    let Some((refs, weights)) = fixtures("image_normalizer.safetensors", &dev) else {
        return;
    };
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev).expect("weights");
    let augmentor = NoiseAugmentor::new(1024, vb).expect("builds");
    assert_eq!(augmentor.output_dim(), 2048);

    // **This checkpoint's normalizer is the identity** — `mean` is all zeros
    // and `std` all ones, the constructor's defaults, so the published weights
    // were never trained. That means the comparison below cannot see a swapped
    // `scale`/`unscale`, or either one dropped. `sd_models::unclip`'s unit
    // tests pin the formula instead; this is here so the gap is recorded
    // rather than discovered by whoever meets a checkpoint that does ship
    // statistics.

    // Both ends of the dial. Level 0 barely noises the embedding and is the
    // "reproduce this image" setting; 250 is a working one. Checking only one
    // would pass with the schedule indexed at a constant.
    for level in [0usize, 250] {
        let got = augmentor
            .augment(&refs["image_embeds"], level, &refs["noise"])
            .expect("augment");
        let expected = &refs[&format!("noised_{level}")];
        assert_eq!(got.dims(), expected.dims());
        let excess = testing::allclose_excess(&got, expected, 0.0).expect("compare");
        assert!(
            excess <= AUGMENT_TOL,
            "level {level}: max diff {excess:.3e}"
        );
        println!("augmented at level {level}: max diff {excess:.3e}");
    }
}

#[test]
fn the_level_reaches_the_embedding_and_the_sinusoid_both() {
    // The noise level conditions the model twice — by being the amount of
    // noise mixed in, and by having its own sinusoid appended. Each half must
    // move when it changes, which is what fails if the sinusoid is computed
    // from a constant or the schedule is indexed at one.
    let dev = Device::Cpu;
    let Some((refs, weights)) = fixtures("image_normalizer.safetensors", &dev) else {
        return;
    };
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev).expect("weights");
    let augmentor = NoiseAugmentor::new(1024, vb).expect("builds");

    let half = |t: &Tensor, i: usize| {
        t.narrow(1, i * 1024, 1024)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let low = augmentor
        .augment(&refs["image_embeds"], 0, &refs["noise"])
        .expect("augment");
    let high = augmentor
        .augment(&refs["image_embeds"], 250, &refs["noise"])
        .expect("augment");
    assert_ne!(half(&low, 0), half(&high, 0), "the embedding half is fixed");
    assert_ne!(half(&low, 1), half(&high, 1), "the sinusoid half is fixed");
}

#[test]
fn this_checkpoints_vision_tower_matches_diffusers() {
    // The tower itself is already verified against IP-Adapter's copy; what
    // this pins is that *this* checkpoint's ViT-H loads into it and produces
    // the projected 1024-wide embedding unCLIP conditions on, rather than the
    // pooled 1280 one.
    let dev = Device::Cpu;
    let Some((refs, weights)) = fixtures("image_encoder.safetensors", &dev) else {
        return;
    };
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev).expect("weights");
    let encoder = ClipVisionEncoder::new(&ClipVisionConfig::vit_h_14(), vb).expect("builds");

    let embeds = encoder.image_embeds(&refs["pixels"]).expect("image_embeds");
    assert_eq!(embeds.dims(), refs["image_embeds"].dims());
    let excess = testing::allclose_excess(&embeds, &refs["image_embeds"], 1e-3).expect("compare");
    assert!(excess <= VISION_TOL, "image_embeds: excess {excess:.3e}");
    println!("image_embeds excess {excess:.3e}");
}

fn unclip_unet(dev: &Device) -> Option<(UNet2DConditionModel, PathBuf)> {
    let weights = golden_dir().join("unet.safetensors");
    if !weights.exists() {
        return None;
    }
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, dev).expect("weights");
    let unet =
        UNet2DConditionModel::new(&UNetConfig::unclip(), vb).expect("building an unCLIP UNet");
    Some((unet, weights))
}

#[test]
fn an_unclip_unet_matches_diffusers() {
    let dev = Device::Cpu;
    let Some((refs, _)) = fixtures("unet.safetensors", &dev) else {
        return;
    };
    let Some((unet, _)) = unclip_unet(&dev) else {
        return;
    };
    assert!(unet.takes_class_labels());

    let out = unet
        .forward_unclip(
            &refs["unet_sample"],
            &refs["unet_timestep"],
            &refs["unet_text"],
            &refs["noised_250"],
        )
        .expect("forward");
    assert_eq!(out.dims(), refs["unet_out"].dims());
    let excess = testing::allclose_excess(&out, &refs["unet_out"], UNET_RTOL).expect("compare");
    assert!(excess <= UNET_TOL, "unclip UNet: excess {excess:.3e}");
    println!("unclip unet excess {excess:.3e}");
}

#[test]
fn the_unconditional_row_is_zeros_and_is_a_different_image() {
    // Guidance needs a row that means "no image", and for this architecture
    // that is a **zero vector of the full 2048** — not an absent argument, and
    // not an augmented zero embedding, which would still carry the level's
    // sinusoid. Checked against diffusers rather than assumed, because all
    // three run.
    let dev = Device::Cpu;
    let Some((refs, _)) = fixtures("unet.safetensors", &dev) else {
        return;
    };
    let Some((unet, _)) = unclip_unet(&dev) else {
        return;
    };

    let zeros = Tensor::zeros(refs["noised_250"].dims(), DType::F32, &dev).expect("zeros");
    let out = unet
        .forward_unclip(
            &refs["unet_sample"],
            &refs["unet_timestep"],
            &refs["unet_text"],
            &zeros,
        )
        .expect("forward");
    let excess =
        testing::allclose_excess(&out, &refs["unet_out_zero"], UNET_RTOL).expect("compare");
    assert!(excess <= UNET_TOL, "unconditional row: excess {excess:.3e}");

    // And the two must not be the same image, or everything above would pass
    // with the class embedding never reaching the model.
    let moved = testing::max_abs_diff(&refs["unet_out"], &refs["unet_out_zero"]).expect("diff");
    assert!(
        moved > 0.1,
        "the image embedding changed the output by only {moved:.3e}"
    );
    println!("unconditional excess {excess:.3e}, conditioning moves the output by {moved:.3}");
}

#[test]
fn an_unclip_unet_refuses_a_forward_with_no_image() {
    // The failure this prevents is silent: with the class embedding skipped,
    // every block still runs and the model returns a perfectly ordinary
    // image — the wrong one, conditioned on text alone.
    let dev = Device::Cpu;
    let Some((refs, _)) = fixtures("unet.safetensors", &dev) else {
        return;
    };
    let Some((unet, _)) = unclip_unet(&dev) else {
        return;
    };
    let err = unet
        .forward(
            &refs["unet_sample"],
            &refs["unet_timestep"],
            &refs["unet_text"],
        )
        .expect_err("a plain forward on an unCLIP UNet must be refused");
    assert!(
        err.to_string().contains("forward_unclip"),
        "the error should name the fix, got: {err}"
    );
}
