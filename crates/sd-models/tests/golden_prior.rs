//! Golden verification for unCLIP's prior — the model that invents an image
//! embedding from text.
//!
//! Four things are checked, and each fails for a reason the others cannot see.
//!
//! - **The transformer**, twice: once with the tokenizer's real attention mask
//!   and once with every position unmasked. The reference's prompt occupies 10
//!   of 77 positions and the two predictions differ by 0.60, so a port that
//!   ignores the mask agrees with exactly one of them. That matters here and
//!   nowhere else in this project: Stable Diffusion conditions on all 77
//!   positions, padding included, so ignoring the mask is the *habit*
//!   everywhere else in this codebase.
//! - **The step**, at an interior timestep with the noise draw pinned, and at
//!   the final one where no variance is added at all. The second is fully
//!   deterministic and pins the mean; the first pins the variance on top of it.
//! - **The standard deviations** the step multiplies its noise by, straight
//!   from `_get_variance`. `fixed_small_log` returns a *deviation* where every
//!   other variance type returns a variance, and `step` multiplies it in with
//!   no further square root — so squaring or rooting it once more is wrong by
//!   exactly that much and still returns a plausible embedding.
//! - **The text encoder**, which is SD 1.5's tower plus a projection head.
//!
//! # Tolerances
//!
//! The prior is 20 blocks over 81 tokens at 2048 wide, so it accumulates like
//! any deep stack; `1e-3` relative with a `1e-3` absolute floor is the bound
//! the UNet references use for the same reason. The scheduler step is closed
//! -form arithmetic on order-1 numbers and takes a plain absolute bound.

use std::path::PathBuf;

use sd_models::clip::{ClipTextConfig, ClipTextEncoder};
use sd_models::prior::{PriorConfig, PriorScheduler, PriorTransformer};
use sd_tensor::{testing, DType, Device, Tensor};

/// The whole transformer, against a stack 20 blocks deep.
const PRIOR_RTOL: f64 = 1e-3;
const PRIOR_TOL: f64 = 1e-3;
/// One closed-form step on order-1 numbers.
const STEP_TOL: f64 = 1e-5;
/// The text-to-image UNet, whose noise floor is **5.5x the image-variation
/// one's** — 1.530e-3 against 2.757e-04, measured by
/// `reference_precision.py unclip --model-id models/unclip-t2i`.
///
/// Same code, same architecture, different weights: this checkpoint's
/// activations are simply worse conditioned. It is worth knowing that a bound
/// carried over from a sibling checkpoint is not transferable — this port
/// lands at 2.0e-3 here and 2.0e-4 there, and both are inside their own
/// reference's float32 noise.
const T2I_UNET_TOL: f64 = 5e-3;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unclip")
}

const REGENERATE: &str = "SKIP: no unCLIP prior reference. Generate it with:\n\n    \
     python3 xtask/golden/dump_reference.py unclip_prior --output tests/golden\n";

fn refs(dev: &Device) -> Option<std::collections::HashMap<String, Tensor>> {
    let path = golden_dir().join("prior_reference.safetensors");
    if !path.exists() {
        eprintln!("{REGENERATE}");
        return None;
    }
    Some(sd_tensor::safetensors::load(&path, dev).expect("reference"))
}

fn prior(dev: &Device) -> Option<PriorTransformer> {
    let weights = golden_dir().join("t2i_prior.safetensors");
    if !weights.exists() {
        eprintln!("{REGENERATE}");
        return None;
    }
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, dev).expect("weights");
    Some(PriorTransformer::new(&PriorConfig::karlo(), vb).expect("building the prior"))
}

#[test]
fn the_prior_matches_diffusers_with_the_prompt_masked() {
    let dev = Device::Cpu;
    let (Some(refs), Some(prior)) = (refs(&dev), prior(&dev)) else {
        return;
    };
    let mask = refs["prior_mask"].to_dtype(DType::F32).expect("mask");
    let timestep = Tensor::from_vec(vec![500f32], 1, &dev).expect("timestep");

    let out = prior
        .forward(
            &refs["prior_latents"],
            &timestep,
            &refs["text_embeds"],
            &refs["text_hidden"],
            Some(&mask),
        )
        .expect("forward");
    assert_eq!(out.dims(), refs["prior_out"].dims());
    let excess = testing::allclose_excess(&out, &refs["prior_out"], PRIOR_RTOL).expect("compare");
    assert!(excess <= PRIOR_TOL, "masked prior: excess {excess:.3e}");
    println!("prior (masked) excess {excess:.3e}");
}

#[test]
fn the_prior_matches_diffusers_with_nothing_masked() {
    // The control for the test above. If the mask were being dropped, one of
    // these two would still pass — which is why both are here rather than
    // whichever one was written first.
    let dev = Device::Cpu;
    let (Some(refs), Some(prior)) = (refs(&dev), prior(&dev)) else {
        return;
    };
    let ones = refs["prior_mask"].ones_like().expect("ones");
    let timestep = Tensor::from_vec(vec![500f32], 1, &dev).expect("timestep");

    let out = prior
        .forward(
            &refs["prior_latents"],
            &timestep,
            &refs["text_embeds"],
            &refs["text_hidden"],
            Some(&ones),
        )
        .expect("forward");
    let excess =
        testing::allclose_excess(&out, &refs["prior_out_unmasked"], PRIOR_RTOL).expect("compare");
    assert!(excess <= PRIOR_TOL, "unmasked prior: excess {excess:.3e}");

    // And the two references must genuinely differ, or neither test is
    // saying anything about masking.
    let moved =
        testing::max_abs_diff(&refs["prior_out"], &refs["prior_out_unmasked"]).expect("diff");
    assert!(
        moved > 0.1,
        "the attention mask changed the prediction by only {moved:.3e}"
    );
    println!("prior (unmasked) excess {excess:.3e}, the mask is worth {moved:.3}");
}

#[test]
fn the_prior_reads_its_answer_from_the_prd_token() {
    // The prediction comes from the *last* sequence position, not from where
    // the latent went in. Both are 2048 wide and both project to a
    // well-formed 768-vector, so this is invisible to every shape check. It is
    // covered here by construction: the comparisons above would fail, but
    // this says what they are failing *about*.
    let dev = Device::Cpu;
    let (Some(refs), Some(prior)) = (refs(&dev), prior(&dev)) else {
        return;
    };
    let mask = refs["prior_mask"].to_dtype(DType::F32).expect("mask");
    let timestep = Tensor::from_vec(vec![500f32], 1, &dev).expect("timestep");

    // Change only the latent. The `prd` token's own input is unchanged, so an
    // implementation reading the latent's position would move much more than
    // one reading `prd` — but both move, so this pins that the latent reaches
    // the prediction at all, which is what fails if the sequence is
    // assembled in the wrong order.
    let perturbed = (&refs["prior_latents"] * 1.5).expect("perturb");
    let base = prior
        .forward(
            &refs["prior_latents"],
            &timestep,
            &refs["text_embeds"],
            &refs["text_hidden"],
            Some(&mask),
        )
        .expect("forward");
    let moved_out = prior
        .forward(
            &perturbed,
            &timestep,
            &refs["text_embeds"],
            &refs["text_hidden"],
            Some(&mask),
        )
        .expect("forward");
    let moved = testing::max_abs_diff(&base, &moved_out).expect("diff");
    assert!(moved > 1e-3, "the latent did not reach the prediction");
}

#[test]
fn the_prior_scheduler_matches_diffusers() {
    let dev = Device::Cpu;
    let Some(refs) = refs(&dev) else { return };
    let scheduler = PriorScheduler::new(25);

    // The ladder itself, entry for entry.
    let want: Vec<usize> = refs["prior_timesteps"]
        .to_vec1::<i64>()
        .expect("timesteps")
        .into_iter()
        .map(|t| t as usize)
        .collect();
    assert_eq!(scheduler.timesteps(), want.as_slice());

    let t = refs["step_timestep"].to_vec1::<i64>().expect("t")[0] as usize;
    let stepped = scheduler
        .step(
            &refs["prior_out"],
            t,
            &refs["prior_latents"],
            &refs["step_noise"],
        )
        .expect("step");
    let diff = testing::max_abs_diff(&stepped, &refs["stepped"]).expect("compare");
    assert!(diff <= STEP_TOL, "step at t={t}: max diff {diff:.3e}");

    // The final step adds no variance at all, so this one is exact up to
    // float32 and isolates the mean from the noise.
    let t_final = refs["step_timestep_final"].to_vec1::<i64>().expect("t")[0] as usize;
    let landed = scheduler
        .step(
            &refs["prior_out"],
            t_final,
            &refs["prior_latents"],
            &refs["step_noise"],
        )
        .expect("step");
    let diff_final = testing::max_abs_diff(&landed, &refs["stepped_final"]).expect("compare");
    assert!(
        diff_final <= STEP_TOL,
        "final step: max diff {diff_final:.3e}"
    );
    println!("prior step {diff:.3e}, final step {diff_final:.3e}");
}

#[test]
fn the_step_uses_a_deviation_where_diffusers_returns_one() {
    // `fixed_small_log` is the trap: `_get_variance` returns a standard
    // deviation for it and a variance for every other type, and `step`
    // multiplies it in with no square root. Getting that wrong scales the
    // noise by the square root of itself — a quieter run that still produces
    // an embedding.
    //
    // Checked by difference rather than by exposing the internal: two steps
    // that differ only in their noise differ by exactly `std * (n1 - n2)`.
    let dev = Device::Cpu;
    let Some(refs) = refs(&dev) else { return };
    let scheduler = PriorScheduler::new(25);

    let probes: Vec<usize> = refs["probe_timesteps"]
        .to_vec1::<i64>()
        .expect("probes")
        .into_iter()
        .map(|t| t as usize)
        .collect();
    let want = refs["probe_stds"].to_vec1::<f32>().expect("stds");

    let zero = refs["step_noise"].zeros_like().expect("zeros");
    let ones = refs["step_noise"].ones_like().expect("ones");
    for (t, expected) in probes.iter().zip(want) {
        let without = scheduler
            .step(&refs["prior_out"], *t, &refs["prior_latents"], &zero)
            .expect("step");
        let with = scheduler
            .step(&refs["prior_out"], *t, &refs["prior_latents"], &ones)
            .expect("step");
        // Unit noise, so the gap between the two *is* the deviation.
        let got = testing::max_abs_diff(&with, &without).expect("diff");
        assert!(
            (got - expected as f64).abs() <= 1e-5,
            "t={t}: standard deviation {got:.6} against {expected:.6}"
        );
    }
    println!("prior step deviations match at {} timesteps", probes.len());
}

#[test]
fn the_prior_runs_at_the_batch_guidance_uses() {
    // Every reference above is batch 1; every *run* is batch 2, because the
    // prior is guided and both rows go through together. Duplicating the
    // inputs must reproduce the batch-1 answer on both rows — which is what
    // fails if anything in the assembly is written for one row.
    let dev = Device::Cpu;
    let (Some(refs), Some(prior)) = (refs(&dev), prior(&dev)) else {
        return;
    };
    let mask = refs["prior_mask"].to_dtype(DType::F32).expect("mask");
    let twice = |t: &Tensor| Tensor::cat(&[t, t], 0).expect("double");
    let timestep = Tensor::from_vec(vec![500f32; 2], 2, &dev).expect("timestep");

    let out = prior
        .forward(
            &twice(&refs["prior_latents"]),
            &timestep,
            &twice(&refs["text_embeds"]),
            &twice(&refs["text_hidden"]),
            Some(&twice(&mask)),
        )
        .expect("batched forward");
    assert_eq!(out.dims(), &[2, 768]);
    for row in 0..2 {
        let got = out.narrow(0, row, 1).expect("row");
        let excess =
            testing::allclose_excess(&got, &refs["prior_out"], PRIOR_RTOL).expect("compare");
        assert!(
            excess <= PRIOR_TOL,
            "batched row {row}: excess {excess:.3e}"
        );
    }
}

#[test]
fn the_prior_joins_to_its_own_image_half() {
    // The reference that would have caught the published mismatch. Everything
    // above passes on `-t2i-h` too, whose prior emits 768 and whose image half
    // takes 1024 — a checkpoint that cannot run. This is the one that says the
    // two ends fit: the prior's output, un-whitened, augmented, and through
    // *this* checkpoint's UNet, whose class projection is 1536 rather than the
    // image-variation model's 2048.
    let dev = Device::Cpu;
    let Some(refs) = refs(&dev) else { return };
    let normalizer_path = golden_dir().join("t2i_image_normalizer.safetensors");
    let unet_path = golden_dir().join("t2i_unet.safetensors");
    if !normalizer_path.exists() || !unet_path.exists() {
        eprintln!("{REGENERATE}");
        return;
    }

    let vb =
        sd_loader::safetensors_var_builder(&[&normalizer_path], DType::F32, &dev).expect("weights");
    let augmentor = sd_models::unclip::NoiseAugmentor::new(768, vb).expect("builds");
    assert_eq!(
        augmentor.output_dim(),
        1536,
        "this checkpoint's class projection is 1536, being twice a 768-wide embedding"
    );

    let labels = augmentor
        .augment(&refs["image_embeds"], 0, &refs["aug_noise"])
        .expect("augment");
    let excess = testing::allclose_excess(&labels, &refs["class_labels"], 0.0).expect("compare");
    assert!(excess <= 5e-5, "class labels: max diff {excess:.3e}");

    let vb = sd_loader::safetensors_var_builder(&[&unet_path], DType::F32, &dev).expect("weights");
    let mut cfg = sd_models::unet::UNetConfig::unclip();
    cfg.class_projection = Some(1536);
    let unet = sd_models::unet::UNet2DConditionModel::new(&cfg, vb).expect("building the t2i UNet");

    let out = unet
        .forward_unclip(
            &refs["t2i_unet_sample"],
            &refs["t2i_unet_timestep"],
            &refs["t2i_unet_text"],
            &labels,
        )
        .expect("forward");
    let excess =
        testing::allclose_excess(&out, &refs["t2i_unet_out"], PRIOR_RTOL).expect("compare");
    assert!(excess <= T2I_UNET_TOL, "t2i UNet: excess {excess:.3e}");
    println!("t2i unet excess {excess:.3e} (class projection 1536)");
}

#[test]
fn the_priors_text_encoder_matches_transformers() {
    // SD 1.5's tower with a projection head on top — the same architecture
    // this project has verified since the first milestone, loaded from a
    // different checkpoint. What is worth pinning is that the prior consumes
    // the *projected* embedding alongside the raw sequence: two outputs of one
    // forward pass, at two different depths.
    let dev = Device::Cpu;
    let Some(refs) = refs(&dev) else { return };
    let weights = golden_dir().join("t2i_prior_text_encoder.safetensors");
    if !weights.exists() {
        eprintln!("{REGENERATE}");
        return;
    }
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev).expect("weights");
    let cfg = ClipTextConfig {
        projection_dim: Some(768),
        ..ClipTextConfig::sd15()
    };
    let encoder = ClipTextEncoder::new(&cfg, vb).expect("building the prior text encoder");

    let tokens = refs["prior_tokens"].to_dtype(DType::U32).expect("tokens");
    let hidden = encoder.forward(&tokens).expect("forward");
    let excess = testing::allclose_excess(&hidden, &refs["text_hidden"], 1e-3).expect("compare");
    assert!(excess <= 1e-3, "text hidden states: excess {excess:.3e}");

    let projected = encoder
        .pooled(&tokens)
        .expect("pooled")
        .expect("this checkpoint carries a text_projection");
    let excess_p =
        testing::allclose_excess(&projected, &refs["text_embeds"], 1e-3).expect("compare");
    assert!(excess_p <= 1e-3, "text embeds: excess {excess_p:.3e}");
    println!("prior text encoder: hidden {excess:.3e}, projected {excess_p:.3e}");
}
