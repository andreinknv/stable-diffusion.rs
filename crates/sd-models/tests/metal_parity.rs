//! CPU against Metal, module by module.
//!
//! Every golden test in this workspace runs on CPU. That leaves a whole class
//! of defect invisible, and it is not hypothetical — this session shipped one:
//! the unCLIP prior reads its answer with `narrow(1, 80, 1)`, leaving a view
//! whose row stride belongs to the sequence rather than the row. CPU computes
//! the right answer from it; candle's Metal matmul refuses it outright. Nine
//! golden tests passed while the model could not run on the default device.
//!
//! `metal_decoder_parity.rs` covers the VAE decoder, which is where the first
//! such bug was found. This covers the rest, and deliberately keeps each case
//! tiny: the point is to touch every kernel a module uses on both backends,
//! not to make an image.
//!
//! # Two different failures, one test
//!
//! A module can fail here by **erroring** on Metal, which is what the prior
//! did, or by **disagreeing** with the CPU, which is what a backend-specific
//! numerical bug looks like. The first needs no tolerance at all; the second
//! needs one that is loose enough for f32 reduction order and tight enough to
//! catch a wrong kernel. `1e-3` relative against tensors that peak in the
//! tens is roughly 100x the drift a correct backend shows here and far below
//! anything a real fault produces — candle's silently-corrupting Metal
//! convolution, for reference, returned a *dark, banded* image.

use std::path::PathBuf;

use sd_tensor::{testing, DType, Device, Tensor};

/// CPU against Metal on the same weights and inputs, as `atol + rtol*|cpu|`.
const RTOL: f64 = 1e-3;
const ATOL: f64 = 1e-3;

fn golden(sub: &str, file: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(sub)
        .join(file);
    p.exists().then_some(p)
}

fn metal() -> Option<Device> {
    match Device::new_metal(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("SKIP: no Metal device ({e})");
            None
        }
    }
}

/// Compare one module's output across the two devices.
///
/// `build` is called once per device so each holds its own copy of the
/// weights, which is the only way to exercise the backend's own kernels.
fn parity<M>(
    label: &str,
    weights: &PathBuf,
    metal: &Device,
    build: impl Fn(&Device) -> M,
    run: impl Fn(&M, &Device) -> Tensor,
) {
    let cpu = Device::Cpu;
    let on_cpu = run(&build(&cpu), &cpu);
    let on_metal = run(&build(metal), metal);
    let _ = weights;

    assert_eq!(
        on_cpu.dims(),
        on_metal.dims(),
        "{label}: the two devices disagree on shape"
    );
    let metal_on_cpu = on_metal.to_device(&cpu).expect("moving back");
    let excess = testing::allclose_excess(&metal_on_cpu, &on_cpu, RTOL).expect("compare");
    assert!(excess <= ATOL, "{label}: CPU/Metal excess {excess:.3e}");
    println!("{label}: CPU/Metal excess {excess:.3e}");
}

#[test]
fn the_unet_agrees_across_devices() {
    let Some(metal) = metal() else { return };
    let Some(w) = golden("unet_full", "unet.safetensors") else {
        sd_tensor::skip_missing_fixture!("SKIP: no unet.safetensors");
        return;
    };
    use sd_models::unet::{UNet2DConditionModel, UNetConfig};

    parity(
        "sd15 unet",
        &w,
        &metal,
        |dev| {
            let vb = sd_loader::safetensors_var_builder(&[&w], DType::F32, dev).expect("weights");
            UNet2DConditionModel::new(&UNetConfig::sd15(), vb).expect("unet")
        },
        |unet, dev| {
            let mut rng = sd_tensor::rng::SeededRng::new(7);
            let sample = rng.randn((1, 4, 16, 16), dev).expect("sample");
            let timestep = Tensor::from_vec(vec![500f32], 1, dev).expect("t");
            let context = rng.randn((1, 77, 768), dev).expect("context");
            unet.forward(&sample, &timestep, &context).expect("forward")
        },
    );
}

#[test]
fn the_clip_text_encoder_agrees_across_devices() {
    let Some(metal) = metal() else { return };
    let Some(w) = golden("clip_encoder", "clip.safetensors") else {
        sd_tensor::skip_missing_fixture!("SKIP: no clip.safetensors");
        return;
    };
    use sd_models::clip::{ClipTextConfig, ClipTextEncoder};

    // The pooled path too, not just the sequence: it indexes a row out of the
    // hidden states, which is exactly the narrow-into-a-buffer shape that
    // broke on Metal elsewhere.
    parity(
        "clip-l pooled",
        &w,
        &metal,
        |dev| {
            let vb = sd_loader::safetensors_var_builder(&[&w], DType::F32, dev).expect("weights");
            ClipTextEncoder::new(&ClipTextConfig::sd15(), vb).expect("encoder")
        },
        |enc, dev| {
            let mut ids = vec![49407u32; 77];
            ids[0] = 49406;
            ids[1] = 320;
            let ids = Tensor::from_vec(ids, (1, 77), dev).expect("ids");
            enc.pooled_hidden(&ids).expect("pooled")
        },
    );
}

#[test]
fn the_unclip_prior_agrees_across_devices() {
    // The module whose Metal-only failure prompted this file. It is also the
    // one with the least backend coverage: every other architecture here has
    // been through a full pipeline on the GPU for months.
    let Some(metal) = metal() else { return };
    let Some(w) = golden("unclip", "t2i_prior.safetensors") else {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no prior weights. Generate them with:\n\n    \
             python3 xtask/golden/dump_reference.py unclip_prior --output tests/golden\n"
        );
        return;
    };
    use sd_models::prior::{PriorConfig, PriorTransformer};

    parity(
        "unclip prior",
        &w,
        &metal,
        |dev| {
            let vb = sd_loader::safetensors_var_builder(&[&w], DType::F32, dev).expect("weights");
            PriorTransformer::new(&PriorConfig::karlo(), vb).expect("prior")
        },
        |prior, dev| {
            // Batch 2, which is what guidance runs and what the reference
            // tensors never exercise.
            let mut rng = sd_tensor::rng::SeededRng::new(11);
            let latents = rng.randn((2, 768), dev).expect("latents");
            let timestep = Tensor::from_vec(vec![500f32; 2], 2, dev).expect("t");
            let proj = rng.randn((2, 768), dev).expect("proj");
            let hidden = rng.randn((2, 77, 768), dev).expect("hidden");
            // A partial mask, so the masked-attention path runs rather than
            // the trivially-unmasked one.
            let mut mask = vec![0f32; 2 * 77];
            for row in 0..2 {
                for i in 0..10 {
                    mask[row * 77 + i] = 1.0;
                }
            }
            let mask = Tensor::from_vec(mask, (2, 77), dev).expect("mask");
            prior
                .forward(&latents, &timestep, &proj, &hidden, Some(&mask))
                .expect("forward")
        },
    );
}

#[test]
fn the_unclip_class_embedding_agrees_across_devices() {
    let Some(metal) = metal() else { return };
    let Some(w) = golden("unclip", "unet.safetensors") else {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no unCLIP UNet. Generate it with:\n\n    \
             python3 xtask/golden/dump_reference.py unclip --output tests/golden\n"
        );
        return;
    };
    use sd_models::unet::{UNet2DConditionModel, UNetConfig};

    parity(
        "unclip unet",
        &w,
        &metal,
        |dev| {
            let vb = sd_loader::safetensors_var_builder(&[&w], DType::F32, dev).expect("weights");
            UNet2DConditionModel::new(&UNetConfig::unclip(), vb).expect("unet")
        },
        |unet, dev| {
            let mut rng = sd_tensor::rng::SeededRng::new(3);
            let sample = rng.randn((1, 4, 16, 16), dev).expect("sample");
            let timestep = Tensor::from_vec(vec![500f32], 1, dev).expect("t");
            let context = rng.randn((1, 77, 1024), dev).expect("context");
            let labels = rng.randn((1, 2048), dev).expect("labels");
            unet.forward_unclip(&sample, &timestep, &context, &labels)
                .expect("forward")
        },
    );
}
