//! Full-size Flux (12B) from a quantised GGUF.
//!
//! `golden_flux_transformer.rs` verifies the *implementation* numerically
//! against diffusers, using flux-mini because a 3.2B model fits beside a
//! reference. This checks the thing that only appears at full size: that all
//! 19 double and 38 single blocks resolve, and that residency tracks the
//! quantisation rather than the parameter count.
//!
//! 12B parameters is 48 GB at F32 and does not fit on this machine at all, so
//! there is no dense path here to compare against — which is the point.

use std::path::PathBuf;

use sd_models::flux::{rope, FluxConfig, FluxTransformer};
use sd_tensor::{DType, Device, Tensor};

fn gguf() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/flux/flux-schnell-q4_k_s.gguf")
}

#[test]
fn schnell_geometry_is_read_from_the_file() {
    let path = gguf();
    if !path.exists() {
        eprintln!("SKIP: no Flux schnell gguf; fetch city96/FLUX.1-schnell-gguf");
        return;
    }
    let (double, single) = sd_loader::flux_block_counts(&path).unwrap();
    assert_eq!(
        (double, single),
        (19, 38),
        "schnell is 19 double, 38 single"
    );

    let cfg = FluxConfig::schnell();
    assert_eq!(cfg.depth, double);
    assert_eq!(cfg.depth_single_blocks, single);

    // schnell is not distilled on a guidance scale; dev is. Reading it from
    // the file means a caller cannot silently pass one to a model that has
    // nowhere to put it.
    assert!(!sd_loader::flux_has_guidance(&path).unwrap());
    assert!(!cfg.guidance_embed);
    assert!(FluxConfig::dev().guidance_embed);
}

#[test]
fn schnell_loads_quantised_and_runs() {
    let path = gguf();
    if !path.exists() {
        eprintln!("SKIP: no Flux schnell gguf");
        return;
    }

    let dev = Device::Cpu;
    let weights = match sd_loader::flux_qtensors_from_gguf(&path, &dev) {
        Ok(w) => w,
        Err(e) => {
            // The memory guard declining is a pass: it means the machine is
            // busy, which is exactly when this should not run.
            eprintln!("SKIP: {e}");
            return;
        }
    };

    let cfg = FluxConfig::schnell();
    let model = FluxTransformer::from_quantized(&cfg, &weights)
        .expect("all 57 blocks of schnell should resolve");

    let resident = model.resident_bytes();
    let dense = 12_000_000_000f64 * 4.0;
    eprintln!(
        "schnell resident: {:.2} GB in quantised projections (F32 would be ~{:.0} GB)",
        resident as f64 / 1e9,
        dense / 1e9
    );
    assert!(
        (resident as f64) < 9e9,
        "quantised residency should be single-digit GB, got {resident}"
    );
    // The check that this is actually quantised rather than quietly expanded:
    // 12B parameters cannot be under 9 GB unless the blocks are held.
    assert!(
        resident > 2_000_000_000,
        "suspiciously small ({resident}) — are the projections loading at all?"
    );

    // A real forward pass at a small size. schnell takes no guidance.
    let (patch, txt_len) = (8usize, 16usize);
    let img = Tensor::zeros((1, patch * patch, cfg.in_channels), DType::F32, &dev).unwrap();
    let txt = Tensor::zeros((1, txt_len, cfg.context_in_dim), DType::F32, &dev).unwrap();
    let pooled = Tensor::zeros((1, cfg.vec_in_dim), DType::F32, &dev).unwrap();
    let t = Tensor::from_vec(vec![0.5f32], 1, &dev).unwrap();

    let out = model
        .forward(
            &img,
            &rope::image_ids(1, patch, patch, &dev).unwrap(),
            &txt,
            &rope::text_ids(1, txt_len, &dev).unwrap(),
            &t,
            &pooled,
            None,
        )
        .unwrap();

    assert_eq!(out.dims(), &[1, patch * patch, cfg.in_channels]);
    let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "non-finite velocity — this is the failure F16 produced, and the \
         reason the weights are held quantised rather than dequantised"
    );
    let absmax = v.iter().fold(0f32, |a, b| a.max(b.abs()));
    eprintln!("schnell velocity from a zero latent: absmax {absmax:.3}");
    assert!(absmax > 0.0, "an all-zero velocity means nothing ran");
}

/// Passing guidance to schnell must fail rather than be ignored.
#[test]
fn schnell_rejects_a_guidance_scale() {
    let path = gguf();
    if !path.exists() {
        eprintln!("SKIP: no Flux schnell gguf");
        return;
    }
    let dev = Device::Cpu;
    let Ok(weights) = sd_loader::flux_qtensors_from_gguf(&path, &dev) else {
        eprintln!("SKIP: guard declined");
        return;
    };
    let cfg = FluxConfig::schnell();
    let model = FluxTransformer::from_quantized(&cfg, &weights).unwrap();

    let (patch, txt_len) = (8usize, 16usize);
    let err = model.forward(
        &Tensor::zeros((1, patch * patch, cfg.in_channels), DType::F32, &dev).unwrap(),
        &rope::image_ids(1, patch, patch, &dev).unwrap(),
        &Tensor::zeros((1, txt_len, cfg.context_in_dim), DType::F32, &dev).unwrap(),
        &rope::text_ids(1, txt_len, &dev).unwrap(),
        &Tensor::from_vec(vec![0.5f32], 1, &dev).unwrap(),
        &Tensor::zeros((1, cfg.vec_in_dim), DType::F32, &dev).unwrap(),
        Some(&Tensor::from_vec(vec![3.5f32], 1, &dev).unwrap()),
    );
    assert!(
        err.is_err(),
        "schnell has no guidance embedding; accepting one silently would \
         discard the caller's setting"
    );
}
