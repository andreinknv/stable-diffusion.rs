//! Flux's MMDiT against `diffusers`.
//!
//! This compares two independent readings of the *same published file*. Our
//! model reads the black-forest-labs names the checkpoint actually uses;
//! diffusers reads it through `from_single_file`, which renames as it loads.
//! Neither side round-trips the weights through its own conventions first —
//! the mistake that let a legacy VAE attention-naming bug survive a fully
//! green suite earlier in this project.
//!
//! Regenerate with:
//! `python3 xtask/golden/dump_reference.py flux_transformer --output tests/golden`

use std::path::PathBuf;

use sd_models::flux::{rope, FluxConfig, FluxTransformer};
use sd_tensor::{testing, DType, Device};

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(name)
}

#[test]
fn flux_transformer_matches_diffusers() {
    let dev = Device::Cpu;
    let refs_path = golden("flux_transformer/reference.safetensors");
    let weights = golden("flux/flux-mini.safetensors");
    if !refs_path.exists() || !weights.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no Flux transformer reference. Generate with \
             `python3 xtask/golden/dump_reference.py flux_transformer --output tests/golden`"
        );
        return;
    }

    let refs = sd_tensor::safetensors::load(&refs_path, &dev).unwrap();
    // F32: 3.2B parameters is 12.8 GB, which fits on this machine but only
    // just. The guard below is what stops it being discovered the hard way.
    let vb = match sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev) {
        Ok(vb) => vb,
        Err(e) => {
            eprintln!("SKIP: {e}");
            return;
        }
    };

    let cfg = FluxConfig::mini();
    let model = FluxTransformer::new(&cfg, vb).expect("flux-mini tensors should all resolve");

    let scalar = |k: &str| refs.get(k).unwrap().to_vec1::<f32>().unwrap()[0] as usize;
    let (lat_h, lat_w) = (scalar("latent_h"), scalar("latent_w"));

    let img = refs.get("hidden_states").unwrap();
    let txt = refs.get("encoder_hidden_states").unwrap();
    let batch = img.dim(0).unwrap();

    let img_ids = rope::image_ids(batch, lat_h, lat_w, &dev).unwrap();
    let txt_ids = rope::text_ids(batch, txt.dim(1).unwrap(), &dev).unwrap();

    let got = model
        .forward(
            img,
            &img_ids,
            txt,
            &txt_ids,
            refs.get("timestep").unwrap(),
            refs.get("pooled_projections").unwrap(),
            Some(refs.get("guidance").unwrap()),
        )
        .unwrap();

    let want = refs.get("output").unwrap();
    assert_eq!(got.dims(), want.dims(), "output shape");

    let c = testing::closeness(&got, want).unwrap();
    let scale = want
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let excess = testing::allclose_excess(&got, want, 1e-3).unwrap();
    eprintln!(
        "flux transformer: max_abs {:.3e}, mean_abs {:.3e} (scale {:.2}), excess {:.3e}",
        c.max_abs, c.mean_abs, scale, excess
    );

    assert!(
        excess < 1e-3,
        "flux transformer diverged by {excess:.3e} beyond rtol=1e-3 \
         (max_abs {:.3e} against a scale of {scale:.2})",
        c.max_abs
    );

    // mean_abs alongside max_abs: 15 blocks of residuals can hide a broad
    // small drift under a max-based bound sized for outliers. Expressed as a
    // fraction of the output's own mean magnitude — the output runs to a
    // scale of ~300, so an absolute bound here would be measuring the
    // model's amplitude rather than our accuracy.
    let mean_magnitude = want
        .abs()
        .unwrap()
        .mean_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64;
    let relative_drift = c.mean_abs / mean_magnitude;
    eprintln!("  mean magnitude {mean_magnitude:.3}, relative drift {relative_drift:.3e}");
    assert!(
        relative_drift < 1e-5,
        "broad drift across the output: mean_abs {:.3e} against a mean \
         magnitude of {mean_magnitude:.3} ({relative_drift:.3e} relative)",
        c.mean_abs
    );
}

/// Guards the geometry against the published `config.json`.
#[test]
fn mini_config_matches_the_checkpoint() {
    let cfg = FluxConfig::mini();
    assert_eq!(cfg.hidden_size, 3072);
    assert_eq!(cfg.num_heads, 24);
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.depth, 5, "flux-mini has 5 double blocks, dev has 19");
    assert_eq!(cfg.depth_single_blocks, 10, "and 10 single, dev has 38");
    assert_eq!(cfg.in_channels, 64, "16 latent channels x 2x2 patch");
    assert_eq!(cfg.context_in_dim, 4096, "T5-XXL width");
    assert_eq!(cfg.vec_in_dim, 768, "CLIP-L pooled width");
    assert!(cfg.guidance_embed, "flux-mini is distilled like dev");
    assert!(!FluxConfig::schnell().guidance_embed, "schnell is not");
    // 3072 divides by 256, so unlike SD 1.5 this model k-quantises cleanly.
    assert_eq!(cfg.hidden_size % 256, 0);
}
