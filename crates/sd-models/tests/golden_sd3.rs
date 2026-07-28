//! SD 3.5's MMDiT against `diffusers`.
//!
//! Two independent readings of one published file. Our model reads Stability's
//! names out of `sd3.5_medium.safetensors`; diffusers reads its own converted
//! copy of the same weights. Neither side round-trips through the other's
//! conventions, which is the property that made an earlier VAE naming bug
//! visible only once it was checked this way.
//!
//! Regenerate with:
//! `python3 xtask/golden/dump_reference.py sd3 --output tests/golden`

use std::path::PathBuf;

use sd_models::sd3::{Sd3Config, Sd3Transformer};
use sd_tensor::{testing, DType, Device};

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(name)
}

#[test]
fn sd3_transformer_matches_diffusers() {
    let dev = Device::Cpu;
    let refs_path = golden("sd3_transformer/reference.safetensors");
    let weights = golden("sd35/sd35-medium.safetensors");
    if !refs_path.exists() || !weights.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no SD 3.5 reference. Generate with \
             `python3 xtask/golden/dump_reference.py sd3 --output tests/golden`"
        );
        return;
    }

    let refs = sd_tensor::safetensors::load(&refs_path, &dev).unwrap();
    let vb = match sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev) {
        Ok(vb) => vb,
        Err(e) => {
            eprintln!("SKIP: {e}");
            return;
        }
    };
    // The single-file checkpoint carries the transformer under
    // `model.diffusion_model.` and the VAE under `first_stage_model.`, so the
    // builder is rooted rather than the names being rewritten.
    let vb = vb.pp("model").pp("diffusion_model");

    let cfg = Sd3Config::medium_35();
    let model = Sd3Transformer::new(&cfg, vb).expect("all 24 joint blocks should resolve");

    let got = model
        .forward(
            refs.get("latents").unwrap(),
            refs.get("context").unwrap(),
            refs.get("pooled").unwrap(),
            refs.get("timestep").unwrap(),
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
        "sd3 transformer: max_abs {:.3e}, mean_abs {:.3e} (scale {:.2}), excess {:.3e}",
        c.max_abs, c.mean_abs, scale, excess
    );

    assert!(
        excess < 1e-3,
        "SD 3.5 transformer diverged by {excess:.3e} beyond rtol=1e-3 \
         (max_abs {:.3e} against a scale of {scale:.2})",
        c.max_abs
    );

    // Relative to the output's own magnitude: 24 blocks of residuals can hide
    // a broad small drift under a max-based bound sized for outliers.
    let mean_magnitude = want
        .abs()
        .unwrap()
        .mean_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64;
    let drift = c.mean_abs / mean_magnitude;
    eprintln!("  mean magnitude {mean_magnitude:.3}, relative drift {drift:.3e}");
    assert!(
        drift < 1e-5,
        "broad drift: mean_abs {:.3e} against mean magnitude {mean_magnitude:.3}",
        c.mean_abs
    );
}
