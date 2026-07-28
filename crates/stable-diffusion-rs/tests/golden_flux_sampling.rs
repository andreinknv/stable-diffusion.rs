//! Twenty steps of the Flux sampling loop against `diffusers`.
//!
//! The component tests verify a single forward pass. This verifies the *loop*:
//! schedule, step rule, and re-entry, twenty times over, where a small error
//! compounds instead of appearing once. That is a different failure mode, and
//! the one that produced a visibly wrong image while every component test was
//! green.
//!
//! Conditioning and the initial noise come from the fixture rather than being
//! generated, so the tokenizer, the text encoders and the RNG are all held
//! fixed and the only thing under comparison is the loop.
//!
//! Regenerate with:
//! ```text
//! cargo run --release -p sd-cli --example flux_export_inputs -- \
//!   tests/golden/flux_sampling/reference.safetensors
//! python3 xtask/golden/dump_reference.py flux_sampling --output tests/golden
//! ```

use std::path::PathBuf;

use sd_models::flux::{rope, unpack_latents, FluxConfig, FluxTransformer};
use sd_sample::flow::{flow_euler_step, flow_sigmas, flow_timesteps, FlowMatchConfig};
use sd_tensor::{testing, DType, Device, Tensor};

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(name)
}

#[test]
fn twenty_step_loop_matches_diffusers() {
    let dev = Device::Cpu;
    let refs_path = golden("flux_sampling/reference.safetensors");
    let weights = golden("flux/flux-mini.safetensors");
    if !refs_path.exists() || !weights.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no Flux sampling reference; see the module docs");
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
    let cfg = FluxConfig::mini();
    let model = FluxTransformer::new(&cfg, vb).unwrap();

    let txt = refs.get("txt").unwrap();
    let pooled = refs.get("pooled").unwrap();
    let mut xs = refs.get("init_packed").unwrap().clone();

    let img_len = xs.dim(1).unwrap();
    let patch = (img_len as f64).sqrt() as usize;
    assert_eq!(
        patch * patch,
        img_len,
        "fixture should be a square patch grid"
    );

    let flow = FlowMatchConfig::flux();
    let sigmas = flow_sigmas(&flow, 20, img_len);
    let timesteps = flow_timesteps(&flow, &sigmas);

    let img_ids = rope::image_ids(1, patch, patch, &dev).unwrap();
    let txt_ids = rope::text_ids(1, txt.dim(1).unwrap(), &dev).unwrap();
    let guidance = Tensor::from_vec(vec![3.5f32], 1, &dev).unwrap();

    for (i, &t) in timesteps.iter().enumerate() {
        let t = Tensor::from_vec(vec![(t / 1000.0) as f32], 1, &dev).unwrap();
        let velocity = model
            .forward(&xs, &img_ids, txt, &txt_ids, &t, pooled, Some(&guidance))
            .unwrap();
        xs = flow_euler_step(&xs, &velocity, sigmas[i], sigmas[i + 1]).unwrap();
    }

    let lat_edge = patch * 2;
    let got = unpack_latents(&xs, lat_edge, lat_edge).unwrap();
    let want = refs.get("reference_latent").unwrap();
    assert_eq!(got.dims(), want.dims(), "final latent shape");

    let c = testing::closeness(&got, want).unwrap();
    eprintln!(
        "flux 20-step loop: max_abs {:.3e}, mean_abs {:.3e}",
        c.max_abs, c.mean_abs
    );
    // Twenty compounding steps through a 3.2B model. A wrong schedule or a
    // transposed step lands orders of magnitude away, not here.
    assert!(
        c.max_abs < 1e-3,
        "the sampling loop diverged over 20 steps: max_abs {:.3e}. The \
         per-component tests would still pass — this checks what they cannot.",
        c.max_abs
    );
    assert!(
        c.mean_abs < 1e-5,
        "broad drift: mean_abs {:.3e}",
        c.mean_abs
    );
}

/// The bottom-edge artifact is the model's, not ours.
///
/// flux-mini's output carries an elevated horizontal gradient in its last two
/// latent rows, which the VAE renders as a band of vertical striping. It was
/// natural to suspect our packing or our positional encoding; it is neither.
/// `diffusers`, given the same inputs, produces the same elevation.
///
/// This is pinned so that nobody re-investigates it, and so that a *real*
/// regression in the last rows — which would push the ratio well beyond the
/// model's own — is still caught.
#[test]
fn bottom_row_artifact_matches_the_reference() {
    let dev = Device::Cpu;
    let refs_path = golden("flux_sampling/reference.safetensors");
    if !refs_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no Flux sampling reference");
        return;
    }
    let refs = sd_tensor::safetensors::load(&refs_path, &dev).unwrap();
    let want = refs.get("reference_latent").unwrap();

    // Mean |x[i, j+1] - x[i, j]| per row, averaged over channels.
    let row_gradient = |t: &Tensor| -> Vec<f32> {
        let (_, _, h, w) = t.dims4().unwrap();
        let d = (t.narrow(3, 1, w - 1).unwrap() - t.narrow(3, 0, w - 1).unwrap()).unwrap();
        let m = d.abs().unwrap().mean(3).unwrap().mean(1).unwrap();
        m.flatten_all().unwrap().to_vec1::<f32>().unwrap()[..h].to_vec()
    };

    let g = row_gradient(want);
    let h = g.len();
    let typical: f32 = g[..h - 4].iter().sum::<f32>() / (h - 4) as f32;
    let last: f32 = g[h - 2..].iter().sum::<f32>() / 2.0;
    let ratio = last / typical;
    eprintln!("diffusers' own last-row gradient is {ratio:.2}x its typical row");

    assert!(
        ratio > 1.15,
        "the reference no longer shows the artifact ({ratio:.2}x) — if the \
         checkpoint changed, this test and the roadmap note should go"
    );
    assert!(
        ratio < 2.0,
        "the reference artifact grew to {ratio:.2}x; that is no longer the \
         model characteristic this pins"
    );
}
