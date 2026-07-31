//! Flux's transformer on MLX, against `tests/golden/flux_transformer`.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_flux -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::flux::{self, FluxConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

fn golden(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(sub)
}

fn relative(got: &Array, want: &Array, s: &Stream, what: &str) -> f32 {
    let g = got.to_vec_f32(s).expect("mlx result");
    let w = want.to_vec_f32(s).expect("reference");
    assert_eq!(g.len(), w.len(), "{what}: element count");
    let (mut worst, mut peak) = (0.0f32, 0.0f32);
    for (a, b) in g.iter().zip(&w) {
        worst = worst.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    let rel = worst / peak.max(f32::MIN_POSITIVE);
    eprintln!("{what:<12} peak {peak:>9.3}  max_abs {worst:.3e}  relative {rel:.2e}");
    rel
}

#[test]
fn the_flux_transformer_matches_the_reference() {
    let refs_path = golden("flux_transformer/reference.safetensors");
    let w_path = golden("flux/flux-mini.safetensors");
    if !refs_path.exists() || !w_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no flux fixture.");
        return;
    }
    let refs = load_safetensors(&refs_path).expect("reference");
    let s = Stream::gpu();
    let cfg = FluxConfig::mini();

    let raw = load_safetensors(&w_path).expect("weights");
    let mut w = std::collections::HashMap::new();
    for (name, t) in &raw {
        w.insert(name.clone(), t.to_f32(&s).expect("f32"));
    }

    let scalar = |k: &str| -> usize { refs.get(k).unwrap().to_vec_f32(&s).unwrap()[0] as usize };
    let (lat_h, lat_w) = (scalar("latent_h"), scalar("latent_w"));
    let ids = flux::image_ids(lat_h, lat_w);

    let got = flux::forward(
        refs.get("hidden_states").expect("hidden_states"),
        &ids,
        refs.get("encoder_hidden_states")
            .expect("encoder_hidden_states"),
        refs.get("timestep").expect("timestep"),
        refs.get("pooled_projections").expect("pooled_projections"),
        Some(refs.get("guidance").expect("guidance")),
        &cfg,
        &w,
        &s,
    )
    .unwrap();

    let want = refs.get("output").expect("output");
    assert_eq!(got.shape(), want.shape(), "output shape");
    let rel = relative(&got, want, &s, "output");
    assert!(rel <= 1e-3, "the Flux transformer is {rel:.3e} relative");
}

/// A checkpoint distilled on a guidance scale must be given one, and one that
/// was not must refuse it. Both mistakes otherwise render a plausible wrong
/// image.
#[test]
fn guidance_is_required_exactly_when_the_checkpoint_has_it() {
    let refs_path = golden("flux_transformer/reference.safetensors");
    let w_path = golden("flux/flux-mini.safetensors");
    if !refs_path.exists() || !w_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no flux fixture.");
        return;
    }
    let refs = load_safetensors(&refs_path).expect("reference");
    let s = Stream::gpu();
    let raw = load_safetensors(&w_path).expect("weights");
    let mut w = std::collections::HashMap::new();
    for (name, t) in &raw {
        w.insert(name.clone(), t.to_f32(&s).expect("f32"));
    }
    let ids = flux::image_ids(16, 16);
    let call = |cfg: &FluxConfig, g: Option<&Array>| {
        flux::forward(
            refs.get("hidden_states").unwrap(),
            &ids,
            refs.get("encoder_hidden_states").unwrap(),
            refs.get("timestep").unwrap(),
            refs.get("pooled_projections").unwrap(),
            g,
            cfg,
            &w,
            &s,
        )
    };
    assert!(
        call(&FluxConfig::mini(), None).is_err(),
        "mini needs guidance"
    );
    assert!(
        call(&FluxConfig::schnell(), refs.get("guidance")).is_err(),
        "schnell takes none"
    );
}
