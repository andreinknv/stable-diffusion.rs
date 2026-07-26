//! Golden verification for the CLIP text encoder.
//!
//! The reference captures every encoder layer's output, not just the final
//! tensor. That is the whole point: a transposed head reshape or the wrong
//! activation produces a correctly-shaped result that is quietly wrong, and a
//! single final number cannot say where it went wrong. Comparing layer by
//! layer names the first one that diverged.

use std::path::PathBuf;

use sd_models::clip::{ClipTextConfig, ClipTextEncoder};
use sd_tensor::nn::{VarBuilder, VarMap};
use sd_tensor::{testing, DType, Device, Tensor};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/clip_encoder")
}

/// Worst violation of `|a - b| <= atol + rtol * |b|`, in units of `atol`.
///
/// `testing::assert_close` compares absolute error only, which is the right
/// criterion for the VAE, where activations are order 1. It is the wrong one
/// here. CLIP's text encoder carries *massive activations*: this reference
/// peaks at 851.03 at `[0, 0, 681]` and holds it through all twelve layers,
/// and an f32 ULP at that magnitude is 6.1e-5. Demanding 1e-4 absolute there
/// asks for more significant digits than f32 has, so it would fail for any
/// implementation, correct or not, and would keep failing however the code was
/// changed.
///
/// So this is the standard combined criterion — the same one `numpy.allclose`
/// and `torch.allclose` use — with the tolerances the repo already declares.
/// `DEFAULT_RTOL` exists in `sd_tensor::testing` and was never wired into
/// `assert_close`; this is what it is for. On order-1 values the bound is
/// still ~`atol`, so nothing is relaxed where accuracy is actually testable.
///
/// Returns the max of `|a - b| - rtol * |b|`. Compare it against `atol`.
fn allclose_excess(a: &Tensor, b: &Tensor, rtol: f64) -> f64 {
    let a = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let b = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let diff = (&a - &b).unwrap().abs().unwrap();
    let allowance = (b.abs().unwrap() * rtol).unwrap();
    (diff - allowance)
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64
}

#[test]
fn config_sd15_has_expected_dimensions() {
    let cfg = ClipTextConfig::sd15();
    assert_eq!(cfg.vocab_size, 49408);
    assert_eq!(cfg.hidden_size, 768);
    assert_eq!(cfg.intermediate_size, 3072);
    assert_eq!(cfg.num_hidden_layers, 12);
    assert_eq!(cfg.num_attention_heads, 12);
    assert_eq!(cfg.max_position_embeddings, 77);
    // 1e-5, not the VAE's 1e-6. Getting this wrong is a small uniform offset
    // that reads as noise rather than as a bug.
    assert!((cfg.layer_norm_eps - 1e-5).abs() < f64::EPSILON);
    // Implied, but it is the number every head reshape depends on.
    assert_eq!(cfg.hidden_size / cfg.num_attention_heads, 64);
}

/// A small encoder, so the structural tests stay fast and need no download.
fn tiny_config() -> ClipTextConfig {
    ClipTextConfig {
        vocab_size: 512,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        max_position_embeddings: 77,
        layer_norm_eps: 1e-5,
    }
}

#[test]
fn encoder_builds_with_random_weights() {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    ClipTextEncoder::new(&tiny_config(), vb).expect("encoder should build from a fresh VarMap");
}

#[test]
fn encoder_output_shape_is_batch_77_768() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let cfg = tiny_config();
    let encoder = ClipTextEncoder::new(&cfg, vb).expect("encoder builds");

    for batch in [1usize, 3] {
        let ids = Tensor::zeros((batch, 77), DType::U32, &dev).unwrap();
        let out = encoder.forward(&ids).expect("forward");
        assert_eq!(out.dims(), &[batch, 77, cfg.hidden_size]);
    }
}

#[test]
fn every_layer_output_is_returned_for_diagnosis() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let cfg = tiny_config();
    let encoder = ClipTextEncoder::new(&cfg, vb).expect("encoder builds");

    let ids = Tensor::zeros((1, 77), DType::U32, &dev).unwrap();
    let (out, layers) = encoder.forward_with_layers(&ids).expect("forward");
    assert_eq!(layers.len(), cfg.num_hidden_layers);
    for (i, layer) in layers.iter().enumerate() {
        assert_eq!(layer.dims(), &[1, 77, cfg.hidden_size], "layer {i}");
    }
    // The final output is the last layer *after* final_layer_norm, so it must
    // not be the same tensor the last layer emitted.
    let c = testing::closeness(&out, layers.last().unwrap()).unwrap();
    assert!(c.max_abs > 0.0, "final_layer_norm appears to be a no-op");
}

#[test]
fn matches_transformers_reference() {
    let dir = golden_dir();
    let refs_path = dir.join("reference.safetensors");
    let weights_path = dir.join("clip.safetensors");
    if !refs_path.exists() || !weights_path.exists() {
        eprintln!(
            "SKIP matches_transformers_reference: no reference data.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py clip_encoder --output tests/golden\n\
             \nSee xtask/golden/README.md."
        );
        return;
    }

    let dev = Device::Cpu;
    let refs = sd_tensor::safetensors::load(&refs_path, &dev).expect("loading reference tensors");
    let vb = sd_loader::safetensors_var_builder(&[&weights_path], DType::F32, &dev)
        .expect("loading CLIP weights");
    let encoder = ClipTextEncoder::new(&ClipTextConfig::sd15(), vb).expect("building encoder");

    let token_ids = refs.get("token_ids").expect("reference has 'token_ids'");
    let expected = refs
        .get("last_hidden_state")
        .expect("reference has 'last_hidden_state'");

    // The embeddings feed every layer, so check them first: if they are wrong,
    // layer_00 would be blamed for a fault that is upstream of it.
    let got_embeddings = encoder.embeddings(token_ids).expect("embeddings");
    let c = testing::closeness(&got_embeddings, refs.get("embeddings").unwrap())
        .expect("comparing embeddings");
    eprintln!("embeddings: {c}");
    testing::assert_close(
        &got_embeddings,
        refs.get("embeddings").unwrap(),
        testing::DEFAULT_ATOL,
        "clip embeddings",
    )
    .unwrap();

    let (got, layers) = encoder.forward_with_layers(token_ids).expect("forward");

    // Report every layer before asserting on any of them, so a failure shows
    // where the divergence starts rather than only that it exists.
    let mut first_bad: Option<(String, f64, f64)> = None;
    for (i, layer) in layers.iter().enumerate() {
        let name = format!("layer_{i:02}");
        let Some(want) = refs.get(&name) else {
            continue;
        };
        let c = testing::closeness(layer, want).expect("comparing layer");
        let excess = allclose_excess(layer, want, testing::DEFAULT_RTOL);
        eprintln!("{name}: {c}, allclose excess={excess:.3e}");
        if excess > testing::DEFAULT_ATOL && first_bad.is_none() {
            first_bad = Some((name, c.max_abs, excess));
        }
    }
    if let Some((name, max_abs, excess)) = first_bad {
        panic!(
            "first divergence at {name}: max_abs={max_abs:.3e}, and {excess:.3e} of that is \
             beyond what rtol={:.0e} allows, exceeding atol={:.0e}",
            testing::DEFAULT_RTOL,
            testing::DEFAULT_ATOL,
        );
    }

    let c = testing::closeness(&got, expected).expect("comparing final output");
    let excess = allclose_excess(&got, expected, testing::DEFAULT_RTOL);
    eprintln!("last_hidden_state: {c}, allclose excess={excess:.3e}");
    assert!(
        excess <= testing::DEFAULT_ATOL,
        "clip last_hidden_state: {excess:.3e} beyond rtol allowance exceeds atol {:.0e}\n  {c}",
        testing::DEFAULT_ATOL
    );

    // The final output is held to plain absolute tolerance as well, and meets
    // it. `final_layer_norm` divides the massive activation out — the peak
    // falls from 851 to 33 — so the tensor SD actually conditions on is back
    // in a range where 1e-4 absolute is a meaningful demand. The relative
    // criterion above is needed only for the raw per-layer captures.
    testing::assert_close(
        &got,
        expected,
        testing::DEFAULT_ATOL,
        "clip last_hidden_state (absolute)",
    )
    .unwrap();
}
