//! SDXL's second text encoder (OpenCLIP ViT-bigG) against `transformers`.
//!
//! Two things differ from SD 1.5's encoder in ways that produce plausible
//! images rather than errors, so both are pinned here:
//!
//! * the activation is plain `gelu`, not `quick_gelu`;
//! * SDXL conditions on the **penultimate** hidden state, raw, while SD 1.5
//!   uses the final one after `final_layer_norm`.
//!
//! The pooled embedding is checked too. It is taken at the EOS position,
//! located by argmax over the token ids — picking the last index instead
//! lands on padding, which is the same token at the wrong position and so the
//! wrong vector.

use std::path::PathBuf;

use sd_models::clip::{ClipActivation, ClipTextConfig, ClipTextEncoder};
use sd_tensor::{testing, DType, Device};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/sdxl_text_encoder_2")
}

#[test]
fn sdxl_second_encoder_config_is_bigg() {
    let cfg = ClipTextConfig::sdxl_2();
    assert_eq!(cfg.hidden_size, 1280);
    assert_eq!(cfg.num_hidden_layers, 32);
    assert_eq!(cfg.num_attention_heads, 20);
    assert_eq!(cfg.intermediate_size, 5120);
    assert_eq!(cfg.projection_dim, Some(1280));
    // The one that is not a size: bigG activates with plain gelu.
    assert_eq!(cfg.activation, ClipActivation::Gelu);
    // 1280 / 20 = 64, same head width as every other CLIP here.
    assert_eq!(cfg.hidden_size / cfg.num_attention_heads, 64);

    // The first encoder is SD 1.5's, unchanged, and keeps quick_gelu.
    assert_eq!(
        ClipTextConfig::sdxl_1().activation,
        ClipActivation::QuickGelu
    );
    assert_eq!(ClipTextConfig::sdxl_1().hidden_size, 768);
}

#[test]
fn sdxl_second_encoder_matches_transformers() {
    let refs_path = golden_dir().join("reference.safetensors");
    let weights = golden_dir().join("text_encoder_2.safetensors");
    if !refs_path.exists() || !weights.exists() {
        eprintln!(
            "SKIP: no reference data.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py sdxl_text_encoder_2 \
             --output tests/golden\n"
        );
        return;
    }

    let dev = Device::Cpu;
    let refs = sd_tensor::safetensors::load(&refs_path, &dev).expect("loading reference");
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev)
        .expect("loading text_encoder_2 weights");
    let encoder = ClipTextEncoder::new(&ClipTextConfig::sdxl_2(), vb).expect("building encoder");

    let token_ids = refs.get("token_ids").expect("token_ids");

    // The final normed state, as SD 1.5 would use.
    let got_last = encoder.forward(token_ids).expect("forward");
    let c =
        testing::closeness(&got_last, refs.get("last_hidden_state").unwrap()).expect("comparing");
    eprintln!("last_hidden_state: {c}");

    // What SDXL actually conditions on.
    let got_pen = encoder
        .penultimate_hidden_state(token_ids)
        .expect("penultimate");
    let want_pen = refs.get("penultimate").expect("penultimate");
    let c = testing::closeness(&got_pen, want_pen).expect("comparing");
    eprintln!("penultimate:       {c}");
    assert_eq!(got_pen.dims(), &[1, 77, 1280]);

    // The two must not be the same tensor, or "penultimate" is not being
    // taken and this test would pass with the final layer.
    let diff = testing::closeness(&got_pen, &got_last).expect("comparing");
    assert!(
        diff.max_abs > 1e-3,
        "penultimate and final are indistinguishable ({diff}); the layer index is likely wrong"
    );

    // bigG's activations are large, like SD 1.5's CLIP, so compare
    // scale-aware: |a-b| <= atol + rtol*|b|.
    let excess = allclose_excess(&got_pen, want_pen, testing::DEFAULT_RTOL);
    assert!(
        excess <= testing::DEFAULT_ATOL,
        "penultimate: {excess:.3e} beyond the rtol allowance exceeds atol"
    );

    // The pooled embedding, projected.
    let got_pooled = encoder
        .pooled(token_ids)
        .expect("pooled")
        .expect("sdxl_2 has a text_projection");
    let want_pooled = refs.get("pooled").expect("pooled");
    assert_eq!(got_pooled.dims(), want_pooled.dims());
    let c = testing::closeness(&got_pooled, want_pooled).expect("comparing");
    eprintln!("pooled:            {c}");
    let excess = allclose_excess(&got_pooled, want_pooled, testing::DEFAULT_RTOL);
    assert!(
        excess <= testing::DEFAULT_ATOL,
        "pooled: {excess:.3e} beyond the rtol allowance exceeds atol"
    );
}

#[test]
fn an_encoder_without_a_projection_has_no_pooled_output() {
    // SD 1.5's encoder carries no `text_projection`, and asking for a pooled
    // embedding must say so rather than inventing one.
    let dev = Device::Cpu;
    let varmap = sd_tensor::nn::VarMap::new();
    let vb = sd_tensor::nn::VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let cfg = ClipTextConfig {
        vocab_size: 512,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        ..ClipTextConfig::sd15()
    };
    let encoder = ClipTextEncoder::new(&cfg, vb).expect("builds");
    let ids = sd_tensor::Tensor::zeros((1, 77), DType::U32, &dev).unwrap();
    assert!(encoder.pooled(&ids).expect("pooled").is_none());
}

/// See `golden_clip_encoder.rs` — same criterion, same reason.
fn allclose_excess(a: &sd_tensor::Tensor, b: &sd_tensor::Tensor, rtol: f64) -> f64 {
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
