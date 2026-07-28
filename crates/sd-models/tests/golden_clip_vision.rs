//! Golden verification for CLIP's vision tower.
//!
//! Both the sequence and the pooled vector are compared. They differ by one
//! `post_layernorm` — `last_hidden_state` is *not* normed and `pooler_output`
//! is — and applying that norm in the wrong place is the mistake this catches:
//! either output alone would still look reasonable.

use std::path::PathBuf;

use sd_models::clip::{ClipVisionConfig, ClipVisionEncoder};
use sd_tensor::{testing, DType, Device, Tensor};

/// 32 layers of accumulated f32, on activations of order 1-30. The UNet's
/// bound is the same and for the same reason: a purely absolute tolerance here
/// would be measuring float32 rather than the port.
const RTOL: f64 = 1e-3;
const ATOL: f64 = 1e-3;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/clip_vision")
}

#[test]
fn config_matches_the_shipped_image_encoder() {
    // ViT-H/14 at 224: 16 patches across, 256 of them, plus the class token.
    let cfg = ClipVisionConfig::vit_h_14();
    assert_eq!(cfg.hidden_size, 1280);
    assert_eq!(cfg.num_hidden_layers, 32);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.grid(), 16);
    assert_eq!(cfg.sequence_length(), 257);
    // 1280 / 16 = 80 wide per head.
    assert_eq!(cfg.hidden_size / cfg.num_attention_heads, 80);
}

#[test]
fn matches_transformers() {
    let dev = Device::Cpu;
    let refs_path = golden_dir().join("reference.safetensors");
    let weights = golden_dir().join("image_encoder.safetensors");
    if !refs_path.exists() || !weights.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no reference data. Generate it with:\n\n    \
             python3 xtask/golden/dump_reference.py clip_vision --output tests/golden\n"
        );
        return;
    }
    let refs = sd_tensor::safetensors::load(&refs_path, &dev).expect("loading reference");
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev).expect("weights");
    let net = ClipVisionEncoder::new(&ClipVisionConfig::vit_h_14(), vb).expect("building");

    let hidden = net.forward(&refs["pixels"]).expect("forward");
    assert_eq!(hidden.dims(), refs["hidden"].dims());
    let excess = testing::allclose_excess(&hidden, &refs["hidden"], RTOL).expect("compare");
    assert!(excess <= ATOL, "hidden: excess {excess:.3e}");
    println!("hidden excess {excess:.3e}");

    let pooled = net.pooled(&refs["pixels"]).expect("pooled");
    assert_eq!(pooled.dims(), refs["pooled"].dims());
    let excess = testing::allclose_excess(&pooled, &refs["pooled"], RTOL).expect("compare");
    assert!(excess <= ATOL, "pooled: excess {excess:.3e}");
    println!("pooled excess {excess:.3e}");
}

#[test]
fn one_token_is_added_to_the_patch_grid() {
    // 224/14 = 16 across, so 256 patches, plus the class token.
    //
    // Deliberately *not* asserting anything about where the class token sits
    // by comparing token magnitudes: an earlier version did, and it failed
    // while the golden comparison passed. After 32 layers of full attention on
    // a blank image, adjacent patches differ by their position embeddings as
    // much as the class token differs from them, so the premise was unfounded.
    // Ordering is covered exactly by `matches_transformers` — prepending
    // versus appending changes every element of `hidden`.
    let dev = Device::Cpu;
    let weights = golden_dir().join("image_encoder.safetensors");
    if !weights.exists() {
        sd_tensor::skip_missing_fixture!("SKIP one_token_is_added_to_the_patch_grid");
        return;
    }
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev).expect("weights");
    let net = ClipVisionEncoder::new(&ClipVisionConfig::vit_h_14(), vb).expect("building");

    let flat = Tensor::zeros((1, 3, 224, 224), DType::F32, &dev).unwrap();
    let hidden = net.forward(&flat).expect("forward");
    assert_eq!(hidden.dims(), &[1, 257, 1280]);
    assert_eq!(net.pooled(&flat).unwrap().dims(), &[1, 1280]);
}

#[test]
fn the_projected_embedding_is_narrower_than_the_tower() {
    // 1280 in the tower, 1024 out of `visual_projection`. IP-Adapter consumes
    // the projected one, and the widths differ, so this pins which is which.
    let dev = Device::Cpu;
    let weights = golden_dir().join("image_encoder.safetensors");
    if !weights.exists() {
        sd_tensor::skip_missing_fixture!("SKIP the_projected_embedding_is_narrower_than_the_tower");
        return;
    }
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev).expect("weights");
    let net = ClipVisionEncoder::new(&ClipVisionConfig::vit_h_14(), vb).expect("building");

    let flat = Tensor::zeros((1, 3, 224, 224), DType::F32, &dev).unwrap();
    assert_eq!(net.pooled(&flat).unwrap().dims(), &[1, 1280]);
    assert_eq!(net.image_embeds(&flat).unwrap().dims(), &[1, 1024]);
}
