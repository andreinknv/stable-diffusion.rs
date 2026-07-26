//! Golden verification for the UNet's spatial transformer.
//!
//! The numerical tests run bottom-up — attn1, attn2, ff, then the whole
//! `Transformer2DModel` — because each stage feeds the next and the first
//! failure names the bug: `attn1` is self-attention or the head reshape,
//! `attn2` is the cross wiring or the kv dim, `ff` is the GEGLU split order,
//! and `transformer_2d` is the permute/reshape sandwich.

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::unet::{Attention, BasicTransformerBlock, FeedForward, Transformer2DModel};
use sd_tensor::nn::{VarBuilder, VarMap};
use sd_tensor::{testing, DType, Device, Tensor};

/// SD 1.5's first down block: 320 channels, 8 heads of 40.
const CHANNELS: usize = 320;
const HEADS: usize = 8;
const DIM_HEAD: usize = 40;
const CROSS_DIM: usize = 768;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unet_attention")
}

fn refs() -> Option<HashMap<String, Tensor>> {
    let path = golden_dir().join("reference.safetensors");
    if !path.exists() {
        eprintln!(
            "SKIP: no reference data.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py unet_attention --output tests/golden\n"
        );
        return None;
    }
    Some(sd_tensor::safetensors::load(&path, &Device::Cpu).expect("loading reference"))
}

/// A VarBuilder over the real weights, or `None` when they are absent.
fn weights(dev: &Device) -> Option<VarBuilder<'static>> {
    let path = golden_dir().join("attention.safetensors");
    if !path.exists() {
        eprintln!("SKIP: no attention.safetensors");
        return None;
    }
    Some(
        sd_loader::safetensors_var_builder(&[&path], DType::F32, dev)
            .expect("loading attention weights"),
    )
}

// -- structural: no reference data needed ---------------------------------

#[test]
fn attention_preserves_shape() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let attn = Attention::new(CHANNELS, None, HEADS, DIM_HEAD, vb).expect("builds");

    let xs = Tensor::zeros((2, 256, CHANNELS), DType::F32, &dev).unwrap();
    let out = attn.forward(&xs, None).expect("forward");
    assert_eq!(out.dims(), &[2, 256, CHANNELS]);
}

#[test]
fn cross_attention_accepts_different_context_length() {
    // 256 queries against 77 keys. Code that assumes the two match is wrong,
    // and this is the shape SD 1.5 actually runs.
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let attn = Attention::new(CHANNELS, Some(CROSS_DIM), HEADS, DIM_HEAD, vb).expect("builds");

    let xs = Tensor::zeros((2, 256, CHANNELS), DType::F32, &dev).unwrap();
    let context = Tensor::zeros((2, 77, CROSS_DIM), DType::F32, &dev).unwrap();
    let out = attn.forward(&xs, Some(&context)).expect("forward");
    assert_eq!(out.dims(), &[2, 256, CHANNELS]);
}

#[test]
fn feedforward_halves_the_gated_projection() {
    // The projection emits 2 * inner and the output is back at `dim`, so a
    // swapped or unsplit gate shows up as a shape error rather than silently.
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let ff = FeedForward::new(CHANNELS, 4, vb).expect("builds");

    let xs = Tensor::zeros((2, 256, CHANNELS), DType::F32, &dev).unwrap();
    let out = ff.forward(&xs).expect("forward");
    assert_eq!(out.dims(), &[2, 256, CHANNELS]);
}

#[test]
fn transformer_preserves_spatial_dims() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let t = Transformer2DModel::new(CHANNELS, HEADS, DIM_HEAD, 1, CROSS_DIM, vb).expect("builds");

    let xs = Tensor::zeros((2, CHANNELS, 16, 16), DType::F32, &dev).unwrap();
    let context = Tensor::zeros((2, 77, CROSS_DIM), DType::F32, &dev).unwrap();
    let out = t.forward(&xs, &context).expect("forward");
    assert_eq!(out.dims(), &[2, CHANNELS, 16, 16]);
}

// -- numerical, bottom-up --------------------------------------------------

/// The reference hooks capture each sub-block's *output*, so comparing means
/// reproducing the input the reference fed it. For attn1 that is
/// `norm1(block_input)`, and so on down the block.
fn block_parts(
    dev: &Device,
) -> Option<(
    HashMap<String, Tensor>,
    BasicTransformerBlock,
    VarBuilder<'static>,
)> {
    let refs = refs()?;
    let vb = weights(dev)?;
    let block = BasicTransformerBlock::new(
        CHANNELS,
        HEADS,
        DIM_HEAD,
        CROSS_DIM,
        vb.pp("transformer_blocks").pp("0"),
    )
    .expect("block builds");
    Some((refs, block, vb))
}

#[test]
fn attn1_self_attention_matches_diffusers() {
    let dev = Device::Cpu;
    let Some((refs, _, vb)) = block_parts(&dev) else {
        return;
    };
    let vb_block = vb.pp("transformer_blocks").pp("0");

    // Rebuild the exact input the hook saw: attn1 runs on norm1(block_input).
    let norm1 = sd_tensor::nn::layer_norm(
        CHANNELS,
        sd_tensor::nn::LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        },
        vb_block.pp("norm1"),
    )
    .expect("norm1");
    let attn1 =
        Attention::new(CHANNELS, None, HEADS, DIM_HEAD, vb_block.pp("attn1")).expect("attn1");

    let input = refs.get("block_input").expect("block_input");
    let normed = sd_tensor::Module::forward(&norm1, input).expect("norm1 forward");
    let got = attn1.forward(&normed, None).expect("attn1 forward");
    let want = refs.get("attn1_output").expect("attn1_output");

    let c = testing::closeness(&got, want).expect("comparing");
    eprintln!("attn1_output: {c}");
    testing::assert_close(&got, want, testing::DEFAULT_ATOL, "attn1 (self-attention)").unwrap();
}

#[test]
fn attn2_cross_attention_matches_diffusers() {
    let dev = Device::Cpu;
    let Some((refs, _, vb)) = block_parts(&dev) else {
        return;
    };
    let vb_block = vb.pp("transformer_blocks").pp("0");

    // attn2 runs on norm2(block_input + attn1_output).
    let norm2 = sd_tensor::nn::layer_norm(
        CHANNELS,
        sd_tensor::nn::LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        },
        vb_block.pp("norm2"),
    )
    .expect("norm2");
    let attn2 = Attention::new(
        CHANNELS,
        Some(CROSS_DIM),
        HEADS,
        DIM_HEAD,
        vb_block.pp("attn2"),
    )
    .expect("attn2");

    let after_attn1 =
        (refs.get("block_input").unwrap() + refs.get("attn1_output").unwrap()).expect("residual");
    let normed = sd_tensor::Module::forward(&norm2, &after_attn1).expect("norm2 forward");
    let got = attn2
        .forward(&normed, Some(refs.get("context").expect("context")))
        .expect("attn2 forward");
    let want = refs.get("attn2_output").expect("attn2_output");

    let c = testing::closeness(&got, want).expect("comparing");
    eprintln!("attn2_output: {c}");
    testing::assert_close(&got, want, testing::DEFAULT_ATOL, "attn2 (cross-attention)").unwrap();
}

#[test]
fn feedforward_matches_diffusers() {
    let dev = Device::Cpu;
    let Some((refs, _, vb)) = block_parts(&dev) else {
        return;
    };
    let vb_block = vb.pp("transformer_blocks").pp("0");

    // ff runs on norm3(block_input + attn1_output + attn2_output).
    let norm3 = sd_tensor::nn::layer_norm(
        CHANNELS,
        sd_tensor::nn::LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        },
        vb_block.pp("norm3"),
    )
    .expect("norm3");
    let ff = FeedForward::new(CHANNELS, 4, vb_block.pp("ff")).expect("ff");

    let after_attn2 = ((refs.get("block_input").unwrap() + refs.get("attn1_output").unwrap())
        .expect("residual 1")
        + refs.get("attn2_output").unwrap())
    .expect("residual 2");
    let normed = sd_tensor::Module::forward(&norm3, &after_attn2).expect("norm3 forward");
    let got = ff.forward(&normed).expect("ff forward");
    let want = refs.get("ff_output").expect("ff_output");

    let c = testing::closeness(&got, want).expect("comparing");
    eprintln!("ff_output: {c}");
    // A swapped GEGLU split lands here, not in the attention tests.
    testing::assert_close(&got, want, testing::DEFAULT_ATOL, "feedforward (GEGLU)").unwrap();
}

#[test]
fn basic_transformer_block_matches_diffusers() {
    let dev = Device::Cpu;
    let Some((refs, block, _)) = block_parts(&dev) else {
        return;
    };
    let got = block
        .forward(
            refs.get("block_input").expect("block_input"),
            refs.get("context").expect("context"),
        )
        .expect("block forward");
    let want = refs.get("block_output").expect("block_output");

    let c = testing::closeness(&got, want).expect("comparing");
    eprintln!("block_output: {c}");
    testing::assert_close(&got, want, testing::DEFAULT_ATOL, "transformer block").unwrap();
}

#[test]
fn transformer_2d_matches_diffusers() {
    let dev = Device::Cpu;
    let Some(refs) = refs() else { return };
    let Some(vb) = weights(&dev) else { return };

    let model =
        Transformer2DModel::new(CHANNELS, HEADS, DIM_HEAD, 1, CROSS_DIM, vb).expect("builds");
    let got = model
        .forward(
            refs.get("attn_input").expect("attn_input"),
            refs.get("context").expect("context"),
        )
        .expect("forward");
    let want = refs.get("attn_output").expect("attn_output");

    let c = testing::closeness(&got, want).expect("comparing");
    eprintln!("attn_output: {c}");
    // The permute/reshape sandwich lands here: a bare reshape gives the right
    // shape and wrong numbers, and only this end-to-end check sees it.
    testing::assert_close(&got, want, testing::DEFAULT_ATOL, "Transformer2DModel").unwrap();
}
