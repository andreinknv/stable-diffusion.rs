//! SDXL's UNet.
//!
//! Structurally different from SD 1.5's in ways that are easy to get backwards:
//! three blocks rather than four, attention on the *last two* rather than the
//! first three, a ten-deep transformer at the bottom, and the `text_time`
//! micro-conditioning that SD 1.5 has no equivalent of.

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::unet::{UNet2DConditionModel, UNetConfig};
use sd_tensor::{testing, DType, Device, Tensor};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/sdxl_unet")
}

fn refs() -> Option<HashMap<String, Tensor>> {
    let path = golden_dir().join("reference.safetensors");
    if !path.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no reference data.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py sdxl_unet --output tests/golden\n"
        );
        return None;
    }
    Some(sd_tensor::safetensors::load(&path, &Device::Cpu).expect("loading reference"))
}

#[test]
fn sdxl_config_differs_from_sd15_where_it_should() {
    let x = UNetConfig::sdxl();
    let s = UNetConfig::sd15();

    assert_eq!(x.block_out_channels, vec![320, 640, 1280]);
    assert_eq!(x.cross_attention_dim, 2048, "both encoders concatenated");
    // Head counts, all giving 64-wide heads.
    assert_eq!(x.attention_head_dim, vec![5, 10, 20]);
    for (i, &c) in x.block_out_channels.iter().enumerate() {
        assert_eq!(c / x.attention_head_dim[i], 64, "block {i} head width");
    }
    // The deepest block carries ten transformers, not one.
    assert_eq!(x.transformer_layers_per_block, vec![1, 2, 10]);

    // The attention pattern is the *opposite end* from SD 1.5: SDXL skips the
    // shallowest block, SD 1.5 skips the deepest. Reversing this loads a
    // plausible-looking model that cannot find its weights.
    assert_eq!(x.down_block_has_attention, vec![false, true, true]);
    assert_eq!(s.down_block_has_attention, vec![true, true, true, false]);

    assert!(x.addition.is_some(), "SDXL has text_time conditioning");
    assert!(s.addition.is_none(), "SD 1.5 has none");
    let add = x.addition.unwrap();
    assert_eq!(add.time_embed_dim, 256);
    // 6 time ids * 256 + 1280 pooled = 2816.
    assert_eq!(add.projection_input_dim, 6 * 256 + 1280);
}

#[test]
fn sdxl_skip_stack_has_nine_entries() {
    // conv_in, then per block two resnets plus a downsampler except the last:
    // 1 + 3 + 3 + 2 = 9. SD 1.5 has 12, and using that count here misaligns
    // every up block.
    let skips = UNetConfig::sdxl().skip_channels();
    assert_eq!(skips.len(), 9, "got {skips:?}");
    assert_eq!(skips, vec![320, 320, 320, 320, 640, 640, 640, 1280, 1280]);
}

#[test]
fn sdxl_unet_matches_diffusers() {
    let Some(refs) = refs() else { return };
    let weights = golden_dir().join("unet.safetensors");
    if !weights.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no unet.safetensors");
        return;
    }

    let dev = Device::Cpu;
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev)
        .expect("loading SDXL UNet weights");
    let unet = UNet2DConditionModel::new(&UNetConfig::sdxl(), vb).expect("building SDXL UNet");

    let got = unet
        .forward_sdxl(
            refs.get("sample").expect("sample"),
            refs.get("timestep").expect("timestep"),
            refs.get("context").expect("context"),
            refs.get("pooled").expect("pooled"),
            refs.get("time_ids").expect("time_ids"),
        )
        .expect("forward_sdxl");
    let want = refs.get("output").expect("output");
    assert_eq!(got.dims(), want.dims());

    let c = testing::closeness(&got, want).expect("comparing");
    eprintln!("sdxl output: {c}");
    testing::assert_close(&got, want, testing::DEFAULT_ATOL, "SDXL UNet").unwrap();
}

#[test]
fn an_sdxl_unet_refuses_to_run_without_micro_conditioning() {
    // The conditioning is not optional: an SDXL UNet run through the SD 1.5
    // entry point would silently drop it and denoise toward the wrong thing.
    let dev = Device::Cpu;
    let varmap = sd_tensor::nn::VarMap::new();
    let vb = sd_tensor::nn::VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let cfg = UNetConfig {
        block_out_channels: vec![32, 64],
        attention_head_dim: vec![2, 2],
        transformer_layers_per_block: vec![1, 1],
        down_block_has_attention: vec![false, true],
        cross_attention_dim: 16,
        norm_num_groups: 8,
        ..UNetConfig::sdxl()
    };
    let unet = UNet2DConditionModel::new(&cfg, vb).expect("builds");

    let sample = Tensor::zeros((1, 4, 16, 16), DType::F32, &dev).unwrap();
    let timestep = Tensor::new(&[500f32], &dev).unwrap();
    let context = Tensor::zeros((1, 77, 16), DType::F32, &dev).unwrap();

    let err = unet
        .forward(&sample, &timestep, &context)
        .expect_err("must refuse without micro-conditioning");
    assert!(
        err.to_string().contains("forward_sdxl"),
        "the error should name the right entry point: {err}"
    );
}
