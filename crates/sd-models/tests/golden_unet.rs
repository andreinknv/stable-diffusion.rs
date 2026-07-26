//! Golden verification for the assembled UNet.
//!
//! The skip stack is dumped and compared entry by entry on purpose. With 25
//! blocks between input and output, a single final number says only that
//! something is wrong. The index of the first bad skip says where: 0 is
//! `conv_in`, 1-3 is down block 0, and everything green through 11 means the
//! down pass is fine and the fault is in the mid block or the up pass.

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::unet::{UNet2DConditionModel, UNetConfig};
use sd_tensor::nn::{VarBuilder, VarMap};
use sd_tensor::{testing, DType, Device, Tensor};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unet_full")
}

fn refs() -> Option<HashMap<String, Tensor>> {
    let path = golden_dir().join("reference.safetensors");
    if !path.exists() {
        eprintln!(
            "SKIP: no reference data.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py unet_full --output tests/golden\n"
        );
        return None;
    }
    Some(sd_tensor::safetensors::load(&path, &Device::Cpu).expect("loading reference"))
}

/// The real checkpoint, symlinked next to the reference by the dump script.
fn real_unet(dev: &Device) -> Option<UNet2DConditionModel> {
    let path = golden_dir().join("unet.safetensors");
    if !path.exists() {
        eprintln!("SKIP: no unet.safetensors");
        return None;
    }
    let vb = sd_loader::safetensors_var_builder(&[&path], DType::F32, dev)
        .expect("loading UNet weights");
    Some(UNet2DConditionModel::new(&UNetConfig::sd15(), vb).expect("building UNet"))
}

// -- structural: no reference data needed ---------------------------------

#[test]
fn config_sd15_has_four_blocks_and_768_cross_dim() {
    let cfg = UNetConfig::sd15();
    assert_eq!(cfg.block_out_channels, vec![320, 640, 1280, 1280]);
    assert_eq!(cfg.cross_attention_dim, 768);
    assert_eq!(cfg.layers_per_block, 2);
    assert_eq!(cfg.in_channels, 4);
    assert_eq!(cfg.out_channels, 4);
    // Head *counts*, despite the name. 320 / 8 = 40 wide at the first block.
    assert_eq!(cfg.attention_head_dim, vec![8; 4]);
    assert_eq!(cfg.block_out_channels[0] / cfg.attention_head_dim[0], 40);
    // SD 1.5 attends on every block but the deepest, one transformer each.
    assert_eq!(cfg.down_block_has_attention, vec![true, true, true, false]);
    assert_eq!(cfg.transformer_layers_per_block, vec![1; 4]);
    // No micro-conditioning: that is SDXL's.
    assert!(cfg.addition.is_none());
    // SD 1.5 projects the spatial transformer with 1x1 convolutions.
    assert!(!cfg.use_linear_projection);
    // 1e-5 in the UNet, unlike the VAE's 1e-6.
    assert!((cfg.norm_eps - 1e-5).abs() < f64::EPSILON);
}

#[test]
fn skip_stack_has_twelve_entries() {
    // One for conv_in, then per down block two resnets plus a downsampler,
    // except the deepest block which has neither attention nor a downsampler.
    let cfg = UNetConfig::sd15();
    let skips = cfg.skip_channels();
    assert_eq!(skips.len(), 12, "got {skips:?}");
    assert_eq!(
        skips,
        vec![320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
    );
}

/// A UNet small enough to build and run without a download.
fn tiny_config() -> UNetConfig {
    UNetConfig {
        in_channels: 4,
        out_channels: 4,
        block_out_channels: vec![32, 64],
        layers_per_block: 1,
        attention_head_dim: vec![2, 2],
        transformer_layers_per_block: vec![1, 1],
        down_block_has_attention: vec![true, false],
        cross_attention_dim: 16,
        norm_num_groups: 8,
        norm_eps: 1e-5,
        use_linear_projection: false,
        addition: None,
    }
}

#[test]
fn output_shape_matches_input_shape() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let cfg = tiny_config();
    let unet = UNet2DConditionModel::new(&cfg, vb).expect("builds");

    let sample = Tensor::zeros((2, 4, 16, 16), DType::F32, &dev).unwrap();
    let timestep = Tensor::new(&[500f32, 500.0], &dev).unwrap();
    let context = Tensor::zeros((2, 77, cfg.cross_attention_dim), DType::F32, &dev).unwrap();

    let out = unet.forward(&sample, &timestep, &context).expect("forward");
    assert_eq!(out.dims(), &[2, 4, 16, 16]);
}

#[test]
fn the_skip_stack_is_fully_consumed() {
    // Every skip pushed by the down pass must be popped by the up pass. A
    // leftover means the two are misaligned, which otherwise only shows up as
    // wrong numbers rather than an error.
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let cfg = tiny_config();
    let unet = UNet2DConditionModel::new(&cfg, vb).expect("builds");

    let sample = Tensor::zeros((1, 4, 16, 16), DType::F32, &dev).unwrap();
    let timestep = Tensor::new(&[500f32], &dev).unwrap();
    let context = Tensor::zeros((1, 77, cfg.cross_attention_dim), DType::F32, &dev).unwrap();

    let (_, skips, _) = unet
        .forward_with_skips(&sample, &timestep, &context, None)
        .expect("forward");
    assert_eq!(skips.len(), cfg.skip_channels().len());
}

// -- numerical -------------------------------------------------------------

#[test]
fn down_pass_skips_match_diffusers() {
    let dev = Device::Cpu;
    let Some(refs) = refs() else { return };
    let Some(unet) = real_unet(&dev) else { return };

    let (_, skips, _) = unet
        .forward_with_skips(
            refs.get("sample").expect("sample"),
            refs.get("timestep").expect("timestep"),
            refs.get("context").expect("context"),
            None,
        )
        .expect("forward");
    assert_eq!(skips.len(), 12, "skip stack must have 12 entries");

    let mut first_bad = None;
    for (i, got) in skips.iter().enumerate() {
        let name = format!("down_{i:02}");
        let want = refs
            .get(&name)
            .unwrap_or_else(|| panic!("reference has no {name}"));
        let c = testing::closeness(got, want).expect("comparing");
        eprintln!("{name}: {c}");
        if c.max_abs > testing::DEFAULT_ATOL && first_bad.is_none() {
            first_bad = Some((i, c.max_abs));
        }
    }
    if let Some((i, max_abs)) = first_bad {
        panic!(
            "first bad skip is index {i} (max_abs={max_abs:.3e}). \
             0 is conv_in; 1-3 is down block 0; all-green means the fault is \
             downstream of the down pass."
        );
    }
}

#[test]
fn mid_block_matches_diffusers() {
    let dev = Device::Cpu;
    let Some(refs) = refs() else { return };
    let Some(unet) = real_unet(&dev) else { return };

    let (_, _, mid) = unet
        .forward_with_skips(
            refs.get("sample").expect("sample"),
            refs.get("timestep").expect("timestep"),
            refs.get("context").expect("context"),
            None,
        )
        .expect("forward");
    let want = refs.get("mid_output").expect("mid_output");

    let c = testing::closeness(&mid, want).expect("comparing");
    eprintln!("mid_output: {c}");
    testing::assert_close(&mid, want, testing::DEFAULT_ATOL, "mid block").unwrap();
}

#[test]
fn full_unet_matches_diffusers() {
    let dev = Device::Cpu;
    let Some(refs) = refs() else { return };
    let Some(unet) = real_unet(&dev) else { return };

    let got = unet
        .forward(
            refs.get("sample").expect("sample"),
            refs.get("timestep").expect("timestep"),
            refs.get("context").expect("context"),
        )
        .expect("forward");
    let want = refs.get("output").expect("output");
    assert_eq!(got.dims(), want.dims());

    let c = testing::closeness(&got, want).expect("comparing");
    eprintln!("output: {c}");
    // The task allows 1e-3 here, on the expectation that 25 blocks of
    // accumulated f32 reordering exceeds 1e-4. Measured, it does not: this
    // comes out at 1.1e-5, a 9x margin under the standard tolerance, because
    // the accumulated error stays in the deep 1280-channel blocks and
    // conv_out projects back down to 4 channels. So hold it to 1e-4 and keep
    // the allowance unused — if another platform's BLAS genuinely needs 1e-3,
    // that is a deliberate decision to make then, with this number to compare
    // against, rather than slack granted up front.
    testing::assert_close(&got, want, testing::DEFAULT_ATOL, "full UNet").unwrap();
}
