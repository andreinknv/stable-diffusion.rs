//! Golden verification for ControlNet.
//!
//! All thirteen corrections are compared individually, for the same reason the
//! UNet's skips are: a ControlNet has no image of its own to look at, so these
//! tensors *are* its entire observable behaviour, and a single summary number
//! could not say which of the thirteen went wrong. The index localises the
//! fault — 0 is `conv_in` plus the hint encoder, 1-3 is down block 0, and a
//! green down stack with a red `mid` puts it in the mid block.
//!
//! The structural tests below need no download and run everywhere. The golden
//! ones skip without `tests/golden/controlnet`.

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::controlnet::{ConditioningEmbedding, ControlNet};
use sd_models::unet::{UNet2DConditionModel, UNetConfig};
use sd_tensor::nn::{VarBuilder, VarMap};
use sd_tensor::{testing, DType, Device, Module, Tensor};

/// Same bound as the UNet's, and for the same measured reason: these
/// activations are large enough that a purely absolute tolerance would demand
/// agreement tighter than float32 delivers. See `golden_unet.rs` for the
/// measurement — `xtask/golden/reference_precision.py unet` puts diffusers'
/// own f32-against-f64 gap at 1.108e-4 absolute, 6.850e-6 relative.
const RTOL: f64 = 1e-3;
const ATOL: f64 = 1e-3;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/controlnet")
}

fn refs() -> Option<HashMap<String, Tensor>> {
    let path = golden_dir().join("reference.safetensors");
    if !path.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no reference data.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py controlnet --output tests/golden\n"
        );
        return None;
    }
    Some(sd_tensor::safetensors::load(&path, &Device::Cpu).expect("loading reference"))
}

fn real_controlnet(dev: &Device) -> Option<ControlNet> {
    let path = golden_dir().join("controlnet.safetensors");
    if !path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no controlnet.safetensors");
        return None;
    }
    let vb = sd_loader::safetensors_var_builder(&[&path], DType::F32, dev)
        .expect("loading ControlNet weights");
    Some(ControlNet::new(&UNetConfig::sd15(), vb).expect("building ControlNet"))
}

// -- structural: no download needed ---------------------------------------

/// A ControlNet small enough to build without weights.
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
        class_projection: None,
    }
}

#[test]
fn the_hint_encoder_reduces_by_exactly_eight() {
    // Three stride-2 convolutions, so the same 8x the VAE gives — reached
    // without the VAE. A hint at pixel resolution has to land exactly on the
    // latent grid or nothing downstream lines up.
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let embed = ConditioningEmbedding::new(3, 320, vb).expect("builds");

    let hint = Tensor::zeros((1, 3, 256, 256), DType::F32, &dev).unwrap();
    let out = embed.forward(&hint).expect("forward");
    assert_eq!(out.dims(), &[1, 320, 32, 32]);
}

#[test]
fn there_is_one_correction_per_skip_at_the_skip_s_width() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let cfg = tiny_config();
    let net = ControlNet::new(&cfg, vb).expect("builds");

    let sample = Tensor::zeros((1, 4, 16, 16), DType::F32, &dev).unwrap();
    let timestep = Tensor::new(&[500f32], &dev).unwrap();
    let context = Tensor::zeros((1, 77, cfg.cross_attention_dim), DType::F32, &dev).unwrap();
    let hint = Tensor::zeros((1, 3, 128, 128), DType::F32, &dev).unwrap();

    let control = net
        .forward(&sample, &timestep, &context, &hint, 1.0)
        .expect("forward");

    let widths = cfg.skip_channels();
    assert_eq!(control.down.len(), widths.len());
    for (i, (c, want)) in control.down.iter().zip(&widths).enumerate() {
        assert_eq!(c.dims()[1], *want, "correction {i} has the wrong width");
    }
    assert_eq!(
        control.mid.dims()[1],
        *cfg.block_out_channels.last().unwrap()
    );
}

#[test]
fn the_corrections_fit_the_unet_they_were_built_from() {
    // The contract that matters, asserted by exercising it rather than by
    // comparing shapes: a ControlNet built from a config must be accepted by a
    // UNet built from the same config.
    let dev = Device::Cpu;
    let cfg = tiny_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let net = ControlNet::new(&cfg, vb.pp("controlnet")).expect("builds");
    let unet = UNet2DConditionModel::new(&cfg, vb.pp("unet")).expect("builds");

    let sample = Tensor::zeros((1, 4, 16, 16), DType::F32, &dev).unwrap();
    let timestep = Tensor::new(&[500f32], &dev).unwrap();
    let context = Tensor::zeros((1, 77, cfg.cross_attention_dim), DType::F32, &dev).unwrap();
    let hint = Tensor::zeros((1, 3, 128, 128), DType::F32, &dev).unwrap();

    let control = net
        .forward(&sample, &timestep, &context, &hint, 1.0)
        .expect("control");
    let out = unet
        .forward_controlled(&sample, &timestep, &context, &control.down, &control.mid)
        .expect("controlled forward");
    assert_eq!(out.dims(), &[1, 4, 16, 16]);
}

#[test]
fn a_scale_of_zero_leaves_the_unet_exactly_as_it_was() {
    // The property that makes `--control-scale` safe to expose: at 0 the
    // corrections are zero, so a controlled run must be *bit-identical* to an
    // uncontrolled one. Approximate equality would not do — it would hide a
    // correction that is merely small.
    let dev = Device::Cpu;
    let cfg = tiny_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let net = ControlNet::new(&cfg, vb.pp("controlnet")).expect("builds");
    let unet = UNet2DConditionModel::new(&cfg, vb.pp("unet")).expect("builds");

    let sample = Tensor::randn(0f32, 1.0, (1, 4, 16, 16), &dev).unwrap();
    let timestep = Tensor::new(&[500f32], &dev).unwrap();
    let context = Tensor::randn(0f32, 1.0, (1, 77, cfg.cross_attention_dim), &dev).unwrap();
    let hint = Tensor::randn(0f32, 1.0, (1, 3, 128, 128), &dev).unwrap();

    let control = net
        .forward(&sample, &timestep, &context, &hint, 0.0)
        .expect("control");
    let with = unet
        .forward_controlled(&sample, &timestep, &context, &control.down, &control.mid)
        .expect("controlled");
    let without = unet.forward(&sample, &timestep, &context).expect("plain");

    let a = with.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = without.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a, b, "scale 0 must be bit-identical to no ControlNet");
}

#[test]
fn a_wrong_length_correction_list_is_refused() {
    // Silently zipping to the shorter list would apply corrections to the
    // wrong skips — every shape still valid, the image merely wrong.
    let dev = Device::Cpu;
    let cfg = tiny_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let unet = UNet2DConditionModel::new(&cfg, vb).expect("builds");

    let sample = Tensor::zeros((1, 4, 16, 16), DType::F32, &dev).unwrap();
    let timestep = Tensor::new(&[500f32], &dev).unwrap();
    let context = Tensor::zeros((1, 77, cfg.cross_attention_dim), DType::F32, &dev).unwrap();
    let mid = Tensor::zeros((1, 64, 8, 8), DType::F32, &dev).unwrap();

    let too_few = vec![Tensor::zeros((1, 32, 16, 16), DType::F32, &dev).unwrap()];
    assert!(unet
        .forward_controlled(&sample, &timestep, &context, &too_few, &mid)
        .is_err());
}

// -- golden ---------------------------------------------------------------

#[test]
fn matches_diffusers_correction_for_correction() {
    let dev = Device::Cpu;
    let Some(refs) = refs() else { return };
    let Some(net) = real_controlnet(&dev) else {
        return;
    };

    let sample = &refs["sample"];
    let timestep = &refs["timestep"];
    let context = &refs["context"];
    let hint = &refs["hint"];

    let control = net
        .forward(sample, timestep, context, hint, 1.0)
        .expect("forward");

    let mut worst = 0f64;
    for (i, got) in control.down.iter().enumerate() {
        let key = format!("down_{i:02}");
        let want = &refs[&key];
        assert_eq!(got.dims(), want.dims(), "{key} shape");
        let excess = testing::allclose_excess(got, want, RTOL).expect("compare");
        // The hint encoder feeds correction 0 and nothing else does, so a
        // failure there and nowhere else points at the encoder rather than at
        // the down stack it shares with the UNet.
        assert!(
            excess <= ATOL,
            "{key}: excess {excess:.3e} over rtol*|want|"
        );
        worst = worst.max(excess);
        println!("  {key}  excess {excess:.3e}");
    }

    let excess = testing::allclose_excess(&control.mid, &refs["mid"], RTOL).expect("compare");
    assert!(excess <= ATOL, "mid: excess {excess:.3e}");
    worst = worst.max(excess);
    println!("  mid       excess {excess:.3e}\nworst {worst:.3e}");
}

#[test]
fn scaling_the_corrections_scales_them_linearly() {
    // Not a tautology against the reference: it pins that `scale` multiplies
    // the *output* of the zero convolutions rather than being folded in
    // somewhere it would interact with a nonlinearity.
    let dev = Device::Cpu;
    let Some(refs) = refs() else { return };
    let Some(net) = real_controlnet(&dev) else {
        return;
    };

    let control = net
        .forward(
            &refs["sample"],
            &refs["timestep"],
            &refs["context"],
            &refs["hint"],
            0.5,
        )
        .expect("forward");

    for (i, got) in control.down.iter().enumerate() {
        let want = (&refs[&format!("down_{i:02}")] * 0.5).unwrap();
        let excess = testing::allclose_excess(got, &want, RTOL).expect("compare");
        assert!(excess <= ATOL, "down_{i:02} at scale 0.5: {excess:.3e}");
    }
}
