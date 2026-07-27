//! Golden verification for IP-Adapter.
//!
//! The comparison that matters is end-to-end through the UNet, because the
//! thing most likely to be wrong is the **index mapping**. The checkpoint
//! numbers its sixteen entries by diffusers' flat processor order — down
//! blocks, up blocks, then mid — while this UNet builds down, mid, up. A wrong
//! mapping puts every correction on a differently-sized layer, which usually
//! fails to load; but between the two 1280-wide regions it would not, and the
//! image would simply be wrong.
//!
//! So: no per-module comparison would catch it, and this one does.

use std::path::PathBuf;

use sd_models::ip_adapter::{ImageProjModel, NUM_TOKENS};
use sd_models::unet::{ip, UNet2DConditionModel, UNetConfig};
use sd_tensor::{testing, DType, Device, Tensor};

/// The UNet's own bound, for the same measured reason — see `golden_unet.rs`.
const RTOL: f64 = 1e-3;
const ATOL: f64 = 1e-3;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/ip_adapter")
}

fn unet_weights() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unet_full/unet.safetensors")
}

#[test]
fn the_entry_order_is_down_then_mid_then_up() {
    // Pinned as a constant because it is derived from diffusers' traversal
    // order, not from anything visible in this codebase, and it is silent when
    // wrong. Mid is entry 15 in the checkpoint and slot 6 here.
    let order = ip::IpSource::sd15_order();
    assert_eq!(order.len(), 16, "sixteen cross-attention layers");
    assert_eq!(
        order,
        vec![0, 1, 2, 3, 4, 5, 15, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    );
}

#[test]
fn the_projection_makes_four_tokens() {
    let dev = Device::Cpu;
    let refs_path = golden_dir().join("reference.safetensors");
    let ip_path = golden_dir().join("ip-adapter_sd15.safetensors");
    if !refs_path.exists() || !ip_path.exists() {
        eprintln!(
            "SKIP: no reference data. Generate it with:\n\n    \
             python3 xtask/golden/dump_reference.py ip_adapter --output tests/golden\n"
        );
        return;
    }
    let refs = sd_tensor::safetensors::load(&refs_path, &dev).expect("reference");
    let vb = sd_loader::safetensors_var_builder(&[&ip_path], DType::F32, &dev).expect("weights");
    let proj = ImageProjModel::new(1024, 768, NUM_TOKENS, vb.pp("image_proj")).expect("builds");

    // The reference carries a [batch, images, embed] embedding; one image.
    let embeds = refs["image_embeds"]
        .squeeze(1)
        .expect("drop the image axis");
    let tokens = proj.forward(&embeds).expect("project");
    let want = refs["image_tokens"]
        .squeeze(1)
        .expect("drop the image axis");
    assert_eq!(tokens.dims(), want.dims());
    let excess = testing::allclose_excess(&tokens, &want, RTOL).expect("compare");
    assert!(excess <= ATOL, "image tokens: excess {excess:.3e}");
    println!("image tokens excess {excess:.3e}");
}

#[test]
fn a_controlled_unet_matches_diffusers() {
    let dev = Device::Cpu;
    let refs_path = golden_dir().join("reference.safetensors");
    let ip_path = golden_dir().join("ip-adapter_sd15.safetensors");
    if !refs_path.exists() || !ip_path.exists() || !unet_weights().exists() {
        eprintln!("SKIP a_controlled_unet_matches_diffusers: missing reference or UNet weights");
        return;
    }
    let refs = sd_tensor::safetensors::load(&refs_path, &dev).expect("reference");
    let ip_vb = sd_loader::safetensors_var_builder(&[&ip_path], DType::F32, &dev).expect("ip");
    let vb =
        sd_loader::safetensors_var_builder(&[&unet_weights()], DType::F32, &dev).expect("unet");
    let unet = UNet2DConditionModel::new_with_ip(
        &UNetConfig::sd15(),
        vb,
        ip_vb.pp("ip_adapter"),
        NUM_TOKENS,
    )
    .expect("building a UNet with IP-Adapter");

    // The image tokens ride on the end of the context; the attention layers
    // split them off. That convention is what let this reach sixteen layers
    // without a new parameter on every block type.
    let tokens = refs["image_tokens"]
        .squeeze(1)
        .expect("drop the image axis");
    let context = Tensor::cat(&[&refs["text"], &tokens], 1).expect("concat");

    let out = unet
        .forward(&refs["sample"], &refs["timestep"], &context)
        .expect("forward");
    let excess = testing::allclose_excess(&out, &refs["output"], RTOL).expect("compare");
    assert!(excess <= ATOL, "output: excess {excess:.3e}");
    println!("ip-adapter unet excess {excess:.3e}");
}

#[test]
fn scale_zero_is_exactly_the_uncontrolled_unet() {
    // The property that makes the strength safe to expose, and a second,
    // independent check on the wiring: at 0 the image path contributes
    // nothing, so the result must match diffusers' own scale-0 output.
    let dev = Device::Cpu;
    let refs_path = golden_dir().join("reference.safetensors");
    let ip_path = golden_dir().join("ip-adapter_sd15.safetensors");
    if !refs_path.exists() || !ip_path.exists() || !unet_weights().exists() {
        eprintln!("SKIP scale_zero_is_exactly_the_uncontrolled_unet");
        return;
    }
    let refs = sd_tensor::safetensors::load(&refs_path, &dev).expect("reference");
    let ip_vb = sd_loader::safetensors_var_builder(&[&ip_path], DType::F32, &dev).expect("ip");
    let vb =
        sd_loader::safetensors_var_builder(&[&unet_weights()], DType::F32, &dev).expect("unet");
    let unet = UNet2DConditionModel::new_with_ip(
        &UNetConfig::sd15(),
        vb,
        ip_vb.pp("ip_adapter"),
        NUM_TOKENS,
    )
    .expect("building");

    let tokens = refs["image_tokens"]
        .squeeze(1)
        .expect("drop the image axis");
    let context = Tensor::cat(&[&refs["text"], &tokens], 1).expect("concat");

    let _scale = ip::with_scale(0.0);
    let out = unet
        .forward(&refs["sample"], &refs["timestep"], &context)
        .expect("forward");
    let excess = testing::allclose_excess(&out, &refs["output_scale0"], RTOL).expect("compare");
    assert!(excess <= ATOL, "scale 0: excess {excess:.3e}");
    println!("scale-0 excess {excess:.3e}");
}
