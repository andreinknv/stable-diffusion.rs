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

/// Tolerance for the UNet's intermediate tensors, as `atol + rtol*|want|`.
///
/// **Relative, not absolute, and the difference is not cosmetic.** These
/// activations peak between 2.7 and 26.6, so the project-wide
/// `DEFAULT_ATOL = 1e-4` asked for agreement to 4e-6 relative on the largest
/// of them — *tighter than float32 can deliver*, which makes it a test of
/// summation order rather than of this port.
///
/// That is measured, not argued. `xtask/golden/reference_precision.py unet`
/// runs the diffusers UNet against **itself** in f64, same weights, same
/// inputs, so neither run has a bug and the gap between them is float32's own
/// noise floor:
///
/// ```text
///   tensor        peak      max_abs     max_rel
///   mid_output   16.169    1.108e-4    6.850e-6
///   down_11      19.219    1.083e-4    5.636e-6
///   down_09      26.601    9.991e-5    3.756e-6
///   output        3.889    9.700e-6    2.494e-6
///   worst across all captured tensors: 1.108e-4 absolute, 6.850e-6 relative
/// ```
///
/// So `mid_output` could never be pinned to 1e-4 absolutely: **diffusers
/// misses that bound against its own f64 by 1.108e-4.** The old bound passed
/// only because candle's summation order happened to sit near PyTorch's, and
/// it failed the moment Apple's Accelerate reordered it — at 1.087e-4, which
/// is *closer to the reference than the reference's own f32 is*.
///
/// Both halves of the bound are needed. `allclose_excess` applies the
/// `rtol*|want|` term and returns the remainder; the assertion supplies
/// `atol`. A relative term alone allows nothing where `want` is near zero, and
/// these tensors have such elements — which is why every skip reports a small
/// non-zero excess. An absolute term alone is the bound that just failed.
///
/// `rtol = 1e-3` is 146x the measured relative floor and the value this
/// project already documents for f32. `atol = 1e-3` is 9x the measured
/// absolute floor. Real porting bugs are nowhere near either: the VAE's
/// asymmetric-padding bug showed 17.32, and the largest excess seen here with
/// a correct port is 4.1e-5.
const UNET_RTOL: f64 = 1e-3;
const UNET_ATOL: f64 = 1e-3;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unet_full")
}

fn refs() -> Option<HashMap<String, Tensor>> {
    let path = golden_dir().join("reference.safetensors");
    if !path.exists() {
        sd_tensor::skip_missing_fixture!(
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
        sd_tensor::skip_missing_fixture!("SKIP: no unet.safetensors");
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
        class_projection: None,
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
        let excess = testing::allclose_excess(got, want, UNET_RTOL).expect("comparing");
        eprintln!("{name}: {c}, excess {excess:.3e}");
        if excess > UNET_ATOL && first_bad.is_none() {
            first_bad = Some((i, c.max_abs, excess));
        }
    }
    if let Some((i, max_abs, excess)) = first_bad {
        panic!(
            "first bad skip is index {i} (max_abs={max_abs:.3e}, {excess:.3e} beyond \
             atol={UNET_ATOL:.0e} + rtol={UNET_RTOL:.0e}). \
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
    let excess = testing::allclose_excess(&mid, want, UNET_RTOL).expect("comparing");
    eprintln!("mid_output: {c}, excess {excess:.3e}");
    assert!(
        excess <= UNET_ATOL,
        "mid block diverged by {excess:.3e} beyond atol={UNET_ATOL:.0e} + \
         rtol={UNET_RTOL:.0e}\n  {c}\n\
         Hint: check axis order and parameter naming before suspecting the kernel."
    );
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

/// The UNet loaded from a real quantised LDM checkpoint, run against the same
/// reference the safetensors path is held to.
///
/// This is what proves the 686-tensor name map. Totality and injectivity say
/// every key translated to *something* unique; only running the model says
/// they translated to the *right* things — a plausible-but-wrong mapping
/// loads without complaint and denoises toward nothing.
#[test]
fn a_quantised_ldm_unet_runs_through_the_name_map() {
    let Some(refs) = refs() else { return };
    let gguf =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/gguf/sd15-q4_0.gguf");
    if !gguf.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no sd15-q4_0.gguf");
        return;
    }

    let dev = Device::Cpu;
    let vb = sd_loader::unet_var_builder_from_gguf(&gguf, DType::F32, &dev)
        .expect("loading the UNet from a GGUF checkpoint");
    let unet = UNet2DConditionModel::new(&UNetConfig::sd15(), vb)
        .expect("every mapped name must be one the UNet asks for");

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

    // An absolute threshold would be arbitrary here: 4-bit weights make some
    // error certain, and "how much is too much" has no principled value.
    // Correlation does have one. A correct mapping predicts the same noise
    // field slightly imprecisely; a wrong mapping predicts a different field
    // entirely, and lands near zero however small its magnitude happens to
    // be. That is the property worth asserting.
    let a = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let cov: f32 = a.iter().zip(&b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f32 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f32 = b.iter().map(|y| (y - mb).powi(2)).sum();
    let corr = cov / (va.sqrt() * vb.sqrt());
    eprintln!("gguf Q4_0 unet vs f32 reference: {c}, correlation {corr:.4}");

    assert!(
        corr > 0.97,
        "the prediction is not the reference's: correlation {corr:.4} ({c})"
    );
}

// -- SD 2.x ---------------------------------------------------------------

fn sd2_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unet_full_cross1024")
}

#[test]
fn config_sd2_is_sd15_geometry_with_a_wider_text_encoder() {
    let cfg = UNetConfig::sd2();
    // The block geometry is SD 1.5's exactly.
    assert_eq!(
        cfg.block_out_channels,
        UNetConfig::sd15().block_out_channels
    );
    assert_eq!(cfg.layers_per_block, 2);
    assert_eq!(cfg.down_block_has_attention, vec![true, true, true, false]);
    // What differs: a 1024-wide text encoder, SDXL-style head counts (all 64
    // wide), and Linear rather than 1x1-conv projections.
    assert_eq!(cfg.cross_attention_dim, 1024);
    assert_eq!(cfg.attention_head_dim, vec![5, 10, 20, 20]);
    for (c, h) in cfg.block_out_channels.iter().zip(&cfg.attention_head_dim) {
        assert_eq!(c / h, 64, "every head is 64 wide");
    }
    assert!(cfg.use_linear_projection);
    // No micro-conditioning: that is SDXL's alone.
    assert!(cfg.addition.is_none());
    // The skip stack is unchanged, so the up blocks are too.
    assert_eq!(cfg.skip_channels(), UNetConfig::sd15().skip_channels());
}

#[test]
fn sd2_matches_diffusers_skip_for_skip() {
    let dev = Device::Cpu;
    let path = sd2_dir().join("reference.safetensors");
    if !path.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no SD 2.x reference. Generate it with:\n\n    \
             python3 xtask/golden/dump_reference.py unet_full \
             --model-id friedrichor/stable-diffusion-2-1-realistic --output tests/golden\n"
        );
        return;
    }
    let refs = sd_tensor::safetensors::load(&path, &dev).expect("loading reference");
    let weights = sd2_dir().join("unet.safetensors");
    if !weights.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no SD 2.x unet.safetensors");
        return;
    }
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev).expect("weights");
    let unet = UNet2DConditionModel::new(&UNetConfig::sd2(), vb).expect("building SD 2 UNet");

    let (out, skips, mid) = unet
        .forward_with_skips(&refs["sample"], &refs["timestep"], &refs["context"], None)
        .expect("forward");

    for (i, got) in skips.iter().enumerate() {
        let key = format!("down_{i:02}");
        let excess = testing::allclose_excess(got, &refs[&key], UNET_RTOL).expect("compare");
        assert!(excess <= UNET_ATOL, "{key}: excess {excess:.3e}");
    }
    let excess = testing::allclose_excess(&mid, &refs["mid_output"], UNET_RTOL).expect("compare");
    assert!(excess <= UNET_ATOL, "mid_output: excess {excess:.3e}");
    let excess = testing::allclose_excess(&out, &refs["output"], UNET_RTOL).expect("compare");
    assert!(excess <= UNET_ATOL, "output: excess {excess:.3e}");
    println!("sd2 output excess {excess:.3e}");
}
