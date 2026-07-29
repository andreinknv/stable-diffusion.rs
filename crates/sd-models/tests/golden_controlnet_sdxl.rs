//! Golden verification for an SDXL ControlNet.
//!
//! The roadmap said this would be "a config and a checkpoint, not new code".
//! It is not: `ControlNet::new` takes a `UNetConfig` but built only a
//! `TimestepEmbedding` from it, ignoring `cfg.addition`, and `forward` had
//! nowhere to put a pooled embedding or time ids. **An SDXL ControlNet is
//! `addition_embed_type: "text_time"`** and is conditioned on both, exactly as
//! the SDXL UNet is.
//!
//! That is the thing this file exists to check, because it is silent when
//! wrong: without the micro-conditioning the ControlNet still emits nine
//! corrections of exactly the right shapes, computed at a timestep embedding
//! that means something else. The image is merely wrong.
//!
//! Compared correction by correction, for the same reason the SD 1.5
//! reference is: a ControlNet has no image of its own, so those ten tensors
//! *are* its whole observable behaviour, and the index of the first bad one
//! localises the fault.

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::controlnet::ControlNet;
use sd_models::unet::UNetConfig;
use sd_tensor::{testing, DType, Device, Tensor};

/// Same bound, and the same reason, as the SD 1.5 ControlNet reference: these
/// activations are far from order-1, so a purely absolute tolerance would ask
/// for more significant digits than f32 has.
const RTOL: f64 = 1e-3;
const ATOL: f64 = 1e-3;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/controlnet_sdxl")
}

const REGENERATE: &str = "SKIP: no SDXL ControlNet reference. Generate it with:\n\n    \
     python3 xtask/golden/dump_reference.py controlnet_sdxl --output tests/golden\n";

fn fixtures(dev: &Device) -> Option<(HashMap<String, Tensor>, ControlNet)> {
    let refs_path = golden_dir().join("reference.safetensors");
    let weights = golden_dir().join("controlnet.safetensors");
    if !refs_path.exists() || !weights.exists() {
        sd_tensor::skip_missing_fixture!("{REGENERATE}");
        return None;
    }
    let refs = sd_tensor::safetensors::load(&refs_path, dev).expect("reference");
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, dev).expect("weights");
    let net = ControlNet::new(&UNetConfig::sdxl(), vb).expect("building an SDXL ControlNet");
    Some((refs, net))
}

#[test]
fn an_sdxl_controlnet_matches_diffusers() {
    let dev = Device::Cpu;
    let Some((refs, net)) = fixtures(&dev) else {
        return;
    };
    assert!(
        net.takes_micro_conditioning(),
        "an SDXL ControlNet must carry an add_embedding"
    );

    let control = net
        .forward_sdxl(
            &refs["sample"],
            &refs["timestep"],
            &refs["context"],
            &refs["hint"],
            1.0,
            &refs["pooled"],
            &refs["time_ids"],
        )
        .expect("forward");

    // SDXL's skip stack is nine, not SD 1.5's twelve: three blocks rather than
    // four, so one fewer resnet pair and one fewer downsampler.
    assert_eq!(control.down.len(), 9, "SDXL has nine skips");
    for (i, got) in control.down.iter().enumerate() {
        let want = &refs[&format!("down_{i:02}")];
        assert_eq!(got.dims(), want.dims(), "correction {i} shape");
        let excess = testing::allclose_excess(got, want, RTOL).expect("compare");
        assert!(excess <= ATOL, "correction {i}: excess {excess:.3e}");
    }
    let excess = testing::allclose_excess(&control.mid, &refs["mid"], RTOL).expect("compare");
    assert!(excess <= ATOL, "mid correction: excess {excess:.3e}");
    println!("sdxl controlnet: 9 corrections plus mid, worst within {ATOL:.0e}");
}

#[test]
fn the_micro_conditioning_is_load_bearing() {
    // The failure this file exists for. Without the pooled embedding and time
    // ids, every shape is still right and every correction is still produced —
    // so the only thing that catches it is a number.
    let dev = Device::Cpu;
    let Some((refs, net)) = fixtures(&dev) else {
        return;
    };

    let err = net
        .forward(
            &refs["sample"],
            &refs["timestep"],
            &refs["context"],
            &refs["hint"],
            1.0,
        )
        .expect_err("an SDXL ControlNet must refuse a plain forward");
    assert!(
        err.to_string().contains("forward_sdxl"),
        "the refusal should name the fix, got: {err}"
    );

    // And it must genuinely change the answer, or refusing it is theatre.
    let zeros_pooled = refs["pooled"].zeros_like().expect("zeros");
    let other = net
        .forward_sdxl(
            &refs["sample"],
            &refs["timestep"],
            &refs["context"],
            &refs["hint"],
            1.0,
            &zeros_pooled,
            &refs["time_ids"],
        )
        .expect("forward");
    let moved = testing::max_abs_diff(&other.mid, &refs["mid"]).expect("diff");
    assert!(
        moved > 1e-3,
        "zeroing the pooled embedding changed the corrections by only {moved:.3e}"
    );
}
