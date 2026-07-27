//! Golden verification for AnimateDiff motion modules.
//!
//! Compared in isolation, because the thing that goes wrong is the permute
//! that makes attention temporal — and it produces correct shapes either way.
//! A module that mixes across pixels instead of frames runs cleanly and looks
//! like a weak motion module rather than a broken one, so only a numeric
//! comparison at this level catches it.

use std::path::PathBuf;

use sd_models::unet::motion::MotionModule;
use sd_tensor::{testing, DType, Device, Tensor};

/// Two attentions and a feed-forward on top of the UNet's own activations.
/// Same bound and same reasoning as `golden_unet.rs`.
const RTOL: f64 = 1e-3;
const ATOL: f64 = 1e-3;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/motion")
}

#[test]
fn matches_the_reference_motion_module() {
    let dev = Device::Cpu;
    let refs_path = golden_dir().join("reference.safetensors");
    let weights = golden_dir().join("motion_adapter.safetensors");
    if !refs_path.exists() || !weights.exists() {
        eprintln!(
            "SKIP: no reference data. Generate it with:\n\n    \
             python3 xtask/golden/dump_reference.py motion --output tests/golden\n"
        );
        return;
    }
    let refs = sd_tensor::safetensors::load(&refs_path, &dev).expect("reference");
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev).expect("weights");
    let module = MotionModule::new(320, 1, vb.pp("down_blocks.0.motion_modules.0"))
        .expect("building a motion module");

    // Four frames of one image: the batch axis *is* the frame axis.
    let out = module.forward(&refs["hidden"], 4).expect("forward");
    assert_eq!(out.dims(), refs["output"].dims());
    let excess = testing::allclose_excess(&out, &refs["output"], RTOL).expect("compare");
    assert!(excess <= ATOL, "motion module: excess {excess:.3e}");
    println!("motion module excess {excess:.3e}");
}

#[test]
fn a_change_in_one_frame_reaches_the_others() {
    // The property that makes it a *motion* module rather than a per-frame
    // one: information crosses the frame axis.
    //
    // An earlier version of this test also asserted that a change does *not*
    // cross the pixel axis, and that is false — the module opens with a
    // GroupNorm, which normalises over all of a frame's pixels, so one changed
    // pixel shifts every other one in that frame by construction. The
    // reference comparison above is what pins the permute; this pins that
    // frames are coupled at all.
    let dev = Device::Cpu;
    let weights = golden_dir().join("motion_adapter.safetensors");
    if !weights.exists() {
        eprintln!("SKIP a_change_in_one_frame_reaches_the_others");
        return;
    }
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev).expect("weights");
    let module =
        MotionModule::new(320, 1, vb.pp("down_blocks.0.motion_modules.0")).expect("building");

    let frames = 4usize;
    let base = Tensor::zeros((frames, 320, 4, 4), DType::F32, &dev).unwrap();
    let mut data = vec![0f32; frames * 320 * 16];
    for c in 0..320 {
        data[c * 16] = 5.0; // frame 0, pixel (0, 0), every channel
    }
    let poked = Tensor::from_vec(data, (frames, 320, 4, 4), &dev).unwrap();

    let a = module.forward(&base, frames).expect("base");
    let b = module.forward(&poked, frames).expect("poked");
    let diff = (b - a).unwrap().abs().unwrap();

    let frame_max = |f: usize| {
        diff.narrow(0, f, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .fold(0f32, |m, v| m.max(*v))
    };
    assert!(
        frame_max(2) > 1e-3,
        "a change in frame 0 did not reach frame 2; is the attention temporal?"
    );
}
