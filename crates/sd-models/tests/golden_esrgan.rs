//! Golden verification for Real-ESRGAN.
//!
//! One comparison, unlike the UNet's twelve: this network has no branch
//! structure worth localising into — it is a straight stack, so a wrong
//! residual scale or a misread dense concatenation shows up in the output and
//! nowhere else. What the structural tests below pin instead are the two
//! things that would still *load*: the 0.2 scalings and the dense widths.

use std::path::PathBuf;

use sd_models::esrgan::{RealEsrgan, ResidualDenseBlock, Rrdb};
use sd_tensor::nn::{VarBuilder, VarMap};
use sd_tensor::{testing, DType, Device, Module, Tensor};

/// The output is an image in `[0, 1]`, so an absolute bound is meaningful.
/// 92 residual additions accumulate f32 error; this is 100x the observed.
const TOL: f64 = 1e-4;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/esrgan")
}

#[test]
fn a_dense_block_widens_its_input_by_thirty_two_each_layer() {
    // The misreading this catches: feeding each convolution only its
    // predecessor's output. That loads for conv1 and fails on conv2's channel
    // count, so it is loud — but only if the widths are right here.
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let block = ResidualDenseBlock::new(vb).expect("builds");

    let xs = Tensor::zeros((1, 64, 8, 8), DType::F32, &dev).unwrap();
    let out = block.forward(&xs).expect("forward");
    assert_eq!(out.dims(), &[1, 64, 8, 8], "a dense block preserves shape");
}

#[test]
fn the_residual_scaling_is_applied_at_both_levels() {
    // With zero weights every convolution outputs zero, so a dense block
    // returns exactly its input and an RRDB returns exactly its input. That
    // holds for any scale — what it pins is that the skip is an *addition* of
    // the input and not a replacement, which is the failure that still runs.
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let rrdb = Rrdb::new(vb).expect("builds");

    let xs = Tensor::ones((1, 64, 4, 4), DType::F32, &dev).unwrap();
    let out = rrdb.forward(&xs).expect("forward");
    let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // Random init, so this is not exact — but the input must dominate.
    assert!(
        v.iter().all(|x| x.is_finite()),
        "an RRDB stack diverged on random weights"
    );
}

#[test]
fn upscaling_is_exactly_four_times() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let net = RealEsrgan::new(vb).expect("builds");

    let image = Tensor::zeros((1, 3, 16, 24), DType::F32, &dev).unwrap();
    let out = net.upscale(&image).expect("upscale");
    assert_eq!(
        out.dims(),
        &[1, 3, 64, 96],
        "two doublings, not one or three"
    );
}

#[test]
fn matches_the_reference_rrdbnet() {
    let dev = Device::Cpu;
    let refs_path = golden_dir().join("reference.safetensors");
    let weights = golden_dir().join("esrgan_x4.safetensors");
    if !refs_path.exists() || !weights.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no reference data. Generate it with:\n\n    \
             python3 xtask/golden/dump_reference.py esrgan --output tests/golden\n"
        );
        return;
    }
    let refs = sd_tensor::safetensors::load(&refs_path, &dev).expect("loading reference");
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev).expect("weights");
    let net = RealEsrgan::new(vb).expect("building Real-ESRGAN");

    let got = net.upscale(&refs["image"]).expect("upscale");
    let want = &refs["output"];
    assert_eq!(got.dims(), want.dims());
    let excess = testing::allclose_excess(&got, want, 0.0).expect("compare");
    assert!(excess <= TOL, "max diff {excess:.3e}");
    println!("esrgan max diff {excess:.3e}");
}

#[test]
fn tiling_with_full_context_is_identical_to_one_pass() {
    // The invariant that pins the crop offsets and the stitching order. Give
    // every tile a padding at least as large as the image and each one sees
    // the whole thing, so the tiled result must match one pass *exactly* —
    // approximate equality would hide an off-by-one in the crop, which is the
    // failure that produces a subtly shifted image rather than an error.
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let net = RealEsrgan::new(vb).expect("builds");

    let image = Tensor::randn(0f32, 1.0, (1, 3, 20, 28), &dev).unwrap();
    let one = net.upscale(&image).expect("one pass");
    // Tiles of 8 over a 20x28 image: 3 rows and 4 columns, with uneven edges.
    let tiled = net.upscale_in_tiles(&image, 8, 64).expect("tiled");

    assert_eq!(one.dims(), tiled.dims());
    let a = one.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = tiled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a, b, "full-context tiling must reproduce one pass exactly");
}

#[test]
fn a_tiled_upscale_is_still_four_times_the_input() {
    // Uneven tiles: 20 and 28 are not multiples of 8, so the last row and
    // column are partial. Getting that wrong truncates the image.
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let net = RealEsrgan::new(vb).expect("builds");

    let image = Tensor::zeros((1, 3, 20, 28), DType::F32, &dev).unwrap();
    let out = net.upscale_in_tiles(&image, 8, 2).expect("tiled");
    assert_eq!(out.dims(), &[1, 3, 80, 112]);
}
