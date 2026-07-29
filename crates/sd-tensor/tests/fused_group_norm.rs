//! Our group norm must agree with candle's, which is what every checkpoint in
//! this project was verified against.
//!
//! It is the riskiest of the three kernels: it does its own reduction *and*
//! applies a per-channel affine, so it can be wrong in two independent ways,
//! and both produce a finite tensor of the right shape.

use sd_tensor::{fused, nn, DType, Device, Module, Tensor};

fn pair(ch: usize, s: usize, groups: usize, dev: &Device) -> (Tensor, Tensor, Tensor) {
    let mut rng = sd_tensor::rng::SeededRng::new(29);
    let xs = rng.randn((2, ch, s, s), dev).expect("input");
    let w = rng.randn((ch,), dev).expect("weight");
    let b = rng.randn((ch,), dev).expect("bias");
    let _ = groups;
    (xs, w, b)
}

#[test]
fn it_agrees_with_candle_at_every_shape_sd15_runs() {
    let dev = sd_tensor::device::best().expect("device");
    // The inventory extracted from the SD 1.5 checkpoint, deduplicated.
    for (ch, s) in [
        (320usize, 64usize),
        (640, 32),
        (1280, 16),
        (1280, 8),
        (2560, 8),
        (1920, 16),
        (960, 64),
        (640, 64),
    ] {
        let groups = 32.min(ch);
        let (xs, w, b) = pair(ch, s, groups, &dev);
        let ours = nn::GroupNorm::new(w.clone(), b.clone(), ch, groups, 1e-5).expect("ours");
        let theirs = fused::candle_group_norm(&w, &b, ch, groups, 1e-5).expect("candle");

        let got = ours.forward(&xs).expect("ours");
        let want = theirs.forward(&xs).expect("candle");
        assert_eq!(got.dims(), want.dims(), "shape at [2,{ch},{s},{s}]");
        let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-4).expect("compare");
        assert!(
            excess <= 1e-4,
            "[2,{ch},{s},{s}]: excess {excess:.3e} over candle"
        );
    }
}

#[test]
fn each_group_is_normalised_independently() {
    // The failure that a whole-tensor normalisation would produce: right
    // shape, right magnitude, wrong statistics per group. Checked by giving
    // each group a wildly different scale and confirming the output does not
    // remember it.
    let dev = sd_tensor::device::best().expect("device");
    let (ch, s, groups) = (64usize, 4usize, 4usize);
    let cpg = ch / groups;
    let mut data = Vec::with_capacity(ch * s * s);
    for c in 0..ch {
        // Group 0 lives near 1, group 1 near 100, and so on.
        let scale = 10f32.powi((c / cpg) as i32);
        for j in 0..s * s {
            data.push(scale * (1.0 + j as f32 * 0.01));
        }
    }
    let xs = Tensor::from_vec(data, (1, ch, s, s), &dev).expect("input");
    let w = Tensor::ones((ch,), DType::F32, &dev).expect("weight");
    let b = Tensor::zeros((ch,), DType::F32, &dev).expect("bias");

    let got = nn::GroupNorm::new(w.clone(), b.clone(), ch, groups, 1e-5)
        .expect("ours")
        .forward(&xs)
        .expect("forward");
    let want = fused::candle_group_norm(&w, &b, ch, groups, 1e-5)
        .expect("candle")
        .forward(&xs)
        .expect("forward");
    let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-3).expect("compare");
    assert!(
        excess <= 1e-3,
        "groups spanning four orders of magnitude: excess {excess:.3e}"
    );

    // Every group must come out with unit-ish spread despite the input scales.
    let flat = got.flatten_all().expect("flat").to_vec1::<f32>().expect("vec");
    let per = cpg * s * s;
    for g in 0..groups {
        let slice = &flat[g * per..(g + 1) * per];
        let mean = slice.iter().sum::<f32>() / per as f32;
        assert!(mean.abs() < 1e-3, "group {g} mean is {mean:.3e}");
    }
}

#[test]
fn the_affine_is_per_channel_not_per_group() {
    // Applying `weight` per group rather than per channel is a real hazard
    // here — the kernel loops channels inside a group row — and it produces a
    // plausible tensor. Pinned with a weight that differs within a group.
    let dev = sd_tensor::device::best().expect("device");
    let (ch, s, groups) = (8usize, 2usize, 2usize);
    let mut rng = sd_tensor::rng::SeededRng::new(31);
    let xs = rng.randn((1, ch, s, s), &dev).expect("input");
    let w = Tensor::from_vec(
        (0..ch).map(|c| (c + 1) as f32).collect::<Vec<_>>(),
        (ch,),
        &dev,
    )
    .expect("weight");
    let b = Tensor::zeros((ch,), DType::F32, &dev).expect("bias");

    let got = nn::GroupNorm::new(w.clone(), b.clone(), ch, groups, 1e-5)
        .expect("ours")
        .forward(&xs)
        .expect("forward");
    let want = fused::candle_group_norm(&w, &b, ch, groups, 1e-5)
        .expect("candle")
        .forward(&xs)
        .expect("forward");
    let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-4).expect("compare");
    assert!(excess <= 1e-4, "per-channel affine: excess {excess:.3e}");
}

#[test]
fn a_constant_row_does_not_produce_nan() {
    // Zero variance is reachable — a dead channel after a ReLU-like path — and
    // the shifted accumulator can land a hair below zero, so the kernel clamps
    // before the reciprocal square root. Without that this is NaN, which then
    // spreads through the whole image.
    let dev = sd_tensor::device::best().expect("device");
    let (ch, s, groups) = (32usize, 4usize, 4usize);
    let xs = Tensor::ones((1, ch, s, s), DType::F32, &dev).expect("constant input");
    let w = Tensor::ones((ch,), DType::F32, &dev).expect("weight");
    let b = Tensor::zeros((ch,), DType::F32, &dev).expect("bias");

    let got = nn::GroupNorm::new(w, b, ch, groups, 1e-5)
        .expect("ours")
        .forward(&xs)
        .expect("forward")
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec");
    assert!(
        got.iter().all(|v| v.is_finite()),
        "a constant input produced non-finite output"
    );
    assert!(
        got.iter().all(|v| v.abs() < 1e-2),
        "a constant input should normalise to zero"
    );
}

#[test]
fn groups_that_do_not_divide_the_channels_are_refused() {
    let dev = Device::Cpu;
    let mut rng = sd_tensor::rng::SeededRng::new(37);
    let xs = rng.randn((1, 30, 2, 2), &dev).expect("input");
    let w = Tensor::ones((30usize,), DType::F32, &dev).expect("weight");
    let b = Tensor::zeros((30usize,), DType::F32, &dev).expect("bias");
    assert!(
        xs.apply_op3_no_bwd(&w, &b, &fused::GroupNormOp { groups: 32, eps: 1e-5 })
            .is_err(),
        "32 groups do not divide 30 channels and must be refused"
    );
}
