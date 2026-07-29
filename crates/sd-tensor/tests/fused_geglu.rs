//! Our GEGLU kernel must agree with the composition it replaced.
//!
//! The kernel runs only on Metal, so on any other machine these fall back to
//! the composition and compare it against itself. That is not a wasted test:
//! the fallback is what CPU users get, and it must stay equivalent too.

use sd_tensor::{fused, ops, DType, Device, Tensor, D};

/// The four ops this replaces, kept here so the test is pinned to the
/// arithmetic rather than to whatever `fused::geglu` currently does.
fn composed(h: &Tensor, inner: usize) -> sd_tensor::Result<Tensor> {
    let hidden = h.narrow(D::Minus1, 0, inner)?;
    let gate = h.narrow(D::Minus1, inner, inner)?;
    hidden * ops::gelu(&gate)?
}

#[test]
fn the_kernel_agrees_with_the_composition() {
    let dev = sd_tensor::device::best().expect("device");
    // The four shapes SD 1.5 runs, batch 2 for guidance.
    for (seq, dim) in [(4096usize, 320usize), (1024, 640), (256, 1280), (64, 1280)] {
        let inner = dim * 4;
        let mut rng = sd_tensor::rng::SeededRng::new(7);
        let h = rng.randn((2, seq, inner * 2), &dev).expect("input");

        let want = composed(&h, inner).expect("composition");
        let got = fused::geglu(&h, inner).expect("kernel");
        assert_eq!(got.dims(), want.dims(), "shape at seq {seq} dim {dim}");

        // 1e-5 relative: the two differ where the erf tail differs, which is
        // by design, and nowhere else by more than f32 rounding.
        let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-5).expect("compare");
        assert!(
            excess <= 1e-5,
            "seq {seq} dim {dim}: excess {excess:.3e} over the composition"
        );
    }
}

#[test]
fn the_gate_and_the_value_are_not_swapped() {
    // The failure this guards is silent: swapping the halves produces a
    // perfectly finite tensor of the right shape and a ruined image. Pinned
    // with a value half of ones, which makes the output the activation alone.
    let dev = Device::Cpu;
    let inner = 4;
    let xs: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, -3.0, -1.0, 0.0, 2.0];
    let h = Tensor::from_vec(xs, (1, inner * 2), &dev).expect("input");
    let got = fused::geglu(&h, inner)
        .expect("kernel")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");

    // gelu(-3) is small and negative, gelu(0) is 0, gelu(2) is close to 2.
    assert!(got[0] < 0.0 && got[0] > -0.02, "gelu(-3) = {}", got[0]);
    assert!(got[2].abs() < 1e-6, "gelu(0) = {}", got[2]);
    assert!(got[3] > 1.9 && got[3] < 2.0, "gelu(2) = {}", got[3]);
    // If the halves were swapped the first output would be 1*gelu(1) = 0.84,
    // which is positive — the opposite sign to the correct answer.
    assert!(got[0] < 0.0, "the halves are swapped");
}

#[test]
fn the_left_tail_does_not_collapse_to_zero() {
    // The reason this GELU is arranged the way it is. Forming `1 + erf(u)` by
    // subtraction rounds the tail away: candle's returns exactly 0.0 for every
    // input below about -6, where the true value is -5.9e-9 and falling. Ours
    // reads erfc off the polynomial before any subtraction, so it does not.
    //
    // Applied directly rather than through `fused::geglu`, because that
    // routes to the composition off Metal — so on CPU it would test candle's
    // arithmetic, which is the arithmetic this is contrasted against. **CPU
    // callers do still get the collapsing tail**; only the Metal path is
    // improved, and closing that would mean replacing a vectorised candle
    // kernel with a scalar loop, which is not obviously a win and has not
    // been measured.
    let dev = Device::Cpu;
    let inner = 4;
    let xs: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, -6.0, -7.0, -8.0, -9.0];
    let h = Tensor::from_vec(xs, (1, inner * 2), &dev).expect("input");
    let got = h
        .apply_op1_no_bwd(&fused::Geglu { inner })
        .expect("kernel")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");

    for (i, x) in [-6.0f32, -7.0, -8.0, -9.0].iter().enumerate() {
        assert!(
            got[i] < 0.0,
            "gelu({x}) came back as {} — the tail cancelled",
            got[i]
        );
    }
    // And the magnitudes must be right, not merely non-zero: truth at -6 is
    // -5.937e-9, and each step left is roughly three orders smaller.
    assert!(
        (got[0] as f64 / -5.937e-9 - 1.0).abs() < 0.01,
        "gelu(-6) = {:.3e}, expected about -5.937e-9",
        got[0]
    );
}

#[test]
fn a_non_contiguous_input_is_refused_rather_than_misread() {
    // The kernel indexes rows by a fixed stride, so a strided input would be
    // read as though it were dense — wrong numbers, no error. Only reachable
    // on Metal, where the kernel is actually used.
    let dev = sd_tensor::device::best().expect("device");
    if !dev.is_metal() {
        eprintln!("SKIP: the kernel only runs on Metal");
        return;
    }
    let mut rng = sd_tensor::rng::SeededRng::new(3);
    let h = rng.randn((2, 8, 64), &dev).expect("input");
    let strided = h.transpose(0, 1).expect("transpose");
    assert!(!strided.is_contiguous(), "the test needs a strided input");

    // `geglu` routes anything non-contiguous to the composition, so it must
    // still produce the right answer rather than an error.
    let got = fused::geglu(&strided, 32).expect("fallback");
    let want = composed(&strided, 32).expect("composition");
    let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-5).expect("compare");
    assert!(excess <= 1e-5, "strided fallback: excess {excess:.3e}");

    // And the kernel itself must decline it rather than read it wrongly.
    let direct = strided.apply_op1_no_bwd(&fused::Geglu { inner: 32 });
    assert!(
        direct.is_err(),
        "the kernel accepted a non-contiguous input instead of refusing it"
    );
}

#[test]
fn only_f32_takes_the_kernel() {
    // Half precision has no kernel yet, and must not silently get the f32 one.
    let dev = sd_tensor::device::best().expect("device");
    let mut rng = sd_tensor::rng::SeededRng::new(5);
    let h = rng
        .randn((1, 8, 64), &dev)
        .expect("input")
        .to_dtype(DType::F16)
        .expect("cast");
    let got = fused::geglu(&h, 32).expect("fallback");
    assert_eq!(got.dtype(), DType::F16, "the fallback changed the dtype");
    let want = composed(&h, 32).expect("composition");
    let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-3).expect("compare");
    assert!(excess <= 1e-3, "f16 fallback: excess {excess:.3e}");
}
