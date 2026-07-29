//! The adaptive layer norm kernel must agree with the four ops it replaced.
//!
//! This one carries more risk than the GEGLU kernel, because it does its own
//! reduction. A wrong GEGLU is wrong pointwise and obvious; a wrong reduction
//! is a slightly wrong mean, which shifts every value in the row by a little
//! and looks exactly like a plausible image.

use sd_tensor::{fused, ops, DType, Device, Tensor, D};

/// The composition, kept here so the tests are pinned to the arithmetic
/// rather than to whatever `fused::ada_layer_norm` currently does.
fn composed(xs: &Tensor, scale: &Tensor, shift: &Tensor, eps: f64) -> sd_tensor::Result<Tensor> {
    ops::plain_layer_norm(xs, eps)?
        .broadcast_mul(&(scale + 1.0)?)?
        .broadcast_add(shift)
}

/// Conditioning as the models actually build it: one projection, narrowed.
/// At batch 1 that narrow is still contiguous; at batch 2 it is not, and the
/// wrapper has to copy it. Both paths matter.
fn conditioning(
    rng: &mut sd_tensor::rng::SeededRng,
    batch: usize,
    width: usize,
    dev: &Device,
) -> (Tensor, Tensor) {
    let proj = rng.randn((batch, 1, width * 6), dev).expect("projection");
    let shift = proj.narrow(D::Minus1, 0, width).expect("shift");
    let scale = proj.narrow(D::Minus1, width, width).expect("scale");
    (scale, shift)
}

#[test]
fn the_kernel_agrees_with_the_composition() {
    let dev = sd_tensor::device::best().expect("device");
    // Flux-dev at 1024: a 4096-token image stream, a 512-token text stream,
    // and the two concatenated. Batch 2 for the strided conditioning path.
    for (batch, tokens, width) in [
        (1usize, 4096usize, 3072usize),
        (1, 512, 3072),
        (1, 4608, 3072),
        (2, 256, 3072),
        (2, 77, 1536),
    ] {
        let mut rng = sd_tensor::rng::SeededRng::new(11);
        let xs = rng.randn((batch, tokens, width), &dev).expect("input");
        let (scale, shift) = conditioning(&mut rng, batch, width, &dev);

        let want = composed(&xs, &scale, &shift, 1e-6).expect("composition");
        let got = fused::ada_layer_norm(&xs, &scale, &shift, 1e-6).expect("kernel");
        assert_eq!(got.dims(), want.dims(), "shape at {batch}x{tokens}x{width}");

        let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-5).expect("compare");
        assert!(
            excess <= 1e-5,
            "{batch}x{tokens}x{width}: excess {excess:.3e} over the composition"
        );
    }
}

#[test]
fn the_normalisation_is_real() {
    // Before the modulation is applied, each row must have zero mean and unit
    // variance. A kernel that skipped the reduction entirely — returning the
    // input scaled and shifted — would still pass a shape check and produce a
    // plausible-looking tensor, so this pins the part that is easy to lose.
    let dev = sd_tensor::device::best().expect("device");
    let mut rng = sd_tensor::rng::SeededRng::new(13);
    let width = 3072;
    let xs = rng.randn((1, 64, width), &dev).expect("input");
    // Unit scale and zero shift make the output the normalisation alone.
    let scale = Tensor::zeros((1, 1, width), DType::F32, &dev).expect("scale");
    let shift = Tensor::zeros((1, 1, width), DType::F32, &dev).expect("shift");

    let got = fused::ada_layer_norm(&xs, &scale, &shift, 1e-6).expect("kernel");
    let mean = got.mean_keepdim(D::Minus1).expect("mean");
    let worst_mean = mean.abs().expect("abs").max_all().expect("max");
    let worst_mean = worst_mean.to_scalar::<f32>().expect("scalar");
    assert!(
        worst_mean < 1e-5,
        "row means are {worst_mean:.3e}, not zero"
    );

    let var = got
        .sqr()
        .expect("sqr")
        .mean_keepdim(D::Minus1)
        .expect("var");
    let spread = (var - 1.0)
        .expect("centre")
        .abs()
        .expect("abs")
        .max_all()
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar");
    assert!(spread < 1e-3, "row variances are off by {spread:.3e}");
}

#[test]
fn the_one_plus_on_the_scale_is_not_dropped() {
    // `(1 + scale)` rather than `scale`: with a zero scale the modulation must
    // be the identity, which is what makes an untrained block a no-op and what
    // published weights assume. Dropping the `1 +` zeroes the whole tensor.
    let dev = sd_tensor::device::best().expect("device");
    let mut rng = sd_tensor::rng::SeededRng::new(17);
    let width = 128;
    let xs = rng.randn((1, 8, width), &dev).expect("input");
    let scale = Tensor::zeros((1, 1, width), DType::F32, &dev).expect("scale");
    let shift = Tensor::zeros((1, 1, width), DType::F32, &dev).expect("shift");

    let got = fused::ada_layer_norm(&xs, &scale, &shift, 1e-6).expect("kernel");
    let plain = ops::plain_layer_norm(&xs, 1e-6).expect("norm");
    let excess = sd_tensor::testing::allclose_excess(&got, &plain, 1e-5).expect("compare");
    assert!(
        excess <= 1e-5,
        "a zero scale should leave the normalisation untouched, excess {excess:.3e}"
    );
}

#[test]
fn the_scale_and_the_shift_are_not_swapped() {
    // Silent if wrong: both are [b, 1, width] and swapping them produces a
    // finite tensor of the right shape. Distinguished with a zero scale, which
    // makes the output `norm(x) + shift` — and a zero *shift* instead would
    // make it `norm(x) * (1 + scale)`, a different tensor.
    let dev = Device::Cpu;
    let width = 4;
    let xs = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, width), &dev).expect("input");
    let scale = Tensor::zeros((1, 1, width), DType::F32, &dev).expect("scale");
    let shift =
        Tensor::from_vec(vec![10.0f32, 10.0, 10.0, 10.0], (1, 1, width), &dev).expect("shift");

    let got = xs
        .apply_op3_no_bwd(&scale, &shift, &fused::AdaLayerNorm { eps: 1e-6 })
        .expect("kernel")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    // norm([1,2,3,4]) is symmetric about zero, so + 10 must average to 10.
    let mean = got.iter().sum::<f32>() / width as f32;
    assert!(
        (mean - 10.0).abs() < 1e-4,
        "shift did not land: mean {mean}, expected 10"
    );
    assert!(got[0] < 10.0 && got[3] > 10.0, "the row is not normalised");
}

#[test]
fn a_row_too_wide_for_threadgroup_memory_falls_back() {
    // The kernel holds a whole row in threadgroup memory, which is 32 KB on
    // Apple GPUs. Past that it must decline and let the composition answer,
    // rather than dispatch something that cannot run.
    let dev = sd_tensor::device::best().expect("device");
    let width = fused::AdaLayerNorm::MAX_WIDTH + 512;
    let mut rng = sd_tensor::rng::SeededRng::new(19);
    let xs = rng.randn((1, 4, width), &dev).expect("input");
    let scale = Tensor::zeros((1, 1, width), DType::F32, &dev).expect("scale");
    let shift = Tensor::zeros((1, 1, width), DType::F32, &dev).expect("shift");

    let got = fused::ada_layer_norm(&xs, &scale, &shift, 1e-6).expect("fallback");
    let want = composed(&xs, &scale, &shift, 1e-6).expect("composition");
    let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-5).expect("compare");
    assert!(excess <= 1e-5, "wide fallback: excess {excess:.3e}");
}

#[test]
fn a_mismatched_conditioning_shape_is_refused() {
    // The kernel indexes conditioning as [b, 1, width]. Anything else would be
    // read at the wrong offsets, which is silent — so it must error instead.
    let dev = Device::Cpu;
    let mut rng = sd_tensor::rng::SeededRng::new(23);
    let xs = rng.randn((2, 4, 64), &dev).expect("input");
    let wrong = rng.randn((2, 4, 64), &dev).expect("wrong shape");
    let ok = Tensor::zeros((2, 1, 64), DType::F32, &dev).expect("shift");
    assert!(
        xs.apply_op3_no_bwd(&wrong, &ok, &fused::AdaLayerNorm { eps: 1e-6 })
            .is_err(),
        "a [b, tokens, width] scale should be refused, not broadcast"
    );
}
