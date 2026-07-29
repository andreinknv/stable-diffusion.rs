//! The norms must keep reducing accurately on CPU.
//!
//! `ops::rms_norm` and `ops::plain_layer_norm` reduce in blocks via
//! `mean_keepdim`. candle ships fused kernels for both, and on Metal those are
//! a tree reduction — as accurate, and 4.5x faster, so that is what the seam
//! uses there. On CPU they sum each row with a sequential `.sum::<f32>()`,
//! whose error grows with row length, and the compositions stay.
//!
//! This file exists because a comment was not enough to prevent that swap
//! being attempted twice. The failure it guards is quiet in the worst way: the
//! fused kernel returns a tensor of the right shape full of plausible numbers,
//! and the only thing downstream that notices is `golden_t5`, three crates
//! away, which needs a T5 checkpoint on disk to run at all. On a machine
//! without fixtures that test skips and the regression ships.
//!
//! So the bound is checked here, against f64, with no checkpoint required.

use sd_tensor::{ops, DType, Device, Tensor, D};

/// The composition lands near 1.3e-7 at every shape tried; candle's sequential
/// CPU sum lands at 8.8e-7. Anything under 3e-7 is the blocked reduction and
/// anything over it is not, which is all this needs to distinguish.
const BOUND: f64 = 3e-7;

fn f64_reference(xs: &Tensor, alpha: &Tensor, eps: f64) -> (Tensor, Tensor) {
    // Longhand, because both helpers cast to f32 internally: calling them at
    // f64 would compare the composition against itself and always pass.
    let x = xs.to_dtype(DType::F64).unwrap();
    let a = alpha.to_dtype(DType::F64).unwrap();
    let rrms = (x.sqr().unwrap().mean_keepdim(D::Minus1).unwrap() + eps)
        .unwrap()
        .sqrt()
        .unwrap();
    let rms = x.broadcast_div(&rrms).unwrap().broadcast_mul(&a).unwrap();
    let mean = x.mean_keepdim(D::Minus1).unwrap();
    let centred = x.broadcast_sub(&mean).unwrap();
    let var = centred.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
    let ln = centred
        .broadcast_div(&(var + eps).unwrap().sqrt().unwrap())
        .unwrap();
    (rms, ln)
}

fn rel_err(got: &Tensor, truth: &Tensor) -> f64 {
    let got = got.to_dtype(DType::F64).unwrap();
    let diff = (&got - truth)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f64>()
        .unwrap();
    let scale = truth
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f64>()
        .unwrap();
    diff / scale
}

#[test]
fn the_cpu_norms_reduce_in_blocks() {
    let dev = Device::Cpu;
    // Row length is the variable a sequential sum is bad at, so the widest row
    // in the project leads: T5-XXL at 4096, then Flux and SD 3.5 at 3072, then
    // CLIP-L at 768.
    for (tokens, width) in [(154usize, 4096usize), (64, 3072), (77, 768)] {
        let mut rng = sd_tensor::rng::SeededRng::new(0);
        let xs = rng.randn((1, tokens, width), &dev).unwrap();
        let alpha = rng.randn((width,), &dev).unwrap();
        let (truth_rms, truth_ln) = f64_reference(&xs, &alpha, 1e-6);

        let rms = rel_err(&ops::rms_norm(&xs, &alpha, 1e-6).unwrap(), &truth_rms);
        let ln = rel_err(&ops::plain_layer_norm(&xs, 1e-6).unwrap(), &truth_ln);
        assert!(
            rms < BOUND,
            "rms_norm at [1,{tokens},{width}]: {rms:.3e} exceeds {BOUND:.0e} — \
             a sequential row sum, not a blocked one"
        );
        assert!(
            ln < BOUND,
            "plain_layer_norm at [1,{tokens},{width}]: {ln:.3e} exceeds {BOUND:.0e} — \
             a sequential row sum, not a blocked one"
        );
    }
}

#[test]
fn the_bound_is_one_the_fused_kernel_fails() {
    // Without this, `BOUND` could be loose enough to admit the very kernel the
    // test above exists to exclude, and the file would assert nothing.
    let dev = Device::Cpu;
    let mut rng = sd_tensor::rng::SeededRng::new(0);
    let xs = rng.randn((1, 154, 4096), &dev).unwrap();
    let alpha = rng.randn((4096,), &dev).unwrap();
    let (truth, _) = f64_reference(&xs, &alpha, 1e-6);

    let fused = rel_err(&ops::fused_rms_norm(&xs, &alpha, 1e-6).unwrap(), &truth);
    assert!(
        fused > BOUND,
        "candle's CPU rms_norm now lands at {fused:.3e}, inside a bound chosen \
         because it did not. Re-measure with `--example norm_accuracy`: if its \
         reduction has changed, the CPU path can use it and this file can go."
    );
}

#[test]
fn half_precision_still_reduces_in_f32() {
    // `t5::RmsNorm` needs this rather than prefers it: at d_model 4096 a bf16
    // sum of squares loses the small terms outright. The dtype guard is what
    // keeps half precision on the composition on every backend, including the
    // one where the fused path is otherwise preferred.
    let dev = Device::Cpu;
    let mut rng = sd_tensor::rng::SeededRng::new(0);
    let xs = rng.randn((1, 64, 4096), &dev).unwrap();
    let alpha = rng.randn((4096,), &dev).unwrap();
    let (truth, _) = f64_reference(&xs, &alpha, 1e-6);

    let bf16 = ops::rms_norm(
        &xs.to_dtype(DType::BF16).unwrap(),
        &alpha.to_dtype(DType::BF16).unwrap(),
        1e-6,
    )
    .unwrap();
    let err = rel_err(&bf16.to_dtype(DType::F32).unwrap(), &truth);
    // bf16 carries 8 bits of mantissa, so the output cannot be better than
    // ~4e-3 however the sum is done. What is being checked is that the *sum*
    // did not also happen in bf16, which costs another order of magnitude.
    assert!(
        err < 2e-2,
        "bf16 rms_norm at 4096 channels: {err:.3e} — the reduction is not in f32"
    );
}
