//! MLX's graph compiler: does it fuse, and is the fusion worth anything here?
//!
//! Two questions, and they have different answers. Correctness is easy to
//! settle. The payoff is not, and is the reason this file measures rather than
//! asserts a speedup: MLX fuses **elementwise chains**, and a diffusion step is
//! roughly three quarters convolution and matmul, which a fuser does not touch.
#![cfg(feature = "mlx")]

use std::time::Instant;

use sd_tensor::mlx::{eval, Array, Compiled, Stream};

/// The modulation chain a DiT block runs: `norm(x) * (1 + scale) + shift`,
/// then a gated residual. Elementwise throughout, which is what makes it the
/// natural candidate — and it is the same shape the candle-era hand-written
/// adaLN kernel targeted at 5.26x.
fn modulate(args: &[Array], s: &Stream) -> sd_tensor::Result<Vec<Array>> {
    let (x, scale, shift, gate) = (&args[0], &args[1], &args[2], &args[3]);
    let normed = x.layer_norm(None, None, 1e-6, s)?;
    let one = Array::scalar_f32(1.0)?;
    let modulated = normed.mul(&scale.add(&one, s)?, s)?.add(shift, s)?;
    let out = x.add(&modulated.mul(gate, s)?, s)?;
    Ok(vec![out])
}

fn inputs(s: &Stream) -> (Array, Array, Array, Array) {
    // One Flux double block's image stream at 512x512: 1024 tokens, 3072 wide.
    let (n, d) = (1024usize, 3072usize);
    let ramp: Vec<f32> = (0..n * d).map(|i| (i % 97) as f32 * 0.01 - 0.5).collect();
    let row: Vec<f32> = (0..d).map(|i| (i % 13) as f32 * 0.02).collect();
    let _ = s;
    (
        Array::from_slice_f32(&ramp, &[1, n, d]).unwrap(),
        Array::from_slice_f32(&row, &[1, 1, d]).unwrap(),
        Array::from_slice_f32(&row, &[1, 1, d]).unwrap(),
        Array::from_slice_f32(&row, &[1, 1, d]).unwrap(),
    )
}

/// **A compiled function returns what the uncompiled one does.**
///
/// The claim everything else depends on. Compilation reorders and fuses; if it
/// changed a result, every measurement below would be timing the wrong thing.
#[test]
fn compiling_does_not_change_the_answer() {
    let s = Stream::gpu();
    let (x, scale, shift, gate) = inputs(&s);

    let plain = modulate(
        &[
            x.contiguous(&s).unwrap(),
            scale.contiguous(&s).unwrap(),
            shift.contiguous(&s).unwrap(),
            gate.contiguous(&s).unwrap(),
        ],
        &s,
    )
    .unwrap();

    let st = Stream::gpu();
    let compiled = Compiled::new(move |args| modulate(args, &st)).expect("compile");
    let fused = compiled.call(&[&x, &scale, &shift, &gate]).expect("call");

    assert_eq!(fused.len(), plain.len());
    let (a, b) = (
        plain[0].to_vec_f32(&s).unwrap(),
        fused[0].to_vec_f32(&s).unwrap(),
    );
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(&b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, f32::max);
    eprintln!("compiled vs plain: max_abs {worst:.3e}");
    // Fusion changes the order of operations, so this is a floating-point
    // agreement rather than a bit-for-bit one.
    assert!(
        worst < 1e-5,
        "compilation changed the result by {worst:.3e}"
    );
}

/// **What compilation is worth on this shape.**
///
/// Reported rather than asserted. A threshold here would pin a number measured
/// on one machine in one state, and the honest output of this test is the
/// measurement itself — which is what the roadmap quotes.
#[test]
fn how_much_does_compiling_the_modulation_chain_buy() {
    let s = Stream::gpu();
    let (x, scale, shift, gate) = inputs(&s);
    let st = Stream::gpu();
    let compiled = Compiled::new(move |args| modulate(args, &st)).expect("compile");

    const REPEATS: usize = 40;
    let run_plain = || {
        let out = modulate(
            &[
                x.contiguous(&s).unwrap(),
                scale.contiguous(&s).unwrap(),
                shift.contiguous(&s).unwrap(),
                gate.contiguous(&s).unwrap(),
            ],
            &s,
        )
        .unwrap();
        eval(&[&out[0]]).unwrap();
    };
    let run_fused = || {
        let out = compiled.call(&[&x, &scale, &shift, &gate]).unwrap();
        eval(&[&out[0]]).unwrap();
    };

    // Warm both: the first compiled call pays for the trace, and the first of
    // either pays for kernel loading. Timing those would measure startup.
    run_plain();
    run_fused();

    // Alternated, so a machine that gets busy halfway does not favour whichever
    // ran second.
    let (mut plain, mut fused) = (0u128, 0u128);
    for _ in 0..REPEATS {
        let t = Instant::now();
        run_plain();
        plain += t.elapsed().as_micros();
        let t = Instant::now();
        run_fused();
        fused += t.elapsed().as_micros();
    }
    let (p, f) = (plain as f64 / REPEATS as f64, fused as f64 / REPEATS as f64);
    eprintln!(
        "modulation chain, 1024x3072:  plain {p:.0} us   compiled {f:.0} us   {:.2}x",
        p / f
    );
}

/// **A compiled function retraces at a new shape**, rather than returning a
/// wrong answer from the cached kernels.
#[test]
fn a_new_shape_is_handled_rather_than_reused() {
    let s = Stream::gpu();
    let st = Stream::gpu();
    let compiled = Compiled::new(move |args| {
        let out = args[0].mul(&args[1], &st)?.add(&args[0], &st)?;
        Ok(vec![out])
    })
    .expect("compile");

    for n in [4usize, 16, 4] {
        let a = Array::from_slice_f32(&vec![2.0; n], &[n]).unwrap();
        let b = Array::from_slice_f32(&vec![3.0; n], &[n]).unwrap();
        let out = compiled.call(&[&a, &b]).expect("call");
        assert_eq!(out[0].shape(), vec![n], "shape {n}");
        assert_eq!(
            out[0].to_vec_f32(&s).unwrap(),
            vec![8.0; n],
            "2*3 + 2 at shape {n}"
        );
    }
}

/// **An error inside the traced function is reported, not a crash.**
///
/// The trampoline catches panics too: unwinding through C is undefined, and a
/// shape mismatch inside a closure is an ordinary mistake to make.
#[test]
fn a_failing_closure_returns_an_error() {
    let st = Stream::gpu();
    let compiled = Compiled::new(move |args| {
        // Deliberately incompatible shapes.
        args[0].matmul(&args[1], &st).map(|v| vec![v])
    })
    .expect("compile");

    let a = Array::from_slice_f32(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Array::from_slice_f32(&[1.0, 2.0], &[2]).unwrap();
    assert!(
        compiled.call(&[&a, &b]).is_err(),
        "a closure that cannot run must surface an error rather than abort"
    );
}
