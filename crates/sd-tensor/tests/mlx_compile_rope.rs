//! Compiling the rotary application — the one place MLX's own Flux does.
//!
//! Apple's `mlx-examples/flux` compiles exactly one function in the model:
//!
//! ```python
//! @partial(mx.compile, shapeless=True)
//! def _ab_plus_cd(a, b, c, d):
//!     return a * b + c * d
//! ```
//!
//! applied to `q` and `k` in every attention. It does **not** compile the
//! modulation chain, and it does **not** use `mx.fast.rope` — it builds the
//! cos/sin tables per axis by hand, exactly as `flux::embed_nd` does. Both
//! choices independently match what this project measured its way to.
//!
//! That leaves one target untried, and it is a better one than the modulation:
//! `rotate` runs on q and k in every attention across 57 blocks, where the
//! modulation runs once per stream per block. This measures it.
#![cfg(feature = "mlx")]

use std::time::Instant;

use sd_tensor::mlx::{eval, Array, Compiled, Stream};

/// The interleaved rotation, as `sd_models::mlx::flux::rotate` spells it:
/// `even*cos - odd*sin` and `even*sin + odd*cos`, which is Apple's
/// `_ab_plus_cd` twice.
fn rotate_pairs(args: &[Array], s: &Stream) -> sd_tensor::Result<Vec<Array>> {
    let (even, odd, cos, sin) = (&args[0], &args[1], &args[2], &args[3]);
    Ok(vec![
        even.mul(cos, s)?.sub(&odd.mul(sin, s)?, s)?,
        even.mul(sin, s)?.add(&odd.mul(cos, s)?, s)?,
    ])
}

/// Flux at 512: 24 heads, 1536 tokens (1024 image + 512 text), head_dim 128,
/// so the pair arrays are `[1, 24, 1536, 64]`.
fn inputs() -> (Array, Array, Array, Array) {
    let (b, h, n, half) = (1usize, 24, 1536, 64);
    let big: Vec<f32> = (0..b * h * n * half)
        .map(|i| (i % 101) as f32 * 0.01 - 0.5)
        .collect();
    let table: Vec<f32> = (0..n * half).map(|i| (i % 37) as f32 * 0.02).collect();
    (
        Array::from_slice_f32(&big, &[b, h, n, half]).unwrap(),
        Array::from_slice_f32(&big, &[b, h, n, half]).unwrap(),
        Array::from_slice_f32(&table, &[1, 1, n, half]).unwrap(),
        Array::from_slice_f32(&table, &[1, 1, n, half]).unwrap(),
    )
}

#[test]
fn compiling_the_rotation_does_not_change_it() {
    let s = Stream::gpu();
    let (even, odd, cos, sin) = inputs();
    let plain = rotate_pairs(
        &[
            even.contiguous(&s).unwrap(),
            odd.contiguous(&s).unwrap(),
            cos.contiguous(&s).unwrap(),
            sin.contiguous(&s).unwrap(),
        ],
        &s,
    )
    .unwrap();

    let st = Stream::gpu();
    let compiled = Compiled::new(move |a| rotate_pairs(a, &st)).expect("compile");
    let fused = compiled.call(&[&even, &odd, &cos, &sin]).expect("call");

    for (i, (p, f)) in plain.iter().zip(&fused).enumerate() {
        let (a, b) = (p.to_vec_f32(&s).unwrap(), f.to_vec_f32(&s).unwrap());
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-6, "output {i} differs by {worst:.3e}");
    }
}

/// What it is worth on Flux's real attention shape.
#[test]
fn how_much_does_compiling_the_rotation_buy() {
    let s = Stream::gpu();
    let (even, odd, cos, sin) = inputs();
    let st = Stream::gpu();
    let compiled = Compiled::new(move |a| rotate_pairs(a, &st)).expect("compile");

    const REPEATS: usize = 40;
    let run_plain = || {
        let out = rotate_pairs(
            &[
                even.contiguous(&s).unwrap(),
                odd.contiguous(&s).unwrap(),
                cos.contiguous(&s).unwrap(),
                sin.contiguous(&s).unwrap(),
            ],
            &s,
        )
        .unwrap();
        eval(&[&out[0], &out[1]]).unwrap();
    };
    let run_fused = || {
        let out = compiled.call(&[&even, &odd, &cos, &sin]).unwrap();
        eval(&[&out[0], &out[1]]).unwrap();
    };
    run_plain();
    run_fused();

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
        "rope rotation, [1,24,1536,64]:  plain {p:.0} us   compiled {f:.0} us   {:.2}x",
        p / f
    );
}
