//! Evidence that the MLX backend computes the same arithmetic as candle.
//!
//! `docs/handoff.md` sets the bar: `sd-tensor` compiles against MLX behind a
//! feature flag, candle stays the default, and the ops agree between the two.
//!
//! ```bash
//! cargo test -p sd-tensor --features mlx --test mlx_agrees_with_candle
//! ```
//!
//! **Bounds.** Elementwise arithmetic is exact in f32 for these inputs, so
//! those are checked with equality — a tolerance would hide a backend quietly
//! computing something else. Ops that reduce or reorder (matmul, softmax,
//! group_norm, conv2d) get a tolerance, because f32 summation order genuinely
//! differs between the two and demanding equality would be testing that MLX
//! implements candle rather than that it implements the arithmetic.
//! `docs/handoff.md` rule 3 applies: these are loose enough not to measure
//! float32 and tight enough that a real porting bug cannot pass. The VAE's
//! asymmetric-padding bug, for calibration, showed 17.32.
#![cfg(feature = "mlx")]

use sd_tensor::mlx::{eval, Array, Stream};
use sd_tensor::{DType, Device, IndexOp, Tensor};

const TOL: f32 = 2e-5;

/// Deterministic, and not all near-identical, so a transposed or truncated
/// buffer cannot pass by luck.
fn ramp(n: usize, scale: f32, shift: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37).sin() * scale + shift)
        .collect()
}

fn candle_vec(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: element count");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: max_abs {worst:.3e} at element {at} exceeds {tol:.0e} \
         (mlx {}, candle {})",
        got[at],
        want[at]
    );
}

// -- elementwise, exact ----------------------------------------------------

#[test]
fn elementwise_matches_candle_exactly() {
    let shape = [2usize, 3, 4];
    let n: usize = shape.iter().product();
    let (a, b) = (ramp(n, 2.0, 0.5), ramp(n, 1.0, 3.0));

    let s = Stream::gpu();
    let ma = Array::from_slice_f32(&a, &shape).unwrap();
    let mb = Array::from_slice_f32(&b, &shape).unwrap();

    let dev = Device::Cpu;
    let ca = Tensor::from_vec(a.clone(), shape.as_slice(), &dev).unwrap();
    let cb = Tensor::from_vec(b.clone(), shape.as_slice(), &dev).unwrap();

    for (name, got, want) in [
        ("add", ma.add(&mb, &s).unwrap(), (&ca + &cb).unwrap()),
        ("sub", ma.sub(&mb, &s).unwrap(), (&ca - &cb).unwrap()),
        ("mul", ma.mul(&mb, &s).unwrap(), (&ca * &cb).unwrap()),
        ("div", ma.div(&mb, &s).unwrap(), (&ca / &cb).unwrap()),
    ] {
        assert_eq!(got.shape(), shape, "{name}: shape");
        let g = got.to_vec_f32(&s).unwrap();
        let w = candle_vec(&want);
        for (i, (x, y)) in g.iter().zip(&w).enumerate() {
            assert_eq!(x, y, "{name} element {i}");
        }
    }
}

#[test]
fn silu_matches_candle() {
    let n = 64;
    let a = ramp(n, 6.0, -1.0); // spans both tails of the sigmoid
    let s = Stream::gpu();
    let got = Array::from_slice_f32(&a, &[n])
        .unwrap()
        .silu(&s)
        .unwrap()
        .to_vec_f32(&s)
        .unwrap();

    let ca = Tensor::from_vec(a, &[n], &Device::Cpu).unwrap();
    let want = candle_vec(&sd_tensor::ops::silu(&ca).unwrap());
    assert_close(&got, &want, TOL, "silu");
}

// -- shape -----------------------------------------------------------------

#[test]
fn reshape_and_transpose_match_candle() {
    let shape = [2usize, 3, 4];
    let n: usize = shape.iter().product();
    let a = ramp(n, 3.0, 0.0);
    let s = Stream::gpu();
    let ma = Array::from_slice_f32(&a, &shape).unwrap();
    let ca = Tensor::from_vec(a, shape.as_slice(), &Device::Cpu).unwrap();

    let r = ma.reshape(&[6, 4], &s).unwrap();
    assert_eq!(r.shape(), vec![6, 4]);
    assert_close(
        &r.to_vec_f32(&s).unwrap(),
        &candle_vec(&ca.reshape((6, 4)).unwrap()),
        0.0,
        "reshape",
    );

    let t = ma.transpose(&[2, 0, 1], &s).unwrap();
    assert_eq!(t.shape(), vec![4, 2, 3]);
    assert_close(
        &t.to_vec_f32(&s).unwrap(),
        &candle_vec(&ca.permute((2, 0, 1)).unwrap().contiguous().unwrap()),
        0.0,
        "transpose",
    );
}

// -- reductions and matmul -------------------------------------------------

#[test]
fn matmul_matches_candle() {
    let (m, k, nn_) = (7usize, 5usize, 3usize);
    let a = ramp(m * k, 2.0, 0.3);
    let b = ramp(k * nn_, 1.5, -0.4);
    let s = Stream::gpu();
    let got = Array::from_slice_f32(&a, &[m, k])
        .unwrap()
        .matmul(&Array::from_slice_f32(&b, &[k, nn_]).unwrap(), &s)
        .unwrap();
    assert_eq!(got.shape(), vec![m, nn_]);

    let dev = Device::Cpu;
    let ca = Tensor::from_vec(a, &[m, k], &dev).unwrap();
    let cb = Tensor::from_vec(b, &[k, nn_], &dev).unwrap();
    assert_close(
        &got.to_vec_f32(&s).unwrap(),
        &candle_vec(&ca.matmul(&cb).unwrap()),
        TOL,
        "matmul",
    );
}

#[test]
fn sum_and_mean_match_candle() {
    let shape = [2usize, 6];
    let a = ramp(12, 4.0, 1.0);
    let s = Stream::gpu();
    let ma = Array::from_slice_f32(&a, &shape).unwrap();
    let ca = Tensor::from_vec(a, shape.as_slice(), &Device::Cpu).unwrap();

    let got = ma.sum(&[1], false, &s).unwrap();
    assert_eq!(got.shape(), vec![2]);
    assert_close(
        &got.to_vec_f32(&s).unwrap(),
        &candle_vec(&ca.sum(1).unwrap()),
        TOL,
        "sum",
    );

    let got = ma.mean(&[1], true, &s).unwrap();
    assert_eq!(got.shape(), vec![2, 1], "keepdims");
    assert_close(
        &got.to_vec_f32(&s).unwrap(),
        &candle_vec(&ca.mean_keepdim(1).unwrap()),
        TOL,
        "mean",
    );
}

#[test]
fn softmax_matches_candle() {
    let shape = [3usize, 16];
    let a = ramp(48, 8.0, 0.0); // wide enough that a missing max-subtraction shows
    let s = Stream::gpu();
    let got = Array::from_slice_f32(&a, &shape)
        .unwrap()
        .softmax(-1, &s)
        .unwrap()
        .to_vec_f32(&s)
        .unwrap();

    let ca = Tensor::from_vec(a, shape.as_slice(), &Device::Cpu).unwrap();
    let want = candle_vec(&sd_tensor::ops::softmax_last_dim(&ca).unwrap());
    assert_close(&got, &want, TOL, "softmax");

    for row in 0..3 {
        let total: f32 = got[row * 16..(row + 1) * 16].iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "row {row} sums to {total}");
    }
}

// -- convolution, the op the migration is for ------------------------------

/// MLX is NHWC with `(out, kh, kw, in)` weights; candle is NCHW with
/// `(out, in, kh, kw)`. Both are given the same logical tensors in their own
/// layout, and the result is compared after transposing MLX's back — so a
/// layout mistake fails rather than cancelling out.
#[test]
fn conv2d_matches_candle() {
    let (n, cin, hw, cout) = (1usize, 4usize, 8usize, 6usize);
    let x_nchw = ramp(n * cin * hw * hw, 2.0, 0.1);
    let w_oihw = ramp(cout * cin * 9, 1.0, -0.2);
    let s = Stream::gpu();
    let dev = Device::Cpu;

    let cx = Tensor::from_vec(x_nchw.clone(), &[n, cin, hw, hw], &dev).unwrap();
    let cw = Tensor::from_vec(w_oihw.clone(), &[cout, cin, 3, 3], &dev).unwrap();
    let want = cx.conv2d(&cw, 1, 1, 1, 1).unwrap();

    // NCHW -> NHWC, OIHW -> OHWI, using MLX's own transpose.
    let mx_x = Array::from_slice_f32(&x_nchw, &[n, cin, hw, hw])
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let mx_w = Array::from_slice_f32(&w_oihw, &[cout, cin, 3, 3])
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();

    let got = mx_x
        .conv2d(&mx_w, (1, 1), (1, 1), (1, 1), 1, &s)
        .unwrap()
        .transpose(&[0, 3, 1, 2], &s)
        .unwrap();

    assert_eq!(got.shape(), want.dims().to_vec(), "conv2d shape");
    assert_close(
        &got.to_vec_f32(&s).unwrap(),
        &candle_vec(&want),
        1e-4,
        "conv2d",
    );
}

// -- group norm ------------------------------------------------------------

/// Against candle's `group_norm`, which the golden tests already hold to
/// diffusers. NHWC in MLX, NCHW in candle, compared in candle's layout.
#[test]
fn group_norm_matches_candle() {
    let (n, c, hw, groups) = (1usize, 32usize, 8usize, 8usize);
    let x_nchw = ramp(n * c * hw * hw, 2.0, 6.0); // large mean beside the spread
    let s = Stream::gpu();
    let dev = Device::Cpu;

    // The arithmetic directly rather than candle's `GroupNorm`, whose
    // constructor wants a `VarBuilder` holding affine parameters this test
    // does not use. This is the same expression `golden_unet` holds to
    // diffusers, so it is a reference and not a restatement of the code under
    // test: candle reduces over NCHW, MLX over NHWC, and they must still agree.
    let cx = Tensor::from_vec(x_nchw.clone(), &[n, c, hw, hw], &dev).unwrap();
    let g = cx.reshape((n, groups, (c / groups) * hw * hw)).unwrap();
    let mean = g.mean_keepdim(2).unwrap();
    let d = g.broadcast_sub(&mean).unwrap();
    let var = (&d * &d).unwrap().mean_keepdim(2).unwrap();
    let want = d
        .broadcast_div(&(var + 1e-6).unwrap().sqrt().unwrap())
        .unwrap()
        .reshape((n, c, hw, hw))
        .unwrap();

    let mx_x = Array::from_slice_f32(&x_nchw, &[n, c, hw, hw])
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let got = mx_x
        .group_norm(groups, 1e-6, None, None, &s)
        .unwrap()
        .transpose(&[0, 3, 1, 2], &s)
        .unwrap();

    assert_eq!(got.shape(), want.dims().to_vec(), "group_norm shape");
    assert_close(
        &got.to_vec_f32(&s).unwrap(),
        &candle_vec(&want),
        1e-4,
        "group_norm",
    );
}

// -- laziness, dtypes, and errors -----------------------------------------

/// The laziness is the point, so check it is there: a chain can be built
/// without evaluating, and an explicit `eval` is what makes it readable.
#[test]
fn results_are_lazy_until_evaluated() {
    let (a, b) = (ramp(8, 2.0, 0.0), ramp(8, 1.0, 1.0));
    let s = Stream::gpu();
    let ma = Array::from_slice_f32(&a, &[8]).unwrap();
    let mb = Array::from_slice_f32(&b, &[8]).unwrap();

    let chained = ma.add(&mb, &s).unwrap().mul(&ma, &s).unwrap();
    assert_eq!(chained.elem_count(), 8);

    eval(&[&chained]).expect("explicit eval");
    let got = chained.to_vec_f32(&s).unwrap();
    for i in 0..8 {
        assert!(((a[i] + b[i]) * a[i] - got[i]).abs() < TOL, "element {i}");
    }
}

#[test]
fn f16_round_trip_keeps_dtype_honest() {
    let a = ramp(16, 3.0, 0.0);
    let s = Stream::gpu();
    let f32_arr = Array::from_slice_f32(&a, &[16]).unwrap();
    assert!(f32_arr.is_f32());

    let half = f32_arr.to_f16(&s).unwrap();
    assert!(!half.is_f32());
    // Reading f32 from an f16 array is refused rather than reinterpreted.
    assert!(half.to_vec_f32(&s).is_err());

    let back = half.to_f32(&s).unwrap().to_vec_f32(&s).unwrap();
    for (i, (g, w)) in back.iter().zip(&a).enumerate() {
        assert!((g - w).abs() < 2e-3, "element {i}: {g} vs {w}");
    }
}

/// The trampoline: a failure MLX raises must arrive with its message, not a
/// bare status code.
#[test]
fn mlx_errors_carry_their_message() {
    let s = Stream::gpu();
    let a = Array::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    // 4 elements cannot become 3x3; MLX raises, mlx-c routes the text to the
    // global handler, and the thread-local must reunite it with the status.
    let err = a.reshape(&[3, 3], &s).unwrap_err().to_string();
    assert!(err.contains("reshape"), "names the op: {err}");
    assert!(
        err.len() > "mlx: reshape failed with status 1".len(),
        "carries MLX's own text rather than only a status: {err}"
    );
}

#[test]
fn wrapper_validates_its_own_contract() {
    let s = Stream::gpu();
    assert!(Array::from_slice_f32(&[1.0, 2.0, 3.0], &[2, 2]).is_err());
    let a = Array::from_slice_f32(&[1.0; 8], &[1, 2, 2, 2]).unwrap();
    assert!(a.group_norm(3, 1e-6, None, None, &s).is_err(), "2 % 3 != 0");
    let flat = Array::from_slice_f32(&[1.0; 4], &[4]).unwrap();
    assert!(
        flat.group_norm(2, 1e-6, None, None, &s).is_err(),
        "group_norm needs NHWC"
    );
}

#[test]
fn index_op_is_still_candles() {
    // Guards the seam claim: candle remains usable from this crate while both
    // backends coexist, which is what `docs/handoff.md` step 5 depends on.
    let t = Tensor::from_vec(ramp(6, 1.0, 0.0), &[2, 3], &Device::Cpu).unwrap();
    assert_eq!(t.i(0).unwrap().dims(), &[3]);
}

// -- second op batch: what the UNet census said it needs -------------------

#[test]
fn unary_transcendentals_match_candle() {
    let n = 48;
    let a = ramp(n, 2.0, 0.05); // stays clear of log/sqrt domain edges
    let s = Stream::gpu();
    let m = Array::from_slice_f32(&a, &[n]).unwrap();
    let c = Tensor::from_vec(a.clone(), &[n], &Device::Cpu).unwrap();

    for (name, got, want) in [
        ("tanh", m.tanh(&s).unwrap(), c.tanh().unwrap()),
        ("exp", m.exp(&s).unwrap(), c.exp().unwrap()),
        ("cos", m.cos(&s).unwrap(), c.cos().unwrap()),
        ("sin", m.sin(&s).unwrap(), c.sin().unwrap()),
    ] {
        assert_close(&got.to_vec_f32(&s).unwrap(), &candle_vec(&want), TOL, name);
    }

    // log and rsqrt need strictly positive input.
    let pos: Vec<f32> = (1..=n).map(|i| i as f32 * 0.25).collect();
    let mp = Array::from_slice_f32(&pos, &[n]).unwrap();
    let cp = Tensor::from_vec(pos, &[n], &Device::Cpu).unwrap();
    assert_close(
        &mp.log(&s).unwrap().to_vec_f32(&s).unwrap(),
        &candle_vec(&cp.log().unwrap()),
        TOL,
        "log",
    );
    assert_close(
        &mp.rsqrt(&s).unwrap().to_vec_f32(&s).unwrap(),
        &candle_vec(&cp.powf(-0.5).unwrap()),
        1e-4,
        "rsqrt",
    );
}

#[test]
fn gelu_matches_candle() {
    let n = 96;
    let a = ramp(n, 5.0, 0.0);
    let s = Stream::gpu();
    let got = Array::from_slice_f32(&a, &[n])
        .unwrap()
        .gelu(&s)
        .unwrap()
        .to_vec_f32(&s)
        .unwrap();
    let c = Tensor::from_vec(a, &[n], &Device::Cpu).unwrap();
    assert_close(&got, &candle_vec(&c.gelu_erf().unwrap()), 1e-5, "gelu");
}

/// **Measured, and it does inherit it.** `docs/handoff.md` records candle's
/// `gelu_erf` returning *exactly zero* below about -6 where the truth is
/// -5.9e-9, because `1 + erf(u)` rounds the tail away by subtraction. MLX's
/// erf has the same formulation and the same collapse, so this is parity with
/// candle rather than a regression — but it is also a gap against the fused
/// kernel this project already wrote, which reads `erfc` off the same
/// polynomial and keeps the tail.
///
/// `mlx-c` exposes no `erfc`, so closing it means a custom op. Until then this
/// test pins the behaviour: it fails if either side changes, rather than
/// leaving the question to be rediscovered.
#[test]
fn gelu_left_tail_collapses_exactly_as_candles_does() {
    let xs: Vec<f32> = vec![-6.0, -7.0, -8.0, -9.0, -10.0];
    let s = Stream::gpu();
    let got = Array::from_slice_f32(&xs, &[xs.len()])
        .unwrap()
        .gelu(&s)
        .unwrap()
        .to_vec_f32(&s)
        .unwrap();
    let candle = candle_vec(
        &Tensor::from_vec(xs.clone(), &[xs.len()], &Device::Cpu)
            .unwrap()
            .gelu_erf()
            .unwrap(),
    );
    // Truth at -6 is about -5.9e-9 and shrinks from there; the failure mode is
    // a hard zero, so that is what is asserted against.
    for (i, x) in xs.iter().enumerate() {
        eprintln!(
            "x={x:>6}  mlx {:>12.4e}  candle {:>12.4e}",
            got[i], candle[i]
        );
    }
    for (i, x) in xs.iter().enumerate() {
        assert_eq!(
            got[i], candle[i],
            "x={x}: MLX and candle must agree on the tail; they both collapse to zero"
        );
        assert_eq!(got[i], 0.0, "x={x}: the collapse is what is being pinned");
    }
}

#[test]
fn layer_norm_matches_candle() {
    let (rows, cols) = (4usize, 32usize);
    let a = ramp(rows * cols, 3.0, 1.0);
    let g = ramp(cols, 0.5, 1.0);
    let b = ramp(cols, 0.2, 0.0);
    let s = Stream::gpu();
    let dev = Device::Cpu;

    let got = Array::from_slice_f32(&a, &[rows, cols])
        .unwrap()
        .layer_norm(
            Some(&Array::from_slice_f32(&g, &[cols]).unwrap()),
            Some(&Array::from_slice_f32(&b, &[cols]).unwrap()),
            1e-5,
            &s,
        )
        .unwrap();

    // The arithmetic directly: candle's LayerNorm wants a VarBuilder for the
    // affine params, and this is the expression golden_unet holds to diffusers.
    let ca = Tensor::from_vec(a, &[rows, cols], &dev).unwrap();
    let mean = ca.mean_keepdim(1).unwrap();
    let d = ca.broadcast_sub(&mean).unwrap();
    let var = (&d * &d).unwrap().mean_keepdim(1).unwrap();
    let want = d
        .broadcast_div(&(var + 1e-5).unwrap().sqrt().unwrap())
        .unwrap()
        .broadcast_mul(&Tensor::from_vec(g, &[cols], &dev).unwrap())
        .unwrap()
        .broadcast_add(&Tensor::from_vec(b, &[cols], &dev).unwrap())
        .unwrap();

    assert_close(
        &got.to_vec_f32(&s).unwrap(),
        &candle_vec(&want),
        1e-4,
        "layer_norm",
    );
}

#[test]
fn concat_and_narrow_match_candle() {
    let s = Stream::gpu();
    let dev = Device::Cpu;
    let (a, b) = (ramp(12, 2.0, 0.0), ramp(8, 1.0, 2.0));

    let ma = Array::from_slice_f32(&a, &[3, 4]).unwrap();
    let mb = Array::from_slice_f32(&b, &[2, 4]).unwrap();
    let joined = sd_tensor::mlx::concat(&[&ma, &mb], 0, &s).unwrap();
    assert_eq!(joined.shape(), vec![5, 4]);

    let ca = Tensor::from_vec(a, &[3, 4], &dev).unwrap();
    let cb = Tensor::from_vec(b, &[2, 4], &dev).unwrap();
    let want = Tensor::cat(&[&ca, &cb], 0).unwrap();
    assert_close(
        &joined.to_vec_f32(&s).unwrap(),
        &candle_vec(&want),
        0.0,
        "concat",
    );

    let cut = joined.narrow(0, 1, 3, &s).unwrap();
    assert_eq!(cut.shape(), vec![3, 4]);
    assert_close(
        &cut.to_vec_f32(&s).unwrap(),
        &candle_vec(&want.narrow(0, 1, 3).unwrap()),
        0.0,
        "narrow",
    );

    assert!(joined.narrow(0, 4, 3, &s).is_err(), "past the end");
    assert!(joined.narrow(9, 0, 1, &s).is_err(), "bad axis");
}

#[test]
fn sdpa_matches_candles_attention() {
    let (b, h, sq, hd) = (1usize, 2usize, 6usize, 8usize);
    let n = b * h * sq * hd;
    let (q, k, v) = (ramp(n, 1.0, 0.0), ramp(n, 0.8, 0.2), ramp(n, 1.2, -0.1));
    let s = Stream::gpu();
    let dev = Device::Cpu;
    let shape = [b, h, sq, hd];

    let got = Array::from_slice_f32(&q, &shape)
        .unwrap()
        .sdpa(
            &Array::from_slice_f32(&k, &shape).unwrap(),
            &Array::from_slice_f32(&v, &shape).unwrap(),
            1.0 / (hd as f32).sqrt(),
            &s,
        )
        .unwrap();
    assert_eq!(got.shape(), shape.to_vec());

    let cq = Tensor::from_vec(q, shape.as_slice(), &dev).unwrap();
    let ck = Tensor::from_vec(k, shape.as_slice(), &dev).unwrap();
    let cv = Tensor::from_vec(v, shape.as_slice(), &dev).unwrap();
    let want = sd_tensor::ops::scaled_dot_product_attention(&cq, &ck, &cv).unwrap();

    assert_close(
        &got.to_vec_f32(&s).unwrap(),
        &candle_vec(&want),
        1e-5,
        "sdpa",
    );
}
