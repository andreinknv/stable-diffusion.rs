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
