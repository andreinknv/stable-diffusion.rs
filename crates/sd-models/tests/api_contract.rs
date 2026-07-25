//! Compile-time proof that every API promised in `AGENTS.md` actually exists.
//!
//! The agent task specs tell a model "these functions exist, use only these".
//! If that list drifts from reality, the model burns a session guessing at
//! signatures — the exact failure mode the specs exist to prevent.
//!
//! This file exercises each promised API. If `AGENTS.md` overclaims, this stops
//! compiling and CI fails, so the docs cannot silently rot.
//!
//! Keep it in sync when the seam changes.

use sd_tensor::nn::{
    conv2d, conv2d_no_bias, embedding, group_norm, layer_norm, linear, linear_no_bias, Conv2d,
    Conv2dConfig, Embedding, GroupNorm, LayerNorm, LayerNormConfig, Linear, VarBuilder, VarMap,
};
use sd_tensor::{ops, testing, DType, Device, IndexOp, Module, Result, Shape, Tensor, D};

fn vb() -> (VarMap, Device) {
    (VarMap::new(), Device::Cpu)
}

#[test]
fn layer_constructors_have_the_documented_signatures() -> Result<()> {
    let (map, dev) = vb();
    let vb = VarBuilder::from_varmap(&map, DType::F32, &dev);

    // Task 03 / 04 / 05 rely on padding AND stride being settable.
    let _c: Conv2d = conv2d(
        4,
        320,
        3,
        Conv2dConfig {
            padding: 1,
            ..Default::default()
        },
        vb.pp("c1"),
    )?;
    let _s: Conv2d = conv2d(
        320,
        320,
        3,
        Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        },
        vb.pp("c2"),
    )?;
    let _n: Conv2d = conv2d_no_bias(4, 4, 1, Conv2dConfig::default(), vb.pp("c3"))?;

    // Task 04: to_q/to_k/to_v are bias-free, to_out.0 is not.
    let _l: Linear = linear(768, 768, vb.pp("l1"))?;
    let _lnb: Linear = linear_no_bias(768, 768, vb.pp("l2"))?;

    let _g: GroupNorm = group_norm(32, 320, 1e-5, vb.pp("g"))?;
    let _ln: LayerNorm = layer_norm(768, LayerNormConfig::default(), vb.pp("ln"))?;
    let _e: Embedding = embedding(49408, 768, vb.pp("e"))?;
    Ok(())
}

#[test]
fn layer_norm_eps_is_configurable() -> Result<()> {
    let (map, dev) = vb();
    let vb = VarBuilder::from_varmap(&map, DType::F32, &dev);
    // Task 02 needs eps = 1e-5 explicitly, not the default.
    let cfg = LayerNormConfig {
        eps: 1e-5,
        ..Default::default()
    };
    let _ln = layer_norm(768, cfg, vb.pp("ln"))?;
    Ok(())
}

#[test]
fn every_documented_op_exists() -> Result<()> {
    let dev = Device::Cpu;
    let x = Tensor::randn(0f32, 1f32, (2, 8, 16), &dev)?;

    let _ = ops::silu(&x)?;
    let _ = ops::swish(&x)?;
    let _ = ops::gelu(&x)?;
    let _ = ops::gelu_approx(&x)?;
    let _ = ops::quick_gelu(&x)?;
    let _ = ops::softmax(&x, D::Minus1)?;
    let _ = ops::softmax_last_dim(&x)?;

    let q = Tensor::randn(0f32, 1f32, (2, 4, 16, 8), &dev)?;
    let out = ops::scaled_dot_product_attention(&q, &q, &q)?;
    assert_eq!(out.dims(), &[2, 4, 16, 8]);

    // Task 02 (CLIP) needs the masked variant plus a causal mask.
    let mask = ops::causal_mask(16, &dev)?;
    assert_eq!(mask.dims(), &[1, 1, 16, 16]);
    let masked = ops::scaled_dot_product_attention_masked(&q, &q, &q, &mask)?;
    assert_eq!(masked.dims(), &[2, 4, 16, 8]);
    Ok(())
}

#[test]
fn causal_mask_blocks_the_future_and_never_the_past() -> Result<()> {
    let dev = Device::Cpu;
    let m = ops::causal_mask(4, &dev)?
        .reshape((4, 4))?
        .to_vec2::<f32>()?;
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if j <= i {
                assert_eq!(v, 0.0, "position ({i},{j}) must be visible");
            } else {
                assert!(v.is_infinite() && v < 0.0, "({i},{j}) must be masked");
            }
        }
    }

    // A causal mask must actually change the result, or it is a no-op bug.
    let q = Tensor::randn(0f32, 1f32, (1, 1, 4, 8), &dev)?;
    let plain = ops::scaled_dot_product_attention(&q, &q, &q)?;
    let mask = ops::causal_mask(4, &dev)?;
    let causal = ops::scaled_dot_product_attention_masked(&q, &q, &q, &mask)?;
    assert!(
        testing::closeness(&plain, &causal)?.max_abs > 1e-6,
        "masking must change the output"
    );
    Ok(())
}

#[test]
fn tensor_methods_used_by_the_task_specs_exist() -> Result<()> {
    let dev = Device::Cpu;
    let x = Tensor::randn(0f32, 1f32, (2, 320, 8, 8), &dev)?;

    // Task 04: the permute/reshape sandwich.
    let (b, c, h, w) = x.dims4()?;
    let y = x
        .permute((0, 2, 3, 1))?
        .contiguous()?
        .reshape((b, h * w, c))?;
    let _ = y
        .reshape((b, h, w, c))?
        .permute((0, 3, 1, 2))?
        .contiguous()?;

    // Task 04: GEGLU split.
    let g = Tensor::randn(0f32, 1f32, (2, 64, 2560), &dev)?;
    let hidden = g.narrow(D::Minus1, 0, 1280)?;
    let gate = g.narrow(D::Minus1, 1280, 1280)?;
    let _ = (hidden * ops::gelu(&gate)?)?;

    // Task 03: time embedding broadcast.
    let t = Tensor::randn(0f32, 1f32, (2, 320), &dev)?;
    let t4 = t.unsqueeze(2)?.unsqueeze(3)?;
    let _ = x.broadcast_add(&t4)?;

    // Task 03: sinusoid construction.
    let ar = Tensor::arange(0f32, 160f32, &dev)?;
    let _ = ((ar * -1.0)?.exp()?.cos()?.sin()?).sqr()?;

    // Task 05: skip concatenation.
    let _ = Tensor::cat(&[&x, &x], 1)?;

    // Task 07: latent scaling and upsampling.
    let _ = (&x / 2.0)?;
    let _ = x.upsample_nearest2d(16, 16)?;
    let _ = x.i(0)?;
    let _ = x.to_dtype(DType::F32)?.clamp(-1.0, 1.0)?;
    let _: Shape = x.shape().clone();
    Ok(())
}

#[test]
fn seeded_randn_is_reproducible_without_a_new_dependency() -> Result<()> {
    // Task 07 needs deterministic seeding. candle's Device::set_seed errors on
    // CPU, so the seam provides its own generator. This proves same-seed
    // reproducibility with no `rand` dependency in any Cargo.toml.
    use sd_tensor::rng::SeededRng;
    let dev = Device::Cpu;

    let a = SeededRng::new(42).randn((1, 4, 8, 8), &dev)?;
    let b = SeededRng::new(42).randn((1, 4, 8, 8), &dev)?;
    let c = testing::closeness(&a, &b)?;
    assert_eq!(c.max_abs, 0.0, "same seed must give identical noise: {c}");

    let d = SeededRng::new(43).randn((1, 4, 8, 8), &dev)?;
    assert!(
        testing::closeness(&a, &d)?.max_abs > 0.0,
        "different seeds must differ"
    );
    assert_eq!(a.dims(), &[1, 4, 8, 8]);
    Ok(())
}

#[test]
fn seeded_noise_is_approximately_standard_normal() -> Result<()> {
    // A Box-Muller bug (wrong constant, missing sqrt) still produces plausible
    // noise. Check the moments so it cannot pass silently.
    use sd_tensor::rng::SeededRng;
    let v = SeededRng::new(7).normals(200_000);
    let n = v.len() as f64;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;

    assert!(mean.abs() < 0.02, "mean should be ~0, got {mean}");
    assert!((var - 1.0).abs() < 0.02, "variance should be ~1, got {var}");
    Ok(())
}

#[test]
fn testing_helpers_exist_and_report_usefully() -> Result<()> {
    let dev = Device::Cpu;
    let a = Tensor::zeros((2, 3), DType::F32, &dev)?;
    let b = Tensor::zeros((2, 3), DType::F32, &dev)?;

    testing::assert_close(&a, &b, testing::DEFAULT_ATOL, "identical zeros")?;
    let c = testing::closeness(&a, &b)?;
    assert_eq!(c.max_abs, 0.0);
    let _ = testing::max_abs_diff(&a, &b)?;
    let _ = testing::DEFAULT_RTOL;

    // Mismatched shapes must report rather than panic — task specs rely on
    // this to distinguish structural from numerical failures.
    let wrong = Tensor::zeros((3, 2), DType::F32, &dev)?;
    let c = testing::closeness(&a, &wrong)?;
    assert!(
        c.max_abs.is_infinite(),
        "shape mismatch must be reported, not panic"
    );
    Ok(())
}

#[test]
fn module_forward_is_in_scope_via_the_seam() -> Result<()> {
    let (map, dev) = vb();
    let vbb = VarBuilder::from_varmap(&map, DType::F32, &dev);
    let l = linear(4, 8, vbb.pp("l"))?;
    let x = Tensor::zeros((2, 4), DType::F32, &dev)?;
    // `Module` must be importable from sd_tensor for `.forward` to resolve.
    let y = l.forward(&x)?;
    assert_eq!(y.dims(), &[2, 8]);
    Ok(())
}
