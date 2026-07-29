//! Our adaptive layer norm kernel against the four ops it replaces.
//!
//! Correctness at the shapes Flux-dev and SD 3.5 run, then speed. The
//! conditioning tensors are narrowed out of a wider projection here exactly as
//! the models produce them, so the non-contiguous case is the one measured.

use anyhow::Result;
use sd_tensor::{fused, ops, Device, Tensor, D};

fn composed(xs: &Tensor, scale: &Tensor, shift: &Tensor, eps: f64) -> Result<Tensor> {
    Ok(ops::plain_layer_norm(xs, eps)?
        .broadcast_mul(&(scale + 1.0)?)?
        .broadcast_add(shift)?)
}

fn bench(dev: &Device, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    dev.synchronize()?;
    let mut best = f64::INFINITY;
    for _ in 0..20 {
        let t0 = std::time::Instant::now();
        f()?;
        dev.synchronize()?;
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(best)
}

fn main() -> Result<()> {
    let dev = sd_tensor::device::best()?;
    println!("device {dev:?}\n");
    let mut rng = sd_tensor::rng::SeededRng::new(0);
    let (mut before, mut after) = (0.0, 0.0);

    // Flux-dev at 1024: 4096 image tokens, 512 text, 4608 concatenated.
    // Counts per forward: 38, 38, 39.
    // Batch 2 as well as 1: at batch 1 a narrow off the last axis is still
    // contiguous (the leading dims are 1), so only batch 2 exercises the
    // strided conditioning the wrapper has to copy.
    for (batch, tokens, n) in [
        (1usize, 4096usize, 38.0),
        (1, 512, 38.0),
        (1, 4608, 39.0),
        (2, 4096, 0.0),
    ] {
        let width = 3072;
        let xs = rng.randn((batch, tokens, width), &dev)?;
        // As the model builds them: one projection, narrowed into six.
        let proj = rng.randn((batch, 1, width * 6), &dev)?;
        let shift = proj.narrow(D::Minus1, 0, width)?;
        let scale = proj.narrow(D::Minus1, width, width)?;

        let want = composed(&xs, &scale, &shift, 1e-6)?;
        let got = fused::ada_layer_norm(&xs, &scale, &shift, 1e-6)?;
        assert_eq!(got.dims(), want.dims(), "shape");
        let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-5)?;
        let worst = sd_tensor::testing::max_abs_diff(&got, &want)?;

        let a = bench(&dev, || {
            composed(&xs, &scale, &shift, 1e-6)?;
            Ok(())
        })?;
        let b = bench(&dev, || {
            fused::ada_layer_norm(&xs, &scale, &shift, 1e-6)?;
            Ok(())
        })?;
        println!(
            "b{batch} {tokens:>4} tokens x{n:<5} composed {a:>6.3} ms  ours {b:>6.3} ms  {:.2}x   \
             max_abs {worst:.3e}  excess {excess:.3e}{}",
            a / b,
            if scale.is_contiguous() {
                ""
            } else {
                "  [strided cond]"
            }
        );
        before += a * n;
        after += b * n;
    }

    println!(
        "\nper Flux-dev forward: {before:.1} ms -> {after:.1} ms  (saves {:.1} ms, {:.2}x)",
        before - after,
        before / after
    );

    // The CPU fallback path must agree too.
    let cpu = Device::Cpu;
    let mut rng = sd_tensor::rng::SeededRng::new(2);
    let xs = rng.randn((2, 5, 64), &cpu)?;
    let scale = rng.randn((2, 1, 64), &cpu)?;
    let shift = rng.randn((2, 1, 64), &cpu)?;
    let want = composed(&xs, &scale, &shift, 1e-6)?;
    let got = xs.apply_op3_no_bwd(&scale, &shift, &fused::AdaLayerNorm { eps: 1e-6 })?;
    println!(
        "cpu kernel vs composition: max_abs {:.3e}",
        sd_tensor::testing::max_abs_diff(&got, &want)?
    );
    Ok(())
}
