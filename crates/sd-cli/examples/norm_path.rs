//! What do the two norms this project owns cost, against candle's fused ones?
//!
//! `sd-tensor` implements `plain_layer_norm` and `rms_norm` as compositions —
//! mean, subtract, square, mean, add, sqrt, divide, plus two dtype casts. That
//! is eight or nine passes over the tensor. candle ships `ops::layer_norm` and
//! `ops::rms_norm` as single fused kernels.
//!
//! This matters beyond these two functions. The strategy question is whether
//! to own more of the tensor layer over time, and the answer plausibly differs
//! by backend: on CPU a composition and a kernel are both memory-bound loops,
//! while on a GPU the difference is one dispatch against nine.
//!
//! Both norms run in **every block of Flux and SD 3.5**, twice or more, so
//! whatever this measures is multiplied by roughly a hundred per step.
//!
//! ```bash
//! cargo run --release -p sd-cli --features metal --example norm_path
//! ```

use anyhow::Result;
use sd_tensor::{ops, DType, Device, Tensor};

fn bench(
    label: &str,
    dev: &Device,
    iters: usize,
    mut f: impl FnMut() -> Result<()>,
) -> Result<f64> {
    f()?;
    dev.synchronize()?;
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        f()?;
        dev.synchronize()?;
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    println!("  {label:<26} {best:>8.3} ms");
    Ok(best)
}

fn main() -> Result<()> {
    let dev = sd_tensor::device::best()?;
    println!("device {dev:?}\n");

    // Flux at 512 and 1024: 1536 and 4608 tokens, 3072 wide.
    for (tokens, width) in [(1536usize, 3072usize), (4608, 3072)] {
        println!("{tokens} tokens x {width}");
        let mut rng = sd_tensor::rng::SeededRng::new(0);
        let xs = rng.randn((1, tokens, width), &dev)?;
        let alpha = rng.randn((width,), &dev)?;
        let beta = Tensor::zeros((width,), DType::F32, &dev)?;
        let ones = Tensor::ones((width,), DType::F32, &dev)?;

        // Agreement first. `plain_layer_norm` has no affine parameters, so
        // candle's equivalent is the same call with unit scale and zero shift.
        let ours = ops::plain_layer_norm(&xs, 1e-6)?;
        let theirs = ops::fused_layer_norm(&xs, &ones, &beta, 1e-6)?;
        println!(
            "  layer_norm agree to {:.3e}",
            sd_tensor::testing::allclose_excess(&theirs, &ours, 1e-4)?
        );
        let ours_r = ops::rms_norm(&xs, &alpha, 1e-6)?;
        let theirs_r = ops::fused_rms_norm(&xs, &alpha, 1e-6)?;
        println!(
            "  rms_norm   agree to {:.3e}",
            sd_tensor::testing::allclose_excess(&theirs_r, &ours_r, 1e-4)?
        );

        let a = bench("ours: plain_layer_norm", &dev, 10, || {
            ops::plain_layer_norm(&xs, 1e-6)?;
            Ok(())
        })?;
        // Allocating the unit scale and zero shift *inside* the timed
        // region, because `plain_layer_norm` takes no affine parameters and a
        // drop-in replacement would have to make them per call.
        let b = bench("candle: layer_norm (+alloc)", &dev, 10, || {
            let o = Tensor::ones((width,), DType::F32, &dev)?;
            let z = Tensor::zeros((width,), DType::F32, &dev)?;
            ops::fused_layer_norm(&xs, &o, &z, 1e-6)?;
            Ok(())
        })?;
        let c = bench("ours: rms_norm", &dev, 10, || {
            ops::rms_norm(&xs, &alpha, 1e-6)?;
            Ok(())
        })?;
        let d = bench("candle: rms_norm", &dev, 10, || {
            ops::fused_rms_norm(&xs, &alpha, 1e-6)?;
            Ok(())
        })?;
        println!(
            "  fused layer_norm {:.2}x, fused rms_norm {:.2}x\n",
            a / b,
            c / d
        );
    }
    Ok(())
}
