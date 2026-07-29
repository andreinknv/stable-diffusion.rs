//! Does candle's fused norm keep the f32 accumulation, and does it win on CPU?
//!
//! `ops::rms_norm` casts to f32 before reducing whatever the input dtype, and
//! `t5/mod.rs` records that as deliberate: a bf16 sum of squares over 4096
//! channels loses the small terms. Replacing the composition with candle's
//! fused kernel is only safe if the kernel does the same thing.
//!
//! Checked by comparing each backend's half-precision result against the *f32*
//! answer, which is the standard both are trying to approximate. A kernel that
//! accumulates in bf16 will show a visibly larger error than one that does not.

use anyhow::Result;
use sd_tensor::{ops, DType, Device, Tensor};

fn main() -> Result<()> {
    for dev in [Device::Cpu, sd_tensor::device::best()?] {
        println!("=== {dev:?} ===");
        let (tokens, width) = (512usize, 4096usize);
        let mut rng = sd_tensor::rng::SeededRng::new(0);
        let xs = rng.randn((1, tokens, width), &dev)?;
        let alpha = rng.randn((width,), &dev)?;
        let ones = Tensor::ones((width,), DType::F32, &dev)?;
        let zeros = Tensor::zeros((width,), DType::F32, &dev)?;

        // The f32 answers both half-precision paths are approximating.
        let truth_r = ops::rms_norm(&xs, &alpha, 1e-6)?;
        let truth_l = ops::plain_layer_norm(&xs, 1e-6)?;

        for dt in [DType::F16, DType::BF16] {
            let x = xs.to_dtype(dt)?;
            let a = alpha.to_dtype(dt)?;
            let ours = ops::rms_norm(&x, &a, 1e-6)?.to_dtype(DType::F32)?;
            let theirs = ops::fused_rms_norm(&x, &a, 1e-6)?.to_dtype(DType::F32)?;
            let ours_l = ops::plain_layer_norm(&x, 1e-6)?.to_dtype(DType::F32)?;
            let theirs_l =
                ops::fused_layer_norm(&x, &ones.to_dtype(dt)?, &zeros.to_dtype(dt)?, 1e-6)?
                    .to_dtype(DType::F32)?;
            println!(
                "  {dt:?}  rms   ours {:.3e}  candle {:.3e}",
                sd_tensor::testing::max_abs_diff(&ours, &truth_r)?,
                sd_tensor::testing::max_abs_diff(&theirs, &truth_r)?
            );
            println!(
                "  {dt:?}  layer ours {:.3e}  candle {:.3e}",
                sd_tensor::testing::max_abs_diff(&ours_l, &truth_l)?,
                sd_tensor::testing::max_abs_diff(&theirs_l, &truth_l)?
            );
        }

        // And the speed question, on this backend.
        let best = |label: &str, mut f: Box<dyn FnMut() -> Result<()>>| -> Result<f64> {
            f()?;
            dev.synchronize()?;
            let mut b = f64::INFINITY;
            for _ in 0..10 {
                let t0 = std::time::Instant::now();
                f()?;
                dev.synchronize()?;
                b = b.min(t0.elapsed().as_secs_f64() * 1e3);
            }
            println!("  {label:<22} {b:>8.3} ms");
            Ok(b)
        };
        let (x1, a1) = (xs.clone(), alpha.clone());
        let o = best(
            "ours: rms_norm",
            Box::new(move || {
                ops::rms_norm(&x1, &a1, 1e-6)?;
                Ok(())
            }),
        )?;
        let (x2, a2) = (xs.clone(), alpha.clone());
        let t = best(
            "candle: rms_norm",
            Box::new(move || {
                ops::fused_rms_norm(&x2, &a2, 1e-6)?;
                Ok(())
            }),
        )?;
        println!("  fused is {:.2}x\n", o / t);
    }
    Ok(())
}
