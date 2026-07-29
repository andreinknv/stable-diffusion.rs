//! Is candle's fused norm less accurate on Metal too, or only on CPU?
//!
//! Commit e5e372a measured candle's fused `rms_norm` on CPU at 10x the error
//! of the composition in `sd-tensor`, because that kernel sums each row with a
//! sequential `.sum::<f32>()` where `mean_keepdim` reduces in blocks, and
//! error in a sequential sum grows with row length. Swapping T5 onto it moves
//! `golden_t5` to 3.891e-3, past a 3e-3 bound.
//!
//! That measurement was CPU-only, and the two backends do not share a
//! reduction. A GPU reduces across a threadgroup in a tree, which is *more*
//! accurate than sequential, not less — so the CPU result does not settle the
//! Metal case, and Metal is where generation runs.
//!
//! Truth here is f64, computed on CPU, which is the same standard the original
//! commit used. Both f32 answers are compared against it.
//!
//! The reference is written out longhand rather than by calling the helpers at
//! f64: **both helpers cast to f32 internally**, so passing them an f64 tensor
//! returns an f32 answer in an f64 container. Doing that compares the
//! composition against itself and reports a relative error of exactly zero.

use anyhow::Result;
use sd_tensor::{ops, DType, Device, Tensor};

fn rel_err(got: &Tensor, truth64: &Tensor) -> Result<f64> {
    let got = got.to_device(&Device::Cpu)?.to_dtype(DType::F64)?;
    let diff = (&got - truth64)?.abs()?.max_all()?.to_scalar::<f64>()?;
    let scale = truth64.abs()?.max_all()?.to_scalar::<f64>()?;
    Ok(diff / scale)
}

fn main() -> Result<()> {
    // The rows that matter: T5-XXL is 4096 wide, Flux and SD 3.5 are 3072,
    // CLIP-L is 768. Row length is the variable the sequential sum is bad at.
    for (tokens, width) in [(154usize, 4096usize), (1536, 3072), (77, 768)] {
        println!("[1, {tokens}, {width}]");

        let mut rng = sd_tensor::rng::SeededRng::new(0);
        let xs_cpu = rng.randn((1, tokens, width), &Device::Cpu)?;
        let alpha_cpu = rng.randn((width,), &Device::Cpu)?;

        // f64 truth, on CPU, from the same numbers — longhand, so that no
        // step of it passes through f32.
        let x64 = xs_cpu.to_dtype(DType::F64)?;
        let a64 = alpha_cpu.to_dtype(DType::F64)?;
        let rrms = (x64.sqr()?.mean_keepdim(sd_tensor::D::Minus1)? + 1e-6)?.sqrt()?;
        let truth_rms = x64.broadcast_div(&rrms)?.broadcast_mul(&a64)?;
        let mean = x64.mean_keepdim(sd_tensor::D::Minus1)?;
        let centred = x64.broadcast_sub(&mean)?;
        let var = centred.sqr()?.mean_keepdim(sd_tensor::D::Minus1)?;
        let truth_ln = centred.broadcast_div(&(var + 1e-6)?.sqrt()?)?;

        for dev in [Device::Cpu, sd_tensor::device::best()?] {
            let xs = xs_cpu.to_device(&dev)?;
            let alpha = alpha_cpu.to_device(&dev)?;
            let ones = Tensor::ones(width, DType::F32, &dev)?;
            let zeros = Tensor::zeros(width, DType::F32, &dev)?;

            let ours = ops::rms_norm(&xs, &alpha, 1e-6)?;
            let theirs = ops::fused_rms_norm(&xs, &alpha, 1e-6)?;
            let ours_l = ops::plain_layer_norm(&xs, 1e-6)?;
            let theirs_l = ops::fused_layer_norm(&xs, &ones, &zeros, 1e-6)?;

            let tag = if matches!(dev, Device::Cpu) {
                "cpu "
            } else {
                "gpu "
            };
            println!(
                "  {tag} rms    ours {:.3e}   candle {:.3e}   ratio {:.1}x",
                rel_err(&ours, &truth_rms)?,
                rel_err(&theirs, &truth_rms)?,
                rel_err(&theirs, &truth_rms)? / rel_err(&ours, &truth_rms)?.max(1e-12)
            );
            println!(
                "  {tag} layer  ours {:.3e}   candle {:.3e}   ratio {:.1}x",
                rel_err(&ours_l, &truth_ln)?,
                rel_err(&theirs_l, &truth_ln)?,
                rel_err(&theirs_l, &truth_ln)? / rel_err(&ours_l, &truth_ln)?.max(1e-12)
            );
        }
        println!();
    }
    Ok(())
}
