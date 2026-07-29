//! Is candle's fused RoPE faster than Flux's own, on Flux's shapes?
//!
//! The roadmap has carried `candle_nn::rotary_emb::{rope, rope_i, rope_thd}`
//! as "worth checking" for a while. It exists; the shapes line up; that is not
//! the question. **The question is whether it is faster here**, and this
//! project has been wrong about exactly that before — CPU flash attention was
//! listed as "potentially the largest single win available" purely on the
//! strength of existing, and turned out to be 2x *slower* above 512 tokens.
//!
//! # What is compared
//!
//! Flux rotates interleaved adjacent pairs using an explicit 2x2 matrix per
//! frequency: reshape, two narrows, two broadcast multiplies, an add, a
//! reshape. candle's `rope_i` is one fused op over `cos`/`sin` vectors. The
//! 2x2's first column *is* `(cos, sin)`, so the two are the same function.
//!
//! The comparison assumes `cos`/`sin` are precomputed, because in a real
//! integration they would be built once per run exactly as the 2x2 is today.
//! Timing the extraction would measure the adapter, not the kernel.
//!
//! Interleaved minimum-of-N, per this project's benchmarking rule: noise on
//! this machine is one-sided, and a mean over back-to-back runs of the same
//! binary has reported figures 10x apart.

use anyhow::Result;
use stable_diffusion_rs as sd;

use sd::models::flux::rope;
use sd_tensor::{ops, DType, Tensor};

/// **Synchronises inside the timed region**, which is not optional on Metal.
///
/// candle queues GPU work and returns immediately, so a timer around the call
/// measures how long it took to *enqueue*. The first version of this benchmark
/// did exactly that and reported 14 million elements rotated in 9 microseconds
/// — 1.5 TB/s, which is the tell. Same family as the trap in handoff.md where
/// a Metal failure is attributed to whatever waits first.
fn bench(
    label: &str,
    dev: &sd_tensor::Device,
    iters: usize,
    mut f: impl FnMut() -> Result<()>,
) -> Result<f64> {
    // One warm-up: the first call pays allocation and, on Metal, pipeline
    // compilation.
    f()?;
    dev.synchronize()?;
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        f()?;
        dev.synchronize()?;
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    println!("  {label:<28} {best:>8.3} ms");
    Ok(best)
}

fn main() -> Result<()> {
    let dev = sd_tensor::device::best()?;
    println!("device {dev:?}\n");

    // Flux at 1024: 4096 image tokens plus 512 text, 24 heads of 128.
    for (seq, heads, head_dim) in [(4608usize, 24usize, 128usize), (1024, 24, 128)] {
        println!("seq {seq}, {heads} heads x {head_dim}");
        let mut rng = sd_tensor::rng::SeededRng::new(0);
        let q = rng.randn((1, heads, seq, head_dim), &dev)?;
        let k = rng.randn((1, heads, seq, head_dim), &dev)?;

        // Flux's own form: a 2x2 rotation per token per frequency.
        let ids = rope::image_ids(1, 64, 64, &dev)?;
        let ids = if seq == 4608 {
            let text = Tensor::zeros((1, 512, 3), DType::F32, &dev)?;
            Tensor::cat(&[&text, &ids], 1)?
        } else {
            ids.narrow(1, 0, seq)?
        };
        let freqs = rope::embed_nd(&ids, &[16, 56, 56], 10000.0)?;

        // The same rotation as cos/sin vectors: the 2x2's first column.
        let col0 = freqs.narrow(5, 0, 1)?.squeeze(5)?;
        let cos = col0.narrow(4, 0, 1)?.squeeze(4)?.squeeze(1)?.contiguous()?;
        let sin = col0.narrow(4, 1, 1)?.squeeze(4)?.squeeze(1)?.contiguous()?;

        // Correctness before speed: a faster wrong answer is not a win.
        let (ours, _) = rope::apply_rope(&q, &k, &freqs)?;
        let theirs = ops::rope_interleaved(&q.contiguous()?, &cos, &sin)?;
        let excess = sd_tensor::testing::allclose_excess(&theirs, &ours, 1e-4)?;
        println!("  agreement excess {excess:.3e}");

        let flux = bench("flux 2x2", &dev, 12, || {
            rope::apply_rope(&q, &k, &freqs)?;
            Ok(())
        })?;
        let qc = q.contiguous()?;
        let kc = k.contiguous()?;
        let fused = bench("candle rope_i", &dev, 12, || {
            ops::rope_interleaved(&qc, &cos, &sin)?;
            ops::rope_interleaved(&kc, &cos, &sin)?;
            Ok(())
        })?;
        let ratio = flux / fused;
        println!(
            "  fused is {ratio:.2}x {}\n",
            if ratio > 1.0 { "faster" } else { "SLOWER" }
        );
    }
    Ok(())
}
