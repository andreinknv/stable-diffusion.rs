//! Does candle's fused attention take a causal mask, and is it faster?
//!
//! The roadmap has carried "broaden fused attention to SD 1.5 by materialising
//! `causal_mask` from `[1,1,s,s]` to `[b,h,s,s]` — a reshape, not a kernel".
//! The premise is that candle's Metal `sdpa` declines the broadcast form.
//!
//! `attention_with_path` already offers whatever mask it is given to `sdpa`
//! and falls back when that errors, so this needs no new code to answer — only
//! a run that reports which path each form takes, and what each costs.
//!
//! The masked attention in the SD family is **CLIP's text encoder**: batch 2
//! for guidance, 12 heads of 64, 77 tokens. SD 1.5's own spatial attention is
//! unmasked and already fused.

use anyhow::Result;
use sd_tensor::{ops, DType, Device, Tensor};

fn run(label: &str, dev: &Device, q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<()> {
    let (_, path) = ops::attention_with_path(q, k, v, Some(mask))?;
    dev.synchronize()?;

    // Minimum of 20, synchronising inside the timed region: these are small
    // enough that enqueue would otherwise be most of what is measured.
    let mut best = f64::INFINITY;
    for _ in 0..20 {
        let t0 = std::time::Instant::now();
        ops::attention_with_path(q, k, v, Some(mask))?;
        dev.synchronize()?;
        best = best.min(t0.elapsed().as_secs_f64() * 1e6);
    }
    println!("  {label:<24} {path:<10?} {best:>8.1} us");
    Ok(())
}

fn main() -> Result<()> {
    let dev = sd_tensor::device::best()?;
    println!("device {dev:?}\n");

    // Two masked shapes exist in this project. CLIP's text tower runs a dozen
    // times per generation; **the unCLIP prior runs 20 blocks x 25 steps**,
    // which is 500 masked attentions and the only place this could matter end
    // to end.
    for (name, b, heads, seq, dim) in [
        ("CLIP-L text tower", 2usize, 12usize, 77usize, 64usize),
        ("unCLIP prior", 2, 32, 81, 64),
    ] {
        measure(name, &dev, b, heads, seq, dim)?;
    }
    Ok(())
}

fn measure(name: &str, dev: &Device, b: usize, heads: usize, seq: usize, dim: usize) -> Result<()> {
    let dev = dev.clone();
    let dev = &dev;
    let mut rng = sd_tensor::rng::SeededRng::new(0);
    let q = rng.randn((b, heads, seq, dim), &dev)?;
    let k = rng.randn((b, heads, seq, dim), &dev)?;
    let v = rng.randn((b, heads, seq, dim), &dev)?;

    let broadcast = ops::causal_mask(seq, &dev)?.to_dtype(DType::F32)?;
    let materialised = broadcast.broadcast_as((b, heads, seq, seq))?.contiguous()?;

    println!("{name}, batch {b}, {heads} heads x {dim}, {seq} tokens");
    run("mask [1,1,s,s]", &dev, &q, &k, &v, &broadcast)?;
    run("mask [b,h,s,s]", &dev, &q, &k, &v, &materialised)?;

    // The masks must agree, or a faster path is just a different answer.
    let a = ops::attention_with_path(&q, &k, &v, Some(&broadcast))?.0;
    let c = ops::attention_with_path(&q, &k, &v, Some(&materialised))?.0;
    let excess = sd_tensor::testing::allclose_excess(&c, &a, 1e-4)?;
    println!("  the two masks agree to {excess:.3e}\n");
    Ok(())
}
