//! How much of candle's convolution is im2col, and how much is the gemm?
//!
//! candle's Metal `conv2d` materialises an im2col buffer and matmuls it. For a
//! 3x3 that buffer is nine times the input — 94 MB at [2,320,64,64] — written
//! and then read straight back.
//!
//! This bounds what any better convolution could win. The gemm underneath is
//! MLX's and is not going to be beaten; the im2col round trip is the part that
//! a direct convolution (Apple's MPSGraph, or our own) would avoid. If it is a
//! small share, there is nothing here worth binding a new framework for.
//!
//! The matmul timed here is the exact one candle performs: `[b, out_positions,
//! ci*k*k] x [ci*k*k, co]`.

use anyhow::Result;
use sd_tensor::{nn, DType, Device, Module, VarBuilder};

fn bench(dev: &Device, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    dev.synchronize()?;
    let mut best = f64::INFINITY;
    for _ in 0..10 {
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
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let (mut t_conv, mut t_gemm) = (0.0, 0.0);

    println!("{:<26}{:>9}{:>9}{:>9}{:>10}", "3x3 conv", "total", "gemm", "im2col", "im2col%");
    for line in include_str!("sd15_inventory.txt").lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.first() != Some(&"conv") || f[3] != "3" {
            continue;
        }
        let (co, ci, s, c): (usize, usize, usize, f64) =
            (f[1].parse()?, f[2].parse()?, f[4].parse()?, f[5].parse::<usize>()? as f64);

        let xs = rng.randn((2, ci, s, s), &dev)?;
        let conv = nn::conv2d(
            ci,
            co,
            3,
            nn::Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp(format!("k{co}_{ci}_{s}")),
        )?;
        // The gemm candle ends up doing: one row per output position, one
        // column per (input channel x kernel tap).
        let k = ci * 9;
        let col = rng.randn((2, s * s, k), &dev)?;
        let ker = rng.randn((k, co), &dev)?;

        let total = bench(&dev, || { conv.forward(&xs)?; Ok(()) })?;
        let gemm = bench(&dev, || { col.broadcast_matmul(&ker)?; Ok(()) })?;
        let im2col = (total - gemm).max(0.0);
        println!(
            "[2,{ci:>4},{s:>2},{s:>2}] -> {co:<5} x{c:<3.0}{total:>9.3}{gemm:>9.3}{im2col:>9.3}{:>9.0}%",
            100.0 * im2col / total
        );
        t_conv += total * c;
        t_gemm += gemm * c;
    }
    let t_im2col = t_conv - t_gemm;
    println!(
        "\n3x3 convs per SD 1.5 forward: {t_conv:.1} ms total = {t_gemm:.1} gemm + {t_im2col:.1} im2col",
    );
    println!(
        "  a perfect direct convolution could remove at most {:.0}% of it",
        100.0 * t_im2col / t_conv
    );
    Ok(())
}
