//! Where does an SD 1.5 step actually go?
//!
//! The question this answers is strategic, not curious: if most of a step is
//! matmul and convolution then owning the elementwise kernels caps the upside,
//! however good those kernels get. And matmul is the one place candle is hard
//! to beat, because its gemm comes from Apple's MLX.
//!
//! The inventory in `sd15_inventory.txt` was extracted from the real SD 1.5
//! UNet checkpoint — every conv2d, linear, group norm and layer norm, with the
//! spatial size implied by which block it sits in — so these are the shapes
//! the model runs, not a guess at them. Batch 2 throughout, for guidance.
//!
//! Each line is timed at its shape and multiplied by how many times it occurs.
//! That ignores overlap between kernels, so the total will overshoot a real
//! step; the ratios are what this is for, and the overshoot is reported rather
//! than hidden.
//!
//! ```bash
//! cargo run --release -p sd-cli --features metal --example step_profile
//! ```

use anyhow::Result;
use sd_tensor::{nn, ops, DType, Device, Module, VarBuilder, D};

fn bench(dev: &Device, iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    dev.synchronize()?;
    let mut best = f64::INFINITY;
    for _ in 0..iters {
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
    let inventory = include_str!("sd15_inventory.txt");
    let mut rng = sd_tensor::rng::SeededRng::new(0);
    let vb = VarBuilder::zeros(DType::F32, &dev);

    let (mut t_conv, mut t_lin, mut t_gn, mut t_ln) = (0.0, 0.0, 0.0, 0.0);
    let (mut n_conv, mut n_lin, mut n_gn, mut n_ln) = (0, 0, 0, 0);
    // Bandwidth is the useful diagnostic: an op far below what the machine can
    // do has headroom, and one near it does not.
    let mut worst: Vec<(f64, String)> = Vec::new();

    for line in inventory.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.is_empty() {
            continue;
        }
        let num = |i: usize| f[i].parse::<usize>().expect("field");
        match f[0] {
            "conv" => {
                let (o, i, k, s, c) = (num(1), num(2), num(3), num(4), num(5));
                let xs = rng.randn((2, i, s, s), &dev)?;
                let conv = nn::conv2d(
                    i,
                    o,
                    k,
                    nn::Conv2dConfig {
                        padding: k / 2,
                        ..Default::default()
                    },
                    vb.pp(format!("c{o}_{i}_{k}_{s}")),
                )?;
                let ms = bench(&dev, 5, || {
                    conv.forward(&xs)?;
                    Ok(())
                })?;
                t_conv += ms * c as f64;
                n_conv += c;
            }
            "linear" => {
                let (o, i, q, c) = (num(1), num(2), num(3), num(4));
                let xs = rng.randn((2, q, i), &dev)?;
                let lin = nn::linear(i, o, vb.pp(format!("l{o}_{i}_{q}")))?;
                let ms = bench(&dev, 5, || {
                    lin.forward(&xs)?;
                    Ok(())
                })?;
                t_lin += ms * c as f64;
                n_lin += c;
            }
            "gnorm" => {
                let (ch, s, c) = (num(1), num(2), num(3));
                let xs = rng.randn((2, ch, s, s), &dev)?;
                let g = nn::group_norm(32.min(ch), ch, 1e-5, vb.pp(format!("g{ch}_{s}")))?;
                let ms = bench(&dev, 5, || {
                    g.forward(&xs)?;
                    Ok(())
                })?;
                t_gn += ms * c as f64;
                n_gn += c;
                let bytes = 2.0 * 2.0 * (ch * s * s) as f64 * 4.0;
                worst.push((
                    bytes / (ms * 1e-3) / 1e9,
                    format!("group_norm [2,{ch},{s},{s}]  {ms:.3} ms x{c}"),
                ));
            }
            "lnorm" => {
                let (ch, q, c) = (num(1), num(2), num(3));
                let xs = rng.randn((2, q, ch), &dev)?;
                let ms = bench(&dev, 5, || {
                    ops::plain_layer_norm(&xs, 1e-5)?;
                    Ok(())
                })?;
                t_ln += ms * c as f64;
                n_ln += c;
            }
            _ => {}
        }
    }

    let total = t_conv + t_lin + t_gn + t_ln;
    println!("per UNet forward, batch 2, 512x512:\n");
    println!("  {:<14} {:>8} {:>9} {:>8}", "op", "calls", "ms", "share");
    for (name, ms, n) in [
        ("matmul/linear", t_lin, n_lin),
        ("conv2d", t_conv, n_conv),
        ("group_norm", t_gn, n_gn),
        ("layer_norm", t_ln, n_ln),
    ] {
        println!("  {name:<14} {n:>8} {ms:>9.1} {:>7.1}%", 100.0 * ms / total);
    }
    println!(
        "  {:<14} {:>8} {total:>9.1}",
        "sum",
        n_conv + n_lin + n_gn + n_ln
    );

    println!("\ncandle's group_norm is a composition, and it shows:");
    worst.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
    for (gbs, label) in worst.iter().take(5) {
        println!("  {label:<44} {gbs:>6.1} GB/s");
    }
    let _ = D::Minus1;
    Ok(())
}
