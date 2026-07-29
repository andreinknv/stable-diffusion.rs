//! Apple's convolution against candle's im2col, at SD 1.5's real 3x3 shapes.
//!
//! candle spends about 169 ms a forward materialising im2col buffers — up to
//! 283 MB per call, moved at ~25 GB/s. A direct convolution never builds them.
//! This is what Apple's own implementation, already installed on every Mac,
//! does with the same tensors.

use anyhow::Result;
use sd_tensor::{mps, nn, DType, Device, Module, VarBuilder};

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
    let (mut t_candle, mut t_mps) = (0.0, 0.0);

    for line in include_str!("sd15_inventory.txt").lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.first() != Some(&"conv") || f[3] != "3" {
            continue;
        }
        let (co, ci, s, c): (usize, usize, usize, f64) = (
            f[1].parse()?,
            f[2].parse()?,
            f[4].parse()?,
            f[5].parse::<usize>()? as f64,
        );
        let xs = rng.randn((2, ci, s, s), &dev)?;
        let k = rng.randn((co, ci, 3, 3), &dev)?;

        // candle, with the same weights and no bias.
        let vb = VarBuilder::from_tensors(
            [("weight".to_string(), k.clone())].into_iter().collect(),
            DType::F32,
            &dev,
        );
        let conv = nn::conv2d_no_bias(
            ci,
            co,
            3,
            nn::Conv2dConfig { padding: 1, ..Default::default() },
            vb,
        )?;

        let want = conv.forward(&xs)?;
        let got = mps::conv2d(&xs, &k, 1)?;
        if got.dims() != want.dims() {
            println!("[2,{ci},{s},{s}] -> {co}: SHAPE {:?} vs {:?}", got.dims(), want.dims());
            continue;
        }
        let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-3)?;

        let a = bench(&dev, || {
            conv.forward(&xs)?;
            Ok(())
        })?;
        let b = bench(&dev, || {
            mps::conv2d(&xs, &k, 1)?;
            Ok(())
        })?;
        println!(
            "[2,{ci:>4},{s:>2},{s:>2}] -> {co:<5} x{c:<3.0} candle {a:>7.3} ms   MPS {b:>7.3} ms   \
             {:>5.2}x   excess {excess:.2e}",
            a / b
        );
        t_candle += a * c;
        t_mps += b * c;
    }
    println!(
        "\n3x3 convs per SD 1.5 forward: candle {t_candle:.1} ms, MPSGraph {t_mps:.1} ms  ({:.2}x)",
        t_candle / t_mps
    );
    Ok(())
}
