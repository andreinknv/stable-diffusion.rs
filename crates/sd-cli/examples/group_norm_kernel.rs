//! Our group norm kernel against candle's composition.
//!
//! The shapes and counts are SD 1.5's, taken from the checkpoint — the same
//! inventory `step_profile` uses, where group norm is 23.5% of a step.

use anyhow::Result;
use sd_tensor::{nn, DType, Device, Module, Tensor, VarBuilder};

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

    for line in include_str!("sd15_inventory.txt").lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.first() != Some(&"gnorm") {
            continue;
        }
        let (ch, s, count): (usize, usize, f64) =
            (f[1].parse()?, f[2].parse()?, f[3].parse::<usize>()? as f64);
        let groups = 32.min(ch);
        let xs = rng.randn((2, ch, s, s), &dev)?;
        let weight = rng.randn((ch,), &dev)?;
        let bias = rng.randn((ch,), &dev)?;

        let ours = nn::GroupNorm::new(weight.clone(), bias.clone(), ch, groups, 1e-5)?;
        let theirs = sd_tensor::fused::candle_group_norm(&weight, &bias, ch, groups, 1e-5)?;

        let got = ours.forward(&xs)?;
        let want = theirs.forward(&xs)?;
        let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-4)?;

        let a = bench(&dev, || {
            theirs.forward(&xs)?;
            Ok(())
        })?;
        let b = bench(&dev, || {
            ours.forward(&xs)?;
            Ok(())
        })?;
        let bytes = 2.0 * 2.0 * (ch * s * s) as f64 * 4.0;
        println!(
            "[2,{ch:>4},{s:>2},{s:>2}] x{count:<3.0}  candle {a:>6.3} ms ({:>5.1} GB/s)  \
             ours {b:>6.3} ms ({:>5.1} GB/s)  {:>5.2}x  excess {excess:.2e}",
            bytes / (a * 1e-3) / 1e9,
            bytes / (b * 1e-3) / 1e9,
            a / b
        );
        before += a * count;
        after += b * count;
    }

    println!(
        "\nper SD 1.5 forward: {before:.1} ms -> {after:.1} ms  (saves {:.1} ms, {:.2}x)",
        before - after,
        before / after
    );
    let _ = (
        DType::F32,
        Tensor::zeros(1usize, DType::F32, &Device::Cpu)?,
        VarBuilder::zeros(DType::F32, &dev),
    );
    Ok(())
}
