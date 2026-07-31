//! Does our group-norm implementation beat candle's composition on CPU too?
//!
//! The Metal kernel is 6.09x, but it is gated on `is_metal()`, so CPU callers
//! still run the ten-op composition. Our `cpu_fwd` exists — it is the
//! reference the kernel is tested against — but has never been timed.
use anyhow::Result;
use sd_tensor::{fused, Device, Module, Tensor};

fn bench(mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        f()?;
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(best)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let mut rng = sd_tensor::rng::SeededRng::new(0);
    let (mut ours_t, mut theirs_t) = (0.0, 0.0);
    for line in include_str!("sd15_inventory.txt").lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.first() != Some(&"gnorm") {
            continue;
        }
        let (ch, s, c): (usize, usize, f64) =
            (f[1].parse()?, f[2].parse()?, f[3].parse::<usize>()? as f64);
        let groups = 32.min(ch);
        let xs = rng.randn((2, ch, s, s), &dev)?;
        let w = rng.randn((ch,), &dev)?;
        let b = rng.randn((ch,), &dev)?;
        let theirs = fused::candle_group_norm(&w, &b, ch, groups, 1e-5)?;
        let op = fused::GroupNormOp { groups, eps: 1e-5 };

        let want = theirs.forward(&xs)?;
        let got = xs.apply_op3_no_bwd(&w, &b, &op)?;
        let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-4)?;

        let a = bench(|| {
            theirs.forward(&xs)?;
            Ok(())
        })?;
        let bb = bench(|| {
            xs.apply_op3_no_bwd(&w, &b, &op)?;
            Ok(())
        })?;
        println!("[2,{ch:>4},{s:>2},{s:>2}] x{c:<3.0} candle {a:>7.3} ms  ours {bb:>7.3} ms  {:>5.2}x  excess {excess:.2e}", a / bb);
        theirs_t += a * c;
        ours_t += bb * c;
    }
    println!(
        "\nCPU, per SD 1.5 forward: candle {theirs_t:.1} ms, ours {ours_t:.1} ms  ({:.2}x)",
        theirs_t / ours_t
    );
    let _ = Tensor::zeros(1usize, sd_tensor::DType::F32, &dev)?;
    Ok(())
}
