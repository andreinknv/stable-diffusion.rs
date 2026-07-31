//! A 1x1 convolution is a matmul. candle runs it through im2col anyway.
//!
//! `candle_core`'s Metal `conv2d` calls `im2col` then `matmul`, with no
//! special case for a 1x1 kernel at stride 1 and no padding — where the im2col
//! buffer is just the input, copied. That copy is written and read back for
//! nothing.
//!
//! Contracting the channel axis directly needs no transpose: `[b, c, h, w]`
//! reshapes to `[b, c, hw]`, and the kernel `[co, ci, 1, 1]` to `[co, ci]`.
//!
//! 37 of the 98 convolutions in an SD 1.5 forward are 1x1.

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

/// `[b, ci, h, w]` x `[co, ci]` -> `[b, co, h, w]`, as one matmul.
fn conv1x1(xs: &Tensor, k: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (b, ci, h, w) = xs.dims4()?;
    let co = k.dim(0)?;
    let flat = xs.reshape((b, ci, h * w))?;
    let k2 = k.reshape((co, ci))?;
    let out = k2.broadcast_matmul(&flat)?.reshape((b, co, h, w))?;
    match bias {
        Some(bs) => Ok(out.broadcast_add(&bs.reshape((1, co, 1, 1))?)?),
        None => Ok(out),
    }
}

fn main() -> Result<()> {
    let dev = sd_tensor::device::best()?;
    println!("device {dev:?}\n");
    let mut rng = sd_tensor::rng::SeededRng::new(0);
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let (mut before, mut after) = (0.0, 0.0);

    for line in include_str!("sd15_inventory.txt").lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.first() != Some(&"conv") || f[3] != "1" {
            continue;
        }
        let (co, ci, s, c): (usize, usize, usize, f64) = (
            f[1].parse()?,
            f[2].parse()?,
            f[4].parse()?,
            f[5].parse::<usize>()? as f64,
        );
        let xs = rng.randn((2, ci, s, s), &dev)?;
        let conv = nn::conv2d(
            ci,
            co,
            1,
            Default::default(),
            vb.pp(format!("k{co}_{ci}_{s}")),
        )?;
        let k = rng.randn((co, ci, 1, 1), &dev)?;

        // Agreement, against candle's own conv with the same weights.
        let vb2 = VarBuilder::from_tensors(
            [("weight".to_string(), k.clone())].into_iter().collect(),
            DType::F32,
            &dev,
        );
        let same = nn::conv2d_no_bias(ci, co, 1, Default::default(), vb2)?;
        let want = same.forward(&xs)?;
        let got = conv1x1(&xs, &k, None)?;
        let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-4)?;

        let a = bench(&dev, || {
            conv.forward(&xs)?;
            Ok(())
        })?;
        let b = bench(&dev, || {
            conv1x1(&xs, &k, None)?;
            Ok(())
        })?;
        println!(
            "[2,{ci:>4},{s:>2},{s:>2}] -> {co:>4}  x{c:<3.0}  im2col {a:>6.3} ms   matmul {b:>6.3} ms   \
             {:>5.2}x   excess {excess:.2e}",
            a / b
        );
        before += a * c;
        after += b * c;
    }
    println!(
        "\n1x1 convs per SD 1.5 forward: {before:.1} ms -> {after:.1} ms  (saves {:.1} ms, {:.2}x)",
        before - after,
        before / after
    );
    Ok(())
}
