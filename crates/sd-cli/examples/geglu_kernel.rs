//! Our own GEGLU kernel against the composition it replaces.
//!
//! Correctness first, at every shape SD 1.5 runs, then speed. A kernel that is
//! fast and slightly different is a bug with a benchmark attached.

use anyhow::Result;
use sd_tensor::{fused, ops, Device, Tensor, D};

fn composed(h: &Tensor, inner: usize) -> Result<Tensor> {
    let hidden = h.narrow(D::Minus1, 0, inner)?;
    let gate = h.narrow(D::Minus1, inner, inner)?;
    Ok((hidden * ops::gelu(&gate)?)?)
}

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
    let mut total_before = 0.0;
    let mut total_after = 0.0;

    for (seq, dim, n) in [
        (4096usize, 320usize, 6.0),
        (1024, 640, 6.0),
        (256, 1280, 3.0),
        (64, 1280, 1.0),
    ] {
        let inner = dim * 4;
        let h = rng.randn((2, seq, inner * 2), &dev)?;

        let want = composed(&h, inner)?;
        let got = fused::geglu(&h, inner)?;
        assert_eq!(got.dims(), want.dims(), "shape");
        let excess = sd_tensor::testing::allclose_excess(&got, &want, 1e-5)?;
        let worst = sd_tensor::testing::max_abs_diff(&got, &want)?;

        let a = bench(&dev, || {
            composed(&h, inner)?;
            Ok(())
        })?;
        let b = bench(&dev, || {
            fused::geglu(&h, inner)?;
            Ok(())
        })?;
        println!(
            "seq {seq:>4} dim {dim:>4} inner {inner:>4}  composed {a:>6.3} ms  ours {b:>6.3} ms  \
             {:.2}x   max_abs {worst:.3e}  excess {excess:.3e}",
            a / b
        );
        total_before += a * n;
        total_after += b * n;
    }

    println!(
        "\nper SD 1.5 forward: {total_before:.1} ms -> {total_after:.1} ms  \
         (saves {:.1} ms, {:.2}x on this op)",
        total_before - total_after,
        total_before / total_after
    );

    // The CPU fallback must agree too, or the two backends diverge.
    let cpu = Device::Cpu;
    let mut rng = sd_tensor::rng::SeededRng::new(1);
    let h = rng.randn((2, 8, 64), &cpu)?;
    let want = composed(&h, 32)?;
    let got = h.apply_op1_no_bwd(&fused::Geglu { inner: 32 })?;
    println!(
        "cpu fallback kernel vs composition: max_abs {:.3e}",
        sd_tensor::testing::max_abs_diff(&got, &want)?
    );
    Ok(())
}
