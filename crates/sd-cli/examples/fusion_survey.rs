//! Which compositions in the hot paths are worth replacing with one kernel.
//!
//! Every entry here is a sequence this project runs today as several candle
//! ops, each of which reads its input from memory and writes its output back.
//! These are all memory-bound: the arithmetic is trivial and the cost is the
//! round trips. Fusing N ops into one kernel turns N reads and N writes into
//! one of each, so the ceiling on the win is roughly N-fold — and the floor is
//! nothing, if the op is small enough that dispatch dominates.
//!
//! Counts are per UNet forward at 512x512 with CFG (batch 2), from the model:
//! SD 1.5 has 22 resnets, each doing GroupNorm->SiLU twice, and 16 transformer
//! blocks, each doing one GEGLU.
//!
//! ```bash
//! cargo run --release -p sd-cli --features metal --example fusion_survey
//! ```

use anyhow::Result;
use sd_tensor::{nn, ops, DType, Device, Module, VarBuilder, D};

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

    // ---- GroupNorm -> SiLU, the resnet prologue ----------------------------
    // SD 1.5 at 512: four resolutions, 44 calls per forward across all of them.
    println!("GroupNorm -> SiLU  (44 calls per SD 1.5 forward)");
    let mut gn_total = 0.0;
    for (ch, hw, n) in [
        (320usize, 64usize, 18.0),
        (640, 32, 12.0),
        (1280, 16, 8.0),
        (1280, 8, 6.0),
    ] {
        let xs = rng.randn((2, ch, hw, hw), &dev)?;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let norm = nn::group_norm(32, ch, 1e-5, vb.pp("g"))?;
        let both = bench(&dev, || {
            ops::silu(&norm.forward(&xs)?)?;
            Ok(())
        })?;
        let norm_only = bench(&dev, || {
            norm.forward(&xs)?;
            Ok(())
        })?;
        println!(
            "  [2,{ch},{hw},{hw}]  norm {norm_only:>6.3}  +silu {both:>6.3}  \
             silu adds {:>6.3} ms  x{n}",
            both - norm_only
        );
        gn_total += (both - norm_only) * n;
    }
    println!("  -> fusing silu into the norm saves {gn_total:.1} ms per forward\n");

    // ---- GEGLU, the transformer feed-forward ------------------------------
    // Two narrows, an erf gelu and a multiply over `inner`, which is 4x the
    // model width. The narrows are views, so the passes are gelu and multiply.
    println!("GEGLU  (16 calls per SD 1.5 forward)");
    let mut ff_total = 0.0;
    for (seq, dim, n) in [
        (4096usize, 320usize, 6.0),
        (1024, 640, 6.0),
        (256, 1280, 3.0),
        (64, 1280, 1.0),
    ] {
        let inner = dim * 4;
        let h = rng.randn((2, seq, inner * 2), &dev)?;
        let composed = bench(&dev, || {
            let hidden = h.narrow(D::Minus1, 0, inner)?;
            let gate = h.narrow(D::Minus1, inner, inner)?;
            (hidden * ops::gelu(&gate)?)?;
            Ok(())
        })?;
        // The floor: one pass that reads 2*inner and writes inner.
        let floor = bench(&dev, || {
            h.narrow(D::Minus1, 0, inner)?.contiguous()?;
            Ok(())
        })?;
        println!(
            "  seq {seq:>4} dim {dim:>4}  composed {composed:>6.3}  one-pass floor {floor:>6.3}  \
             headroom {:>6.3} ms  x{n}",
            composed - floor
        );
        ff_total += (composed - floor) * n;
    }
    println!("  -> a fused GEGLU could save up to {ff_total:.1} ms per forward\n");

    // ---- adaLN, the DiT conditioning path ---------------------------------
    // Estimated only: no Flux or SD 3 checkpoint is present on this machine.
    println!("adaLN: norm -> modulate  (115 calls per Flux-dev forward, estimated)");
    let mut ada_total = 0.0;
    for (tokens, n) in [(4096usize, 38.0), (512, 38.0), (4608, 39.0)] {
        let xs = rng.randn((1, tokens, 3072), &dev)?;
        let scale = rng.randn((1, 1, 3072), &dev)?;
        let shift = rng.randn((1, 1, 3072), &dev)?;
        let both = bench(&dev, || {
            let hnorm = ops::plain_layer_norm(&xs, 1e-6)?;
            hnorm
                .broadcast_mul(&(&scale + 1.0)?)?
                .broadcast_add(&shift)?;
            Ok(())
        })?;
        let norm_only = bench(&dev, || {
            ops::plain_layer_norm(&xs, 1e-6)?;
            Ok(())
        })?;
        println!(
            "  {tokens:>4} tokens  norm {norm_only:>6.3}  +modulate {both:>6.3}  \
             modulate adds {:>6.3} ms  x{n}",
            both - norm_only
        );
        ada_total += (both - norm_only) * n;
    }
    println!("  -> fusing modulate into the norm saves {ada_total:.1} ms per step\n");

    println!("ranked, per forward:");
    println!("  adaLN (Flux, estimated)  {ada_total:>7.1} ms");
    println!("  GEGLU (SD 1.5)           {ff_total:>7.1} ms");
    println!("  GroupNorm+SiLU (SD 1.5)  {gn_total:>7.1} ms");
    Ok(())
}
