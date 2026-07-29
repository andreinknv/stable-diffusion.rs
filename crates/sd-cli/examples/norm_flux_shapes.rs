//! The norm shapes Flux-dev actually runs, and what the seam's split is worth.
//!
//! Counted from `flux/mod.rs`. A double block holds four `PlainLayerNorm`
//! (`img_norm1/2`, `txt_norm1/2`) and four `RmsNorm` (a QK norm of two, per
//! stream); a single block holds one and two; there is one final layer norm.
//! At dev's 19 double and 38 single blocks that is 115 layer norms and 152 QK
//! norms per forward.
//!
//! The counts alone would overstate the case, because the two streams are not
//! the same length. In a double block the image stream is 4096 tokens at
//! 1024x1024 and the **text stream is 512**; single blocks run on the two
//! concatenated, 4608. So the calls are weighted by where they actually run:
//!
//! ```text
//!   layer norm   38 @ 4096    38 @ 512    39 @ 4608
//!   QK norm      38 @ 4096    38 @ 512    76 @ 4608
//! ```
//!
//! The two norms also do not share a regime. Layer norms reduce rows of 3072;
//! QK norms run after `split_qkv`, so they reduce rows of 128 — 24x shorter,
//! and row length is the variable that decides whether candle's reduction is
//! worth using.
//!
//! This is an estimate assembled from per-call measurements and counted call
//! sites, not an end-to-end run: there is no Flux checkpoint on this machine,
//! and instantiating dev at f32 would ask for about 48 GB.

use anyhow::Result;
use sd_tensor::{ops, DType, Device, Tensor, D};

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
    let (heads, head_dim, width) = (24usize, 128usize, 3072usize);
    let mut rng = sd_tensor::rng::SeededRng::new(0);

    // (tokens, layer-norm calls, QK-norm calls)
    let work = [
        (4096usize, 38.0, 38.0),
        (512, 38.0, 38.0),
        (4608, 39.0, 76.0),
    ];
    let (mut saved_ln, mut saved_qk) = (0.0, 0.0);

    for (tokens, n_ln, n_qk) in work {
        let xs = rng.randn((1, tokens, width), &dev)?;
        let seam = bench(&dev, || {
            ops::plain_layer_norm(&xs, 1e-6)?;
            Ok(())
        })?;
        let comp = bench(&dev, || {
            let mean = xs.mean_keepdim(D::Minus1)?;
            let centred = xs.broadcast_sub(&mean)?;
            let var = centred.sqr()?.mean_keepdim(D::Minus1)?;
            centred.broadcast_div(&(var + 1e-6)?.sqrt()?)?;
            Ok(())
        })?;

        let q = rng.randn((1, heads, tokens, head_dim), &dev)?;
        let scale = rng.randn((head_dim,), &dev)?;
        let seam_q = bench(&dev, || {
            ops::rms_norm(&q, &scale, 1e-6)?;
            Ok(())
        })?;
        let comp_q = bench(&dev, || {
            let rrms = (q.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
            q.broadcast_div(&rrms)?.broadcast_mul(&scale)?;
            Ok(())
        })?;

        println!("{tokens} tokens");
        println!("  layer norm  composition {comp:>6.3} ms   seam {seam:>6.3} ms   x{n_ln}");
        println!("  QK norm     composition {comp_q:>6.3} ms   seam {seam_q:>6.3} ms   x{n_qk}");
        saved_ln += (comp - seam) * n_ln;
        saved_qk += (comp_q - seam_q) * n_qk;

        // Agreement, at every shape, or the saving is a different answer.
        let a = ops::rms_norm(&q, &scale, 1e-6)?;
        let b = ops::fused_rms_norm(&q, &scale, 1e-6)?;
        println!(
            "  agree to {:.3e}\n",
            sd_tensor::testing::allclose_excess(&b, &a, 1e-4)?
        );
        let _ = (DType::F32, Tensor::ones(1usize, DType::F32, &dev)?);
    }

    println!("estimated saving per Flux-dev forward at 1024x1024:");
    println!("  layer norms {saved_ln:>8.1} ms");
    println!("  QK norms    {saved_qk:>8.1} ms");
    println!("  total       {:>8.1} ms per step", saved_ln + saved_qk);
    Ok(())
}
