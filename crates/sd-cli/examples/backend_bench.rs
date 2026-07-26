//! Backend benchmark: runs the SD 1.5 VAE decoder and times it.
//!
//! ```bash
//! cargo run --release -p sd-cli --example backend_bench -- 64
//! cargo run --release -p sd-cli --features metal --example backend_bench -- 64
//! ```
//!
//! The argument is the latent edge length; the image is 8x that. Attention
//! sequence length is `n*n`, and our attention is O(seq^2) in memory, which is
//! the whole point of this benchmark — see docs/backends.md.
//!
//! Because that makes memory grow as `n^4`, an oversized run is refused rather
//! than left to the operating system. The refusal itself lives in the seam
//! (`sd_tensor::ops::check_attention_budget`) so that it covers every caller,
//! not just this file; what this file adds is doing the check *up front*, so a
//! bad size costs nothing instead of failing partway through a decode.
//!
//! Always discard the first call: candle compiles Metal pipelines lazily, so
//! it includes shader compilation and is not representative.

use anyhow::{bail, Result};
use sd_tensor::nn::{VarBuilder, VarMap};
use sd_tensor::ops::{
    attention_chunk_rows, attention_score_bytes, check_alloc_budget, human_bytes,
    DEFAULT_ATTENTION_CHUNK_BYTES,
};
use sd_tensor::{device, DType, Tensor};
use stable_diffusion_rs::models::vae::{Decoder, DecoderConfig};

/// Latent edge used when the argument is absent.
const DEFAULT_LATENT_EDGE: u64 = 64;

/// Parse the latent edge, rejecting a malformed argument rather than silently
/// benchmarking a different size than the one that was asked for.
fn latent_edge(arg: Option<&str>) -> Result<usize> {
    let Some(arg) = arg else {
        return Ok(DEFAULT_LATENT_EDGE as usize);
    };
    let n: u64 = arg
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("latent edge must be a positive integer, got {arg:?}"))?;
    if n == 0 {
        bail!("latent edge must be greater than zero");
    }
    // The budget check is what refuses oversized runs; this only keeps the
    // `seq = n * n` arithmetic from overflowing before it gets there.
    usize::try_from(n)
        .ok()
        .filter(|n| n.checked_mul(*n).is_some())
        .ok_or_else(|| anyhow::anyhow!("latent {n}x{n} overflows an attention sequence length"))
}

fn main() -> Result<()> {
    let arg = std::env::args().nth(1);
    let n = latent_edge(arg.as_deref())?;
    let seq = n * n;

    // SD 1.5 geometry, 64x64 latent -> 512x512 image.
    let cfg = DecoderConfig {
        latent_channels: 4,
        out_channels: 3,
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        norm_num_groups: 32,
        norm_eps: 1e-6,
    };

    // Cost the run before building anything, so a refusal is free. The gate is
    // the peak activation, not the score matrix: chunked attention bounds the
    // latter, which leaves a full-resolution up block as the biggest single
    // allocation. `Decoder::forward` checks this too — doing it here as well
    // only buys a cheaper refusal.
    check_alloc_budget(
        cfg.peak_activation_bytes(1, n, n, DType::F32),
        &format!("VAE decode activation for a {n}x{n} latent"),
    )?;

    let dev = device::best()?;
    println!("device: {dev:?}");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let d = Decoder::new(&cfg, vb)?;

    // The VAE attention block is single-head over a batch of one.
    let unchunked = attention_score_bytes(1, 1, seq, seq, DType::F32);
    let chunk_rows = attention_chunk_rows(1, 1, seq, DType::F32, DEFAULT_ATTENTION_CHUNK_BYTES);
    println!(
        "latent {n}x{n} -> image {}x{}  (attention seq = {seq}, score matrix = {}{})",
        n * 8,
        n * 8,
        unchunked.map_or_else(|| "overflow".to_string(), human_bytes),
        if chunk_rows >= seq {
            String::new()
        } else {
            format!(
                ", chunked to {} rows = {}",
                chunk_rows,
                attention_score_bytes(1, 1, chunk_rows, seq, DType::F32)
                    .map_or_else(|| "overflow".to_string(), human_bytes)
            )
        },
    );
    let z = Tensor::zeros((1, 4, n, n), DType::F32, &dev)?;

    // Warm-up: triggers Metal pipeline compilation and any lazy init.
    let warm = std::time::Instant::now();
    let out = d.forward(&z)?;
    let _ = out.flatten_all()?.to_vec1::<f32>()?;
    println!("warm-up (incl. shader compilation): {:?}", warm.elapsed());

    let mut best = std::time::Duration::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let out = d.forward(&z)?;
        let _ = out.flatten_all()?.to_vec1::<f32>()?; // force sync to host
        best = best.min(t.elapsed());
    }
    println!("best of 3 (steady state):         {best:?}");
    println!("output: {:?}", out.dims());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_malformed_argument_is_rejected_not_silently_defaulted() {
        // Falling back to 64 would benchmark a different size than the one
        // asked for and report it as if it were the requested run.
        assert!(latent_edge(Some("64x")).is_err());
        assert!(latent_edge(Some("-1")).is_err());
        assert!(latent_edge(Some("0")).is_err());
        assert_eq!(latent_edge(None).unwrap(), DEFAULT_LATENT_EDGE as usize);
        assert_eq!(latent_edge(Some(" 32 ")).unwrap(), 32);
    }

    #[test]
    fn an_edge_whose_square_overflows_is_rejected() {
        assert!(latent_edge(Some(&u64::MAX.to_string())).is_err());
    }
}
