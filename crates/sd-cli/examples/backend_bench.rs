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

    let repeats = repeat_count(std::env::var(REPEATS_ENV).ok().as_deref())?;
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t = std::time::Instant::now();
        let out = d.forward(&z)?;
        let _ = out.flatten_all()?.to_vec1::<f32>()?; // force sync to host
        samples.push(t.elapsed());
    }

    let stats = Stats::from(&mut samples);
    println!("steady state, {repeats} runs:");
    println!("  median: {:?}", stats.median);
    println!("  range:  {:?} .. {:?}", stats.min, stats.max);
    println!("  spread: {:.0}% of median", stats.spread_pct());
    if let Some(warning) = stats.reliability_warning() {
        println!("\n{warning}");
    }
    println!("output: {:?}", out.dims());
    Ok(())
}

/// Environment override for the number of timed runs.
const REPEATS_ENV: &str = "SD_BENCH_REPEATS";

/// Runs to time, after the warm-up.
///
/// Five, not three, and reported as a median rather than a minimum. A minimum
/// answers "how fast could this go on a quiet machine", which is the wrong
/// question when comparing two implementations on a machine you do not
/// control: it reports whichever configuration happened to catch the quietest
/// moment. The median plus the spread answers "what does this cost, and do I
/// believe the number".
const DEFAULT_REPEATS: usize = 5;

fn repeat_count(raw: Option<&str>) -> Result<usize> {
    let Some(raw) = raw else {
        return Ok(DEFAULT_REPEATS);
    };
    let n: usize = raw
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("{REPEATS_ENV} must be a positive integer, got {raw:?}"))?;
    if n == 0 {
        bail!("{REPEATS_ENV} must be at least 1");
    }
    Ok(n)
}

struct Stats {
    median: std::time::Duration,
    min: std::time::Duration,
    max: std::time::Duration,
}

impl Stats {
    /// Sorts `samples` in place and summarises them.
    fn from(samples: &mut [std::time::Duration]) -> Self {
        assert!(!samples.is_empty(), "at least one sample is required");
        samples.sort_unstable();
        Self {
            median: samples[samples.len() / 2],
            min: samples[0],
            max: samples[samples.len() - 1],
        }
    }

    /// Spread as a percentage of the median.
    fn spread_pct(&self) -> f64 {
        let median = self.median.as_secs_f64();
        if median == 0.0 {
            return 0.0;
        }
        (self.max.as_secs_f64() - self.min.as_secs_f64()) / median * 100.0
    }

    /// Refuse to let a noisy measurement be quoted as if it were a result.
    ///
    /// A chunk-size comparison on this benchmark once produced 9.1/11.8/9.6 s
    /// for one configuration and 17.3/12.3/8.0 s for another — overlapping
    /// ranges that looked like a 2x regression if you took one run from each.
    /// A number without its spread is not a measurement, so the spread is
    /// always printed and a bad one says so out loud.
    fn reliability_warning(&self) -> Option<String> {
        const NOISE_THRESHOLD_PCT: f64 = 15.0;
        let spread = self.spread_pct();
        (spread > NOISE_THRESHOLD_PCT).then(|| {
            format!(
                "WARNING: {spread:.0}% spread is too noisy to compare against another run.\n\
                 Something else is likely competing for the CPU or GPU. Quieten the machine \
                 and re-run, or raise {REPEATS_ENV}; do not quote this median as a result."
            )
        })
    }
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

    #[test]
    fn repeat_count_is_validated() {
        assert_eq!(repeat_count(None).unwrap(), DEFAULT_REPEATS);
        assert_eq!(repeat_count(Some(" 9 ")).unwrap(), 9);
        // Zero would leave no samples and panic in Stats::from.
        assert!(repeat_count(Some("0")).is_err());
        assert!(repeat_count(Some("lots")).is_err());
    }

    fn ms(n: u64) -> std::time::Duration {
        std::time::Duration::from_millis(n)
    }

    #[test]
    fn stats_report_the_median_not_the_minimum() {
        // The minimum would report 100ms here and hide the outlier entirely.
        let mut s = [ms(100), ms(400), ms(110), ms(120), ms(105)];
        let stats = Stats::from(&mut s);
        assert_eq!(stats.median, ms(110));
        assert_eq!(stats.min, ms(100));
        assert_eq!(stats.max, ms(400));
    }

    #[test]
    fn an_even_sample_count_still_picks_a_real_sample() {
        let mut s = [ms(100), ms(200)];
        let stats = Stats::from(&mut s);
        assert_eq!(stats.median, ms(200));
    }

    #[test]
    fn a_noisy_run_refuses_to_be_quoted_as_a_result() {
        // The real numbers that made a chunk-size comparison meaningless.
        let mut noisy = [ms(17300), ms(12300), ms(8000)];
        let stats = Stats::from(&mut noisy);
        assert!(stats.spread_pct() > 60.0, "{}", stats.spread_pct());
        assert!(stats.reliability_warning().is_some());

        // A tight run says nothing, because there is nothing to warn about.
        let mut tight = [ms(1000), ms(1020), ms(1010)];
        let stats = Stats::from(&mut tight);
        assert!(stats.spread_pct() < 5.0);
        assert!(stats.reliability_warning().is_none());
    }
}
