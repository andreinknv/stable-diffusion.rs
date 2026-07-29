//! End-to-end SD 1.5 generation time, for A/B against `SD_FUSED_KERNELS=0`.
//!
//! Times generation only — the pipeline is loaded first, outside the clock —
//! and synchronises before stopping, because Metal queues work and timing the
//! enqueue measures nothing.
//!
//! ```bash
//! SD_FUSED_KERNELS=1 cargo run --release -p sd-cli --features metal \
//!   --example sd15_step_time -- models/sd15 20
//! ```

use anyhow::Result;
use stable_diffusion_rs::pipeline::{Txt2ImgConfig, Txt2ImgPipeline};

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).unwrap_or("models/sd15".into());
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let dev = sd_tensor::device::best()?;
    let pipeline = Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev)?;

    let cfg = Txt2ImgConfig {
        prompt: "a rusty crab on a beach".into(),
        width: 512,
        height: 512,
        steps,
        seed: 42,
        ..Default::default()
    };

    // One warm run so shader compilation and first-touch allocation are not
    // in the measurement, then the run that counts.
    let _ = pipeline.run(&cfg)?;
    dev.synchronize()?;

    let t0 = std::time::Instant::now();
    let image = pipeline.run(&cfg)?;
    dev.synchronize()?;
    let elapsed = t0.elapsed().as_secs_f64();

    let on = std::env::var("SD_FUSED_KERNELS").as_deref() != Ok("0");
    println!(
        "fused_kernels={} steps={steps} generate={elapsed:.3}s ({:.1} ms/step) dims={:?}",
        if on { "on" } else { "off" },
        elapsed * 1000.0 / steps as f64,
        image.dims()
    );
    Ok(())
}
