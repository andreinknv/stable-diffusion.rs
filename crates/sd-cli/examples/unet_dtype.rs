//! Is f16 worth it for SD 1.5?
//!
//! The roadmap has carried this as "if it is ever worth it... measure first".
//! SDXL needed f16 to fit at 1024; SD 1.5 does not need it to fit at all, so
//! the only reason to switch would be speed — and switching means casting at
//! the sampler boundary in the most-verified loop in the project, plus
//! re-verifying every golden test against f16 tolerances.
//!
//! That is a lot to spend on an unmeasured hunch, so this measures the thing
//! that decides it, and nothing else: **one UNet forward, f32 against f16, on
//! the same weights and the same input.** If f16 is not meaningfully faster
//! here, the rest of the work is not worth starting.
//!
//! Reports residency too, since that is the other half of the trade — and the
//! half SDXL actually needed.
//!
//! ```bash
//! cargo run --release -p sd-cli --features metal --example unet_dtype -- models/sd15
//! ```

use anyhow::{Context, Result};
use stable_diffusion_rs as sd;

use sd::models::unet::{UNet2DConditionModel, UNetConfig};
use sd_tensor::{DType, Device, Tensor};

fn time_forward(
    label: &str,
    dev: &Device,
    dtype: DType,
    weights: &std::path::Path,
    iters: usize,
) -> Result<f64> {
    let vb = sd::loader::safetensors_var_builder(&[weights], dtype, dev)
        .with_context(|| format!("loading in {dtype:?}"))?;
    let unet = UNet2DConditionModel::new(&UNetConfig::sd15(), vb)?;

    let mut rng = sd_tensor::rng::SeededRng::new(0);
    // A guidance batch at 512, which is what a real step runs.
    let sample = rng.randn((2, 4, 64, 64), dev)?.to_dtype(dtype)?;
    let timestep = Tensor::from_vec(vec![500f32; 2], 2, dev)?;
    let context = rng.randn((2, 77, 768), dev)?.to_dtype(dtype)?;

    // Warm-up, then synchronise inside the timed region: candle queues Metal
    // work, so a naive timer measures enqueue rather than execution.
    unet.forward(&sample, &timestep, &context)?;
    dev.synchronize()?;
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        unet.forward(&sample, &timestep, &context)?;
        dev.synchronize()?;
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    println!("  {label:<10} {best:>9.1} ms");
    Ok(best)
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/sd15".to_string());
    let weights = std::path::Path::new(&dir).join("unet/diffusion_pytorch_model.safetensors");
    if !weights.exists() {
        anyhow::bail!("no UNet at {}", weights.display());
    }
    let dev = sd_tensor::device::best()?;
    println!("device {dev:?}\n");

    let bytes = std::fs::metadata(&weights)?.len() as f64;
    println!(
        "weights on disk {:.2} GB; resident f32 {:.2} GB, f16 {:.2} GB\n",
        bytes / 1e9,
        bytes / 1e9,
        bytes / 2e9
    );

    // f32 first, then f16, then f32 again — alternating, because this machine
    // drifts and a single ordered pair has misled this project before.
    let a32 = time_forward("f32", &dev, DType::F32, &weights, 8)?;
    let a16 = time_forward("f16", &dev, DType::F16, &weights, 8)?;
    let b32 = time_forward("f32 again", &dev, DType::F32, &weights, 8)?;
    let f32_best = a32.min(b32);
    let ratio = f32_best / a16;
    println!(
        "\nf16 is {ratio:.2}x {} on the forward",
        if ratio > 1.0 { "faster" } else { "SLOWER" }
    );
    println!(
        "A 20-step run does 20 of these, so this is worth about {:.1} s per image.",
        (f32_best - a16) * 20.0 / 1e3
    );
    Ok(())
}
