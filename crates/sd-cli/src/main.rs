//! `sdrs` — diffusion inference from the command line, on MLX.
//!
//! One subcommand per kind of generation. The flags describe *generation*, not
//! a backend, which is why they are unchanged from the candle CLI this
//! replaces — a command that worked before works now.

#[cfg(feature = "mlx")]
mod mlx_cli;

#[cfg(not(feature = "mlx"))]
fn main() {
    eprintln!(
        "sdrs was built without a compute backend.\n\n\
         Build it with `--features mlx` (and `brew install mlx-c` first). The \
         feature is optional so that a machine without MLX can still check that \
         the crate graph is intact — not so that the binary can run without one."
    );
    std::process::exit(2);
}

#[cfg(feature = "mlx")]
use anyhow::{Context, Result};
#[cfg(feature = "mlx")]
use clap::{Parser, Subcommand};
#[cfg(feature = "mlx")]
use stable_diffusion_rs as sd;
#[cfg(feature = "mlx")]
use stable_diffusion_rs::config::Txt2ImgConfig;
#[cfg(feature = "mlx")]
use stable_diffusion_rs::tensor::mlx::Device;

#[cfg(feature = "mlx")]
#[derive(Parser)]
#[command(
    name = "sdrs",
    version,
    about = "Diffusion inference in pure Rust, on MLX"
)]
struct Cli {
    /// Run on the CPU instead of the GPU.
    ///
    /// **Slow — a diffusion step is thousands of matmuls and the GPU is the
    /// point.** It exists because a machine whose GPU is busy should still be
    /// able to run, and because "GPU or nothing" should be your choice rather
    /// than a constant compiled in.
    #[arg(long, global = true)]
    cpu: bool,

    #[command(subcommand)]
    command: Command,
}

#[cfg(feature = "mlx")]
#[derive(Subcommand)]
enum Command {
    /// Generate an image from a text prompt.
    #[command(name = "txt2img")]
    Txt2Img {
        /// Model directory in the standard diffusers layout.
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "")]
        negative_prompt: String,
        #[arg(long, default_value_t = 512)]
        width: usize,
        #[arg(long, default_value_t = 512)]
        height: usize,
        #[arg(long, default_value_t = 20)]
        steps: usize,
        #[arg(long, default_value_t = 7.5)]
        cfg_scale: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// `euler-a`, `dpmpp2m`, or `lcm` (needs an LCM model or --lora).
        #[arg(long, default_value = "euler-a")]
        sampler: String,
        #[arg(short, long, default_value = "out.png")]
        output: String,
        /// Treat the model directory as SDXL (two text encoders).
        #[arg(long)]
        sdxl: bool,
        /// LoRA adapter to merge into the UNet.
        #[arg(long)]
        lora: Option<String>,
        #[arg(long, default_value_t = 1.0)]
        lora_scale: f64,
        /// ControlNet weights, with `--control-map`.
        #[arg(long, requires = "control_map")]
        controlnet: Option<String>,
        /// The control map, at the run's own size.
        #[arg(long)]
        control_map: Option<String>,
        #[arg(long, default_value_t = 1.0)]
        control_scale: f64,
        /// Textual inversion, as `trigger=path.safetensors`. Repeatable.
        #[arg(long)]
        embedding: Vec<String>,
        /// AnimateDiff motion adapter, so frames become one motion.
        #[arg(long, requires = "frames")]
        motion_adapter: Option<String>,
        /// Frames per clip. Above 1 writes `out-000.png`, `out-001.png`, ...
        #[arg(long, default_value_t = 1)]
        frames: usize,
        /// Two-pass generation: compose at --width/--height, then refine at
        /// this size. `1024x1024`, or `1024` for a square.
        ///
        /// **Not the same as generating big.** A model composes at its
        /// training resolution and duplicates subjects above it.
        #[arg(long, value_name = "WxH")]
        hires: Option<String>,
        #[arg(long, default_value_t = 0.55)]
        hires_strength: f64,
        /// Reuse the model's prediction while it is estimated not to have
        /// moved much. 0 disables it. Needs a deterministic sampler.
        #[arg(long, default_value_t = 0.0)]
        cache_threshold: f64,
        /// A region prompt, as `mask.png=a prompt`. Repeatable.
        #[arg(long)]
        region: Vec<String>,
        /// Upscale the result 4x with these Real-ESRGAN weights.
        #[arg(long)]
        upscale: Option<String>,
    },

    /// Generate from an existing image.
    #[command(name = "img2img")]
    Img2Img {
        #[arg(long)]
        model: String,
        #[arg(long)]
        init: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "")]
        negative_prompt: String,
        /// How much of the schedule to replace. 0 returns the input.
        #[arg(long, default_value_t = 0.75)]
        strength: f64,
        /// Repaint only where this mask is white.
        #[arg(long)]
        mask: Option<String>,
        #[arg(long, default_value_t = 512)]
        width: usize,
        #[arg(long, default_value_t = 512)]
        height: usize,
        #[arg(long, default_value_t = 20)]
        steps: usize,
        #[arg(long, default_value_t = 7.5)]
        cfg_scale: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value = "euler-a")]
        sampler: String,
        #[arg(short, long, default_value = "out.png")]
        output: String,
    },

    /// Upscale an image 4x with Real-ESRGAN.
    Upscale {
        /// A model directory, for the pipeline the upscaler runs on.
        #[arg(long)]
        model: String,
        /// Real-ESRGAN weights.
        #[arg(long)]
        weights: String,
        #[arg(long)]
        input: String,
        #[arg(short, long, default_value = "out.png")]
        output: String,
    },

    /// Report what is on this machine.
    Info,
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "mlx")]
fn config(
    prompt: &str,
    negative: &str,
    width: usize,
    height: usize,
    steps: usize,
    cfg_scale: f64,
    seed: u64,
    sampler: &str,
) -> Result<Txt2ImgConfig> {
    Ok(Txt2ImgConfig {
        prompt: prompt.to_string(),
        negative_prompt: negative.to_string(),
        width,
        height,
        steps,
        cfg_scale,
        seed,
        sampler: mlx_cli::parse_sampler(sampler)?,
    })
}

/// Split `a=b` on its **first** `=`, so a prompt may contain one.
#[cfg(feature = "mlx")]
fn split_pair(spec: &str) -> Result<(String, String)> {
    spec.split_once('=')
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .with_context(|| format!("{spec:?} should be `key=value`"))
}

#[cfg(feature = "mlx")]
fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let device = if cli.cpu { Device::Cpu } else { Device::Gpu };
    match cli.command {
        Command::Txt2Img {
            model,
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            cfg_scale,
            seed,
            sampler,
            output,
            sdxl,
            lora,
            lora_scale,
            controlnet,
            control_map,
            control_scale,
            embedding,
            motion_adapter,
            frames,
            hires,
            hires_strength,
            cache_threshold,
            region,
            upscale,
        } => {
            let cfg = config(
                &prompt,
                &negative_prompt,
                width,
                height,
                steps,
                cfg_scale,
                seed,
                &sampler,
            )?;
            let args = mlx_cli::Txt2ImgArgs {
                model,
                cfg,
                output,
                sdxl,
                lora: lora.map(|p| (p, lora_scale)),
                controlnet: match (controlnet, control_map) {
                    (Some(w), Some(m)) => vec![(w, m, control_scale)],
                    _ => Vec::new(),
                },
                embeddings: embedding
                    .iter()
                    .map(|s| split_pair(s))
                    .collect::<Result<Vec<_>>>()?,
                motion: motion_adapter.map(|p| (p, frames)),
                hires: match &hires {
                    Some(spec) => Some((mlx_cli::parse_size(spec)?, hires_strength)),
                    None => None,
                },
                cache_threshold,
                regions: region
                    .iter()
                    .map(|s| split_pair(s))
                    .collect::<Result<Vec<_>>>()?,
                upscale,
            };
            for path in mlx_cli::run_txt2img(&args, device)? {
                println!("wrote {}", path.display());
            }
        }

        Command::Img2Img {
            model,
            init,
            prompt,
            negative_prompt,
            strength,
            mask,
            width,
            height,
            steps,
            cfg_scale,
            seed,
            sampler,
            output,
        } => {
            let cfg = config(
                &prompt,
                &negative_prompt,
                width,
                height,
                steps,
                cfg_scale,
                seed,
                &sampler,
            )?;
            for path in mlx_cli::run_img2img(
                &model,
                &cfg,
                &init,
                strength,
                mask.as_deref(),
                &output,
                device,
            )? {
                println!("wrote {}", path.display());
            }
        }

        Command::Upscale {
            model,
            weights,
            input,
            output,
        } => {
            for path in mlx_cli::run_upscale(&model, &weights, &input, &output, device)? {
                println!("wrote {}", path.display());
            }
        }

        Command::Info => {
            println!("sdrs {}", sd::VERSION);
            println!("backend: MLX ({device})");
            match sd_tensor::sysmem::available_bytes() {
                Some(free) => println!("free memory: {}", sd_tensor::ops::human_bytes(free)),
                None => println!("free memory: unknown"),
            }
        }
    }
    Ok(())
}
