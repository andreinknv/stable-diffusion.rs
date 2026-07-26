//! `sdrs` — command-line interface.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

// The umbrella crate is `stable-diffusion-rs`; `sd` was already taken on
// crates.io. Aliasing keeps call sites short — users can do the same.
use stable_diffusion_rs as sd;

use std::path::Path;

use sd::models::vae::{AutoencoderKlDecoder, VaeConfig};
use sd::pipeline::{Img2ImgConfig, SamplerKind, Strength, Txt2ImgConfig, Txt2ImgPipeline};
use sd_tensor::{device, DType, Tensor};

#[derive(Parser)]
#[command(
    name = "sdrs",
    version,
    about = "Diffusion model inference in pure Rust",
    long_about = None
)]
struct Cli {
    /// Force CPU even when a GPU backend is compiled in.
    #[arg(long, global = true)]
    cpu: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Decode a latent tensor to an image using the VAE decoder.
    ///
    /// This is milestone 1. txt2img lands once CLIP and the UNet are verified.
    Decode {
        /// VAE weights (`.safetensors`).
        #[arg(long)]
        vae: String,

        /// Latent tensor `[1, 4, h, w]` stored as safetensors under key `latent`.
        #[arg(long)]
        latent: String,

        /// Output PNG path.
        #[arg(short, long, default_value = "out.png")]
        output: String,

        /// Treat the latent as already unscaled (skip `scaling_factor`).
        #[arg(long)]
        raw: bool,
    },

    /// Generate an image from a text prompt.
    ///
    /// Named explicitly: clap would otherwise derive `txt2-img` from the
    /// variant name, which is not what anyone will type.
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

        /// `euler-a` or `dpmpp2m`.
        #[arg(long, default_value = "euler-a")]
        sampler: String,

        #[arg(short, long, default_value = "out.png")]
        output: String,
    },

    /// Generate an image from a text prompt and an existing image.
    #[command(name = "img2img")]
    Img2Img {
        #[arg(long)]
        model: String,

        #[arg(long)]
        prompt: String,

        /// Source image. Resized to --width x --height.
        #[arg(long)]
        init_image: String,

        /// 0.0 returns the input, 1.0 ignores it.
        #[arg(long, default_value_t = 0.75)]
        strength: f64,

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

        #[arg(long, default_value = "euler-a")]
        sampler: String,

        #[arg(short, long, default_value = "out.png")]
        output: String,
    },

    /// Report the active compute device and build configuration.
    Info,
}

fn parse_sampler(name: &str) -> Result<SamplerKind> {
    match name {
        "euler-a" | "euler_a" | "euler" => Ok(SamplerKind::EulerAncestral),
        "dpmpp2m" | "dpm++2m" | "dpmpp-2m" => Ok(SamplerKind::DpmPlusPlus2M),
        other => anyhow::bail!("unknown sampler {other:?}; expected 'euler-a' or 'dpmpp2m'"),
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sd=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let dev = if cli.cpu {
        device::cpu()
    } else {
        device::best().context("selecting compute device")?
    };

    match cli.command {
        Command::Info => {
            println!("stable-diffusion.rs {}", sd::VERSION);
            println!("device:  {dev:?}");
            println!(
                "backends: cpu{}{}",
                if cfg!(feature = "cuda") { ", cuda" } else { "" },
                if cfg!(feature = "metal") {
                    ", metal"
                } else {
                    ""
                },
            );
        }

        Command::Decode {
            vae,
            latent,
            output,
            raw,
        } => {
            let cfg = VaeConfig::sd15();
            let vb = sd::loader::safetensors_var_builder(&[&vae], DType::F32, &dev)
                .with_context(|| format!("loading VAE weights from {vae}"))?;
            let decoder = AutoencoderKlDecoder::new(&cfg, vb).context("building VAE decoder")?;

            let tensors = sd_tensor::safetensors::load(&latent, &dev)
                .with_context(|| format!("loading latent from {latent}"))?;
            let z = tensors
                .get("latent")
                .with_context(|| format!("{latent} has no tensor named 'latent'"))?
                .to_dtype(DType::F32)?;

            tracing::info!(shape = ?z.dims(), "decoding latent");
            let img: Tensor = if raw {
                decoder.decode_raw(&z)?
            } else {
                decoder.decode(&z)?
            };

            sd::image_io::save_png(&img, &output).with_context(|| format!("writing {output}"))?;
            println!("wrote {output}");
        }

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
        } => {
            let cfg = Txt2ImgConfig {
                prompt,
                negative_prompt,
                width,
                height,
                steps,
                cfg_scale,
                seed,
                sampler: parse_sampler(&sampler)?,
            };

            tracing::info!(model = %model, "loading pipeline");
            let pipeline = Txt2ImgPipeline::load(Path::new(&model), &dev)
                .with_context(|| format!("loading pipeline from {model}"))?;

            tracing::info!(
                prompt = %cfg.prompt,
                steps = cfg.steps,
                seed = cfg.seed,
                "generating"
            );
            let started = std::time::Instant::now();
            // A 20-step CPU run takes minutes; without per-step output it
            // looks hung.
            let img = pipeline
                .run_with_progress(&cfg, &mut |step, total, sigma| {
                    tracing::info!(step, total, sigma = format!("{sigma:.3}"), "denoise");
                })
                .context("running txt2img")?;

            sd::image_io::save_png(&img, &output).with_context(|| format!("writing {output}"))?;
            println!("wrote {output} in {:.1?}", started.elapsed());
        }

        Command::Img2Img {
            model,
            prompt,
            init_image,
            strength,
            negative_prompt,
            width,
            height,
            steps,
            cfg_scale,
            seed,
            sampler,
            output,
        } => {
            let cfg = Img2ImgConfig {
                base: Txt2ImgConfig {
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    steps,
                    cfg_scale,
                    seed,
                    sampler: parse_sampler(&sampler)?,
                },
                init_image: std::path::PathBuf::from(&init_image),
                strength: Strength::new(strength),
            };

            tracing::info!(model = %model, "loading pipeline");
            let pipeline = Txt2ImgPipeline::load(Path::new(&model), &dev)
                .with_context(|| format!("loading pipeline from {model}"))?;

            tracing::info!(
                prompt = %cfg.base.prompt,
                init = %init_image,
                strength = cfg.strength.get(),
                "generating"
            );
            let started = std::time::Instant::now();
            let img = pipeline
                .run_img2img_with_progress(&cfg, &mut |step, total, sigma| {
                    tracing::info!(step, total, sigma = format!("{sigma:.3}"), "denoise");
                })
                .context("running img2img")?;

            sd::image_io::save_png(&img, &output).with_context(|| format!("writing {output}"))?;
            println!("wrote {output} in {:.1?}", started.elapsed());
        }
    }

    Ok(())
}
