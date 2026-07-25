//! `sdrs` — command-line interface.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

// The umbrella crate is `stable-diffusion-rs`; `sd` was already taken on
// crates.io. Aliasing keeps call sites short — users can do the same.
use stable_diffusion_rs as sd;

use sd::models::vae::{AutoencoderKlDecoder, VaeConfig};
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

    /// Report the active compute device and build configuration.
    Info,
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
    }

    Ok(())
}
