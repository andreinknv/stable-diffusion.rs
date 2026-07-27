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
        #[arg(long, conflicts_with = "gguf")]
        model: Option<String>,

        /// A single LDM-layout `.gguf` checkpoint instead of a directory.
        ///
        /// These carry no tokenizer — stable-diffusion.cpp writes no GGUF
        /// metadata at all — so `--tokenizer` is required with it.
        #[arg(long, requires = "tokenizer")]
        gguf: Option<String>,

        /// `tokenizer.json`, for `--gguf`. Copy it from
        /// `openai/clip-vit-large-patch14`.
        #[arg(long)]
        tokenizer: Option<String>,

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

        /// LoRA adapter to merge into the UNet. SD 1.5 only for now.
        #[arg(long)]
        lora: Option<String>,

        /// LoRA strength. 0 is identical to not passing --lora.
        #[arg(long, default_value_t = 1.0)]
        lora_scale: f64,
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

        /// Treat the model directory as SDXL (two text encoders).
        #[arg(long)]
        sdxl: bool,
    },

    /// Report what a GGUF checkpoint contains, without loading it.
    Inspect {
        /// Path to a `.gguf` file.
        #[arg(long)]
        gguf: String,

        /// Print every metadata key, not just the summary.
        #[arg(long)]
        verbose: bool,
    },

    /// Report the active compute device and build configuration.
    Info,
}

fn parse_sampler(name: &str) -> Result<SamplerKind> {
    match name {
        "euler-a" | "euler_a" | "euler" => Ok(SamplerKind::EulerAncestral),
        "dpmpp2m" | "dpm++2m" | "dpmpp-2m" => Ok(SamplerKind::DpmPlusPlus2M),
        "lcm" => Ok(SamplerKind::Lcm),
        other => anyhow::bail!(
            "unknown sampler {other:?}; expected 'euler-a', 'dpmpp2m' or 'lcm'.\n\
             'lcm' needs an LCM-distilled model or --lora, 4-8 steps, and --cfg-scale near 1."
        ),
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
        Command::Inspect { gguf, verbose } => {
            // Header only — no tensor data is read, so this is instant even
            // for a checkpoint of several gigabytes.
            let info =
                sd::loader::GgufInfo::open(&gguf).with_context(|| format!("reading {gguf}"))?;
            println!("file:         {}", info.path.display());
            println!(
                "architecture: {}",
                info.architecture().unwrap_or("(not declared)")
            );
            println!("tensors:      {}", info.tensors.len());
            println!("parameters:   {:.2} M", info.parameter_count() as f64 / 1e6);
            // The number that matters before loading: there is no quantised
            // inference path, so a 4-bit file costs what its expanded weights
            // cost, not what the file does.
            println!(
                "as f32:       {:.2} GB in memory once dequantised",
                info.dequantised_bytes(DType::F32) as f64 / 1e9
            );
            println!("quantisation:");
            for (dtype, count) in info.quantisations() {
                println!("  {count:>6} x {dtype:?}");
            }
            if verbose {
                println!("metadata:");
                if info.metadata.is_empty() {
                    // Not hypothetical: stable-diffusion.cpp writes none.
                    println!("  (none — the file declares nothing about itself)");
                } else {
                    let mut keys: Vec<_> = info.metadata.keys().collect();
                    keys.sort();
                    for k in keys {
                        println!("  {k}");
                    }
                }
                // Grouped by top-level prefix: 1131 individual names is not
                // something anyone reads, but the prefixes say what is inside.
                println!("tensor name prefixes:");
                let mut groups: std::collections::BTreeMap<String, usize> = Default::default();
                for name in info.tensors.keys() {
                    let prefix: String = name.split('.').take(2).collect::<Vec<_>>().join(".");
                    *groups.entry(prefix).or_default() += 1;
                }
                for (prefix, count) in groups {
                    println!("  {count:>6}  {prefix}.*");
                }
            }
        }

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
            gguf,
            tokenizer,
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

            let source = gguf.clone().or_else(|| model.clone()).unwrap_or_default();
            tracing::info!(model = %source, sdxl, gguf = gguf.is_some(), "loading pipeline");
            let started = std::time::Instant::now();
            let mut report = |step, total, sigma: f64| {
                tracing::info!(step, total, sigma = format!("{sigma:.3}"), "denoise");
            };
            // A 20-step CPU run takes minutes; without per-step output it
            // looks hung.
            let img = match (gguf.as_deref(), sdxl) {
                (Some(_), true) => {
                    anyhow::bail!("--gguf is SD 1.5 only; SDXL GGUF checkpoints are not supported")
                }
                (Some(g), false) => {
                    let tok = tokenizer.as_deref().expect("clap requires it with --gguf");
                    let pipeline = Txt2ImgPipeline::load_gguf(Path::new(g), Path::new(tok), &dev)
                        .with_context(|| format!("loading pipeline from {g}"))?;
                    pipeline
                        .run_with_progress(&cfg, &mut report)
                        .context("running txt2img from gguf")?
                }
                (None, true) => {
                    let m = model.as_deref().context("--model or --gguf is required")?;
                    let pipeline = sd::pipeline::SdxlPipeline::load(Path::new(m), &dev)
                        .with_context(|| format!("loading SDXL pipeline from {m}"))?;
                    pipeline
                        .run_with_progress(&cfg, &mut report)
                        .context("running SDXL txt2img")?
                }
                (None, false) => {
                    let m = model.as_deref().context("--model or --gguf is required")?;
                    let pipeline = match lora.as_deref() {
                        Some(l) => {
                            tracing::info!(lora = %l, scale = lora_scale, "merging LoRA");
                            Txt2ImgPipeline::load_with_lora(
                                Path::new(m),
                                &dev,
                                Path::new(l),
                                lora_scale,
                            )
                            .with_context(|| format!("loading {m} with LoRA {l}"))?
                        }
                        None => Txt2ImgPipeline::load(Path::new(m), &dev)
                            .with_context(|| format!("loading pipeline from {m}"))?,
                    };
                    pipeline
                        .run_with_progress(&cfg, &mut report)
                        .context("running txt2img")?
                }
            };

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
            sdxl,
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

            tracing::info!(model = %model, sdxl, "loading pipeline");
            tracing::info!(
                prompt = %cfg.base.prompt,
                init = %init_image,
                strength = cfg.strength.get(),
                "generating"
            );
            let started = std::time::Instant::now();
            let mut report = |step, total, sigma: f64| {
                tracing::info!(step, total, sigma = format!("{sigma:.3}"), "denoise");
            };
            let img = if sdxl {
                let pipeline = sd::pipeline::SdxlPipeline::load(Path::new(&model), &dev)
                    .with_context(|| format!("loading SDXL pipeline from {model}"))?;
                pipeline
                    .run_img2img_with_progress(&cfg, &mut report)
                    .context("running SDXL img2img")?
            } else {
                let pipeline = Txt2ImgPipeline::load(Path::new(&model), &dev)
                    .with_context(|| format!("loading pipeline from {model}"))?;
                pipeline
                    .run_img2img_with_progress(&cfg, &mut report)
                    .context("running img2img")?
            };

            sd::image_io::save_png(&img, &output).with_context(|| format!("writing {output}"))?;
            println!("wrote {output} in {:.1?}", started.elapsed());
        }
    }

    Ok(())
}
