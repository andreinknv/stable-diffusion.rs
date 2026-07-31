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
use clap::{Args, Parser, Subcommand};
#[cfg(feature = "mlx")]
use stable_diffusion_rs as sd;
#[cfg(feature = "mlx")]
use stable_diffusion_rs::config::Txt2ImgConfig;
#[cfg(feature = "mlx")]
use stable_diffusion_rs::tensor::mlx::Device;

/// The flags that describe a *generation*, shared by every command that does
/// one.
///
/// Flattened rather than repeated per subcommand: the previous shape passed
/// eight of these positionally into a `config` helper, which is how `--seed`
/// and `--steps` come to be swapped at one call site and not another.
#[cfg(feature = "mlx")]
#[derive(Args, Clone)]
struct Generation {
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
    /// A batch runs `seed`, `seed + 1`, ... so each image is individually
    /// reproducible rather than only the batch as a whole.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// `euler-a`, `euler`, `heun`, `dpmpp2m`, `dpmpp2s-a`, `ddim`, or `lcm`.
    ///
    /// **`heun` and `dpmpp2s-a` evaluate the model twice per step**, so twenty
    /// of their steps cost about what forty Euler steps do.
    #[arg(long, default_value = "euler-a")]
    sampler: String,
    /// `discrete`, `karras`, `exponential`, or `sgm-uniform`.
    ///
    /// **Karras is what most published step counts assume.** Running someone
    /// else's "20 steps, DPM++ 2M" recipe on the discrete ladder compares two
    /// different schedules.
    #[arg(long, default_value = "discrete")]
    scheduler: String,
    /// How many CLIP layers to discard from the end. 1 is what SD 1.5 was
    /// trained with; **2 is what most community finetunes expect**, and using
    /// 1 on one of those is not an error, just a flatter picture.
    #[arg(long, default_value_t = 1)]
    clip_skip: usize,
    /// Generate this many images. The model is loaded once for all of them.
    #[arg(long, short = 'n', default_value_t = 1)]
    batch_count: usize,
    /// `f32` or `f16`. Measured on SD 1.5 at 768: f16 is 1.10x faster and
    /// 1.15 GB smaller, at PSNR 36.8 dB from the f32 image — a different
    /// sample of comparable quality, not a degraded one.
    #[arg(long, default_value = "f32")]
    precision: String,
    /// Make the image tile: convolutions wrap at the edge, in the decoder as
    /// well as the UNet, so opposite edges agree where they meet.
    #[arg(long)]
    seamless: bool,
}

#[cfg(feature = "mlx")]
impl Generation {
    fn to_config(&self) -> Result<Txt2ImgConfig> {
        use stable_diffusion_rs::config::{ClipSkip, Precision, SamplerKind, Scheduler};
        Ok(Txt2ImgConfig {
            prompt: self.prompt.clone(),
            negative_prompt: self.negative_prompt.clone(),
            width: self.width,
            height: self.height,
            steps: self.steps,
            cfg_scale: self.cfg_scale,
            seed: self.seed,
            sampler: SamplerKind::parse(&self.sampler).with_context(|| {
                format!(
                    "unknown sampler {:?}; try one of: {}",
                    self.sampler,
                    SamplerKind::all()
                        .iter()
                        .map(|s| s.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?,
            scheduler: Scheduler::parse(&self.scheduler).with_context(|| {
                format!(
                    "unknown scheduler {:?}; try discrete, karras, exponential or sgm-uniform",
                    self.scheduler
                )
            })?,
            clip_skip: ClipSkip::new(self.clip_skip),
            batch_count: self.batch_count.max(1),
            precision: Precision::parse(&self.precision).with_context(|| {
                format!("unknown precision {:?}; try f32 or f16", self.precision)
            })?,
            seamless: self.seamless,
        })
    }
}

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

// `txt2img` carries far more flags than the other subcommands, so the enum is
// as large as its biggest variant. Boxing the payload would mean a manual
// `FromArgMatches`, which is a lot of hand-written parsing to save one
// short-lived allocation on a command-line binary.
#[allow(clippy::large_enum_variant)]
#[cfg(feature = "mlx")]
#[derive(Subcommand)]
enum Command {
    /// Generate an image from a text prompt.
    #[command(name = "txt2img")]
    Txt2Img {
        /// Model directory in the standard diffusers layout.
        /// Model directory in the standard diffusers layout.
        #[arg(long)]
        model: String,
        #[command(flatten)]
        gen: Generation,
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
        /// IP-Adapter weights, with `--reference` and `--image-encoder`.
        ///
        /// Conditions on an image's *style and content* rather than its
        /// geometry, which is what separates it from a ControlNet.
        #[arg(long, requires = "reference", requires = "image_encoder")]
        ip_adapter: Option<String>,
        /// The CLIP vision tower the adapter was trained against. A separate
        /// checkpoint, and a large one.
        #[arg(long)]
        image_encoder: Option<String>,
        /// The reference image for `--ip-adapter`.
        #[arg(long)]
        reference: Option<String>,
        #[arg(long, default_value_t = 1.0)]
        ip_scale: f64,
        /// A GLIGEN grounding box, as `x0,y0,x1,y1=phrase`. Repeatable.
        ///
        /// **Coordinates are normalised to `[0, 1]`, not pixels.** The model
        /// was trained on normalised boxes, and pixel values put every box off
        /// the canvas with no error.
        #[arg(long)]
        grounded_box: Vec<String>,
        /// Derive the control map by running Canny edge detection on this
        /// image, instead of passing a prepared map to `--control-map`.
        #[arg(long, conflicts_with = "control_map", requires = "controlnet")]
        canny: Option<String>,
        #[arg(long, default_value_t = 0.1)]
        canny_low: f32,
        #[arg(long, default_value_t = 0.3)]
        canny_high: f32,
        /// Decode with TAESD instead of the full VAE.
        ///
        /// A tiny distilled decoder: far faster and visibly softer. For
        /// previews and for sweeping seeds, not for a final image.
        #[arg(long)]
        taesd: Option<String>,
    },

    /// Generate from an existing image.
    #[command(name = "img2img")]
    Img2Img {
        #[arg(long)]
        model: String,
        #[arg(long)]
        init: String,
        /// How much of the schedule to replace. 0 returns the input.
        #[arg(long, default_value_t = 0.75)]
        strength: f64,
        /// Repaint only where this mask is white.
        #[arg(long)]
        mask: Option<String>,
        #[command(flatten)]
        gen: Generation,
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

    /// Edit an image by instruction: "change the sky to sunset".
    ///
    /// **Not img2img.** That re-noises a picture and denoises it against a
    /// prompt describing the result; this conditions on the source directly
    /// and takes a prompt describing the *change*. Needs an InstructPix2Pix
    /// checkpoint, whose UNet takes eight input channels.
    Instruct {
        #[arg(long)]
        model: String,
        #[arg(long)]
        init: String,
        /// How strongly to follow the instruction. This is `--cfg-scale`.
        #[command(flatten)]
        gen: Generation,
        /// How strongly to stay faithful to the source image. Raise it when
        /// the edit changes too much, lower it when nothing changes.
        #[arg(long, default_value_t = 1.5)]
        image_guidance: f64,
        #[arg(short, long, default_value = "out.png")]
        output: String,
    },

    /// Generate with Flux — schnell, dev, or flux-mini.
    ///
    /// **Quantised at rest by default.** Flux schnell is 12B parameters; held
    /// dense it needs about 48 GB, and at 4 bits with the sensitive layers at
    /// 8 it runs in 13.3 GB alongside T5-XXL.
    Flux {
        /// A `diffusers` Flux directory. With `--transformer-gguf` this is
        /// only used for the pieces not given explicitly.
        #[arg(long)]
        model: String,
        /// `schnell`, `dev`, or `mini`.
        ///
        /// **Not cosmetic.** schnell has 19 double blocks and no guidance
        /// embedding; dev has 19 and one. Passing a guidance scale to schnell
        /// is an error rather than a silent no-op.
        #[arg(long, default_value = "schnell")]
        variant: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 512)]
        width: usize,
        #[arg(long, default_value_t = 512)]
        height: usize,
        #[arg(long, default_value_t = 4)]
        steps: usize,
        /// Distilled guidance. **dev only** — schnell refuses it.
        #[arg(long, default_value_t = 3.5)]
        guidance: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Bits for the bulk of the weights. The layers `quantized::sensitive`
        /// names stay at 8 whatever this says.
        #[arg(long, default_value_t = 4)]
        bits: usize,
        /// A Flux transformer GGUF, in black-forest-labs naming.
        #[arg(long, requires = "t5_gguf")]
        transformer_gguf: Option<String>,
        /// A T5-XXL encoder GGUF, in llama.cpp naming.
        #[arg(long)]
        t5_gguf: Option<String>,
        /// CLIP-L weights, when they are not under `--model`.
        #[arg(long)]
        clip: Option<String>,
        /// VAE weights, when they are not under `--model`.
        #[arg(long)]
        vae: Option<String>,
        /// The T5 tokenizer, when it is not under `--model`.
        #[arg(long)]
        t5_tokenizer: Option<String>,
        /// FLUX.1-Kontext: edit this image by instruction rather than
        /// generating from scratch.
        ///
        /// The reference is encoded and its tokens **appended to the sequence**
        /// the transformer attends over, marked in the rotary embedding's third
        /// axis so the model can tell them from the image it is making.
        /// Needs a Kontext checkpoint; `--variant dev` is the base.
        #[arg(long)]
        reference: Option<String>,
        #[arg(short, long, default_value = "out.png")]
        output: String,
    },

    /// Generate with SD 3.5.
    ///
    /// Quantised at rest, like `flux`: SD 3.5 medium plus T5-XXL is 10.1 GB
    /// at 4 bits against roughly 40 dense.
    Sd3 {
        /// An SD 3.5 directory. **The published one is incomplete** — it ships
        /// no `text_encoder_3` and no tokenisers — so the explicit flags below
        /// are the path a real checkpoint usually needs.
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
        #[arg(long, default_value_t = 4.5)]
        cfg_scale: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 4)]
        bits: usize,
        /// Skip-layer guidance: steer away from a prediction made with these
        /// blocks bypassed. `7,8,9` is Stability's published set for SD 3.5,
        /// and helps most with anatomy — hands, limb counts.
        ///
        /// **Costs a third model evaluation** for every step inside the
        /// window, so it is not free.
        #[arg(long, value_name = "7,8,9")]
        slg_layers: Option<String>,
        #[arg(long, default_value_t = 2.8)]
        slg_scale: f64,
        /// The fraction of the schedule over which it applies. Early steps set
        /// composition and late ones texture; pushing away from a degraded
        /// prediction at either end distorts rather than fixes.
        #[arg(long, default_value_t = 0.01)]
        slg_start: f64,
        #[arg(long, default_value_t = 0.2)]
        slg_end: f64,
        /// The MMDiT weights, in Stability's original naming.
        #[arg(long)]
        transformer: Option<String>,
        #[arg(long)]
        vae: Option<String>,
        #[arg(long)]
        clip_l: Option<String>,
        #[arg(long)]
        clip_g: Option<String>,
        /// T5-XXL, as safetensors or GGUF.
        #[arg(long)]
        t5: Option<String>,
        #[arg(long)]
        t5_tokenizer: Option<String>,
        #[arg(short, long, default_value = "out.png")]
        output: String,
    },

    /// Generate a variation of an image, or from a prompt through the prior.
    ///
    /// unCLIP conditions on a CLIP *image* embedding rather than a text one,
    /// which is what makes "another picture like this one" a thing you can
    /// ask for directly.
    Unclip {
        /// An unCLIP model directory.
        #[arg(long)]
        model: String,
        /// A reference image. Without one the prompt goes through the prior.
        #[arg(long)]
        image: Option<String>,
        #[arg(long, default_value = "")]
        prompt: String,
        #[arg(long, default_value_t = 768)]
        width: usize,
        #[arg(long, default_value_t = 768)]
        height: usize,
        #[arg(long, default_value_t = 20)]
        steps: usize,
        #[arg(long, default_value_t = 10.0)]
        cfg_scale: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// How much noise to add to the image embedding. Higher is more
        /// variation and less fidelity to the reference.
        #[arg(long, default_value_t = 0)]
        noise_level: usize,
        #[arg(short, long, default_value = "out.png")]
        output: String,
    },

    /// Merge two checkpoints by weighted average.
    ///
    /// `(1 - alpha) * a + alpha * b`, tensor by tensor. Only meaningful
    /// between checkpoints of the same architecture, and mismatches are
    /// refused rather than skipped — an SD 1.5 and an SDXL share enough tensor
    /// names to produce a file that loads and renders noise.
    Merge {
        #[arg(long)]
        a: String,
        #[arg(long)]
        b: String,
        /// Weight of `--b`. 0 is `--a` exactly, 1 is `--b` exactly.
        #[arg(long, default_value_t = 0.5)]
        alpha: f64,
        /// Carry through tensors only one side has, instead of refusing.
        #[arg(long)]
        allow_unmatched: bool,
        #[arg(short, long, default_value = "merged.safetensors")]
        output: String,
    },

    /// Report what is on this machine.
    Info,
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
            gen,
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
            ip_adapter,
            image_encoder,
            reference,
            ip_scale,
            grounded_box,
            canny,
            canny_low,
            canny_high,
            taesd,
        } => {
            let cfg = gen.to_config()?;
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
                ip: match (ip_adapter, image_encoder, reference) {
                    (Some(a), Some(e), Some(r)) => Some((a, e, r, ip_scale)),
                    _ => None,
                },
                boxes: grounded_box
                    .iter()
                    .map(|s| mlx_cli::parse_box(s))
                    .collect::<Result<Vec<_>>>()?,
                canny: canny.map(|c| (c, canny_low, canny_high)),
                taesd,
            };
            for path in mlx_cli::run_txt2img(&args, device)? {
                println!("wrote {}", path.display());
            }
        }

        Command::Img2Img {
            model,
            init,
            strength,
            mask,
            gen,
            output,
        } => {
            let cfg = gen.to_config()?;
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

        Command::Instruct {
            model,
            init,
            gen,
            image_guidance,
            output,
        } => {
            let cfg = gen.to_config()?;
            for path in mlx_cli::run_instruct(&model, &cfg, &init, image_guidance, &output, device)?
            {
                println!("wrote {}", path.display());
            }
        }

        Command::Flux {
            model,
            variant,
            prompt,
            width,
            height,
            steps,
            guidance,
            seed,
            bits,
            transformer_gguf,
            t5_gguf,
            clip,
            vae,
            t5_tokenizer,
            reference,
            output,
        } => {
            let args = mlx_cli::FluxArgs {
                model,
                variant,
                cfg: stable_diffusion_rs::mlx::FluxRunConfig {
                    prompt,
                    width,
                    height,
                    steps,
                    guidance,
                    seed,
                },
                bits,
                transformer_gguf,
                t5_gguf,
                clip,
                vae,
                t5_tokenizer,
                reference,
                output,
            };
            for path in mlx_cli::run_flux(&args, device)? {
                println!("wrote {}", path.display());
            }
        }

        Command::Sd3 {
            model,
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            cfg_scale,
            seed,
            bits,
            slg_layers,
            slg_scale,
            slg_start,
            slg_end,
            transformer,
            vae,
            clip_l,
            clip_g,
            t5,
            t5_tokenizer,
            output,
        } => {
            let slg = match &slg_layers {
                Some(spec) => Some(stable_diffusion_rs::mlx::SkipLayerGuidance {
                    layers: spec
                        .split(',')
                        .map(|p| {
                            p.trim()
                                .parse::<usize>()
                                .with_context(|| format!("block index {p:?} in {spec:?}"))
                        })
                        .collect::<Result<Vec<_>>>()?,
                    scale: slg_scale,
                    start: slg_start,
                    end: slg_end,
                }),
                None => None,
            };
            let args = mlx_cli::Sd3Args {
                model,
                cfg: stable_diffusion_rs::mlx::Sd3RunConfig {
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    steps,
                    cfg_scale,
                    seed,
                    slg,
                },
                bits,
                transformer,
                vae,
                clip_l,
                clip_g,
                t5,
                t5_tokenizer,
                output,
            };
            for path in mlx_cli::run_sd3(&args, device)? {
                println!("wrote {}", path.display());
            }
        }

        Command::Unclip {
            model,
            image,
            prompt,
            width,
            height,
            steps,
            cfg_scale,
            seed,
            noise_level,
            output,
        } => {
            let cfg = Txt2ImgConfig {
                prompt,
                width,
                height,
                steps,
                cfg_scale,
                seed,
                ..Default::default()
            };
            for path in
                mlx_cli::run_unclip(&model, &cfg, image.as_deref(), noise_level, &output, device)?
            {
                println!("wrote {}", path.display());
            }
        }

        Command::Merge {
            a,
            b,
            alpha,
            allow_unmatched,
            output,
        } => {
            let report = mlx_cli::run_merge(&a, &b, alpha, allow_unmatched, &output)?;
            println!(
                "wrote {output}: {} blended, {} carried",
                report.blended, report.carried
            );
        }

        Command::Info => {
            use sd_tensor::ops::human_bytes;
            println!("sdrs {}", sd::VERSION);
            println!("backend: MLX ({device})");
            match sd_tensor::sysmem::available_bytes() {
                Some(free) => println!("free memory: {}", human_bytes(free)),
                None => println!("free memory: unknown"),
            }
            // MLX's own accounting, not resident set size. On unified memory
            // the two answer different questions, and RSS is the one that
            // includes the memory-mapped checkpoint.
            if let (Ok(active), Ok(cache)) = (
                sd_tensor::mlx::active_memory(),
                sd_tensor::mlx::cache_memory(),
            ) {
                println!(
                    "mlx memory: {} live, {} cached",
                    human_bytes(active as u64),
                    human_bytes(cache as u64)
                );
            }
            println!(
                "samplers: {}",
                stable_diffusion_rs::config::SamplerKind::all()
                    .iter()
                    .map(|s| s.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!("schedulers: discrete, karras, exponential, sgm-uniform");
        }
    }
    Ok(())
}
