//! The command line on MLX.
//!
//! One handler per command, over `stable_diffusion_rs::mlx`. The argument
//! surface is unchanged from the candle CLI it replaces — a flag that worked
//! before works now — because the flags describe generation, not a backend.
//!
//! # What is not here, and why
//!
//! **`--placement`.** Splitting a model across devices is a CUDA idea; on
//! Apple silicon the GPU and the CPU share one pool, so there is nothing to
//! place and moving a tensor is a no-op that costs a copy.
//!
//! **`--upscale-mode`.** The candle path offers latent-nearest,
//! latent-bilinear and pixel-lanczos between hires passes. Only nearest is
//! here: it is the default there too, and it is the one that introduces no
//! colours that were not already in the image. The other two can come back
//! when something needs them.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use sd_tensor::mlx::{load_safetensors, Array, Device};
use stable_diffusion_rs::mlx::{Cancel, MlxPipeline, Progress, Region, SdxlPipeline};
use stable_diffusion_rs::pipeline::{SamplerKind, Strength, Txt2ImgConfig};

/// Parse `WxH`, or `N` for a square.
pub fn parse_size(spec: &str) -> Result<(usize, usize)> {
    let (w, h) = match spec.split_once(['x', 'X']) {
        Some((a, b)) => (a.trim(), b.trim()),
        None => (spec.trim(), spec.trim()),
    };
    Ok((
        w.parse().with_context(|| format!("width in {spec:?}"))?,
        h.parse().with_context(|| format!("height in {spec:?}"))?,
    ))
}

pub fn parse_sampler(name: &str) -> Result<SamplerKind> {
    Ok(match name {
        "euler-a" | "euler_a" => SamplerKind::EulerAncestral,
        "dpmpp2m" | "dpmpp-2m" => SamplerKind::DpmPlusPlus2M,
        "lcm" => SamplerKind::Lcm,
        other => bail!("unknown sampler {other:?}; try euler-a, dpmpp2m or lcm"),
    })
}

/// Write RGB bytes as a PNG.
///
/// With more than one frame the name gains an index — `clip.png` becomes
/// `clip-000.png`, `clip-001.png` — because a clip is several pictures and
/// silently writing only the last one is the wrong answer.
pub fn write_images(path: &str, width: usize, height: usize, rgb: &[u8]) -> Result<Vec<PathBuf>> {
    let per = width * height * 3;
    if per == 0 || rgb.len() % per != 0 {
        bail!(
            "{} bytes is not a whole number of {width}x{height} images",
            rgb.len()
        );
    }
    let frames = rgb.len() / per;
    let base = Path::new(path);
    let mut written = Vec::with_capacity(frames);

    for i in 0..frames {
        let out = if frames == 1 {
            base.to_path_buf()
        } else {
            let stem = base.file_stem().unwrap_or_default().to_string_lossy();
            let ext = base
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_else(|| "png".into());
            base.with_file_name(format!("{stem}-{i:03}.{ext}"))
        };
        let buf = image::RgbImage::from_raw(
            width as u32,
            height as u32,
            rgb[i * per..(i + 1) * per].to_vec(),
        )
        .context("assembling the image")?;
        buf.save(&out)
            .with_context(|| format!("writing {}", out.display()))?;
        written.push(out);
    }
    Ok(written)
}

/// A progress callback that prints one line per step.
///
/// `evaluated` is shown alongside `step` because with caching on they differ,
/// and the difference is the whole saving.
pub fn print_progress(p: Progress<'_>) {
    if p.evaluated == p.step {
        eprint!("\rstep {}/{}  sigma {:.3}   ", p.step, p.total, p.sigma);
    } else {
        eprint!(
            "\rstep {}/{}  sigma {:.3}  evaluated {}   ",
            p.step, p.total, p.sigma, p.evaluated
        );
    }
    if p.step == p.total {
        eprintln!();
    }
}

/// Load a greyscale image as a `[1, h, w, 1]` mask in `[0, 1]`.
pub fn load_mask(path: &str, width: usize, height: usize) -> Result<Array> {
    let img = image::open(path)
        .with_context(|| format!("opening {path}"))?
        .resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        )
        .to_luma8();
    let v: Vec<f32> = img.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    Ok(Array::from_slice_f32(&v, &[1, height, width, 1])?)
}

/// Load an image as `[1, h, w, 3]` in `[-1, 1]` — the VAE's range.
pub fn load_signed(path: &str, width: usize, height: usize) -> Result<Array> {
    let img = image::open(path)
        .with_context(|| format!("opening {path}"))?
        .resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();
    let v: Vec<f32> = img
        .pixels()
        .flat_map(|p| p.0)
        .map(|b| b as f32 / 127.5 - 1.0)
        .collect();
    Ok(Array::from_slice_f32(&v, &[1, height, width, 3])?)
}

/// Load an image as `[1, h, w, 3]` in `[0, 1]` — CLIP's range, and ESRGAN's.
///
/// **Not the same as [`load_signed`]**, and the two are the same shape and
/// dtype, so the wrong one is accepted and describes the wrong picture.
pub fn load_unit(path: &str, width: usize, height: usize) -> Result<Array> {
    let img = image::open(path)
        .with_context(|| format!("opening {path}"))?
        .resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();
    let v: Vec<f32> = img
        .pixels()
        .flat_map(|p| p.0)
        .map(|b| b as f32 / 255.0)
        .collect();
    Ok(Array::from_slice_f32(&v, &[1, height, width, 3])?)
}

/// Everything the txt2img command carries, already parsed.
pub struct Txt2ImgArgs {
    pub model: String,
    pub cfg: Txt2ImgConfig,
    pub output: String,
    pub sdxl: bool,
    pub lora: Option<(String, f64)>,
    pub controlnet: Vec<(String, String, f64)>,
    pub embeddings: Vec<(String, String)>,
    pub motion: Option<(String, usize)>,
    pub hires: Option<((usize, usize), f64)>,
    pub cache_threshold: f64,
    pub regions: Vec<(String, String)>,
    pub upscale: Option<String>,
}

/// Run txt2img, with whatever was attached.
pub fn run_txt2img(args: &Txt2ImgArgs, device: Device) -> Result<Vec<PathBuf>> {
    if args.sdxl {
        // SDXL has no adapter surface here yet; the flags that imply one are
        // refused rather than silently dropped.
        if !args.controlnet.is_empty()
            || args.lora.is_some()
            || args.motion.is_some()
            || !args.embeddings.is_empty()
        {
            bail!("--sdxl does not yet take LoRA, ControlNet, motion or embeddings on MLX");
        }
        let pipe = SdxlPipeline::load_on(Path::new(&args.model), device)?;
        let (w, h, bytes) = pipe.txt2img(&args.cfg)?;
        return write_images(&args.output, w, h, &bytes);
    }

    let mut pipe = MlxPipeline::load_on(Path::new(&args.model), device)?;
    if let Some((path, multiplier)) = &args.lora {
        let merged = pipe.attach_lora(Path::new(path), *multiplier)?;
        eprintln!("merged {merged} LoRA layers");
    }
    for (trigger, path) in &args.embeddings {
        let mut raw = load_safetensors(Path::new(path))?;
        // Textual inversion files store one tensor under a name that varies
        // by trainer — `emb_params`, `string_to_param`, others — so the tensor
        // is taken rather than looked up. More than one is ambiguous and said
        // so, instead of picking whichever the hash map yields first.
        if raw.len() != 1 {
            bail!(
                "{path} holds {} tensors; a textual-inversion file should hold one",
                raw.len()
            );
        }
        let name = raw.keys().next().expect("exactly one").clone();
        let vectors = raw.remove(&name).expect("just read");
        pipe.attach_embedding(trigger, vectors)?;
    }
    let mut hints = Vec::new();
    for (weights, map, scale) in &args.controlnet {
        pipe.attach_controlnet(Path::new(weights), *scale)?;
        hints.push(load_signed(map, args.cfg.width, args.cfg.height)?);
    }
    if let Some((path, frames)) = &args.motion {
        pipe.attach_motion(Path::new(path), *frames)?;
    }

    let regions = args
        .regions
        .iter()
        .map(|(mask, prompt)| -> Result<Region> {
            Ok(Region {
                mask: load_mask(mask, args.cfg.width, args.cfg.height)?,
                prompt: prompt.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Several ControlNets stack, and each wants its own map — but the pipeline
    // takes one hint for all of them, so more than one map is refused rather
    // than quietly using the first.
    if hints.len() > 1 {
        bail!(
            "MLX takes one control map for now; {} were given",
            hints.len()
        );
    }
    let hint = hints.first();

    let (w, h, bytes) = match &args.hires {
        Some(((hw, hh), strength)) => {
            if hint.is_some() || !regions.is_empty() {
                bail!("--hires does not yet combine with a control map or regions");
            }
            pipe.txt2img_hires(
                &args.cfg,
                *hw,
                *hh,
                Strength::new(*strength),
                &mut print_progress,
            )?
        }
        None => pipe.txt2img_with(
            &args.cfg,
            hint,
            None,
            &[],
            &regions,
            args.cache_threshold,
            Some(Cancel::new()),
            &mut print_progress,
        )?,
    };

    let (w, h, bytes) = match &args.upscale {
        Some(path) => {
            let weights = load_safetensors(Path::new(path))?;
            eprintln!("upscaling {w}x{h} by 4");
            pipe.upscale_bytes(w, h, &bytes, &weights)?
        }
        None => (w, h, bytes),
    };
    write_images(&args.output, w, h, &bytes)
}

/// img2img, and inpainting when a mask is given.
pub fn run_img2img(
    model: &str,
    cfg: &Txt2ImgConfig,
    init: &str,
    strength: f64,
    mask: Option<&str>,
    output: &str,
    device: Device,
) -> Result<Vec<PathBuf>> {
    let pipe = MlxPipeline::load_on(Path::new(model), device)?;
    let image = load_signed(init, cfg.width, cfg.height)?;
    let (w, h, bytes) = match mask {
        Some(m) => {
            let mask = load_mask(m, cfg.width, cfg.height)?;
            pipe.inpaint(cfg, &image, Strength::new(strength), &mask)?
        }
        None => pipe.img2img(cfg, &image, Strength::new(strength))?,
    };
    write_images(output, w, h, &bytes)
}

/// Upscale an existing image 4x.
pub fn run_upscale(
    model: &str,
    weights: &str,
    input: &str,
    output: &str,
    device: Device,
) -> Result<Vec<PathBuf>> {
    let dims = image::image_dimensions(input).with_context(|| format!("reading {input}"))?;
    let pipe = MlxPipeline::load_on(Path::new(model), device)?;
    let w = load_safetensors(Path::new(weights))?;
    // **`[0, 1]`, which is ESRGAN's range and not the VAE's.**
    let image = load_unit(input, dims.0 as usize, dims.1 as usize)?;
    let (uw, uh, bytes) = pipe.upscale(&image, &w)?;
    write_images(output, uw, uh, &bytes)
}
