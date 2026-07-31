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
use stable_diffusion_rs::mlx::{Cancel, MlxPipeline, Progress, Region, Request, SdxlPipeline};
use stable_diffusion_rs::pipeline::{Strength, Txt2ImgConfig};

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

/// Write RGB bytes as a PNG, with the generation parameters embedded.
///
/// Two things make the naming non-obvious, and both are about not silently
/// discarding a picture:
///
/// - **A clip is several frames.** With more than one, the name gains a frame
///   index: `clip.png` becomes `clip-000.png`, `clip-001.png`.
/// - **A batch is several images.** `batch` is `Some(i)` when the run produced
///   more than one, and every image of a batch is then indexed — including the
///   first. Indexing from the second onward, which is what "apply the suffix
///   when the index is non-zero" produces, gives `out.png` and `out-001.png`:
///   a set whose members are named by two different rules.
///
/// `parameters`, when given, is written as a PNG `tEXt` chunk under the key
/// A1111 and every viewer that follows it use. Written through `png` directly
/// rather than `image::save`, which has no way to attach one.
pub fn write_images(
    path: &str,
    width: usize,
    height: usize,
    rgb: &[u8],
    parameters: Option<&str>,
    batch: Option<usize>,
) -> Result<Vec<PathBuf>> {
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
        let out = if frames == 1 && batch.is_none() {
            base.to_path_buf()
        } else {
            let stem = base.file_stem().unwrap_or_default().to_string_lossy();
            let ext = base
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_else(|| "png".into());
            // Batch first, then frame: `out-001-000.png` is the second image's
            // first frame, which sorts the way a reader expects.
            let name = match (batch, frames) {
                (None, _) => format!("{stem}-{i:03}.{ext}"),
                (Some(b), 1) => format!("{stem}-{b:03}.{ext}"),
                (Some(b), _) => format!("{stem}-{b:03}-{i:03}.{ext}"),
            };
            base.with_file_name(name)
        };
        write_png(
            &out,
            width,
            height,
            &rgb[i * per..(i + 1) * per],
            parameters,
        )?;
        written.push(out);
    }
    Ok(written)
}

/// One PNG, with an optional `parameters` text chunk.
fn write_png(
    out: &Path,
    width: usize,
    height: usize,
    rgb: &[u8],
    parameters: Option<&str>,
) -> Result<()> {
    let file = std::fs::File::create(out).with_context(|| format!("creating {}", out.display()))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    if let Some(text) = parameters {
        // Non-fatal: an image that was generated is worth writing even if its
        // provenance chunk is not, and the only way this fails is a key or
        // value the PNG spec rejects.
        if let Err(e) = encoder.add_text_chunk("parameters".into(), text.into()) {
            eprintln!("warning: could not attach generation parameters: {e}");
        }
    }
    let mut writer = encoder
        .write_header()
        .with_context(|| format!("writing the header of {}", out.display()))?;
    writer
        .write_image_data(rgb)
        .with_context(|| format!("writing {}", out.display()))?;
    Ok(())
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
        let meta = metadata(&args.cfg, args.cfg.seed);
        return write_images(&args.output, w, h, &bytes, Some(&meta), None);
    }

    let mut pipe = MlxPipeline::load_with(Path::new(&args.model), device, args.cfg.precision)?;
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

    // A batch loads the model once and runs `seed`, `seed + 1`, ... so each
    // image is individually reproducible. `--hires` runs its own two passes and
    // has its own seed discipline, so it is a batch of one.
    let images = match &args.hires {
        Some(((hw, hh), strength)) => {
            if hint.is_some() || !regions.is_empty() {
                bail!("--hires does not yet combine with a control map or regions");
            }
            vec![pipe.txt2img_hires(
                &args.cfg,
                *hw,
                *hh,
                Strength::new(*strength),
                &mut print_progress,
            )?]
        }
        None => {
            let mut request = Request::new(&args.cfg)
                .regions(&regions)
                .cache_threshold(args.cache_threshold)
                .cancel(Cancel::new());
            if let Some(h) = hint {
                request = request.hint(h);
            }
            pipe.run_batch(request, &mut print_progress)?
        }
    };

    // `None` when there is one image, so a single run still writes the name it
    // was given rather than `out-000.png`.
    let indexed = images.len() > 1;
    let mut written = Vec::new();
    for (i, (w, h, bytes)) in images.into_iter().enumerate() {
        let (w, h, bytes) = match &args.upscale {
            Some(path) => {
                let weights = load_safetensors(Path::new(path))?;
                eprintln!("upscaling {w}x{h} by 4");
                pipe.upscale_bytes(w, h, &bytes, &weights)?
            }
            None => (w, h, bytes),
        };
        // The seed recorded is the one this image actually used, not the batch's
        // base — otherwise every image in a batch would claim to be the first.
        let meta = metadata(&args.cfg, args.cfg.seed_for(i));
        written.extend(write_images(
            &args.output,
            w,
            h,
            &bytes,
            Some(&meta),
            indexed.then_some(i),
        )?);
    }
    Ok(written)
}

/// The generation parameters, in the A1111 format every viewer already reads.
///
/// Written into the PNG so an image explains itself a year later. The format is
/// one prose line, then `Key: value` pairs comma-separated — not a design this
/// project chose, but the one that existing tooling parses, and a format nothing
/// reads is the same as no metadata.
pub fn metadata(cfg: &Txt2ImgConfig, seed: u64) -> String {
    let mut out = cfg.prompt.clone();
    if !cfg.negative_prompt.is_empty() {
        out.push_str(&format!("\nNegative prompt: {}", cfg.negative_prompt));
    }
    out.push_str(&format!(
        "\nSteps: {}, Sampler: {}, Schedule type: {}, CFG scale: {}, Seed: {}, \
         Size: {}x{}, Clip skip: {}, Model precision: {}, Version: sdrs {}",
        cfg.steps,
        cfg.sampler.name(),
        cfg.scheduler.name(),
        cfg.cfg_scale,
        seed,
        cfg.width,
        cfg.height,
        cfg.clip_skip.get(),
        cfg.precision.name(),
        stable_diffusion_rs::VERSION,
    ));
    out
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
    let pipe = MlxPipeline::load_with(Path::new(model), device, cfg.precision)?;
    let image = load_signed(init, cfg.width, cfg.height)?;
    let (w, h, bytes) = match mask {
        Some(m) => {
            let mask = load_mask(m, cfg.width, cfg.height)?;
            pipe.inpaint(cfg, &image, Strength::new(strength), &mask)?
        }
        None => pipe.img2img(cfg, &image, Strength::new(strength))?,
    };
    write_images(output, w, h, &bytes, None, None)
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
    write_images(output, uw, uh, &bytes, None, None)
}

/// Merge two checkpoints by weighted average.
///
/// **On the CPU regardless of `--cpu`.** This is file arithmetic: moving
/// gigabytes onto the GPU to add them pays a transfer for no gain, and the
/// result goes straight back out to disk.
pub fn run_merge(
    a: &str,
    b: &str,
    alpha: f64,
    allow_unmatched: bool,
    output: &str,
) -> Result<stable_diffusion_rs::models::mlx::merge::Merged> {
    use stable_diffusion_rs::models::mlx::merge::{merge, MergeOptions};
    use stable_diffusion_rs::tensor::mlx::{save_safetensors, Stream};

    let s = Stream::cpu();
    let left = load_safetensors(Path::new(a)).with_context(|| format!("reading {a}"))?;
    let right = load_safetensors(Path::new(b)).with_context(|| format!("reading {b}"))?;
    eprintln!(
        "merging {} tensors with {} at alpha {alpha}",
        left.len(),
        right.len()
    );
    let (merged, report) = merge(
        &left,
        &right,
        &MergeOptions {
            alpha,
            allow_unmatched,
        },
        &s,
    )?;
    save_safetensors(Path::new(output), &merged).with_context(|| format!("writing {output}"))?;
    Ok(report)
}

/// Edit an image by instruction.
pub fn run_instruct(
    model: &str,
    cfg: &Txt2ImgConfig,
    init: &str,
    image_guidance: f64,
    output: &str,
    device: Device,
) -> Result<Vec<PathBuf>> {
    let pipe = MlxPipeline::load_with(Path::new(model), device, cfg.precision)?;
    // **`[-1, 1]`, the VAE's range** — not CLIP's `[0, 1]`. The two are the
    // same shape and dtype.
    let image = load_signed(init, cfg.width, cfg.height)?;
    let (w, h, bytes) = pipe.instruct_with(cfg, &image, image_guidance, &mut print_progress)?;
    let meta = metadata(cfg, cfg.seed);
    write_images(output, w, h, &bytes, Some(&meta), None)
}

/// Everything the `flux` command carries.
pub struct FluxArgs {
    pub model: String,
    pub variant: String,
    pub cfg: stable_diffusion_rs::mlx::FluxRunConfig,
    pub bits: usize,
    pub transformer_gguf: Option<String>,
    pub t5_gguf: Option<String>,
    pub clip: Option<String>,
    pub vae: Option<String>,
    pub t5_tokenizer: Option<String>,
    pub output: String,
}

/// Run Flux.
///
/// Two ways in, and which one applies is decided by whether a transformer GGUF
/// was named rather than by probing the directory — a checkpoint half-assembled
/// from both is a confusing thing to debug.
pub fn run_flux(args: &FluxArgs, device: Device) -> Result<Vec<PathBuf>> {
    use stable_diffusion_rs::mlx::{FluxPaths, FluxPipeline};
    use stable_diffusion_rs::models::mlx::flux::FluxConfig;

    let cfg = match args.variant.as_str() {
        "schnell" => FluxConfig::schnell(),
        "dev" => FluxConfig::dev(),
        "mini" => FluxConfig::mini(),
        other => bail!("unknown Flux variant {other:?}; try schnell, dev or mini"),
    };
    // schnell distils guidance into the model, so there is nothing to scale.
    // Refused rather than ignored: a user who passed it believes it did
    // something.
    if !cfg.guidance_embed && args.cfg.guidance != FluxRunDefaults::GUIDANCE {
        bail!(
            "--guidance {} was given, but {} has no guidance embedding; it is distilled in. \
             Drop the flag, or use --variant dev",
            args.cfg.guidance,
            args.variant
        );
    }

    let root = Path::new(&args.model);
    let defaults = FluxPaths::in_dir(root);
    let pipe = match (&args.transformer_gguf, &args.t5_gguf) {
        (Some(tr), Some(t5)) => {
            let clip = args
                .clip
                .as_ref()
                .map(PathBuf::from)
                .unwrap_or(defaults.clip);
            let vae = args.vae.as_ref().map(PathBuf::from).unwrap_or(defaults.vae);
            let t5_tok = args
                .t5_tokenizer
                .as_ref()
                .map(PathBuf::from)
                .unwrap_or(defaults.t5_tokenizer);
            eprintln!("loading Flux from GGUF at {} bits", args.bits);
            FluxPipeline::from_gguf(
                Path::new(tr),
                Path::new(t5),
                &clip,
                &vae,
                &defaults.clip_tokenizer,
                &t5_tok,
                cfg,
                args.bits,
                device,
            )?
        }
        _ => {
            eprintln!("loading Flux from {} at {} bits", args.model, args.bits);
            FluxPipeline::load_quantized(root, cfg, args.bits, device)?
        }
    };
    report_memory();
    let (w, h, bytes) = pipe.txt2img(&args.cfg)?;
    write_images(&args.output, w, h, &bytes, None, None)
}

/// The default `--guidance`, so that "was it given?" can be answered without
/// making the flag an `Option` in the parser.
struct FluxRunDefaults;
impl FluxRunDefaults {
    const GUIDANCE: f64 = 3.5;
}

/// Everything the `sd3` command carries.
pub struct Sd3Args {
    pub model: String,
    pub cfg: stable_diffusion_rs::mlx::Sd3RunConfig,
    pub bits: usize,
    pub transformer: Option<String>,
    pub vae: Option<String>,
    pub clip_l: Option<String>,
    pub clip_g: Option<String>,
    pub t5: Option<String>,
    pub t5_tokenizer: Option<String>,
    pub output: String,
}

/// Run SD 3.5.
pub fn run_sd3(args: &Sd3Args, device: Device) -> Result<Vec<PathBuf>> {
    use stable_diffusion_rs::mlx::{Sd3Paths, Sd3Pipeline};

    let root = Path::new(&args.model);
    let d = Sd3Paths::in_dir(root);
    let pick = |given: &Option<String>, fallback: PathBuf| -> PathBuf {
        given.as_ref().map(PathBuf::from).unwrap_or(fallback)
    };
    // The published SD 3.5 directory is incomplete, so `from_parts` is the
    // ordinary path rather than the escape hatch — every piece may be
    // overridden and the directory supplies whatever was not.
    eprintln!("loading SD 3.5 at {} bits", args.bits);
    let pipe = Sd3Pipeline::from_parts(
        &pick(&args.transformer, d.transformer),
        &pick(&args.vae, d.vae),
        &pick(&args.clip_l, d.clip_l),
        &pick(&args.clip_g, d.clip_g),
        &pick(&args.t5, d.t5.first().cloned().unwrap_or_default()),
        &d.clip_tokenizer,
        &pick(&args.t5_tokenizer, d.t5_tokenizer),
        args.bits,
        device,
    )?;
    report_memory();
    let (w, h, bytes) = pipe.txt2img(&args.cfg)?;
    write_images(&args.output, w, h, &bytes, None, None)
}

/// Run unCLIP: a variation of an image, or a prompt through the prior.
#[allow(clippy::too_many_arguments)]
pub fn run_unclip(
    model: &str,
    cfg: &Txt2ImgConfig,
    image: Option<&str>,
    noise_level: usize,
    output: &str,
    device: Device,
) -> Result<Vec<PathBuf>> {
    let pipe = MlxPipeline::load_unclip_on(Path::new(model), device)?;
    let (w, h, bytes) = match image {
        // CLIP's range, `[0, 1]` — **not** the VAE's `[-1, 1]`. The two are the
        // same shape and dtype, so the wrong one describes a different picture.
        Some(path) => {
            let px = load_unit(path, 224, 224)?;
            pipe.variation(cfg, &px, noise_level)?
        }
        None => pipe.txt2img(cfg)?,
    };
    write_images(output, w, h, &bytes, None, None)
}

/// What MLX is actually holding, after a load.
///
/// Reported because the whole point of quantising at rest is a number, and
/// "it loaded" is not one. This is MLX's own accounting rather than resident
/// set size, which on unified memory also counts the memory-mapped checkpoint.
fn report_memory() {
    if let Ok(active) = sd_tensor::mlx::active_memory() {
        eprintln!("resident: {}", sd_tensor::ops::human_bytes(active as u64));
    }
}
