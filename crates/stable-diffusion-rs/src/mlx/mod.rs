//! The generation pipeline on MLX.
//!
//! Every model this needs is already ported and gated against diffusers. What
//! is here is the orchestration between them — and it is the part a caller
//! actually invokes, so until it exists the MLX port is a set of verified
//! pieces rather than a working backend.
//!
//! # What is shared with the candle pipeline, and why
//!
//! The configuration types — [`Txt2ImgConfig`], [`SamplerKind`], [`Strength`],
//! [`Progress`] — are plain data and touch no tensor, so they are *imported*
//! rather than mirrored. So is `sd_sample::Schedule`, which returns `Vec<f64>`.
//! A second copy of any of them is how the two backends would come to disagree
//! about what a seed or a strength means.
//!
//! What is genuinely rewritten is the tensor work: the guidance batch, the
//! sampler step, and the NCHW/NHWC boundary.
//!
//! # The one thing that must not drift: the noise
//!
//! Noise is drawn through `SeededRng` on the CPU and handed to MLX as plain
//! data, exactly as `mlx_end_to_end` does. That is deliberate rather than
//! lazy: it makes the two backends see **identical draws**, so any difference
//! between their images is the models and not the dice. Drawing on the GPU
//! would be faster and would make every cross-backend comparison meaningless.

use std::path::{Path, PathBuf};

use sd_models::clip::ClipTokenizer;
use sd_models::mlx::{
    clip, clip_vision, controlnet, gligen, ip, lora::Lora, motion, normalise_legacy_attention,
    sample, taesd, timestep_embedding, unclip, unet_forward_adapters, vae, Adapters, Motion,
    UNetConfig, Weights,
};
use sd_sample::{steps, Schedule};
use sd_tensor::mlx::{concat, load_safetensors, Array, Device, Stream};
use sd_tensor::rng::SeededRng;

use crate::pipeline::{
    cache_rescale, PipelineError, Precision, SamplerKind, Strength, Txt2ImgConfig,
};

pub mod flux;
pub mod sd3;
pub mod sdxl;

pub use flux::{FluxPaths, FluxPipeline, FluxRunConfig};
pub use sd3::{Sd3Paths, Sd3Pipeline, Sd3RunConfig, SkipLayerGuidance};
pub use sdxl::SdxlPipeline;

/// `PipelineError` carries no free-form variant of its own, so a message goes
/// through the tensor error the way the candle pipeline's do.
pub(crate) fn msg(text: String) -> PipelineError {
    PipelineError::Tensor(sd_tensor::Error::Msg(text))
}

/// The discrete training timestep nearest a continuous sigma.
///
/// The UNet takes a training timestep, not a sigma. Handing it the sigma runs —
/// both are one number — and conditions on the wrong point of the schedule
/// entirely.
pub(crate) fn timestep_for(schedule: &Schedule, sigma: f64) -> f32 {
    schedule
        .sigmas()
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - sigma)
                .abs()
                .partial_cmp(&(*b - sigma).abs())
                .expect("finite sigmas")
        })
        .map(|(i, _)| i as f32)
        .unwrap_or(0.0)
}

/// A standard-normal draw of `[1, c, h, w]`, delivered in NHWC.
///
/// **Through `SeededRng` on the CPU**, which is the module's one deliberate
/// inefficiency: it makes both backends see identical draws, so a difference
/// between their images is the models and not the dice.
pub(crate) fn draw_noise(
    rng: &mut SeededRng,
    c: usize,
    h: usize,
    w: usize,
) -> Result<Array, PipelineError> {
    Ok(sd_tensor::rng::randn_nhwc(rng, 1, c, h, w)?)
}

/// `<|startoftext|>` and `<|endoftext|>` in CLIP's vocabulary.
const BOS: i32 = 49406;
const EOS: i32 = 49407;

/// Where a checkpoint's four pieces live under a model directory.
///
/// Laid out as `diffusers` does, which is what `SD_TEST_MODEL_DIR` points at.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub unet: PathBuf,
    pub vae: PathBuf,
    pub text_encoder: PathBuf,
    pub tokenizer: PathBuf,
}

impl ModelPaths {
    /// The `diffusers` layout under `root`.
    pub fn in_dir(root: &Path) -> Self {
        Self {
            unet: root.join("unet/diffusion_pytorch_model.safetensors"),
            vae: root.join("vae/diffusion_pytorch_model.safetensors"),
            text_encoder: root.join("text_encoder/model.safetensors"),
            tokenizer: root.join("tokenizer"),
        }
    }

    /// Every path, for an existence check that names what is missing rather
    /// than failing at whichever one is loaded first.
    pub fn missing(&self) -> Vec<&Path> {
        // **The tokenizer is not on this list.** It cannot be missing:
        // `ClipTokenizer::open` reads whichever form the checkpoint ships and
        // falls back to the vocabulary vendored in `sd-models` when it ships
        // none. Listing it here would refuse to load a model over a file that
        // is not needed.
        [
            self.unet.as_path(),
            self.vae.as_path(),
            self.text_encoder.as_path(),
        ]
        .into_iter()
        .filter(|p| !p.exists())
        .collect()
    }
}

/// One ControlNet and how hard it steers.
///
/// `scale` multiplies every correction; **0 contributes exactly nothing**
/// rather than merely almost nothing, which is what makes it a usable off
/// switch.
pub struct Control {
    pub weights: Weights,
    pub scale: f64,
}

/// An attached IP-Adapter: its own weights, the image tower's, and the scale.
///
/// The adapter's attention weights live in the same map as `image_proj` — it
/// ships as one file — but the **vision tower is a separate checkpoint**, and a
/// large one. Held apart so a run that never conditions on an image does not
/// pay for it.
pub struct IpAdapter {
    weights: Weights,
    vision: Weights,
    scale: f32,
}

/// One grounded box for GLIGEN: where, and what.
///
/// Coordinates are `[x0, y0, x1, y1]` in `[0, 1]`, **not pixels** — the model
/// was trained on normalised boxes and pixel values put every box off the
/// canvas without an error.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundedBox {
    pub bbox: [f32; 4],
    pub phrase: String,
}

/// Progress after one step.
///
/// `denoised` is the model's estimate of the **finished image** as a latent —
/// not the sampler's latent, and that difference is the whole value of the
/// field. The latent at step 5 of 20 is `x0 + sigma*noise` with sigma still
/// around 4, so decoding it shows noise; this is the `x0` the model predicts,
/// which decodes blurry and sharpens as the run proceeds.
pub struct Progress<'a> {
    /// 1-based; equal to `total` on the last step.
    pub step: usize,
    pub total: usize,
    pub sigma: f64,
    pub denoised: &'a Array,
    /// How many steps so far actually ran the model rather than reusing a
    /// cached prediction. Equal to `step` when caching is off.
    pub evaluated: usize,
}

/// A callback invoked after each step.
pub type ProgressFn<'a> = &'a mut dyn FnMut(Progress<'_>);

/// One region of the canvas, with its own prompt.
pub struct Region {
    /// `[1, h, w, 1]` in `[0, 1]` at **pixel** resolution. Downsampled to the
    /// latent grid by **mean**, not max: a region boundary should fade over a
    /// latent cell rather than claim it outright, which is the opposite of
    /// what an inpainting mask wants and is worth not copying by reflex.
    pub mask: Array,
    pub prompt: String,
}

/// One generation, with everything optional named.
///
/// The public counterpart of `Extras`. Built by chaining, so the common call
/// stays `Request::new(&cfg)` and the rare one names what it adds — which is
/// the property the eight-parameter `txt2img_with` did not have, where two
/// adjacent `Option<&Array>` meant a control map and a reference image were
/// interchangeable to the compiler and not to the model.
pub struct Request<'a> {
    pub(crate) cfg: &'a Txt2ImgConfig,
    pub(crate) hint: Option<&'a Array>,
    pub(crate) reference: Option<&'a Array>,
    pub(crate) boxes: &'a [GroundedBox],
    pub(crate) regions: &'a [Region],
    pub(crate) cache_threshold: f64,
    pub(crate) cancel: Option<Cancel>,
}

impl<'a> Request<'a> {
    pub fn new(cfg: &'a Txt2ImgConfig) -> Self {
        Self {
            cfg,
            hint: None,
            reference: None,
            boxes: &[],
            regions: &[],
            cache_threshold: 0.0,
            cancel: None,
        }
    }

    /// A ControlNet's control map, `[1, h, w, 3]` in **`[-1, 1]`**.
    pub fn hint(mut self, hint: &'a Array) -> Self {
        self.hint = Some(hint);
        self
    }

    /// An IP-Adapter reference image, `[1, h, w, 3]` in **`[0, 1]`** — CLIP's
    /// range, not the VAE's. The two are the same shape and dtype.
    pub fn reference(mut self, reference: &'a Array) -> Self {
        self.reference = Some(reference);
        self
    }

    /// GLIGEN grounding boxes, in normalised `[0, 1]` coordinates.
    pub fn boxes(mut self, boxes: &'a [GroundedBox]) -> Self {
        self.boxes = boxes;
        self
    }

    /// Per-region prompts, blended into the prediction before each step.
    pub fn regions(mut self, regions: &'a [Region]) -> Self {
        self.regions = regions;
        self
    }

    /// Reuse the model's prediction while it is estimated not to have moved
    /// much. **Predicted relative change accumulated since the last real
    /// evaluation**, so 0.2 means "reuse until the output is estimated to have
    /// drifted 20%" — a statement about the model rather than an arbitrary
    /// dial. 0 disables it bit-identically, and an ancestral sampler refuses it.
    pub fn cache_threshold(mut self, threshold: f64) -> Self {
        self.cache_threshold = threshold;
        self
    }

    pub fn cancel(mut self, cancel: Cancel) -> Self {
        self.cancel = Some(cancel);
        self
    }
}

/// The per-run conditioning that is not the prompt.
///
/// Bundled because the sampling loop needs all of it and none of it belongs to
/// [`Txt2ImgConfig`], which is shared with the candle pipeline and holds no
/// tensors.
struct Extras<'a> {
    /// A ControlNet's control map, `[1, h, w, 3]` in `[-1, 1]`.
    hint: Option<&'a Array>,
    /// The IP-Adapter's four tokens, already doubled for the guidance batch.
    ip_tokens: Option<Array>,
    /// GLIGEN's grounding tokens, likewise doubled.
    objs: Option<&'a Array>,
    /// Per-region prompts, blended into the noise prediction each step.
    regions: &'a [Region],
    /// Reuse the model's prediction between steps while it is estimated not to
    /// have moved much. 0 disables it bit-identically.
    cache_threshold: f64,
    /// Checked once per step. A step is not interruptible internally, so
    /// cancelling costs at most one step of latency.
    cancel: Option<Cancel>,
    /// Wrap every convolution at the image edge, so the result tiles.
    seamless: bool,
}

impl Default for Extras<'_> {
    fn default() -> Self {
        Self {
            hint: None,
            ip_tokens: None,
            objs: None,
            regions: &[],
            cache_threshold: 0.0,
            cancel: None,
            seamless: false,
        }
    }
}

/// Replace one sequence position of `[1, seq, width]`.
///
/// Rebuilt rather than written in place: MLX arrays are immutable, so this
/// concatenates the part before, the new row, and the part after.
fn splice_row(embeds: &Array, row: &Array, at: usize, s: &Stream) -> Result<Array, PipelineError> {
    let [_, seq, _] = embeds.shape()[..] else {
        return Err(msg(format!("mlx: embeds {:?}", embeds.shape())));
    };
    if at >= seq {
        return Err(msg(format!(
            "mlx: position {at} is past the sequence's {seq}"
        )));
    }
    let mut parts: Vec<Array> = Vec::with_capacity(3);
    if at > 0 {
        parts.push(embeds.narrow(1, 0, at, s)?);
    }
    parts.push(row.contiguous(s)?);
    if at + 1 < seq {
        parts.push(embeds.narrow(1, at + 1, seq - at - 1, s)?);
    }
    let refs: Vec<&Array> = parts.iter().collect();
    Ok(concat(&refs, 1, s)?)
}

/// Nearest-neighbour 2x-and-beyond upsample of a latent, `[n, h, w, c]`.
///
/// Integer scaling only, which is what a hires pass wants: it is
/// `broadcast_to` between two reshapes, so each source cell is copied into an
/// exact block and no intermediate value is invented.
fn nearest_upsample(
    x: &Array,
    out_h: usize,
    out_w: usize,
    s: &Stream,
) -> Result<Array, PipelineError> {
    let [n, h, w, c] = x.shape()[..] else {
        return Err(msg(format!("mlx: upsample got {:?}", x.shape())));
    };
    if out_h % h != 0 || out_w % w != 0 {
        return Err(msg(format!(
            "mlx: {h}x{w} does not scale to {out_h}x{out_w} by an integer factor; a hires \
             pass wants a whole multiple so no intermediate value is invented"
        )));
    }
    let (fh, fw) = (out_h / h, out_w / w);
    Ok(x.reshape(&[n, h, 1, w, 1, c], s)?
        .broadcast_to(&[n, h, fh, w, fw, c], s)?
        .contiguous(s)?
        .reshape(&[n, out_h, out_w, c], s)?)
}

/// Reduce a pixel-resolution region mask to the latent grid by **mean**.
///
/// Not max. `sample::latent_mask` uses max because an inpaint needs a cell
/// freed if *any* pixel under it is writeable; a region wants the opposite —
/// a boundary that fades across a cell rather than claiming it. Reusing the
/// inpainting reduction here gives every region a hard edge at latent
/// resolution.
fn mean_pool_to_latent(mask_px: &Array, s: &Stream) -> Result<Array, PipelineError> {
    let [n, h, w, c] = mask_px.shape()[..] else {
        return Err(msg(format!(
            "mlx: a region mask should be [n, h, w, 1], got {:?}",
            mask_px.shape()
        )));
    };
    if h % 8 != 0 || w % 8 != 0 {
        return Err(msg(format!(
            "mlx: a {h}x{w} region mask does not divide into latent cells"
        )));
    }
    Ok(mask_px
        .reshape(&[n, h / 8, 8, w / 8, 8, c], s)?
        .mean(&[2, 4], false, s)?)
}

/// A cancellation token, shared with whatever wants to stop a generation.
///
/// A token rather than a callback return value, so the ordinary progress
/// callback stays a plain `FnMut` and callers who never cancel write nothing.
#[derive(Debug, Clone, Default)]
pub struct Cancel(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the generation to stop. Safe to call from any thread.
    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// A loaded SD 1.5-family pipeline on MLX.
pub struct MlxPipeline {
    tokenizer: ClipTokenizer,
    text_encoder: Weights,
    unet: Weights,
    vae: Weights,
    cfg: UNetConfig,
    /// **The text tower's geometry differs by checkpoint.** SD 1.5 is CLIP-L
    /// at 768; unCLIP conditions on SD 2.x's OpenCLIP ViT-H at 1024. Reusing
    /// SD 1.5's here fails at the first reshape, which is the loud direction —
    /// but only because the widths disagree, not because anything checks.
    clip_cfg: clip::ClipConfig,
    vae_cfg: vae::VaeConfig,
    schedule: Schedule,
    stream: Stream,
    device: Device,
    /// Spatial conditioning, in attachment order. Empty is the common case.
    controlnets: Vec<Control>,
    ip: Option<IpAdapter>,
    /// A tiny distilled decoder, used in place of the VAE when attached.
    taesd: Option<Weights>,
    /// unCLIP's image conditioning: the normalizer's statistics and the
    /// vision tower. Only an unCLIP checkpoint has these.
    unclip: Option<(Weights, Weights)>,
    /// Textual-inversion embeddings, spliced into prompts by trigger word.
    ///
    /// Each is `(trigger, [vectors, width])`.
    embeddings: Vec<(String, Array)>,
    /// An AnimateDiff motion adapter, and the clip length it will run at.
    ///
    /// The adapter is a separate checkpoint from the UNet, so it is held apart
    /// rather than merged in — and a run with no adapter pays nothing.
    motion: Option<(Weights, usize)>,
}

impl MlxPipeline {
    /// Load SD 1.5 from a `diffusers` model directory, on the GPU.
    pub fn load(root: &Path) -> Result<Self, PipelineError> {
        Self::load_on(root, Device::default())
    }

    /// [`Self::load`] on a named device, at a named precision.
    ///
    /// **f16 is a different sample, not a worse one.** Measured on SD 1.5 at
    /// 768x768: 1.10x faster, 1.15 GB smaller, PSNR 36.8 dB against the f32
    /// image. It is not the default because every golden test in this project
    /// is verified at f32 tolerances, and because SD 1.5 fits either way — the
    /// residency saving is the interesting half, and that matters for SDXL and
    /// the DiTs rather than here.
    pub fn load_with(
        root: &Path,
        device: Device,
        precision: Precision,
    ) -> Result<Self, PipelineError> {
        let mut pipe = Self::load_on(root, device)?;
        if precision == Precision::F16 {
            pipe.cast_to_f16()?;
        }
        Ok(pipe)
    }

    /// Cast the diffusion weights to f16, in place.
    ///
    /// **The UNet and the VAE only.** The text encoder stays f32: CLIP's
    /// activations peak around 851, and while that is inside f16's range the
    /// conditioning is computed once per run rather than once per step, so
    /// casting it saves nothing measurable and spends the one place in the
    /// pipeline where a magnitude problem has already been found once.
    fn cast_to_f16(&mut self) -> Result<(), PipelineError> {
        let s = &self.stream;
        for w in [&mut self.unet, &mut self.vae] {
            for v in w.values_mut() {
                *v = v.to_f16(s)?;
            }
        }
        Ok(())
    }

    /// [`Self::load`] on a named device.
    ///
    /// The GPU is the point — a diffusion step is thousands of matmuls — but
    /// the choice belongs to whoever runs the program, not to a constant.
    pub fn load_on(root: &Path, device: Device) -> Result<Self, PipelineError> {
        let paths = ModelPaths::in_dir(root);
        let missing = paths.missing();
        if !missing.is_empty() {
            return Err(msg(format!(
                "mlx: {} is missing {}",
                root.display(),
                missing
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let stream = Stream::for_device(device);
        let tokenizer = ClipTokenizer::open(&paths.tokenizer)?;

        // **The stock checkpoint may use the legacy attention names.** The
        // decoder asks for `to_q` and a published VAE has `query`; converting
        // once here is what keeps every model module free of the concern.
        let mut vae = load_safetensors(&paths.vae)?;
        normalise_legacy_attention(&mut vae);
        let mut unet = load_safetensors(&paths.unet)?;
        normalise_legacy_attention(&mut unet);

        Ok(Self {
            tokenizer,
            text_encoder: load_safetensors(&paths.text_encoder)?,
            unet,
            vae,
            cfg: UNetConfig::sd15(),
            clip_cfg: clip::ClipConfig::sd15(),
            vae_cfg: vae::VaeConfig::sd15(),
            schedule: Schedule::sd15(),
            stream,
            device,
            controlnets: Vec::new(),
            ip: None,
            taesd: None,
            unclip: None,
            embeddings: Vec::new(),
            motion: None,
        })
    }

    /// Load an unCLIP checkpoint, whose UNet conditions on a CLIP **image**
    /// embedding rather than only on text.
    ///
    /// `UNetConfig::unclip()` sets `class_projection`, so the UNet refuses to
    /// run without one — which is the loud direction. The normalizer is
    /// mandatory and the image tower is not: text-to-image unCLIP checkpoints
    /// ship no `image_encoder` at all, because a prompt is their only input.
    pub fn load_unclip(root: &Path) -> Result<Self, PipelineError> {
        Self::load_unclip_on(root, Device::default())
    }

    /// [`Self::load_unclip`] on a named device.
    pub fn load_unclip_on(root: &Path, device: Device) -> Result<Self, PipelineError> {
        let mut pipe = Self::load_on(root, device)?;
        pipe.cfg = UNetConfig::unclip();
        // unCLIP's text encoder is SD 2.x's OpenCLIP ViT-H — 1024 wide, 23
        // layers, plain gelu — not SD 1.5's CLIP-L.
        pipe.clip_cfg = clip::ClipConfig::sd2();

        let normalizer = root.join("image_normalizer/diffusion_pytorch_model.safetensors");
        if !normalizer.exists() {
            return Err(PipelineError::MissingFile(normalizer));
        }
        let encoder = root.join("image_encoder/model.safetensors");
        let vision = if encoder.exists() {
            load_safetensors(&encoder)?
        } else {
            // A text-to-image unCLIP has no tower. Asking it for a variation
            // is then a clear error rather than a missing file.
            Weights::new()
        };
        pipe.unclip = Some((load_safetensors(&normalizer)?, vision));
        Ok(pipe)
    }

    /// An image variation: condition on a picture rather than only a prompt.
    ///
    /// `image` is `[1, 224, 224, 3]` in **`[0, 1]`** — CLIP's range. `level`
    /// says how much noise to add to the embedding before conditioning on it;
    /// higher means the model is told to trust it less, which is how unCLIP
    /// trades fidelity for variety.
    pub fn variation(
        &self,
        cfg: &Txt2ImgConfig,
        image: &Array,
        level: usize,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        let s = &self.stream;
        let Some((normalizer, vision)) = &self.unclip else {
            return Err(msg(
                "mlx: this is not an unCLIP pipeline; load it with `load_unclip`".into(),
            ));
        };
        if vision.is_empty() {
            return Err(msg(
                "mlx: this unCLIP checkpoint ships no image_encoder, so it cannot read a \
                 reference image — it is a text-to-image variant"
                    .into(),
            ));
        }
        let pixels = clip_vision::preprocess(image, s)?;
        let embeds =
            clip_vision::image_embeds(&pixels, &clip_vision::VisionConfig::vit_h_14(), vision, s)?;

        let mut rng = SeededRng::new(cfg.seed);
        let dim = embeds.shape()[1];
        // `[1, dim]`, drawn in the same order the image path draws in so a
        // seed means one thing across the crate.
        let noise = Array::from_slice_f32(&rng.normals(dim), &[1, dim])?;

        let alphas = sd_models::unclip::cosine_alphas_cumprod(unclip::TRAIN_TIMESTEPS);
        let conditioned = unclip::augment(&embeds, level, &noise, &alphas, normalizer, s)?;
        // **The unconditional row is zeros of the whole width**, not an
        // augmented zero embedding.
        let uncond = unclip::unconditional(1, dim)?;
        let class_embeds = concat(&[&uncond, &conditioned], 0, s)?;

        self.generate_with_class(cfg, Some(&class_embeds), &mut rng)
    }

    /// Upscale an image 4x with Real-ESRGAN.
    ///
    /// **`image` is `[1, h, w, 3]` in `[0, 1]`, not the VAE's `[-1, 1]`.**
    /// The two are the same shape and dtype, so the wrong range is accepted
    /// and returns a washed-out picture with no error — the same trap
    /// `clip_vision::preprocess` carries. [`Self::upscale_bytes`] takes the
    /// bytes `txt2img` returns and does the conversion.
    ///
    /// Separate from the pipeline's own weights, because an upscaler is a
    /// different model: loading it costs what a caller who never upscales
    /// should not pay.
    pub fn upscale(
        &self,
        image: &Array,
        weights: &Weights,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        let s = &self.stream;
        let out = sd_models::mlx::esrgan::upscale(image, weights, s)?;
        let [_, h, w, _] = out.shape()[..] else {
            return Err(msg(format!("mlx: esrgan returned {:?}", out.shape())));
        };
        // `[0, 1]` out, matching what went in.
        let bytes = out
            .to_vec_f32(s)?
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        Ok((w, h, bytes))
    }

    /// [`Self::upscale`] over the RGB bytes the generation entry points
    /// return, so a caller never has to name a range at all.
    pub fn upscale_bytes(
        &self,
        width: usize,
        height: usize,
        rgb: &[u8],
        weights: &Weights,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        if rgb.len() != width * height * 3 {
            return Err(msg(format!(
                "mlx: {} bytes is not {width}x{height} RGB",
                rgb.len()
            )));
        }
        let unit: Vec<f32> = rgb.iter().map(|&b| b as f32 / 255.0).collect();
        let image = Array::from_slice_f32(&unit, &[1, height, width, 3])?;
        self.upscale(&image, weights)
    }

    /// Attach a textual-inversion embedding under a trigger word.
    ///
    /// **The width is checked here**, because an SDXL embedding in an SD 1.5
    /// prompt is the common mistake and would otherwise surface as a shape
    /// error from deep inside the transformer.
    pub fn attach_embedding(&mut self, trigger: &str, vectors: Array) -> Result<(), PipelineError> {
        let [n, width] = vectors.shape()[..] else {
            return Err(msg(format!(
                "mlx: an embedding should be [vectors, width], got {:?}",
                vectors.shape()
            )));
        };
        if width != self.clip_cfg.hidden {
            return Err(msg(format!(
                "mlx: the embedding `{trigger}` is {width} wide and this text encoder is {}; \
                 an SDXL embedding in an SD 1.5 prompt is the usual cause",
                self.clip_cfg.hidden
            )));
        }
        if n == 0 {
            return Err(msg(format!(
                "mlx: the embedding `{trigger}` has no vectors"
            )));
        }
        self.embeddings.push((trigger.to_string(), vectors));
        Ok(())
    }

    /// Attach an AnimateDiff motion adapter and fix the clip length.
    ///
    /// **Frames ride on the batch axis**, so a clip of `f` frames makes every
    /// activation `[f, h, w, c]` and every step `f` times the work. The count
    /// is fixed at attach time rather than per call because the modules are
    /// installed into the UNet's blocks and every one of them has to agree
    /// about it.
    ///
    /// The adapter is refused if it carries no motion modules, rather than
    /// installing nothing and rendering `f` unrelated images.
    pub fn attach_motion(&mut self, path: &Path, frames: usize) -> Result<(), PipelineError> {
        if frames == 0 {
            return Err(msg("mlx: a clip of zero frames".into()));
        }
        let weights = load_safetensors(path)?;
        if !motion::present(&weights, "down_blocks.0.motion_modules.0") {
            return Err(msg(format!(
                "mlx: {} carries no motion modules; an AnimateDiff adapter names them \
                 `down_blocks.0.motion_modules.0` and so on",
                path.display()
            )));
        }
        self.motion = Some((weights, frames));
        Ok(())
    }

    /// How many frames a run will produce. 1 with no adapter attached.
    pub fn frames(&self) -> usize {
        self.motion.as_ref().map_or(1, |(_, f)| *f)
    }

    /// Attach an IP-Adapter.
    ///
    /// `adapter` carries `image_proj` and the per-layer `to_k_ip`/`to_v_ip`
    /// weights; `image_encoder` is the CLIP vision tower, which is a separate
    /// and much larger checkpoint.
    ///
    /// **Scale 0 contributes exactly nothing**, so it is a usable off switch
    /// rather than an approximation of one.
    pub fn attach_ip_adapter(
        &mut self,
        adapter: &Path,
        image_encoder: &Path,
        scale: f64,
    ) -> Result<(), PipelineError> {
        self.ip = Some(IpAdapter {
            weights: load_safetensors(adapter)?,
            vision: load_safetensors(image_encoder)?,
            scale: scale as f32,
        });
        Ok(())
    }

    /// The adapter's four tokens for a reference image.
    ///
    /// `image` is `[1, 224, 224, 3]` in **`[0, 1]`** — CLIP's own range, not
    /// the `[-1, 1]` a VAE uses. The wrong range is accepted and describes the
    /// wrong picture.
    fn ip_tokens(&self, image: &Array) -> Result<Option<Array>, PipelineError> {
        let Some(adapter) = &self.ip else {
            return Ok(None);
        };
        let s = &self.stream;
        let pixels = clip_vision::preprocess(image, s)?;
        // The **projected** embedding, 1024 wide for ViT-H — not the pooled
        // 1280, which is a different vector of a different width.
        let embeds = clip_vision::image_embeds(
            &pixels,
            &clip_vision::VisionConfig::vit_h_14(),
            &adapter.vision,
            s,
        )?;
        let tokens = ip::image_proj(&embeds, self.clip_cfg.hidden, &adapter.weights, s)?;
        // Doubled to match the guidance batch, unconditional row first. The
        // unconditional row gets the *same* tokens: dropping the image there
        // would make guidance push away from it.
        Ok(Some(concat(&[&tokens, &tokens], 0, s)?))
    }

    /// Merge a LoRA into the UNet, in place.
    ///
    /// **Coverage is the thing that matters**, not the arithmetic. The merge is
    /// three lines and hard to get subtly wrong; the name mapping is where an
    /// adapter silently half-applies, and a half-applied adapter still renders
    /// a plausible image. So this errors on any layer that found no home rather
    /// than applying the rest.
    pub fn attach_lora(&mut self, path: &Path, multiplier: f64) -> Result<usize, PipelineError> {
        let raw = load_safetensors(path)?;
        let lora = Lora::from_weights(&raw, &self.stream)?;
        let applied = lora.merge_into(&mut self.unet, multiplier as f32, &self.stream)?;
        if !applied.unmatched.is_empty() {
            return Err(msg(format!(
                "mlx: {} of the LoRA's layers have no weight in this UNet, first `{}`. \
                 A LoRA names the layers it corrects, so entries with nowhere to go mean it \
                 was trained for a different architecture.",
                applied.unmatched.len(),
                applied.unmatched.first().map(String::as_str).unwrap_or("?")
            )));
        }
        Ok(applied.merged)
    }

    /// Attach a ControlNet. Several may be attached, and their corrections sum.
    ///
    /// **Built from the same config as this UNet**, which is checked where the
    /// corrections are added rather than here — a ControlNet for a different
    /// architecture emits a plausible number of plausible tensors and only the
    /// count catches it.
    /// Decode with TAESD instead of the full VAE.
    ///
    /// **Faster and visibly softer**, by design: TAESD is a distilled decoder
    /// with a fraction of the VAE's parameters, so it is for previews and for
    /// sweeping seeds rather than for a final image. It reads the same scaled
    /// latent, so nothing else in the pipeline changes.
    pub fn attach_taesd(&mut self, weights: Weights) {
        self.taesd = Some(weights);
    }

    /// Whether a TAESD decoder is attached.
    pub fn has_taesd(&self) -> bool {
        self.taesd.is_some()
    }

    pub fn attach_controlnet(&mut self, path: &Path, scale: f64) -> Result<(), PipelineError> {
        let mut weights = load_safetensors(path)?;
        normalise_legacy_attention(&mut weights);
        self.controlnets.push(Control { weights, scale });
        Ok(())
    }

    /// How many ControlNets are attached.
    pub fn controlnet_count(&self) -> usize {
        self.controlnets.len()
    }

    /// GLIGEN's grounding tokens for a set of boxes.
    ///
    /// Requires a checkpoint whose UNet carries `fuser` layers; an ordinary SD
    /// 1.5 UNet has nowhere to put them, so this errors rather than dropping
    /// the boxes silently.
    fn grounding(&self, boxes: &[GroundedBox]) -> Result<Option<Array>, PipelineError> {
        if boxes.is_empty() {
            return Ok(None);
        }
        if !gligen::present(
            &self.unet,
            "down_blocks.0.attentions.0.transformer_blocks.0",
        ) {
            return Err(msg(
                "mlx: grounded boxes were supplied but this UNet has no GLIGEN fuser layers".into(),
            ));
        }
        let s = &self.stream;
        let n = boxes.len();
        let mut rows = Vec::with_capacity(n);
        let mut coords = Vec::with_capacity(n * 4);
        for b in boxes {
            // **The phrase's pooled hidden state**, which is the EOS position
            // and not position 0. `clip::pool` takes the *first* highest token
            // id, which matters here because CLIP-L pads with EOS itself — the
            // last one is 60-odd positions past the end of the phrase.
            let ids = self.token_ids(&b.phrase)?;
            let hidden = clip::text_encoder_with(&ids, &self.clip_cfg, &self.text_encoder, s)?;
            rows.push(clip::pool(&hidden, &ids, s)?);
            coords.extend_from_slice(&b.bbox);
        }
        let refs: Vec<&Array> = rows.iter().collect();
        let phrases = concat(&refs, 0, s)?.reshape(&[1, n, self.clip_cfg.hidden], s)?;
        let boxes_arr = Array::from_slice_f32(&coords, &[1, n, 4])?;
        // Every slot is real here, so every mask is 1. The learned nulls exist
        // for callers batching a fixed number of slots.
        let masks = Array::from_slice_f32(&vec![1.0; n], &[1, n])?;

        let objs = gligen::position_net(&boxes_arr, &masks, &phrases, &self.unet, s)?;
        // Doubled for the guidance batch, like every other conditioning here.
        Ok(Some(concat(&[&objs, &objs], 0, s)?))
    }

    /// Encode one prompt to `[1, 77, 768]`.
    ///
    /// An empty prompt is *not* an empty sequence: it is BOS followed by 76
    /// EOS, which is what the tokenizer produces and what the model was trained
    /// against. Feeding a zero tensor instead is a different unconditional.
    fn encode(&self, prompt: &str, skip: usize) -> Result<Array, PipelineError> {
        if self.embeddings.is_empty() {
            let ids = self.token_ids(prompt)?;
            return Ok(clip::text_encoder_skipping(
                &ids,
                &self.clip_cfg,
                skip,
                &self.text_encoder,
                &self.stream,
            )?);
        }
        self.encode_with_embeddings(prompt, skip)
    }

    /// Encode a prompt with textual-inversion embeddings spliced in.
    ///
    /// The trigger is first **expanded** to as many copies of itself as the
    /// embedding has vectors, so the tokeniser reserves that many positions;
    /// then each position's vector is overwritten. Reserving is all the word
    /// is for — its own token embeddings are discarded.
    ///
    /// The positions are found by matching token ids, **not character
    /// offsets**: BPE splits are not positions in the string.
    fn encode_with_embeddings(&self, prompt: &str, skip: usize) -> Result<Array, PipelineError> {
        let s = &self.stream;
        let mut expanded = prompt.to_string();
        for (trigger, vectors) in &self.embeddings {
            if !expanded.contains(trigger.as_str()) {
                continue;
            }
            let n = vectors.shape()[0];
            let repeated = std::iter::repeat_n(trigger.as_str(), n)
                .collect::<Vec<_>>()
                .join(" ");
            expanded = expanded.replace(trigger.as_str(), &repeated);
        }

        let ids = self.token_ids(&expanded)?;
        let mut embeds = clip::embeddings(&ids, &self.text_encoder, s)?;
        let flat: Vec<i32> = ids
            .to_f32(s)?
            .to_vec_f32(s)?
            .iter()
            .map(|&x| x as i32)
            .collect();
        let width = self.clip_cfg.hidden;

        for (trigger, vectors) in &self.embeddings {
            let trigger_ids: Vec<i32> = self
                .tokenizer
                .encode_content(trigger)?
                .iter()
                .map(|&x| x as i32)
                .collect();
            if trigger_ids.is_empty() {
                continue;
            }
            let n = vectors.shape()[0];
            let mut placed = 0usize;
            let mut i = 0usize;
            while i + trigger_ids.len() <= flat.len() && placed < n {
                if flat[i..i + trigger_ids.len()] == trigger_ids[..] {
                    let row = vectors
                        .narrow(0, placed, 1, s)?
                        .reshape(&[1, 1, width], s)?;
                    embeds = splice_row(&embeds, &row, i, s)?;
                    placed += 1;
                    i += trigger_ids.len();
                } else {
                    i += 1;
                }
            }
        }
        Ok(clip::encode_from_embeds_skipping(
            &embeds,
            &self.clip_cfg,
            skip,
            &self.text_encoder,
            s,
        )?)
    }

    /// A prompt's 77 token ids.
    ///
    /// An empty prompt is *not* an empty sequence: it is BOS followed by 76
    /// EOS, which is what the tokenizer produces and what the model was trained
    /// against. A zero tensor instead is a different unconditional.
    fn token_ids(&self, prompt: &str) -> Result<Array, PipelineError> {
        let ids: Vec<i32> = if prompt.is_empty() {
            let mut v = vec![EOS; clip::MAX_POSITION];
            v[0] = BOS;
            v
        } else {
            self.tokenizer
                .encode(prompt)?
                .iter()
                .map(|&x| x as i32)
                .collect()
        };
        if ids.len() != clip::MAX_POSITION {
            return Err(msg(format!(
                "mlx: the tokenizer produced {} ids, CLIP takes {}",
                ids.len(),
                clip::MAX_POSITION
            )));
        }
        Ok(Array::from_slice_i32(&ids, &[1, clip::MAX_POSITION])?)
    }

    /// The guidance batch: unconditional row **first**.
    ///
    /// The order is a contract with [`sample::guidance`], which reads row 0 as
    /// the unconditional. Reversing it runs and drives the image away from the
    /// prompt instead of toward it.
    fn conditioning(&self, cfg: &Txt2ImgConfig) -> Result<Array, PipelineError> {
        let skip = cfg.clip_skip.get();
        let cond = self.encode(&cfg.prompt, skip)?;
        let uncond = self.encode(&cfg.negative_prompt, skip)?;
        Ok(concat(&[&uncond, &cond], 0, &self.stream)?)
    }

    /// The sigma ladder for a run.
    ///
    /// **LCM ignores the scheduler**, and that is not an oversight: its ladder
    /// comes from the distillation's own timestep subset, so respacing it is
    /// not a different schedule but a broken one.
    fn sigmas(&self, cfg: &Txt2ImgConfig) -> Vec<f64> {
        match cfg.sampler {
            SamplerKind::Lcm => sd_sample::lcm_sigmas(
                &self.schedule,
                &sd_sample::lcm_timesteps(
                    self.schedule.alphas_cumprod.len(),
                    sd_sample::ORIGINAL_INFERENCE_STEPS,
                    cfg.steps,
                ),
            ),
            _ => sd_sample::sigmas_for(cfg.scheduler, &self.schedule.sigmas(), cfg.steps),
        }
    }

    fn timestep_for(&self, sigma: f64) -> f32 {
        timestep_for(&self.schedule, sigma)
    }

    fn draw(
        &self,
        rng: &mut SeededRng,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<Array, PipelineError> {
        draw_noise(rng, c, h, w)
    }

    /// The sampling loop, shared by txt2img, img2img and inpaint.
    ///
    /// `sigmas` is a ladder of `n + 1` boundaries; img2img passes a suffix of
    /// one. `rng` is threaded in rather than created here so the caller
    /// controls draw order, which is what makes a seed reproducible.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        mut latent: Array,
        context: &Array,
        sigmas: &[f64],
        cfg_scale: f64,
        sampler: SamplerKind,
        keep: Option<(&Array, &Array)>,
        extras: &Extras<'_>,
        class_embeds: Option<&Array>,
        rng: &mut SeededRng,
        progress: ProgressFn<'_>,
    ) -> Result<Array, PipelineError> {
        // **Held here, not at the public entry points.** Seamless has to reach
        // every convolution, and there are eight ways into this function; a
        // guard placed at each is a guard that one of them will come to miss,
        // and a run that tiles in the sampler but not the decoder has a seam
        // with nothing to point at. `decode` holds the other half.
        let _seamless = sd_models::mlx::SeamlessGuard::new(extras.seamless);
        self.refuse_if_instruct()?;
        let s = &self.stream;
        let [nframes, lh, lw, lc] = latent.shape()[..] else {
            return Err(msg(format!(
                "mlx: a latent should be [n, h, w, c], got {:?}",
                latent.shape()
            )));
        };
        // **One draw per frame.** A `[1, h, w, c]` noise tensor broadcasts
        // against a clip and gives every frame the *same* ancestral noise,
        // which the motion modules then cannot move apart.
        let draw_all = |rng: &mut SeededRng| -> Result<Array, PipelineError> {
            let mut rows = Vec::with_capacity(nframes);
            for _ in 0..nframes {
                rows.push(draw_noise(rng, lc, lh, lw)?);
            }
            let refs: Vec<&Array> = rows.iter().collect();
            Ok(concat(&refs, 0, s)?)
        };
        let mut dpm = sample::DpmSolverPlusPlus2M::new();

        // **The context is repeated per frame, in the batch's own order.**
        // `scale_model_input` doubles a `[f, ...]` latent to `[2f, ...]` with
        // the unconditional clip first, so the context has to be the
        // unconditional row f times then the conditional row f times. Repeating
        // the *pair* f times instead has the right shape and pairs every other
        // frame with the wrong prompt.
        let context = if nframes == 1 {
            context.contiguous(s)?
        } else {
            let mut rows: Vec<Array> = Vec::with_capacity(2 * nframes);
            for half in 0..2 {
                let row = context.narrow(0, half, 1, s)?;
                for _ in 0..nframes {
                    rows.push(row.contiguous(s)?);
                }
            }
            let refs: Vec<&Array> = rows.iter().collect();
            concat(&refs, 0, s)?
        };

        // **Caching is refused with an ancestral sampler**, not ignored. Those
        // draw fresh noise every step, so consecutive predictions never stop
        // moving and there is nothing to reuse — a caller who asked for
        // caching and got none would wonder why, and one who got it anyway
        // would get colour speckle.
        if extras.cache_threshold > 0.0 && sampler.is_ancestral() {
            return Err(msg(format!(
                "mlx: step caching needs a deterministic sampler; {} re-noises every step and \
                 leaves nothing to reuse. Try dpmpp2m, euler, heun or ddim",
                sampler.name()
            )));
        }
        // The reused prediction, the last timestep embedding, and the
        // accumulated *predicted* relative change in the model's output.
        let mut cached: Option<Array> = None;
        let mut previous_temb: Option<Array> = None;
        let mut drift = 0f64;
        let mut evaluated = 0usize;
        let total = sigmas.len().saturating_sub(1);

        for i in 0..total {
            // Checked before the work, so a cancel between steps costs nothing
            // and the error says how far it got.
            if extras.cancel.as_ref().is_some_and(Cancel::is_cancelled) {
                return Err(msg(format!("mlx: cancelled after {i} of {total} steps")));
            }
            let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);

            // **The cache predictor is the timestep embedding**, not the
            // latent. TeaCache's method: measure how far the embedding moved,
            // rescale it through a fitted polynomial into an estimate of how
            // far the *output* would move, and accumulate. `cache_rescale` is
            // scalar and shared with the config crate, so a caller cannot fit
            // a different curve.
            let reuse = if extras.cache_threshold > 0.0 {
                let t = self.timestep_for(sigma);
                let timestep = Array::from_slice_f32(&vec![t; 2 * nframes], &[2 * nframes])?;
                let temb = timestep_embedding(&timestep, 320, &self.unet, s)?;
                let moved = match &previous_temb {
                    Some(prev) => sample::relative_l1(&temb, prev, s)?,
                    None => f64::INFINITY,
                };
                previous_temb = Some(temb);
                if moved.is_finite() {
                    drift += cache_rescale(moved);
                }
                cached.is_some() && drift < extras.cache_threshold
            } else {
                false
            };

            let noise_pred = if reuse {
                cached
                    .as_ref()
                    .expect("reuse implies a cached prediction")
                    .contiguous(s)?
            } else {
                let guided = self.predict(
                    &latent,
                    sigma,
                    &context,
                    cfg_scale,
                    class_embeds,
                    extras,
                    nframes,
                    s,
                )?;
                evaluated += 1;
                drift = 0.0;
                cached = Some(guided.contiguous(s)?);
                guided
            };
            let denoised = sample::denoise_epsilon(&latent, &noise_pred, sigma, s)?;

            latent = match sampler {
                // Ancestral: the deterministic part lands at `sigma_down`,
                // below `sigma_next`, and fresh noise brings it back up.
                // Stepping to `sigma_next` *and* adding noise overshoots.
                SamplerKind::EulerAncestral | SamplerKind::Lcm => {
                    let noise = draw_all(rng)?;
                    sample::euler_ancestral_step(&latent, &denoised, sigma, sigma_next, &noise, s)?
                }
                // Euler and DDIM at eta = 0 are the same update written in two
                // coordinate systems; both names are offered because every
                // paper reports against one of them.
                SamplerKind::Euler | SamplerKind::Ddim => {
                    sample::euler_step(&latent, &denoised, sigma, sigma_next, s)?
                }
                SamplerKind::DpmPlusPlus2M => dpm.step(&latent, &denoised, sigma, sigma_next, s)?,
                // **Two evaluations.** An Euler step, then the model again at
                // where it landed, then the average of the two derivatives.
                // At the last step `sigma_next` is 0 and there is nothing to
                // evaluate, so it degenerates to Euler rather than dividing by
                // zero — the weights say so, and the branch avoids the wasted
                // forward.
                SamplerKind::Heun => {
                    if sigma_next <= 0.0 {
                        sample::euler_step(&latent, &denoised, sigma, sigma_next, s)?
                    } else {
                        let euler = sample::euler_step(&latent, &denoised, sigma, sigma_next, s)?;
                        let pred_next = self.predict(
                            &euler,
                            sigma_next,
                            &context,
                            cfg_scale,
                            class_embeds,
                            extras,
                            nframes,
                            s,
                        )?;
                        evaluated += 1;
                        let denoised_next =
                            sample::denoise_epsilon(&euler, &pred_next, sigma_next, s)?;
                        sample::heun_step(&latent, &denoised, &denoised_next, sigma, sigma_next, s)?
                    }
                }
                // **Two evaluations**, and the midpoint is geometric in sigma
                // — DPM-Solver integrates in `lambda = -log(sigma)`, so an
                // arithmetic midpoint silently drops the method to first order.
                SamplerKind::DpmPlusPlus2SAncestral => {
                    let (sigma_up, sigma_down) = steps::ancestral_split(sigma, sigma_next, 1.0);
                    let stepped = if sigma_down <= 0.0 {
                        denoised.contiguous(s)?
                    } else {
                        let (sigma_mid, a, b) = steps::dpmpp_2s_midpoint(sigma, sigma_down);
                        let mid = sample::blend(&latent, &denoised, a, b, s)?;
                        let pred_mid = self.predict(
                            &mid,
                            sigma_mid,
                            &context,
                            cfg_scale,
                            class_embeds,
                            extras,
                            nframes,
                            s,
                        )?;
                        evaluated += 1;
                        let denoised_mid = sample::denoise_epsilon(&mid, &pred_mid, sigma_mid, s)?;
                        let (c, d) = steps::dpmpp_2s_step(sigma, sigma_down);
                        sample::blend(&latent, &denoised_mid, c, d, s)?
                    };
                    let noise = draw_all(rng)?;
                    sample::add_noise(&stepped, &noise, sigma_up, s)?
                }
            };

            // Inpainting: restore outside the mask at every step, so the model
            // sees the true surroundings and what it paints joins up with them.
            if let Some((mask, init)) = keep {
                let noise = draw_all(rng)?;
                latent = sample::restore_outside_mask(&latent, init, mask, &noise, sigma_next, s)?;
            }

            progress(Progress {
                step: i + 1,
                total,
                sigma,
                denoised: &denoised,
                evaluated,
            });
        }
        Ok(latent)
    }

    /// One model evaluation: latent and sigma in, a guided noise prediction out.
    ///
    /// Extracted from the sampling loop because **the second-order samplers
    /// call it twice per step.** Heun evaluates again at the point its Euler
    /// half landed on; DPM++ 2S evaluates at a midpoint. Inlining this in the
    /// loop, as it used to be, made those samplers unimplementable without
    /// duplicating the adapter plumbing — which is exactly where a ControlNet
    /// or an IP-Adapter would come to be applied on one evaluation and not the
    /// other.
    ///
    /// Returns the **guided epsilon**, not `denoised`, because that is what the
    /// step cache stores: caching after the epsilon-to-x0 conversion would tie
    /// a cached value to the sigma it was computed at.
    #[allow(clippy::too_many_arguments)]
    fn predict(
        &self,
        latent: &Array,
        sigma: f64,
        context: &Array,
        cfg_scale: f64,
        class_embeds: Option<&Array>,
        extras: &Extras<'_>,
        nframes: usize,
        s: &Stream,
    ) -> Result<Array, PipelineError> {
        let latent_in = sample::scale_model_input(latent, sigma, s)?;
        let t = self.timestep_for(sigma);
        // One entry per row of the doubled batch, not one per guidance half.
        let timestep = Array::from_slice_f32(&vec![t; 2 * nframes], &[2 * nframes])?;

        // Several ControlNets sum. Running them here rather than once outside
        // the loop is not an optimisation missed: each is conditioned on the
        // current latent and timestep, so its corrections differ every step —
        // and differ between the two halves of a second-order step too.
        let control = self.control_for(&latent_in, &timestep, context, extras.hint)?;
        // The adapter walks its layers in visit order, so its counter has to
        // start from zero on every evaluation rather than continue across a
        // run. `unet_forward_adapters` rewinds it.
        let ip = extras.ip_tokens.as_ref().map(|tokens| {
            ip::IpAdapter::new(
                &self.ip.as_ref().expect("tokens imply an adapter").weights,
                tokens.contiguous(s).expect("tokens"),
                self.ip.as_ref().expect("adapter").scale,
            )
        });
        let m = self.motion.as_ref().map(|(w, frames)| Motion {
            weights: w,
            // The guidance batch doubles the rows, so the UNet sees `2 * f` and
            // each half is one clip. Passing `f` here is what makes the
            // regrouping split them correctly.
            frames: *frames,
        });
        let ad = Adapters {
            control: control.as_ref(),
            ip: ip.as_ref(),
            objs: extras.objs,
            motion: m.as_ref(),
        };
        let out = unet_forward_adapters(
            &latent_in,
            &timestep,
            context,
            None,
            class_embeds,
            &ad,
            &self.cfg,
            &self.unet,
            s,
        )?;
        let guided = sample::guidance(&out, cfg_scale, s)?;
        // Regions blend *before* the step, not after: compositing two finished
        // images produces visible joins because neither half ever saw the other.
        self.blend_regions(
            &guided,
            &latent_in,
            &timestep,
            cfg_scale,
            class_embeds,
            &ad,
            extras,
            s,
        )
    }

    /// Blend each region's own noise prediction into the base one.
    ///
    /// **Before the step, not after.** Generating separately and compositing
    /// produces visible joins because neither half ever saw the other; blending
    /// the predictions means every region is denoised in the context of its
    /// neighbours.
    ///
    /// The mask is reduced to the latent grid by **mean**, not max — a region
    /// boundary should fade over a latent cell rather than claim it outright,
    /// which is the opposite of what an inpainting mask wants.
    #[allow(clippy::too_many_arguments)]
    fn blend_regions(
        &self,
        base: &Array,
        latent_in: &Array,
        timestep: &Array,
        cfg_scale: f64,
        class_embeds: Option<&Array>,
        ad: &Adapters<'_>,
        extras: &Extras<'_>,
        s: &Stream,
    ) -> Result<Array, PipelineError> {
        if extras.regions.is_empty() {
            return Ok(base.contiguous(s)?);
        }
        let mut out = base.contiguous(s)?;
        for region in extras.regions {
            let ctx = self.conditioning(&Txt2ImgConfig {
                prompt: region.prompt.clone(),
                ..Default::default()
            })?;
            let pred = unet_forward_adapters(
                latent_in,
                timestep,
                &ctx,
                None,
                class_embeds,
                ad,
                &self.cfg,
                &self.unet,
                s,
            )?;
            let guided = sample::guidance(&pred, cfg_scale, s)?;
            let mask = mean_pool_to_latent(&region.mask, s)?;
            let keep = Array::scalar_f32(1.0)?.sub(&mask, s)?;
            out = out.mul(&keep, s)?.add(&guided.mul(&mask, s)?, s)?;
        }
        Ok(out)
    }

    /// Every attached ControlNet's corrections, summed.
    ///
    /// `None` when nothing is attached, so the ordinary path allocates
    /// nothing. A hint is required once one is: a ControlNet with no map to
    /// read would emit corrections from a blank image, which steers the run
    /// toward an empty picture rather than doing nothing.
    fn control_for(
        &self,
        latent_in: &Array,
        timestep: &Array,
        context: &Array,
        hint: Option<&Array>,
    ) -> Result<Option<controlnet::Control>, PipelineError> {
        if self.controlnets.is_empty() {
            return Ok(None);
        }
        let s = &self.stream;
        let Some(hint) = hint else {
            return Err(msg(
                "mlx: a ControlNet is attached but no control map was supplied".into(),
            ));
        };
        // The hint is doubled to match the guidance batch, exactly as the
        // latent is — the ControlNet sees both rows.
        let hint = concat(&[hint, hint], 0, s)?;

        let mut total: Option<controlnet::Control> = None;
        for net in &self.controlnets {
            let c = controlnet::forward(
                latent_in,
                timestep,
                context,
                &hint,
                net.scale,
                &self.cfg,
                &net.weights,
                s,
            )?;
            total = Some(match total {
                None => c,
                Some(acc) => {
                    if acc.down.len() != c.down.len() {
                        return Err(msg(format!(
                            "mlx: two ControlNets emitted {} and {} corrections",
                            acc.down.len(),
                            c.down.len()
                        )));
                    }
                    let down = acc
                        .down
                        .iter()
                        .zip(&c.down)
                        .map(|(a, b)| a.add(b, s))
                        .collect::<Result<Vec<_>, _>>()?;
                    controlnet::Control {
                        down,
                        mid: acc.mid.add(&c.mid, s)?,
                    }
                }
            });
        }
        Ok(total)
    }

    /// A latent to `[h, w, 3]` bytes.
    /// A latent to RGB bytes, `frames * h * w * 3` of them.
    ///
    /// **One frame at a time.** A clip is a batch and the VAE has no
    /// cross-frame interaction, so decoding `n` together simply multiplies the
    /// largest single allocation by `n` — which is how a three-frame 512
    /// decode reaches 6.8 GiB on the candle side. Looping gives identical
    /// output at one frame's peak.
    fn decode(
        &self,
        latent: &Array,
        seamless: bool,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        // A parameter rather than a field, so that adding a decode path is a
        // compile error until it says whether it tiles. The decoder is
        // convolutional too: a latent that wraps, decoded with zero padding,
        // has a seam again.
        let _seamless = sd_models::mlx::SeamlessGuard::new(seamless);
        let s = &self.stream;
        let n = latent.shape()[0];
        let mut out: Vec<u8> = Vec::new();
        let (mut width, mut height) = (0usize, 0usize);

        for i in 0..n {
            let frame = latent.narrow(0, i, 1, s)?.contiguous(s)?;
            let image = match &self.taesd {
                // **TAESD takes the sampler's latent directly**, where the VAE
                // wants it unscaled first: `AutoencoderTiny` carries a scaling
                // factor of 1, so it was distilled on the 0.18215-scaled space
                // the sampler already works in. Unscaling here would hand it a
                // latent 5.5x too small, which decodes to a flat grey field.
                //
                // Both emit the signed range — measured, min -1.557 max 1.399
                // on the golden latent — so the byte conversion below is
                // shared rather than branched.
                Some(w) => taesd::decode(&frame, w, s)?,
                None => {
                    let unscaled = self.vae_cfg.unscale(&frame, s)?;
                    vae::decode_with(&unscaled, &self.vae_cfg, &self.vae, s)?
                }
            };
            let [_, h, w, _] = image.shape()[..] else {
                return Err(msg(format!(
                    "mlx: the decoder returned {:?}",
                    image.shape()
                )));
            };
            (width, height) = (w, h);
            // The VAE emits roughly [-1, 1]; the caller wants bytes.
            out.extend(
                image
                    .to_vec_f32(s)?
                    .iter()
                    .map(|&v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8),
            );
        }
        Ok((width, height, out))
    }

    /// Prompt to pixels. Returns `(width, height, RGB bytes)`.
    ///
    /// With a motion adapter attached the bytes are **`frames()` images back to
    /// back**, each `width * height * 3`.
    pub fn txt2img(&self, cfg: &Txt2ImgConfig) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        self.txt2img_controlled(cfg, None)
    }

    /// [`Self::txt2img`] with a control map for the attached ControlNets.
    ///
    /// `hint` is `[1, h, w, 3]` in `[-1, 1]` at the run's own resolution — a
    /// Canny edge map, a depth map, whatever the ControlNet was trained on.
    pub fn txt2img_controlled(
        &self,
        cfg: &Txt2ImgConfig,
        hint: Option<&Array>,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        self.generate(cfg, hint, None, &[])
    }

    /// Everything at once: a control map, an IP-Adapter reference image, and
    /// GLIGEN boxes.
    ///
    /// `reference` is `[1, 224, 224, 3]` in **`[0, 1]`** — CLIP's range, not
    /// the `[-1, 1]` a VAE uses.
    pub fn generate(
        &self,
        cfg: &Txt2ImgConfig,
        hint: Option<&Array>,
        reference: Option<&Array>,
        boxes: &[GroundedBox],
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        let mut rng = SeededRng::new(cfg.seed);
        self.generate_inner(
            cfg,
            hint,
            reference,
            boxes,
            None,
            &mut rng,
            None,
            &mut |_| {},
        )
    }

    /// [`Self::generate`] with unCLIP's image conditioning, and a caller-owned
    /// RNG so the draw order stays under one seed.
    fn generate_with_class(
        &self,
        cfg: &Txt2ImgConfig,
        class_embeds: Option<&Array>,
        rng: &mut SeededRng,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        self.generate_inner(cfg, None, None, &[], class_embeds, rng, None, &mut |_| {})
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_inner(
        &self,
        cfg: &Txt2ImgConfig,
        hint: Option<&Array>,
        reference: Option<&Array>,
        boxes: &[GroundedBox],
        class_embeds: Option<&Array>,
        rng: &mut SeededRng,
        extras_in: Option<&Extras<'_>>,
        progress: ProgressFn<'_>,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        if cfg.width % 8 != 0 || cfg.height % 8 != 0 {
            return Err(msg(format!(
                "mlx: {}x{} does not divide into 8-pixel latent cells",
                cfg.width, cfg.height
            )));
        }
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);
        let context = self.conditioning(cfg)?;
        let sigmas = self.sigmas(cfg);

        let ip_tokens = match reference {
            Some(image) => self.ip_tokens(image)?,
            None => None,
        };
        let objs = self.grounding(boxes)?;
        let extras = Extras {
            hint,
            ip_tokens,
            objs: objs.as_ref(),
            regions: extras_in.map_or(&[][..], |e| e.regions),
            cache_threshold: extras_in.map_or(0.0, |e| e.cache_threshold),
            cancel: extras_in.and_then(|e| e.cancel.clone()),
            seamless: cfg.seamless,
        };

        // **One draw per frame, in order.** A clip is a batch, so the latent is
        // `[f, h, w, 4]`; drawing one and repeating it would give f identical
        // frames that the motion modules then fail to move apart.
        let frames = self.frames();
        let mut rows = Vec::with_capacity(frames);
        for _ in 0..frames {
            rows.push(self.draw(rng, 4, lh, lw)?);
        }
        let refs: Vec<&Array> = rows.iter().collect();
        let latent = concat(&refs, 0, &self.stream)?
            .mul(&Array::scalar_f32(sigmas[0] as f32)?, &self.stream)?;

        let latent = self.denoise(
            latent,
            &context,
            &sigmas,
            cfg.cfg_scale,
            cfg.sampler,
            None,
            &extras,
            class_embeds,
            rng,
            progress,
        )?;
        self.decode(&latent, cfg.seamless)
    }

    /// [`Self::txt2img`] with everything optional supplied by name.
    ///
    /// This used to take eight positional parameters, two of them adjacent
    /// `Option<&Array>` that the compiler cannot tell apart — so passing a
    /// control map where a reference image belonged compiled and produced a
    /// wrong picture. Build a [`Request`] instead:
    ///
    /// ```no_run
    /// # use stable_diffusion_rs::mlx::{MlxPipeline, Request};
    /// # use stable_diffusion_rs::config::Txt2ImgConfig;
    /// # fn f(pipe: &MlxPipeline, cfg: &Txt2ImgConfig) -> Result<(), Box<dyn std::error::Error>> {
    /// let (w, h, rgb) = pipe.run(Request::new(cfg).cache_threshold(0.2))?;
    /// # Ok(()) }
    /// ```
    pub fn run(&self, request: Request<'_>) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        self.run_with(request, &mut |_| {})
    }

    /// [`Self::run`], reporting progress after each step.
    pub fn run_with(
        &self,
        request: Request<'_>,
        progress: ProgressFn<'_>,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        let Request {
            cfg,
            hint,
            reference,
            boxes,
            regions,
            cache_threshold,
            cancel,
        } = request;
        let extras = Extras {
            regions,
            cache_threshold,
            cancel,
            seamless: cfg.seamless,
            ..Default::default()
        };
        let mut rng = SeededRng::new(cfg.seed);
        self.generate_inner(
            cfg,
            hint,
            reference,
            boxes,
            None,
            &mut rng,
            Some(&extras),
            progress,
        )
    }

    /// Every image a [`Txt2ImgConfig::batch_count`] asks for, from one load.
    ///
    /// **Seeds run `seed`, `seed + 1`, ...**, so image 3 of a batch is the same
    /// picture as a single run at `seed + 2`. A batch that shared one seed
    /// would produce one image N times; a batch seeded from a clock would not
    /// be reproducible at all.
    pub fn run_batch(
        &self,
        request: Request<'_>,
        progress: ProgressFn<'_>,
    ) -> Result<Vec<(usize, usize, Vec<u8>)>, PipelineError> {
        let n = request.cfg.batch_count.max(1);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let cfg = Txt2ImgConfig {
                seed: request.cfg.seed_for(i),
                ..request.cfg.clone()
            };
            let one = Request {
                cfg: &cfg,
                hint: request.hint,
                reference: request.reference,
                boxes: request.boxes,
                regions: request.regions,
                cache_threshold: request.cache_threshold,
                cancel: request.cancel.clone(),
            };
            out.push(self.run_with(one, progress)?);
        }
        Ok(out)
    }

    /// Two passes: compose at `cfg`'s size, enlarge the latent, refine at the
    /// larger one.
    ///
    /// **This fixes a real failure, not a cosmetic one.** SD 1.5 asked to
    /// compose at 1024 directly produces duplicated subjects — three knights
    /// where one was asked for — because it was trained at 512 and the extra
    /// canvas reads as more room for content. Composing small and refining
    /// large gives one, sharp, and is *faster*, because the first pass runs at
    /// the smaller size.
    ///
    /// The second pass draws from `seed + 1`, so the two passes do not draw the
    /// same noise for differently-sized latents.
    pub fn txt2img_hires(
        &self,
        cfg: &Txt2ImgConfig,
        width: usize,
        height: usize,
        strength: Strength,
        progress: ProgressFn<'_>,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        if width % 8 != 0 || height % 8 != 0 {
            return Err(msg(format!(
                "mlx: {width}x{height} does not divide into 8-pixel latent cells"
            )));
        }
        if width < cfg.width || height < cfg.height {
            return Err(msg(format!(
                "mlx: the second pass is {width}x{height}, smaller than the first at {}x{} — \
                 hires enlarges",
                cfg.width, cfg.height
            )));
        }
        let s = &self.stream;

        // Pass one, at the size the model composes well at.
        let mut rng = SeededRng::new(cfg.seed);
        let first = self.latent_for(cfg, &mut rng, progress)?;

        // **Nearest, by default, introduces no colours that were not already
        // there.** Every interpolating mode invents intermediate values, which
        // is right for photographic work and destructive for a fixed palette.
        let (lh, lw) = (height / 8, width / 8);
        let enlarged = nearest_upsample(&first, lh, lw, s)?;

        // Pass two: noise the enlarged latent to where `strength` starts and
        // run the tail of the schedule.
        let second = Txt2ImgConfig {
            width,
            height,
            // A different seed, so the two passes do not draw the same noise
            // for differently-sized latents.
            seed: cfg.seed.wrapping_add(1),
            ..cfg.clone()
        };
        let sigmas = self.sigmas(&second);
        let start = strength.start_index(second.steps);
        if start >= second.steps {
            return self.decode(&enlarged, second.seamless);
        }
        let mut rng = SeededRng::new(second.seed);
        let noise = draw_noise(&mut rng, 4, lh, lw)?;
        let latent = sample::noise_to_sigma(&enlarged, &noise, sigmas[start], s)?;
        let context = self.conditioning(&second)?;
        let latent = self.denoise(
            latent,
            &context,
            &sigmas[start..],
            second.cfg_scale,
            second.sampler,
            None,
            &Extras::default(),
            None,
            &mut rng,
            progress,
        )?;
        self.decode(&latent, second.seamless)
    }

    /// One txt2img pass, stopping at the latent.
    fn latent_for(
        &self,
        cfg: &Txt2ImgConfig,
        rng: &mut SeededRng,
        progress: ProgressFn<'_>,
    ) -> Result<Array, PipelineError> {
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);
        let context = self.conditioning(cfg)?;
        let sigmas = self.sigmas(cfg);
        let latent =
            draw_noise(rng, 4, lh, lw)?.mul(&Array::scalar_f32(sigmas[0] as f32)?, &self.stream)?;
        self.denoise(
            latent,
            &context,
            &sigmas,
            cfg.cfg_scale,
            cfg.sampler,
            None,
            &Extras::default(),
            None,
            rng,
            progress,
        )
    }

    /// An image and a prompt to pixels.
    ///
    /// `image` is `[1, h, w, 3]` in `[-1, 1]`, already at `cfg.width` x
    /// `cfg.height`. `strength` selects where in the ladder the run begins: at
    /// strength `s` with `n` steps it starts at `n - round(n*s)`.
    ///
    /// **Does not yet carry a control map, a reference image or boxes.** With
    /// a ControlNet attached this errors rather than running unsteered — which
    /// is the right failure, but it does mean img2img and ControlNet cannot be
    /// combined here yet. `txt2img_controlled` and `generate` are the
    /// conditioned entry points.
    pub fn img2img(
        &self,
        cfg: &Txt2ImgConfig,
        image: &Array,
        strength: Strength,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        self.run_masked(cfg, image, strength, None)
    }

    /// img2img bounded by a mask. `mask_px` is `[1, h, w, 1]`, 1 where the
    /// model may write.
    ///
    /// Same conditioning limits as [`Self::img2img`].
    pub fn inpaint(
        &self,
        cfg: &Txt2ImgConfig,
        image: &Array,
        strength: Strength,
        mask_px: &Array,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        self.run_masked(cfg, image, strength, Some(mask_px))
    }

    fn run_masked(
        &self,
        cfg: &Txt2ImgConfig,
        image: &Array,
        strength: Strength,
        mask_px: Option<&Array>,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        let s = &self.stream;
        // The distribution's mean, scaled — the sampler supplies all the
        // randomness, so drawing here too would add variance the seed does not
        // control.
        let init = vae::encode_scaled(image, &self.vae_cfg, &self.vae, s)?;
        let [_, lh, lw, _] = init.shape()[..] else {
            return Err(msg(format!("mlx: the encoder returned {:?}", init.shape())));
        };

        let sigmas = self.sigmas(cfg);
        let start = strength.start_index(cfg.steps);
        // Strength 0 means "return the input", and there is nothing to run.
        if start >= cfg.steps {
            return self.decode(&init, cfg.seamless);
        }

        let mut rng = SeededRng::new(cfg.seed);
        let noise = self.draw(&mut rng, 4, lh, lw)?;
        let latent = sample::noise_to_sigma(&init, &noise, sigmas[start], s)?;

        let mask = mask_px.map(|m| sample::latent_mask(m, s)).transpose()?;
        let context = self.conditioning(cfg)?;
        let latent = self.denoise(
            latent,
            &context,
            &sigmas[start..],
            cfg.cfg_scale,
            cfg.sampler,
            mask.as_ref().map(|m| (m, &init)),
            &Extras::default(),
            None,
            &mut rng,
            &mut |_| {},
        )?;
        self.decode(&latent, cfg.seamless)
    }

    /// How many channels this UNet's `conv_in` takes.
    ///
    /// 4 for an ordinary latent model, **8 for InstructPix2Pix** — four for
    /// the latent and four for the source image.
    pub fn unet_in_channels(&self) -> usize {
        self.unet
            .get("conv_in.weight")
            .and_then(|w| w.shape().get(1).copied())
            .unwrap_or(0)
    }

    /// Refuse a plain generation on an InstructPix2Pix checkpoint.
    ///
    /// Without this the run fails inside the first convolution with
    /// `Expect the input channels ... to match but got (2,32,32,4) and
    /// (320,3,3,8)`, which is true and tells the caller nothing about what to
    /// do. The checkpoint is not broken; it is being asked the wrong question.
    fn refuse_if_instruct(&self) -> Result<(), PipelineError> {
        if self.unet_in_channels() == 8 {
            return Err(msg(
                "mlx: this is an InstructPix2Pix checkpoint — its UNet takes eight input \
                 channels, four of them the source image. Use `instruct` (`sdrs instruct`), \
                 which supplies them"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Edit an image by instruction: "change the sky to sunset".
    ///
    /// **Not img2img.** img2img partially re-noises a picture and denoises it
    /// against a prompt describing the *result*; this conditions on the source
    /// image directly and takes a prompt describing the *change*. The
    /// checkpoint is different too — InstructPix2Pix's UNet takes **eight**
    /// input channels, four for the latent and four for the source.
    ///
    /// Three things here fail quietly if got wrong, and all three cost a
    /// plausible image rather than an error:
    ///
    /// - **The guidance batch has three rows, not two**: instruction, image,
    ///   and a true unconditional that sees neither. Two scales combine them.
    /// - **The image latent is not scaled by `0.18215`.** Every other latent
    ///   in this crate is; InstructPix2Pix was trained on the raw encoder
    ///   output, and scaling multiplies the conditioning by 5.5, which returns
    ///   a clean image that ignores the source.
    /// - **The source joins on the channel axis**, not the batch axis, which
    ///   is what the extra four input channels are for.
    ///
    /// `image` is `[1, h, w, 3]` in `[-1, 1]` — the VAE's range.
    pub fn instruct(
        &self,
        cfg: &Txt2ImgConfig,
        image: &Array,
        image_guidance: f64,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        self.instruct_with(cfg, image, image_guidance, &mut |_| {})
    }

    /// [`Self::instruct`], reporting progress after each step.
    pub fn instruct_with(
        &self,
        cfg: &Txt2ImgConfig,
        image: &Array,
        image_guidance: f64,
        progress: ProgressFn<'_>,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        let s = &self.stream;
        if cfg.width % 8 != 0 || cfg.height % 8 != 0 {
            return Err(msg(format!(
                "mlx: {}x{} does not divide into 8-pixel latent cells",
                cfg.width, cfg.height
            )));
        }
        if cfg.steps == 0 {
            return Err(PipelineError::NoSteps);
        }
        // A plain SD 1.5 UNet has a four-channel `conv_in` and would fail on
        // the eight-channel input with a shape error deep in the first block.
        // Checked here so the message names the actual problem.
        let conv_in = self
            .unet
            .get("conv_in.weight")
            .ok_or_else(|| msg("mlx: the UNet has no conv_in.weight".into()))?;
        let in_channels = conv_in.shape().get(1).copied().unwrap_or(0);
        if in_channels != 8 {
            return Err(msg(format!(
                "mlx: this UNet takes {in_channels} input channels; instruction editing needs                  an InstructPix2Pix checkpoint, whose conv_in takes 8 — four for the latent                  and four for the source image"
            )));
        }

        let _seamless = sd_models::mlx::SeamlessGuard::new(cfg.seamless);
        let skip = cfg.clip_skip.get();
        let cond = self.encode(&cfg.prompt, skip)?;
        let uncond = self.encode(&cfg.negative_prompt, skip)?;
        // Instruction, then the unconditional twice — matching the three rows
        // the guidance below reads.
        let context = concat(&[&cond, &uncond, &uncond], 0, s)?;

        // **Unscaled**, unlike every other latent here. See the doc comment.
        // `.0` is the distribution's mean, **unscaled** — `encode_scaled`,
        // which every other path here calls, would multiply it by 0.18215.
        let (image_latents, _) = vae::encode_dist_with(image, &self.vae_cfg, &self.vae, s)?;
        let zeros = image_latents.mul(&Array::scalar_f32(0.0)?, s)?;
        // The third row sees no image at all, which is what makes it the true
        // unconditional rather than a second image row.
        let image_rows = concat(&[&image_latents, &image_latents, &zeros], 0, s)?;

        let sigmas = self.sigmas(cfg);
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);
        let mut rng = SeededRng::new(cfg.seed);
        let mut latent = self
            .draw(&mut rng, 4, lh, lw)?
            .mul(&Array::scalar_f32(sigmas[0] as f32)?, s)?;

        let total = sigmas.len().saturating_sub(1);
        let mut dpm = sample::DpmSolverPlusPlus2M::new();
        for i in 0..total {
            let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
            let tripled = concat(&[&latent, &latent, &latent], 0, s)?;
            let latent_in =
                tripled.div(&Array::scalar_f32((sigma * sigma + 1.0).sqrt() as f32)?, s)?;
            // NHWC, so the channel axis is 3.
            let latent_in = concat(&[&latent_in, &image_rows], 3, s)?;

            let t = self.timestep_for(sigma);
            let timestep = Array::from_slice_f32(&[t, t, t], &[3])?;
            let out = unet_forward_adapters(
                &latent_in,
                &timestep,
                &context,
                None,
                None,
                &Adapters::default(),
                &self.cfg,
                &self.unet,
                s,
            )?;

            let (text, img, none) = (
                out.narrow(0, 0, 1, s)?,
                out.narrow(0, 1, 1, s)?,
                out.narrow(0, 2, 1, s)?,
            );
            // uncond + (text - img) * cfg_scale + (img - uncond) * image_guidance
            let noise_pred = none
                .add(
                    &text
                        .sub(&img, s)?
                        .mul(&Array::scalar_f32(cfg.cfg_scale as f32)?, s)?,
                    s,
                )?
                .add(
                    &img.sub(&none, s)?
                        .mul(&Array::scalar_f32(image_guidance as f32)?, s)?,
                    s,
                )?;

            let denoised = sample::denoise_epsilon(&latent, &noise_pred, sigma, s)?;
            latent = match cfg.sampler {
                SamplerKind::EulerAncestral | SamplerKind::Lcm => {
                    let noise = self.draw(&mut rng, 4, lh, lw)?;
                    sample::euler_ancestral_step(&latent, &denoised, sigma, sigma_next, &noise, s)?
                }
                SamplerKind::Euler | SamplerKind::Ddim | SamplerKind::Heun => {
                    // Heun's second evaluation would need the three-row batch
                    // rebuilt at the midpoint; first-order is used rather than
                    // silently running a half-Heun.
                    sample::euler_step(&latent, &denoised, sigma, sigma_next, s)?
                }
                SamplerKind::DpmPlusPlus2M | SamplerKind::DpmPlusPlus2SAncestral => {
                    dpm.step(&latent, &denoised, sigma, sigma_next, s)?
                }
            };
            progress(Progress {
                step: i + 1,
                total,
                sigma,
                denoised: &denoised,
                evaluated: i + 1,
            });
        }
        self.decode(&latent, cfg.seamless)
    }

    /// The stream this pipeline runs on, for callers that build their own
    /// tensors to hand in.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Which device this pipeline loaded onto.
    pub fn device(&self) -> Device {
        self.device
    }
}
