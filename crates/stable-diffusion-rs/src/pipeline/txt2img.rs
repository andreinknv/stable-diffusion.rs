//! Text-to-image: tokenizer, text encoder, UNet, sampler, VAE.

use std::path::{Path, PathBuf};

use sd_models::clip::{ClipTextConfig, ClipTextEncoder, ClipTokenizer};
use sd_models::controlnet::ControlNet;
use sd_models::unet::{UNet2DConditionModel, UNetConfig};
use sd_models::vae::{AutoencoderKlDecoder, AutoencoderKlEncoder, TinyDecoder, VaeConfig};
use sd_sample::{
    euler_ancestral_step, lcm_sigmas, lcm_step, lcm_timesteps, sigmas_for_steps,
    DpmSolverPlusPlus2M, Schedule,
};
use sd_tensor::rng::SeededRng;
use sd_tensor::{DType, Device, Tensor};

/// Which sampler to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SamplerKind {
    #[default]
    EulerAncestral,
    DpmPlusPlus2M,
    /// Latent consistency sampling. **Only meaningful with an LCM-distilled
    /// model or adapter**, and wants 4-8 steps at `cfg_scale` near 1 — the
    /// guidance is distilled in, so applying more on top double-counts it and
    /// blows the image out.
    Lcm,
}

/// Everything a single generation needs.
#[derive(Debug, Clone)]
pub struct Txt2ImgConfig {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub cfg_scale: f64,
    pub seed: u64,
    pub sampler: SamplerKind,
}

impl Default for Txt2ImgConfig {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 512,
            height: 512,
            steps: 20,
            cfg_scale: 7.5,
            seed: 0,
            sampler: SamplerKind::default(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("missing model file: {0}\nExpected the standard diffusers layout under the model directory.")]
    MissingFile(PathBuf),
    #[error(
        "missing {0}\n\n\
         A stock SD 1.5 download does not contain this file: the repository ships the slow \
         tokenizer (vocab.json + merges.txt) instead. Copy `tokenizer.json` from \
         `openai/clip-vit-large-patch14`, which is the tokenizer SD 1.5 uses."
    )]
    MissingTokenizerJson(PathBuf),
    #[error(
        "loading the VAE from {path}: {source}\n\n\
         sd-loader adapts the legacy diffusers attention names \
         (`query`/`key`/`value`/`proj_attn`) that stock SD 1.5 weights use, so a missing \
         `to_q`-style tensor here means the checkpoint is neither layout — check it is an \
         SD 1.5 VAE and not, say, an SDXL one."
    )]
    VaeWeights {
        path: PathBuf,
        source: sd_tensor::Error,
    },
    #[error("{0} must be a multiple of 8 (latents are 1/8 scale), got {1}")]
    NotMultipleOfEight(&'static str, usize),
    #[error("steps must be at least 1")]
    NoSteps,
    #[error("tokenizer: {0}")]
    Tokenize(#[from] sd_models::clip::TokenizeError),
    #[error("loading weights: {0}")]
    Load(#[from] sd_loader::LoadError),
    #[error(
        "the LoRA does not match this model: {unmatched} of its layers have no weight here, \
         first `{first}`.\n\n\
         A LoRA names the layers it corrects, so entries with nowhere to go mean it was \
         trained for a different architecture — an SDXL adapter on SD 1.5, say. Applying \
         the rest would render a plausible image that is not the one the adapter describes, \
         which nothing downstream can detect, so it is refused here instead."
    )]
    LoraMismatch { unmatched: usize, first: String },
    #[error(
        "this pipeline has no ControlNet.\n\n\
         Attach one with `Txt2ImgPipeline::with_controlnet(path)` before calling \
         `run_control`. Running the control config without one would silently ignore \
         the control image and return an ordinary generation."
    )]
    NoControlNet,
    #[error("tensor: {0}")]
    Tensor(#[from] sd_tensor::Error),
}

/// What a progress callback is told after each denoising step.
///
/// A struct rather than positional arguments because the interesting field is
/// `latent`, and a fourth positional `&Tensor` would be easy to ignore — which
/// is the opposite of what it is for.
pub struct Progress<'a> {
    /// 1-based; equal to `total` on the last step.
    pub step: usize,
    pub total: usize,
    pub sigma: f64,
    /// The model's current estimate of the finished image, as a latent.
    ///
    /// **Not the sampler's latent**, and the difference is the whole value of
    /// this field. The latent at step 5 of 20 is `x0 + sigma*noise` with sigma
    /// still around 4, so decoding it shows noise; this is the `x0` the model
    /// predicts, which decodes to a blurry version of the final image and
    /// sharpens as the run proceeds. That is what a preview is for, and it is
    /// what every diffusion UI shows.
    ///
    /// Borrowed, and decoding it is the caller's choice: a full VAE decode per
    /// step costs more than the denoising does, which is exactly why
    /// [`Txt2ImgPipeline::with_taesd`] exists. `Txt2ImgPipeline::preview`
    /// decodes it with whichever decoder is attached.
    pub denoised: &'a Tensor,
}

/// Called after each denoising step.
///
/// A callback rather than a log line because this crate has no logging
/// dependency and adding one is out of scope. The CLI owns the reporting, and
/// a library caller can render progress however it likes.
pub type ProgressFn<'a> = &'a mut dyn FnMut(Progress<'_>);

/// How much of the schedule an img2img run replaces.
///
/// `1.0` ignores the input entirely; `0.0` returns it unchanged. The value
/// selects where in the sigma ladder to start: at strength `s` with `n` steps,
/// the run begins at index `n - round(n*s)` and executes the remaining steps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Strength(f64);

impl Strength {
    /// Clamped to `[0, 1]`; anything outside is a caller error, not a mode.
    pub fn new(v: f64) -> Self {
        Self(v.clamp(0.0, 1.0))
    }

    pub fn get(self) -> f64 {
        self.0
    }

    /// Index into a `steps + 1` sigma ladder at which to begin.
    ///
    /// Public because it is the whole meaning of the parameter: `steps -
    /// start_index(steps)` is how much work a run will actually do, and a
    /// caller sizing a progress bar or a time estimate needs it.
    pub fn start_index(self, steps: usize) -> usize {
        let run = (steps as f64 * self.0).round() as usize;
        steps.saturating_sub(run.min(steps))
    }
}

impl Default for Strength {
    fn default() -> Self {
        Self(0.75)
    }
}

/// Everything an inpaint needs: an img2img plus the mask that bounds it.
#[derive(Debug, Clone)]
pub struct InpaintConfig {
    pub base: Img2ImgConfig,
    /// Greyscale mask, resized to the run's dimensions. **White repaints.**
    pub mask: std::path::PathBuf,
}

/// What the sampler must leave alone, and what it is restoring.
///
/// Held together because they are only meaningful together: the mask says
/// where the original applies and `init` is the original to apply.
struct Keep<'a> {
    /// Latent-resolution mask, 1 where the model may write.
    mask: &'a Tensor,
    /// The encoded original, restored outside the mask at every step.
    init: &'a Tensor,
}

/// Everything an img2img generation needs.
#[derive(Debug, Clone)]
pub struct Img2ImgConfig {
    pub base: Txt2ImgConfig,
    /// Source image, resized to `base.width` x `base.height` on load.
    pub init_image: std::path::PathBuf,
    pub strength: Strength,
}

/// A generation steered by a ControlNet.
#[derive(Debug, Clone)]
pub struct ControlConfig {
    pub base: Txt2ImgConfig,
    /// The control map as `[1, 3, height, width]` in `[0, 1]` at **pixel**
    /// resolution — a Canny edge map, a depth map, a pose skeleton.
    ///
    /// A prepared tensor rather than a path, because what counts as a control
    /// map depends on which ControlNet is loaded and this crate cannot check
    /// the two agree. [`crate::canny`] makes one for the canny models.
    pub hint: Tensor,
    /// How strongly the control applies. 1.0 is the published strength; 0.0 is
    /// exactly an uncontrolled run.
    pub scale: f64,
}

/// A control map and its strength, for one run.
struct Hint<'a> {
    /// `[2, 3, h, w]`, already doubled for the guidance batch.
    hint: &'a Tensor,
    scale: f64,
}

/// A loaded SD 1.5 pipeline.
pub struct Txt2ImgPipeline {
    tokenizer: ClipTokenizer,
    text_encoder: ClipTextEncoder,
    unet: UNet2DConditionModel,
    vae: AutoencoderKlDecoder,
    vae_encoder: AutoencoderKlEncoder,
    schedule: Schedule,
    device: Device,
    /// Optional spatial conditioning, attached by [`Txt2ImgPipeline::with_controlnet`].
    controlnet: Option<ControlNet>,
    /// Optional tiny decoder, attached by [`Txt2ImgPipeline::with_taesd`].
    tiny: Option<TinyDecoder>,
}

impl std::fmt::Debug for Txt2ImgPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Txt2ImgPipeline")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

fn require(path: PathBuf) -> Result<PathBuf, PipelineError> {
    if path.exists() {
        Ok(path)
    } else {
        Err(PipelineError::MissingFile(path))
    }
}

/// Reduce a pixel-resolution mask to latent resolution by 8x8 **maximum**.
///
/// Maximum, not average. A latent cell covers 64 pixels, and if any of them is
/// to be repainted the cell has to be free to change — averaging would leave
/// edge cells partly frozen and produce a visible seam of half-old, half-new
/// content exactly where the join needs to be cleanest. Erring toward
/// repainting dilates the mask by up to one latent cell, which the composite
/// in pixel space then trims back to the user's actual boundary.
fn latent_mask(mask_px: &Tensor, lh: usize, lw: usize) -> Result<Tensor, PipelineError> {
    let (_, _, h, w) = mask_px.dims4()?;
    if h != lh * 8 || w != lw * 8 {
        return Err(PipelineError::Tensor(sd_tensor::Error::Msg(format!(
            "mask is {h}x{w}, expected {}x{}",
            lh * 8,
            lw * 8
        ))));
    }
    Ok(mask_px
        .reshape((1, 1, lh, 8, lw, 8))?
        .max(5)?
        .max(3)?
        .contiguous()?)
}

/// Nearest index in the training sigmas to `sigma`, as a timestep.
///
/// The UNet takes a discrete timestep, not a sigma, so the continuous sampling
/// sigma has to be mapped back onto the 1000-entry training schedule. A linear
/// search is ample: this runs once per step, twenty times per image.
pub fn sigma_to_timestep(schedule: &Schedule, sigma: f64) -> f64 {
    let sigmas = schedule.sigmas();
    let mut best = 0usize;
    let mut best_dist = f64::INFINITY;
    for (i, &s) in sigmas.iter().enumerate() {
        let d = (s - sigma).abs();
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    best as f64
}

impl Txt2ImgPipeline {
    /// Load with a LoRA merged into the UNet.
    ///
    /// `multiplier` is the adapter's strength; 0 reproduces [`Self::load`]
    /// exactly, bit for bit.
    ///
    /// **A partially-applied adapter is refused rather than rendered.** If any
    /// of the LoRA's entries finds no weight in this UNet, the adapter is for
    /// a different architecture — or the name mapping missed — and the result
    /// would be a plausible image that is not the one the adapter describes.
    /// That is the failure worth erroring on, because nothing downstream can
    /// detect it.
    pub fn load_with_lora(
        model_dir: &Path,
        device: &Device,
        lora_path: &Path,
        multiplier: f64,
    ) -> Result<Self, PipelineError> {
        let lora = sd_loader::Lora::load(lora_path, device)?;
        Self::load_inner(model_dir, device, Some((&lora, multiplier)))
    }

    /// Load from the standard diffusers directory layout.
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self, PipelineError> {
        Self::load_inner(model_dir, device, None)
    }

    fn load_inner(
        model_dir: &Path,
        device: &Device,
        lora: Option<(&sd_loader::Lora, f64)>,
    ) -> Result<Self, PipelineError> {
        let tokenizer_path = model_dir.join("tokenizer/tokenizer.json");
        if !tokenizer_path.exists() {
            // Its own variant: this one is missing from a *correct* SD 1.5
            // download, so "file not found" would send people looking for a
            // broken download rather than for the file they actually need.
            return Err(PipelineError::MissingTokenizerJson(tokenizer_path));
        }
        let text_encoder_path = require(model_dir.join("text_encoder/model.safetensors"))?;
        let unet_path = require(model_dir.join("unet/diffusion_pytorch_model.safetensors"))?;
        let vae_path = require(model_dir.join("vae/diffusion_pytorch_model.safetensors"))?;

        // See the same check in sdxl.rs: weights stay resident for the whole
        // run and dominate, so the projection has to include them.
        let weights =
            sd_loader::resident_bytes(&[&text_encoder_path, &unet_path, &vae_path], DType::F32)?;
        // The *active* tile, not the default: see the note in sdxl.rs.
        let tile = sd_models::vae::tile_latent_edge()?;
        let decode_peak = sd_models::vae::DecoderConfig::from(&VaeConfig::sd15())
            .peak_alloc_bytes(1, tile, tile, DType::F32)
            .unwrap_or(0);
        sd_tensor::sysmem::check_headroom(
            weights.saturating_add(decode_peak),
            &format!("loading the pipeline from {}", model_dir.display()),
        )?;

        let tokenizer = ClipTokenizer::from_file(&tokenizer_path)?;

        let vb = sd_loader::safetensors_var_builder(&[&text_encoder_path], DType::F32, device)?;
        let text_encoder = ClipTextEncoder::new(&ClipTextConfig::sd15(), vb)?;

        let vb = match lora {
            None => sd_loader::safetensors_var_builder(&[&unet_path], DType::F32, device)?,
            Some((lora, multiplier)) => {
                let (vb, applied) = sd_loader::safetensors_var_builder_with_lora(
                    &[&unet_path],
                    DType::F32,
                    device,
                    lora,
                    multiplier,
                )?;
                if !applied.unmatched.is_empty() {
                    return Err(PipelineError::LoraMismatch {
                        unmatched: applied.unmatched.len(),
                        first: applied.unmatched[0].clone(),
                    });
                }
                // No log line: this crate deliberately has no logging
                // dependency (see `ProgressFn`). The count is on `Applied` for
                // a caller that wants to report it.
                vb
            }
        };
        let unet = UNet2DConditionModel::new(&UNetConfig::sd15(), vb)?;

        let vb = sd_loader::safetensors_var_builder(&[&vae_path], DType::F32, device)?;
        let vae = AutoencoderKlDecoder::new(&VaeConfig::sd15(), vb).map_err(|source| {
            PipelineError::VaeWeights {
                path: vae_path.clone(),
                source,
            }
        })?;

        // Same file, both halves. The encoder is only used by img2img, but it
        // is cheap to build and mmap means the weights are not read twice.
        let vb = sd_loader::safetensors_var_builder(&[&vae_path], DType::F32, device)?;
        let vae_encoder = AutoencoderKlEncoder::new(&VaeConfig::sd15(), vb).map_err(|source| {
            PipelineError::VaeWeights {
                path: vae_path.clone(),
                source,
            }
        })?;

        Ok(Self {
            tokenizer,
            text_encoder,
            unet,
            vae,
            vae_encoder,
            schedule: Schedule::sd15(),
            device: device.clone(),
            controlnet: None,
            tiny: None,
        })
    }

    /// Load every tower from a single LDM-layout GGUF checkpoint.
    ///
    /// `tokenizer` is a separate path because **the checkpoint does not carry
    /// one**. `stable-diffusion.cpp` writes no GGUF metadata at all — not the
    /// vocabulary, not even `general.architecture` — so unlike a language
    /// model in this format, an SD checkpoint cannot supply its own
    /// tokenizer. Copy `tokenizer.json` from `openai/clip-vit-large-patch14`.
    ///
    /// Weights are dequantised on load: a 4-bit checkpoint costs what its
    /// expanded weights cost, not what the file does. The memory guard sizes
    /// each tower against that.
    pub fn load_gguf(
        gguf: &Path,
        tokenizer: &Path,
        device: &Device,
    ) -> Result<Self, PipelineError> {
        if !gguf.exists() {
            return Err(PipelineError::MissingFile(gguf.to_path_buf()));
        }
        if !tokenizer.exists() {
            return Err(PipelineError::MissingTokenizerJson(tokenizer.to_path_buf()));
        }

        let tokenizer = ClipTokenizer::from_file(tokenizer)?;

        let vb = sd_loader::clip_var_builder_from_gguf(gguf, DType::F32, device)?;
        let text_encoder = ClipTextEncoder::new(&ClipTextConfig::sd15(), vb)?;

        let vb = sd_loader::unet_var_builder_from_gguf(gguf, DType::F32, device)?;
        let unet = UNet2DConditionModel::new(&UNetConfig::sd15(), vb)?;

        let vb = sd_loader::vae_var_builder_from_gguf(gguf, DType::F32, device)?;
        let vae = AutoencoderKlDecoder::new(&VaeConfig::sd15(), vb).map_err(|source| {
            PipelineError::VaeWeights {
                path: gguf.to_path_buf(),
                source,
            }
        })?;
        let vb = sd_loader::vae_var_builder_from_gguf(gguf, DType::F32, device)?;
        let vae_encoder = AutoencoderKlEncoder::new(&VaeConfig::sd15(), vb).map_err(|source| {
            PipelineError::VaeWeights {
                path: gguf.to_path_buf(),
                source,
            }
        })?;

        Ok(Self {
            tokenizer,
            text_encoder,
            unet,
            vae,
            vae_encoder,
            schedule: Schedule::sd15(),
            device: device.clone(),
            controlnet: None,
            tiny: None,
        })
    }

    /// Attach a ControlNet.
    ///
    /// Takes the pipeline by value and returns it, so a pipeline either has a
    /// ControlNet from the moment it is built or never does — there is no
    /// window in which a caller holds one it believes is controlled and is not.
    ///
    /// The ControlNet is built from `UNetConfig::sd15()`, the same config the
    /// UNet is, which is what guarantees a correction per skip at the right
    /// width. An SDXL ControlNet here will fail to load rather than run wrong.
    pub fn with_controlnet(mut self, path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(PipelineError::MissingFile(path.to_path_buf()));
        }
        let vb = sd_loader::safetensors_var_builder(&[path], DType::F32, &self.device)?;
        self.controlnet = Some(ControlNet::new(&UNetConfig::sd15(), vb)?);
        Ok(self)
    }

    /// Use TAESD instead of the VAE for decoding.
    ///
    /// About 5 MB against the VAE's 330, and correspondingly faster. Lossier —
    /// fine detail is softened — so this is a speed and memory trade, not a
    /// free win, and it is opt-in for that reason.
    ///
    /// Only the *decoder* is replaced. Encoding (img2img, inpainting) still
    /// goes through the VAE, because a latent produced by TAESD's encoder and
    /// then denoised is not the same starting point as the VAE's.
    pub fn with_taesd(mut self, path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(PipelineError::MissingFile(path.to_path_buf()));
        }
        let vb = sd_loader::safetensors_var_builder(&[path], DType::F32, &self.device)?;
        self.tiny = Some(TinyDecoder::new(4, 3, vb)?);
        Ok(self)
    }

    /// Decode a latent with whichever decoder is attached.
    ///
    /// TAESD takes the sampler's latent unscaled, where the VAE divides by
    /// `0.18215` first — each decoder owns its own convention, so the choice
    /// is a single branch here rather than a scaling the caller has to get
    /// right.
    fn decode(&self, latent: &Tensor) -> Result<Tensor, PipelineError> {
        match &self.tiny {
            Some(tiny) => Ok(tiny.decode(latent)?),
            None => Ok(self.vae.decode_tiled(latent)?),
        }
    }

    /// Decode a latent for previewing, with whichever decoder is attached.
    ///
    /// The same decode the final image gets — there is no reduced-quality
    /// preview path, because a preview that does not look like the result is
    /// worse than none. Attach TAESD first if this is called every step.
    pub fn preview(&self, latent: &Tensor) -> Result<Tensor, PipelineError> {
        self.decode(latent)
    }

    /// Whether a ControlNet is attached.
    pub fn has_controlnet(&self) -> bool {
        self.controlnet.is_some()
    }

    /// Generate under spatial control. Returns `[1, 3, height, width]`.
    pub fn run_control(&self, cfg: &ControlConfig) -> Result<Tensor, PipelineError> {
        self.run_control_with_progress(cfg, &mut |_| {})
    }

    /// [`Self::run_control`], reporting progress after each step.
    pub fn run_control_with_progress(
        &self,
        cfg: &ControlConfig,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        if self.controlnet.is_none() {
            return Err(PipelineError::NoControlNet);
        }
        let base = &cfg.base;
        if base.width % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("width", base.width));
        }
        if base.height % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("height", base.height));
        }
        if base.steps == 0 {
            return Err(PipelineError::NoSteps);
        }
        let (hh, hw) = {
            let d = cfg.hint.dims4()?;
            (d.2, d.3)
        };
        if hh != base.height || hw != base.width {
            return Err(PipelineError::Tensor(sd_tensor::Error::Msg(format!(
                "control map is {hh}x{hw}, expected {}x{} — it is at pixel resolution, \
                 not latent",
                base.height, base.width
            ))));
        }

        let cond = self.encode(&base.prompt)?;
        let uncond = self.encode(&base.negative_prompt)?;
        let context = Tensor::cat(&[&uncond, &cond], 0)?;

        // The hint is doubled to match the guidance batch. Both halves get the
        // same control: guidance contrasts the *prompts*, and giving the
        // unconditional half no control would make the contrast partly about
        // the control map instead.
        let hint = Tensor::cat(&[&cfg.hint, &cfg.hint], 0)?.to_dtype(self.unet.dtype())?;

        let mut rng = SeededRng::new(base.seed);
        let (lh, lw) = (base.height / 8, base.width / 8);
        let sigmas = self.sigmas_for(base.sampler, base.steps);
        let latent = (rng.randn((1, 4, lh, lw), &self.device)? * sigmas[0])?;

        let latent = self.denoise_controlled(
            base,
            latent,
            &sigmas,
            &context,
            &mut rng,
            None,
            Some(Hint {
                hint: &hint,
                scale: cfg.scale,
            }),
            progress,
        )?;
        self.decode(&latent)
    }

    /// Encode a prompt to `[1, 77, 768]`.
    fn encode(&self, text: &str) -> Result<Tensor, PipelineError> {
        let ids = self.tokenizer.encode(text)?;
        let ids = Tensor::from_vec(ids, (1, self.tokenizer.max_length()), &self.device)?;
        Ok(self.text_encoder.forward(&ids)?)
    }

    /// Generate. Returns `[1, 3, height, width]` in `[-1, 1]`.
    pub fn run(&self, cfg: &Txt2ImgConfig) -> Result<Tensor, PipelineError> {
        self.run_with_progress(cfg, &mut |_| {})
    }

    /// [`Self::run`], reporting progress after each step.
    pub fn run_with_progress(
        &self,
        cfg: &Txt2ImgConfig,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        if cfg.width % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("width", cfg.width));
        }
        if cfg.height % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("height", cfg.height));
        }
        if cfg.steps == 0 {
            return Err(PipelineError::NoSteps);
        }

        // Uncond first. This order and the split below must agree; reversing
        // exactly one of them inverts guidance and produces the opposite of
        // the prompt, which is a confusing symptom to debug from the image.
        let cond = self.encode(&cfg.prompt)?;
        let uncond = self.encode(&cfg.negative_prompt)?;
        let context = Tensor::cat(&[&uncond, &cond], 0)?;

        let sigmas = self.sigmas_for(cfg.sampler, cfg.steps);
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);

        // One generator per image, drawn in order: initial latent first, then
        // one noise draw per step. A fresh generator inside the loop would
        // give every step identical noise.
        let mut rng = SeededRng::new(cfg.seed);
        // Scaled by the first sigma — unit-variance noise gives washed-out
        // output.
        let latent = (rng.randn((1, 4, lh, lw), &self.device)? * sigmas[0])?;

        let latent = self.denoise(cfg, latent, &sigmas, &context, &mut rng, progress)?;

        // `decode_tiled` applies the scaling factor, like `decode`, and falls
        // through to a whole-image decode for latents that already fit — so
        // 512px output is bit-identical to before. Above that it tiles, which
        // is what keeps a 1024px decode inside GPU memory.
        self.decode(&latent)
    }

    /// Generate inside a mask, leaving everything else alone.
    ///
    /// Latent blending rather than a dedicated inpainting checkpoint: at every
    /// step the region outside the mask is restored to the original, so any
    /// ordinary model can inpaint and no 9-channel UNet is required. The trade
    /// is that the model never *sees* a mask, so it infers the boundary from
    /// context alone — a dedicated checkpoint does better on large holes.
    ///
    /// The untouched region is exact: latent blending alone would return it
    /// through a VAE round trip, which is not lossless, so the result is
    /// composited against the original in pixel space at the end.
    pub fn run_inpaint(&self, cfg: &InpaintConfig) -> Result<Tensor, PipelineError> {
        self.run_inpaint_with_progress(cfg, &mut |_| {})
    }

    /// [`Self::run_inpaint`], reporting progress after each step.
    pub fn run_inpaint_with_progress(
        &self,
        cfg: &InpaintConfig,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        let base = &cfg.base.base;
        if base.width % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("width", base.width));
        }
        if base.height % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("height", base.height));
        }
        if base.steps == 0 {
            return Err(PipelineError::NoSteps);
        }

        let cond = self.encode(&base.prompt)?;
        let uncond = self.encode(&base.negative_prompt)?;
        let context = Tensor::cat(&[&uncond, &cond], 0)?;

        let (w, h) = (base.width as u32, base.height as u32);
        let image = crate::image_io::load_image(&cfg.base.init_image, w, h, &self.device)?;
        let mask_px = crate::image_io::load_mask(&cfg.mask, w, h, &self.device)?;
        let init = self.vae_encoder.encode(&image)?;
        let mask = latent_mask(&mask_px, base.height / 8, base.width / 8)?;

        let sigmas = self.sigmas_for(base.sampler, base.steps);
        let start = cfg.base.strength.start_index(base.steps);
        if start >= base.steps {
            // Strength 0 repaints nothing, so the original is the answer —
            // and returning it directly avoids a pointless VAE round trip.
            return Ok(image);
        }

        let mut rng = SeededRng::new(base.seed);
        let (lh, lw) = (base.height / 8, base.width / 8);
        let noise = rng.randn((1, 4, lh, lw), &self.device)?;
        let latent = (&init + (noise * sigmas[start])?)?;

        let latent = self.denoise_keeping(
            base,
            latent,
            &sigmas[start..],
            &context,
            &mut rng,
            Some(Keep {
                mask: &mask,
                init: &init,
            }),
            progress,
        )?;
        let decoded = self.decode(&latent)?;
        Ok(crate::image_io::composite(&decoded, &image, &mask_px)?)
    }

    /// The sigma ladder a sampler wants.
    ///
    /// LCM does not take an even spread: it visits the subset of timesteps its
    /// distillation used, so it builds its own ladder. Everything else shares
    /// the usual one.
    fn sigmas_for(&self, sampler: SamplerKind, steps: usize) -> Vec<f64> {
        match sampler {
            SamplerKind::Lcm => lcm_sigmas(
                &self.schedule,
                &lcm_timesteps(
                    self.schedule.alphas_cumprod.len(),
                    sd_sample::ORIGINAL_INFERENCE_STEPS,
                    steps,
                ),
            ),
            _ => sigmas_for_steps(&self.schedule, steps),
        }
    }

    /// The sampling loop, shared by txt2img and img2img.
    ///
    /// `sigmas` is a full ladder of `n + 1` boundaries; img2img passes a
    /// suffix of one. `rng` is threaded in rather than created here so the
    /// caller controls draw order, which is what makes a seed reproducible.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        cfg: &Txt2ImgConfig,
        latent: Tensor,
        sigmas: &[f64],
        context: &Tensor,
        rng: &mut SeededRng,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        self.denoise_keeping(cfg, latent, sigmas, context, rng, None, progress)
    }

    /// [`Self::denoise`], optionally holding a region at the original.
    #[allow(clippy::too_many_arguments)]
    fn denoise_keeping(
        &self,
        cfg: &Txt2ImgConfig,
        latent: Tensor,
        sigmas: &[f64],
        context: &Tensor,
        rng: &mut SeededRng,
        keep: Option<Keep<'_>>,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        self.denoise_controlled(cfg, latent, sigmas, context, rng, keep, None, progress)
    }

    /// [`Self::denoise_keeping`], optionally under a ControlNet.
    #[allow(clippy::too_many_arguments)]
    fn denoise_controlled(
        &self,
        cfg: &Txt2ImgConfig,
        mut latent: Tensor,
        sigmas: &[f64],
        context: &Tensor,
        rng: &mut SeededRng,
        keep: Option<Keep<'_>>,
        control: Option<Hint<'_>>,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);
        let steps = sigmas.len().saturating_sub(1);
        let mut dpm = DpmSolverPlusPlus2M::new();

        for i in 0..steps {
            let sigma = sigmas[i];
            let sigma_next = sigmas[i + 1];

            // Classifier-free guidance: run both conditionings in one batch.
            let latent_in = Tensor::cat(&[&latent, &latent], 0)?;
            // k-diffusion input scaling. Omitting it gives noisy, oversaturated
            // results.
            let latent_in = (latent_in / (sigma * sigma + 1.0).sqrt())?;

            let t = sigma_to_timestep(&self.schedule, sigma);
            let timestep = Tensor::new(&[t as f32, t as f32], &self.device)?;

            let out = match (&self.controlnet, &control) {
                (Some(net), Some(h)) => {
                    // The ControlNet sees the same scaled latent and timestep
                    // the UNet does. Feeding it the unscaled latent is a
                    // natural mistake that produces corrections of plausible
                    // magnitude for the wrong noise level.
                    let c = net.forward(&latent_in, &timestep, context, h.hint, h.scale)?;
                    self.unet
                        .forward_controlled(&latent_in, &timestep, context, &c.down, &c.mid)?
                }
                _ => self.unet.forward(&latent_in, &timestep, context)?,
            };
            let out_uncond = out.narrow(0, 0, 1)?;
            let out_cond = out.narrow(0, 1, 1)?;
            let noise_pred = (&out_uncond + ((out_cond - &out_uncond)? * cfg.cfg_scale)?)?;

            // The UNet predicts noise; the samplers want x0.
            let denoised = (&latent - (&noise_pred * sigma)?)?;

            latent = match cfg.sampler {
                SamplerKind::EulerAncestral => {
                    let noise = rng.randn((1, 4, lh, lw), &self.device)?;
                    euler_ancestral_step(&latent, &denoised, sigma, sigma_next, &noise)?
                }
                SamplerKind::DpmPlusPlus2M => dpm.step(&latent, &denoised, sigma, sigma_next)?,
                SamplerKind::Lcm => {
                    // Fresh noise each step: LCM re-noises rather than
                    // integrating, so a reused draw correlates the steps.
                    let noise = rng.randn((1, 4, lh, lw), &self.device)?;
                    lcm_step(&latent, &denoised, sigma, sigma_next, t, &noise)?
                }
            };

            // Restore everything outside the mask to the original, noised to
            // the level the next step expects. Doing this *inside* the loop
            // rather than once at the end is what keeps the model's context
            // honest: it sees the true surroundings at every step, so what it
            // paints actually joins up with them.
            if let Some(k) = &keep {
                let restored = if sigma_next > 0.0 {
                    let n = rng.randn((1, 4, lh, lw), &self.device)?;
                    (k.init + (n * sigma_next)?)?
                } else {
                    k.init.clone()
                };
                latent =
                    (latent.broadcast_mul(k.mask)? + restored.broadcast_mul(&(1.0 - k.mask)?)?)?;
            }

            progress(Progress {
                step: i + 1,
                total: steps,
                sigma,
                denoised: &denoised,
            });
        }
        Ok(latent)
    }

    /// Generate from an existing image. Returns `[1, 3, height, width]`.
    pub fn run_img2img(&self, cfg: &Img2ImgConfig) -> Result<Tensor, PipelineError> {
        self.run_img2img_with_progress(cfg, &mut |_| {})
    }

    /// [`Self::run_img2img`], reporting progress after each step.
    pub fn run_img2img_with_progress(
        &self,
        cfg: &Img2ImgConfig,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        let base = &cfg.base;
        if base.width % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("width", base.width));
        }
        if base.height % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("height", base.height));
        }
        if base.steps == 0 {
            return Err(PipelineError::NoSteps);
        }

        let cond = self.encode(&base.prompt)?;
        let uncond = self.encode(&base.negative_prompt)?;
        let context = Tensor::cat(&[&uncond, &cond], 0)?;

        let image = crate::image_io::load_image(
            &cfg.init_image,
            base.width as u32,
            base.height as u32,
            &self.device,
        )?;
        // The distribution mean, not a draw from it: the sampler supplies all
        // the randomness, so this stays a function of the seed alone.
        let latent = self.vae_encoder.encode(&image)?;

        let sigmas = self.sigmas_for(base.sampler, base.steps);
        let start = cfg.strength.start_index(base.steps);
        // Strength 0 means "return the input", and there is nothing to run.
        if start >= base.steps {
            return self.decode(&latent);
        }

        let mut rng = SeededRng::new(base.seed);
        let (lh, lw) = (base.height / 8, base.width / 8);
        // Noise the encoded latent to the sigma the run starts at. This is
        // what makes strength mean something: a later start is less noise and
        // so a smaller departure from the input.
        let noise = rng.randn((1, 4, lh, lw), &self.device)?;
        let latent = (latent + (noise * sigmas[start])?)?;

        let latent = self.denoise(base, latent, &sigmas[start..], &context, &mut rng, progress)?;
        self.decode(&latent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_masked_pixel_frees_its_whole_latent_cell() {
        // Max, not mean, and this is the case that distinguishes them. One
        // white pixel in an 8x8 block means that block's latent cell must be
        // free to change: a latent cell is not a pixel, and averaging would
        // give 1/64 — an almost-frozen cell, producing a hard seam exactly at
        // the mask edge, where it is most visible.
        let dev = sd_tensor::Device::Cpu;
        let mut px = vec![0f32; 16 * 16];
        px[0] = 1.0; // top-left pixel only
        let m = sd_tensor::Tensor::from_vec(px, (1, 1, 16, 16), &dev).unwrap();

        let lm = latent_mask(&m, 2, 2).unwrap();
        assert_eq!(lm.dims(), &[1, 1, 2, 2]);
        let v = lm.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![1.0, 0.0, 0.0, 0.0], "only the covering cell frees");
    }

    #[test]
    fn a_mask_that_is_not_a_multiple_of_eight_is_refused() {
        // Silently truncating would shift the mask relative to the image,
        // repainting the wrong region and still returning a plausible picture.
        let dev = sd_tensor::Device::Cpu;
        let m = sd_tensor::Tensor::zeros((1, 1, 12, 16), sd_tensor::DType::F32, &dev).unwrap();
        assert!(latent_mask(&m, 2, 2).is_err());
    }
}
