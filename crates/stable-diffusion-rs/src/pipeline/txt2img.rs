//! Text-to-image: tokenizer, text encoder, UNet, sampler, VAE.

use std::path::{Path, PathBuf};

use sd_models::clip::{ClipTextConfig, ClipTextEncoder, ClipTokenizer};
use sd_models::unet::{UNet2DConditionModel, UNetConfig};
use sd_models::vae::{AutoencoderKlDecoder, AutoencoderKlEncoder, VaeConfig};
use sd_sample::{euler_ancestral_step, sigmas_for_steps, DpmSolverPlusPlus2M, Schedule};
use sd_tensor::rng::SeededRng;
use sd_tensor::{DType, Device, Tensor};

/// Which sampler to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SamplerKind {
    #[default]
    EulerAncestral,
    DpmPlusPlus2M,
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
    #[error("tensor: {0}")]
    Tensor(#[from] sd_tensor::Error),
}

/// Called after each denoising step: `(step, total, sigma)`.
///
/// A callback rather than a log line because this crate has no logging
/// dependency and adding one is out of scope. The CLI owns the reporting, and
/// a 20-step CPU run needs it — it takes minutes and otherwise looks hung.
pub type ProgressFn<'a> = &'a mut dyn FnMut(usize, usize, f64);

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

/// Everything an img2img generation needs.
#[derive(Debug, Clone)]
pub struct Img2ImgConfig {
    pub base: Txt2ImgConfig,
    /// Source image, resized to `base.width` x `base.height` on load.
    pub init_image: std::path::PathBuf,
    pub strength: Strength,
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
    /// Load from the standard diffusers directory layout.
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self, PipelineError> {
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

        let tokenizer = ClipTokenizer::from_file(&tokenizer_path)?;

        let vb = sd_loader::safetensors_var_builder(&[&text_encoder_path], DType::F32, device)?;
        let text_encoder = ClipTextEncoder::new(&ClipTextConfig::sd15(), vb)?;

        let vb = sd_loader::safetensors_var_builder(&[&unet_path], DType::F32, device)?;
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
        })
    }

    /// Encode a prompt to `[1, 77, 768]`.
    fn encode(&self, text: &str) -> Result<Tensor, PipelineError> {
        let ids = self.tokenizer.encode(text)?;
        let ids = Tensor::from_vec(ids, (1, self.tokenizer.max_length()), &self.device)?;
        Ok(self.text_encoder.forward(&ids)?)
    }

    /// Generate. Returns `[1, 3, height, width]` in `[-1, 1]`.
    pub fn run(&self, cfg: &Txt2ImgConfig) -> Result<Tensor, PipelineError> {
        self.run_with_progress(cfg, &mut |_, _, _| {})
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

        let sigmas = sigmas_for_steps(&self.schedule, cfg.steps);
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);

        // One generator per image, drawn in order: initial latent first, then
        // one noise draw per step. A fresh generator inside the loop would
        // give every step identical noise.
        let mut rng = SeededRng::new(cfg.seed);
        // Scaled by the first sigma — unit-variance noise gives washed-out
        // output.
        let latent = (rng.randn((1, 4, lh, lw), &self.device)? * sigmas[0])?;

        let latent = self.denoise(cfg, latent, &sigmas, &context, &mut rng, progress)?;

        // `decode` applies the scaling factor; `decode_raw` does not. Using
        // the latter here would double-scale.
        Ok(self.vae.decode(&latent)?)
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
        mut latent: Tensor,
        sigmas: &[f64],
        context: &Tensor,
        rng: &mut SeededRng,
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

            let out = self.unet.forward(&latent_in, &timestep, context)?;
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
            };

            progress(i + 1, steps, sigma);
        }
        Ok(latent)
    }

    /// Generate from an existing image. Returns `[1, 3, height, width]`.
    pub fn run_img2img(&self, cfg: &Img2ImgConfig) -> Result<Tensor, PipelineError> {
        self.run_img2img_with_progress(cfg, &mut |_, _, _| {})
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

        let sigmas = sigmas_for_steps(&self.schedule, base.steps);
        let start = cfg.strength.start_index(base.steps);
        // Strength 0 means "return the input", and there is nothing to run.
        if start >= base.steps {
            return Ok(self.vae.decode(&latent)?);
        }

        let mut rng = SeededRng::new(base.seed);
        let (lh, lw) = (base.height / 8, base.width / 8);
        // Noise the encoded latent to the sigma the run starts at. This is
        // what makes strength mean something: a later start is less noise and
        // so a smaller departure from the input.
        let noise = rng.randn((1, 4, lh, lw), &self.device)?;
        let latent = (latent + (noise * sigmas[start])?)?;

        let latent = self.denoise(base, latent, &sigmas[start..], &context, &mut rng, progress)?;
        Ok(self.vae.decode(&latent)?)
    }
}
