//! SDXL text-to-image.
//!
//! Structurally the same loop as SD 1.5, with a different conditioning path:
//!
//! * **two** text encoders, whose penultimate hidden states are concatenated
//!   along the feature axis to give the 2048-wide context;
//! * the second encoder's **pooled** embedding, fed to the UNet separately;
//! * six **time ids** — original size, crop offset, target size — which SDXL
//!   was trained with and expects.
//!
//! The two tokenizers are not interchangeable: the second pads with `!`
//! rather than with EOS.

use std::path::{Path, PathBuf};

use sd_models::clip::{ClipTextConfig, ClipTextEncoder, ClipTokenizer};
use sd_models::unet::{UNet2DConditionModel, UNetConfig};
use sd_models::vae::{AutoencoderKlDecoder, VaeConfig};
use sd_sample::{euler_ancestral_step, sigmas_for_steps, DpmSolverPlusPlus2M, Schedule};
use sd_tensor::rng::SeededRng;
use sd_tensor::{DType, Device, Tensor};

use super::{PipelineError, ProgressFn, SamplerKind, Txt2ImgConfig};

/// SDXL's second tokenizer pads with `!`, not `<|endoftext|>`.
const TOKENIZER_2_PAD: &str = "!";

/// A loaded SDXL pipeline.
pub struct SdxlPipeline {
    tokenizer: ClipTokenizer,
    tokenizer_2: ClipTokenizer,
    text_encoder: ClipTextEncoder,
    text_encoder_2: ClipTextEncoder,
    unet: UNet2DConditionModel,
    vae: AutoencoderKlDecoder,
    schedule: Schedule,
    device: Device,
}

impl std::fmt::Debug for SdxlPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SdxlPipeline")
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

impl SdxlPipeline {
    /// Load from the standard diffusers directory layout.
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self, PipelineError> {
        let tok_path = model_dir.join("tokenizer/tokenizer.json");
        if !tok_path.exists() {
            return Err(PipelineError::MissingTokenizerJson(tok_path));
        }
        let tok2_path = model_dir.join("tokenizer_2/tokenizer.json");
        if !tok2_path.exists() {
            return Err(PipelineError::MissingTokenizerJson(tok2_path));
        }
        let te_path = require(model_dir.join("text_encoder/model.safetensors"))?;
        let te2_path = require(model_dir.join("text_encoder_2/model.safetensors"))?;
        let unet_path = require(model_dir.join("unet/diffusion_pytorch_model.safetensors"))?;
        let vae_path = require(model_dir.join("vae/diffusion_pytorch_model.safetensors"))?;

        let tokenizer = ClipTokenizer::from_file(&tok_path)?;
        let tokenizer_2 = ClipTokenizer::from_file(&tok2_path)?.with_pad_token(TOKENIZER_2_PAD)?;

        let vb = sd_loader::safetensors_var_builder(&[&te_path], DType::F32, device)?;
        let text_encoder = ClipTextEncoder::new(&ClipTextConfig::sdxl_1(), vb)?;

        let vb = sd_loader::safetensors_var_builder(&[&te2_path], DType::F32, device)?;
        let text_encoder_2 = ClipTextEncoder::new(&ClipTextConfig::sdxl_2(), vb)?;

        let vb = sd_loader::safetensors_var_builder(&[&unet_path], DType::F32, device)?;
        let unet = UNet2DConditionModel::new(&UNetConfig::sdxl(), vb)?;

        let vb = sd_loader::safetensors_var_builder(&[&vae_path], DType::F32, device)?;
        // SDXL's VAE differs from SD 1.5's only in `scaling_factor`.
        let vae = AutoencoderKlDecoder::new(&VaeConfig::sdxl(), vb).map_err(|source| {
            PipelineError::VaeWeights {
                path: vae_path.clone(),
                source,
            }
        })?;

        Ok(Self {
            tokenizer,
            tokenizer_2,
            text_encoder,
            text_encoder_2,
            unet,
            vae,
            schedule: Schedule::sd15(),
            device: device.clone(),
        })
    }

    /// Encode a prompt through both towers.
    ///
    /// Returns the 2048-wide sequence and the 1280-wide pooled embedding.
    /// Both encoders contribute their **penultimate** hidden state; the
    /// pooled vector comes only from the second.
    fn encode(&self, text: &str) -> Result<(Tensor, Tensor), PipelineError> {
        let ids = self.tokenizer.encode(text)?;
        let ids = Tensor::from_vec(ids, (1, self.tokenizer.max_length()), &self.device)?;
        let hidden_1 = self.text_encoder.penultimate_hidden_state(&ids)?;

        let ids_2 = self.tokenizer_2.encode(text)?;
        let ids_2 = Tensor::from_vec(ids_2, (1, self.tokenizer_2.max_length()), &self.device)?;
        let hidden_2 = self.text_encoder_2.penultimate_hidden_state(&ids_2)?;
        let pooled = self.text_encoder_2.pooled(&ids_2)?.ok_or_else(|| {
            PipelineError::Tensor(sd_tensor::Error::Msg(
                "SDXL's second encoder must have a text_projection".to_string(),
            ))
        })?;

        // Concatenated on the feature axis: 768 + 1280 = 2048.
        let context = Tensor::cat(&[&hidden_1, &hidden_2], 2)?;
        Ok((context, pooled))
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

        let (cond, cond_pooled) = self.encode(&cfg.prompt)?;
        let (uncond, uncond_pooled) = self.encode(&cfg.negative_prompt)?;
        // Uncond first, matching the split in the loop.
        let context = Tensor::cat(&[&uncond, &cond], 0)?;
        let pooled = Tensor::cat(&[&uncond_pooled, &cond_pooled], 0)?;

        // original h/w, crop top/left, target h/w. Telling SDXL the image was
        // cropped from a larger original, or produced at a lower resolution
        // than it was, measurably degrades output — these are conditioning
        // inputs the model was trained on, not metadata.
        let (h, w) = (cfg.height as f32, cfg.width as f32);
        let time_ids = Tensor::new(&[h, w, 0.0, 0.0, h, w], &self.device)?
            .reshape((1, 6))?
            .repeat((2, 1))?;

        let sigmas = sigmas_for_steps(&self.schedule, cfg.steps);
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);

        let mut rng = SeededRng::new(cfg.seed);
        let mut latent = (rng.randn((1, 4, lh, lw), &self.device)? * sigmas[0])?;
        let mut dpm = DpmSolverPlusPlus2M::new();

        for i in 0..cfg.steps {
            let sigma = sigmas[i];
            let sigma_next = sigmas[i + 1];

            let latent_in = Tensor::cat(&[&latent, &latent], 0)?;
            let latent_in = (latent_in / (sigma * sigma + 1.0).sqrt())?;

            let t = super::sigma_to_timestep(&self.schedule, sigma);
            let timestep = Tensor::new(&[t as f32, t as f32], &self.device)?;

            let out = self
                .unet
                .forward_sdxl(&latent_in, &timestep, &context, &pooled, &time_ids)?;
            let out_uncond = out.narrow(0, 0, 1)?;
            let out_cond = out.narrow(0, 1, 1)?;
            let noise_pred = (&out_uncond + ((out_cond - &out_uncond)? * cfg.cfg_scale)?)?;

            let denoised = (&latent - (&noise_pred * sigma)?)?;

            latent = match cfg.sampler {
                SamplerKind::EulerAncestral => {
                    let noise = rng.randn((1, 4, lh, lw), &self.device)?;
                    euler_ancestral_step(&latent, &denoised, sigma, sigma_next, &noise)?
                }
                SamplerKind::DpmPlusPlus2M => dpm.step(&latent, &denoised, sigma, sigma_next)?,
            };

            progress(i + 1, cfg.steps, sigma);
        }

        Ok(self.vae.decode(&latent)?)
    }
}
