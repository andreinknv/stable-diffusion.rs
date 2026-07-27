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
use sd_models::vae::{AutoencoderKlDecoder, AutoencoderKlEncoder, VaeConfig};
use sd_sample::{euler_ancestral_step, sigmas_for_steps, DpmSolverPlusPlus2M, Schedule};
use sd_tensor::rng::SeededRng;
use sd_tensor::{DType, Device, Tensor};

use super::{Img2ImgConfig, PipelineError, ProgressFn, SamplerKind, Txt2ImgConfig};

/// SDXL's second tokenizer pads with `!`, not `<|endoftext|>`.
const TOKENIZER_2_PAD: &str = "!";

/// Dtype for the UNet and text encoders.
///
/// SDXL's checkpoints ship as fp16. Upcasting them to f32 doubles what sits
/// on the GPU — 13.9 GB for this model, 10.3 GB of it the UNet — which does
/// not leave room for a decode on a 36 GiB machine. Holding them in f16
/// roughly halves that, and is what every other implementation does.
const MODEL_DTYPE: DType = DType::F16;

/// Dtype for the VAE, and for the sampler's own arithmetic.
///
/// Deliberately not f16. SDXL's VAE overflows in fp16 — a well-known defect,
/// which is why `madebyollin/sdxl-vae-fp16-fix` exists — and it is small
/// enough (167 MB) that keeping it in f32 costs nothing worth having.
const VAE_DTYPE: DType = DType::F32;

/// A loaded SDXL pipeline.
pub struct SdxlPipeline {
    tokenizer: ClipTokenizer,
    tokenizer_2: ClipTokenizer,
    text_encoder: ClipTextEncoder,
    text_encoder_2: ClipTextEncoder,
    unet: UNet2DConditionModel,
    vae: AutoencoderKlDecoder,
    vae_encoder: AutoencoderKlEncoder,
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

        // Cost the whole thing before committing to it. The weights dominate
        // — 6.9 GB of files here — and they stay resident for the entire run,
        // so checking a single activation against a fixed ceiling would never
        // have caught the case that mattered.
        let weights = sd_loader::resident_bytes(&[&te_path, &te2_path, &unet_path], MODEL_DTYPE)?
            .saturating_add(sd_loader::resident_bytes(&[&vae_path], VAE_DTYPE)?);
        // Plus the largest single allocation a decode will make. Tiling keeps
        // this to one tile — the *active* tile, since lowering it is exactly
        // how a caller makes a decode fit, and projecting the default here
        // would refuse loads that the smaller tile allows.
        let tile = sd_models::vae::tile_latent_edge()?;
        let decode_peak = sd_models::vae::DecoderConfig::from(&VaeConfig::sdxl())
            .peak_alloc_bytes(1, tile, tile, VAE_DTYPE)
            .unwrap_or(0);
        sd_tensor::sysmem::check_headroom(
            weights.saturating_add(decode_peak),
            &format!("loading SDXL from {}", model_dir.display()),
        )?;

        let tokenizer = ClipTokenizer::from_file(&tok_path)?;
        let tokenizer_2 = ClipTokenizer::from_file(&tok2_path)?.with_pad_token(TOKENIZER_2_PAD)?;

        let vb = sd_loader::safetensors_var_builder(&[&te_path], MODEL_DTYPE, device)?;
        let text_encoder = ClipTextEncoder::new(&ClipTextConfig::sdxl_1(), vb)?;

        let vb = sd_loader::safetensors_var_builder(&[&te2_path], MODEL_DTYPE, device)?;
        let text_encoder_2 = ClipTextEncoder::new(&ClipTextConfig::sdxl_2(), vb)?;

        let vb = sd_loader::safetensors_var_builder(&[&unet_path], MODEL_DTYPE, device)?;
        let unet = UNet2DConditionModel::new(&UNetConfig::sdxl(), vb)?;

        let vb = sd_loader::safetensors_var_builder(&[&vae_path], VAE_DTYPE, device)?;
        // SDXL's VAE differs from SD 1.5's only in `scaling_factor`.
        let vae = AutoencoderKlDecoder::new(&VaeConfig::sdxl(), vb).map_err(|source| {
            PipelineError::VaeWeights {
                path: vae_path.clone(),
                source,
            }
        })?;

        // Same file, both halves; mmap means the weights are not read twice.
        let vb = sd_loader::safetensors_var_builder(&[&vae_path], VAE_DTYPE, device)?;
        let vae_encoder = AutoencoderKlEncoder::new(&VaeConfig::sdxl(), vb).map_err(|source| {
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
            vae_encoder,
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

        // original h/w, crop top/left, target h/w. Telling SDXL the image was
        // cropped from a larger original, or produced at a lower resolution
        // than it was, measurably degrades output — these are conditioning
        // inputs the model was trained on, not metadata.
        let (context, pooled, time_ids) = self.conditioning(cfg)?;

        let sigmas = sigmas_for_steps(&self.schedule, cfg.steps);
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);

        let mut rng = SeededRng::new(cfg.seed);
        let latent = (rng.randn((1, 4, lh, lw), &self.device)? * sigmas[0])?;

        let latent = self.denoise(
            cfg, latent, &sigmas, &context, &pooled, &time_ids, &mut rng, progress,
        )?;
        // Tiled: SDXL's native 1024 needs a 9.66 GB conv intermediate as a
        // single decode, which does not fit in GPU memory. See
        // docs/backends.md.
        Ok(self.vae.decode_tiled(&latent)?)
    }

    /// The sampling loop, shared by txt2img and img2img.
    ///
    /// `sigmas` is a full ladder; img2img passes a suffix of one.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        cfg: &Txt2ImgConfig,
        mut latent: Tensor,
        sigmas: &[f64],
        context: &Tensor,
        pooled: &Tensor,
        time_ids: &Tensor,
        rng: &mut SeededRng,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);
        let steps = sigmas.len().saturating_sub(1);
        let mut dpm = DpmSolverPlusPlus2M::new();

        for i in 0..steps {
            let sigma = sigmas[i];
            let sigma_next = sigmas[i + 1];

            let latent_in = Tensor::cat(&[&latent, &latent], 0)?;
            let latent_in = (latent_in / (sigma * sigma + 1.0).sqrt())?;

            let t = super::sigma_to_timestep(&self.schedule, sigma);
            let timestep = Tensor::new(&[t as f32, t as f32], &self.device)?;

            // Into the model's dtype for the forward, and straight back out.
            // The latent and every sigma calculation stay f32: they are tiny
            // next to the weights, and f16 has neither the range for a sigma
            // of 14.6 squared nor the precision for the guidance subtraction.
            let out = self
                .unet
                .forward_sdxl(
                    &latent_in.to_dtype(self.unet.dtype())?,
                    &timestep,
                    context,
                    pooled,
                    time_ids,
                )?
                .to_dtype(VAE_DTYPE)?;
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

            progress(i + 1, steps, sigma);
        }
        Ok(latent)
    }

    /// Conditioning for a prompt pair: context, pooled, time ids.
    ///
    /// Uncond first in both batched tensors, matching the split in the loop.
    fn conditioning(&self, cfg: &Txt2ImgConfig) -> Result<(Tensor, Tensor, Tensor), PipelineError> {
        let (cond, cond_pooled) = self.encode(&cfg.prompt)?;
        let (uncond, uncond_pooled) = self.encode(&cfg.negative_prompt)?;
        let context = Tensor::cat(&[&uncond, &cond], 0)?;
        let pooled = Tensor::cat(&[&uncond_pooled, &cond_pooled], 0)?;

        // original h/w, crop top/left, target h/w. These are conditioning
        // inputs SDXL was trained on, not metadata: telling it the image was
        // cropped from a larger original, or produced at a lower resolution
        // than it was, measurably degrades the output.
        //
        // f32, because the sinusoid inside the UNet is computed in f32 and
        // cast to the model dtype there.
        let (h, w) = (cfg.height as f32, cfg.width as f32);
        let time_ids = Tensor::new(&[h, w, 0.0, 0.0, h, w], &self.device)?
            .reshape((1, 6))?
            .repeat((2, 1))?;
        Ok((context, pooled, time_ids))
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

        let (context, pooled, time_ids) = self.conditioning(base)?;

        let image = crate::image_io::load_image(
            &cfg.init_image,
            base.width as u32,
            base.height as u32,
            &self.device,
        )?;
        // The distribution mean, not a draw from it — the sampler owns all the
        // randomness, so a run stays a function of the seed alone.
        let latent = self.vae_encoder.encode(&image)?;

        let sigmas = sigmas_for_steps(&self.schedule, base.steps);
        let start = cfg.strength.start_index(base.steps);
        if start >= base.steps {
            return Ok(self.vae.decode_tiled(&latent)?);
        }

        let mut rng = SeededRng::new(base.seed);
        let (lh, lw) = (base.height / 8, base.width / 8);
        let noise = rng.randn((1, 4, lh, lw), &self.device)?;
        let latent = (latent + (noise * sigmas[start])?)?;

        let latent = self.denoise(
            base,
            latent,
            &sigmas[start..],
            &context,
            &pooled,
            &time_ids,
            &mut rng,
            progress,
        )?;
        Ok(self.vae.decode_tiled(&latent)?)
    }
}
