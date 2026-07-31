//! SDXL on MLX.
//!
//! Three things separate this from [`super::MlxPipeline`], and each is a silent
//! failure when wrong.
//!
//! - **Two text encoders, concatenated on the feature axis.** CLIP-L's 768 and
//!   OpenCLIP-bigG's 1280 make the UNet's 2048. Either one alone is the wrong
//!   width and fails loudly; swapping their order keeps the width and
//!   conditions on nonsense.
//! - **The penultimate hidden state**, from both, and without
//!   `final_layer_norm`. SDXL conditions on `hidden_states[-2]`; taking the
//!   last layer produces images recognisably related to the prompt and
//!   consistently worse.
//! - **Micro-conditioning**: the pooled embedding from the *second* encoder,
//!   plus six time ids. Those are conditioning inputs SDXL was trained on, not
//!   metadata — telling it the image was cropped from a larger original, or
//!   produced at a lower resolution than it was, measurably degrades the
//!   output.
//!
//! **The second tokenizer pads with `!`, not `<|endoftext|>`.** That is what
//! makes the argmax pooling unambiguous there — see `clip::pool`, where the
//! same rule costs 67 positions on a CLIP-L sequence.

use std::path::{Path, PathBuf};

use sd_models::clip::ClipTokenizer;
use sd_models::mlx::{
    clip::{self, ClipConfig},
    normalise_legacy_attention, sample, unet_forward_adapters, vae, Adapters, UNetConfig, Weights,
};
use sd_sample::{sigmas_for_steps, steps, Schedule};
use sd_tensor::mlx::{concat, load_safetensors, Array, Device, Stream};
use sd_tensor::rng::SeededRng;

use super::{draw_noise, msg, timestep_for};
use crate::pipeline::{PipelineError, SamplerKind, Strength, Txt2ImgConfig};

/// SDXL's second tokenizer pads with `!`, not `<|endoftext|>`.
const TOKENIZER_2_PAD: &str = "!";

/// SDXL's VAE scaling factor is 0.13025, not SD 1.5's 0.18215.
const VAE: fn() -> vae::VaeConfig = vae::VaeConfig::sdxl;

/// A loaded SDXL pipeline on MLX.
pub struct SdxlPipeline {
    tokenizer: ClipTokenizer,
    tokenizer_2: ClipTokenizer,
    text_encoder: Weights,
    text_encoder_2: Weights,
    unet: Weights,
    vae: Weights,
    cfg: UNetConfig,
    vae_cfg: vae::VaeConfig,
    schedule: Schedule,
    stream: Stream,
}

fn require(path: PathBuf) -> Result<PathBuf, PipelineError> {
    if path.exists() {
        Ok(path)
    } else {
        Err(PipelineError::MissingFile(path))
    }
}

impl SdxlPipeline {
    /// Load SDXL from a `diffusers` model directory, on the GPU.
    pub fn load(root: &Path) -> Result<Self, PipelineError> {
        Self::load_on(root, Device::default())
    }

    /// [`Self::load`] on a named device.
    pub fn load_on(root: &Path, device: Device) -> Result<Self, PipelineError> {
        // Not `require`: a tokenizer directory that is empty, or absent, is
        // handled by `ClipTokenizer::open` falling back to the vendored
        // vocabulary. Both towers use the same one — SDXL's `tokenizer_2`
        // ships it byte for byte identical — and differ only in padding.
        let tok = root.join("tokenizer");
        let tok2 = root.join("tokenizer_2");
        let te = require(root.join("text_encoder/model.safetensors"))?;
        let te2 = require(root.join("text_encoder_2/model.safetensors"))?;
        let unet_p = require(root.join("unet/diffusion_pytorch_model.safetensors"))?;
        let vae_p = require(root.join("vae/diffusion_pytorch_model.safetensors"))?;

        let stream = Stream::for_device(device);
        let mut vae_w = load_safetensors(&vae_p)?;
        normalise_legacy_attention(&mut vae_w);
        let mut unet = load_safetensors(&unet_p)?;
        normalise_legacy_attention(&mut unet);

        Ok(Self {
            tokenizer: ClipTokenizer::open(&tok)?,
            tokenizer_2: ClipTokenizer::open(&tok2)?.with_pad_token(TOKENIZER_2_PAD)?,
            text_encoder: load_safetensors(&te)?,
            text_encoder_2: load_safetensors(&te2)?,
            unet,
            vae: vae_w,
            cfg: UNetConfig::sdxl(),
            vae_cfg: VAE(),
            schedule: Schedule::sd15(),
            stream,
        })
    }

    /// Each tower's contribution, separately: CLIP-L's `[1, 77, 768]`,
    /// bigG's `[1, 77, 1280]`, and bigG's pooled `[1, 1280]`.
    ///
    /// The primitive, so [`Self::encode`] is only the concatenation and a test
    /// can check that order against halves it obtained independently.
    pub fn encode_halves(&self, text: &str) -> Result<(Array, Array, Array), PipelineError> {
        let s = &self.stream;
        let ids = self.tokenizer.encode(text)?;
        let v: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids = Array::from_slice_i32(&v, &[1, v.len()])?;
        let hidden_1 = clip::penultimate(&ids, &ClipConfig::sd15(), &self.text_encoder, s)?;

        let ids_2 = self.tokenizer_2.encode(text)?;
        let v2: Vec<i32> = ids_2.iter().map(|&x| x as i32).collect();
        let ids_2 = Array::from_slice_i32(&v2, &[1, v2.len()])?;
        // One forward, two outputs — `penultimate` then `pool` would encode
        // the prompt twice.
        let (hidden_2, pooled) =
            clip::sdxl_conditioning(&ids_2, &ClipConfig::sdxl_2(), &self.text_encoder_2, s)?;
        Ok((hidden_1, hidden_2, pooled))
    }

    /// One prompt through both towers: `([1, 77, 2048], [1, 1280])`.
    ///
    /// **CLIP-L first.** 768 + 1280 and 1280 + 768 are both 2048, so the wrong
    /// order loads, runs, and conditions on nonsense.
    fn encode(&self, text: &str) -> Result<(Array, Array), PipelineError> {
        let (hidden_1, hidden_2, pooled) = self.encode_halves(text)?;
        // Concatenated on the **feature** axis.
        let context = concat(&[&hidden_1, &hidden_2], 2, &self.stream)?;
        Ok((context, pooled))
    }

    /// Context, pooled and time ids for a prompt pair. Unconditional first in
    /// all three, matching the split in the loop.
    fn conditioning(&self, cfg: &Txt2ImgConfig) -> Result<(Array, Array, Array), PipelineError> {
        let s = &self.stream;
        let (cond, cond_pooled) = self.encode(&cfg.prompt)?;
        let (uncond, uncond_pooled) = self.encode(&cfg.negative_prompt)?;
        let context = concat(&[&uncond, &cond], 0, s)?;
        let pooled = concat(&[&uncond_pooled, &cond_pooled], 0, s)?;

        // original h/w, crop top/left, target h/w — conditioning SDXL was
        // trained on, not metadata.
        let (h, w) = (cfg.height as f32, cfg.width as f32);
        let row = [h, w, 0.0, 0.0, h, w];
        let both: Vec<f32> = row.iter().chain(row.iter()).copied().collect();
        let time_ids = Array::from_slice_f32(&both, &[2, 6])?;
        Ok((context, pooled, time_ids))
    }

    fn sigmas(&self, sampler: SamplerKind, steps: usize) -> Vec<f64> {
        match sampler {
            SamplerKind::Lcm => sd_sample::lcm_sigmas(
                &self.schedule,
                &sd_sample::lcm_timesteps(
                    self.schedule.alphas_cumprod.len(),
                    sd_sample::ORIGINAL_INFERENCE_STEPS,
                    steps,
                ),
            ),
            _ => sigmas_for_steps(&self.schedule, steps),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        mut latent: Array,
        context: &Array,
        pooled: &Array,
        time_ids: &Array,
        sigmas: &[f64],
        cfg_scale: f64,
        sampler: SamplerKind,
        rng: &mut SeededRng,
    ) -> Result<Array, PipelineError> {
        let s = &self.stream;
        let [_, lh, lw, lc] = latent.shape()[..] else {
            return Err(msg(format!("mlx: sdxl latent {:?}", latent.shape())));
        };
        let mut dpm = sample::DpmSolverPlusPlus2M::new();

        // One model evaluation, as a closure rather than inline: the
        // second-order samplers call it twice per step, at a point the first
        // half of the step chooses.
        let predict = |x: &Array, sigma: f64| -> Result<Array, PipelineError> {
            let latent_in = sample::scale_model_input(x, sigma, s)?;
            let t = timestep_for(&self.schedule, sigma);
            let timestep = Array::from_slice_f32(&[t, t], &[2])?;
            let out = unet_forward_adapters(
                &latent_in,
                &timestep,
                context,
                Some((pooled, time_ids)),
                None,
                &Adapters::default(),
                &self.cfg,
                &self.unet,
                s,
            )?;
            let noise_pred = sample::guidance(&out, cfg_scale, s)?;
            Ok(sample::denoise_epsilon(x, &noise_pred, sigma, s)?)
        };

        for i in 0..sigmas.len().saturating_sub(1) {
            let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
            let denoised = predict(&latent, sigma)?;

            latent = match sampler {
                SamplerKind::EulerAncestral | SamplerKind::Lcm => {
                    let noise = draw_noise(rng, lc, lh, lw)?;
                    sample::euler_ancestral_step(&latent, &denoised, sigma, sigma_next, &noise, s)?
                }
                SamplerKind::Euler | SamplerKind::Ddim => {
                    sample::euler_step(&latent, &denoised, sigma, sigma_next, s)?
                }
                SamplerKind::DpmPlusPlus2M => dpm.step(&latent, &denoised, sigma, sigma_next, s)?,
                SamplerKind::Heun => {
                    if sigma_next <= 0.0 {
                        sample::euler_step(&latent, &denoised, sigma, sigma_next, s)?
                    } else {
                        let euler = sample::euler_step(&latent, &denoised, sigma, sigma_next, s)?;
                        let denoised_next = predict(&euler, sigma_next)?;
                        sample::heun_step(&latent, &denoised, &denoised_next, sigma, sigma_next, s)?
                    }
                }
                SamplerKind::DpmPlusPlus2SAncestral => {
                    let (sigma_up, sigma_down) = steps::ancestral_split(sigma, sigma_next, 1.0);
                    let stepped = if sigma_down <= 0.0 {
                        denoised.contiguous(s)?
                    } else {
                        let (sigma_mid, a, b) = steps::dpmpp_2s_midpoint(sigma, sigma_down);
                        let mid = sample::blend(&latent, &denoised, a, b, s)?;
                        let denoised_mid = predict(&mid, sigma_mid)?;
                        let (c, d) = steps::dpmpp_2s_step(sigma, sigma_down);
                        sample::blend(&latent, &denoised_mid, c, d, s)?
                    };
                    let noise = draw_noise(rng, lc, lh, lw)?;
                    sample::add_noise(&stepped, &noise, sigma_up, s)?
                }
            };
        }
        Ok(latent)
    }

    fn decode(&self, latent: &Array) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        let s = &self.stream;
        let unscaled = self.vae_cfg.unscale(latent, s)?;
        let image = vae::decode_with(&unscaled, &self.vae_cfg, &self.vae, s)?;
        let [_, h, w, _] = image.shape()[..] else {
            return Err(msg(format!("mlx: sdxl decode {:?}", image.shape())));
        };
        let bytes = image
            .to_vec_f32(s)?
            .iter()
            .map(|&v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        Ok((w, h, bytes))
    }

    /// Prompt to pixels. Returns `(width, height, RGB bytes)`.
    ///
    /// **SDXL below its native 1024 is out of distribution**, not merely
    /// smaller — `docs/handoff.md` records what 512 looks like. This does not
    /// refuse it, because the same is true of every resolution to a degree and
    /// a hard floor would be a different opinion than the model's; but a caller
    /// asking for 512 and getting mush is not a bug here.
    pub fn txt2img(&self, cfg: &Txt2ImgConfig) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        if cfg.width % 8 != 0 || cfg.height % 8 != 0 {
            return Err(msg(format!(
                "mlx: {}x{} does not divide into 8-pixel latent cells",
                cfg.width, cfg.height
            )));
        }
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);
        let (context, pooled, time_ids) = self.conditioning(cfg)?;
        let sigmas = self.sigmas(cfg.sampler, cfg.steps);

        let mut rng = SeededRng::new(cfg.seed);
        let latent = draw_noise(&mut rng, 4, lh, lw)?
            .mul(&Array::scalar_f32(sigmas[0] as f32)?, &self.stream)?;

        let latent = self.denoise(
            latent,
            &context,
            &pooled,
            &time_ids,
            &sigmas,
            cfg.cfg_scale,
            cfg.sampler,
            &mut rng,
        )?;
        self.decode(&latent)
    }

    /// The stream this pipeline runs on.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// [`Self::encode`], exposed so a test can check the concatenation order
    /// without running the UNet. The order is invisible to a shape check —
    /// 768 + 1280 and 1280 + 768 are both 2048.
    pub fn encode_for_test(&self, text: &str) -> Result<(Array, Array), PipelineError> {
        self.encode(text)
    }

    /// An image and a prompt to pixels. `image` is `[1, h, w, 3]` in `[-1, 1]`.
    pub fn img2img(
        &self,
        cfg: &Txt2ImgConfig,
        image: &Array,
        strength: Strength,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        let s = &self.stream;
        let init = vae::encode_scaled(image, &self.vae_cfg, &self.vae, s)?;
        let [_, lh, lw, _] = init.shape()[..] else {
            return Err(msg(format!("mlx: sdxl encode {:?}", init.shape())));
        };
        let sigmas = self.sigmas(cfg.sampler, cfg.steps);
        let start = strength.start_index(cfg.steps);
        if start >= cfg.steps {
            return self.decode(&init);
        }

        let mut rng = SeededRng::new(cfg.seed);
        let noise = draw_noise(&mut rng, 4, lh, lw)?;
        let latent = sample::noise_to_sigma(&init, &noise, sigmas[start], s)?;
        let (context, pooled, time_ids) = self.conditioning(cfg)?;

        let latent = self.denoise(
            latent,
            &context,
            &pooled,
            &time_ids,
            &sigmas[start..],
            cfg.cfg_scale,
            cfg.sampler,
            &mut rng,
        )?;
        self.decode(&latent)
    }
}
