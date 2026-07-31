//! Flux on MLX.
//!
//! Rectified flow over a 16-channel latent packed into 2x2 patches, conditioned
//! on T5's sequence and CLIP's pooled vector. Four things differ from every
//! other pipeline here, and each is silent when wrong.
//!
//! - **One forward per step, not two.** Guidance is *distilled in* rather than
//!   applied by contrasting a conditional and an unconditional prediction. A
//!   pipeline that batched two rows here would be twice as slow and would
//!   double-count guidance the model already carries.
//! - **CLIP contributes only its pooled vector.** Flux never sees CLIP's
//!   sequence — that job belongs to T5 — so feeding the sequence conditions the
//!   model on something it was never trained with. And it is the *raw* pooled
//!   hidden state, not the projected one SDXL and SD 3 take.
//! - **The timestep is the sigma itself**, in `[0, 1]`, not an index into a
//!   training schedule.
//! - **`guidance_embed` is a property of the checkpoint, not the caller.**
//!   schnell is not distilled on a guidance scale and rejects one; dev and
//!   flux-mini require one. Driven by the config so a setting cannot be
//!   silently discarded.
//!
//! # Not verified end to end here
//!
//! T5-XXL is not on this machine in safetensors — only as a 4-bit GGUF, which
//! dequantises to 18.8 GB and does not fit. The candle `FluxPipeline` has no
//! end-to-end test either. What *is* gated is every piece: the transformer at
//! `mlx_golden_flux` (3.49e-6), T5 at `mlx_golden_t5`, the 16-channel VAE at
//! `mlx_golden_flux_vae`, and the latent packing against candle's element for
//! element at `mlx_flux_packing_agrees`.

use std::path::{Path, PathBuf};

use sd_models::clip::ClipTokenizer;
use sd_models::mlx::{
    clip::{self, ClipConfig},
    flux, normalise_legacy_attention, t5,
    vae::{self, VaeConfig},
    Weights,
};
use sd_models::t5::T5Tokenizer;
use sd_sample::flow::{flow_sigmas, flow_timesteps, FlowMatchConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};
use sd_tensor::rng::SeededRng;

use super::{draw_noise, msg};
use crate::pipeline::PipelineError;

/// T5's sequence length for Flux. 512 for dev, 256 for schnell — the shorter
/// one is what schnell was distilled with.
pub const T5_LENGTH_SCHNELL: usize = 256;
pub const T5_LENGTH_DEV: usize = 512;

/// Where Flux's pieces live.
#[derive(Debug, Clone)]
pub struct FluxPaths {
    pub transformer: Vec<PathBuf>,
    pub vae: PathBuf,
    pub clip: PathBuf,
    pub t5: Vec<PathBuf>,
    pub clip_tokenizer: PathBuf,
    pub t5_tokenizer: PathBuf,
}

impl FluxPaths {
    /// The `diffusers` layout under `root`.
    pub fn in_dir(root: &Path) -> Self {
        Self {
            transformer: shards(&root.join("transformer"), "diffusion_pytorch_model"),
            vae: root.join("vae/diffusion_pytorch_model.safetensors"),
            clip: root.join("text_encoder/model.safetensors"),
            t5: shards(&root.join("text_encoder_2"), "model"),
            clip_tokenizer: root.join("tokenizer/tokenizer.json"),
            t5_tokenizer: root.join("tokenizer_2/spiece.model"),
        }
    }
}

/// Every shard of a sharded checkpoint, or the single file if it is not
/// sharded.
///
/// **Flux's transformer and T5-XXL both ship sharded**, and a single-file
/// assumption drops most of the model and surfaces as a missing tensor naming
/// one arbitrary layer. The directory is read rather than a shard count
/// guessed, because the count differs between releases.
fn shards(dir: &Path, stem: &str) -> Vec<PathBuf> {
    let single = dir.join(format!("{stem}.safetensors"));
    if single.exists() {
        return vec![single];
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "safetensors")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(stem))
        })
        .collect();
    // Sorted, so `00001-of-00003` precedes `00002-of-00003`. The order does
    // not affect the merged map, but a stable one makes a failure repeatable.
    found.sort();
    found
}

/// A loaded Flux pipeline on MLX.
pub struct FluxPipeline {
    clip_tokenizer: ClipTokenizer,
    t5_tokenizer: T5Tokenizer,
    clip: Weights,
    t5: Weights,
    transformer: Weights,
    vae: Weights,
    cfg: flux::FluxConfig,
    vae_cfg: VaeConfig,
    flow: FlowMatchConfig,
    stream: Stream,
}

/// One run's settings.
#[derive(Debug, Clone)]
pub struct FluxRunConfig {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// The distilled guidance scale. Ignored — and rejected — by schnell.
    pub guidance: f64,
    pub seed: u64,
}

impl Default for FluxRunConfig {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: 4,
            guidance: 3.5,
            seed: 0,
        }
    }
}

impl FluxPipeline {
    /// Load Flux from a `diffusers` model directory.
    ///
    /// `cfg` says which variant this is — `schnell()`, `dev()` or `mini()` —
    /// and with it whether a guidance scale is expected.
    pub fn load(root: &Path, cfg: flux::FluxConfig) -> Result<Self, PipelineError> {
        let paths = FluxPaths::in_dir(root);
        for p in [
            &paths.vae,
            &paths.clip,
            &paths.clip_tokenizer,
            &paths.t5_tokenizer,
        ] {
            if !p.exists() {
                return Err(PipelineError::MissingFile(p.clone()));
            }
        }
        if paths.transformer.is_empty() {
            return Err(PipelineError::MissingFile(root.join("transformer")));
        }
        if paths.t5.is_empty() {
            return Err(PipelineError::MissingFile(root.join("text_encoder_2")));
        }
        let stream = Stream::gpu();

        let mut transformer = Weights::new();
        for shard in &paths.transformer {
            transformer.extend(load_safetensors(shard)?);
        }
        let mut t5w = Weights::new();
        for shard in &paths.t5 {
            t5w.extend(load_safetensors(shard)?);
        }
        let mut vae_w = load_safetensors(&paths.vae)?;
        normalise_legacy_attention(&mut vae_w);

        let t5_len = if cfg.guidance_embed {
            T5_LENGTH_DEV
        } else {
            T5_LENGTH_SCHNELL
        };
        Ok(Self {
            clip_tokenizer: ClipTokenizer::from_file(&paths.clip_tokenizer)?,
            t5_tokenizer: T5Tokenizer::from_file(&paths.t5_tokenizer, t5_len)?,
            clip: load_safetensors(&paths.clip)?,
            t5: t5w,
            transformer,
            vae: vae_w,
            cfg,
            vae_cfg: VaeConfig::flux(),
            flow: FlowMatchConfig::flux(),
            stream,
        })
    }

    /// `(txt, pooled)`: T5's sequence and CLIP's **raw pooled** vector.
    pub fn conditioning(&self, prompt: &str) -> Result<(Array, Array), PipelineError> {
        let s = &self.stream;
        let t5_ids = self.t5_tokenizer.encode(prompt)?;
        let v: Vec<i32> = t5_ids.iter().map(|&x| x as i32).collect();
        let t5_ids = Array::from_slice_i32(&v, &[1, v.len()])?;
        let txt = t5::encode(&t5_ids, &t5::T5Config::xxl(), &self.t5, s)?;

        let ids = self.clip_tokenizer.encode(prompt)?;
        let v: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids = Array::from_slice_i32(&v, &[1, v.len()])?;
        // **The pooled output only, and unprojected.** Flux takes
        // `transformers`' raw pooler output; SDXL and SD 3 take the projected
        // one, and the two are different vectors.
        let hidden = clip::text_encoder_with(&ids, &ClipConfig::sd15(), &self.clip, s)?;
        let pooled = clip::pool(&hidden, &ids, s)?;
        Ok((txt, pooled))
    }

    /// Prompt to pixels. Returns `(width, height, RGB bytes)`.
    pub fn txt2img(&self, cfg: &FluxRunConfig) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        // 16, not 8: the VAE downsamples by 8 and the patchifier halves again,
        // so an odd number of latent rows cannot pack into 2x2 patches.
        for (what, v) in [("width", cfg.width), ("height", cfg.height)] {
            if v % 16 != 0 {
                return Err(msg(format!(
                    "mlx: {what} {v} must be a multiple of 16 for Flux's 2x2 patches"
                )));
            }
        }
        if cfg.steps == 0 {
            return Err(PipelineError::NoSteps);
        }
        let s = &self.stream;
        let (lat_h, lat_w) = (cfg.height / 8, cfg.width / 8);
        let (patch_h, patch_w) = (lat_h / 2, lat_w / 2);
        let img_len = patch_h * patch_w;

        let (txt, pooled) = self.conditioning(&cfg.prompt)?;

        let mut rng = SeededRng::new(cfg.seed);
        // NCHW, because `pack_latents` reads `[b, c, h, w]`.
        let noise = draw_noise(&mut rng, 16, lat_h, lat_w)?;
        let latents = noise.transpose(&[0, 3, 1, 2], s)?.contiguous(s)?;
        let mut xs = flux::pack_latents(&latents, s)?;

        // The ladder depends on how many tokens the transformer will see.
        let sigmas = flow_sigmas(&self.flow, cfg.steps, img_len);
        let timesteps = flow_timesteps(&self.flow, &sigmas);
        let img_ids = flux::image_ids(lat_h, lat_w);

        // Driven by the checkpoint, so a `guidance` setting cannot be silently
        // discarded — nor a required one silently omitted.
        let guidance = if self.cfg.guidance_embed {
            Some(Array::from_slice_f32(&[cfg.guidance as f32], &[1])?)
        } else {
            None
        };

        for (i, &t) in timesteps.iter().enumerate() {
            // Flux's timestep is the sigma itself, in [0, 1], not an index.
            let timestep = Array::from_slice_f32(&[(t / 1000.0) as f32], &[1])?;
            let velocity = flux::forward(
                &xs,
                &img_ids,
                &txt,
                &timestep,
                &pooled,
                guidance.as_ref(),
                &self.cfg,
                &self.transformer,
                s,
            )?;
            // `x + v * (sigma_next - sigma)`.
            xs = xs.add(
                &velocity.mul(&Array::scalar_f32((sigmas[i + 1] - sigmas[i]) as f32)?, s)?,
                s,
            )?;
        }

        // Unpack before decoding: handing the decoder the packed form would
        // decode nonsense at a shape that still looks reasonable.
        let latents = flux::unpack_latents(&xs, lat_h, lat_w, s)?;
        let nhwc = latents.transpose(&[0, 2, 3, 1], s)?.contiguous(s)?;
        let unscaled = self.vae_cfg.unscale(&nhwc, s)?;
        let image = vae::decode_with(&unscaled, &self.vae_cfg, &self.vae, s)?;
        let [_, h, w, _] = image.shape()[..] else {
            return Err(msg(format!("mlx: flux decode {:?}", image.shape())));
        };
        let bytes = image
            .to_vec_f32(s)?
            .iter()
            .map(|&v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        Ok((w, h, bytes))
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}
