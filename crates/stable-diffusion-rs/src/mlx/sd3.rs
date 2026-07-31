//! SD 3.5 on MLX.
//!
//! Three text encoders, a flow-matching sampler, and a 16-channel latent. Each
//! differs from the SD 1.5 path in a way that is silent when wrong.
//!
//! # The context is three encoders in a fixed layout
//!
//! ```text
//!   [ CLIP-L 768 | CLIP-G 1280 | zeros to 4096 ]   77 tokens
//!   [ T5 4096                                  ]  256 tokens
//! ```
//!
//! CLIP's two sequences concatenate to 2048 and are then **zero-padded out to
//! T5's 4096** — the transformer sees one context width and CLIP simply
//! occupies the first half of each of its tokens. The padding is deliberate,
//! not a placeholder. Then the two blocks join along the *token* axis, CLIP
//! first.
//!
//! The two *pooled* vectors concatenate to 2048 and become `y`, the vector the
//! adaLN modulation is built from. They come from the **projection heads**,
//! where the sequences come from the penultimate layer — different depths of
//! the same forward, which is why each tower runs once and both are taken from
//! it.
//!
//! # Flow matching, not diffusion
//!
//! The model predicts a **velocity**, and a step is `x + v * (sigma_next -
//! sigma)`. There is no epsilon, no `denoised`, and no ancestral noise. The
//! sigma ladder is resolution-dependent — `flow_sigmas` takes the image
//! sequence length — so a ladder computed for a different size is wrong in a
//! way that still runs.
//!
//! All of that arithmetic is `sd_sample::flow`, which returns `Vec<f64>` and
//! touches no tensor, so both backends call it. Only the step itself is here.

use std::path::{Path, PathBuf};

use sd_models::clip::ClipTokenizer;
use sd_models::mlx::{
    clip::{self, ClipConfig},
    normalise_legacy_attention, sd3, t5,
    vae::{self, VaeConfig},
    Weights,
};
use sd_models::t5::T5Tokenizer;
use sd_sample::flow::{flow_sigmas, flow_timesteps, FlowMatchConfig};
use sd_tensor::mlx::{concat, load_safetensors, Array, Stream};
use sd_tensor::rng::SeededRng;

use super::{draw_noise, msg};
use crate::pipeline::PipelineError;

/// T5's sequence length for SD 3. Not CLIP's 77.
const T5_LENGTH: usize = 256;
/// The width the transformer sees. CLIP is padded up to it.
const CONTEXT_WIDTH: usize = 4096;

/// Where SD 3.5's six pieces live.
#[derive(Debug, Clone)]
pub struct Sd3Paths {
    pub transformer: PathBuf,
    pub vae: PathBuf,
    pub clip_l: PathBuf,
    pub clip_g: PathBuf,
    pub t5: Vec<PathBuf>,
    pub clip_tokenizer: PathBuf,
    pub t5_tokenizer: PathBuf,
}

impl Sd3Paths {
    /// The `diffusers` layout under `root`.
    ///
    /// T5-XXL ships as two shards; both are loaded and merged, because a
    /// single-file assumption silently drops half the encoder and produces a
    /// missing-tensor error naming one arbitrary layer.
    pub fn in_dir(root: &Path) -> Self {
        Self {
            transformer: root.join("transformer/diffusion_pytorch_model.safetensors"),
            vae: root.join("vae/diffusion_pytorch_model.safetensors"),
            clip_l: root.join("text_encoder/model.safetensors"),
            clip_g: root.join("text_encoder_2/model.safetensors"),
            t5: vec![
                root.join("text_encoder_3/model-00001-of-00002.safetensors"),
                root.join("text_encoder_3/model-00002-of-00002.safetensors"),
            ],
            clip_tokenizer: root.join("tokenizer/tokenizer.json"),
            t5_tokenizer: root.join("tokenizer_3/spiece.model"),
        }
    }
}

/// A loaded SD 3.5 pipeline on MLX.
pub struct Sd3Pipeline {
    clip_tokenizer: ClipTokenizer,
    t5_tokenizer: T5Tokenizer,
    clip_l: Weights,
    clip_g: Weights,
    t5: Weights,
    transformer: Weights,
    vae: Weights,
    cfg: sd3::Sd3Config,
    vae_cfg: VaeConfig,
    flow: FlowMatchConfig,
    stream: Stream,
}

/// One run's settings. Flow matching has no `cfg_scale` ladder of samplers to
/// choose between, so this is smaller than [`crate::pipeline::Txt2ImgConfig`].
#[derive(Debug, Clone)]
pub struct Sd3RunConfig {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub cfg_scale: f64,
    pub seed: u64,
}

impl Default for Sd3RunConfig {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: 28,
            cfg_scale: 4.5,
            seed: 0,
        }
    }
}

fn require(path: &Path) -> Result<(), PipelineError> {
    if path.exists() {
        Ok(())
    } else {
        Err(PipelineError::MissingFile(path.to_path_buf()))
    }
}

impl Sd3Pipeline {
    /// Load SD 3.5 medium from a `diffusers` model directory.
    pub fn load(root: &Path) -> Result<Self, PipelineError> {
        let paths = Sd3Paths::in_dir(root);
        for p in [
            &paths.transformer,
            &paths.vae,
            &paths.clip_l,
            &paths.clip_g,
            &paths.clip_tokenizer,
            &paths.t5_tokenizer,
        ] {
            require(p)?;
        }
        let stream = Stream::gpu();

        // Both T5 shards, merged. A single-file assumption drops half the
        // encoder and surfaces as a missing tensor naming one arbitrary layer.
        let mut t5w: Weights = Weights::new();
        for shard in &paths.t5 {
            require(shard)?;
            t5w.extend(load_safetensors(shard)?);
        }

        let mut vae_w = load_safetensors(&paths.vae)?;
        normalise_legacy_attention(&mut vae_w);

        Ok(Self {
            clip_tokenizer: ClipTokenizer::from_file(&paths.clip_tokenizer)?,
            t5_tokenizer: T5Tokenizer::from_file(&paths.t5_tokenizer, T5_LENGTH)?,
            clip_l: load_safetensors(&paths.clip_l)?,
            clip_g: load_safetensors(&paths.clip_g)?,
            t5: t5w,
            transformer: load_safetensors(&paths.transformer)?,
            vae: vae_w,
            cfg: sd3::Sd3Config::medium_35(),
            vae_cfg: VaeConfig::sd35(),
            flow: FlowMatchConfig::sd3(),
            stream,
        })
    }

    /// `(context, pooled)` for one prompt: `[1, 333, 4096]` and `[1, 2048]`.
    pub fn conditioning(&self, prompt: &str) -> Result<(Array, Array), PipelineError> {
        let s = &self.stream;
        let ids = self.clip_tokenizer.encode(prompt)?;
        let v: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let ids = Array::from_slice_i32(&v, &[1, v.len()])?;

        // Penultimate for the sequences, projection head for the pooled
        // vectors — different depths of the same forward, so each tower runs
        // once.
        let (final_l, layers_l) =
            clip::text_encoder_layers_with(&ids, &ClipConfig::sd3_l(), &self.clip_l, s)?;
        let (final_g, layers_g) =
            clip::text_encoder_layers_with(&ids, &ClipConfig::sdxl_2(), &self.clip_g, s)?;
        let seq_l = penultimate(&layers_l, s)?;
        let seq_g = penultimate(&layers_g, s)?;

        let pooled_l = clip::project(&clip::pool(&final_l, &ids, s)?, &self.clip_l, s)?;
        let pooled_g = clip::project(&clip::pool(&final_g, &ids, s)?, &self.clip_g, s)?;

        // 768 + 1280 = 2048, zero-padded out to T5's 4096.
        let clip_seq = concat(&[&seq_l, &seq_g], 2, s)?;
        let [b, n, w] = clip_seq.shape()[..] else {
            return Err(msg(format!("mlx: sd3 clip seq {:?}", clip_seq.shape())));
        };
        if w > CONTEXT_WIDTH {
            return Err(msg(format!(
                "mlx: CLIP's {w} features exceed the context width {CONTEXT_WIDTH}"
            )));
        }
        let pad = Array::from_slice_f32(
            &vec![0.0; b * n * (CONTEXT_WIDTH - w)],
            &[b, n, CONTEXT_WIDTH - w],
        )?;
        let clip_seq = concat(&[&clip_seq, &pad], 2, s)?;

        let t5_ids = self.t5_tokenizer.encode(prompt)?;
        let t5v: Vec<i32> = t5_ids.iter().map(|&x| x as i32).collect();
        let t5_ids = Array::from_slice_i32(&t5v, &[1, t5v.len()])?;
        let t5_seq = t5::encode(&t5_ids, &t5::T5Config::xxl(), &self.t5, s)?;

        // CLIP's tokens first, then T5's, along the token axis.
        let context = concat(&[&clip_seq, &t5_seq], 1, s)?;
        let pooled = concat(&[&pooled_l, &pooled_g], 1, s)?;
        Ok((context, pooled))
    }

    /// Prompt to pixels. Returns `(width, height, RGB bytes)`.
    pub fn txt2img(&self, cfg: &Sd3RunConfig) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        if cfg.width % 16 != 0 || cfg.height % 16 != 0 {
            return Err(msg(format!(
                "mlx: {}x{} does not divide into 2x2 patches of 8-pixel latent cells",
                cfg.width, cfg.height
            )));
        }
        let s = &self.stream;
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);
        let p = self.cfg.patch_size;
        // The sigma ladder is resolution-dependent, so this is the run's own
        // token count and not a constant.
        let seq_len = (lh / p) * (lw / p);

        let (cond, cond_pooled) = self.conditioning(&cfg.prompt)?;
        let (uncond, uncond_pooled) = self.conditioning(&cfg.negative_prompt)?;
        // Unconditional row first, as everywhere else here.
        let context = concat(&[&uncond, &cond], 0, s)?;
        let pooled = concat(&[&uncond_pooled, &cond_pooled], 0, s)?;

        let sigmas = flow_sigmas(&self.flow, cfg.steps, seq_len);
        let timesteps = flow_timesteps(&self.flow, &sigmas);

        let mut rng = SeededRng::new(cfg.seed);
        // **NCHW here, not NHWC.** SD 3's latent is patchified rather than
        // convolved, and `pack_latents` reads `[b, c, h, w]`.
        let noise = draw_noise(&mut rng, self.cfg.in_channels, lh, lw)?;
        let mut latent = noise.transpose(&[0, 3, 1, 2], s)?.contiguous(s)?;

        for i in 0..cfg.steps {
            let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
            let doubled = concat(&[&latent, &latent], 0, s)?;
            let t = timesteps[i] as f32;
            let timestep = Array::from_slice_f32(&[t, t], &[2])?;

            let velocity = sd3::forward(
                &doubled,
                &context,
                &pooled,
                &timestep,
                &self.cfg,
                &self.transformer,
                s,
            )?;
            // Guidance on the velocity, which is what this model predicts.
            let uncond_v = velocity.narrow(0, 0, 1, s)?;
            let cond_v = velocity.narrow(0, 1, 1, s)?;
            let guided = cond_v
                .sub(&uncond_v, s)?
                .mul(&Array::scalar_f32(cfg.cfg_scale as f32)?, s)?
                .add(&uncond_v, s)?;

            // `x + v * (sigma_next - sigma)` — no epsilon, no ancestral noise.
            latent = latent.add(
                &guided.mul(&Array::scalar_f32((sigma_next - sigma) as f32)?, s)?,
                s,
            )?;
        }

        // Back to NHWC for the VAE, then unscale and decode.
        let nhwc = latent.transpose(&[0, 2, 3, 1], s)?.contiguous(s)?;
        let unscaled = self.vae_cfg.unscale(&nhwc, s)?;
        let image = vae::decode_with(&unscaled, &self.vae_cfg, &self.vae, s)?;
        let [_, h, w, _] = image.shape()[..] else {
            return Err(msg(format!("mlx: sd3 decode {:?}", image.shape())));
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

/// The penultimate layer's output, which is what SD 3 conditions on.
fn penultimate(layers: &[Array], s: &Stream) -> Result<Array, PipelineError> {
    let idx = layers
        .len()
        .checked_sub(2)
        .ok_or_else(|| msg("mlx: an encoder with fewer than two layers".into()))?;
    Ok(layers[idx].contiguous(s)?)
}
