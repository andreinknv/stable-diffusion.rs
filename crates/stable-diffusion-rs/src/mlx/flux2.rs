//! The FLUX.2 pipeline: a prompt in, an image out.
//!
//! Everything below the transformer is different from Flux.1, and the
//! differences are the parts that fail quietly:
//!
//! **The text encoder is a language model, and three of its layers are used.**
//! FLUX.2 conditions on Qwen3's hidden states at layers **10, 20 and 30**,
//! concatenated per token into a 7680-wide vector. Not the last layer, not one
//! layer — three, in that order. `context_embedder` is `[hidden, 7680]`, which
//! is the only visible sign if the concatenation is wrong.
//!
//! **The latent normalisation is a BatchNorm, not a scalar.** Every other model
//! here multiplies by `scaling_factor` and subtracts `shift_factor`; FLUX.2's
//! VAE carries running statistics over the 128 patchified channels.
//!
//! **There are two 2x2 operations.** The VAE's `patch_size` folds 32 latent
//! channels into 128; the transformer's `patch_size` is 1 and folds nothing.
//! Applying the first twice gives a latent of the right shape and the wrong
//! content.

use std::path::{Path, PathBuf};

use sd_models::clip::TokenizeError;
use sd_models::mlx::llm::{self, LlmConfig};
use sd_models::mlx::quantized::{self, QuantizedWeights};
use sd_models::mlx::{flux2, normalise_legacy_attention, vae, Weights};
use sd_sample::flow::{flow_sigmas, flow_timesteps, FlowMatchConfig};
use sd_tensor::mlx::{load_safetensors, Array, Device, Stream};
use sd_tensor::rng::SeededRng;

use super::{draw_noise, msg};
use crate::pipeline::PipelineError;

/// **Layers 10, 20 and 30 of the text encoder**, concatenated per token.
///
/// `diffusers` calls this `text_encoder_out_layers` and defaults to exactly
/// this. Index 0 is the embedding, so layer 10 is the output of the tenth
/// block.
pub const CONDITIONING_LAYERS: [usize; 3] = [10, 20, 30];

/// How many tokens the prompt is padded or truncated to.
const MAX_SEQUENCE_LENGTH: usize = 512;

/// The system message FLUX.2 was trained with.
///
/// **Not decoration.** The encoder is an instruction-tuned chat model, and the
/// checkpoint learned to read a prompt that arrives inside this conversation.
/// Feeding the bare prompt gives a conditioning the model has never seen: the
/// palette still follows the words, and the composition is soft and painterly.
/// Verbatim from `black-forest-labs/flux2`.
const SYSTEM_MESSAGE: &str = "You are an AI that reasons about image descriptions. You give \
     structured responses focusing on object relationships, object\nattribution and actions \
     without speculation.";

/// Qwen3's ChatML wrapper, with `add_generation_prompt=False` — so no trailing
/// assistant turn, because nothing is being generated from this model.
fn chat_format(prompt: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
    )
}

/// Where FLUX.2's pieces live under a model directory.
#[derive(Debug, Clone)]
pub struct Flux2Paths {
    pub transformer: PathBuf,
    pub vae: PathBuf,
    /// The Qwen3 encoder's shards.
    pub text_encoder: Vec<PathBuf>,
    pub tokenizer: PathBuf,
}

impl Flux2Paths {
    pub fn in_dir(root: &Path) -> Self {
        Self {
            transformer: root.join("transformer/diffusion_pytorch_model.safetensors"),
            vae: root.join("vae/diffusion_pytorch_model.safetensors"),
            text_encoder: shards(&root.join("text_encoder"), "model"),
            tokenizer: root.join("tokenizer/tokenizer.json"),
        }
    }
}

/// Every shard of a sharded checkpoint, or the single file if it is not.
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
    found.sort();
    found
}

/// One run's settings.
#[derive(Debug, Clone)]
pub struct Flux2RunConfig {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// The distilled guidance scale. **Refused by a checkpoint without a
    /// guidance embedder**, which the klein releases are.
    pub guidance: f64,
    pub seed: u64,
}

impl Default for Flux2RunConfig {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: 4,
            guidance: 4.0,
            seed: 0,
        }
    }
}

/// A loaded FLUX.2 pipeline.
pub struct Flux2Pipeline {
    tokenizer: tokenizers::Tokenizer,
    /// Qwen's `<|endoftext|>`, read from the vocabulary rather than hardcoded.
    pad_token: u32,
    /// **Quantised at rest.** klein-4B's transformer is 7.75 GB in bf16 and
    /// the encoder another 8; dense f32 is well past this machine.
    transformer: QuantizedWeights,
    text_encoder: QuantizedWeights,
    vae: Weights,
    cfg: flux2::Flux2Config,
    llm_cfg: LlmConfig,
    vae_cfg: vae::VaeConfig,
    flow: FlowMatchConfig,
    /// `batch_norm_eps` from the VAE's config.
    bn_eps: f32,
    stream: Stream,
}

impl Flux2Pipeline {
    /// Load from a `diffusers` FLUX.2 directory, quantised at rest.
    pub fn load(root: &Path, cfg: flux2::Flux2Config) -> Result<Self, PipelineError> {
        Self::load_quantized(root, cfg, quantized::DEFAULT_BITS, Device::default())
    }

    /// [`Self::load`] at an explicit bit width, on a named device.
    pub fn load_quantized(
        root: &Path,
        cfg: flux2::Flux2Config,
        bits: usize,
        device: Device,
    ) -> Result<Self, PipelineError> {
        let paths = Flux2Paths::in_dir(root);
        for p in [&paths.transformer, &paths.vae, &paths.tokenizer] {
            if !p.exists() {
                return Err(PipelineError::MissingFile(p.clone()));
            }
        }
        if paths.text_encoder.is_empty() {
            return Err(PipelineError::MissingFile(root.join("text_encoder")));
        }
        let stream = Stream::for_device(device);

        // Qwen3's tokenizer is an ordinary HuggingFace one. It is not CLIP's,
        // so `ClipTokenizer` and its vendored vocabulary do not apply.
        let tokenizer = tokenizers::Tokenizer::from_file(&paths.tokenizer)
            .map_err(|e| PipelineError::Tokenize(TokenizeError::Load(e.to_string())))?;

        let mut dense_te = Weights::new();
        for shard in &paths.text_encoder {
            dense_te.extend(load_safetensors(shard)?);
        }
        let text_encoder = quantized::from_dense(&dense_te, bits, &stream)?;
        drop(dense_te);

        let dense_tr = load_safetensors(&paths.transformer)?;
        let transformer = quantized::from_dense(&dense_tr, bits, &stream)?;
        drop(dense_tr);

        let mut vae_w = load_safetensors(&paths.vae)?;
        normalise_legacy_attention(&mut vae_w);

        let pad_token = tokenizer.token_to_id("<|endoftext|>").ok_or_else(|| {
            msg("mlx: this tokenizer has no <|endoftext|>; it is not Qwen's".into())
        })?;

        Ok(Self {
            tokenizer,
            pad_token,
            transformer,
            text_encoder,
            vae: vae_w,
            cfg,
            llm_cfg: LlmConfig::qwen3_4b(),
            // 32 latent channels, and `use_quant_conv` because FLUX.2's VAE
            // config sets it — unlike Flux.1's, which does not.
            vae_cfg: vae::VaeConfig {
                latent_channels: 32,
                use_quant_conv: true,
                // Unused: FLUX.2 normalises with the BatchNorm below instead.
                scaling_factor: 1.0,
                shift_factor: 0.0,
            },
            flow: FlowMatchConfig::flux(),
            bn_eps: 1e-4,
            stream,
        })
    }

    /// The prompt's conditioning, `[1, seq, 3 * hidden]`.
    ///
    /// **Three layers concatenated per token**, in the order
    /// [`CONDITIONING_LAYERS`] gives. Taking the last hidden state instead
    /// produces a third of the width the `context_embedder` expects, which is
    /// the one failure here that is loud.
    pub fn conditioning(&self, prompt: &str) -> Result<Array, PipelineError> {
        let s = &self.stream;
        // The template already carries the special tokens, so the tokenizer
        // must not add its own on top.
        let encoded = self
            .tokenizer
            .encode(chat_format(prompt), false)
            .map_err(|e| PipelineError::Tokenize(TokenizeError::Load(e.to_string())))?;
        let mut ids: Vec<i32> = encoded.get_ids().iter().map(|&i| i as i32).collect();
        // Padded to a fixed length so the token count does not change the
        // sigma ladder between prompts. **With the pad token**, not by
        // repeating the last real one — which would read as the prompt saying
        // its final word four hundred more times.
        ids.truncate(MAX_SEQUENCE_LENGTH);
        ids.resize(MAX_SEQUENCE_LENGTH, self.pad_token as i32);

        let token_ids = Array::from_slice_i32(&ids, &[1, ids.len()])?;
        let states = llm::hidden_states(&token_ids, &self.llm_cfg, &self.text_encoder, s)?;

        let mut picked = Vec::with_capacity(CONDITIONING_LAYERS.len());
        for &layer in &CONDITIONING_LAYERS {
            let state = states.get(layer).ok_or_else(|| {
                msg(format!(
                    "mlx: the text encoder produced {} hidden states, so layer {layer} does not \
                     exist — this is not the {}-layer encoder FLUX.2 expects",
                    states.len(),
                    self.llm_cfg.layers
                ))
            })?;
            picked.push(state.contiguous(s)?);
        }
        // Concatenated on the **feature** axis, so each token carries all
        // three layers. On the token axis instead it would be three times as
        // long and a third as wide — a shape the embedder rejects, which is
        // the one merciful part of this.
        let refs: Vec<&Array> = picked.iter().collect();
        Ok(sd_tensor::mlx::concat(&refs, 2, s)?)
    }

    /// Prompt to pixels. Returns `(width, height, RGB bytes)`.
    pub fn txt2img(&self, cfg: &Flux2RunConfig) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        let s = &self.stream;
        // 16, not 8: the VAE downsamples by 8 and the latent patchifier halves
        // again, so an odd number of latent rows cannot fold into 2x2 cells.
        for (what, v) in [("width", cfg.width), ("height", cfg.height)] {
            if v % 16 != 0 {
                return Err(msg(format!(
                    "mlx: {what} {v} must be a multiple of 16 for FLUX.2's 2x2 latent cells"
                )));
            }
        }
        if cfg.steps == 0 {
            return Err(PipelineError::NoSteps);
        }

        let (lat_h, lat_w) = (cfg.height / 8, cfg.width / 8);
        let (patch_h, patch_w) = (lat_h / 2, lat_w / 2);
        let img_len = patch_h * patch_w;

        let txt = self.conditioning(&cfg.prompt)?;
        let txt_len = txt.shape()[1];

        let mut rng = SeededRng::new(cfg.seed);
        // 128 channels: the patchified width the transformer takes, so the
        // noise is drawn in the space the sampler works in.
        let noise = draw_noise(&mut rng, 128, patch_h, patch_w)?;
        let nchw = noise.transpose(&[0, 3, 1, 2], s)?.contiguous(s)?;
        let mut xs = flux2::pack_latents(&nchw, s)?;

        let sigmas = flow_sigmas(&self.flow, cfg.steps, img_len);
        let timesteps = flow_timesteps(&self.flow, &sigmas);
        let img_ids = flux2::image_ids(patch_h, patch_w);
        let txt_ids = flux2::text_ids(txt_len);

        let guidance = self
            .cfg
            .guidance_embed
            .then(|| Array::from_slice_f32(&[(cfg.guidance * 1000.0) as f32], &[1]))
            .transpose()?;

        for (i, &t) in timesteps.iter().enumerate() {
            // FLUX.2 scales both the timestep and the guidance by 1000 before
            // embedding; `flow_timesteps` already returns the model's units.
            let timestep = Array::from_slice_f32(&[t as f32], &[1])?;
            let velocity = flux2::forward(
                &xs,
                &img_ids,
                &txt,
                &txt_ids,
                &timestep,
                guidance.as_ref(),
                &self.cfg,
                &self.transformer,
                s,
            )?;
            xs = xs.add(
                &velocity.mul(&Array::scalar_f32((sigmas[i + 1] - sigmas[i]) as f32)?, s)?,
                s,
            )?;
        }

        // Unpack, denormalise with the VAE's running statistics, unfold the
        // 2x2 cells, then decode. Each of those has to happen, in that order.
        let packed = flux2::unpack_latents(&xs, patch_h, patch_w, s)?;
        let denorm = flux2::normalize_latents(&packed, &self.vae, self.bn_eps, false, s)?;
        let latent = flux2::unpatchify_latents(&denorm, s)?;

        let nhwc = latent.transpose(&[0, 2, 3, 1], s)?.contiguous(s)?;
        let image = vae::decode_with(&nhwc, &self.vae_cfg, &self.vae, s)?;
        let [_, h, w, _] = image.shape()[..] else {
            return Err(msg(format!("mlx: flux2 decode {:?}", image.shape())));
        };
        let bytes = image
            .to_vec_f32(s)?
            .iter()
            .map(|&v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        Ok((w, h, bytes))
    }

    /// What this pipeline holds resident, in bytes.
    pub fn resident_bytes(&self) -> usize {
        self.transformer.resident_bytes()
            + self.text_encoder.resident_bytes()
            // The VAE stays dense; f32 is four bytes an element.
            + self.vae.values().map(|v| v.elem_count() * 4).sum::<usize>()
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}
