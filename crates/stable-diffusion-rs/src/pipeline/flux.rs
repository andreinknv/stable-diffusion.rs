//! The Flux text-to-image pipeline.
//!
//! Four models rather than SD's three: CLIP-L supplies a single *pooled*
//! vector (not a sequence — Flux uses no cross-attention on it), T5-XXL
//! supplies the token sequence that actually carries the prompt, the MMDiT
//! transformer predicts flow velocity, and the 16-channel VAE decodes.
//!
//! Two things differ from every other pipeline here:
//!
//! - **No classifier-free guidance.** Flux dev and flux-mini are *distilled*
//!   on a guidance scale, which is fed in as a conditioning input. That means
//!   one forward pass per step rather than two, and `guidance` is not a CFG
//!   weight even though it occupies the same place in the interface.
//! - **The schedule depends on resolution.** `flow_sigmas` warps by token
//!   count, so a 512x512 run and a 1024x1024 run do not share sigmas.
//!
//! **Precision is not a free choice here, and F16 does not work.** Both big
//! models carry activations far outside F16's range, which tops out at 65504:
//! T5 peaks near 200,000 partway up its stack, and the transformer produces
//! NaN velocities. Both were measured rather than predicted.
//!
//! So T5's weights are held *quantised* and expanded per matmul, which keeps
//! every activation in F32 and costs 2.7 GB instead of 18.8, and the
//! transformer runs at F32 for 12.8 GB. About 15.5 GB in total, which fits on
//! a 36 GB machine. bfloat16 has F32's exponent range and would suit both,
//! but candle's CPU backend has no bf16 matmul.

use std::path::{Path, PathBuf};

use sd_models::clip::{ClipTextConfig, ClipTextEncoder, ClipTokenizer};
use sd_models::flux::{pack_latents, rope, unpack_latents, FluxConfig, FluxTransformer};
use sd_models::t5::{T5Config, T5EncoderModel, T5Tokenizer, FLUX_MAX_LENGTH};
use sd_models::vae::{AutoencoderKlDecoder, VaeConfig};
use sd_sample::flow::{flow_euler_step, flow_sigmas, flow_timesteps, FlowMatchConfig};
use sd_tensor::{DType, Device, Tensor};

use super::placement::{self, file_bytes, Placement, StageBytes};
use super::PipelineError;

/// F32. Not a default — F16 produces NaN in both big models. See the module
/// note.
const MODEL_DTYPE: DType = DType::F32;
/// The VAE stays F32: it is small, and its config sets `force_upcast`.
const VAE_DTYPE: DType = DType::F32;

/// Where each of the four models lives.
///
/// Explicit paths rather than a directory, because a working Flux setup is
/// assembled from several sources — the transformer from one repository, T5
/// from a GGUF, CLIP from a third — and pretending otherwise would mean
/// inventing a layout nobody publishes.
#[derive(Debug, Clone)]
pub struct FluxPaths {
    /// black-forest-labs-layout transformer weights.
    pub transformer: PathBuf,
    /// T5-XXL encoder, llama.cpp-layout GGUF.
    pub t5_gguf: PathBuf,
    pub t5_tokenizer: PathBuf,
    /// CLIP-L weights — the same encoder SD 1.5 uses.
    pub clip: PathBuf,
    pub clip_tokenizer: PathBuf,
    pub vae: PathBuf,
}

/// A Flux run.
#[derive(Debug, Clone)]
pub struct FluxConfigRun {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// Distilled guidance, *not* a CFG weight. 3.5 is what Flux dev is
    /// distilled around.
    pub guidance: f64,
    pub seed: u64,
}

impl Default for FluxConfigRun {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 512,
            height: 512,
            steps: 20,
            guidance: 3.5,
            seed: 0,
        }
    }
}

/// Text-to-image with Flux.
pub struct FluxPipeline {
    clip_tokenizer: ClipTokenizer,
    clip: ClipTextEncoder,
    t5_tokenizer: T5Tokenizer,
    t5: T5EncoderModel,
    transformer: FluxTransformer,
    vae: AutoencoderKlDecoder,
    flow: FlowMatchConfig,
    placement: Placement,
}

impl FluxPipeline {
    /// Load with every stage on one device.
    pub fn load(
        paths: &FluxPaths,
        cfg: &FluxConfig,
        device: &Device,
    ) -> Result<Self, PipelineError> {
        Self::load_with_placement(paths, cfg, &Placement::on(device))
    }

    /// What each stage costs resident, for [`Placement::auto`].
    pub fn stage_bytes(paths: &FluxPaths) -> Result<StageBytes, PipelineError> {
        let quantised = paths.transformer.extension().is_some_and(|e| e == "gguf");
        Ok(StageBytes {
            text_encoders: placement::resident_bytes(&[&paths.clip], MODEL_DTYPE)?
                .saturating_add(file_bytes(&paths.t5_gguf)),
            diffusion: if quantised {
                file_bytes(&paths.transformer)
            } else {
                placement::resident_bytes(&[&paths.transformer], MODEL_DTYPE)?
            },
            vae: placement::resident_bytes(&[&paths.vae], VAE_DTYPE)?,
        })
    }

    /// Load with each stage on the device [`Placement`] assigns it.
    pub fn load_with_placement(
        paths: &FluxPaths,
        cfg: &FluxConfig,
        placement: &Placement,
    ) -> Result<Self, PipelineError> {
        let device = placement.compute();
        let text_device = placement.text_encoders();
        let vae_device = placement.vae();
        for p in [&paths.transformer, &paths.t5_gguf, &paths.clip, &paths.vae] {
            if !p.exists() {
                return Err(PipelineError::MissingFile(p.clone()));
            }
        }
        for p in [&paths.t5_tokenizer, &paths.clip_tokenizer] {
            if !p.exists() {
                return Err(PipelineError::MissingTokenizerJson(p.clone()));
            }
        }

        let clip_tokenizer = ClipTokenizer::from_file(&paths.clip_tokenizer)?;
        let t5_tokenizer = T5Tokenizer::from_file(&paths.t5_tokenizer, FLUX_MAX_LENGTH)?;

        let vb = sd_loader::safetensors_var_builder(&[&paths.clip], MODEL_DTYPE, text_device)?;
        let clip = ClipTextEncoder::new(&ClipTextConfig::sd15(), vb)?;

        // T5's weights stay quantised: 2.7 GB against 18.8 at F32. That is
        // not only a saving. Its activations peak near 200,000 partway up the
        // stack, and f16 tops out at 65504, so a dequantise-to-f16 load
        // silently becomes NaN around block 10 — measured, not assumed.
        // Holding the blocks and expanding per matmul keeps every activation
        // in f32. bf16 would also work; candle's CPU backend has no bf16
        // matmul.
        let weights = sd_loader::t5_qtensors_from_gguf(&paths.t5_gguf, text_device)?;
        let t5 = T5EncoderModel::from_quantized(&T5Config::xxl(), &weights)?;

        // A GGUF transformer keeps its weights quantised, which is what makes
        // full-size Flux reachable: schnell and dev are 12B parameters, 48 GB
        // at F32 and 4.9 GB held as Q4_K. safetensors are read dense, since
        // that is the only form they come in.
        let transformer = if paths.transformer.extension().is_some_and(|e| e == "gguf") {
            let weights = sd_loader::flux_qtensors_from_gguf(&paths.transformer, device)?;
            FluxTransformer::from_quantized(cfg, &weights)?
        } else {
            let vb =
                sd_loader::safetensors_var_builder(&[&paths.transformer], MODEL_DTYPE, device)?;
            FluxTransformer::new(cfg, vb)?
        };

        let vb = sd_loader::safetensors_var_builder(&[&paths.vae], VAE_DTYPE, vae_device)?;
        let vae = AutoencoderKlDecoder::new(&VaeConfig::flux(), vb)?;

        Ok(Self {
            clip_tokenizer,
            clip,
            t5_tokenizer,
            t5,
            transformer,
            vae,
            flow: FlowMatchConfig::flux(),
            placement: placement.clone(),
        })
    }

    /// Encode the prompt into T5's sequence and CLIP's pooled vector.
    fn conditioning(&self, prompt: &str) -> Result<(Tensor, Tensor), PipelineError> {
        let ids = self.t5_tokenizer.encode(prompt)?;
        let ids = Tensor::from_vec(
            ids,
            (1, self.t5_tokenizer.max_length()),
            self.placement.text_encoders(),
        )?;
        // f32 out of the quantised stack; the transformer runs narrower. The
        // *output* is order 10 and casts safely — it is the intermediates
        // that overflow, and those never leave T5.
        let txt = self.t5.forward(&ids)?.to_dtype(MODEL_DTYPE)?;

        let ids = self.clip_tokenizer.encode(prompt)?;
        let ids = Tensor::from_vec(
            ids,
            (1, self.clip_tokenizer.max_length()),
            self.placement.text_encoders(),
        )?;
        // The *pooled* output only. Flux never sees CLIP's sequence — that
        // job belongs to T5 — so a pipeline that fed the sequence here would
        // be conditioning the model on something it was never trained with.
        //
        // `pooled_hidden`, not `pooled`: Flux wants transformers' raw
        // pooler_output, while `pooled` applies text_projection for SDXL's
        // second encoder. CLIP-L has no projection to apply anyway.
        let pooled = self.clip.pooled_hidden(&ids)?.to_dtype(MODEL_DTYPE)?;

        // Cross to the compute device — the entire cost of a split placement:
        // two tensors, once per prompt.
        let compute = self.placement.compute();
        Ok((
            placement::to(&txt, compute)?,
            placement::to(&pooled, compute)?,
        ))
    }

    pub fn run(&self, cfg: &FluxConfigRun) -> Result<Tensor, PipelineError> {
        self.run_with_progress(cfg, |_, _| {})
    }

    pub fn run_with_progress(
        &self,
        cfg: &FluxConfigRun,
        progress: impl FnMut(usize, usize),
    ) -> Result<Tensor, PipelineError> {
        let latents = self.denoise(cfg, progress)?;
        let latents = placement::to(&latents, self.placement.vae())?;
        Ok(self.vae.decode_tiled(&latents)?)
    }

    /// [`Self::run_with_progress`], but everything the decode does not need is
    /// dropped first. See `Sd3Pipeline::run_releasing` for the reasoning and
    /// for what it is actually worth on Metal.
    pub fn run_releasing(
        self,
        cfg: &FluxConfigRun,
        progress: impl FnMut(usize, usize),
    ) -> Result<Tensor, PipelineError> {
        let latents = self.denoise(cfg, progress)?;
        let latents = placement::to(&latents, self.placement.vae())?;
        let device = self.placement.compute().clone();
        let Self { vae, .. } = self;
        device.synchronize()?;
        Ok(vae.decode_tiled(&latents)?)
    }

    /// The sampling loop, returning the unpacked latent before decoding.
    fn denoise(
        &self,
        cfg: &FluxConfigRun,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Tensor, PipelineError> {
        if cfg.steps == 0 {
            return Err(PipelineError::NoSteps);
        }
        // 16, not 8: the VAE downsamples by 8 and the patchifier halves again,
        // so an odd number of latent rows cannot be packed into 2x2 patches.
        for (what, v) in [("width", cfg.width), ("height", cfg.height)] {
            if v % 16 != 0 {
                return Err(PipelineError::NotMultipleOfEight(what, v));
            }
        }

        let (txt, pooled) = self.conditioning(&cfg.prompt)?;

        let (lat_h, lat_w) = (cfg.height / 8, cfg.width / 8);
        let (patch_h, patch_w) = (lat_h / 2, lat_w / 2);
        let img_len = patch_h * patch_w;

        let mut rng = sd_tensor::rng::SeededRng::new(cfg.seed);
        let noise = rng.normals(16 * lat_h * lat_w);
        let latents = Tensor::from_vec(noise, (1, 16, lat_h, lat_w), self.placement.compute())?
            .to_dtype(MODEL_DTYPE)?;
        let mut xs = pack_latents(&latents)?;

        // The schedule depends on how many tokens the transformer will see.
        let sigmas = flow_sigmas(&self.flow, cfg.steps, img_len);
        let timesteps = flow_timesteps(&self.flow, &sigmas);

        let img_ids = rope::image_ids(1, patch_h, patch_w, self.placement.compute())?;
        let txt_ids = rope::text_ids(1, txt.dim(1)?, self.placement.compute())?;
        // schnell is not distilled on a guidance scale and rejects one; dev
        // and flux-mini require it. Driven by the model rather than by the
        // caller, so a `guidance` setting cannot be silently discarded.
        let guidance = if self.transformer.config().guidance_embed {
            Some(
                Tensor::from_vec(vec![cfg.guidance as f32], 1, self.placement.compute())?
                    .to_dtype(MODEL_DTYPE)?,
            )
        } else {
            None
        };

        for (i, &t) in timesteps.iter().enumerate() {
            // Flux's timestep is the sigma itself, in [0, 1], not an index.
            let t = Tensor::from_vec(vec![(t / 1000.0) as f32], 1, self.placement.compute())?
                .to_dtype(MODEL_DTYPE)?;

            // One pass, not two: guidance is distilled in rather than applied
            // by contrasting a conditional and unconditional prediction.
            let velocity = self.transformer.forward(
                &xs,
                &img_ids,
                &txt,
                &txt_ids,
                &t,
                &pooled,
                guidance.as_ref(),
            )?;

            // The step itself in F32. It is a scaled add over the whole
            // latent and accumulating 20 of them in F16 visibly quantises the
            // trajectory, for a cost of nothing.
            xs = flow_euler_step(
                &xs.to_dtype(DType::F32)?,
                &velocity.to_dtype(DType::F32)?,
                sigmas[i],
                sigmas[i + 1],
            )?
            .to_dtype(MODEL_DTYPE)?;

            progress(i + 1, cfg.steps);
        }

        Ok(unpack_latents(&xs.to_dtype(VAE_DTYPE)?, lat_h, lat_w)?)
    }

    /// Everything the sampling loop consumes, for a controlled comparison
    /// against another implementation.
    ///
    /// Handing a reference pipeline our conditioning *and* our initial noise
    /// removes every difference except the loop itself — otherwise a
    /// mismatch could be the tokenizer, the encoder, or the RNG.
    pub fn sampling_inputs(
        &self,
        cfg: &FluxConfigRun,
    ) -> Result<(Tensor, Tensor, Tensor), PipelineError> {
        let (txt, pooled) = self.conditioning(&cfg.prompt)?;
        let (lat_h, lat_w) = (cfg.height / 8, cfg.width / 8);
        let mut rng = sd_tensor::rng::SeededRng::new(cfg.seed);
        let noise = rng.normals(16 * lat_h * lat_w);
        let latents = Tensor::from_vec(noise, (1, 16, lat_h, lat_w), self.placement.compute())?
            .to_dtype(MODEL_DTYPE)?;
        Ok((txt, pooled, pack_latents(&latents)?))
    }

    /// Like [`Self::run`], but also returns the unpacked latent it decoded.
    ///
    /// For locating an artifact: if it is present in the latent then the VAE
    /// is not the cause, and vice versa. Cheaper than bisecting the stack.
    pub fn run_capturing_latent(
        &self,
        cfg: &FluxConfigRun,
    ) -> Result<(Tensor, Tensor), PipelineError> {
        let latents = self.denoise(cfg, |_, _| {})?;
        let image = self
            .vae
            .decode_tiled(&placement::to(&latents, self.placement.vae())?)?;
        Ok((latents, image))
    }
}

/// Token count the transformer will see for a given image size.
///
/// Exposed because the sigma schedule depends on it, so anything that wants to
/// reproduce a run needs the same number.
pub fn image_token_count(width: usize, height: usize) -> usize {
    (width / 16) * (height / 16)
}

/// Convenience for the common layout: everything under one directory, with the
/// names this project's fixtures use.
pub fn paths_in(dir: &Path) -> FluxPaths {
    // Prefer a full-size quantised checkpoint when one is present, since it is
    // the better model; fall back to flux-mini.
    let schnell = dir.join("flux-schnell-q4_k_s.gguf");
    let transformer = if schnell.exists() {
        schnell
    } else {
        dir.join("flux-mini.safetensors")
    };
    FluxPaths {
        transformer,
        t5_gguf: dir.join("t5-xxl-q4_k_s.gguf"),
        t5_tokenizer: dir.join("t5-tokenizer.json"),
        clip: dir.join("clip-l.safetensors"),
        clip_tokenizer: dir.join("clip-tokenizer.json"),
        vae: dir.join("flux-vae.safetensors"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_count_matches_the_patch_grid() {
        // 1024x1024 is the 4096 tokens the scheduler's max_image_seq_len
        // refers to; 512x512 is a quarter of that.
        assert_eq!(image_token_count(1024, 1024), 4096);
        assert_eq!(image_token_count(512, 512), 1024);
    }

    #[test]
    fn sigmas_differ_with_resolution() {
        let flow = FlowMatchConfig::flux();
        let small = flow_sigmas(&flow, 20, image_token_count(512, 512));
        let large = flow_sigmas(&flow, 20, image_token_count(1024, 1024));
        assert_ne!(
            small, large,
            "the Flux schedule is resolution-dependent; sharing sigmas across \
             sizes is a real bug that produces merely-worse images"
        );
    }
}
