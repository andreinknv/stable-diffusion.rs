//! The SD 3 / SD 3.5 text-to-image pipeline.
//!
//! Three text encoders, which is one more than SDXL and one more than Flux
//! uses meaningfully, and they are combined in a way worth spelling out:
//!
//! - **CLIP-L and CLIP-G** each contribute a *sequence* and a *pooled* vector.
//!   The two sequences are concatenated on the feature axis (768 + 1280 =
//!   2048) and then **zero-padded to T5's 4096**, so most of each CLIP token
//!   is empty. That looks like a bug and is not.
//! - **T5** contributes only a sequence, which is concatenated onto the CLIP
//!   one along the *token* axis.
//! - The two pooled vectors concatenate to 2048 and become the conditioning
//!   vector alongside the timestep.
//!
//! The CLIP sequences come from the **penultimate** layer, not the last —
//! the same "clip skip" convention SDXL uses — while the pooled vectors come
//! from the projection head. Using the final layer produces a slightly worse
//! image rather than an error.
//!
//! Unlike Flux, SD 3 **does** use classifier-free guidance, so each step runs
//! the transformer twice.

use std::path::{Path, PathBuf};

use sd_models::clip::{ClipTextConfig, ClipTextEncoder, ClipTokenizer};
use sd_models::sd3::{Sd3Config, Sd3Transformer};
use sd_models::t5::{T5Config, T5EncoderModel, T5Tokenizer};
use sd_models::vae::{AutoencoderKlDecoder, VaeConfig};
use sd_sample::flow::{flow_euler_step, flow_sigmas, flow_timesteps, FlowMatchConfig};
use sd_tensor::{DType, Device, Tensor, D};

use super::PipelineError;

/// F32 throughout. As with Flux, F16 is not a safe halving here: T5's
/// activations exceed its range, and SD 3.5's blocks reach ~97,000.
const DTYPE: DType = DType::F32;

/// T5 tokens SD 3 conditions on. Shorter than Flux's 512.
const T5_LENGTH: usize = 154;

/// Where each model lives.
#[derive(Debug, Clone)]
pub struct Sd3Paths {
    /// Single-file transformer, `.safetensors` or `.gguf`.
    pub transformer: PathBuf,
    pub clip_l: PathBuf,
    pub clip_g: PathBuf,
    pub clip_tokenizer: PathBuf,
    pub t5_gguf: PathBuf,
    pub t5_tokenizer: PathBuf,
    pub vae: PathBuf,
}

/// Fixture layout used by this repository.
///
/// Prefers the quantised transformer when one is present, exactly as Flux's
/// `paths_in` does, and for a reason that outweighs the small quantisation
/// error: the dense checkpoint is 10.2 GB at f32 against 1.79 GB at Q4_K_M,
/// and on a 36 GB Mac that decides whether SD 3.5 runs on the GPU at all.
/// Loading the dense form leaves about 1.1 GB free and the run then fails in
/// the first denoise step.
pub fn sd3_paths_in(dir: &Path) -> Sd3Paths {
    let quantised = dir.join("sd35-medium-q4_k_m.gguf");
    let transformer = if quantised.exists() {
        quantised
    } else {
        dir.join("sd35-medium.safetensors")
    };
    Sd3Paths {
        transformer,
        clip_l: dir.join("clip-l.safetensors"),
        clip_g: dir.join("clip-g.safetensors"),
        clip_tokenizer: dir.join("../flux/clip-tokenizer.json"),
        t5_gguf: dir.join("../flux/t5-xxl-q4_k_s.gguf"),
        t5_tokenizer: dir.join("../flux/t5-tokenizer.json"),
        vae: dir.join("sd35-vae.safetensors"),
    }
}

/// An SD 3 run.
#[derive(Debug, Clone)]
pub struct Sd3RunConfig {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// A genuine CFG weight here, unlike Flux's distilled guidance.
    pub cfg_scale: f64,
    pub seed: u64,
}

impl Default for Sd3RunConfig {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 512,
            height: 512,
            steps: 28,
            cfg_scale: 4.5,
            seed: 0,
        }
    }
}

pub struct Sd3Pipeline {
    clip_tokenizer: ClipTokenizer,
    clip_l: ClipTextEncoder,
    clip_g: ClipTextEncoder,
    t5_tokenizer: T5Tokenizer,
    t5: T5EncoderModel,
    transformer: Sd3Transformer,
    vae: AutoencoderKlDecoder,
    flow: FlowMatchConfig,
    device: Device,
}

impl Sd3Pipeline {
    pub fn load(paths: &Sd3Paths, cfg: &Sd3Config, device: &Device) -> Result<Self, PipelineError> {
        for p in [
            &paths.transformer,
            &paths.clip_l,
            &paths.clip_g,
            &paths.t5_gguf,
            &paths.vae,
        ] {
            if !p.exists() {
                return Err(PipelineError::MissingFile(p.clone()));
            }
        }

        let clip_tokenizer = ClipTokenizer::from_file(&paths.clip_tokenizer)?;
        let t5_tokenizer = T5Tokenizer::from_file(&paths.t5_tokenizer, T5_LENGTH)?;

        let vb = sd_loader::safetensors_var_builder(&[&paths.clip_l], DTYPE, device)?;
        let clip_l = ClipTextEncoder::new(&ClipTextConfig::sd3_l(), vb)?;
        let vb = sd_loader::safetensors_var_builder(&[&paths.clip_g], DTYPE, device)?;
        let clip_g = ClipTextEncoder::new(&ClipTextConfig::sdxl_2(), vb)?;

        let weights = sd_loader::t5_qtensors_from_gguf(&paths.t5_gguf, device)?;
        let t5 = T5EncoderModel::from_quantized(&T5Config::xxl(), &weights)?;

        let transformer = if paths.transformer.extension().is_some_and(|e| e == "gguf") {
            let w = sd_loader::sd3_qtensors_from_gguf(&paths.transformer, device)?;
            Sd3Transformer::from_quantized(cfg, &w)?
        } else {
            let vb = sd_loader::safetensors_var_builder(&[&paths.transformer], DTYPE, device)?;
            // The single-file checkpoint nests the transformer; the VAE lives
            // beside it under `first_stage_model.`.
            Sd3Transformer::new(cfg, vb.pp("model").pp("diffusion_model"))?
        };

        let vb = sd_loader::safetensors_var_builder(&[&paths.vae], DTYPE, device)?;
        let vae = AutoencoderKlDecoder::new(&VaeConfig::sd35(), vb)?;

        Ok(Self {
            clip_tokenizer,
            clip_l,
            clip_g,
            t5_tokenizer,
            t5,
            transformer,
            vae,
            flow: FlowMatchConfig::sd3(),
            device: device.clone(),
        })
    }

    /// Build `(context, pooled)` for one prompt.
    fn conditioning(&self, prompt: &str) -> Result<(Tensor, Tensor), PipelineError> {
        let ids = self.clip_tokenizer.encode(prompt)?;
        let ids = Tensor::from_vec(ids, (1, self.clip_tokenizer.max_length()), &self.device)?;

        // Penultimate layer for the sequences, projection head for the pooled
        // vectors — the two come from different depths of the same forward.
        let (_, layers_l) = self.clip_l.forward_with_layers(&ids)?;
        let (_, layers_g) = self.clip_g.forward_with_layers(&ids)?;
        let seq_l = penultimate(&layers_l)?;
        let seq_g = penultimate(&layers_g)?;

        let pooled_l = self
            .clip_l
            .pooled(&ids)?
            .ok_or_else(|| PipelineError::MissingFile(PathBuf::from("CLIP-L text_projection")))?;
        let pooled_g = self
            .clip_g
            .pooled(&ids)?
            .ok_or_else(|| PipelineError::MissingFile(PathBuf::from("CLIP-G text_projection")))?;

        // 768 + 1280 = 2048, then zero-padded out to T5's 4096. The padding is
        // deliberate: the transformer sees one context width, and CLIP simply
        // occupies the first half of each of its tokens.
        let clip_seq = Tensor::cat(&[&seq_l, &seq_g], D::Minus1)?;
        let (b, n, w) = clip_seq.dims3()?;
        let pad = Tensor::zeros((b, n, 4096 - w), clip_seq.dtype(), &self.device)?;
        let clip_seq = Tensor::cat(&[&clip_seq, &pad], D::Minus1)?;

        let t5_ids = self.t5_tokenizer.encode(prompt)?;
        let t5_ids = Tensor::from_vec(t5_ids, (1, T5_LENGTH), &self.device)?;
        let t5_seq = self.t5.forward(&t5_ids)?.to_dtype(DTYPE)?;

        // CLIP tokens first, then T5's, along the token axis.
        let context = Tensor::cat(&[&clip_seq, &t5_seq], 1)?;
        let pooled = Tensor::cat(&[&pooled_l, &pooled_g], D::Minus1)?;
        Ok((context, pooled))
    }

    pub fn run(&self, cfg: &Sd3RunConfig) -> Result<Tensor, PipelineError> {
        self.run_with_progress(cfg, |_, _| {})
    }

    pub fn run_with_progress(
        &self,
        cfg: &Sd3RunConfig,
        progress: impl FnMut(usize, usize),
    ) -> Result<Tensor, PipelineError> {
        let latents = self.denoise(cfg, progress)?;
        Ok(self.vae.decode_tiled(&latents)?)
    }

    /// [`Self::run_with_progress`], but the transformer and text encoders are
    /// dropped before the VAE decode.
    ///
    /// Consuming rather than borrowing, because that is the honest signature:
    /// the pipeline cannot generate again afterwards. Worth it for a one-shot
    /// run, which is what the CLI and the examples do.
    ///
    /// The decode is where this pipeline peaks. It allocates a convolution
    /// im2col — 2.42 GB for a 512px image in one tile — while the transformer
    /// that produced the latent is still resident and will never be used
    /// again. For SD 3.5 that is 10 GB held for nothing, and on Metal it was
    /// the difference between rendering at 512 and dying after all 20 denoise
    /// steps had run.
    pub fn run_releasing(
        self,
        cfg: &Sd3RunConfig,
        progress: impl FnMut(usize, usize),
    ) -> Result<Tensor, PipelineError> {
        let latents = self.denoise(cfg, progress)?;
        // Keep the VAE and the device, drop everything else. `latents` is
        // already computed and holds no borrow of `self`.
        let Self { vae, device, .. } = self;
        // Dropping is not enough on Metal: candle pools its buffers and hands
        // them back only when something synchronises, because that is where
        // it runs `drop_unused_buffers`. Without this the drop above frees
        // nothing at all — measured, not assumed.
        device.synchronize()?;
        Ok(vae.decode_tiled(&latents)?)
    }

    /// The sampling loop, returning the latent before decoding.
    fn denoise(
        &self,
        cfg: &Sd3RunConfig,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Tensor, PipelineError> {
        if cfg.steps == 0 {
            return Err(PipelineError::NoSteps);
        }
        for (what, v) in [("width", cfg.width), ("height", cfg.height)] {
            // 8 for the VAE, then 2 for the patch grid.
            if v % 16 != 0 {
                return Err(PipelineError::NotMultipleOfEight(what, v));
            }
        }

        let (context, pooled) = self.conditioning(&cfg.prompt)?;
        let (neg_context, neg_pooled) = self.conditioning(&cfg.negative_prompt)?;

        let (lh, lw) = (cfg.height / 8, cfg.width / 8);
        let mut rng = sd_tensor::rng::SeededRng::new(cfg.seed);
        let mut latents =
            Tensor::from_vec(rng.normals(16 * lh * lw), (1, 16, lh, lw), &self.device)?;

        // Static shift, so the schedule does not depend on resolution the way
        // Flux's does; the token count is passed but unused.
        let sigmas = flow_sigmas(&self.flow, cfg.steps, (lh / 2) * (lw / 2));
        let timesteps = flow_timesteps(&self.flow, &sigmas);

        for (i, &t) in timesteps.iter().enumerate() {
            // SD 3 takes the timestep already in the training range, unlike
            // Flux which takes a sigma in [0, 1].
            let t = Tensor::from_vec(vec![t as f32], 1, &self.device)?;

            let cond = self.transformer.forward(&latents, &context, &pooled, &t)?;
            let uncond = self
                .transformer
                .forward(&latents, &neg_context, &neg_pooled, &t)?;
            // Real classifier-free guidance, two passes per step.
            let velocity = (&uncond + ((cond - &uncond)? * cfg.cfg_scale)?)?;

            latents = flow_euler_step(&latents, &velocity, sigmas[i], sigmas[i + 1])?;
            progress(i + 1, cfg.steps);
        }

        Ok(latents)
    }
}

/// The second-to-last hidden state.
///
/// `forward_with_layers` returns one entry per encoder layer, so the
/// penultimate *layer output* is the second from the end.
fn penultimate(layers: &[Tensor]) -> Result<Tensor, PipelineError> {
    let n = layers.len();
    if n < 2 {
        return Err(PipelineError::NoSteps);
    }
    Ok(layers[n - 2].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_context_is_padded_to_the_t5_width() {
        // 768 + 1280 = 2048, and the transformer expects 4096. Half of every
        // CLIP token is zero by design.
        assert_eq!(768 + 1280, 2048);
        assert_eq!(ClipTextConfig::sd3_l().projection_dim, Some(768));
        assert_eq!(ClipTextConfig::sdxl_2().projection_dim, Some(1280));
        assert_eq!(Sd3Config::medium_35().pooled_dim, 768 + 1280);
        assert_eq!(Sd3Config::medium_35().context_dim, 4096);
    }

    #[test]
    fn the_schedule_is_resolution_independent() {
        // The opposite of Flux, whose whole point is that it is not.
        let flow = FlowMatchConfig::sd3();
        assert!(!flow.use_dynamic_shifting);
        assert_eq!(flow_sigmas(&flow, 10, 256), flow_sigmas(&flow, 10, 4096));
    }
}
