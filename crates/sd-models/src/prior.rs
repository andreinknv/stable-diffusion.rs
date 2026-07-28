//! unCLIP's prior: inventing an image embedding from text.
//!
//! [`crate::unclip`] takes a CLIP image embedding and conditions a UNet on it.
//! That embedding normally comes from a picture. The prior is what produces one
//! from a *prompt* instead, and it is what makes unCLIP a text-to-image model
//! rather than only an image-variation one.
//!
//! It is a diffusion model in its own right, and an unusual one: the thing it
//! denoises is a single 768-vector, not a latent image. Twenty-five steps of
//! DDPM in embedding space, and the result is handed to the image half exactly
//! as if a photograph had produced it.
//!
//! # It is a CLIP encoder layer wearing different names
//!
//! Each of the twenty blocks is, mathematically, [`crate::clip`]'s
//! `ClipEncoderLayer`: pre-norm, self-attention with bias on every projection,
//! an additive mask, a residual, pre-norm, a plain-GELU MLP, a residual. The
//! two differ only in what the tensors are *called* —
//! `self_attn.q_proj` against `attn1.to_q`, `layer_norm1` against `norm1`,
//! `mlp.fc1` against `ff.net.0.proj` — because one was exported by
//! `transformers` and the other by `diffusers`.
//!
//! They are written out separately here rather than unified behind a naming
//! table, which is a deliberate call and not an oversight. A weight name is
//! the one thing in this codebase that must be copied character by character
//! from the checkpoint; putting a layer of indirection between the name and
//! the module that loads it is precisely how a silent mismatch gets in. Forty
//! lines of duplication is the cheaper of the two costs.
//!
//! # The sequence is five things concatenated
//!
//! ```text
//!   [ 77 text tokens | projected text embedding | timestep | latent | prd ]
//! ```
//!
//! 81 positions, which is why `positional_embedding` is 81 wide. The answer is
//! read from the **last** one — the learned `prd` token, which exists to have
//! somewhere to put it. Reading position 79 (where the latent went in) instead
//! returns a well-shaped vector that is not the prediction.
//!
//! # The attention mask is load-bearing here and nowhere else
//!
//! Every other CLIP consumer in this project ignores the tokenizer's attention
//! mask: Stable Diffusion conditions on all 77 positions, padding included.
//! The prior does not — padded positions are masked out, and on top of that
//! the whole sequence is causally masked. Ignoring the mask runs, and produces
//! a different image.

use sd_tensor::nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear};
use sd_tensor::{ops, DType, Device, Module, Result, Tensor, VarBuilder};

use crate::unet::{timestep_embedding, TimestepEmbedding};

/// LayerNorm epsilon throughout the prior. PyTorch's default, and diffusers
/// does not override it.
const NORM_EPS: f64 = 1e-5;

/// The value diffusers masks with. **Not `-inf`**, though both underflow to
/// zero through a softmax: the text mask and the causal mask are *added*, and
/// two finite penalties add to a finite one where two infinities do not.
const MASKED: f64 = -10_000.0;

/// Geometry of the prior.
#[derive(Debug, Clone)]
pub struct PriorConfig {
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub num_layers: usize,
    /// Width of what goes in and comes out: a CLIP image embedding.
    pub embedding_dim: usize,
    /// Text positions the prior attends over — CLIP's 77.
    pub num_embeddings: usize,
    /// The extra sequence positions: the projected embedding, the timestep,
    /// the latent, and `prd`.
    pub additional_embeddings: usize,
}

impl PriorConfig {
    /// `kakaobrain/karlo-v1-alpha`'s prior, which is what
    /// `stable-diffusion-2-1-unclip` ships.
    pub fn karlo() -> Self {
        Self {
            num_attention_heads: 32,
            attention_head_dim: 64,
            num_layers: 20,
            embedding_dim: 768,
            num_embeddings: 77,
            additional_embeddings: 4,
        }
    }

    /// 2048 for Karlo: the transformer is far wider than the embedding it
    /// carries, which is why `proj_in` and `proj_to_clip_embeddings` exist.
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn sequence_length(&self) -> usize {
        self.num_embeddings + self.additional_embeddings
    }
}

/// Self-attention with bias on every projection.
///
/// The bias is the difference from [`crate::unet::Attention`], which loads
/// `to_q`/`to_k`/`to_v` without one — so the two cannot share an
/// implementation without a flag, and a checkpoint loaded through the wrong
/// one fails at load naming the missing tensor.
#[derive(Debug)]
struct PriorAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
    dim_head: usize,
}

impl PriorAttention {
    fn new(dim: usize, heads: usize, dim_head: usize, vb: VarBuilder) -> Result<Self> {
        let inner = heads * dim_head;
        Ok(Self {
            to_q: linear(dim, inner, vb.pp("to_q"))?,
            to_k: linear(dim, inner, vb.pp("to_k"))?,
            to_v: linear(dim, inner, vb.pp("to_v"))?,
            // `to_out.0` — index 0 because diffusers wraps it in a list whose
            // second entry is dropout.
            to_out: linear(inner, dim, vb.pp("to_out").pp("0"))?,
            heads,
            dim_head,
        })
    }

    fn split_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        xs.reshape((b, s, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// `mask` is additive, `[b, 1, seq, seq]`, and broadcasts over heads.
    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;
        let q = self.split_heads(&self.to_q.forward(xs)?)?;
        let k = self.split_heads(&self.to_k.forward(xs)?)?;
        let v = self.split_heads(&self.to_v.forward(xs)?)?;
        let out = ops::scaled_dot_product_attention_masked(&q, &k, &v, mask)?;
        let out =
            out.transpose(1, 2)?
                .contiguous()?
                .reshape((b, seq, self.heads * self.dim_head))?;
        self.to_out.forward(&out)
    }
}

/// Plain-GELU feed-forward.
///
/// **Not GEGLU**, which is what [`crate::unet::FeedForward`] implements and
/// what every Stable Diffusion transformer uses. GEGLU's projection emits
/// twice the inner width and splits it into a value and a gate; this one emits
/// the inner width and applies GELU to all of it. The shapes differ — 8192
/// against 16384 — so the wrong choice fails to load rather than running.
#[derive(Debug)]
struct PriorFeedForward {
    proj: Linear,
    out: Linear,
}

impl PriorFeedForward {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = dim * 4;
        Ok(Self {
            proj: linear(dim, inner, vb.pp("net").pp("0").pp("proj"))?,
            // `net.1` is dropout and holds nothing, hence the jump.
            out: linear(inner, dim, vb.pp("net").pp("2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // The erf gelu, matching `nn.GELU()`'s default — not the tanh
        // approximation.
        self.out.forward(&ops::gelu(&self.proj.forward(xs)?)?)
    }
}

/// Pre-norm self-attention, pre-norm feed-forward, residuals on both.
///
/// **`norm1` and `norm3`, with no `norm2`.** The block is diffusers'
/// `BasicTransformerBlock` built without cross-attention, and that leaves a
/// hole in the numbering rather than closing it: `norm2` and `attn2` simply do
/// not exist in the checkpoint. Renumbering `norm3` to `norm2` would be the
/// tidy thing and would fail to load.
#[derive(Debug)]
struct PriorBlock {
    norm1: LayerNorm,
    attn1: PriorAttention,
    norm3: LayerNorm,
    ff: PriorFeedForward,
}

impl PriorBlock {
    fn new(dim: usize, heads: usize, dim_head: usize, vb: VarBuilder) -> Result<Self> {
        let norm_cfg = LayerNormConfig {
            eps: NORM_EPS,
            ..Default::default()
        };
        Ok(Self {
            norm1: layer_norm(dim, norm_cfg, vb.pp("norm1"))?,
            attn1: PriorAttention::new(dim, heads, dim_head, vb.pp("attn1"))?,
            norm3: layer_norm(dim, norm_cfg, vb.pp("norm3"))?,
            ff: PriorFeedForward::new(dim, vb.pp("ff"))?,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let xs = (xs + self.attn1.forward(&self.norm1.forward(xs)?, mask)?)?;
        &xs + self.ff.forward(&self.norm3.forward(&xs)?)?
    }
}

/// The prior.
#[derive(Debug)]
pub struct PriorTransformer {
    time_embedding: TimestepEmbedding,
    proj_in: Linear,
    embedding_proj: Linear,
    encoder_hidden_states_proj: Linear,
    positional_embedding: Tensor,
    prd_embedding: Tensor,
    blocks: Vec<PriorBlock>,
    norm_out: LayerNorm,
    proj_to_clip_embeddings: Linear,
    /// Statistics the sampled embedding is un-whitened by at the very end.
    clip_mean: Tensor,
    clip_std: Tensor,
    /// `[1, 1, seq, seq]`, additive, zero on and below the diagonal.
    causal_mask: Tensor,
    cfg: PriorConfig,
    dtype: DType,
}

impl PriorTransformer {
    pub fn new(cfg: &PriorConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.inner_dim();
        let seq = cfg.sequence_length();
        let device = vb.device();

        let vb_blocks = vb.pp("transformer_blocks");
        let blocks = (0..cfg.num_layers)
            .map(|i| {
                PriorBlock::new(
                    inner,
                    cfg.num_attention_heads,
                    cfg.attention_head_dim,
                    vb_blocks.pp(i.to_string()),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            // Both widths are `inner`: diffusers passes `out_dim=inner_dim`,
            // so this MLP does not widen the way the UNet's does.
            time_embedding: TimestepEmbedding::new(inner, inner, vb.pp("time_embedding"))?,
            proj_in: linear(cfg.embedding_dim, inner, vb.pp("proj_in"))?,
            embedding_proj: linear(cfg.embedding_dim, inner, vb.pp("embedding_proj"))?,
            encoder_hidden_states_proj: linear(
                cfg.embedding_dim,
                inner,
                vb.pp("encoder_hidden_states_proj"),
            )?,
            positional_embedding: vb.get((1, seq, inner), "positional_embedding")?,
            prd_embedding: vb.get((1, 1, inner), "prd_embedding")?,
            blocks,
            norm_out: layer_norm(
                inner,
                LayerNormConfig {
                    eps: NORM_EPS,
                    ..Default::default()
                },
                vb.pp("norm_out"),
            )?,
            proj_to_clip_embeddings: linear(
                inner,
                cfg.embedding_dim,
                vb.pp("proj_to_clip_embeddings"),
            )?,
            clip_mean: vb.get((1, cfg.embedding_dim), "clip_mean")?,
            clip_std: vb.get((1, cfg.embedding_dim), "clip_std")?,
            causal_mask: causal_penalty(seq, device)?.to_dtype(vb.dtype())?,
            cfg: cfg.clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn config(&self) -> &PriorConfig {
        &self.cfg
    }

    /// Predict the clean image embedding from a noised one.
    ///
    /// `latents` is `[b, embedding_dim]`, `timestep` is `[b]`,
    /// `proj_embedding` is the prompt's **projected** CLIP embedding
    /// `[b, embedding_dim]`, and `encoder_hidden_states` is its token sequence
    /// `[b, num_embeddings, embedding_dim]`. `text_mask` is `[b, num_embeddings]`
    /// with 1 for a real token and 0 for padding.
    ///
    /// Note the prediction is the **sample**, not the noise — see
    /// [`PriorScheduler`].
    pub fn forward(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        proj_embedding: &Tensor,
        encoder_hidden_states: &Tensor,
        text_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, _) = latents.dims2()?;
        let inner = self.cfg.inner_dim();

        let temb = timestep_embedding(timestep, inner)?.to_dtype(self.dtype)?;
        let temb = self.time_embedding.forward(&temb)?;

        let proj = self.embedding_proj.forward(proj_embedding)?.unsqueeze(1)?;
        let encoded = self
            .encoder_hidden_states_proj
            .forward(encoder_hidden_states)?;
        let latent = self.proj_in.forward(latents)?.unsqueeze(1)?;
        let prd = self
            .prd_embedding
            .broadcast_as((b, 1, inner))?
            .contiguous()?;

        // Order is the whole layout, and every permutation of it is 81 tokens
        // wide and loads against the same positional embedding.
        let xs = Tensor::cat(&[&encoded, &proj, &temb.unsqueeze(1)?, &latent, &prd], 1)?;
        let xs = xs.broadcast_add(&self.positional_embedding)?;

        let mask = self.attention_mask(b, text_mask)?;
        let mut xs = xs;
        for block in &self.blocks {
            xs = block.forward(&xs, &mask)?;
        }
        let xs = self.norm_out.forward(&xs)?;

        // The **last** position: the `prd` token, which is what the model was
        // trained to write its answer into.
        //
        // `contiguous()` is not decoration. The narrow leaves a `[b, 2048]`
        // view whose row stride is the *sequence's* 81*2048, and candle's
        // Metal matmul refuses that outright — "Invalid matmul arguments
        // [165888, 1] [1, 2048] (2, 768, 2048)", from inside the kernel, with
        // nothing naming this line. CPU accepts it and computes the right
        // answer, so every golden test here passes without the copy. This is
        // the same family as the quantised `start_offset` trap in
        // `sd_tensor::quantized`: a tensor that does not own its buffer is a
        // different tensor.
        let seq = self.cfg.sequence_length();
        let xs = xs.narrow(1, seq - 1, 1)?.squeeze(1)?.contiguous()?;
        self.proj_to_clip_embeddings.forward(&xs)
    }

    /// The additive mask: padded text positions, plus causality.
    ///
    /// The four extra positions are **never masked** — they are the
    /// conditioning and the answer, and masking them would hide the prompt
    /// from the token that has to produce the prediction.
    ///
    /// # Materialised to `[b, heads, seq, seq]`, not left to broadcast
    ///
    /// A `[b, 1, seq, seq]` mask is what the arithmetic naturally produces and
    /// it broadcasts correctly on CPU. **On Metal it does not run**: the seam's
    /// dispatcher documents that its fast paths want every leading axis at 1,
    /// and a mask that varies per batch element is outside that — the failure
    /// is a matmul shape error from inside the kernel, not a wrong number.
    ///
    /// Expanding here also matches the reference, which does
    /// `repeat_interleave(num_attention_heads)` for its own reasons. It costs
    /// 1.7 MB at batch 2.
    fn attention_mask(&self, batch: usize, text_mask: Option<&Tensor>) -> Result<Tensor> {
        let seq = self.cfg.sequence_length();
        let heads = self.cfg.num_attention_heads;
        let Some(text_mask) = text_mask else {
            return self
                .causal_mask
                .broadcast_as((batch, heads, seq, seq))?
                .contiguous();
        };
        let (b, n) = text_mask.dims2()?;
        if n != self.cfg.num_embeddings {
            return Err(sd_tensor::Error::Msg(format!(
                "text mask covers {n} positions, the prior takes {}",
                self.cfg.num_embeddings
            )));
        }
        // 1 -> 0 (attend), 0 -> -10000 (do not).
        let penalty = ((text_mask.to_dtype(self.dtype)? * -1.0)? + 1.0)?;
        let penalty = (penalty * MASKED)?;
        let tail = Tensor::zeros(
            (b, self.cfg.additional_embeddings),
            self.dtype,
            text_mask.device(),
        )?;
        let row = Tensor::cat(&[&penalty, &tail], 1)?;
        // `[b, 1, 1, seq]` against the causal `[1, 1, seq, seq]`: the text
        // penalty applies to a *key* whatever the query, and the causal one to
        // the pair.
        let row = row.reshape((b, 1, 1, seq))?;
        row.broadcast_add(&self.causal_mask)?
            .broadcast_as((b, heads, seq, seq))?
            .contiguous()
    }

    /// Un-whiten a sampled embedding into the units the image half expects.
    ///
    /// The prior works in a whitened space; `clip_mean` and `clip_std` are how
    /// it gets back. Skipping this returns a vector of the right shape whose
    /// scale is wrong by whatever the statistics say, and the UNet then
    /// conditions on something it has never seen.
    pub fn post_process(&self, latents: &Tensor) -> Result<Tensor> {
        latents
            .broadcast_mul(&self.clip_std)?
            .broadcast_add(&self.clip_mean)
    }
}

/// `[1, 1, seq, seq]`, `0` on and below the diagonal and [`MASKED`] above.
///
/// Deliberately not [`ops::causal_mask`], which uses `-inf`. The values here
/// are added to the text mask's own penalty, and this one has to match what
/// diffusers registered as a buffer.
fn causal_penalty(seq: usize, device: &Device) -> Result<Tensor> {
    let mut data = Vec::with_capacity(seq * seq);
    for i in 0..seq {
        for j in 0..seq {
            data.push(if j <= i { 0f32 } else { MASKED as f32 });
        }
    }
    Tensor::from_vec(data, (1, 1, seq, seq), device)
}

/// The prior's own sampler: DDPM over a 768-vector.
///
/// **Nothing else in this project samples like this.** Every other sampler
/// here is k-diffusion's: continuous sigmas, a model that predicts noise, and
/// an ODE step. This one is the original DDPM formulation — discrete
/// timesteps, ancestral, with variance added back — and the model predicts the
/// **sample** rather than the noise, so there is no `x0` to compute.
#[derive(Debug)]
pub struct PriorScheduler {
    alphas_cumprod: Vec<f64>,
    /// The subset of the 1000 training steps a run visits, descending.
    timesteps: Vec<usize>,
    train_timesteps: usize,
    /// The prior clamps its prediction. 5.0, not 1.0 — the embedding is
    /// whitened, so its natural range is several standard deviations.
    clip_sample_range: f64,
}

impl PriorScheduler {
    /// `steps` is the number of denoising steps; diffusers' pipeline uses 25.
    pub fn new(steps: usize) -> Self {
        let train_timesteps = crate::unclip::TRAIN_TIMESTEPS;
        // Exactly `DDPMScheduler.set_timesteps`: evenly spaced *by integer
        // ratio*, descending, and the arithmetic is `(arange(n) * ratio)`
        // reversed — not a linspace, which would land on different integers.
        let steps = steps.clamp(1, train_timesteps);
        let ratio = train_timesteps / steps;
        let timesteps = (0..steps).map(|i| i * ratio).rev().collect();
        Self {
            alphas_cumprod: crate::unclip::cosine_alphas_cumprod(train_timesteps),
            timesteps,
            train_timesteps,
            clip_sample_range: 5.0,
        }
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// One DDPM step. `predicted` is the model's estimate of the *clean*
    /// sample; `noise` is a fresh standard normal draw of the same shape.
    ///
    /// The final step adds no variance, which is what lands the run on a
    /// definite answer rather than a sample from a distribution around it.
    pub fn step(
        &self,
        predicted: &Tensor,
        timestep: usize,
        sample: &Tensor,
        noise: &Tensor,
    ) -> Result<Tensor> {
        let t = timestep.min(self.train_timesteps - 1);
        let prev = self.previous_timestep(t);

        let alpha_prod_t = self.alphas_cumprod[t];
        let alpha_prod_prev = match prev {
            Some(p) => self.alphas_cumprod[p],
            // `set_alpha_to_one` is not set, so the step off the end of the
            // ladder uses 1.0 — a noiseless starting point.
            None => 1.0,
        };
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_prev = 1.0 - alpha_prod_prev;
        let current_alpha = alpha_prod_t / alpha_prod_prev;
        let current_beta = 1.0 - current_alpha;

        // The model predicts x0 directly. Clamped, because an unclamped
        // prediction early in the run can be far outside the whitened range
        // and the coefficients below amplify it.
        let x0 = predicted.clamp(-self.clip_sample_range, self.clip_sample_range)?;

        let x0_coeff = alpha_prod_prev.sqrt() * current_beta / beta_prod_t;
        let sample_coeff = current_alpha.sqrt() * beta_prod_prev / beta_prod_t;
        let mean = ((x0 * x0_coeff)? + (sample * sample_coeff)?)?;

        // **Guarded on `t > 0`, not on "is this the last step"**, which is
        // what diffusers does. For this schedule the two coincide — the ladder
        // always ends at 0 — but they are different conditions and copying the
        // literal one costs nothing.
        if t == 0 {
            return Ok(mean);
        }
        // `fixed_small_log`: the *log* of the variance is what is clamped, and
        // the standard deviation is its exponential of a half. Using the
        // variance directly gives noise that is too small by its own square
        // root — a quieter, blurrier result with nothing to catch it.
        let variance = (beta_prod_prev / beta_prod_t) * current_beta;
        let log_variance = variance.max(1e-20).ln();
        let std = (0.5 * log_variance).exp();
        mean + (noise * std)?
    }

    /// The timestep the next step lands on, or `None` at the end of the run.
    ///
    /// Read from the schedule rather than computed as `t - ratio`: with a step
    /// count that does not divide 1000 the two disagree, and the reference
    /// walks the list.
    fn previous_timestep(&self, t: usize) -> Option<usize> {
        let i = self.timesteps.iter().position(|&x| x == t)?;
        self.timesteps.get(i + 1).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schedule_descends_from_near_the_top() {
        let s = PriorScheduler::new(25);
        assert_eq!(s.timesteps().len(), 25);
        assert_eq!(s.timesteps()[0], 960);
        assert_eq!(*s.timesteps().last().expect("non-empty"), 0);
        for w in s.timesteps().windows(2) {
            assert!(w[1] < w[0], "timesteps must descend");
        }
    }

    #[test]
    fn the_last_step_is_the_one_with_no_variance() {
        // Everything before it samples; the last lands. A run that added
        // variance at the end would return a noisy embedding and there is
        // nothing downstream to notice.
        let s = PriorScheduler::new(25);
        assert!(s.previous_timestep(960).is_some());
        assert_eq!(s.previous_timestep(0), None);
    }

    #[test]
    fn the_causal_penalty_is_finite_and_upper_triangular() {
        let m = causal_penalty(3, &Device::Cpu).expect("mask");
        let v = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            v,
            vec![0.0, -10000.0, -10000.0, 0.0, 0.0, -10000.0, 0.0, 0.0, 0.0]
        );
        // Finite on purpose: this is added to the text mask's own penalty.
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
