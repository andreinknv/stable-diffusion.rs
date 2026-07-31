//! unCLIP's prior on MLX: inventing an image embedding from text.
//!
//! A diffusion model in its own right, and an unusual one — the thing it
//! denoises is a single 768-vector, not a latent image. Twenty-five steps of
//! DDPM in embedding space, and the result is handed to the image half exactly
//! as if a photograph had produced it.
//!
//! # The sequence is five things concatenated
//!
//! ```text
//!   [ 77 text tokens | projected text embedding | timestep | latent | prd ]
//! ```
//!
//! 81 positions, which is why `positional_embedding` is 81 wide. The answer is
//! read from the **last** one — the learned `prd` token, which exists to have
//! somewhere to put it. Reading position 79, where the latent went in, returns
//! a well-shaped vector that is not the prediction. Every permutation of that
//! order is 81 tokens wide and loads against the same positional embedding.
//!
//! # The attention mask is load-bearing here and nowhere else
//!
//! Every other CLIP consumer in this project ignores the tokenizer's attention
//! mask: Stable Diffusion conditions on all 77 positions, padding included.
//! The prior does not — padded positions are masked out, and on top of that the
//! whole sequence is causally masked. Ignoring the mask runs and produces a
//! different image; on the reference's 10-of-77-position prompt the two
//! predictions differ by 0.60.
//!
//! **The mask value is `-10000`, not `-inf`.** Both underflow to zero through a
//! softmax, but the text mask and the causal mask are *added*, and two finite
//! penalties add to a finite one where two infinities do not.
//!
//! # The scheduler is not reimplemented here
//!
//! [`crate::prior::PriorScheduler::coefficients`] returns the DDPM scalars and
//! touches no tensor, so both backends call it and the formulation exists once.
//! Only the three tensor operations are below.

use sd_tensor::mlx::{concat, Array, Stream};
use sd_tensor::{Error, Result};

use super::{get, linear, sinusoid_embedding, Weights};

/// LayerNorm epsilon throughout the prior. PyTorch's default, which diffusers
/// does not override.
pub const NORM_EPS: f32 = 1e-5;

/// The value diffusers masks with. Not `-inf`; see the module docs.
pub const MASKED: f32 = -10_000.0;

/// Geometry of the prior.
#[derive(Debug, Clone, Copy)]
pub struct PriorConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    /// Width of what goes in and comes out: a CLIP image embedding.
    pub embedding_dim: usize,
    /// Text positions the prior attends over — CLIP's 77.
    pub num_embeddings: usize,
    /// The extra sequence positions: the projected embedding, the timestep,
    /// the latent, and `prd`.
    pub additional_embeddings: usize,
}

impl PriorConfig {
    /// `kakaobrain/karlo-v1-alpha`'s prior, which `stable-diffusion-2-1-unclip`
    /// ships.
    pub fn karlo() -> Self {
        Self {
            heads: 32,
            head_dim: 64,
            layers: 20,
            embedding_dim: 768,
            num_embeddings: 77,
            additional_embeddings: 4,
        }
    }

    /// 2048 for Karlo: the transformer is far wider than the embedding it
    /// carries, which is why `proj_in` and `proj_to_clip_embeddings` exist.
    pub fn inner_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn sequence_length(&self) -> usize {
        self.num_embeddings + self.additional_embeddings
    }
}

/// Self-attention with bias on **every** projection.
///
/// That bias is the difference from the UNet's attention, which loads
/// `to_q`/`to_k`/`to_v` without one.
fn attention(
    x: &Array,
    mask: &Array,
    cfg: &PriorConfig,
    w: &Weights,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let p = |n: &str| format!("{prefix}.{n}");
    let [b, seq, dim] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: prior attention {:?}", x.shape())));
    };
    let proj = |name: &str| -> Result<Array> {
        linear(
            x,
            get(w, &p(&format!("{name}.weight")))?,
            Some(get(w, &p(&format!("{name}.bias")))?),
            s,
        )?
        .reshape(&[b, seq, cfg.heads, cfg.head_dim], s)?
        .transpose(&[0, 2, 1, 3], s)
    };
    let out = proj("to_q")?.sdpa_masked(
        &proj("to_k")?,
        &proj("to_v")?,
        1.0 / (cfg.head_dim as f32).sqrt(),
        mask,
        s,
    )?;
    let merged = out
        .transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[b, seq, dim], s)?;
    // `to_out.0` — index 0 because diffusers wraps it in a list whose second
    // entry is dropout.
    linear(
        &merged,
        get(w, &p("to_out.0.weight"))?,
        Some(get(w, &p("to_out.0.bias"))?),
        s,
    )
}

/// The block's MLP: a plain-GELU two-layer, not the UNet's GEGLU.
fn feed_forward(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let p = |n: &str| format!("{prefix}.{n}");
    let h = linear(
        x,
        get(w, &p("ff.net.0.proj.weight"))?,
        Some(get(w, &p("ff.net.0.proj.bias"))?),
        s,
    )?
    .gelu(s)?;
    linear(
        &h,
        get(w, &p("ff.net.2.weight"))?,
        Some(get(w, &p("ff.net.2.bias"))?),
        s,
    )
}

/// One block: pre-norm, attention, residual, pre-norm, MLP, residual.
///
/// Mathematically a CLIP encoder layer; the tensors are named differently
/// because one was exported by `diffusers` and the other by `transformers`.
fn block(
    x: &Array,
    mask: &Array,
    cfg: &PriorConfig,
    w: &Weights,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let p = |n: &str| format!("{prefix}.{n}");
    let norm = |t: &Array, which: &str| -> Result<Array> {
        t.layer_norm(
            Some(get(w, &p(&format!("{which}.weight")))?),
            Some(get(w, &p(&format!("{which}.bias")))?),
            NORM_EPS,
            s,
        )
    };
    let h = x.add(
        &attention(&norm(x, "norm1")?, mask, cfg, w, &p("attn1"), s)?,
        s,
    )?;
    let ff = feed_forward(&norm(&h, "norm3")?, w, prefix, s)?;
    h.add(&ff, s)
}

/// `[1, 1, seq, seq]`, zero on and below the diagonal and [`MASKED`] above.
///
/// Deliberately not an `-inf` causal mask: these values are *added* to the text
/// mask's own penalty and have to match what diffusers registered as a buffer.
fn causal_penalty(seq: usize, s: &Stream) -> Result<Array> {
    let mut data = Vec::with_capacity(seq * seq);
    for i in 0..seq {
        for j in 0..seq {
            data.push(if j <= i { 0.0 } else { MASKED });
        }
    }
    Array::from_slice_f32(&data, &[seq * seq])?.reshape(&[1, 1, seq, seq], s)
}

/// The additive mask: padded text positions, plus causality.
///
/// The four extra positions are **never masked** — they are the conditioning
/// and the answer, and masking them would hide the prompt from the token that
/// has to produce the prediction.
///
/// `text_mask` is `[b, num_embeddings]` with 1 for a real token and 0 for
/// padding; `None` leaves only the causal penalty.
pub fn attention_mask(
    batch: usize,
    text_mask: Option<&Array>,
    cfg: &PriorConfig,
    s: &Stream,
) -> Result<Array> {
    let seq = cfg.sequence_length();
    let causal = causal_penalty(seq, s)?;
    let Some(text_mask) = text_mask else {
        return causal.broadcast_to(&[batch, 1, seq, seq], s)?.contiguous(s);
    };
    let [b, n] = text_mask.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: a text mask should be [b, n], got {:?}",
            text_mask.shape()
        )));
    };
    if n != cfg.num_embeddings {
        return Err(Error::Msg(format!(
            "mlx: the text mask covers {n} positions, the prior takes {}",
            cfg.num_embeddings
        )));
    }
    // 1 -> 0 (attend), 0 -> -10000 (do not).
    let penalty = Array::scalar_f32(1.0)?
        .sub(text_mask, s)?
        .mul(&Array::scalar_f32(MASKED)?, s)?;
    let tail = Array::from_slice_f32(
        &vec![0.0; b * cfg.additional_embeddings],
        &[b, cfg.additional_embeddings],
    )?;
    let row = concat(&[&penalty, &tail], 1, s)?.reshape(&[b, 1, 1, seq], s)?;
    // `[b, 1, 1, seq]` against the causal `[1, 1, seq, seq]`: the text penalty
    // applies to a *key* whatever the query, and the causal one to the pair.
    row.add(&causal, s)?
        .broadcast_to(&[b, 1, seq, seq], s)?
        .contiguous(s)
}

/// Predict the clean image embedding from a noised one.
///
/// `latents` is `[b, embedding_dim]`, `timestep` is `[b]`, `proj_embedding` is
/// the prompt's **projected** CLIP embedding `[b, embedding_dim]`, and
/// `encoder_hidden_states` is its token sequence
/// `[b, num_embeddings, embedding_dim]`.
///
/// The prediction is the **sample**, not the noise.
pub fn forward(
    latents: &Array,
    timestep: &Array,
    proj_embedding: &Array,
    encoder_hidden_states: &Array,
    text_mask: Option<&Array>,
    cfg: &PriorConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let [b, _] = latents.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: prior latents should be [b, dim], got {:?}",
            latents.shape()
        )));
    };
    let inner = cfg.inner_dim();

    // Both widths are `inner`: diffusers passes `out_dim=inner_dim`, so this
    // MLP does not widen the way the UNet's does.
    let temb = sinusoid_embedding(timestep, inner, s)?;
    let temb = linear(
        &temb,
        get(w, "time_embedding.linear_1.weight")?,
        Some(get(w, "time_embedding.linear_1.bias")?),
        s,
    )?
    .silu(s)?;
    let temb = linear(
        &temb,
        get(w, "time_embedding.linear_2.weight")?,
        Some(get(w, "time_embedding.linear_2.bias")?),
        s,
    )?;

    let proj = linear(
        proj_embedding,
        get(w, "embedding_proj.weight")?,
        Some(get(w, "embedding_proj.bias")?),
        s,
    )?
    .reshape(&[b, 1, inner], s)?;
    let encoded = linear(
        encoder_hidden_states,
        get(w, "encoder_hidden_states_proj.weight")?,
        Some(get(w, "encoder_hidden_states_proj.bias")?),
        s,
    )?;
    let latent = linear(
        latents,
        get(w, "proj_in.weight")?,
        Some(get(w, "proj_in.bias")?),
        s,
    )?
    .reshape(&[b, 1, inner], s)?;
    let prd = get(w, "prd_embedding")?
        .reshape(&[1, 1, inner], s)?
        .broadcast_to(&[b, 1, inner], s)?
        .contiguous(s)?;
    let temb = temb.reshape(&[b, 1, inner], s)?;

    // Order is the whole layout, and every permutation is 81 tokens wide.
    let x = concat(&[&encoded, &proj, &temb, &latent, &prd], 1, s)?;
    let x = x.add(get(w, "positional_embedding")?, s)?;

    let mask = attention_mask(b, text_mask, cfg, s)?;
    let mut x = x;
    for i in 0..cfg.layers {
        x = block(&x, &mask, cfg, w, &format!("transformer_blocks.{i}"), s)?;
    }
    let x = x.layer_norm(
        Some(get(w, "norm_out.weight")?),
        Some(get(w, "norm_out.bias")?),
        NORM_EPS,
        s,
    )?;

    // The **last** position: the `prd` token, which is what the model was
    // trained to write its answer into.
    let seq = cfg.sequence_length();
    let x = x.narrow(1, seq - 1, 1, s)?.reshape(&[b, inner], s)?;
    linear(
        &x,
        get(w, "proj_to_clip_embeddings.weight")?,
        Some(get(w, "proj_to_clip_embeddings.bias")?),
        s,
    )
}

/// Un-whiten a sampled embedding into the units the image half expects.
///
/// The prior works in a whitened space; `clip_mean` and `clip_std` are how it
/// gets back. Skipping this returns a vector of the right shape whose scale is
/// wrong by whatever the statistics say, and the UNet then conditions on
/// something it has never seen.
pub fn post_process(latents: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    latents
        .mul(get(w, "clip_std")?, s)?
        .add(get(w, "clip_mean")?, s)
}

/// One DDPM step, from [`crate::prior::StepCoefficients`].
///
/// The scalars come from the shared scheduler; this is only the three tensor
/// operations. `noise` is ignored at the final step, where `std` is `None`.
pub fn step(
    predicted: &Array,
    sample: &Array,
    noise: &Array,
    c: crate::prior::StepCoefficients,
    s: &Stream,
) -> Result<Array> {
    // Clamped, because an unclamped prediction early in the run can be far
    // outside the whitened range and the coefficients amplify it.
    // `clamp(x, -r, r)` as `-max(-max(x, -r), -r)`: MLX exposes an elementwise
    // maximum and no minimum, and `min(a, b) == -max(-a, -b)`.
    let lo = Array::scalar_f32(-c.clip_range as f32)?;
    let neg_hi = Array::scalar_f32(-c.clip_range as f32)?;
    let raised = predicted.maximum(&lo, s)?;
    let x0 = raised
        .mul(&Array::scalar_f32(-1.0)?, s)?
        .maximum(&neg_hi, s)?
        .mul(&Array::scalar_f32(-1.0)?, s)?;

    let mean = x0.mul(&Array::scalar_f32(c.x0_coeff as f32)?, s)?.add(
        &sample.mul(&Array::scalar_f32(c.sample_coeff as f32)?, s)?,
        s,
    )?;
    match c.std {
        Some(std) => mean.add(&noise.mul(&Array::scalar_f32(std as f32)?, s)?, s),
        None => Ok(mean),
    }
}
