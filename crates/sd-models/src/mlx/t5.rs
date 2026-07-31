//! T5 v1.1 encoder on MLX — the text tower Flux and SD 3 use alongside CLIP.
//!
//! Structurally unlike the CLIP encoder beside it, in four ways that each have
//! a quiet failure mode:
//!
//! - **RMSNorm, not LayerNorm.** No mean subtraction and no bias. LayerNorm
//!   gives plausible activations and a wrong result.
//! - **No `1/sqrt(d_kv)` in attention.** T5 folds that scale into its
//!   initialisation. Applying it anyway sharpens every attention distribution
//!   and degrades the conditioning subtly. Here the scale is passed as `1.0`
//!   rather than cancelled by pre-multiplying the query, which the candle path
//!   has to do because its helper always divides.
//! - **Relative position bias, not absolute embeddings.** Computed once from a
//!   bucketed distance table and reused by every block.
//! - **Gated GELU.** Two input projections, one gating the other, so the
//!   feed-forward has three matrices rather than two.
//!
//! No layer carries a bias anywhere.

use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{Error, Result};

use super::{get, linear, Weights};

/// T5 v1.1 encoder geometry.
#[derive(Debug, Clone)]
pub struct T5Config {
    pub d_model: usize,
    pub d_ff: usize,
    /// Per-head width. 64 for every T5 size.
    pub d_kv: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_epsilon: f32,
}

impl T5Config {
    /// `google/t5-v1_1-xxl`, which Flux and SD 3 use.
    pub fn xxl() -> Self {
        Self {
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_layers: 24,
            num_heads: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_epsilon: 1e-6,
        }
    }

    /// `google/t5-v1_1-small`. Same architecture, 1/50th the size — which is
    /// what makes it a practical golden reference.
    pub fn v1_1_small() -> Self {
        Self {
            d_model: 512,
            d_ff: 1024,
            num_layers: 8,
            num_heads: 6,
            ..Self::xxl()
        }
    }

    fn inner_dim(&self) -> usize {
        self.num_heads * self.d_kv
    }
}

/// Bucket index for every `(query, key)` pair, row-major `[q_len * k_len]`.
///
/// Small integer arithmetic on the host: it runs once per forward pass and
/// building it as a tensor graph would be slower and much harder to read.
///
/// With `bidirectional` — which is what an encoder wants — the table is split
/// in half, one half for keys before the query and one for keys after, so
/// direction is preserved rather than collapsed onto distance.
pub fn relative_position_bucket(
    q_len: usize,
    k_len: usize,
    bidirectional: bool,
    num_buckets: usize,
    max_distance: usize,
) -> Vec<i32> {
    let mut out = Vec::with_capacity(q_len * k_len);
    // Halved for the bidirectional split, and every later bound is in terms of
    // the halved value — easy to lose, and losing it silently halves the
    // model's usable position range.
    let n_buckets = if bidirectional {
        num_buckets / 2
    } else {
        num_buckets
    };
    let max_exact = n_buckets / 2;

    for q in 0..q_len {
        for k in 0..k_len {
            let relative = k as i64 - q as i64;
            let (mut bucket, distance) = if bidirectional {
                let sign_offset = if relative > 0 { n_buckets } else { 0 };
                (sign_offset, relative.unsigned_abs() as usize)
            } else {
                (0, (-relative).max(0) as usize)
            };
            bucket += if distance < max_exact {
                distance
            } else {
                let ratio = (distance as f64 / max_exact as f64).ln()
                    / (max_distance as f64 / max_exact as f64).ln();
                let scaled = max_exact + (ratio * (n_buckets - max_exact) as f64) as usize;
                // Clamped: the formula keeps growing past the end of the table
                // for distances over `max_distance`, and indexing off the end
                // of the embedding is the failure this prevents.
                scaled.min(n_buckets - 1)
            };
            out.push(bucket as i32);
        }
    }
    out
}

/// The additive per-head bias, `[1, heads, seq, seq]`.
pub fn position_bias(cfg: &T5Config, seq: usize, w: &Weights, s: &Stream) -> Result<Array> {
    let buckets = relative_position_bucket(
        seq,
        seq,
        true,
        cfg.relative_attention_num_buckets,
        cfg.relative_attention_max_distance,
    );
    let idx = Array::from_slice_i32(&buckets, &[seq * seq])?;
    // The table lives on the first block only.
    let table = get(
        w,
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
    )?;
    // [seq*seq, heads] -> [seq, seq, heads] -> [1, heads, seq, seq]
    table
        .take(&idx, 0, s)?
        .reshape(&[seq, seq, cfg.num_heads], s)?
        .transpose(&[2, 0, 1], s)?
        .reshape(&[1, cfg.num_heads, seq, seq], s)
}

/// `x * rsqrt(mean(x^2) + eps) * weight`, composed rather than fused.
///
/// **Deliberately not `Array::rms_norm`**, which is the obvious thing to reach
/// for. `ops::rms_norm` on the candle side records the same choice and the
/// reason: a fused kernel that sums the row sequentially accumulates error
/// with row length, where a blocked reduction does not — and T5's rows are
/// long while its activations reach ~40,000 by the last block, so the two
/// compound.
///
/// Measured on `hidden_7` (peak 39,794): the fused kernel gives max_abs
/// 5.469e-2, this composition 2.148e-2 — 2.5x closer to transformers, and
/// better than the candle port's 1.211e-1 on the same tensor.
fn rms_norm(x: &Array, weight: &Array, eps: f32, s: &Stream) -> Result<Array> {
    let dims = x.shape();
    let last = dims.len() - 1;
    let mean_sq = x.mul(x, s)?.mean(&[last], true, s)?;
    let scaled = x.mul(&mean_sq.add(&Array::scalar_f32(eps)?, s)?.rsqrt(s)?, s)?;
    scaled.mul(weight, s)
}

fn attention(
    x: &Array,
    bias: &Array,
    cfg: &T5Config,
    w: &Weights,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let p = |n: &str| format!("{prefix}.{n}");
    let [b, n, _] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: t5 attention got {:?}", x.shape())));
    };
    let heads = cfg.num_heads;
    let d_kv = cfg.d_kv;

    let proj = |name: &str| -> Result<Array> {
        // No biases anywhere in T5.
        linear(x, get(w, &p(&format!("{name}.weight")))?, None, s)?
            .reshape(&[b, n, heads, d_kv], s)?
            .transpose(&[0, 2, 1, 3], s)
    };

    // **scale 1.0**: T5 folds 1/sqrt(d_kv) into its initialisation.
    let ctx = proj("q")?.sdpa_masked(&proj("k")?, &proj("v")?, 1.0, bias, s)?;
    let ctx = ctx
        .transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[b, n, cfg.inner_dim()], s)?;
    linear(&ctx, get(w, &p("o.weight"))?, None, s)
}

/// Gated GELU: `wo(gelu_new(wi_0(x)) * wi_1(x))`.
///
/// **`gelu_new` — the tanh approximation, not the erf form.** They differ by
/// ~1e-3, far above this model's noise floor, and the difference compounds
/// through the stack as a systematic drift rather than a visible break.
/// Measured with the erf form here: every hidden state was 200-300x further
/// from transformers than the candle port, with no single layer obviously
/// wrong.
fn feed_forward(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let p = |n: &str| format!("{prefix}.{n}");
    let gate = linear(x, get(w, &p("wi_0.weight"))?, None, s)?.gelu_approx(s)?;
    let up = linear(x, get(w, &p("wi_1.weight"))?, None, s)?;
    linear(&gate.mul(&up, s)?, get(w, &p("wo.weight"))?, None, s)
}

fn block(
    x: &Array,
    bias: &Array,
    cfg: &T5Config,
    w: &Weights,
    index: usize,
    s: &Stream,
) -> Result<Array> {
    let l = |i: usize, n: &str| format!("encoder.block.{index}.layer.{i}.{n}");

    let normed = rms_norm(
        x,
        get(w, &l(0, "layer_norm.weight"))?,
        cfg.layer_norm_epsilon,
        s,
    )?;
    let x = x.add(
        &attention(&normed, bias, cfg, w, &l(0, "SelfAttention"), s)?,
        s,
    )?;

    let normed = rms_norm(
        &x,
        get(w, &l(1, "layer_norm.weight"))?,
        cfg.layer_norm_epsilon,
        s,
    )?;
    x.add(&feed_forward(&normed, w, &l(1, "DenseReluDense"), s)?, s)
}

/// Every intermediate state, matching `output_hidden_states=True`.
///
/// `states[0]` is the token embedding and `states[i+1]` the output of block
/// `i` — **except the last, which has the final RMSNorm applied.**
/// transformers collects hidden states *before* each block and then appends
/// the normalised result once the loop ends, so the last entry is the model
/// output rather than the last block's raw output.
///
/// That asymmetry is invisible until compared: T5's activations grow to
/// ~40,000 by the last block and the final norm brings them back to order 1,
/// so mismatching it is a four-orders-of-magnitude discrepancy in one tensor
/// and none elsewhere. Measured here before it was fixed: 3.50e4 relative on
/// that one state, with every earlier state inside 1e-3.
pub fn encode_hidden_states(
    token_ids: &Array,
    cfg: &T5Config,
    w: &Weights,
    s: &Stream,
) -> Result<Vec<Array>> {
    let [_, seq] = token_ids.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: t5 token ids should be [n, seq], got {:?}",
            token_ids.shape()
        )));
    };
    let bias = position_bias(cfg, seq, w, s)?;
    let mut x = get(w, "shared.weight")?.take(token_ids, 0, s)?;

    let mut states = Vec::with_capacity(cfg.num_layers + 1);
    states.push(x.contiguous(s)?);
    for i in 0..cfg.num_layers {
        x = block(&x, &bias, cfg, w, i, s)?;
        states.push(x.contiguous(s)?);
    }
    // Replace the last raw state with the normalised model output, which is
    // what transformers reports.
    if let Some(last) = states.last_mut() {
        *last = rms_norm(
            last,
            get(w, "encoder.final_layer_norm.weight")?,
            cfg.layer_norm_epsilon,
            s,
        )?;
    }
    Ok(states)
}

/// Encode `[batch, seq]` token ids to `[batch, seq, d_model]`.
///
/// **The final RMSNorm matters more than it looks.** T5's activations grow to
/// roughly 40,000 by the last block and this norm brings them back to order 1,
/// so omitting it is a four-orders-of-magnitude discrepancy in one tensor and
/// none elsewhere.
pub fn encode(token_ids: &Array, cfg: &T5Config, w: &Weights, s: &Stream) -> Result<Array> {
    let mut states = encode_hidden_states(token_ids, cfg, w, s)?;
    // The last state is already normalised, per `encode_hidden_states`.
    states
        .pop()
        .ok_or_else(|| Error::Msg("mlx: a T5 stack must produce at least one state".into()))
}
