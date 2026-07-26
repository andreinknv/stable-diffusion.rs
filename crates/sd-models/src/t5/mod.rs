//! T5 v1.1 encoder — the text tower Flux and SD 3 use alongside CLIP.
//!
//! Structurally unlike the CLIP encoder already in this crate, in four ways
//! that each have a quiet failure mode:
//!
//! - **RMSNorm, not LayerNorm.** No mean subtraction and no bias. Using
//!   LayerNorm gives plausible activations and a wrong result.
//! - **No `1/sqrt(d_k)` in attention.** T5 folds that scale into its
//!   initialisation instead. Applying it anyway sharpens every attention
//!   distribution and degrades the conditioning subtly.
//! - **Relative position bias, not absolute embeddings.** Computed once in
//!   the first block from a bucketed distance table and reused by every
//!   later block.
//! - **Gated GELU.** Two input projections, one gated by the other, so the
//!   feed-forward has three matrices rather than two.
//!
//! No layer carries a bias anywhere.

mod bucket;
mod tokenizer;

pub use bucket::relative_position_bucket;
pub use tokenizer::{T5Tokenizer, FLUX_MAX_LENGTH};

use sd_tensor::gguf::QTensor;
use sd_tensor::nn::{linear_no_bias, Embedding, Linear, VarBuilder};
use sd_tensor::quantized::QLinear;
use sd_tensor::{ops, DType, Module, Result, Tensor, D};

/// T5 v1.1 encoder geometry.
#[derive(Debug, Clone)]
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    /// Per-head width. 64 for every T5 size, which is what makes the
    /// attention-scale cancellation in [`T5Attention`] exact.
    pub d_kv: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_epsilon: f64,
}

impl T5Config {
    /// `google/t5-v1_1-xxl`, the encoder Flux conditions on.
    pub fn xxl() -> Self {
        Self {
            vocab_size: 32128,
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
    /// what makes it a practical golden reference for this file.
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

/// A T5 projection, dense or quantised.
///
/// Quantised weights are not merely smaller here, they are what makes the
/// model *run*: T5's activations reach tens of thousands and f16 tops out at
/// 65504, so loading dequantised-to-f16 produces NaN around block 10. Holding
/// the blocks and expanding per matmul keeps every activation in f32.
#[derive(Debug)]
pub enum T5Proj {
    Dense(Linear),
    Quantized(QLinear),
}

impl T5Proj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(xs),
            Self::Quantized(q) => q.forward(xs),
        }
    }
}

/// Where a block's weights come from.
#[derive(Clone, Copy)]
pub enum T5Source<'a> {
    Dense(&'a VarBuilder<'a>),
    Quantized(&'a QuantizedWeights),
}

/// Quantised tensors keyed by HuggingFace name, as produced by
/// `sd_loader::t5_qtensors_from_gguf`.
pub type QuantizedWeights = std::collections::HashMap<String, std::sync::Arc<QTensor>>;

impl<'a> T5Source<'a> {
    fn proj(&self, path: &str, in_dim: usize, out_dim: usize) -> Result<T5Proj> {
        match self {
            Self::Dense(vb) => {
                let mut sub = (*vb).clone();
                for part in path.split('.') {
                    sub = sub.pp(part);
                }
                Ok(T5Proj::Dense(linear_no_bias(in_dim, out_dim, sub)?))
            }
            Self::Quantized(w) => {
                let key = format!("{path}.weight");
                let t = w.get(&key).ok_or_else(|| {
                    sd_tensor::Error::Msg(format!("quantised T5 is missing {key}"))
                })?;
                Ok(T5Proj::Quantized(QLinear::new(t.clone(), None)?))
            }
        }
    }

    /// Norm scales and the embedding stay dense: they are stored F32 in the
    /// file already and are a rounding error next to the projections.
    fn dense_tensor(&self, path: &str, dim: usize) -> Result<Tensor> {
        match self {
            Self::Dense(vb) => {
                let mut sub = (*vb).clone();
                for part in path.split('.') {
                    sub = sub.pp(part);
                }
                sub.get(dim, "weight")
            }
            Self::Quantized(w) => {
                let key = format!("{path}.weight");
                let t = w.get(&key).ok_or_else(|| {
                    sd_tensor::Error::Msg(format!("quantised T5 is missing {key}"))
                })?;
                t.dequantize(&t.device())
            }
        }
    }

    fn dense_2d(&self, path: &str, rows: usize, cols: usize) -> Result<Tensor> {
        match self {
            Self::Dense(vb) => {
                let mut sub = (*vb).clone();
                for part in path.split('.') {
                    sub = sub.pp(part);
                }
                sub.get((rows, cols), "weight")
            }
            Self::Quantized(w) => {
                let key = format!("{path}.weight");
                let t = w.get(&key).ok_or_else(|| {
                    sd_tensor::Error::Msg(format!("quantised T5 is missing {key}"))
                })?;
                t.dequantize(&t.device())
            }
        }
    }
}

/// Root-mean-square normalisation.
///
/// `x * rsqrt(mean(x^2) + eps) * weight`. No mean subtraction and no bias,
/// unlike LayerNorm.
#[derive(Debug)]
pub struct T5RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl T5RmsNorm {
    pub fn new(dim: usize, eps: f64, src: T5Source, path: &str) -> Result<Self> {
        Ok(Self {
            weight: src.dense_tensor(path, dim)?,
            eps,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // The variance is accumulated in f32 even when the weights are f16.
        // transformers does the same, and at d_model = 4096 the sum of
        // squares overflows f16 for perfectly ordinary activations.
        let dtype = xs.dtype();
        let xs32 = xs.to_dtype(DType::F32)?;
        let variance = xs32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = xs32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        normed
            .to_dtype(dtype)?
            .broadcast_mul(&self.weight.to_dtype(dtype)?)
    }
}

/// T5 self-attention with an additive relative position bias.
#[derive(Debug)]
pub struct T5Attention {
    q: T5Proj,
    k: T5Proj,
    v: T5Proj,
    o: T5Proj,
    /// Present only in the first block; every other block reuses the bias it
    /// computes.
    relative_attention_bias: Option<Embedding>,
    num_heads: usize,
    d_kv: usize,
    num_buckets: usize,
    max_distance: usize,
}

impl T5Attention {
    pub fn new(cfg: &T5Config, has_relative_bias: bool, src: T5Source, path: &str) -> Result<Self> {
        let inner = cfg.inner_dim();
        Ok(Self {
            q: src.proj(&format!("{path}.q"), cfg.d_model, inner)?,
            k: src.proj(&format!("{path}.k"), cfg.d_model, inner)?,
            v: src.proj(&format!("{path}.v"), cfg.d_model, inner)?,
            o: src.proj(&format!("{path}.o"), inner, cfg.d_model)?,
            relative_attention_bias: if has_relative_bias {
                Some(Embedding::new(
                    src.dense_2d(
                        &format!("{path}.relative_attention_bias"),
                        cfg.relative_attention_num_buckets,
                        cfg.num_heads,
                    )?,
                    cfg.num_heads,
                ))
            } else {
                None
            },
            num_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
            num_buckets: cfg.relative_attention_num_buckets,
            max_distance: cfg.relative_attention_max_distance,
        })
    }

    /// The `[1, heads, q_len, k_len]` bias table, if this block owns one.
    pub fn compute_bias(
        &self,
        q_len: usize,
        k_len: usize,
        device: &sd_tensor::Device,
    ) -> Result<Option<Tensor>> {
        let Some(emb) = &self.relative_attention_bias else {
            return Ok(None);
        };
        // Bidirectional: the encoder sees the whole sequence.
        let buckets =
            relative_position_bucket(q_len, k_len, true, self.num_buckets, self.max_distance);
        let idx = Tensor::from_vec(buckets, (q_len, k_len), device)?;
        // [q, k] -> [q, k, heads] -> [1, heads, q, k]
        let bias = emb.forward(&idx)?.permute((2, 0, 1))?.unsqueeze(0)?;
        Ok(Some(bias.contiguous()?))
    }

    fn shape_heads(&self, xs: &Tensor, b: usize, n: usize) -> Result<Tensor> {
        xs.reshape((b, n, self.num_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// `position_bias` is added to the attention scores. `mask` marks padding
    /// with a large negative value; both are additive and are summed here.
    pub fn forward(&self, xs: &Tensor, position_bias: &Tensor) -> Result<Tensor> {
        let (b, n, _) = xs.dims3()?;

        let q = self.shape_heads(&self.q.forward(xs)?, b, n)?;
        let k = self.shape_heads(&self.k.forward(xs)?, b, n)?;
        let v = self.shape_heads(&self.v.forward(xs)?, b, n)?;

        // T5 does *not* divide the scores by sqrt(d_kv) — the scale is folded
        // into its initialisation. Our attention helper always divides, so
        // pre-multiplying q cancels it. This is exact rather than approximate:
        // d_kv is 64 for every T5 size, sqrt(64) = 8, and both 8 and 1/8 are
        // representable in binary floating point, so the round trip introduces
        // no error. Reusing the helper this way keeps the chunking and memory
        // budget that a hand-rolled matmul here would bypass.
        let q = (q * (self.d_kv as f64).sqrt())?;

        let bias = position_bias
            .broadcast_as((b, self.num_heads, n, n))?
            .contiguous()?;
        let ctx = ops::scaled_dot_product_attention_masked(&q, &k, &v, &bias)?;

        let ctx = ctx
            .transpose(1, 2)?
            .reshape((b, n, self.num_heads * self.d_kv))?;
        self.o.forward(&ctx)
    }
}

/// Gated GELU feed-forward: `wo(gelu(wi_0(x)) * wi_1(x))`.
#[derive(Debug)]
pub struct T5FeedForward {
    wi_0: T5Proj,
    wi_1: T5Proj,
    wo: T5Proj,
}

impl T5FeedForward {
    pub fn new(cfg: &T5Config, src: T5Source, path: &str) -> Result<Self> {
        Ok(Self {
            wi_0: src.proj(&format!("{path}.wi_0"), cfg.d_model, cfg.d_ff)?,
            wi_1: src.proj(&format!("{path}.wi_1"), cfg.d_model, cfg.d_ff)?,
            wo: src.proj(&format!("{path}.wo"), cfg.d_ff, cfg.d_model)?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // `gelu_new` in transformers — the tanh approximation, not the erf
        // form. They differ by ~1e-3, which is far above this model's noise
        // floor and would show up as a systematic drift.
        let gate = ops::gelu_approx(&self.wi_0.forward(xs)?)?;
        self.wo.forward(&(gate * self.wi_1.forward(xs)?)?)
    }
}

/// One encoder block: pre-normed attention, then pre-normed feed-forward,
/// each with a residual.
#[derive(Debug)]
pub struct T5Block {
    attention: T5Attention,
    attention_norm: T5RmsNorm,
    ff: T5FeedForward,
    ff_norm: T5RmsNorm,
}

impl T5Block {
    pub fn new(cfg: &T5Config, has_relative_bias: bool, src: T5Source, path: &str) -> Result<Self> {
        let l0 = format!("{path}.layer.0");
        let l1 = format!("{path}.layer.1");
        Ok(Self {
            attention: T5Attention::new(
                cfg,
                has_relative_bias,
                src,
                &format!("{l0}.SelfAttention"),
            )?,
            attention_norm: T5RmsNorm::new(
                cfg.d_model,
                cfg.layer_norm_epsilon,
                src,
                &format!("{l0}.layer_norm"),
            )?,
            ff: T5FeedForward::new(cfg, src, &format!("{l1}.DenseReluDense"))?,
            ff_norm: T5RmsNorm::new(
                cfg.d_model,
                cfg.layer_norm_epsilon,
                src,
                &format!("{l1}.layer_norm"),
            )?,
        })
    }

    pub fn forward(&self, xs: &Tensor, position_bias: &Tensor) -> Result<Tensor> {
        let h = self
            .attention
            .forward(&self.attention_norm.forward(xs)?, position_bias)?;
        let xs = (xs + h)?;
        let h = self.ff.forward(&self.ff_norm.forward(&xs)?)?;
        xs + h
    }

    pub fn attention(&self) -> &T5Attention {
        &self.attention
    }
}

/// The T5 encoder stack.
#[derive(Debug)]
pub struct T5EncoderModel {
    embed_tokens: Embedding,
    blocks: Vec<T5Block>,
    final_norm: T5RmsNorm,
}

impl T5EncoderModel {
    /// `vb` should be rooted so that `shared` / `encoder.block.N` resolve
    /// beneath it — i.e. at the model root, not at `encoder`.
    pub fn new(cfg: &T5Config, vb: VarBuilder) -> Result<Self> {
        Self::from_source(cfg, T5Source::Dense(&vb))
    }

    /// Build with the weights left quantised.
    ///
    /// Preferred for T5-XXL: 2.7 GB resident against 18.8 at F32, and every
    /// activation stays f32, which a dequantise-to-f16 load cannot manage —
    /// see [`T5Proj`].
    pub fn from_quantized(cfg: &T5Config, weights: &QuantizedWeights) -> Result<Self> {
        Self::from_source(cfg, T5Source::Quantized(weights))
    }

    fn from_source(cfg: &T5Config, src: T5Source) -> Result<Self> {
        // The embedding is a lookup, not a matmul, so it is dequantised: at
        // 32128 x 4096 that is 526 MB, against 9 GB for the projections.
        let embed_tokens = Embedding::new(
            src.dense_2d("shared", cfg.vocab_size, cfg.d_model)?,
            cfg.d_model,
        );
        let blocks = (0..cfg.num_layers)
            // Only block 0 owns a relative attention bias; the rest share it.
            .map(|i| T5Block::new(cfg, i == 0, src, &format!("encoder.block.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens,
            blocks,
            final_norm: T5RmsNorm::new(
                cfg.d_model,
                cfg.layer_norm_epsilon,
                src,
                "encoder.final_layer_norm",
            )?,
        })
    }

    /// The `[1, heads, n, n]` relative position bias the whole stack shares.
    ///
    /// Public because it is the single most error-prone part of this model and
    /// worth verifying on its own — a mistake here perturbs every attention
    /// score slightly rather than failing.
    pub fn position_bias(&self, n: usize, device: &sd_tensor::Device) -> Result<Tensor> {
        // Computed once by the first block and threaded through the rest.
        // Recomputing it per block would be correct but pointlessly slow, and
        // *loading* it per block would be wrong — only block 0 has weights.
        self.blocks[0]
            .attention()
            .compute_bias(n, n, device)?
            .ok_or_else(|| {
                sd_tensor::Error::Msg(
                    "the first T5 block must own the relative attention bias".into(),
                )
            })
    }

    /// Encode `[batch, seq]` token ids to `[batch, seq, d_model]`.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let states = self.forward_with_hidden_states(token_ids)?;
        states.into_iter().next_back().ok_or_else(|| {
            sd_tensor::Error::Msg("a T5 stack must produce at least one state".into())
        })
    }

    /// Every intermediate state, matching `output_hidden_states=True`.
    ///
    /// `states[0]` is the token embedding, `states[i+1]` the output of block
    /// `i` — **except** the last, which has the final RMSNorm applied.
    /// transformers collects hidden states *before* each block and then
    /// appends the normalised result once the loop ends, so the last entry is
    /// the model output rather than the last block's raw output.
    ///
    /// That asymmetry is worth stating because it is invisible until
    /// compared: T5's activations grow to ~40,000 by the last block and the
    /// final norm brings them back to order 1, so mismatching it produces a
    /// four-orders-of-magnitude discrepancy in one tensor and none elsewhere.
    pub fn forward_with_hidden_states(&self, token_ids: &Tensor) -> Result<Vec<Tensor>> {
        let mut xs = self.embed_tokens.forward(token_ids)?;
        let (_, n, _) = xs.dims3()?;
        let position_bias = self.position_bias(n, xs.device())?.to_dtype(xs.dtype())?;

        let mut states = Vec::with_capacity(self.blocks.len() + 1);
        states.push(xs.clone());
        for block in &self.blocks {
            xs = block.forward(&xs, &position_bias)?;
            states.push(xs.clone());
        }
        // Replace the raw last-block output with the normalised one.
        if let Some(last) = states.last_mut() {
            *last = self.final_norm.forward(last)?;
        }
        Ok(states)
    }
}
