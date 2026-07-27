//! CLIP text encoder — 77 token ids to a `[batch, 77, 768]` conditioning
//! tensor.
//!
//! Parameter names follow `transformers` (`q_proj`, `layer_norm1`), not
//! `diffusers` (`to_q`, `norm1`). CLIP ships from the former and the VAE from
//! the latter; the two conventions coexist on purpose and unifying them would
//! break weight loading.
//!
//! Three details here are easy to get wrong in ways that still run and still
//! produce correctly-shaped output:
//!
//! * the activation is `quick_gelu`, not `gelu` — wrong by ~1e-2, which reads
//!   as noise rather than as a bug;
//! * `layer_norm_eps` is `1e-5`, where the VAE uses `1e-6`;
//! * the layer norm comes *before* attention with the residual added after.

use sd_tensor::nn::{embedding, layer_norm, linear, Embedding, LayerNorm, LayerNormConfig, Linear};
use sd_tensor::{ops, DType, IndexOp, Module, Result, Tensor, VarBuilder};

/// Which GELU the feed-forward uses.
///
/// Not cosmetic: `quick_gelu` and `gelu` differ by about 1e-2, which is far
/// too small to look like a bug and far too large to be right. SD 1.5's
/// encoder and SDXL's *first* encoder use `quick_gelu`; SDXL's second encoder
/// (OpenCLIP bigG) uses plain `gelu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipActivation {
    QuickGelu,
    Gelu,
}

/// Geometry of the text tower.
#[derive(Debug, Clone)]
pub struct ClipTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub activation: ClipActivation,
    /// `Some(dim)` when the checkpoint carries a `text_projection`, which
    /// SDXL's second encoder uses to produce the pooled embedding.
    pub projection_dim: Option<usize>,
}

impl ClipTextConfig {
    /// SD 1.5's text encoder: `openai/clip-vit-large-patch14`.
    pub fn sd15() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            max_position_embeddings: 77,
            // 1e-5, not the VAE's 1e-6. Copying the VAE value produces a small
            // uniform offset that is easy to misread as noise.
            layer_norm_eps: 1e-5,
            activation: ClipActivation::QuickGelu,
            projection_dim: None,
        }
    }

    /// SDXL's first text encoder — identical to [`Self::sd15`].
    pub fn sdxl_1() -> Self {
        Self::sd15()
    }

    /// SD 3's first text encoder: SD 1.5's geometry with a projection head.
    ///
    /// The projection is the difference that matters. SD 1.5 ships a plain
    /// `CLIPTextModel` and Flux takes its raw `pooler_output`; SD 3 ships
    /// `CLIPTextModelWithProjection` and takes the *projected* embedding, so
    /// [`ClipTextEncoder::pooled`] is correct here where
    /// [`ClipTextEncoder::pooled_hidden`] is correct for Flux.
    pub fn sd3_l() -> Self {
        Self {
            projection_dim: Some(768),
            ..Self::sd15()
        }
    }

    /// SD 2.x's text encoder: OpenCLIP ViT-H/14.
    ///
    /// **23 layers, not 24**, and that is not a typo. SD 2.x conditions on the
    /// *penultimate* hidden state, so the conversion to diffusers format drops
    /// the final layer outright — the shipped checkpoint has 23 and the normal
    /// "last layer, then `final_layer_norm`" path is then exactly right.
    /// Building 24 here would fail to load, which is the good direction, but
    /// reaching for [`ClipTextEncoder::penultimate_hidden_state`] instead would
    /// silently condition on layer 22.
    pub fn sd2() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 23,
            num_attention_heads: 16,
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            // OpenCLIP activates with plain gelu; only OpenAI's CLIP uses the
            // quick approximation.
            activation: ClipActivation::Gelu,
            projection_dim: None,
        }
    }

    /// SDXL's second text encoder, and SD 3's: OpenCLIP ViT-bigG.
    ///
    /// Bigger in every dimension, and it activates with plain `gelu`.
    pub fn sdxl_2() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 20,
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            activation: ClipActivation::Gelu,
            projection_dim: Some(1280),
        }
    }

    /// Width of one attention head.
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Multi-head causal self-attention.
#[derive(Debug)]
struct ClipAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    heads: usize,
    head_dim: usize,
}

impl ClipAttention {
    fn new(cfg: &ClipTextConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            q_proj: linear(h, h, vb.pp("q_proj"))?,
            k_proj: linear(h, h, vb.pp("k_proj"))?,
            v_proj: linear(h, h, vb.pp("v_proj"))?,
            out_proj: linear(h, h, vb.pp("out_proj"))?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// `[b, seq, hidden]` -> `[b, heads, seq, head_dim]`.
    ///
    /// Via `[b, seq, heads, head_dim]` and a transpose. Reshaping straight to
    /// `[b, heads, seq, head_dim]` interleaves the heads wrongly and yields
    /// garbage of exactly the right shape.
    fn split_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;
        xs.reshape((b, seq, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, seq, hidden) = xs.dims3()?;

        let q = self.split_heads(&self.q_proj.forward(xs)?)?;
        let k = self.split_heads(&self.k_proj.forward(xs)?)?;
        let v = self.split_heads(&self.v_proj.forward(xs)?)?;

        let out = ops::scaled_dot_product_attention_masked(&q, &k, &v, mask)?;

        // Back to [b, seq, hidden]. `contiguous` is required before `reshape`
        // on a transposed tensor.
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq, hidden))?;
        self.out_proj.forward(&out)
    }
}

/// Feed-forward block: `fc2(quick_gelu(fc1(x)))`.
#[derive(Debug)]
struct ClipMlp {
    fc1: Linear,
    fc2: Linear,
    activation: ClipActivation,
}

impl ClipMlp {
    fn new(cfg: &ClipTextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?,
            fc2: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?,
            activation: cfg.activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = match self.activation {
            ClipActivation::QuickGelu => ops::quick_gelu(&xs)?,
            ClipActivation::Gelu => ops::gelu(&xs)?,
        };
        self.fc2.forward(&xs)
    }
}

/// One transformer layer. Pre-layernorm.
#[derive(Debug)]
struct ClipEncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: ClipAttention,
    layer_norm2: LayerNorm,
    mlp: ClipMlp,
}

impl ClipEncoderLayer {
    fn new(cfg: &ClipTextConfig, vb: VarBuilder) -> Result<Self> {
        let norm_cfg = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        Ok(Self {
            layer_norm1: layer_norm(cfg.hidden_size, norm_cfg, vb.pp("layer_norm1"))?,
            self_attn: ClipAttention::new(cfg, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(cfg.hidden_size, norm_cfg, vb.pp("layer_norm2"))?,
            mlp: ClipMlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Norm before the sublayer, residual added after. Reversing this still
        // runs and still emits [b, 77, 768].
        let residual = xs;
        let xs = self.layer_norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, mask)?;
        let xs = (residual + xs)?;

        let residual = &xs;
        let ys = self.layer_norm2.forward(&xs)?;
        let ys = self.mlp.forward(&ys)?;
        residual + ys
    }
}

/// The full text tower.
#[derive(Debug)]
pub struct ClipTextEncoder {
    token_embedding: Embedding,
    position_embedding: Embedding,
    layers: Vec<ClipEncoderLayer>,
    final_layer_norm: LayerNorm,
    /// `[1, 1, 77, 77]`, built once. Rebuilding it per call would allocate a
    /// 77x77 tensor on every forward for a value that never changes.
    causal_mask: Tensor,
    /// `0..77`, likewise fixed. CLIP attends over the full context, including
    /// the EOS padding, so positions never depend on the prompt.
    positions: Tensor,
    /// Present when the checkpoint has one. SDXL's second encoder projects the
    /// pooled hidden state through this to produce `text_embeds`.
    text_projection: Option<Linear>,
    /// The weights' dtype. Everything this module builds itself — the causal
    /// mask especially — has to match it, or the first `broadcast_add` fails.
    dtype: DType,
}

impl ClipTextEncoder {
    /// `vb` is rooted at the checkpoint root, so `text_model.*` resolves
    /// directly beneath it.
    pub fn new(cfg: &ClipTextConfig, vb: VarBuilder) -> Result<Self> {
        let vb_text = vb.pp("text_model");
        let vb_emb = vb_text.pp("embeddings");

        let token_embedding = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_emb.pp("token_embedding"),
        )?;
        let position_embedding = embedding(
            cfg.max_position_embeddings,
            cfg.hidden_size,
            vb_emb.pp("position_embedding"),
        )?;

        let vb_layers = vb_text.pp("encoder").pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(ClipEncoderLayer::new(cfg, vb_layers.pp(i.to_string()))?);
        }

        let final_layer_norm = layer_norm(
            cfg.hidden_size,
            LayerNormConfig {
                eps: cfg.layer_norm_eps,
                ..Default::default()
            },
            vb_text.pp("final_layer_norm"),
        )?;

        let device = vb.device();
        let dtype = vb.dtype();
        let seq = cfg.max_position_embeddings;
        Ok(Self {
            token_embedding,
            position_embedding,
            layers,
            final_layer_norm,
            // Built in f32 and cast: the mask is -inf and 0, both exact in
            // f16, so this loses nothing.
            causal_mask: ops::causal_mask(seq, device)?.to_dtype(dtype)?,
            positions: Tensor::arange(0u32, seq as u32, device)?,
            dtype,
            text_projection: match cfg.projection_dim {
                // No bias: `CLIPTextModelWithProjection` uses
                // `nn.Linear(hidden, projection, bias=False)`.
                Some(dim) => Some(sd_tensor::nn::linear_no_bias(
                    cfg.hidden_size,
                    dim,
                    vb.pp("text_projection"),
                )?),
                None => None,
            },
        })
    }

    /// `token_ids` is `[batch, 77]`. Returns `[batch, 77, 768]`.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.forward_with_layers(token_ids).map(|(out, _)| out)
    }

    /// [`Self::forward`], also returning each encoder layer's output.
    ///
    /// Exists for the golden test: when the final tensor disagrees with the
    /// reference, the per-layer outputs say *which* layer diverged first,
    /// which localizes the bug far better than a single final number.
    pub fn forward_with_layers(&self, token_ids: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        // The reference stores ids as int64; embedding lookup wants U32.
        let token_ids = token_ids.to_dtype(DType::U32)?;

        let xs = self.token_embedding.forward(&token_ids)?;
        let pos = self.position_embedding.forward(&self.positions)?;
        let mut xs = xs.broadcast_add(&pos)?;

        let mut per_layer = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            xs = layer.forward(&xs, &self.causal_mask)?;
            per_layer.push(xs.clone());
        }

        // SD uses the output *after* the final norm, and without the text
        // projection — that projection belongs to CLIP retrieval, not to
        // diffusion conditioning.
        let out = self.final_layer_norm.forward(&xs)?;
        Ok((out, per_layer))
    }

    /// The hidden state SDXL conditions on: the **penultimate** layer, not the
    /// final one, and without `final_layer_norm`.
    ///
    /// SD 1.5 uses the last layer normed; SDXL uses `hidden_states[-2]` raw.
    /// Taking the wrong one produces images that are recognisably related to
    /// the prompt but consistently worse, which is easy to blame on the model.
    pub fn penultimate_hidden_state(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (_, layers) = self.forward_with_layers(token_ids)?;
        let idx = layers.len().checked_sub(2).ok_or_else(|| {
            sd_tensor::Error::Msg("encoder has fewer than two layers".to_string())
        })?;
        Ok(layers[idx].clone())
    }

    /// The pooled text embedding: the final-layer hidden state at the EOS
    /// position, projected.
    ///
    /// `None` when the checkpoint carries no `text_projection`. SDXL feeds
    /// this into the UNet's additional conditioning alongside the sequence.
    ///
    /// EOS is located as the **argmax of the token ids**, which is how
    /// `transformers` does it and works because EOS is the highest id in
    /// CLIP's vocabulary. Using the last index instead would pick a padding
    /// slot — the same token, but at the wrong position, so the wrong vector.
    pub fn pooled(&self, token_ids: &Tensor) -> Result<Option<Tensor>> {
        let Some(projection) = &self.text_projection else {
            return Ok(None);
        };
        let pooled = self.pooled_hidden(token_ids)?;
        Ok(Some(projection.forward(&pooled)?))
    }

    /// The EOS hidden state **without** the text projection.
    ///
    /// This is `transformers`' `pooler_output`, and it is what Flux
    /// conditions on. [`Self::pooled`] additionally applies
    /// `text_projection`, which is what SDXL's second text encoder wants —
    /// the two are different vectors of different widths and are not
    /// interchangeable.
    ///
    /// Available regardless of whether the checkpoint carries a projection,
    /// which matters because CLIP-L as shipped with SD 1.5 (and reused by
    /// Flux) is a plain `CLIPTextModel` and has none.
    pub fn pooled_hidden(&self, token_ids: &Tensor) -> Result<Tensor> {
        let hidden = self.forward(token_ids)?;
        let ids = token_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;

        let mut rows = Vec::with_capacity(ids.len());
        for (b, row) in ids.iter().enumerate() {
            let eos = row
                .iter()
                .enumerate()
                .max_by_key(|(_, &id)| id)
                .map(|(i, _)| i)
                .unwrap_or(0);
            rows.push(hidden.i(b)?.narrow(0, eos, 1)?);
        }
        Tensor::cat(&rows, 0)
    }

    /// The dtype this encoder's weights are in.
    ///
    /// Callers hand it token ids and get this back; anything concatenated with
    /// the result has to match. Exposed because the pipeline runs the models
    /// in f16 and the sampler in f32, so it needs to know where the boundary
    /// is rather than assume.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The embedding sum, before any encoder layer. Step 3 of the forward.
    ///
    /// Separate entry point so the golden reference's `embeddings` capture can
    /// be checked on its own: if the embeddings are wrong, every layer after
    /// them is wrong too, and the per-layer report would blame layer 0.
    pub fn embeddings(&self, token_ids: &Tensor) -> Result<Tensor> {
        let token_ids = token_ids.to_dtype(DType::U32)?;
        let xs = self.token_embedding.forward(&token_ids)?;
        let pos = self.position_embedding.forward(&self.positions)?;
        xs.broadcast_add(&pos)
    }
}
