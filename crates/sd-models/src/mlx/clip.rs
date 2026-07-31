//! CLIP's text tower on MLX, gated on `tests/golden/clip_encoder`.
//!
//! Prompt embeddings for the UNet's cross-attention. With the UNet and the VAE
//! already here, this is the last model a txt2img pass needs.
//!
//! **`layer_norm_eps` is 1e-5, not the VAE's 1e-6.** `clip/text_encoder.rs`
//! says it where the config is declared, and says why: copying the VAE value
//! produces a small uniform offset that is easy to misread as noise. This is
//! the third distinct epsilon in the port and none of them are interchangeable.
//!
//! **The activation is QuickGelu, `x * sigmoid(1.702 * x)`.** Not the erf GELU
//! the UNet's GEGLU uses. `ClipTextConfig::sd15` selects it explicitly.
//!
//! **The attention is causal.** CLIP's text tower masks future tokens, and MLX
//! implements that in its fused kernel — `sdpa_causal` rather than a mask
//! tensor built by hand.

use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{Error, Result};

use super::{get, linear, Weights};

/// LayerNorm epsilon in the text tower. Not the VAE's 1e-6.
pub const CLIP_EPS: f32 = 1e-5;
/// `openai/clip-vit-large-patch14`, which is SD 1.5's text encoder.
pub const HIDDEN: usize = 768;
pub const HEADS: usize = 12;
pub const LAYERS: usize = 12;
pub const MAX_POSITION: usize = 77;

/// Which GELU the feed-forward uses.
///
/// Not cosmetic: the two differ by about 1e-2, which is far too small to look
/// like a bug and far too large to be right. OpenAI's CLIP uses the quick
/// approximation; OpenCLIP uses plain `gelu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    QuickGelu,
    Gelu,
}

/// Geometry of a CLIP text tower.
#[derive(Debug, Clone, Copy)]
pub struct ClipConfig {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub activation: Activation,
    /// True when the checkpoint carries a `text_projection`, which SDXL's
    /// second encoder uses to produce the pooled embedding.
    pub projection: bool,
}

impl ClipConfig {
    /// SD 1.5's, and SDXL's *first*: `openai/clip-vit-large-patch14`.
    pub fn sd15() -> Self {
        Self {
            hidden: HIDDEN,
            heads: HEADS,
            layers: LAYERS,
            activation: Activation::QuickGelu,
            projection: false,
        }
    }

    /// SD 3's first text encoder: SD 1.5's geometry with a projection head.
    ///
    /// The projection is the difference that matters. SD 1.5 ships a plain
    /// `CLIPTextModel` and Flux takes its raw pooled hidden state; SD 3 ships
    /// `CLIPTextModelWithProjection` and takes the *projected* one. The two are
    /// different vectors and are not interchangeable.
    pub fn sd3_l() -> Self {
        Self {
            projection: true,
            ..Self::sd15()
        }
    }

    /// SD 2.x: OpenCLIP ViT-H/14.
    ///
    /// **23 layers, not 24.** SD 2.x conditions on the penultimate hidden
    /// state, so the conversion to diffusers drops the final layer outright and
    /// the ordinary "last layer, then `final_layer_norm`" path is then exactly
    /// right. Reaching for [`penultimate`] here instead would silently
    /// condition on layer 22.
    pub fn sd2() -> Self {
        Self {
            hidden: 1024,
            heads: 16,
            layers: 23,
            activation: Activation::Gelu,
            projection: false,
        }
    }

    /// SDXL's second text encoder, and SD 3's: OpenCLIP ViT-bigG.
    pub fn sdxl_2() -> Self {
        Self {
            hidden: 1280,
            heads: 20,
            layers: 32,
            activation: Activation::Gelu,
            projection: true,
        }
    }
}

/// Token and position embeddings, summed.
///
/// Positions are `0..seq`, not the token values — a lookup either way, and
/// confusing them yields a plausible tensor of the right shape.
pub fn embeddings(token_ids: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    let shape = token_ids.shape();
    let [_n, seq] = shape[..] else {
        return Err(Error::Msg(format!(
            "mlx: token ids should be [n, seq], got {shape:?}"
        )));
    };
    if seq > MAX_POSITION {
        return Err(Error::Msg(format!(
            "mlx: {seq} tokens exceeds CLIP's {MAX_POSITION} positions"
        )));
    }

    let tokens = get(w, "text_model.embeddings.token_embedding.weight")?.take(token_ids, 0, s)?;
    let positions: Vec<i32> = (0..seq as i32).collect();
    let positions = Array::from_slice_i32(&positions, &[seq])?;
    let pos = get(w, "text_model.embeddings.position_embedding.weight")?.take(&positions, 0, s)?;
    // [n, seq, hidden] + [seq, hidden] broadcasts over the batch.
    tokens.add(&pos, s)
}

/// One encoder layer: pre-norm, causal self-attention, pre-norm, MLP, both
/// residual.
fn encoder_layer(
    x: &Array,
    cfg: &ClipConfig,
    w: &Weights,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let p = |name: &str| format!("{prefix}.{name}");
    let [n, seq, hidden] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: clip layer got {:?}", x.shape())));
    };
    let head_dim = hidden / cfg.heads;

    // Attention.
    let y = x.layer_norm(
        Some(get(w, &p("layer_norm1.weight"))?),
        Some(get(w, &p("layer_norm1.bias"))?),
        CLIP_EPS,
        s,
    )?;
    let proj = |name: &str, src: &Array| -> Result<Array> {
        linear(
            src,
            get(w, &p(&format!("self_attn.{name}.weight")))?,
            w.get(&p(&format!("self_attn.{name}.bias"))),
            s,
        )
    };
    let split = |t: &Array| -> Result<Array> {
        t.reshape(&[n, seq, cfg.heads, head_dim], s)?
            .transpose(&[0, 2, 1, 3], s)
    };
    let attended = split(&proj("q_proj", &y)?)?.sdpa_causal(
        &split(&proj("k_proj", &y)?)?,
        &split(&proj("v_proj", &y)?)?,
        1.0 / (head_dim as f32).sqrt(),
        s,
    )?;
    let merged = attended
        .transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[n, seq, hidden], s)?;
    let x = x.add(&proj("out_proj", &merged)?, s)?;

    // MLP.
    let y = x.layer_norm(
        Some(get(w, &p("layer_norm2.weight"))?),
        Some(get(w, &p("layer_norm2.bias"))?),
        CLIP_EPS,
        s,
    )?;
    let y = linear(
        &y,
        get(w, &p("mlp.fc1.weight"))?,
        w.get(&p("mlp.fc1.bias")),
        s,
    )?;
    let y = match cfg.activation {
        Activation::QuickGelu => y.quick_gelu(s)?,
        Activation::Gelu => y.gelu(s)?,
    };
    let y = linear(
        &y,
        get(w, &p("mlp.fc2.weight"))?,
        w.get(&p("mlp.fc2.bias")),
        s,
    )?;
    x.add(&y, s)
}

/// The text tower: embeddings, twelve layers, final layer norm.
///
/// Returns `last_hidden_state`, which is what SD 1.5 conditions on — the
/// penultimate hidden state is SDXL's convention, not this one.
pub fn text_encoder(token_ids: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    text_encoder_with(token_ids, &ClipConfig::sd15(), w, s)
}

/// The tower run on an **already-embedded** sequence.
///
/// Textual inversion needs this: a trained embedding replaces the vectors at
/// its trigger's positions, which happens after the token lookup and before
/// the first layer. Re-tokenising afterwards would discard the substitution,
/// so the two halves have to be separable.
pub fn encode_from_embeds(
    embeds: &Array,
    cfg: &ClipConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let mut h = embeds.contiguous(s)?;
    for i in 0..cfg.layers {
        h = encoder_layer(&h, cfg, w, &format!("text_model.encoder.layers.{i}"), s)?;
    }
    h.layer_norm(
        Some(get(w, "text_model.final_layer_norm.weight")?),
        Some(get(w, "text_model.final_layer_norm.bias")?),
        CLIP_EPS,
        s,
    )
}

/// [`text_encoder`] for any of the towers in [`ClipConfig`].
pub fn text_encoder_with(
    token_ids: &Array,
    cfg: &ClipConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let mut h = embeddings(token_ids, w, s)?;
    for i in 0..cfg.layers {
        h = encoder_layer(&h, cfg, w, &format!("text_model.encoder.layers.{i}"), s)?;
    }
    h.layer_norm(
        Some(get(w, "text_model.final_layer_norm.weight")?),
        Some(get(w, "text_model.final_layer_norm.bias")?),
        CLIP_EPS,
        s,
    )
}

/// The hidden state SDXL conditions on: the **penultimate** layer, raw.
///
/// SD 1.5 uses the last layer normed; SDXL uses `hidden_states[-2]` *without*
/// `final_layer_norm`. Taking the wrong one produces images that are
/// recognisably related to the prompt but consistently worse — easy to blame on
/// the model.
pub fn penultimate(token_ids: &Array, cfg: &ClipConfig, w: &Weights, s: &Stream) -> Result<Array> {
    let (_, layers) = text_encoder_layers_with(token_ids, cfg, w, s)?;
    let idx = layers
        .len()
        .checked_sub(2)
        .ok_or_else(|| Error::Msg("mlx: an encoder with fewer than two layers".into()))?;
    layers[idx].contiguous(s)
}

/// The EOS hidden state, **without** the text projection.
///
/// # The *first* highest id, not the last
///
/// `transformers` locates EOS with `argmax`, which returns the first maximum.
/// The distinction is invisible for a tokenizer that pads with something other
/// than EOS — SDXL's second pads with `!`, id 0, so there is exactly one 49407
/// and either rule finds it. **SD 1.5's tokenizer pads with EOS itself**, so a
/// 10-token prompt has 68 copies and the two rules are 67 positions apart.
/// `docs/handoff.md` records that this cost 1.72 on the candle side, silently,
/// in every caller that pools a CLIP-L sequence.
pub fn pool(hidden: &Array, token_ids: &Array, s: &Stream) -> Result<Array> {
    let ids = token_ids.to_f32(s)?.to_vec_f32(s)?;
    let [n, seq] = token_ids.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: token ids should be [n, seq], got {:?}",
            token_ids.shape()
        )));
    };
    let mut positions = Vec::with_capacity(n);
    for b in 0..n {
        let row = &ids[b * seq..(b + 1) * seq];
        let highest = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let eos = row.iter().position(|&id| id == highest).unwrap_or(0);
        positions.push((b * seq + eos) as i32);
    }
    // Flatten the batch so one `take` gathers every row's own EOS.
    let hidden_dim = hidden.shape()[2];
    let flat = hidden.reshape(&[n * seq, hidden_dim], s)?;
    let idx = Array::from_slice_i32(&positions, &[n])?;
    flat.take(&idx, 0, s)
}

/// Apply `text_projection` to a pooled vector.
///
/// **`transformers` stores it without a bias**, and SDXL's UNet takes the
/// projected vector — not the raw `pooler_output`, which is what Flux takes.
/// The two are different vectors and are not interchangeable.
pub fn project(pooled: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    linear(pooled, get(w, "text_projection.weight")?, None, s)
}

/// The sequence SDXL conditions on and the pooled vector it micro-conditions
/// with, from **one** forward pass.
///
/// The obvious spelling — `penultimate()` then `pool(text_encoder(..))` —
/// encodes the prompt twice.
pub fn sdxl_conditioning(
    token_ids: &Array,
    cfg: &ClipConfig,
    w: &Weights,
    s: &Stream,
) -> Result<(Array, Array)> {
    let (final_state, layers) = text_encoder_layers_with(token_ids, cfg, w, s)?;
    let idx = layers
        .len()
        .checked_sub(2)
        .ok_or_else(|| Error::Msg("mlx: an encoder with fewer than two layers".into()))?;
    let pooled = pool(&final_state, token_ids, s)?;
    let pooled = if cfg.projection {
        project(&pooled, w, s)?
    } else {
        pooled
    };
    Ok((layers[idx].contiguous(s)?, pooled))
}

/// Every layer's output as well as the final state, for localising a failure.
pub fn text_encoder_layers(
    token_ids: &Array,
    w: &Weights,
    s: &Stream,
) -> Result<(Array, Vec<Array>)> {
    text_encoder_layers_with(token_ids, &ClipConfig::sd15(), w, s)
}

/// [`text_encoder_layers`] for any of the towers in [`ClipConfig`].
pub fn text_encoder_layers_with(
    token_ids: &Array,
    cfg: &ClipConfig,
    w: &Weights,
    s: &Stream,
) -> Result<(Array, Vec<Array>)> {
    let mut h = embeddings(token_ids, w, s)?;
    let mut per_layer = Vec::with_capacity(cfg.layers);
    for i in 0..cfg.layers {
        h = encoder_layer(&h, cfg, w, &format!("text_model.encoder.layers.{i}"), s)?;
        per_layer.push(h.contiguous(s)?);
    }
    let final_state = h.layer_norm(
        Some(get(w, "text_model.final_layer_norm.weight")?),
        Some(get(w, "text_model.final_layer_norm.bias")?),
        CLIP_EPS,
        s,
    )?;
    Ok((final_state, per_layer))
}
