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
fn encoder_layer(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let p = |name: &str| format!("{prefix}.{name}");
    let [n, seq, hidden] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: clip layer got {:?}", x.shape())));
    };
    let head_dim = hidden / HEADS;

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
        t.reshape(&[n, seq, HEADS, head_dim], s)?
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
    )?
    .quick_gelu(s)?;
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
    let mut h = embeddings(token_ids, w, s)?;
    for i in 0..LAYERS {
        h = encoder_layer(&h, w, &format!("text_model.encoder.layers.{i}"), s)?;
    }
    h.layer_norm(
        Some(get(w, "text_model.final_layer_norm.weight")?),
        Some(get(w, "text_model.final_layer_norm.bias")?),
        CLIP_EPS,
        s,
    )
}

/// Every layer's output as well as the final state, for localising a failure.
pub fn text_encoder_layers(
    token_ids: &Array,
    w: &Weights,
    s: &Stream,
) -> Result<(Array, Vec<Array>)> {
    let mut h = embeddings(token_ids, w, s)?;
    let mut per_layer = Vec::with_capacity(LAYERS);
    for i in 0..LAYERS {
        h = encoder_layer(&h, w, &format!("text_model.encoder.layers.{i}"), s)?;
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
