//! IP-Adapter's decoupled cross-attention on MLX.
//!
//! An IP-Adapter conditions on an *image* alongside the text. Each
//! cross-attention gains a second key/value pair and returns
//!
//! ```text
//!   attn(q, k_text, v_text)  +  scale * attn(q, k_image, v_image)
//! ```
//!
//! with `to_out` applied once, to the sum.
//!
//! **This is not the same as appending the image tokens to the text ones.**
//! Attention is not linear in K and V, so a concatenation is a different
//! function — and a plausible-looking one, which is why it is worth stating.
//!
//! # The index order is not the construction order
//!
//! The checkpoint numbers its entries by diffusers' flat processor list, which
//! visits **down blocks, then up blocks, then the mid block** — while this UNet
//! runs down, mid, up. And the entries sit at *odd* indices, because that list
//! alternates self- and cross-attention and only cross-attention has them.
//!
//! So slot `i` of the visit order maps to key `2 * order[i] + 1`. Get this
//! wrong and every correction lands on a differently-sized layer, which usually
//! fails to load — but between the two 1280-wide regions it would not.

use std::cell::Cell;

use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{Error, Result};

use super::Weights;

/// SD 1.5's mapping from visit order to checkpoint slot.
///
/// Derived rather than typed out: six down cross-attentions, then the mid one,
/// then nine up. diffusers lists them down, up, mid — so the mid entry is last
/// in that list (15) while it is seventh here.
pub fn sd15_order() -> Vec<usize> {
    let mut order: Vec<usize> = (0..6).collect();
    order.push(15);
    order.extend(6..15);
    order
}

/// An attached adapter: its per-layer projections and the projected image
/// tokens every layer attends to.
pub struct IpAdapter<'a> {
    weights: &'a Weights,
    /// `[batch, tokens, cross_dim]`, already through `image_proj`.
    pub tokens: Array,
    /// Strength. **0 contributes exactly nothing**, not merely almost nothing.
    pub scale: f32,
    order: Vec<usize>,
    next: Cell<usize>,
}

impl<'a> IpAdapter<'a> {
    pub fn new(weights: &'a Weights, tokens: Array, scale: f32) -> Self {
        Self {
            weights,
            tokens,
            scale,
            order: sd15_order(),
            next: Cell::new(0),
        }
    }

    /// Reset before each forward, since the counter tracks position within one
    /// pass rather than across the run.
    pub fn rewind(&self) {
        self.next.set(0);
    }

    /// The `(to_k_ip, to_v_ip)` for the next cross-attention to be visited.
    fn take(&self) -> Result<Option<(&'a Array, &'a Array)>> {
        let i = self.next.get();
        let Some(&slot) = self.order.get(i) else {
            return Ok(None);
        };
        self.next.set(i + 1);
        let key = 2 * slot + 1;
        let k = self
            .weights
            .get(&format!("ip_adapter.{key}.to_k_ip.weight"));
        let v = self
            .weights
            .get(&format!("ip_adapter.{key}.to_v_ip.weight"));
        match (k, v) {
            (Some(k), Some(v)) => Ok(Some((k, v))),
            _ => Ok(None),
        }
    }
}

/// The image half of one decoupled cross-attention, already scaled.
///
/// Returns `None` when there is no adapter, when it has run out of layers, or
/// when the strength is zero — the last so that `scale = 0` reproduces an
/// unadapted run exactly rather than adding a scaled-to-nothing tensor.
#[allow(clippy::too_many_arguments)]
pub fn image_attention(
    adapter: Option<&IpAdapter<'_>>,
    q: &Array,
    heads: usize,
    head_dim: usize,
    s: &Stream,
) -> Result<Option<Array>> {
    let Some(ip) = adapter else { return Ok(None) };
    let Some((k_w, v_w)) = ip.take()? else {
        return Ok(None);
    };
    if ip.scale == 0.0 {
        return Ok(None);
    }

    let [n, _, seq_q, _] = q.shape()[..] else {
        return Ok(None);
    };
    let tokens = &ip.tokens;
    let seq_kv = tokens.shape()[1];

    let project = |w: &Array| -> Result<Array> {
        super::linear(tokens, w, None, s)?
            .reshape(&[n, seq_kv, heads, head_dim], s)?
            .transpose(&[0, 2, 1, 3], s)
    };
    let out = q.sdpa(
        &project(k_w)?,
        &project(v_w)?,
        1.0 / (head_dim as f32).sqrt(),
        s,
    )?;
    let _ = seq_q;
    Ok(Some(out.mul(&Array::scalar_f32(ip.scale)?, s)?))
}

/// The IP-Adapter's own token count. Four, not the text tower's 77.
pub const NUM_TOKENS: usize = 4;
/// LayerNorm epsilon in the projection. PyTorch's default.
const PROJ_EPS: f32 = 1e-5;

/// `image_proj`: a CLIP image embedding to the tokens the UNet attends over.
///
/// `[b, embed_dim]` -> `[b, NUM_TOKENS, cross_dim]`. **One Linear producing
/// every token at once**, then reshaped — which is why its output width is
/// `tokens * cross_dim` rather than `cross_dim`, and why splitting it into
/// four projections would load nothing.
///
/// The input is the vision tower's **projected** embedding — 1024 for ViT-H,
/// not the pooled 1280. `clip_vision::image_embeds` is the one that gives it.
pub fn image_proj(
    image_embeds: &Array,
    cross_dim: usize,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let [b, _] = image_embeds.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: image embeds should be [b, dim], got {:?}",
            image_embeds.shape()
        )));
    };
    let projected = super::linear(
        image_embeds,
        super::get(w, "image_proj.proj.weight")?,
        w.get("image_proj.proj.bias"),
        s,
    )?
    .reshape(&[b, NUM_TOKENS, cross_dim], s)?;
    projected.layer_norm(
        Some(super::get(w, "image_proj.norm.weight")?),
        Some(super::get(w, "image_proj.norm.bias")?),
        PROJ_EPS,
        s,
    )
}
