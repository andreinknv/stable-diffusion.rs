//! AnimateDiff motion modules on MLX: attention over *time*.
//!
//! A motion module sits after every resnet in the UNet — 21 for SD 1.5 — and
//! each is a small transformer whose attention runs across frames rather than
//! across pixels. That is the whole difference between "N images that happen to
//! share a seed" and "frames of one motion".
//!
//! # The reshape is the mechanism
//!
//! ```text
//!   [b*f, c, h, w]     the UNet's ordinary activation
//!   -> [b*f, h*w, c]   pixels as sequence  (spatial)
//!   -> [b*h*w, f, c]   frames as sequence  (temporal)
//! ```
//!
//! Attention then mixes across `f`, so each pixel sees itself at every other
//! frame and at no other pixel. Getting this permute wrong leaves a module that
//! runs, keeps every shape, and blurs across space instead of time — which
//! looks like a weak motion module rather than a broken one.
//!
//! # Three things that are wrong by about 3 while keeping every shape
//!
//! Each of these was found the hard way on the candle side and is stated here
//! rather than rediscovered:
//!
//! - **The normalisation spans frames.** `[b*f, c, h, w]` is regrouped to
//!   `[b, c, f, h, w]` first, so each group's statistics are taken over the
//!   whole clip rather than one frame.
//! - **The positional encoding goes on the normed states inside each attention
//!   path, and is applied twice** — once before each attention — not once onto
//!   the residual stream.
//! - **Both attentions are self-attention over time.** The adapter's config
//!   sets `motion_cross_attention_dim: null`, so `attn2` attends over frames
//!   too rather than over the text.
//!
//! With one frame, temporal attention over a sequence of length 1 is the
//! identity up to the projections, so a module left installed for a still image
//! is nearly — but not exactly — a no-op.

use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{Error, Result};

use super::{get, linear, Weights, NORM_GROUPS};

/// Heads in every motion module, per the adapter config.
pub const HEADS: usize = 8;

/// Is a motion module attached at this path?
pub fn present(w: &Weights, prefix: &str) -> bool {
    w.contains_key(&format!("{prefix}.proj_in.weight"))
}

/// Self-attention over the frame axis, `[b*h*w, f, c]` in and out.
fn temporal_attention(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let p = |n: &str| format!("{prefix}.{n}");
    let [b, n, c] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: motion attention {:?}", x.shape())));
    };
    let hd = c / HEADS;
    let proj = |name: &str| -> Result<Array> {
        linear(
            x,
            get(w, &p(&format!("{name}.weight")))?,
            w.get(&p(&format!("{name}.bias"))),
            s,
        )?
        .reshape(&[b, n, HEADS, hd], s)?
        .transpose(&[0, 2, 1, 3], s)
    };
    let out = proj("to_q")?.sdpa(&proj("to_k")?, &proj("to_v")?, 1.0 / (hd as f32).sqrt(), s)?;
    let merged = out
        .transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[b, n, c], s)?;
    linear(
        &merged,
        get(w, &p("to_out.0.weight"))?,
        w.get(&p("to_out.0.bias")),
        s,
    )
}

/// One temporal transformer block. `x` is `[b*h*w, f, c]` — already temporal.
///
/// The permute that makes it temporal happens once, in [`forward`], not here.
fn block(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let p = |n: &str| format!("{prefix}.{n}");
    let frames = x.shape()[1];
    // The learned table is 32 long; take the prefix this clip needs.
    let pe = get(w, &p("pos_embed.pe"))?.narrow(1, 0, frames, s)?;

    let norm = |t: &Array, which: &str| -> Result<Array> {
        t.layer_norm(
            Some(get(w, &p(&format!("{which}.weight")))?),
            Some(get(w, &p(&format!("{which}.bias")))?),
            1e-5,
            s,
        )
    };

    // The positional encoding goes on the **normed** states, twice, not once
    // onto the residual stream.
    let normed = norm(x, "norm1")?.add(&pe, s)?;
    let h = temporal_attention(&normed, w, &p("attn1"), s)?.add(x, s)?;

    let normed = norm(&h, "norm2")?.add(&pe, s)?;
    let h = temporal_attention(&normed, w, &p("attn2"), s)?.add(&h, s)?;

    // GEGLU, as everywhere else in this UNet.
    let y = norm(&h, "norm3")?;
    let projected = linear(
        &y,
        get(w, &p("ff.net.0.proj.weight"))?,
        w.get(&p("ff.net.0.proj.bias")),
        s,
    )?;
    let dims = projected.shape();
    let last = dims.len() - 1;
    let inner = dims[last] / 2;
    let value = projected.narrow(last, 0, inner, s)?;
    let gate = projected.narrow(last, inner, inner, s)?;
    let ff = linear(
        &value.mul(&gate.gelu(s)?, s)?,
        get(w, &p("ff.net.2.weight"))?,
        w.get(&p("ff.net.2.bias")),
        s,
    )?;
    h.add(&ff, s)
}

/// One motion module. `x` is the UNet's NHWC activation `[b*f, h, w, c]`.
///
/// Residual around the whole module, so a zero-initialised `proj_out` makes it
/// an exact identity — which is how these are trained against a frozen base.
pub fn forward(
    x: &Array,
    num_frames: usize,
    w: &Weights,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let [bf, height, width, c] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: motion module {:?}", x.shape())));
    };
    if num_frames == 0 || bf % num_frames != 0 {
        return Err(Error::Msg(format!(
            "mlx: a batch of {bf} does not divide into {num_frames} frames"
        )));
    }
    let b = bf / num_frames;

    // **The normalisation spans frames.** Regroup so the clip is one group's
    // population, not one frame. NHWC keeps channels last, so this is
    // [b, f, h, w, c] folded to [b, f*h, w, c] — the group statistics are over
    // everything but the batch and the channel either way.
    let grouped = x.reshape(&[b, num_frames * height, width, c], s)?;
    let normed = grouped
        .group_norm(
            NORM_GROUPS,
            1e-5,
            Some(get(w, &format!("{prefix}.norm.weight"))?),
            Some(get(w, &format!("{prefix}.norm.bias"))?),
            s,
        )?
        .reshape(&[b, num_frames, height, width, c], s)?;

    // -> [b*h*w, f, c]: pixels become the batch, frames the sequence.
    let temporal = normed
        .transpose(&[0, 2, 3, 1, 4], s)?
        .contiguous(s)?
        .reshape(&[b * height * width, num_frames, c], s)?;

    let mut h = linear(
        &temporal,
        get(w, &format!("{prefix}.proj_in.weight"))?,
        w.get(&format!("{prefix}.proj_in.bias")),
        s,
    )?;
    let mut i = 0usize;
    while w.contains_key(&format!("{prefix}.transformer_blocks.{i}.norm1.weight")) {
        h = block(&h, w, &format!("{prefix}.transformer_blocks.{i}"), s)?;
        i += 1;
    }
    let h = linear(
        &h,
        get(w, &format!("{prefix}.proj_out.weight"))?,
        w.get(&format!("{prefix}.proj_out.bias")),
        s,
    )?;

    // Back to [b*f, h, w, c].
    h.reshape(&[b, height, width, num_frames, c], s)?
        .transpose(&[0, 3, 1, 2, 4], s)?
        .contiguous(s)?
        .reshape(&[bf, height, width, c], s)?
        .add(x, s)
}
