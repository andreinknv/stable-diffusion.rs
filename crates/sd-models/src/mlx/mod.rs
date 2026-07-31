//! SD 1.5's UNet on MLX.
//!
//! The port described in `docs/handoff.md`, decision (b): built on
//! `sd_tensor::mlx`'s own surface rather than on a candle-shaped shim. Additive
//! — the candle modules beside this one are untouched and remain the default —
//! and every block here is gated against `tests/golden/unet_full`, the same
//! fixture and the same bound `golden_unet.rs` holds the candle path to.
//!
//! **Everything is NHWC.** MLX convolves channels-last, so activations carry
//! `[n, h, w, c]` and convolution weights `(out, kh, kw, in)`. diffusers ships
//! NCHW and `(out, in, kh, kw)`, so weights are transposed once at load rather
//! than per call. The one place this is nicer than candle: a per-channel bias
//! broadcasts over the last axis with no reshape.
//!
//! **Three epsilons, and they are not the same.** `unet/attention.rs` says it
//! for the candle path and it is just as true here: `Transformer2DModel`'s
//! GroupNorm is 1e-6, the LayerNorms inside a block are 1e-5, and the resnets
//! are 1e-5. Unifying them is tidier and wrong — it is exactly the bug found
//! in `mlx-examples` and reported as ml-explore/mlx-examples#1434, worth
//! 5.6e-4 at the UNet output.

use std::collections::HashMap;

use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{Error, Result};

/// GroupNorm epsilon on `Transformer2DModel`'s spatial wrapper.
pub const SPATIAL_NORM_EPS: f32 = 1e-6;
/// LayerNorm epsilon inside a transformer block. Not the same as the above.
pub const BLOCK_EPS: f32 = 1e-5;
/// GroupNorm epsilon in the resnets.
pub const RESNET_EPS: f32 = 1e-5;
/// SD 1.5 normalises over 32 groups throughout.
pub const NORM_GROUPS: usize = 32;

/// The tensors of a checkpoint, by their diffusers names.
pub type Weights = HashMap<String, Array>;

fn get<'a>(w: &'a Weights, key: &str) -> Result<&'a Array> {
    w.get(key)
        .ok_or_else(|| Error::Msg(format!("mlx: checkpoint has no `{key}`")))
}

/// `x @ w.T + b`, the diffusers `Linear` convention where `w` is `(out, in)`.
fn linear(x: &Array, w: &Array, b: Option<&Array>, s: &Stream) -> Result<Array> {
    let wt = w.transpose(&[1, 0], s)?;
    let y = x.matmul(&wt, s)?;
    match b {
        Some(b) => y.add(b, s),
        None => Ok(y),
    }
}

/// A convolution whose weights arrive in diffusers' `(out, in, kh, kw)` and are
/// used in MLX's `(out, kh, kw, in)`.
fn conv(x: &Array, w: &Array, b: Option<&Array>, padding: usize, s: &Stream) -> Result<Array> {
    let k = w.transpose(&[0, 2, 3, 1], s)?;
    let y = x.conv2d(&k, (1, 1), (padding, padding), (1, 1), 1, s)?;
    match b {
        Some(b) => y.add(b, s),
        None => Ok(y),
    }
}

/// diffusers' `ResnetBlock2D`.
///
/// `prefix` is the checkpoint path, e.g. `down_blocks.0.resnets.0`.
/// `temb` is the time embedding, `[n, temb_dim]`, already through its own MLP.
pub fn resnet_block(
    x: &Array,
    temb: &Array,
    w: &Weights,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let p = |name: &str| format!("{prefix}.{name}");

    let h = x.group_norm(
        NORM_GROUPS,
        RESNET_EPS,
        Some(get(w, &p("norm1.weight"))?),
        Some(get(w, &p("norm1.bias"))?),
        s,
    )?;
    let h = h.silu(s)?;
    let h = conv(
        &h,
        get(w, &p("conv1.weight"))?,
        Some(get(w, &p("conv1.bias"))?),
        1,
        s,
    )?;

    // The time embedding enters after the first convolution, as a per-channel
    // shift. SiLU first: diffusers applies the activation to `temb` here, not
    // when the embedding is built.
    let emb = linear(
        &temb.silu(s)?,
        get(w, &p("time_emb_proj.weight"))?,
        Some(get(w, &p("time_emb_proj.bias"))?),
        s,
    )?;
    let [n, c] = emb.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: time_emb_proj produced {:?}, expected [n, c]",
            emb.shape()
        )));
    };
    // NHWC, so the channel axis is last and this broadcasts over h and w.
    let h = h.add(&emb.reshape(&[n, 1, 1, c], s)?, s)?;

    let h = h.group_norm(
        NORM_GROUPS,
        RESNET_EPS,
        Some(get(w, &p("norm2.weight"))?),
        Some(get(w, &p("norm2.bias"))?),
        s,
    )?;
    let h = h.silu(s)?;
    let h = conv(
        &h,
        get(w, &p("conv2.weight"))?,
        Some(get(w, &p("conv2.bias"))?),
        1,
        s,
    )?;

    // The skip is projected only when the channel count changes. Borrowed
    // rather than cloned: an MLX handle is refcounted and there is no reason
    // to take a second reference just to add.
    let projected;
    let skip: &Array = match w.get(&p("conv_shortcut.weight")) {
        Some(sw) => {
            projected = conv(x, sw, w.get(&p("conv_shortcut.bias")), 0, s)?;
            &projected
        }
        None => x,
    };
    skip.add(&h, s)
}

/// One attention, `to_q`/`to_k`/`to_v`/`to_out.0`.
///
/// `context` is `None` for self-attention. SD 1.5's projections have no bias
/// except `to_out.0`.
fn attention(
    x: &Array,
    context: Option<&Array>,
    heads: usize,
    w: &Weights,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let p = |name: &str| format!("{prefix}.{name}");
    let kv = context.unwrap_or(x);

    let q = linear(x, get(w, &p("to_q.weight"))?, None, s)?;
    let k = linear(kv, get(w, &p("to_k.weight"))?, None, s)?;
    let v = linear(kv, get(w, &p("to_v.weight"))?, None, s)?;

    let [n, seq_q, inner] = q.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: attention query is {:?}",
            q.shape()
        )));
    };
    let head_dim = inner / heads;
    let seq_kv = k.shape()[1];

    // [n, seq, inner] -> [n, heads, seq, head_dim]
    let split = |t: &Array, seq: usize| -> Result<Array> {
        t.reshape(&[n, seq, heads, head_dim], s)?
            .transpose(&[0, 2, 1, 3], s)
    };
    let out = split(&q, seq_q)?.sdpa(
        &split(&k, seq_kv)?,
        &split(&v, seq_kv)?,
        1.0 / (head_dim as f32).sqrt(),
        s,
    )?;

    let merged = out
        .transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[n, seq_q, inner], s)?;
    linear(
        &merged,
        get(w, &p("to_out.0.weight"))?,
        Some(get(w, &p("to_out.0.bias"))?),
        s,
    )
}

/// GEGLU: one projection to twice the width, split into value and gate.
fn feed_forward(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let p = |name: &str| format!("{prefix}.{name}");
    let projected = linear(
        x,
        get(w, &p("net.0.proj.weight"))?,
        Some(get(w, &p("net.0.proj.bias"))?),
        s,
    )?;
    let dims = projected.shape();
    let inner = dims[dims.len() - 1] / 2;
    let last = dims.len() - 1;
    let value = projected.narrow(last, 0, inner, s)?;
    let gate = projected.narrow(last, inner, inner, s)?;
    let activated = value.mul(&gate.gelu(s)?, s)?;
    // `net.1` is dropout and has no parameters, hence the jump to 2. The gap is
    // real; renumbering breaks weight loading.
    linear(
        &activated,
        get(w, &p("net.2.weight"))?,
        Some(get(w, &p("net.2.bias"))?),
        s,
    )
}

/// diffusers' `BasicTransformerBlock`: self-attention, cross-attention, GEGLU,
/// each pre-normed and residual.
fn transformer_block(
    x: &Array,
    context: &Array,
    heads: usize,
    w: &Weights,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let p = |name: &str| format!("{prefix}.{name}");

    let y = x.layer_norm(
        Some(get(w, &p("norm1.weight"))?),
        Some(get(w, &p("norm1.bias"))?),
        BLOCK_EPS,
        s,
    )?;
    let x = x.add(&attention(&y, None, heads, w, &p("attn1"), s)?, s)?;

    let y = x.layer_norm(
        Some(get(w, &p("norm2.weight"))?),
        Some(get(w, &p("norm2.bias"))?),
        BLOCK_EPS,
        s,
    )?;
    let x = x.add(&attention(&y, Some(context), heads, w, &p("attn2"), s)?, s)?;

    let y = x.layer_norm(
        Some(get(w, &p("norm3.weight"))?),
        Some(get(w, &p("norm3.bias"))?),
        BLOCK_EPS,
        s,
    )?;
    x.add(&feed_forward(&y, w, &p("ff"), s)?, s)
}

/// diffusers' `Transformer2DModel`, the spatial wrapper around the blocks.
///
/// SD 1.5 has `use_linear_projection: false`, so `proj_in`/`proj_out` are 1x1
/// convolutions rather than linear layers.
pub fn transformer_2d(
    x: &Array,
    context: &Array,
    heads: usize,
    layers: usize,
    w: &Weights,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let p = |name: &str| format!("{prefix}.{name}");
    let [n, h, wd, c] = x.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: transformer_2d got {:?}",
            x.shape()
        )));
    };

    // 1e-6 here, unlike the 1e-5 LayerNorms inside the blocks.
    let y = x.group_norm(
        NORM_GROUPS,
        SPATIAL_NORM_EPS,
        Some(get(w, &p("norm.weight"))?),
        Some(get(w, &p("norm.bias"))?),
        s,
    )?;
    let y = conv(
        &y,
        get(w, &p("proj_in.weight"))?,
        Some(get(w, &p("proj_in.bias"))?),
        0,
        s,
    )?;

    let mut seq = y.reshape(&[n, h * wd, c], s)?;
    for i in 0..layers {
        seq = transformer_block(
            &seq,
            context,
            heads,
            w,
            &p(&format!("transformer_blocks.{i}")),
            s,
        )?;
    }

    let y = seq.reshape(&[n, h, wd, c], s)?;
    let y = conv(
        &y,
        get(w, &p("proj_out.weight"))?,
        Some(get(w, &p("proj_out.bias"))?),
        0,
        s,
    )?;
    y.add(x, s)
}

/// diffusers' `get_timestep_embedding` followed by the `time_embedding` MLP.
///
/// The frequencies are built on the CPU and uploaded once, because `mlx-c` has
/// no `arange` and this is a 160-element constant.
///
/// **`flip_sin_to_cos` is true and `downscale_freq_shift` is 0 for SD 1.5**,
/// so cosine comes first and the exponent is divided by `half`, not by
/// `half - 1`. Getting that denominator wrong is worth 2.1e-4 on the embedding
/// and is what `mlx-examples` does differently — measured while tracking down
/// ml-explore/mlx-examples#1434, and not the cause of that bug, but wrong all
/// the same.
pub fn timestep_embedding(
    timestep: &Array,
    channels: usize,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let half = channels / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10000f32.ln()) * i as f32 / half as f32).exp())
        .collect();
    let freqs = Array::from_slice_f32(&freqs, &[1, half])?;

    let t = timestep.reshape(&[timestep.elem_count(), 1], s)?;
    let angles = t.matmul(&freqs, s)?;
    // flip_sin_to_cos: cosine first.
    let emb = sd_tensor::mlx::concat(&[&angles.cos(s)?, &angles.sin(s)?], 1, s)?;

    let h = linear(
        &emb,
        get(w, "time_embedding.linear_1.weight")?,
        Some(get(w, "time_embedding.linear_1.bias")?),
        s,
    )?;
    linear(
        &h.silu(s)?,
        get(w, "time_embedding.linear_2.weight")?,
        Some(get(w, "time_embedding.linear_2.bias")?),
        s,
    )
}
