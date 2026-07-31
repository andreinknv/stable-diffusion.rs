//! Wan's video transformer.
//!
//! **The first model here with a frame axis**, and it is a different shape from
//! every image DiT in this crate:
//!
//! **Cross-attention, not joint.** `attn1` is self-attention over the video
//! tokens with rotary positions; `attn2` attends to the text with none, and the
//! text never receives an update. There is no second stream to modulate.
//!
//! **The modulation is a learned table plus the timestep.** Each block carries
//! a `scale_shift_table` of `[1, 6, dim]` which is *added* to a projection of
//! the timestep shared by every block — where SD 3 and Qwen-Image project the
//! timestep separately in each block. The table is the per-block part.
//!
//! **QK-norm is across heads, not per head.** `norm_q` is `[dim]`, the whole
//! width, so it applies to the flat projection *before* the head split. Every
//! other model in this crate norms per head with a `[head_dim]` weight, and the
//! two are indistinguishable at a glance — one is 4096 wide and the other 128.
//!
//! **Three norms per block, two of them affine-free.** `norm1` and `norm3` have
//! no weights; `norm2`, which precedes the cross-attention, has both.
//!
//! The patch embedding is a 3D convolution whose stride equals its kernel, so
//! it is implemented here as a patchify and a linear — the same arithmetic
//! without needing a `conv3d`.

use sd_tensor::mlx::{concat, Array, Stream};
use sd_tensor::{Error, Result};

use super::quantized::WeightSource;

/// Wan transformer geometry.
#[derive(Debug, Clone)]
pub struct WanConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    /// The text encoder's width. 4096 — Wan conditions on umT5.
    pub text_dim: usize,
    /// Sinusoid width before the timestep embedder.
    pub freq_dim: usize,
    pub ffn_dim: usize,
    /// `(t, h, w)`, and 1 on the frame axis for every published Wan.
    pub patch_size: (usize, usize, usize),
    pub theta: f64,
    pub eps: f32,
}

impl WanConfig {
    /// `Wan-AI/Wan2.1-T2V-1.3B`.
    pub fn t2v_1_3b() -> Self {
        Self {
            num_heads: 12,
            head_dim: 128,
            layers: 30,
            in_channels: 16,
            out_channels: 16,
            text_dim: 4096,
            freq_dim: 256,
            ffn_dim: 8960,
            patch_size: (1, 2, 2),
            theta: 10_000.0,
            eps: 1e-6,
        }
    }

    /// `Wan-AI/Wan2.1-T2V-14B`.
    pub fn t2v_14b() -> Self {
        Self {
            num_heads: 40,
            head_dim: 128,
            layers: 40,
            ffn_dim: 13824,
            ..Self::t2v_1_3b()
        }
    }

    pub fn dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// `(t_dim, h_dim, w_dim)` — how the head dimension splits across the
    /// three rotary axes.
    ///
    /// **Height and width take `2 * (head_dim / 6)` each and time takes the
    /// remainder**, so the split is not even: at 128 it is `(44, 42, 42)`. An
    /// even three-way split builds tables of the right total width whose
    /// per-axis boundaries are wrong.
    pub fn rope_dims(&self) -> (usize, usize, usize) {
        let hw = 2 * (self.head_dim / 6);
        (self.head_dim - 2 * hw, hw, hw)
    }
}

/// Rotary tables for a `f x h x w` patch grid, each `[1, 1, seq, head_dim]`.
///
/// **Full width, duplicated.** `repeat_interleave_real` in the reference, so
/// each frequency occupies two adjacent slots and covers one `(x[2i], x[2i+1])`
/// pair — the same convention FLUX.2 uses, and not Z-Image's half-width tables.
pub fn rope_tables(
    frames: usize,
    h: usize,
    w: usize,
    cfg: &WanConfig,
    s: &Stream,
) -> Result<(Array, Array)> {
    let (t_dim, h_dim, w_dim) = cfg.rope_dims();
    let total = t_dim + h_dim + w_dim;
    let seq = frames * h * w;
    let mut cos = vec![0.0f32; seq * total];
    let mut sin = vec![0.0f32; seq * total];

    for f in 0..frames {
        for y in 0..h {
            for x in 0..w {
                let token = (f * h + y) * w + x;
                let mut off = 0usize;
                for (dim, pos) in [(t_dim, f), (h_dim, y), (w_dim, x)] {
                    for i in 0..dim / 2 {
                        let omega = 1.0 / cfg.theta.powf(2.0 * i as f64 / dim as f64);
                        let angle = pos as f64 * omega;
                        // Duplicated adjacently.
                        let base = token * total + off + 2 * i;
                        cos[base] = angle.cos() as f32;
                        cos[base + 1] = angle.cos() as f32;
                        sin[base] = angle.sin() as f32;
                        sin[base + 1] = angle.sin() as f32;
                    }
                    off += dim;
                }
            }
        }
    }
    Ok((
        Array::from_slice_f32(&cos, &[1, 1, seq, total])?.contiguous(s)?,
        Array::from_slice_f32(&sin, &[1, 1, seq, total])?.contiguous(s)?,
    ))
}

/// Interleaved rotation of `[b, heads, seq, head_dim]`.
fn apply_rope(x: &Array, cos: &Array, sin: &Array, s: &Stream) -> Result<Array> {
    let [b, h, n, d] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: wan rope {:?}", x.shape())));
    };
    let pairs = x.reshape(&[b, h, n, d / 2, 2], s)?;
    let even = pairs.narrow(4, 0, 1, s)?.reshape(&[b, h, n, d / 2], s)?;
    let odd = pairs.narrow(4, 1, 1, s)?.reshape(&[b, h, n, d / 2], s)?;
    // The reference reads `cos[..., 0::2]` and `sin[..., 1::2]` out of the
    // duplicated tables; taking every other entry of either gives the same
    // half-width sequence, so one stride serves both.
    let c = cos
        .reshape(&[1, 1, n, d / 2, 2], s)?
        .narrow(4, 0, 1, s)?
        .reshape(&[1, 1, n, d / 2], s)?;
    let sn = sin
        .reshape(&[1, 1, n, d / 2, 2], s)?
        .narrow(4, 1, 1, s)?
        .reshape(&[1, 1, n, d / 2], s)?;

    let out_even = even.mul(&c, s)?.sub(&odd.mul(&sn, s)?, s)?;
    let out_odd = even.mul(&sn, s)?.add(&odd.mul(&c, s)?, s)?;
    concat(
        &[
            &out_even.reshape(&[b, h, n, d / 2, 1], s)?,
            &out_odd.reshape(&[b, h, n, d / 2, 1], s)?,
        ],
        4,
        s,
    )?
    .reshape(&[b, h, n, d], s)
}

/// `[b, n, dim]` to `[b, heads, n, head_dim]`.
fn to_heads(x: &Array, heads: usize, hd: usize, s: &Stream) -> Result<Array> {
    let [b, n, _] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: wan heads {:?}", x.shape())));
    };
    x.reshape(&[b, n, heads, hd], s)?
        .transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)
}

fn merge_heads(x: &Array, s: &Stream) -> Result<Array> {
    let [b, h, n, d] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: wan merge {:?}", x.shape())));
    };
    x.transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[b, n, h * d], s)
}

/// One attention, self or cross.
///
/// `context` is `None` for self-attention. `rope` is `None` for the cross
/// attention, which carries no positions at all — the text is unordered as far
/// as this model is concerned.
#[allow(clippy::too_many_arguments)]
fn attention(
    x: &Array,
    context: Option<&Array>,
    rope: Option<(&Array, &Array)>,
    cfg: &WanConfig,
    w: &impl WeightSource,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let (heads, hd) = (cfg.num_heads, cfg.head_dim);
    let p = |n: &str| format!("{prefix}.{n}");
    let proj = |src: &Array, name: &str| -> Result<Array> {
        w.linear(
            src,
            &p(&format!("{name}.weight")),
            w.optional(&p(&format!("{name}.bias"))),
            s,
        )
    };
    let kv_src = context.unwrap_or(x);

    // **Normed across the whole width, before the head split.** The weights
    // are `[dim]`, not `[head_dim]`.
    let q = proj(x, "to_q")?.rms_norm(Some(w.dense(&p("norm_q.weight"))?), cfg.eps, s)?;
    let k = proj(kv_src, "to_k")?.rms_norm(Some(w.dense(&p("norm_k.weight"))?), cfg.eps, s)?;
    let v = proj(kv_src, "to_v")?;

    let mut q = to_heads(&q, heads, hd, s)?;
    let mut k = to_heads(&k, heads, hd, s)?;
    let v = to_heads(&v, heads, hd, s)?;
    if let Some((cos, sin)) = rope {
        q = apply_rope(&q, cos, sin, s)?;
        k = apply_rope(&k, cos, sin, s)?;
    }

    let attended = merge_heads(&q.sdpa(&k, &v, 1.0 / (hd as f32).sqrt(), s)?, s)?;
    w.linear(
        &attended,
        &p("to_out.0.weight"),
        w.optional(&p("to_out.0.bias")),
        s,
    )
}

/// The `net.0.proj` / `net.2` feed-forward, approximate GELU.
fn feed_forward(x: &Array, w: &impl WeightSource, prefix: &str, s: &Stream) -> Result<Array> {
    let h = w
        .linear(
            x,
            &format!("{prefix}.net.0.proj.weight"),
            w.optional(&format!("{prefix}.net.0.proj.bias")),
            s,
        )?
        .gelu_approx(s)?;
    w.linear(
        &h,
        &format!("{prefix}.net.2.weight"),
        w.optional(&format!("{prefix}.net.2.bias")),
        s,
    )
}

/// One block: modulated self-attention, cross-attention, modulated MLP.
#[allow(clippy::too_many_arguments)]
fn block(
    x: &Array,
    context: &Array,
    temb: &Array,
    cos: &Array,
    sin: &Array,
    index: usize,
    cfg: &WanConfig,
    w: &impl WeightSource,
    s: &Stream,
) -> Result<Array> {
    let path = format!("blocks.{index}");
    let dim = cfg.dim();

    // **The learned table is added to the timestep projection**, not
    // multiplied and not replacing it. `[1, 6, dim]` against `[b, 6, dim]`.
    let m = w
        .dense(&format!("{path}.scale_shift_table"))?
        .reshape(&[1, 6, dim], s)?
        .add(temb, s)?;
    let take =
        |i: usize| -> Result<Array> { m.narrow(1, i, 1, s)?.reshape(&[m.shape()[0], 1, dim], s) };
    let (shift, scale, gate) = (take(0)?, take(1)?, take(2)?);
    let (c_shift, c_scale, c_gate) = (take(3)?, take(4)?, take(5)?);

    // norm1: affine-free.
    let h = x
        .layer_norm(None, None, cfg.eps, s)?
        .mul(&scale.add(&Array::scalar_f32(1.0)?, s)?, s)?
        .add(&shift, s)?;
    let attn = attention(
        &h,
        None,
        Some((cos, sin)),
        cfg,
        w,
        &format!("{path}.attn1"),
        s,
    )?;
    let x = x.add(&attn.mul(&gate, s)?, s)?;

    // norm2: **has weight and bias** — the only affine norm in the block.
    let h = x.layer_norm(
        Some(w.dense(&format!("{path}.norm2.weight"))?),
        Some(w.dense(&format!("{path}.norm2.bias"))?),
        cfg.eps,
        s,
    )?;
    // Cross-attention: no rotary positions, and **ungated** — its output goes
    // straight into the residual.
    let cross = attention(&h, Some(context), None, cfg, w, &format!("{path}.attn2"), s)?;
    let x = x.add(&cross, s)?;

    // norm3: affine-free.
    let h = x
        .layer_norm(None, None, cfg.eps, s)?
        .mul(&c_scale.add(&Array::scalar_f32(1.0)?, s)?, s)?
        .add(&c_shift, s)?;
    let ff = feed_forward(&h, w, &format!("{path}.ffn"), s)?;
    x.add(&ff.mul(&c_gate, s)?, s)
}

/// Sinusoidal timestep features, cosine half first.
fn timestep_features(t: &Array, channels: usize, s: &Stream) -> Result<Array> {
    let half = channels / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Array::from_slice_f32(&freqs, &[1, half])?;
    let args = t.reshape(&[t.shape()[0], 1], s)?.matmul(&freqs, s)?;
    concat(&[&args.cos(s)?, &args.sin(s)?], 1, s)
}

/// Patchify `[b, c, f, h, w]` into `[b, tokens, pt*ph*pw*c]`.
///
/// The 3D patch embedding is a convolution whose stride equals its kernel, so
/// this plus a linear is the same arithmetic. The permutation puts **channel
/// first** inside a patch, because the convolution kernel is stored
/// `(out, in, pt, ph, pw)` and flattens in that order.
pub fn patchify(x: &Array, patch: (usize, usize, usize), s: &Stream) -> Result<Array> {
    let [b, c, f, h, w] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: wan patchify {:?}", x.shape())));
    };
    let (pt, ph, pw) = patch;
    if f % pt != 0 || h % ph != 0 || w % pw != 0 {
        return Err(Error::Msg(format!(
            "mlx: a {f}x{h}x{w} latent does not divide into {pt}x{ph}x{pw} patches"
        )));
    }
    let (ft, ht, wt) = (f / pt, h / ph, w / pw);
    x.reshape(&[b, c, ft, pt, ht, ph, wt, pw], s)?
        .transpose(&[0, 2, 4, 6, 1, 3, 5, 7], s)?
        .contiguous(s)?
        .reshape(&[b, ft * ht * wt, c * pt * ph * pw], s)
}

/// The inverse of [`patchify`].
pub fn unpatchify(
    x: &Array,
    frames: usize,
    h: usize,
    w: usize,
    patch: (usize, usize, usize),
    channels: usize,
    s: &Stream,
) -> Result<Array> {
    let (pt, ph, pw) = patch;
    let (ft, ht, wt) = (frames / pt, h / ph, w / pw);
    let b = x.shape()[0];
    x.reshape(&[b, ft, ht, wt, pt, ph, pw, channels], s)?
        .transpose(&[0, 7, 1, 4, 2, 5, 3, 6], s)?
        .contiguous(s)?
        .reshape(&[b, channels, frames, h, w], s)
}

/// The velocity Wan predicts, `[b, c, f, h, w]` in and out.
pub fn forward(
    latent: &Array,
    text: &Array,
    timestep: &Array,
    cfg: &WanConfig,
    w: &impl WeightSource,
    s: &Stream,
) -> Result<Array> {
    let [b, _, frames, h, wd] = latent.shape()[..] else {
        return Err(Error::Msg(format!("mlx: wan latent {:?}", latent.shape())));
    };
    let (pt, ph, pw) = cfg.patch_size;
    let (ft, ht, wt) = (frames / pt, h / ph, wd / pw);
    let dim = cfg.dim();

    // The patch embedding, as a patchify plus the convolution kernel read as
    // a linear.
    let patches = patchify(latent, cfg.patch_size, s)?;
    let kernel = w.dense("patch_embedding.weight")?;
    let flat = kernel.shape();
    let kernel = kernel.reshape(&[flat[0], flat[1..].iter().product()], s)?;
    let mut x = super::linear(&patches, &kernel, w.optional("patch_embedding.bias"), s)?;

    // Conditioning: a sinusoid, an embedder, then a 6-way projection shared by
    // every block — the per-block part is each block's own table.
    let t_feat = timestep_features(timestep, cfg.freq_dim, s)?;
    let temb = w.linear(
        &w.linear(
            &t_feat,
            "condition_embedder.time_embedder.linear_1.weight",
            w.optional("condition_embedder.time_embedder.linear_1.bias"),
            s,
        )?
        .silu(s)?,
        "condition_embedder.time_embedder.linear_2.weight",
        w.optional("condition_embedder.time_embedder.linear_2.bias"),
        s,
    )?;
    let projected = w
        .linear(
            &temb.silu(s)?,
            "condition_embedder.time_proj.weight",
            w.optional("condition_embedder.time_proj.bias"),
            s,
        )?
        .reshape(&[b, 6, dim], s)?;

    let context = w.linear(
        &w.linear(
            text,
            "condition_embedder.text_embedder.linear_1.weight",
            w.optional("condition_embedder.text_embedder.linear_1.bias"),
            s,
        )?
        .gelu_approx(s)?,
        "condition_embedder.text_embedder.linear_2.weight",
        w.optional("condition_embedder.text_embedder.linear_2.bias"),
        s,
    )?;

    let (cos, sin) = rope_tables(ft, ht, wt, cfg, s)?;
    for i in 0..cfg.layers {
        x = block(&x, &context, &projected, &cos, &sin, i, cfg, w, s)?;
    }

    // The output head's table is `[1, 2, dim]` and is added to the *unprojected*
    // timestep embedding, not the six-way one.
    let m = w
        .dense("scale_shift_table")?
        .reshape(&[1, 2, dim], s)?
        .add(&temb.reshape(&[b, 1, dim], s)?, s)?;
    let shift = m.narrow(1, 0, 1, s)?.reshape(&[b, 1, dim], s)?;
    let scale = m.narrow(1, 1, 1, s)?.reshape(&[b, 1, dim], s)?;
    let out = x
        .layer_norm(None, None, cfg.eps, s)?
        .mul(&scale.add(&Array::scalar_f32(1.0)?, s)?, s)?
        .add(&shift, s)?;
    let out = w.linear(&out, "proj_out.weight", w.optional("proj_out.bias"), s)?;

    unpatchify(&out, frames, h, wd, cfg.patch_size, cfg.out_channels, s)
}
