//! Qwen-Image's transformer.
//!
//! SD 3.5's silhouette — 60 joint blocks over two streams, six-way modulation
//! per stream per block — with three things that are its own:
//!
//! **The spatial positions are centred.** A 4-row patch grid is rotated at
//! `[-2, -1, 0, 1]`, not `[0, 1, 2, 3]`; the rotary table is built from
//! `arange(4096)` *and* a mirrored negative half, and the grid takes a slice
//! straddling zero. The text stream then starts past the image's largest
//! index. Positions running `0..h` instead produce a coherent image with its
//! geometry shifted, which is not an error anywhere.
//!
//! **The rotation is applied per stream, before the concatenation.** The image
//! gets its own frequencies and the text its own, and only then are the two
//! joined for the attention — where FLUX.2 concatenates first and rotates the
//! whole sequence against one table.
//!
//! **Text first in the joint attention**, image first out of it. The block
//! returns the streams separately, so the split has to undo the order the
//! concatenation used.
//!
//! Its text encoder is Qwen2.5-VL — `LlmConfig::qwen2_5_vl_7b()`, and
//! `joint_attention_dim` is 3584 because that is the encoder's width.

use sd_tensor::mlx::{concat, Array, Stream};
use sd_tensor::{Error, Result};

use super::quantized::WeightSource;

/// Qwen-Image transformer geometry.
#[derive(Debug, Clone)]
pub struct QwenImageConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    /// The text encoder's width. 3584 — Qwen2.5-VL.
    pub joint_attention_dim: usize,
    /// Per-axis head-dimension split, `(frame, height, width)`.
    pub axes_dims: Vec<usize>,
    pub theta: f64,
    pub eps: f32,
}

impl QwenImageConfig {
    /// The published `Qwen/Qwen-Image`.
    pub fn base() -> Self {
        Self {
            num_heads: 24,
            head_dim: 128,
            layers: 60,
            in_channels: 64,
            out_channels: 16,
            joint_attention_dim: 3584,
            axes_dims: vec![16, 56, 56],
            theta: 10_000.0,
            eps: 1e-6,
        }
    }

    pub fn dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

/// Rotary tables for the image grid and the text run.
///
/// Returns `((img_cos, img_sin), (txt_cos, txt_sin))`, each `[1, 1, seq, d/2]`.
///
/// **The spatial axes are centred on zero.** For a height of `h` the positions
/// run from `-(h - h/2)` to `h/2 - 1` — so 4 rows give `[-2, -1, 0, 1]` and 3
/// columns give `[-2, -1, 0]`. The frame axis is *not* centred; it counts up
/// from 0. The text stream starts at `max(h/2, w/2)` and counts up from there,
/// which is what keeps it clear of the image's positive half.
pub fn rope_tables(
    frames: usize,
    h: usize,
    w: usize,
    txt_len: usize,
    cfg: &QwenImageConfig,
    s: &Stream,
) -> Result<((Array, Array), (Array, Array))> {
    if cfg.axes_dims.len() != 3 {
        return Err(Error::Msg(format!(
            "mlx: qwen-image wants three rotary axes, got {}",
            cfg.axes_dims.len()
        )));
    }
    let halves: Vec<usize> = cfg.axes_dims.iter().map(|d| d / 2).collect();
    let total: usize = halves.iter().sum();

    // One frequency per axis and slot, shared by every position on that axis.
    let omega = |axis: usize, i: usize| -> f64 {
        1.0 / cfg.theta.powf(2.0 * i as f64 / cfg.axes_dims[axis] as f64)
    };

    // Centred coordinates: `-(n - n/2) ..= n/2 - 1`.
    let centred = |n: usize| -> Vec<i64> {
        let lo = -((n - n / 2) as i64);
        (0..n).map(|i| lo + i as i64).collect()
    };
    let (rows, cols) = (centred(h), centred(w));

    let seq = frames * h * w;
    let mut cos = vec![0.0f32; seq * total];
    let mut sin = vec![0.0f32; seq * total];
    for f in 0..frames {
        for (yi, &y) in rows.iter().enumerate() {
            for (xi, &x) in cols.iter().enumerate() {
                let token = (f * h + yi) * w + xi;
                let mut off = 0usize;
                for (axis, pos) in [(0usize, f as i64), (1, y), (2, x)] {
                    for i in 0..halves[axis] {
                        let angle = pos as f64 * omega(axis, i);
                        cos[token * total + off + i] = angle.cos() as f32;
                        sin[token * total + off + i] = angle.sin() as f32;
                    }
                    off += halves[axis];
                }
            }
        }
    }

    // The text run starts past the image's largest *positive* index, so the
    // two streams never share a coordinate on any axis.
    let start = (h / 2).max(w / 2) as i64;
    let mut tcos = vec![0.0f32; txt_len * total];
    let mut tsin = vec![0.0f32; txt_len * total];
    for t in 0..txt_len {
        let pos = start + t as i64;
        let mut off = 0usize;
        // **Every axis takes the same coordinate here.** The text stream has
        // no geometry, so its one position is written into all three axes
        // rather than into a single one.
        for (axis, &half) in halves.iter().enumerate() {
            for i in 0..half {
                let angle = pos as f64 * omega(axis, i);
                tcos[t * total + off + i] = angle.cos() as f32;
                tsin[t * total + off + i] = angle.sin() as f32;
            }
            off += half;
        }
    }

    let wrap = |v: &[f32], n: usize| -> Result<Array> {
        Array::from_slice_f32(v, &[1, 1, n, total])?.contiguous(s)
    };
    Ok((
        (wrap(&cos, seq)?, wrap(&sin, seq)?),
        (wrap(&tcos, txt_len)?, wrap(&tsin, txt_len)?),
    ))
}

/// Interleaved rotation of `[b, h, seq, head_dim]` by half-width tables.
fn apply_rope(x: &Array, cos: &Array, sin: &Array, s: &Stream) -> Result<Array> {
    let [b, h, n, d] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: qwen rope {:?}", x.shape())));
    };
    let half = d / 2;
    let pairs = x.reshape(&[b, h, n, half, 2], s)?;
    let even = pairs.narrow(4, 0, 1, s)?.reshape(&[b, h, n, half], s)?;
    let odd = pairs.narrow(4, 1, 1, s)?.reshape(&[b, h, n, half], s)?;
    let out_even = even.mul(cos, s)?.sub(&odd.mul(sin, s)?, s)?;
    let out_odd = even.mul(sin, s)?.add(&odd.mul(cos, s)?, s)?;
    concat(
        &[
            &out_even.reshape(&[b, h, n, half, 1], s)?,
            &out_odd.reshape(&[b, h, n, half, 1], s)?,
        ],
        4,
        s,
    )?
    .reshape(&[b, h, n, d], s)
}

/// `norm(x) * (1 + scale) + shift`, with an affine-free LayerNorm.
fn modulate(x: &Array, shift: &Array, scale: &Array, eps: f32, s: &Stream) -> Result<Array> {
    x.layer_norm(None, None, eps, s)?
        .mul(&scale.add(&Array::scalar_f32(1.0)?, s)?, s)?
        .add(shift, s)
}

/// The `net.0.proj` / `net.2` feed-forward, with an approximate GELU.
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

/// Six modulation vectors from the timestep embedding.
///
/// `img_mod` is `Sequential(SiLU, Linear)`, so the weight is at index **1**
/// and the activation happens before it.
fn modulation(
    temb: &Array,
    dim: usize,
    w: &impl WeightSource,
    prefix: &str,
    s: &Stream,
) -> Result<Vec<Array>> {
    let out = w.linear(
        &temb.silu(s)?,
        &format!("{prefix}.1.weight"),
        w.optional(&format!("{prefix}.1.bias")),
        s,
    )?;
    let last = out.shape().len() - 1;
    (0..6)
        .map(|i| {
            out.narrow(last, i * dim, dim, s)?
                .reshape(&[out.shape()[0], 1, dim], s)
        })
        .collect()
}

/// One joint block.
#[allow(clippy::too_many_arguments)]
fn block(
    img: &Array,
    txt: &Array,
    img_rope: (&Array, &Array),
    txt_rope: (&Array, &Array),
    temb: &Array,
    index: usize,
    cfg: &QwenImageConfig,
    w: &impl WeightSource,
    s: &Stream,
) -> Result<(Array, Array)> {
    let path = format!("transformer_blocks.{index}");
    let a = |n: &str| format!("{path}.attn.{n}");
    let (heads, hd, dim) = (cfg.num_heads, cfg.head_dim, cfg.dim());

    let im = modulation(temb, dim, w, &format!("{path}.img_mod"), s)?;
    let tm = modulation(temb, dim, w, &format!("{path}.txt_mod"), s)?;

    // `chunk(2)` then `chunk(3)`: the first three drive the attention, the
    // second three the MLP, and within each it is (shift, scale, gate).
    let img_n = modulate(img, &im[0], &im[1], cfg.eps, s)?;
    let txt_n = modulate(txt, &tm[0], &tm[1], cfg.eps, s)?;

    let split = |t: Array, n: usize| -> Result<Array> {
        t.reshape(&[1, n, heads, hd], s)?
            .transpose(&[0, 2, 1, 3], s)?
            .contiguous(s)
    };
    let img_len = img.shape()[1];
    let txt_len = txt.shape()[1];
    let proj = |x: &Array, name: &str, n: usize| -> Result<Array> {
        split(
            w.linear(
                x,
                &a(&format!("{name}.weight")),
                w.optional(&a(&format!("{name}.bias"))),
                s,
            )?,
            n,
        )
    };
    let qn = |x: &Array, name: &str| -> Result<Array> {
        x.rms_norm(Some(w.dense(&a(&format!("{name}.weight")))?), cfg.eps, s)
    };

    // **Rotated per stream, before the concatenation**, each against its own
    // table.
    let img_q = apply_rope(
        &qn(&proj(&img_n, "to_q", img_len)?, "norm_q")?,
        img_rope.0,
        img_rope.1,
        s,
    )?;
    let img_k = apply_rope(
        &qn(&proj(&img_n, "to_k", img_len)?, "norm_k")?,
        img_rope.0,
        img_rope.1,
        s,
    )?;
    let img_v = proj(&img_n, "to_v", img_len)?;
    let txt_q = apply_rope(
        &qn(&proj(&txt_n, "add_q_proj", txt_len)?, "norm_added_q")?,
        txt_rope.0,
        txt_rope.1,
        s,
    )?;
    let txt_k = apply_rope(
        &qn(&proj(&txt_n, "add_k_proj", txt_len)?, "norm_added_k")?,
        txt_rope.0,
        txt_rope.1,
        s,
    )?;
    let txt_v = proj(&txt_n, "add_v_proj", txt_len)?;

    // **Text first in**, image first out.
    let q = concat(&[&txt_q, &img_q], 2, s)?;
    let k = concat(&[&txt_k, &img_k], 2, s)?;
    let v = concat(&[&txt_v, &img_v], 2, s)?;
    let attended = q
        .sdpa(&k, &v, 1.0 / (hd as f32).sqrt(), s)?
        .transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[1, txt_len + img_len, heads * hd], s)?;
    let txt_attn = attended.narrow(1, 0, txt_len, s)?;
    let img_attn = attended.narrow(1, txt_len, img_len, s)?;

    let img = img.add(
        &w.linear(
            &img_attn,
            &a("to_out.0.weight"),
            w.optional(&a("to_out.0.bias")),
            s,
        )?
        .mul(&im[2], s)?,
        s,
    )?;
    let img = img.add(
        &feed_forward(
            &modulate(&img, &im[3], &im[4], cfg.eps, s)?,
            w,
            &format!("{path}.img_mlp"),
            s,
        )?
        .mul(&im[5], s)?,
        s,
    )?;

    let txt = txt.add(
        &w.linear(
            &txt_attn,
            &a("to_add_out.weight"),
            w.optional(&a("to_add_out.bias")),
            s,
        )?
        .mul(&tm[2], s)?,
        s,
    )?;
    let txt = txt.add(
        &feed_forward(
            &modulate(&txt, &tm[3], &tm[4], cfg.eps, s)?,
            w,
            &format!("{path}.txt_mlp"),
            s,
        )?
        .mul(&tm[5], s)?,
        s,
    )?;
    Ok((img, txt))
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

/// The velocity Qwen-Image predicts.
///
/// `img` is `[1, frames*h*w, in_channels]` — already packed — and `txt` is
/// `[1, txt_len, joint_attention_dim]`, the encoder's hidden states.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    img: &Array,
    txt: &Array,
    timestep: &Array,
    frames: usize,
    h: usize,
    w_grid: usize,
    cfg: &QwenImageConfig,
    w: &impl WeightSource,
    s: &Stream,
) -> Result<Array> {
    let dim = cfg.dim();
    // The timestep arrives in [0, 1] and is embedded at 1000x, as Flux's is.
    let scaled = timestep.mul(&Array::scalar_f32(1000.0)?, s)?;
    let t_feat = timestep_features(&scaled, 256, s)?;
    let temb = w.linear(
        &w.linear(
            &t_feat,
            "time_text_embed.timestep_embedder.linear_1.weight",
            w.optional("time_text_embed.timestep_embedder.linear_1.bias"),
            s,
        )?
        .silu(s)?,
        "time_text_embed.timestep_embedder.linear_2.weight",
        w.optional("time_text_embed.timestep_embedder.linear_2.bias"),
        s,
    )?;

    let mut img = w.linear(img, "img_in.weight", w.optional("img_in.bias"), s)?;
    // **RMSNorm on the raw encoder output**, before the projection.
    let txt_n = txt.rms_norm(Some(w.dense("txt_norm.weight")?), cfg.eps, s)?;
    let mut txt = w.linear(&txt_n, "txt_in.weight", w.optional("txt_in.bias"), s)?;

    let txt_len = txt.shape()[1];
    let (img_rope, txt_rope) = rope_tables(frames, h, w_grid, txt_len, cfg, s)?;

    for i in 0..cfg.layers {
        let (a, b) = block(
            &img,
            &txt,
            (&img_rope.0, &img_rope.1),
            (&txt_rope.0, &txt_rope.1),
            &temb,
            i,
            cfg,
            w,
            s,
        )?;
        img = a;
        txt = b;
    }

    // The output head modulates with shift and scale — **scale first**, as
    // `AdaLayerNormContinuous` chunks it.
    let m = w.linear(
        &temb.silu(s)?,
        "norm_out.linear.weight",
        w.optional("norm_out.linear.bias"),
        s,
    )?;
    let last = m.shape().len() - 1;
    let scale = m.narrow(last, 0, dim, s)?.reshape(&[1, 1, dim], s)?;
    let shift = m.narrow(last, dim, dim, s)?.reshape(&[1, 1, dim], s)?;
    let out = modulate(&img, &shift, &scale, cfg.eps, s)?;
    w.linear(&out, "proj_out.weight", w.optional("proj_out.bias"), s)
}
