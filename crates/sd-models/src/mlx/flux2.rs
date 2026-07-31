//! FLUX.2's transformer.
//!
//! **Not Flux.1 with different numbers.** The two share a silhouette — double
//! blocks over two streams, then single blocks over one — and almost nothing
//! below it. Read out of `FLUX.2-klein-4B`'s own safetensors header before any
//! of this was written, because sd.cpp's config struct makes the differences
//! look like flags and they are not:
//!
//! | | Flux.1 | FLUX.2 |
//! |---|---|---|
//! | names | `double_blocks.0.img_attn.qkv` | diffusers' `transformer_blocks.0.attn.to_q` |
//! | q/k/v | one fused projection | three, in the double blocks |
//! | MLP | `mlp.0` → GELU → `mlp.2` | **gated SiLU**, `linear_in` twice as wide |
//! | modulation | one `img_mod.lin` **per block** | **three tensors for the whole model** |
//! | rotary axes | 3, `(t, h, w)` | **4** |
//! | pooled CLIP | `vector_in` | none — timestep and guidance only |
//! | biases | on qkv and mlp | **none anywhere** |
//! | patch size | 2 | 1, over 128 input channels |
//!
//! # The shared modulation
//!
//! `double_stream_modulation_img` emits `6 * hidden` and every one of the
//! double blocks uses **the same six vectors**; `single_stream_modulation`
//! emits `3 * hidden` for all the single blocks. Fifteen modulation vectors
//! for a 25-block model, against Flux.1's six per block.
//!
//! It looks like a bug when you first see the shapes — `[18432, 3072]` where a
//! per-block scheme would need one of these per block — which is why it is
//! stated here rather than left for someone to rediscover.
//!
//! # The text stream comes first
//!
//! `cat([encoder, image])`, in both the joint attention and the single blocks.
//! Flux.1 concatenates the same way, so this is one of the few things that
//! carries over; getting it backwards runs and attends every token to the
//! wrong positions.

use sd_tensor::mlx::{concat, Array, Stream};
use sd_tensor::{Error, Result};

use super::{get, linear, Weights};

/// FLUX.2 transformer geometry.
#[derive(Debug, Clone)]
pub struct Flux2Config {
    pub hidden_size: usize,
    pub num_heads: usize,
    /// Double (two-stream) blocks.
    pub depth: usize,
    /// Single (merged-stream) blocks.
    pub depth_single_blocks: usize,
    /// 3.0, where Flux.1 is 4.0 — and the MLP is gated, so `linear_in` emits
    /// twice this.
    pub mlp_ratio: f64,
    /// **Four** axes, 32 each. Flux.1 has three.
    pub axes_dim: Vec<usize>,
    /// 2000, where Flux.1 is 10,000.
    pub theta: f32,
    pub eps: f32,
    /// Width of the sinusoid before the timestep and guidance embedders.
    pub time_channels: usize,
    /// Whether the checkpoint carries a `guidance_embedder`. The distilled
    /// klein releases set `guidance_embeds: false` and ship only the timestep
    /// half; asking for the other tensor there is a missing-weight error.
    pub guidance_embed: bool,
}

impl Flux2Config {
    /// `FLUX.2-klein-4B`.
    pub fn klein_4b() -> Self {
        Self {
            hidden_size: 3072,
            num_heads: 24,
            depth: 5,
            depth_single_blocks: 20,
            mlp_ratio: 3.0,
            axes_dim: vec![32, 32, 32, 32],
            theta: 2000.0,
            eps: 1e-6,
            time_channels: 256,
            guidance_embed: false,
        }
    }

    /// `FLUX.2-dev`: 8 double and 48 single blocks at 6144 wide.
    pub fn dev() -> Self {
        Self {
            hidden_size: 6144,
            num_heads: 48,
            depth: 8,
            depth_single_blocks: 48,
            guidance_embed: true,
            ..Self::klein_4b()
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// The gated MLP's inner width. `linear_in` emits **twice** this, because
    /// half of it is the gate.
    pub fn mlp_hidden(&self) -> usize {
        (self.hidden_size as f64 * self.mlp_ratio) as usize
    }
}

/// `(t, y, x, extra)` coordinates for an `h x w` patch grid, `[h*w, 4]`.
///
/// **Four axes.** The fourth is zero for an ordinary image; it exists for the
/// same reason Flux.1's first does — distinguishing reference images from the
/// one being generated.
pub fn image_ids(h: usize, w: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(h * w * 4);
    for row in 0..h {
        for col in 0..w {
            v.push(0.0);
            v.push(row as f32);
            v.push(col as f32);
            v.push(0.0);
        }
    }
    v
}

/// Rotary tables for `[seq, 4]` coordinates.
///
/// `repeat_interleave_real` in diffusers, so each frequency is **duplicated
/// adjacently** to cover a `(x[2i], x[2i+1])` pair — the interleaved
/// convention, matching Flux.1 and not the split-half one the LLM encoder
/// uses. Built in f64 for the same reason: the low frequencies are the ones
/// that survive f32 badly and they encode long-range position.
pub fn rope_tables(
    ids: &[f32],
    seq: usize,
    axes_dim: &[usize],
    theta: f32,
    s: &Stream,
) -> Result<(Array, Array)> {
    let n_axes = axes_dim.len();
    if ids.len() != seq * n_axes {
        return Err(Error::Msg(format!(
            "mlx: flux2 got {} ids for {seq} tokens across {n_axes} axes",
            ids.len()
        )));
    }
    let total: usize = axes_dim.iter().sum();
    let mut cos = vec![0.0f32; seq * total];
    let mut sin = vec![0.0f32; seq * total];

    let mut offset = 0usize;
    for (axis, &dim) in axes_dim.iter().enumerate() {
        let half = dim / 2;
        for t in 0..seq {
            let pos = ids[t * n_axes + axis] as f64;
            for i in 0..half {
                let omega = 1.0 / (theta as f64).powf(2.0 * i as f64 / dim as f64);
                let angle = pos * omega;
                // Duplicated adjacently: entries 2i and 2i+1 share a frequency.
                let base = t * total + offset + 2 * i;
                cos[base] = angle.cos() as f32;
                cos[base + 1] = angle.cos() as f32;
                sin[base] = angle.sin() as f32;
                sin[base + 1] = angle.sin() as f32;
            }
        }
        offset += dim;
    }
    Ok((
        Array::from_slice_f32(&cos, &[1, 1, seq, total])?.contiguous(s)?,
        Array::from_slice_f32(&sin, &[1, 1, seq, total])?.contiguous(s)?,
    ))
}

/// Interleaved rotation over `[b, h, seq, head_dim]`.
fn apply_rope(x: &Array, cos: &Array, sin: &Array, s: &Stream) -> Result<Array> {
    let [b, h, n, d] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: flux2 rope got {:?}", x.shape())));
    };
    // `[..., -x2, x1, -x4, x3, ...]`, which is what pairs (2i, 2i+1) under a
    // duplicated frequency table.
    let pairs = x.reshape(&[b, h, n, d / 2, 2], s)?;
    let even = pairs.narrow(4, 0, 1, s)?.reshape(&[b, h, n, d / 2], s)?;
    let odd = pairs.narrow(4, 1, 1, s)?.reshape(&[b, h, n, d / 2], s)?;
    let rotated = concat(
        &[
            &odd.mul(&Array::scalar_f32(-1.0)?, s)?
                .reshape(&[b, h, n, d / 2, 1], s)?,
            &even.reshape(&[b, h, n, d / 2, 1], s)?,
        ],
        4,
        s,
    )?
    .reshape(&[b, h, n, d], s)?;
    x.mul(cos, s)?.add(&rotated.mul(sin, s)?, s)
}

/// `silu(x1) * x2` over a projection that is twice the MLP width.
fn swiglu(x: &Array, s: &Stream) -> Result<Array> {
    let last = x.shape().len() - 1;
    let width = x.shape()[last];
    if width % 2 != 0 {
        return Err(Error::Msg(format!(
            "mlx: flux2 swiglu wants an even width, got {width}"
        )));
    }
    let half = width / 2;
    let gate = x.narrow(last, 0, half, s)?;
    let value = x.narrow(last, half, half, s)?;
    gate.silu(s)?.mul(&value, s)
}

/// The gated feed-forward: `linear_out(swiglu(linear_in(x)))`.
fn feed_forward(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let inner = linear(x, get(w, &format!("{prefix}.linear_in.weight"))?, None, s)?;
    linear(
        &swiglu(&inner, s)?,
        get(w, &format!("{prefix}.linear_out.weight"))?,
        None,
        s,
    )
}

/// `norm(x) * (1 + scale) + shift`, with an affine-free LayerNorm.
fn modulate(x: &Array, shift: &Array, scale: &Array, eps: f32, s: &Stream) -> Result<Array> {
    x.layer_norm(None, None, eps, s)?
        .mul(&scale.add(&Array::scalar_f32(1.0)?, s)?, s)?
        .add(shift, s)
}

/// Split a modulation projection into `sets` triples of `(shift, scale, gate)`.
fn modulation(
    temb: &Array,
    sets: usize,
    hidden: usize,
    w: &Weights,
    prefix: &str,
    s: &Stream,
) -> Result<Vec<Array>> {
    // **SiLU before the projection**, as everywhere else in this family.
    let out = linear(
        &temb.silu(s)?,
        get(w, &format!("{prefix}.linear.weight"))?,
        None,
        s,
    )?;
    let last = out.shape().len() - 1;
    let mut parts = Vec::with_capacity(3 * sets);
    for i in 0..3 * sets {
        // `[b, 1, hidden]` so it broadcasts across the sequence.
        parts.push(
            out.narrow(last, i * hidden, hidden, s)?
                .reshape(&[out.shape()[0], 1, hidden], s)?,
        );
    }
    Ok(parts)
}

/// Split a `[b, n, 3*heads*hd]` projection into q, k, v as `[b, heads, n, hd]`.
fn split_qkv(qkv: &Array, heads: usize, hd: usize, s: &Stream) -> Result<(Array, Array, Array)> {
    let [b, n, _] = qkv.shape()[..] else {
        return Err(Error::Msg(format!("mlx: flux2 qkv got {:?}", qkv.shape())));
    };
    let width = heads * hd;
    let take = |i: usize| -> Result<Array> {
        qkv.narrow(2, i * width, width, s)?
            .reshape(&[b, n, heads, hd], s)?
            .transpose(&[0, 2, 1, 3], s)?
            .contiguous(s)
    };
    Ok((take(0)?, take(1)?, take(2)?))
}

/// One head-split projection, `[b, n, heads*hd]` to `[b, heads, n, hd]`.
fn to_heads(x: &Array, heads: usize, hd: usize, s: &Stream) -> Result<Array> {
    let [b, n, _] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: flux2 heads got {:?}", x.shape())));
    };
    x.reshape(&[b, n, heads, hd], s)?
        .transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)
}

fn merge_heads(x: &Array, s: &Stream) -> Result<Array> {
    let [b, h, n, d] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: flux2 merge got {:?}", x.shape())));
    };
    x.transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[b, n, h * d], s)
}

/// One double block: two streams, joint attention, gated MLPs.
#[allow(clippy::too_many_arguments)]
fn double_block(
    img: &Array,
    txt: &Array,
    im: &[Array],
    tm: &[Array],
    cos: &Array,
    sin: &Array,
    index: usize,
    cfg: &Flux2Config,
    w: &Weights,
    s: &Stream,
) -> Result<(Array, Array)> {
    let path = format!("transformer_blocks.{index}");
    let a = |n: &str| format!("{path}.attn.{n}.weight");
    let (heads, hd) = (cfg.num_heads, cfg.head_dim());

    let img_n = modulate(img, &im[0], &im[1], cfg.eps, s)?;
    let txt_n = modulate(txt, &tm[0], &tm[1], cfg.eps, s)?;

    let proj = |x: &Array, name: &str| -> Result<Array> {
        to_heads(&linear(x, get(w, &a(name))?, None, s)?, heads, hd, s)
    };
    // **QK-norm before the concatenation**, per stream, with its own weights:
    // the image stream uses `norm_q`, the text stream `norm_added_q`. They are
    // the same shape.
    let qn = |x: &Array, name: &str| -> Result<Array> {
        x.rms_norm(Some(get(w, &a(name))?), cfg.eps, s)
    };

    let img_q = qn(&proj(&img_n, "to_q")?, "norm_q")?;
    let img_k = qn(&proj(&img_n, "to_k")?, "norm_k")?;
    let img_v = proj(&img_n, "to_v")?;
    let txt_q = qn(&proj(&txt_n, "add_q_proj")?, "norm_added_q")?;
    let txt_k = qn(&proj(&txt_n, "add_k_proj")?, "norm_added_k")?;
    let txt_v = proj(&txt_n, "add_v_proj")?;

    // **Text first.** The rotary tables are built for that order.
    let q = apply_rope(&concat(&[&txt_q, &img_q], 2, s)?, cos, sin, s)?;
    let k = apply_rope(&concat(&[&txt_k, &img_k], 2, s)?, cos, sin, s)?;
    let v = concat(&[&txt_v, &img_v], 2, s)?;

    let attended = merge_heads(&q.sdpa(&k, &v, 1.0 / (hd as f32).sqrt(), s)?, s)?;
    let txt_len = txt.shape()[1];
    let img_len = img.shape()[1];
    let txt_attn = attended.narrow(1, 0, txt_len, s)?;
    let img_attn = attended.narrow(1, txt_len, img_len, s)?;

    // Image stream: gated attention residual, then gated MLP.
    let img = img.add(
        &linear(&img_attn, get(w, &a("to_out.0"))?, None, s)?.mul(&im[2], s)?,
        s,
    )?;
    let img = img.add(
        &feed_forward(
            &modulate(&img, &im[3], &im[4], cfg.eps, s)?,
            w,
            &format!("{path}.ff"),
            s,
        )?
        .mul(&im[5], s)?,
        s,
    )?;

    let txt = txt.add(
        &linear(&txt_attn, get(w, &a("to_add_out"))?, None, s)?.mul(&tm[2], s)?,
        s,
    )?;
    let txt = txt.add(
        &feed_forward(
            &modulate(&txt, &tm[3], &tm[4], cfg.eps, s)?,
            w,
            &format!("{path}.ff_context"),
            s,
        )?
        .mul(&tm[5], s)?,
        s,
    )?;
    Ok((img, txt))
}

/// One single block: attention and MLP from **one fused projection**.
#[allow(clippy::too_many_arguments)]
fn single_block(
    x: &Array,
    m: &[Array],
    cos: &Array,
    sin: &Array,
    index: usize,
    cfg: &Flux2Config,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let path = format!("single_transformer_blocks.{index}.attn");
    let (heads, hd) = (cfg.num_heads, cfg.head_dim());
    let normed = modulate(x, &m[0], &m[1], cfg.eps, s)?;

    // `to_qkv_mlp_proj` emits `3*hidden` of qkv followed by `2*mlp_hidden` of
    // gated MLP — one projection, split by width rather than chunked evenly.
    let projected = linear(
        &normed,
        get(w, &format!("{path}.to_qkv_mlp_proj.weight"))?,
        None,
        s,
    )?;
    let qkv_width = 3 * heads * hd;
    let last = projected.shape().len() - 1;
    let qkv = projected.narrow(last, 0, qkv_width, s)?;
    let mlp = projected.narrow(last, qkv_width, projected.shape()[last] - qkv_width, s)?;

    let (q, k, v) = split_qkv(&qkv, heads, hd, s)?;
    let q = apply_rope(
        &q.rms_norm(Some(get(w, &format!("{path}.norm_q.weight"))?), cfg.eps, s)?,
        cos,
        sin,
        s,
    )?;
    let k = apply_rope(
        &k.rms_norm(Some(get(w, &format!("{path}.norm_k.weight"))?), cfg.eps, s)?,
        cos,
        sin,
        s,
    )?;

    let attended = merge_heads(&q.sdpa(&k, &v, 1.0 / (hd as f32).sqrt(), s)?, s)?;
    // Attention output and gated MLP output side by side, then one projection.
    let joined = concat(&[&attended, &swiglu(&mlp, s)?], 2, s)?;
    let out = linear(&joined, get(w, &format!("{path}.to_out.weight"))?, None, s)?;
    x.add(&out.mul(&m[2], s)?, s)
}

/// Sinusoidal timestep features, `flip_sin_to_cos` — cosine half first.
fn timestep_features(t: &Array, channels: usize, s: &Stream) -> Result<Array> {
    let half = channels / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Array::from_slice_f32(&freqs, &[1, half])?;
    let args = t.reshape(&[t.shape()[0], 1], s)?.matmul(&freqs, s)?;
    concat(&[&args.cos(s)?, &args.sin(s)?], 1, s)
}

/// A two-layer embedder: `linear_2(silu(linear_1(x)))`.
fn embedder(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let h = linear(x, get(w, &format!("{prefix}.linear_1.weight"))?, None, s)?;
    linear(
        &h.silu(s)?,
        get(w, &format!("{prefix}.linear_2.weight"))?,
        None,
        s,
    )
}

/// The velocity FLUX.2 predicts.
///
/// `img_ids` and `txt_ids` are flat `[seq * 4]` coordinate lists. `timestep`
/// and `guidance` are `[1]` in the model's own units — **already multiplied by
/// 1000**, which is the caller's job because the sampler works in `[0, 1]`.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    img: &Array,
    img_ids: &[f32],
    txt: &Array,
    txt_ids: &[f32],
    timestep: &Array,
    guidance: Option<&Array>,
    cfg: &Flux2Config,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let hidden = cfg.hidden_size;

    // Timestep and guidance are embedded separately and **summed**, not
    // concatenated — one `temb` drives every modulation in the model.
    let t_feat = timestep_features(timestep, cfg.time_channels, s)?;
    let mut temb = embedder(&t_feat, w, "time_guidance_embed.timestep_embedder", s)?;
    match (cfg.guidance_embed, guidance) {
        (true, Some(g)) => {
            let g_feat = timestep_features(g, cfg.time_channels, s)?;
            temb = temb.add(
                &embedder(&g_feat, w, "time_guidance_embed.guidance_embedder", s)?,
                s,
            )?;
        }
        (true, None) => {
            return Err(Error::Msg(
                "mlx: this FLUX.2 checkpoint has a guidance embedder and needs a guidance scale"
                    .into(),
            ))
        }
        (false, Some(_)) => {
            return Err(Error::Msg(
                "mlx: this FLUX.2 checkpoint is distilled and takes no guidance scale".into(),
            ))
        }
        (false, None) => {}
    }

    let mut img = linear(img, get(w, "x_embedder.weight")?, None, s)?;
    let mut txt = linear(txt, get(w, "context_embedder.weight")?, None, s)?;

    // **One modulation for the whole model**, not one per block.
    let mod_img = modulation(&temb, 2, hidden, w, "double_stream_modulation_img", s)?;
    let mod_txt = modulation(&temb, 2, hidden, w, "double_stream_modulation_txt", s)?;
    let mod_single = modulation(&temb, 1, hidden, w, "single_stream_modulation", s)?;

    // Text ids precede image ids, matching the concatenation inside every
    // block.
    let mut ids = txt_ids.to_vec();
    ids.extend_from_slice(img_ids);
    let axes = cfg.axes_dim.len();
    let seq = ids.len() / axes;
    let (cos, sin) = rope_tables(&ids, seq, &cfg.axes_dim, cfg.theta, s)?;

    for i in 0..cfg.depth {
        let (a, b) = double_block(&img, &txt, &mod_img, &mod_txt, &cos, &sin, i, cfg, w, s)?;
        img = a;
        txt = b;
    }

    let mut xs = concat(&[&txt, &img], 1, s)?;
    for i in 0..cfg.depth_single_blocks {
        xs = single_block(&xs, &mod_single, &cos, &sin, i, cfg, w, s)?;
    }
    let img_len = img.shape()[1];
    let xs = xs.narrow(1, txt.shape()[1], img_len, s)?;

    // The output head modulates from `temb` with shift and scale only —
    // nothing to gate at the end.
    let out_mod = linear(&temb.silu(s)?, get(w, "norm_out.linear.weight")?, None, s)?;
    let last = out_mod.shape().len() - 1;
    // **Scale first, then shift** — `AdaLayerNormContinuous` chunks in that
    // order, which is the *reverse* of `Flux2Modulation`'s `(shift, scale,
    // gate)`. Two conventions in one model, and swapping them costs about 10%
    // of the output with no error anywhere.
    let scale = out_mod
        .narrow(last, 0, hidden, s)?
        .reshape(&[out_mod.shape()[0], 1, hidden], s)?;
    let shift = out_mod
        .narrow(last, hidden, hidden, s)?
        .reshape(&[out_mod.shape()[0], 1, hidden], s)?;
    let xs = modulate(&xs, &shift, &scale, cfg.eps, s)?;
    linear(&xs, get(w, "proj_out.weight")?, None, s)
}
