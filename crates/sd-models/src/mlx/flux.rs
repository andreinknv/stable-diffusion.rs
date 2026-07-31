//! Flux's MMDiT transformer on MLX.
//!
//! Not a UNet: no downsampling, no skips, no convolution. The latent is cut
//! into 2x2 patches, flattened to tokens, and pushed through transformer blocks
//! at constant width. Position comes from explicit rotary embeddings rather
//! than convolutional locality.
//!
//! Two halves. **Double-stream** blocks keep image and text as separate
//! residual streams with their own weights, joining them only inside attention.
//! **Single-stream** blocks concatenate the two and run one shared stream, with
//! attention and the feed-forward fused into a single pair of matrices.
//!
//! Conditioning is modulation, not cross-attention: a vector built from the
//! timestep, the guidance scale and CLIP's pooled embedding is projected per
//! block into `(shift, scale, gate)` triples.
//!
//! Names follow the black-forest-labs checkpoint layout
//! (`double_blocks.0.img_attn.qkv`), not the diffusers renaming.

use sd_tensor::mlx::{concat, Array, Stream};
use sd_tensor::{Error, Result};

use super::{get, linear, Weights};

const TIME_EMBED_DIM: usize = 256;
/// Flux scales the timestep by this before embedding it. The SD UNet does not.
const TIME_FACTOR: f32 = 1000.0;
const EPS: f32 = 1e-6;

/// Flux transformer geometry.
#[derive(Debug, Clone)]
pub struct FluxConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
    pub depth_single_blocks: usize,
    pub mlp_ratio: f64,
    /// Head-dimension split across the `(t, h, w)` rotary axes. Must sum to
    /// `hidden_size / num_heads`.
    pub axes_dim: Vec<usize>,
    pub theta: f32,
    /// Whether the model takes a distilled guidance scale. dev and mini do;
    /// schnell does not.
    pub guidance_embed: bool,
}

impl FluxConfig {
    /// `TencentARC/flux-mini`: full Flux width, 5 double and 10 single blocks.
    pub fn mini() -> Self {
        Self {
            hidden_size: 3072,
            num_heads: 24,
            depth: 5,
            depth_single_blocks: 10,
            mlp_ratio: 4.0,
            axes_dim: vec![16, 56, 56],
            theta: 10_000.0,
            guidance_embed: true,
        }
    }

    /// `FLUX.1-dev`.
    pub fn dev() -> Self {
        Self {
            depth: 19,
            depth_single_blocks: 38,
            ..Self::mini()
        }
    }

    /// `FLUX.1-schnell`: dev's shape with no guidance embedding.
    pub fn schnell() -> Self {
        Self {
            guidance_embed: false,
            ..Self::dev()
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    fn mlp_hidden(&self) -> usize {
        (self.hidden_size as f64 * self.mlp_ratio) as usize
    }
}

/// `(t, h, w)` coordinates for an `h x w` patch grid, `[1, h*w, 3]`.
pub fn image_ids(h: usize, w: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(h * w * 3);
    for row in 0..h {
        for col in 0..w {
            v.push(0.0);
            v.push(row as f32);
            v.push(col as f32);
        }
    }
    v
}

/// Rotary tables for one run: `cos` and `sin`, each `[1, seq, head_dim/2]`.
///
/// Built on the host in f64 for the frequencies: the exponent spans several
/// orders of magnitude and f32 loses the low ones, which are exactly the
/// frequencies that encode long-range position.
pub struct Rope {
    pub cos: Array,
    pub sin: Array,
}

/// `(cos, sin)` for `[seq, 3]` integer coordinates.
pub fn embed_nd(ids: &[f32], seq: usize, axes_dim: &[usize], theta: f32) -> Result<Rope> {
    let n_axes = axes_dim.len();
    if ids.len() != seq * n_axes {
        return Err(Error::Msg(format!(
            "mlx: {} ids for {seq} tokens across {n_axes} axes",
            ids.len()
        )));
    }
    let half_total: usize = axes_dim.iter().map(|d| d / 2).sum();
    let mut cos = vec![0.0f32; seq * half_total];
    let mut sin = vec![0.0f32; seq * half_total];

    let mut offset = 0usize;
    for (axis, &dim) in axes_dim.iter().enumerate() {
        let half = dim / 2;
        for t in 0..seq {
            let pos = ids[t * n_axes + axis] as f64;
            for i in 0..half {
                let omega = 1.0 / (theta as f64).powf(2.0 * i as f64 / dim as f64);
                let angle = pos * omega;
                cos[t * half_total + offset + i] = angle.cos() as f32;
                sin[t * half_total + offset + i] = angle.sin() as f32;
            }
        }
        offset += half;
    }
    Ok(Rope {
        cos: Array::from_slice_f32(&cos, &[1, seq, half_total])?,
        sin: Array::from_slice_f32(&sin, &[1, seq, half_total])?,
    })
}

/// Interleaved rotary application: pairs `(x[2i], x[2i+1])` rotated by
/// `(cos[i], sin[i])`.
///
/// **Interleaved, not split-half.** The rotation is stored as an explicit 2x2
/// per frequency because that is what the reference does, and a transposed or
/// half-split rotation is still a rotation — it produces a coherent image with
/// the geometry subtly wrong.
pub fn rotate(x: &Array, pe: &Rope, s: &Stream) -> Result<Array> {
    let [b, h, n, d] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: rope got {:?}", x.shape())));
    };
    let half = d / 2;
    let pairs = x.reshape(&[b, h, n, half, 2], s)?;
    let even = pairs.narrow(4, 0, 1, s)?.reshape(&[b, h, n, half], s)?;
    let odd = pairs.narrow(4, 1, 1, s)?.reshape(&[b, h, n, half], s)?;

    // [1, n, half] -> [1, 1, n, half], broadcasting over heads.
    let cos = pe.cos.reshape(&[1, 1, n, half], s)?;
    let sin = pe.sin.reshape(&[1, 1, n, half], s)?;

    let out_even = even.mul(&cos, s)?.sub(&odd.mul(&sin, s)?, s)?;
    let out_odd = even.mul(&sin, s)?.add(&odd.mul(&cos, s)?, s)?;

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

/// LayerNorm with **no learned parameters at all**.
///
/// Every norm inside these blocks is `elementwise_affine=False` — the scale and
/// shift come from the modulation vector instead, which is the whole mechanism
/// by which Flux is conditioned.
fn norm_modulate(x: &Array, shift: &Array, scale: &Array, s: &Stream) -> Result<Array> {
    x.layer_norm(None, None, EPS, s)?
        .mul(&scale.add(&Array::scalar_f32(1.0)?, s)?, s)?
        .add(shift, s)
}

/// `(shift, scale, gate)` triples. Six values for a double block, three for a
/// single one. **SiLU before the projection, not after.**
fn modulation(
    vec: &Array,
    double: bool,
    w: &Weights,
    prefix: &str,
    s: &Stream,
) -> Result<Vec<Array>> {
    let out = linear(
        &vec.silu(s)?,
        get(w, &format!("{prefix}.lin.weight"))?,
        w.get(&format!("{prefix}.lin.bias")),
        s,
    )?;
    let [b, total] = out.shape()[..] else {
        return Err(Error::Msg(format!("mlx: modulation got {:?}", out.shape())));
    };
    let n = if double { 6 } else { 3 };
    let dim = total / n;
    let out = out.reshape(&[b, 1, total], s)?;
    (0..n).map(|i| out.narrow(2, i * dim, dim, s)).collect()
}

/// `[b, n, 3*heads*hd]` into three `[b, heads, n, hd]`.
fn split_qkv(qkv: &Array, heads: usize, hd: usize, s: &Stream) -> Result<(Array, Array, Array)> {
    let [b, n, _] = qkv.shape()[..] else {
        return Err(Error::Msg(format!("mlx: qkv got {:?}", qkv.shape())));
    };
    let t = qkv
        .reshape(&[b, n, 3, heads, hd], s)?
        .transpose(&[2, 0, 3, 1, 4], s)?;
    let take = |i: usize| -> Result<Array> { t.narrow(0, i, 1, s)?.reshape(&[b, heads, n, hd], s) };
    Ok((take(0)?, take(1)?, take(2)?))
}

fn merge_heads(x: &Array, s: &Stream) -> Result<Array> {
    let [b, h, n, d] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: merge got {:?}", x.shape())));
    };
    x.transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[b, n, h * d], s)
}

/// RMSNorm over the head dimension, applied to queries and keys.
fn qk_norm(x: &Array, weight: &Array, s: &Stream) -> Result<Array> {
    let dims = x.shape();
    let last = dims.len() - 1;
    let mean_sq = x.mul(x, s)?.mean(&[last], true, s)?;
    x.mul(&mean_sq.add(&Array::scalar_f32(EPS)?, s)?.rsqrt(s)?, s)?
        .mul(weight, s)
}

/// A double-stream block: separate image and text streams, joined in attention.
#[allow(clippy::too_many_arguments)]
fn double_block(
    img: &Array,
    txt: &Array,
    vec: &Array,
    pe: &Rope,
    index: usize,
    cfg: &FluxConfig,
    w: &Weights,
    s: &Stream,
) -> Result<(Array, Array)> {
    let path = format!("double_blocks.{index}");
    let (heads, hd) = (cfg.num_heads, cfg.head_dim());
    let scale = 1.0 / (hd as f32).sqrt();

    let im = modulation(vec, true, w, &format!("{path}.img_mod"), s)?;
    let tm = modulation(vec, true, w, &format!("{path}.txt_mod"), s)?;

    let stream_qkv = |x: &Array, m: &[Array], tag: &str| -> Result<(Array, Array, Array)> {
        let normed = norm_modulate(x, &m[0], &m[1], s)?;
        let qkv = linear(
            &normed,
            get(w, &format!("{path}.{tag}_attn.qkv.weight"))?,
            w.get(&format!("{path}.{tag}_attn.qkv.bias")),
            s,
        )?;
        let (q, k, v) = split_qkv(&qkv, heads, hd, s)?;
        Ok((
            qk_norm(
                &q,
                get(w, &format!("{path}.{tag}_attn.norm.query_norm.scale"))?,
                s,
            )?,
            qk_norm(
                &k,
                get(w, &format!("{path}.{tag}_attn.norm.key_norm.scale"))?,
                s,
            )?,
            v,
        ))
    };
    let (img_q, img_k, img_v) = stream_qkv(img, &im, "img")?;
    let (txt_q, txt_k, txt_v) = stream_qkv(txt, &tm, "txt")?;

    // Text first, then image — the same order the position ids were
    // concatenated in, which is what makes the rotary embedding line up.
    let q = rotate(&concat(&[&txt_q, &img_q], 2, s)?, pe, s)?;
    let k = rotate(&concat(&[&txt_k, &img_k], 2, s)?, pe, s)?;
    let v = concat(&[&txt_v, &img_v], 2, s)?;
    let attn = merge_heads(&q.sdpa(&k, &v, scale, s)?, s)?;

    let txt_len = txt.shape()[1];
    let total = attn.shape()[1];
    let txt_attn = attn.narrow(1, 0, txt_len, s)?;
    let img_attn = attn.narrow(1, txt_len, total - txt_len, s)?;

    let finish = |x: &Array, a: &Array, m: &[Array], tag: &str| -> Result<Array> {
        let x = x.add(
            &linear(
                a,
                get(w, &format!("{path}.{tag}_attn.proj.weight"))?,
                w.get(&format!("{path}.{tag}_attn.proj.bias")),
                s,
            )?
            .mul(&m[2], s)?,
            s,
        )?;
        let ff_in = norm_modulate(&x, &m[3], &m[4], s)?;
        let ff = linear(
            &ff_in,
            get(w, &format!("{path}.{tag}_mlp.0.weight"))?,
            w.get(&format!("{path}.{tag}_mlp.0.bias")),
            s,
        )?
        .gelu_approx(s)?;
        let ff = linear(
            &ff,
            get(w, &format!("{path}.{tag}_mlp.2.weight"))?,
            w.get(&format!("{path}.{tag}_mlp.2.bias")),
            s,
        )?;
        x.add(&ff.mul(&m[5], s)?, s)
    };
    Ok((
        finish(img, &img_attn, &im, "img")?,
        finish(txt, &txt_attn, &tm, "txt")?,
    ))
}

/// A single-stream block: one sequence, attention and feed-forward fused into
/// `linear1` / `linear2`.
fn single_block(
    x: &Array,
    vec: &Array,
    pe: &Rope,
    index: usize,
    cfg: &FluxConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let path = format!("single_blocks.{index}");
    let (heads, hd) = (cfg.num_heads, cfg.head_dim());
    let h = cfg.hidden_size;

    let m = modulation(vec, false, w, &format!("{path}.modulation"), s)?;
    let normed = norm_modulate(x, &m[0], &m[1], s)?;

    let projected = linear(
        &normed,
        get(w, &format!("{path}.linear1.weight"))?,
        w.get(&format!("{path}.linear1.bias")),
        s,
    )?;
    let last = projected.shape().len() - 1;
    let qkv = projected.narrow(last, 0, 3 * h, s)?;
    let mlp = projected.narrow(last, 3 * h, cfg.mlp_hidden(), s)?;

    let (q, k, v) = split_qkv(&qkv.contiguous(s)?, heads, hd, s)?;
    let q = rotate(
        &qk_norm(&q, get(w, &format!("{path}.norm.query_norm.scale"))?, s)?,
        pe,
        s,
    )?;
    let k = rotate(
        &qk_norm(&k, get(w, &format!("{path}.norm.key_norm.scale"))?, s)?,
        pe,
        s,
    )?;
    let attn = merge_heads(&q.sdpa(&k, &v, 1.0 / (hd as f32).sqrt(), s)?, s)?;

    let joined = concat(&[&attn, &mlp.contiguous(s)?.gelu_approx(s)?], 2, s)?;
    let out = linear(
        &joined,
        get(w, &format!("{path}.linear2.weight"))?,
        w.get(&format!("{path}.linear2.bias")),
        s,
    )?;
    x.add(&out.mul(&m[2], s)?, s)
}

/// Flux's timestep embedding.
///
/// **Scaled by 1000 first**, which the SD UNet's is not, and cosine before
/// sine.
fn timestep_embedding(t: &Array, dim: usize, theta: f32, s: &Stream) -> Result<Array> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(theta as f64).ln() * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Array::from_slice_f32(&freqs, &[1, half])?;
    let scaled = t
        .reshape(&[t.elem_count(), 1], s)?
        .mul(&Array::scalar_f32(TIME_FACTOR)?, s)?;
    let args = scaled.matmul(&freqs, s)?;
    concat(&[&args.cos(s)?, &args.sin(s)?], 1, s)
}

/// Two-layer MLP with SiLU between, used for every conditioning input.
fn mlp_embedder(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let p = |n: &str| format!("{prefix}.{n}");
    let h = linear(
        x,
        get(w, &p("in_layer.weight"))?,
        w.get(&p("in_layer.bias")),
        s,
    )?
    .silu(s)?;
    linear(
        &h,
        get(w, &p("out_layer.weight"))?,
        w.get(&p("out_layer.bias")),
        s,
    )
}

/// The Flux transformer.
///
/// `img` is `[b, img_tokens, in_channels]` already packed; `txt` is T5's
/// sequence. `guidance` is required exactly when the checkpoint was distilled
/// on one.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    img: &Array,
    img_ids: &[f32],
    txt: &Array,
    timestep: &Array,
    pooled: &Array,
    guidance: Option<&Array>,
    cfg: &FluxConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let mut img = linear(img, get(w, "img_in.weight")?, w.get("img_in.bias"), s)?;
    let mut txt = linear(txt, get(w, "txt_in.weight")?, w.get("txt_in.bias"), s)?;

    let mut vec = mlp_embedder(
        &timestep_embedding(timestep, TIME_EMBED_DIM, cfg.theta, s)?,
        w,
        "time_in",
        s,
    )?;
    match (cfg.guidance_embed, guidance) {
        (true, Some(g)) => {
            vec = vec.add(
                &mlp_embedder(
                    &timestep_embedding(g, TIME_EMBED_DIM, cfg.theta, s)?,
                    w,
                    "guidance_in",
                    s,
                )?,
                s,
            )?;
        }
        (true, None) => {
            return Err(Error::Msg(
                "mlx: this checkpoint has a guidance embedding and needs a guidance scale".into(),
            ))
        }
        (false, Some(_)) => {
            return Err(Error::Msg(
                "mlx: this checkpoint takes no guidance scale (schnell is not distilled on one)"
                    .into(),
            ))
        }
        (false, None) => {}
    }
    vec = vec.add(&mlp_embedder(pooled, w, "vector_in", s)?, s)?;

    // Text ids precede image ids, matching the concatenation order inside
    // every block. Text ids are all zero.
    let txt_len = txt.shape()[1];
    let mut ids = vec![0.0f32; txt_len * 3];
    ids.extend_from_slice(img_ids);
    let seq = txt_len + img_ids.len() / 3;
    let pe = embed_nd(&ids, seq, &cfg.axes_dim, cfg.theta)?;

    for i in 0..cfg.depth {
        let (a, b) = double_block(&img, &txt, &vec, &pe, i, cfg, w, s)?;
        img = a;
        txt = b;
    }

    let mut xs = concat(&[&txt, &img], 1, s)?;
    for i in 0..cfg.depth_single_blocks {
        xs = single_block(&xs, &vec, &pe, i, cfg, w, s)?;
    }
    let img = xs.narrow(1, txt_len, xs.shape()[1] - txt_len, s)?;

    // The output head: modulate, then project back to patch channels. **Shift
    // comes first here**, unlike `modulation`, which yields (shift, scale,
    // gate) — the ordering is per-module, not global.
    let params = linear(
        &vec.silu(s)?,
        get(w, "final_layer.adaLN_modulation.1.weight")?,
        w.get("final_layer.adaLN_modulation.1.bias"),
        s,
    )?;
    let [b, total] = params.shape()[..] else {
        return Err(Error::Msg(format!("mlx: final mod {:?}", params.shape())));
    };
    let dim = total / 2;
    let params = params.reshape(&[b, 1, total], s)?;
    let shift = params.narrow(2, 0, dim, s)?;
    let scale = params.narrow(2, dim, dim, s)?;

    let out = norm_modulate(&img, &shift, &scale, s)?;
    linear(
        &out,
        get(w, "final_layer.linear.weight")?,
        w.get("final_layer.linear.bias"),
        s,
    )
}
