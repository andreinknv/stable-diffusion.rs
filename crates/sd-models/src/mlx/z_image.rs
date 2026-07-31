//! Z-Image's transformer.
//!
//! A Lumina-descended DiT, and its shape is its own. Read from
//! `ZImageTransformer2DModel` and its own state dict rather than inferred:
//!
//! **Sandwich normalisation.** Every sublayer is normed on the way *in* and
//! again on its output before the residual — `x + gate * norm2(attn(norm1(x)))`
//! — where a DiT normally norms only the input. Dropping `norm2` runs and
//! drifts.
//!
//! **The gates go through `tanh`.** `gate_msa` and `gate_mlp` are squashed to
//! `[-1, 1]` before use. Nothing else in this project does that, and omitting
//! it leaves the arithmetic valid and the residuals unbounded.
//!
//! **Four modulation parameters, and no shift.** `scale_msa, gate_msa,
//! scale_mlp, gate_mlp` — the scales multiply the *normed* input as
//! `1 + scale`, and there is no additive term at all. A six-way `(shift,
//! scale, gate)` reading takes the wrong slices.
//!
//! **Three stacks, not one.** `noise_refiner` runs over the image tokens and
//! `context_refiner` over the text tokens, *before* the two are concatenated
//! and the main `layers` run over both. The context refiner is the only one
//! without modulation — it has no `adaLN_modulation` weights at all.
//!
//! **Sequences are padded to a multiple of 32**, with learned `x_pad_token`
//! and `cap_pad_token`, and an attention mask marks the padding. That is why
//! the reference implementation takes lists of images rather than a batch.

use sd_tensor::mlx::{concat, Array, Stream};
use sd_tensor::{Error, Result};

use super::quantized::WeightSource;

/// Z-Image pads every stream to a multiple of this.
pub const SEQ_MULTIPLE: usize = 32;

/// The conditioning vector is capped at this width, however wide the model is.
pub const ADALN_EMBED_DIM: usize = 256;

/// The timestep arrives in `[0, 1]` and the model works in `[0, 1000]`.
pub const T_SCALE: f32 = 1000.0;

/// Z-Image transformer geometry.
#[derive(Debug, Clone)]
pub struct ZImageConfig {
    pub dim: usize,
    pub layers: usize,
    /// Blocks that run on each stream *before* they are joined.
    pub refiner_layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub in_channels: usize,
    /// The text encoder's width — 2560, because Z-Image conditions on Qwen3.
    pub cap_feat_dim: usize,
    pub norm_eps: f32,
    pub rope_theta: f64,
    /// Per-axis head-dimension split, `(t, h, w)`. Sums to `head_dim`.
    pub axes_dims: Vec<usize>,
    pub patch_size: usize,
}

impl ZImageConfig {
    /// The published `Tongyi-MAI/Z-Image` geometry.
    pub fn base() -> Self {
        Self {
            dim: 3840,
            layers: 30,
            refiner_layers: 2,
            heads: 30,
            kv_heads: 30,
            in_channels: 16,
            cap_feat_dim: 2560,
            norm_eps: 1e-5,
            rope_theta: 256.0,
            axes_dims: vec![32, 48, 48],
            patch_size: 2,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }

    /// The width of the timestep embedding that drives every modulation.
    ///
    /// **`min(dim, 256)`, not `dim`.** At the published 3840 that caps the
    /// conditioning vector at 256, so `adaLN_modulation` is `[4*dim, 256]`
    /// rather than square. Deriving it as `dim` builds a projection whose
    /// input width does not match its weight — loud at 3840, and silently
    /// *correct* at any width below 256, which is exactly the size a fixture
    /// uses.
    pub fn temb_dim(&self) -> usize {
        self.dim.min(ADALN_EMBED_DIM)
    }

    /// The key the checkpoint files its patch embedder and final layer under:
    /// `{patch_size}-{frame_patch_size}`, and the frame patch is always 1 for
    /// a still.
    pub fn variant_key(&self) -> String {
        format!("{}-1", self.patch_size)
    }
}

/// Round `n` up to the next multiple of [`SEQ_MULTIPLE`].
pub fn padded_len(n: usize) -> usize {
    n.div_ceil(SEQ_MULTIPLE) * SEQ_MULTIPLE
}

/// `(cos, sin)` for `[seq, axes]` coordinates, each `[1, 1, seq, head_dim/2]`.
///
/// **Half-width tables, applied to adjacent pairs.** The reference builds these
/// as complex numbers and multiplies; that is the interleaved convention, so
/// each frequency covers one `(x[2i], x[2i+1])` pair rather than being
/// duplicated across two slots as FLUX.2's are.
pub fn rope_tables(
    ids: &[i32],
    seq: usize,
    axes_dims: &[usize],
    theta: f64,
    s: &Stream,
) -> Result<(Array, Array)> {
    let n_axes = axes_dims.len();
    if ids.len() != seq * n_axes {
        return Err(Error::Msg(format!(
            "mlx: z-image got {} ids for {seq} tokens across {n_axes} axes",
            ids.len()
        )));
    }
    let half_total: usize = axes_dims.iter().map(|d| d / 2).sum();
    let mut cos = vec![0.0f32; seq * half_total];
    let mut sin = vec![0.0f32; seq * half_total];

    let mut offset = 0usize;
    for (axis, &dim) in axes_dims.iter().enumerate() {
        let half = dim / 2;
        for t in 0..seq {
            let pos = ids[t * n_axes + axis] as f64;
            for i in 0..half {
                // f64 throughout: theta is 256 here rather than 10,000, so the
                // frequencies are closer together and f32 loses more of them.
                let omega = 1.0 / theta.powf(2.0 * i as f64 / dim as f64);
                let angle = pos * omega;
                cos[t * half_total + offset + i] = angle.cos() as f32;
                sin[t * half_total + offset + i] = angle.sin() as f32;
            }
        }
        offset += half;
    }
    Ok((
        Array::from_slice_f32(&cos, &[1, 1, seq, half_total])?.contiguous(s)?,
        Array::from_slice_f32(&sin, &[1, 1, seq, half_total])?.contiguous(s)?,
    ))
}

/// Interleaved rotation of `[b, h, seq, head_dim]` by half-width tables.
fn apply_rope(x: &Array, cos: &Array, sin: &Array, s: &Stream) -> Result<Array> {
    let [b, h, n, d] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: z-image rope {:?}", x.shape())));
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

/// `w2(silu(w1(x)) * w3(x))` — SwiGLU with three separate projections.
///
/// **`w1` is the gate and `w3` the value**, which is the opposite of the
/// `gate_proj`/`up_proj` naming the LLM encoder uses. Swapping them runs.
fn feed_forward(x: &Array, w: &impl WeightSource, prefix: &str, s: &Stream) -> Result<Array> {
    let gate = w
        .linear(x, &format!("{prefix}.w1.weight"), None, s)?
        .silu(s)?;
    let value = w.linear(x, &format!("{prefix}.w3.weight"), None, s)?;
    w.linear(
        &gate.mul(&value, s)?,
        &format!("{prefix}.w2.weight"),
        None,
        s,
    )
}

/// Self-attention with per-head QK normalisation and rotary positions.
#[allow(clippy::too_many_arguments)]
fn attention(
    x: &Array,
    mask: Option<&Array>,
    cos: &Array,
    sin: &Array,
    cfg: &ZImageConfig,
    w: &impl WeightSource,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let (heads, hd) = (cfg.heads, cfg.head_dim());
    let [b, n, _] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: z-image attn {:?}", x.shape())));
    };
    let split = |t: Array| -> Result<Array> {
        t.reshape(&[b, n, heads, hd], s)?
            .transpose(&[0, 2, 1, 3], s)?
            .contiguous(s)
    };
    let q = split(w.linear(x, &format!("{prefix}.to_q.weight"), None, s)?)?;
    let k = split(w.linear(x, &format!("{prefix}.to_k.weight"), None, s)?)?;
    let v = split(w.linear(x, &format!("{prefix}.to_v.weight"), None, s)?)?;

    // Per head, before the rotation — the weights are `[head_dim]`.
    let q = q.rms_norm(
        Some(w.dense(&format!("{prefix}.norm_q.weight"))?),
        cfg.norm_eps,
        s,
    )?;
    let k = k.rms_norm(
        Some(w.dense(&format!("{prefix}.norm_k.weight"))?),
        cfg.norm_eps,
        s,
    )?;
    let q = apply_rope(&q, cos, sin, s)?;
    let k = apply_rope(&k, cos, sin, s)?;

    let scale = 1.0 / (hd as f32).sqrt();
    let attended = match mask {
        Some(m) => q.sdpa_masked(&k, &v, scale, m, s)?,
        None => q.sdpa(&k, &v, scale, s)?,
    };
    let merged = attended
        .transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[b, n, heads * hd], s)?;
    w.linear(&merged, &format!("{prefix}.to_out.0.weight"), None, s)
}

/// One block. `adaln` is `None` for the context refiner, which has no
/// modulation weights at all.
#[allow(clippy::too_many_arguments)]
fn block(
    x: &Array,
    mask: Option<&Array>,
    cos: &Array,
    sin: &Array,
    adaln: Option<&Array>,
    cfg: &ZImageConfig,
    w: &impl WeightSource,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let norm = |t: &Array, which: &str| -> Result<Array> {
        t.rms_norm(
            Some(w.dense(&format!("{prefix}.{which}.weight"))?),
            cfg.norm_eps,
            s,
        )
    };

    let Some(temb) = adaln else {
        // No modulation: the sandwich norms and nothing else.
        let attn = attention(
            &norm(x, "attention_norm1")?,
            mask,
            cos,
            sin,
            cfg,
            w,
            &format!("{prefix}.attention"),
            s,
        )?;
        let x = x.add(&norm(&attn, "attention_norm2")?, s)?;
        let ff = feed_forward(
            &norm(&x, "ffn_norm1")?,
            w,
            &format!("{prefix}.feed_forward"),
            s,
        )?;
        return x.add(&norm(&ff, "ffn_norm2")?, s);
    };

    // `(scale_msa, gate_msa, scale_mlp, gate_mlp)` — four, and no shift.
    let m = w.linear(
        temb,
        &format!("{prefix}.adaLN_modulation.0.weight"),
        w.optional(&format!("{prefix}.adaLN_modulation.0.bias")),
        s,
    )?;
    let dim = cfg.dim;
    let last = m.shape().len() - 1;
    let take = |i: usize| -> Result<Array> {
        m.narrow(last, i * dim, dim, s)?
            .reshape(&[m.shape()[0], 1, dim], s)
    };
    let one = Array::scalar_f32(1.0)?;
    let scale_msa = take(0)?.add(&one, s)?;
    // **tanh on the gates.** Nothing else here does this.
    let gate_msa = take(1)?.tanh(s)?;
    let scale_mlp = take(2)?.add(&one, s)?;
    let gate_mlp = take(3)?.tanh(s)?;

    let attn = attention(
        &norm(x, "attention_norm1")?.mul(&scale_msa, s)?,
        mask,
        cos,
        sin,
        cfg,
        w,
        &format!("{prefix}.attention"),
        s,
    )?;
    let x = x.add(&norm(&attn, "attention_norm2")?.mul(&gate_msa, s)?, s)?;
    let ff = feed_forward(
        &norm(&x, "ffn_norm1")?.mul(&scale_mlp, s)?,
        w,
        &format!("{prefix}.feed_forward"),
        s,
    )?;
    x.add(&norm(&ff, "ffn_norm2")?.mul(&gate_mlp, s)?, s)
}

/// Sinusoidal timestep features, sine half first.
fn timestep_features(t: &Array, channels: usize, s: &Stream) -> Result<Array> {
    let half = channels / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Array::from_slice_f32(&freqs, &[1, half])?;
    let args = t.reshape(&[t.shape()[0], 1], s)?.matmul(&freqs, s)?;
    concat(&[&args.cos(s)?, &args.sin(s)?], 1, s)
}

/// `(t, h, w)` coordinates for a `f x h x w` patch grid, row-major.
pub fn image_ids(frames: usize, h: usize, w: usize) -> Vec<i32> {
    let mut v = Vec::with_capacity(frames * h * w * 3);
    for f in 0..frames {
        for y in 0..h {
            for x in 0..w {
                v.push(f as i32);
                v.push(y as i32);
                v.push(x as i32);
            }
        }
    }
    v
}

/// `(t, 0, 0)` coordinates for `len` caption tokens.
///
/// **One-based.** The first caption token sits at `t = 1`, not 0, and a
/// 32-token caption ends at 32 — which is why the image stream starts at
/// `padded_caption_length + 1` rather than at the caption's length.
///
/// Position 0 is left to the *padding*: an image's pad tokens all sit at the
/// origin. Starting the caption at 0 puts its first real token exactly on top
/// of them, and costs about 5% of the output with nothing reporting a problem.
/// Read out of the reference's own `cap_pos_ids`, not from its source.
pub fn caption_ids(len: usize) -> Vec<i32> {
    let mut v = Vec::with_capacity(len * 3);
    for i in 0..len {
        v.push(i as i32 + 1);
        v.push(0);
        v.push(0);
    }
    v
}

/// Patchify `[c, f, h, w]` into `[tokens, pF*pH*pW*c]`.
///
/// The permutation is `(f, h, w, pf, ph, pw, c)` — **channel last**, after all
/// three sub-patch axes. A DiT that folds channel-first, as SD 3 does, gives
/// the right token count with each token's features interleaved wrongly.
pub fn patchify(x: &Array, patch: usize, s: &Stream) -> Result<Array> {
    let [c, f, h, w] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: z-image patchify {:?}", x.shape())));
    };
    if h % patch != 0 || w % patch != 0 {
        return Err(Error::Msg(format!(
            "mlx: a {h}x{w} latent does not divide into {patch}x{patch} patches"
        )));
    }
    let (ht, wt) = (h / patch, w / patch);
    // The frame patch is 1 for a still, so `f` is already the token count.
    x.reshape(&[c, f, 1, ht, patch, wt, patch], s)?
        .transpose(&[1, 3, 5, 2, 4, 6, 0], s)?
        .contiguous(s)?
        .reshape(&[f * ht * wt, patch * patch * c], s)
}

/// The inverse of [`patchify`], back to `[c, f, h, w]`.
pub fn unpatchify(
    x: &Array,
    f: usize,
    h: usize,
    w: usize,
    patch: usize,
    channels: usize,
    s: &Stream,
) -> Result<Array> {
    let (ht, wt) = (h / patch, w / patch);
    x.reshape(&[f, ht, wt, 1, patch, patch, channels], s)?
        .transpose(&[6, 0, 3, 1, 4, 2, 5], s)?
        .contiguous(s)?
        .reshape(&[channels, f, h, w], s)
}

/// Run one stack of blocks.
#[allow(clippy::too_many_arguments)]
fn stack(
    mut x: Array,
    cos: &Array,
    sin: &Array,
    adaln: Option<&Array>,
    count: usize,
    cfg: &ZImageConfig,
    w: &impl WeightSource,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    for i in 0..count {
        x = block(
            &x,
            None,
            cos,
            sin,
            adaln,
            cfg,
            w,
            &format!("{prefix}.{i}"),
            s,
        )?;
    }
    Ok(x)
}

/// Grow `[1, real, dim]` to `[1, target, dim]`, filling with a learned token.
///
/// **This lengthens the sequence, it does not just overwrite a tail.** Z-Image
/// pads every stream up to a multiple of 32 and the pad slots carry
/// `x_pad_token` or `cap_pad_token` — a learned vector, not zeros, and not a
/// repeat of the last real token.
fn pad_to(x: &Array, real: usize, target: usize, pad: &Array, s: &Stream) -> Result<Array> {
    let [b, n, dim] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: z-image pad {:?}", x.shape())));
    };
    if target <= n {
        return x.narrow(1, 0, target, s)?.contiguous(s);
    }
    let head = x.narrow(1, 0, real.min(n), s)?;
    let tail = pad
        .reshape(&[1, 1, dim], s)?
        .broadcast_to(&[b, target - real.min(n), dim], s)?;
    concat(&[&head, &tail], 1, s)
}

/// Every stage's output, for localising a divergence.
///
/// Z-Image runs three stacks in sequence and a mismatch at the end says
/// nothing about which. The golden test compares each.
pub struct Stages {
    /// The joined sequence handed to `layers`, before any of them run.
    pub unified_input: Array,
    pub after_layer0: Array,
    pub after_noise_refiner: Array,
    pub after_context_refiner: Array,
    pub after_layers: Array,
    pub output: Array,
}

/// [`forward`], keeping each stack's output.
pub fn forward_stages(
    latent: &Array,
    cap: &Array,
    timestep: &Array,
    cfg: &ZImageConfig,
    w: &impl WeightSource,
    s: &Stream,
) -> Result<Stages> {
    forward_inner(latent, cap, timestep, cfg, w, s)
}

/// The velocity Z-Image predicts, for one image and one caption.
///
/// `latent` is `[c, f, h, w]` and `cap` is `[cap_len, cap_feat_dim]` — the
/// reference takes lists because it batches images of differing sizes, and a
/// single pair is the case that matters here.
///
/// **Both streams are padded to a multiple of 32** with their own learned pad
/// tokens, which is why the shapes below are rounded up.
pub fn forward(
    latent: &Array,
    cap: &Array,
    timestep: &Array,
    cfg: &ZImageConfig,
    w: &impl WeightSource,
    s: &Stream,
) -> Result<Array> {
    Ok(forward_inner(latent, cap, timestep, cfg, w, s)?.output)
}

fn forward_inner(
    latent: &Array,
    cap: &Array,
    timestep: &Array,
    cfg: &ZImageConfig,
    w: &impl WeightSource,
    s: &Stream,
) -> Result<Stages> {
    let [c, f, h, wd] = latent.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: z-image latent {:?}",
            latent.shape()
        )));
    };
    let [cap_len, _] = cap.shape()[..] else {
        return Err(Error::Msg(format!("mlx: z-image cap {:?}", cap.shape())));
    };
    let key = cfg.variant_key();
    let (ht, wt) = (h / cfg.patch_size, wd / cfg.patch_size);
    let img_real = f * ht * wt;
    let img_len = padded_len(img_real);
    let cap_padded = padded_len(cap_len);

    // **The scaling happens here, not in the caller.** The sampler works in
    // `[0, 1]` and this model embeds `[0, 1000]`; the reference multiplies
    // inside its own forward, so doing it outside would double-scale for
    // anyone following that code.
    let scaled = timestep.mul(&Array::scalar_f32(T_SCALE)?, s)?;
    let t_feat = timestep_features(&scaled, ADALN_EMBED_DIM, s)?;
    let temb = w.linear(
        &w.linear(
            &t_feat,
            "t_embedder.mlp.0.weight",
            w.optional("t_embedder.mlp.0.bias"),
            s,
        )?
        .silu(s)?,
        "t_embedder.mlp.2.weight",
        w.optional("t_embedder.mlp.2.bias"),
        s,
    )?;

    // -- image stream --------------------------------------------------
    let patches = patchify(latent, cfg.patch_size, s)?;
    let x = w.linear(
        &patches.reshape(&[1, img_real, patches.shape()[1]], s)?,
        &format!("all_x_embedder.{key}.weight"),
        w.optional(&format!("all_x_embedder.{key}.bias")),
        s,
    )?;
    let x = pad_to(&x, img_real, img_len, w.dense("x_pad_token")?, s)?;

    // **The image's `t` coordinate starts after the caption's.** The two
    // streams share one axis, so an image token at t=0 would collide with the
    // first caption token.
    let mut img_ids = image_ids(f, ht, wt);
    for i in 0..img_real {
        img_ids[i * 3] += (cap_padded + 1) as i32;
    }
    // Padded image tokens sit at the origin, which is what the reference does.
    img_ids.resize(img_len * 3, 0);
    let (img_cos, img_sin) = rope_tables(&img_ids, img_len, &cfg.axes_dims, cfg.rope_theta, s)?;

    let x = stack(
        x,
        &img_cos,
        &img_sin,
        Some(&temb),
        cfg.refiner_layers,
        cfg,
        w,
        "noise_refiner",
        s,
    )?;
    let after_noise_refiner = x.contiguous(s)?;

    // -- caption stream ------------------------------------------------
    // `cap_embedder` is an RMSNorm then a Linear, so the norm's weight is
    // `cap_embedder.0.weight` and it has no bias.
    let cap_in = cap.reshape(&[1, cap_len, cap.shape()[1]], s)?;
    let normed = cap_in.rms_norm(Some(w.dense("cap_embedder.0.weight")?), cfg.norm_eps, s)?;
    let caps = w.linear(
        &normed,
        "cap_embedder.1.weight",
        w.optional("cap_embedder.1.bias"),
        s,
    )?;
    let caps = pad_to(&caps, cap_len, cap_padded, w.dense("cap_pad_token")?, s)?;

    let cap_ids = caption_ids(cap_padded);
    let (cap_cos, cap_sin) = rope_tables(&cap_ids, cap_padded, &cfg.axes_dims, cfg.rope_theta, s)?;
    // **No modulation here.** The context refiner has no `adaLN_modulation`
    // weights at all, which is how it differs from the noise refiner.
    let caps = stack(
        caps,
        &cap_cos,
        &cap_sin,
        None,
        cfg.refiner_layers,
        cfg,
        w,
        "context_refiner",
        s,
    )?;
    let after_context_refiner = caps.contiguous(s)?;

    // -- joined --------------------------------------------------------
    // **Image first, then caption** — the opposite of FLUX.2's order.
    let unified = concat(&[&x, &caps], 1, s)?;
    let mut ids = img_ids.clone();
    ids.truncate(img_len * 3);
    ids.extend_from_slice(&cap_ids);
    let total = img_len + cap_padded;
    let (cos, sin) = rope_tables(&ids, total, &cfg.axes_dims, cfg.rope_theta, s)?;

    let unified_input = unified.contiguous(s)?;
    let first = block(
        &unified,
        None,
        &cos,
        &sin,
        Some(&temb),
        cfg,
        w,
        "layers.0",
        s,
    )?;
    let after_layer0 = first.contiguous(s)?;
    let mut unified = first;
    for i in 1..cfg.layers {
        unified = block(
            &unified,
            None,
            &cos,
            &sin,
            Some(&temb),
            cfg,
            w,
            &format!("layers.{i}"),
            s,
        )?;
    }
    let after_layers = unified.contiguous(s)?;

    // -- output --------------------------------------------------------
    // **Scale only, no shift** — the final layer modulates with one vector.
    let scale = w
        .linear(
            &temb.silu(s)?,
            &format!("all_final_layer.{key}.adaLN_modulation.1.weight"),
            w.optional(&format!("all_final_layer.{key}.adaLN_modulation.1.bias")),
            s,
        )?
        .add(&Array::scalar_f32(1.0)?, s)?;
    let dim = cfg.dim;
    let scale = scale.reshape(&[1, 1, dim], s)?;
    let out = unified.layer_norm(None, None, 1e-6, s)?.mul(&scale, s)?;
    let out = w.linear(
        &out,
        &format!("all_final_layer.{key}.linear.weight"),
        w.optional(&format!("all_final_layer.{key}.linear.bias")),
        s,
    )?;

    // Only the real image tokens survive; the padding and the caption do not.
    let out = out.narrow(1, 0, img_real, s)?.contiguous(s)?;
    let width = out.shape()[2];
    let output = unpatchify(
        &out.reshape(&[img_real, width], s)?,
        f,
        h,
        wd,
        cfg.patch_size,
        c,
        s,
    )?;
    Ok(Stages {
        unified_input,
        after_layer0,
        after_noise_refiner,
        after_context_refiner,
        after_layers,
        output,
    })
}
