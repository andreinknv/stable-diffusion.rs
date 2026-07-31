//! SD 3 / SD 3.5's MMDiT transformer on MLX.
//!
//! Patchified latent, joint attention over image and text tokens, conditioning
//! by modulation. Four things differ from Flux and each fails quietly if
//! assumed from it:
//!
//! - **Learned positional embeddings, not RoPE.** A `384 x 384` table cropped
//!   from the *centre* for the image size in use. Cropping from a corner gives
//!   a coherent image whose composition is systematically offset.
//! - **Every block is a joint block.** No single-stream half; image and text
//!   keep separate weights throughout.
//! - **The last block's context half is `pre_only`.** It contributes keys and
//!   values to the final attention and its own output is then discarded, so it
//!   carries no projection and no MLP.
//! - **SD 3.5 adds a second image self-attention** in its first 13 blocks —
//!   "MMDiT-X". Those blocks modulate nine ways instead of six.
//!
//! Names follow the original Stability layout (`joint_blocks.0.x_block.attn`),
//! which is what the published checkpoints use.

use sd_tensor::mlx::{concat, Array, Stream};
use sd_tensor::{Error, Result};

use super::quantized::WeightSource;
use super::sinusoid_embedding;

/// Width of the raw timestep sinusoid, before `t_embedder`.
const TIME_EMBED_DIM: usize = 256;
/// Every norm in this transformer.
const EPS: f32 = 1e-6;

/// SD 3 transformer geometry.
#[derive(Debug, Clone)]
pub struct Sd3Config {
    pub in_channels: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
    /// Blocks carrying a second image self-attention. Empty for SD 3.
    pub dual_attention_layers: Vec<usize>,
    /// Width of the stored positional table, in patches per side.
    pub pos_embed_max_size: usize,
}

impl Sd3Config {
    /// `stabilityai/stable-diffusion-3.5-medium`.
    pub fn medium_35() -> Self {
        Self {
            in_channels: 16,
            patch_size: 2,
            hidden_size: 1536,
            num_heads: 24,
            depth: 24,
            dual_attention_layers: (0..13).collect(),
            pos_embed_max_size: 384,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    fn is_dual(&self, block: usize) -> bool {
        self.dual_attention_layers.contains(&block)
    }
}

/// `layer_norm(x) * (1 + scale) + shift`, the adaLN the whole model is built
/// on. No affine in the norm itself — the modulation supplies it.
fn norm_modulate(x: &Array, shift: &Array, scale: &Array, s: &Stream) -> Result<Array> {
    let normed = x.layer_norm(None, None, EPS, s)?;
    normed
        .mul(&scale.add(&Array::scalar_f32(1.0)?, s)?, s)?
        .add(shift, s)
}

/// RMSNorm over the head dimension, applied to queries and keys.
fn head_rms_norm(x: &Array, weight: &Array, s: &Stream) -> Result<Array> {
    let dims = x.shape();
    let last = dims.len() - 1;
    let mean_sq = x.mul(x, s)?.mean(&[last], true, s)?;
    x.mul(&mean_sq.add(&Array::scalar_f32(EPS)?, s)?.rsqrt(s)?, s)?
        .mul(weight, s)
}

/// Project and split into per-head `q`, `k`, `v`, normalising q and k.
fn qkv(
    x: &Array,
    cfg: &Sd3Config,
    w: &impl WeightSource,
    prefix: &str,
    s: &Stream,
) -> Result<(Array, Array, Array)> {
    let p = |n: &str| format!("{prefix}.{n}");
    let [b, n, _] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: sd3 qkv got {:?}", x.shape())));
    };
    let (heads, hd) = (cfg.num_heads, cfg.head_dim());

    let t = w
        .linear(x, &p("qkv.weight"), w.optional(&p("qkv.bias")), s)?
        .reshape(&[b, n, 3, heads, hd], s)?
        .transpose(&[2, 0, 3, 1, 4], s)?;
    let take = |i: usize| -> Result<Array> { t.narrow(0, i, 1, s)?.reshape(&[b, heads, n, hd], s) };
    Ok((
        head_rms_norm(&take(0)?, w.dense(&p("ln_q.weight"))?, s)?,
        head_rms_norm(&take(1)?, w.dense(&p("ln_k.weight"))?, s)?,
        take(2)?,
    ))
}

/// Merge heads and project. Absent on a `pre_only` block, whose output is
/// discarded.
fn post(attn: &Array, w: &impl WeightSource, prefix: &str, s: &Stream) -> Result<Array> {
    let [b, h, n, d] = attn.shape()[..] else {
        return Err(Error::Msg(format!("mlx: sd3 post got {:?}", attn.shape())));
    };
    let merged = attn
        .transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[b, n, h * d], s)?;
    w.linear(
        &merged,
        &format!("{prefix}.proj.weight"),
        w.optional(&format!("{prefix}.proj.bias")),
        s,
    )
}

/// `fc2(gelu_new(fc1(x)))`. The tanh approximation, as SD 3 uses.
fn mlp(x: &Array, w: &impl WeightSource, prefix: &str, s: &Stream) -> Result<Array> {
    let p = |n: &str| format!("{prefix}.{n}");
    let h = w
        .linear(x, &p("fc1.weight"), w.optional(&p("fc1.bias")), s)?
        .gelu_approx(s)?;
    w.linear(&h, &p("fc2.weight"), w.optional(&p("fc2.bias")), s)
}

/// This half-block's modulation parameters, `chunks` of `hidden_size` each.
///
/// `adaLN_modulation` is `Sequential(SiLU, Linear)`; index 1 is the Linear and
/// index 0 has no weights.
fn modulation(
    c: &Array,
    chunks: usize,
    hidden: usize,
    w: &impl WeightSource,
    prefix: &str,
    s: &Stream,
) -> Result<Vec<Array>> {
    let out = w.linear(
        &c.silu(s)?,
        &format!("{prefix}.adaLN_modulation.1.weight"),
        w.optional(&format!("{prefix}.adaLN_modulation.1.bias")),
        s,
    )?;
    let [b, _] = out.shape()[..] else {
        return Err(Error::Msg(format!("mlx: modulation got {:?}", out.shape())));
    };
    let out = out.reshape(&[b, 1, chunks * hidden], s)?;
    (0..chunks)
        .map(|i| out.narrow(2, i * hidden, hidden, s))
        .collect()
}

/// One joint block: image and text streams with their own weights, mixed
/// inside one attention over the concatenated tokens.
///
/// Returns `(context_out, x_out)`; `context_out` is `None` for the final
/// block, whose context half is `pre_only`.
#[allow(clippy::too_many_arguments)]
fn joint_block(
    context: &Array,
    x: &Array,
    c: &Array,
    index: usize,
    cfg: &Sd3Config,
    w: &impl WeightSource,
    s: &Stream,
) -> Result<(Option<Array>, Array)> {
    let path = format!("joint_blocks.{index}");
    let ctx_p = format!("{path}.context_block");
    let x_p = format!("{path}.x_block");
    let pre_only = index + 1 == cfg.depth;
    let dual = cfg.is_dual(index);
    let h = cfg.hidden_size;

    // 2 for pre_only (shift and scale, nothing to gate), 9 with a second
    // attention, 6 otherwise.
    let cm = modulation(c, if pre_only { 2 } else { 6 }, h, w, &ctx_p, s)?;
    let xm = modulation(c, if dual { 9 } else { 6 }, h, w, &x_p, s)?;

    let c_in = norm_modulate(context, &cm[0], &cm[1], s)?;
    let (cq, ck, cv) = qkv(&c_in, cfg, w, &format!("{ctx_p}.attn"), s)?;
    let x_in = norm_modulate(x, &xm[0], &xm[1], s)?;
    let (xq, xk, xv) = qkv(&x_in, cfg, w, &format!("{x_p}.attn"), s)?;

    // Context first, then image — the order SD 3 concatenates them, and the
    // order they are split back out below.
    let q = concat(&[&cq, &xq], 2, s)?;
    let k = concat(&[&ck, &xk], 2, s)?;
    let v = concat(&[&cv, &xv], 2, s)?;
    let attn = q.sdpa(&k, &v, 1.0 / (cfg.head_dim() as f32).sqrt(), s)?;

    let c_len = cq.shape()[2];
    let total = attn.shape()[2];
    let c_attn = attn.narrow(2, 0, c_len, s)?;
    let x_attn = attn.narrow(2, c_len, total - c_len, s)?;

    let context_out = if pre_only {
        // Feeds the attention and is then dropped: nothing to gate or project.
        None
    } else {
        let ctx = context.add(
            &post(&c_attn, w, &format!("{ctx_p}.attn"), s)?.mul(&cm[2], s)?,
            s,
        )?;
        let ff = mlp(
            &norm_modulate(&ctx, &cm[3], &cm[4], s)?,
            w,
            &format!("{ctx_p}.mlp"),
            s,
        )?;
        Some(ctx.add(&ff.mul(&cm[5], s)?, s)?)
    };

    let mut xs = x.add(
        &post(&x_attn, w, &format!("{x_p}.attn"), s)?.mul(&xm[2], s)?,
        s,
    )?;
    if dual {
        // A second self-attention over the image tokens alone, modulated
        // independently and added to the same residual.
        let a2 = format!("{x_p}.attn2");
        let x2_in = norm_modulate(x, &xm[6], &xm[7], s)?;
        let (q2, k2, v2) = qkv(&x2_in, cfg, w, &a2, s)?;
        let attn2 = q2.sdpa(&k2, &v2, 1.0 / (cfg.head_dim() as f32).sqrt(), s)?;
        xs = xs.add(&post(&attn2, w, &a2, s)?.mul(&xm[8], s)?, s)?;
    }
    let ff = mlp(
        &norm_modulate(&xs, &xm[3], &xm[4], s)?,
        w,
        &format!("{x_p}.mlp"),
        s,
    )?;
    let xs = xs.add(&ff.mul(&xm[5], s)?, s)?;
    Ok((context_out, xs))
}

/// `[b, c, h, w]` into `[b, (h/2)*(w/2), c*4]`, channel-major within a patch.
///
/// This is the order the patch-embedding convolution's flattened kernel runs.
pub fn pack_latents(latents: &Array, s: &Stream) -> Result<Array> {
    let [b, c, h, w] = latents.shape()[..] else {
        return Err(Error::Msg(format!("mlx: pack got {:?}", latents.shape())));
    };
    if h % 2 != 0 || w % 2 != 0 {
        return Err(Error::Msg(format!(
            "mlx: latent {h}x{w} must have even sides to pack into 2x2 patches"
        )));
    }
    latents
        .reshape(&[b, c, h / 2, 2, w / 2, 2], s)?
        .transpose(&[0, 2, 4, 1, 3, 5], s)?
        .contiguous(s)?
        .reshape(&[b, (h / 2) * (w / 2), c * 4], s)
}

/// The slice of the stored positional table for an `h x w` patch grid.
///
/// **Cropped from the centre.** Taking it from a corner yields a coherent
/// image whose composition is systematically offset — plausible enough to miss.
fn cropped_pos_embed(
    cfg: &Sd3Config,
    h: usize,
    w: usize,
    wt: &impl WeightSource,
    s: &Stream,
) -> Result<Array> {
    let max = cfg.pos_embed_max_size;
    if h > max || w > max {
        return Err(Error::Msg(format!(
            "mlx: a {h}x{w} patch grid exceeds the stored positional table ({max}x{max})"
        )));
    }
    let (top, left) = ((max - h) / 2, (max - w) / 2);
    wt.dense("pos_embed")?
        .reshape(&[1, max, max, cfg.hidden_size], s)?
        .narrow(1, top, h, s)?
        .narrow(2, left, w, s)?
        .contiguous(s)?
        .reshape(&[1, h * w, cfg.hidden_size], s)
}

/// The inverse of the *final linear*, which is **not** the inverse of
/// [`pack_latents`].
///
/// The patch embedding is a convolution, whose flattened kernel runs
/// `(channel, ph, pw)`. The final linear instead emits `(ph, pw, channel)`,
/// channel varying fastest, which is what SD 3's `einsum("nhwpqc->nchpwq")`
/// unpacks. Reusing the packing inverse here gives an image of the right shape
/// whose every 2x2 patch has its channels and positions transposed — coherent
/// colours, destroyed detail, and no error.
fn unpatchify(x: &Array, lh: usize, lw: usize, p: usize, c: usize, s: &Stream) -> Result<Array> {
    let [b, n, _] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: unpatchify got {:?}", x.shape())));
    };
    let (ph, pw) = (lh / p, lw / p);
    if n != ph * pw {
        return Err(Error::Msg(format!(
            "mlx: {n} tokens do not fill a {ph}x{pw} patch grid"
        )));
    }
    x.reshape(&[b, ph, pw, p, p, c], s)?
        .transpose(&[0, 5, 1, 3, 2, 4], s)?
        .contiguous(s)?
        .reshape(&[b, c, ph * p, pw * p], s)
}

/// `t_embedder` / `y_embedder`: `fc2(silu(fc1(x)))`, stored as `mlp.0`/`mlp.2`.
fn embedder(x: &Array, w: &impl WeightSource, prefix: &str, s: &Stream) -> Result<Array> {
    let p = |n: &str| format!("{prefix}.{n}");
    let h = w
        .linear(x, &p("mlp.0.weight"), w.optional(&p("mlp.0.bias")), s)?
        .silu(s)?;
    w.linear(&h, &p("mlp.2.weight"), w.optional(&p("mlp.2.bias")), s)
}

/// The MMDiT forward: latents, T5 context, pooled CLIP and a timestep in; a
/// velocity prediction of the latent's shape out.
pub fn forward(
    latents: &Array,
    context: &Array,
    pooled: &Array,
    timestep: &Array,
    cfg: &Sd3Config,
    w: &impl WeightSource,
    s: &Stream,
) -> Result<Array> {
    forward_skipping(latents, context, pooled, timestep, cfg, &[], w, s)
}

/// [`forward`] with some joint blocks **not run at all**.
///
/// This is the machinery behind skip-layer guidance. Stability found that SD
/// 3.5's anatomy failures — hands, limb counts — come from a small set of
/// middle blocks, and that a third model pass with those blocks bypassed gives
/// a prediction to steer *away* from. Blocks 7, 8 and 9 are the published set.
///
/// **Skipped, not zeroed.** A skipped block passes its inputs through
/// untouched; zeroing its output would delete the residual stream rather than
/// leave it alone, which is a different and much more destructive edit.
///
/// Out-of-range indices are ignored rather than refused: the set is a tuning
/// knob a caller sweeps, and failing a long render because index 40 does not
/// exist in a 24-block model would be the wrong trade. `Sd3Config::depth` is
/// the bound.
#[allow(clippy::too_many_arguments)]
pub fn forward_skipping(
    latents: &Array,
    context: &Array,
    pooled: &Array,
    timestep: &Array,
    cfg: &Sd3Config,
    skip: &[usize],
    w: &impl WeightSource,
    s: &Stream,
) -> Result<Array> {
    let [_, _, lh, lw] = latents.shape()[..] else {
        return Err(Error::Msg(format!("mlx: sd3 got {:?}", latents.shape())));
    };
    let p = cfg.patch_size;
    let (ph, pw) = (lh / p, lw / p);

    let patches = pack_latents(latents, s)?;
    // The patch embedding is stored as a convolution, `[hidden, c, p, p]`.
    // Flattened it is `[hidden, c*p*p]` running `(channel, ph, pw)` — which is
    // exactly the order `pack_latents` produces, and the reason that packing is
    // channel-major rather than position-major.
    // A convolution kernel reshaped into a linear, so it is read as a tensor
    // rather than dispatched through `linear` — hence dense.
    let pw_raw = w.dense("x_embedder.proj.weight")?;
    let flat = pw_raw.shape();
    let patch_weight = if flat.len() == 4 {
        pw_raw.reshape(&[flat[0], flat[1] * flat[2] * flat[3]], s)?
    } else {
        pw_raw.contiguous(s)?
    };
    let mut xs = super::linear(
        &patches,
        &patch_weight,
        w.optional("x_embedder.proj.bias"),
        s,
    )?;
    xs = xs.add(&cropped_pos_embed(cfg, ph, pw, w, s)?, s)?;

    let t = embedder(
        &sinusoid_embedding(timestep, TIME_EMBED_DIM, s)?,
        w,
        "t_embedder",
        s,
    )?;
    let c = t.add(&embedder(pooled, w, "y_embedder", s)?, s)?;
    let mut context = Some(w.linear(
        context,
        "context_embedder.weight",
        w.optional("context_embedder.bias"),
        s,
    )?);

    for i in 0..cfg.depth {
        // A skipped block is a no-op on both streams, not a zeroed one.
        if skip.contains(&i) {
            continue;
        }
        let ctx = context
            .as_ref()
            .ok_or_else(|| Error::Msg("mlx: only the last block may drop context".into()))?;
        let (next_ctx, next_x) = joint_block(ctx, &xs, &c, i, cfg, w, s)?;
        context = next_ctx;
        xs = next_x;
    }

    // The final layer modulates twice — shift and scale, nothing to gate.
    let fm = modulation(&c, 2, cfg.hidden_size, w, "final_layer", s)?;
    let xs = norm_modulate(&xs, &fm[0], &fm[1], s)?;
    let xs = w.linear(
        &xs,
        "final_layer.linear.weight",
        w.optional("final_layer.linear.bias"),
        s,
    )?;
    unpatchify(&xs, lh, lw, p, cfg.in_channels, s)
}
