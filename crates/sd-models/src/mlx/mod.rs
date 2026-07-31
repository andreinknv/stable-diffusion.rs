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

/// CLIP's text tower. Its epsilon is 1e-5 and its activation is QuickGelu.
pub mod clip;
/// ControlNet: a copy of the UNet's down stack that emits corrections.
pub mod controlnet;
/// LoRA adapters, merged into a weight map before any model is built.
pub mod lora;
/// The txt2img sampling loop.
pub mod sample;
/// SD 3 / SD 3.5's MMDiT transformer.
pub mod sd3;
/// T5 v1.1, the text tower Flux and SD 3 use alongside CLIP.
pub mod t5;
/// TAESD, the tiny autoencoder. Its latent scaling is 1.0, not the VAE's.
pub mod taesd;
/// The VAE decoder. Its epsilon is 1e-6, not the UNet's 1e-5.
pub mod vae;

use std::collections::HashMap;

use sd_tensor::mlx::{concat, Array, Stream};
use sd_tensor::{Error, Result};

/// GroupNorm epsilon on `Transformer2DModel`'s spatial wrapper.
pub const SPATIAL_NORM_EPS: f32 = 1e-6;
/// LayerNorm epsilon inside a transformer block. Not the same as the above.
pub const BLOCK_EPS: f32 = 1e-5;
/// GroupNorm epsilon in the resnets.
pub const RESNET_EPS: f32 = 1e-5;
/// SD 1.5 normalises over 32 groups throughout.
pub const NORM_GROUPS: usize = 32;

/// What differs between UNets of this family.
///
/// SD 1.5 and SD 2.x share every block shape and differ only here, so the
/// modules below take a config rather than hard-coding SD 1.5 and being copied
/// for the next architecture.
#[derive(Debug, Clone)]
pub struct UNetConfig {
    /// Attention **head counts** per block, despite diffusers naming the field
    /// `attention_head_dim`. SD 1.5 is `[8; 4]`, giving 40-wide heads at the
    /// first block; SD 2.x is `[5, 10, 20, 20]`, giving 64 throughout.
    pub heads: Vec<usize>,
    /// Resnets per down block. The up blocks take one more.
    pub layers_per_block: usize,
    /// Which down blocks carry a transformer. SD 1.5 and SD 2.x attend on all
    /// but the deepest.
    pub down_has_attention: Vec<bool>,
    /// `proj_in`/`proj_out` are `Linear` when true and 1x1 convolutions when
    /// false. **SD 1.5 is false, SD 2.x is true** — the weights differ in rank,
    /// so the wrong choice fails to load rather than rendering wrongly.
    pub use_linear_projection: bool,
    /// Transformer blocks per attention, per block. SD 1.5 and SD 2.x are 1
    /// throughout; SDXL is `[1, 2, 10]` and is much deeper at the bottom.
    pub transformer_layers: Vec<usize>,
    /// SDXL's micro-conditioning. `None` for everything else here.
    pub addition: Option<AdditionEmbedding>,
    /// unCLIP's image conditioning: `class_embedding` projects a vector into
    /// the timestep embedding. `true` for stable-diffusion-2-1-unclip, `false`
    /// for everything else.
    ///
    /// A separate field from `addition` because they are separate tensors in
    /// the checkpoint — `class_embedding` against `add_embedding` — and no
    /// checkpoint carries both. Collapsing them would make that an assumption
    /// rather than an observation.
    pub class_projection: bool,
}

/// SDXL's extra conditioning: image size and crop offsets, sinusoidally
/// embedded and concatenated with the pooled text embedding.
#[derive(Debug, Clone, Copy)]
pub struct AdditionEmbedding {
    /// Width of the sinusoid applied to each of the six time ids.
    pub time_embed_dim: usize,
}

impl UNetConfig {
    pub fn sd15() -> Self {
        Self {
            heads: vec![8; 4],
            layers_per_block: 2,
            down_has_attention: vec![true, true, true, false],
            use_linear_projection: false,
            transformer_layers: vec![1; 4],
            addition: None,
            class_projection: false,
        }
    }

    /// unCLIP — **SD 2.x exactly**, plus a `class_embedding` that projects a
    /// CLIP *image* embedding into the timestep embedding. Every block, every
    /// width and the text encoder behind it are unchanged, which is why this is
    /// one field rather than an architecture.
    pub fn unclip() -> Self {
        Self {
            class_projection: true,
            ..Self::sd2()
        }
    }

    /// SDXL base: three blocks rather than four, attention on the **last two**
    /// rather than the first three, a much deeper transformer at the bottom,
    /// and the text_time micro-conditioning.
    pub fn sdxl() -> Self {
        Self {
            // 320/5 = 640/10 = 1280/20 = 64 wide throughout.
            heads: vec![5, 10, 20],
            layers_per_block: 2,
            down_has_attention: vec![false, true, true],
            use_linear_projection: true,
            transformer_layers: vec![1, 2, 10],
            addition: Some(AdditionEmbedding {
                time_embed_dim: 256,
            }),
            class_projection: false,
        }
    }

    /// SD 2.x: 64-wide heads throughout, cross-attention at 1024, and linear
    /// projections in the transformer.
    pub fn sd2() -> Self {
        Self {
            heads: vec![5, 10, 20, 20],
            use_linear_projection: true,
            ..Self::sd15()
        }
    }
}

/// The tensors of a checkpoint, by their diffusers names.
pub type Weights = HashMap<String, Array>;

pub(crate) fn get<'a>(w: &'a Weights, key: &str) -> Result<&'a Array> {
    w.get(key)
        .ok_or_else(|| Error::Msg(format!("mlx: checkpoint has no `{key}`")))
}

/// `x @ w.T + b`, the diffusers `Linear` convention where `w` is `(out, in)`.
pub(crate) fn linear(x: &Array, w: &Array, b: Option<&Array>, s: &Stream) -> Result<Array> {
    let wt = w.transpose(&[1, 0], s)?;
    let y = x.matmul(&wt, s)?;
    match b {
        Some(b) => y.add(b, s),
        None => Ok(y),
    }
}

/// A convolution whose weights arrive in diffusers' `(out, in, kh, kw)` and are
/// used in MLX's `(out, kh, kw, in)`.
pub(crate) fn conv_strided(
    x: &Array,
    w: &Array,
    b: Option<&Array>,
    stride: usize,
    padding: usize,
    s: &Stream,
) -> Result<Array> {
    let k = w.transpose(&[0, 2, 3, 1], s)?;
    let y = x.conv2d(&k, (stride, stride), (padding, padding), (1, 1), 1, s)?;
    match b {
        Some(b) => y.add(b, s),
        None => Ok(y),
    }
}

pub fn conv(x: &Array, w: &Array, b: Option<&Array>, padding: usize, s: &Stream) -> Result<Array> {
    conv_strided(x, w, b, 1, padding, s)
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
#[allow(clippy::too_many_arguments)]
pub fn transformer_2d(
    x: &Array,
    context: &Array,
    heads: usize,
    layers: usize,
    linear_projection: bool,
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
    // A linear projection consumes the flattened sequence; a 1x1 convolution
    // consumes the spatial grid. Same arithmetic, different weight rank — SD
    // 1.5 ships 4-D weights here and SD 2.x ships 2-D, so the wrong branch
    // fails to load rather than rendering wrongly.
    let mut seq = if linear_projection {
        let flat = y.reshape(&[n, h * wd, c], s)?;
        linear(
            &flat,
            get(w, &p("proj_in.weight"))?,
            w.get(&p("proj_in.bias")),
            s,
        )?
    } else {
        conv(
            &y,
            get(w, &p("proj_in.weight"))?,
            Some(get(w, &p("proj_in.bias"))?),
            0,
            s,
        )?
        .reshape(&[n, h * wd, c], s)?
    };
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

    let y = if linear_projection {
        linear(
            &seq,
            get(w, &p("proj_out.weight"))?,
            w.get(&p("proj_out.bias")),
            s,
        )?
        .reshape(&[n, h, wd, c], s)?
    } else {
        conv(
            &seq.reshape(&[n, h, wd, c], s)?,
            get(w, &p("proj_out.weight"))?,
            Some(get(w, &p("proj_out.bias"))?),
            0,
            s,
        )?
    };
    y.add(x, s)
}

/// diffusers' `get_timestep_embedding`, without the MLP.
///
/// `flip_sin_to_cos` is true and `downscale_freq_shift` is 0, so cosine comes
/// first and the exponent divides by `half` rather than `half - 1`. The
/// frequencies are built on the CPU and uploaded, because `mlx-c` has no
/// `arange` and this is a small constant.
pub fn sinusoid_embedding(values: &Array, channels: usize, s: &Stream) -> Result<Array> {
    let half = channels / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10000f32.ln()) * i as f32 / half as f32).exp())
        .collect();
    let freqs = Array::from_slice_f32(&freqs, &[1, half])?;
    let v = values.reshape(&[values.elem_count(), 1], s)?;
    let angles = v.matmul(&freqs, s)?;
    // flip_sin_to_cos: cosine first.
    concat(&[&angles.cos(s)?, &angles.sin(s)?], 1, s)
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
    let emb = sinusoid_embedding(timestep, channels, s)?;

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

/// diffusers' `Downsample2D`: a 3x3 convolution at stride 2.
///
/// `padding: 1` here, not the asymmetric pad diffusers uses for
/// `padding=0` downsamplers — SD 1.5's UNet configures `downsample_padding: 1`,
/// so this is the symmetric case. The VAE's is the asymmetric one, and getting
/// that wrong is the bug `docs/handoff.md` records at 17.32.
fn downsample(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    conv_strided(
        x,
        get(w, &format!("{prefix}.conv.weight"))?,
        Some(get(w, &format!("{prefix}.conv.bias"))?),
        2,
        1,
        s,
    )
}

/// One down block: `layers` resnets, each optionally followed by a transformer,
/// then an optional downsampler.
///
/// Returns the block output and the skip entries it contributes, in the order
/// `golden_unet.rs` expects them — one per resnet(+attention) pair, then one
/// for the downsampler.
#[allow(clippy::too_many_arguments)]
pub fn down_block(
    x: &Array,
    temb: &Array,
    context: &Array,
    w: &Weights,
    prefix: &str,
    layers: usize,
    heads: Option<usize>,
    transformer_layers: usize,
    linear_projection: bool,
    has_downsample: bool,
    s: &Stream,
) -> Result<(Array, Vec<Array>)> {
    let mut h = x.contiguous(s)?;
    let mut skips = Vec::with_capacity(layers + usize::from(has_downsample));

    for i in 0..layers {
        h = resnet_block(&h, temb, w, &format!("{prefix}.resnets.{i}"), s)?;
        if let Some(heads) = heads {
            h = transformer_2d(
                &h,
                context,
                heads,
                transformer_layers,
                linear_projection,
                w,
                &format!("{prefix}.attentions.{i}"),
                s,
            )?;
        }
        skips.push(h.contiguous(s)?);
    }

    if has_downsample {
        h = downsample(&h, w, &format!("{prefix}.downsamplers.0"), s)?;
        skips.push(h.contiguous(s)?);
    }
    Ok((h, skips))
}

/// The whole down pass: `conv_in` and the four down blocks.
///
/// Returns the deepest activation and the twelve skip entries, which is exactly
/// the stack `skip_stack_has_twelve_entries` describes — one for `conv_in`,
/// then per block two resnets plus a downsampler, except the deepest block
/// which has neither attention nor a downsampler.
pub fn down_pass(
    sample_nhwc: &Array,
    temb: &Array,
    context: &Array,
    cfg: &UNetConfig,
    w: &Weights,
    s: &Stream,
) -> Result<(Array, Vec<Array>)> {
    let mut h = conv(
        sample_nhwc,
        get(w, "conv_in.weight")?,
        Some(get(w, "conv_in.bias")?),
        1,
        s,
    )?;
    let mut skips = vec![h.contiguous(s)?];

    // The deepest block has neither attention nor a downsampler.
    let blocks = cfg.down_has_attention.len();
    for i in 0..blocks {
        let heads = cfg.down_has_attention[i].then(|| cfg.heads[i]);
        let (out, mut block_skips) = down_block(
            &h,
            temb,
            context,
            w,
            &format!("down_blocks.{i}"),
            cfg.layers_per_block,
            heads,
            cfg.transformer_layers[i],
            cfg.use_linear_projection,
            i + 1 < blocks,
            s,
        )?;
        h = out;
        skips.append(&mut block_skips);
    }
    Ok((h, skips))
}

/// Nearest-neighbour 2x upsample over NHWC, then diffusers' 3x3 convolution.
///
/// MLX has no upsample op, so the doubling is `broadcast_to` between two
/// reshapes: `[n,h,w,c]` -> `[n,h,1,w,1,c]` -> `[n,h,2,w,2,c]` -> `[n,2h,2w,c]`.
/// That is nearest by construction — each source pixel is copied into a 2x2
/// block — rather than by asking for an interpolation mode and hoping it is the
/// one diffusers used.
fn upsample(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let [n, h, wd, c] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: upsample got {:?}", x.shape())));
    };
    let doubled = x
        .reshape(&[n, h, 1, wd, 1, c], s)?
        .broadcast_to(&[n, h, 2, wd, 2, c], s)?
        .contiguous(s)?
        .reshape(&[n, h * 2, wd * 2, c], s)?;
    conv(
        &doubled,
        get(w, &format!("{prefix}.conv.weight"))?,
        Some(get(w, &format!("{prefix}.conv.bias"))?),
        1,
        s,
    )
}

/// The mid block: resnet, transformer, resnet.
pub fn mid_block(
    x: &Array,
    temb: &Array,
    context: &Array,
    cfg: &UNetConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let h = resnet_block(x, temb, w, "mid_block.resnets.0", s)?;
    // The mid block runs at the deepest width, so it takes the last head count.
    let heads = *cfg.heads.last().expect("at least one block");
    let layers = *cfg.transformer_layers.last().expect("at least one block");
    let h = transformer_2d(
        &h,
        context,
        heads,
        layers,
        cfg.use_linear_projection,
        w,
        "mid_block.attentions.0",
        s,
    )?;
    resnet_block(&h, temb, w, "mid_block.resnets.1", s)
}

/// One up block: `layers` resnets, each fed the concatenation of the running
/// activation with a skip popped from the stack, then an optional upsampler.
#[allow(clippy::too_many_arguments)]
pub fn up_block(
    x: &Array,
    temb: &Array,
    context: &Array,
    skips: &mut Vec<Array>,
    w: &Weights,
    prefix: &str,
    layers: usize,
    heads: Option<usize>,
    transformer_layers: usize,
    linear_projection: bool,
    has_upsample: bool,
    s: &Stream,
) -> Result<Array> {
    let mut h = x.contiguous(s)?;
    for i in 0..layers {
        let skip = skips.pop().ok_or_else(|| {
            Error::Msg("mlx: the up pass ran out of skips; the stack is the wrong depth".into())
        })?;
        // Channels last, so the join is on the last axis rather than dim 1.
        h = concat(&[&h, &skip], 3, s)?;
        h = resnet_block(&h, temb, w, &format!("{prefix}.resnets.{i}"), s)?;
        if let Some(heads) = heads {
            h = transformer_2d(
                &h,
                context,
                heads,
                transformer_layers,
                linear_projection,
                w,
                &format!("{prefix}.attentions.{i}"),
                s,
            )?;
        }
    }
    if has_upsample {
        h = upsample(&h, w, &format!("{prefix}.upsamplers.0"), s)?;
    }
    Ok(h)
}

/// SD 1.5's UNet, end to end, in NHWC.
///
/// `sample_nhwc` is `[n, h, w, 4]` and the result is `[n, h, w, 4]`. Callers
/// holding diffusers' NCHW transpose on the way in and out.
pub fn unet_forward(
    sample_nhwc: &Array,
    timestep: &Array,
    context: &Array,
    cfg: &UNetConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    unet_forward_with(sample_nhwc, timestep, context, None, None, cfg, w, s)
}

/// `unet_forward` plus SDXL's micro-conditioning: the pooled text embedding
/// `[n, 1280]` and the six time ids `[n, 6]`.
///
/// **Pooled first, then the sinusoid.** The halves are 1280 and 1536, so either
/// order sums to 2816 and loads and runs — the reversed one just conditions on
/// nonsense.
#[allow(clippy::too_many_arguments)]
pub fn unet_forward_with(
    sample_nhwc: &Array,
    timestep: &Array,
    context: &Array,
    added: Option<(&Array, &Array)>,
    class_embeds: Option<&Array>,
    cfg: &UNetConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let temb = timestep_embedding(timestep, 320, w, s)?;
    let temb = match (cfg.addition, added) {
        (Some(add), Some((pooled, time_ids))) => {
            let [n, ids] = time_ids.shape()[..] else {
                return Err(Error::Msg(format!(
                    "mlx: time_ids should be [n, 6], got {:?}",
                    time_ids.shape()
                )));
            };
            // Each id gets its own sinusoid, then they are flattened.
            let flat = time_ids.reshape(&[n * ids], s)?;
            let sinusoid = sinusoid_embedding(&flat, add.time_embed_dim, s)?
                .reshape(&[n, ids * add.time_embed_dim], s)?;
            let combined = concat(&[pooled, &sinusoid], 1, s)?;
            let projected = linear(
                &combined,
                get(w, "add_embedding.linear_1.weight")?,
                Some(get(w, "add_embedding.linear_1.bias")?),
                s,
            )?
            .silu(s)?;
            let projected = linear(
                &projected,
                get(w, "add_embedding.linear_2.weight")?,
                Some(get(w, "add_embedding.linear_2.bias")?),
                s,
            )?;
            // Added to the timestep embedding, not concatenated with it.
            temb.add(&projected, s)?
        }
        (Some(_), None) => {
            return Err(Error::Msg(
                "mlx: this UNet expects SDXL micro-conditioning".into(),
            ))
        }
        (None, Some(_)) => {
            return Err(Error::Msg(
                "mlx: micro-conditioning supplied to a UNet that has no add_embedding".into(),
            ))
        }
        (None, None) => temb,
    };

    // unCLIP's image conditioning enters the same slot the micro-conditioning
    // does — added to the timestep embedding — but from a different tensor.
    let temb = match (cfg.class_projection, class_embeds) {
        (true, Some(v)) => {
            let h = linear(
                v,
                get(w, "class_embedding.linear_1.weight")?,
                Some(get(w, "class_embedding.linear_1.bias")?),
                s,
            )?
            .silu(s)?;
            let h = linear(
                &h,
                get(w, "class_embedding.linear_2.weight")?,
                Some(get(w, "class_embedding.linear_2.bias")?),
                s,
            )?;
            temb.add(&h, s)?
        }
        (true, None) => {
            return Err(Error::Msg(
                "mlx: this UNet expects an unCLIP image embedding".into(),
            ))
        }
        (false, Some(_)) => {
            return Err(Error::Msg(
                "mlx: an image embedding was supplied to a UNet with no class_embedding".into(),
            ))
        }
        (false, None) => temb,
    };
    let (h, mut skips) = down_pass(sample_nhwc, &temb, context, cfg, w, s)?;
    let mut h = mid_block(&h, &temb, context, cfg, w, s)?;

    // UpBlock2D first, then three CrossAttnUpBlock2D — the reverse of the down
    // pass, and the deepest block is the one without attention. Three resnets
    // each, one more than the down side, because the extra one consumes
    // conv_in's skip at the end.
    // The up pass mirrors the down pass, so its attention flags and head counts
    // are the down-side ones reversed.
    let blocks = cfg.down_has_attention.len();
    for i in 0..blocks {
        let mirrored = blocks - 1 - i;
        let heads = cfg.down_has_attention[mirrored].then(|| cfg.heads[mirrored]);
        h = up_block(
            &h,
            &temb,
            context,
            &mut skips,
            w,
            &format!("up_blocks.{i}"),
            cfg.layers_per_block + 1,
            heads,
            cfg.transformer_layers[mirrored],
            cfg.use_linear_projection,
            i + 1 < blocks,
            s,
        )?;
    }
    if !skips.is_empty() {
        return Err(Error::Msg(format!(
            "mlx: the up pass left {} skips unconsumed",
            skips.len()
        )));
    }

    let h = h.group_norm(
        NORM_GROUPS,
        RESNET_EPS,
        Some(get(w, "conv_norm_out.weight")?),
        Some(get(w, "conv_norm_out.bias")?),
        s,
    )?;
    let h = h.silu(s)?;
    conv(
        &h,
        get(w, "conv_out.weight")?,
        Some(get(w, "conv_out.bias")?),
        1,
        s,
    )
}
