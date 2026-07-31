//! The VAE decoder on MLX, gated on `tests/golden/vae_decoder`.
//!
//! Latents in, image out, which is the second half of every pipeline in this
//! repository. Same NHWC convention as the UNet beside it.
//!
//! **The VAE's epsilon is 1e-6, the UNet's resnets are 1e-5.** `vae/mod.rs`
//! sets `norm_eps: 1e-6` and `unet/attention.rs` spells out why the two must
//! not be unified. This module keeps its own constant rather than reaching for
//! the UNet's.
//!
//! **The resnet here has no time embedding.** It is the same shape as the
//! UNet's otherwise — norm, silu, conv, norm, silu, conv, skip — but a VAE has
//! no timestep to condition on, so sharing the UNet's block would mean passing
//! a zero and hoping. It is written out instead.

use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{Error, Result};

use super::{conv, get, linear, Weights, NORM_GROUPS};

/// GroupNorm epsilon throughout the VAE. Not the UNet's 1e-5.
pub const VAE_EPS: f32 = 1e-6;

/// A VAE resnet: no time embedding, otherwise the UNet's shape.
fn resnet(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let p = |name: &str| format!("{prefix}.{name}");

    let h = x
        .group_norm(
            NORM_GROUPS,
            VAE_EPS,
            Some(get(w, &p("norm1.weight"))?),
            Some(get(w, &p("norm1.bias"))?),
            s,
        )?
        .silu(s)?;
    let h = conv(
        &h,
        get(w, &p("conv1.weight"))?,
        Some(get(w, &p("conv1.bias"))?),
        1,
        s,
    )?;

    let h = h
        .group_norm(
            NORM_GROUPS,
            VAE_EPS,
            Some(get(w, &p("norm2.weight"))?),
            Some(get(w, &p("norm2.bias"))?),
            s,
        )?
        .silu(s)?;
    let h = conv(
        &h,
        get(w, &p("conv2.weight"))?,
        Some(get(w, &p("conv2.bias"))?),
        1,
        s,
    )?;

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

/// The VAE's spatial self-attention: one head over `h*w` positions.
///
/// Unlike the UNet's, `to_q`/`to_k`/`to_v` carry bias here. The attention is
/// over space rather than tokens, so the sequence is the flattened spatial
/// grid and there is a single head.
fn attention(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let p = |name: &str| format!("{prefix}.{name}");
    let [n, h, wd, c] = x.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: vae attention got {:?}",
            x.shape()
        )));
    };

    let y = x.group_norm(
        NORM_GROUPS,
        VAE_EPS,
        Some(get(w, &p("group_norm.weight"))?),
        Some(get(w, &p("group_norm.bias"))?),
        s,
    )?;
    // NHWC already has channels last, so flattening space needs no transpose —
    // the candle path has to permute here and this does not.
    let seq = y.reshape(&[n, h * wd, c], s)?;

    let q = linear(&seq, get(w, &p("to_q.weight"))?, w.get(&p("to_q.bias")), s)?;
    let k = linear(&seq, get(w, &p("to_k.weight"))?, w.get(&p("to_k.bias")), s)?;
    let v = linear(&seq, get(w, &p("to_v.weight"))?, w.get(&p("to_v.bias")), s)?;

    // One head: [n, 1, hw, c].
    let head = |t: &Array| -> Result<Array> { t.reshape(&[n, 1, h * wd, c], s) };
    let out = head(&q)?.sdpa(&head(&k)?, &head(&v)?, 1.0 / (c as f32).sqrt(), s)?;
    let out = out.reshape(&[n, h * wd, c], s)?;

    let out = linear(
        &out,
        get(w, &p("to_out.0.weight"))?,
        w.get(&p("to_out.0.bias")),
        s,
    )?;
    out.reshape(&[n, h, wd, c], s)?.add(x, s)
}

/// Nearest 2x upsample, then a 3x3 convolution — diffusers' `Upsample2D`.
fn upsample(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let [n, h, wd, c] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: vae upsample got {:?}", x.shape())));
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

/// The decoder's mid block: resnet, attention, resnet.
pub fn mid_block(x: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    let h = resnet(x, w, "decoder.mid_block.resnets.0", s)?;
    let h = attention(&h, w, "decoder.mid_block.attentions.0", s)?;
    resnet(&h, w, "decoder.mid_block.resnets.1", s)
}

/// One decoder up block: three resnets, then an optional upsampler.
pub fn up_block(
    x: &Array,
    w: &Weights,
    index: usize,
    has_upsample: bool,
    s: &Stream,
) -> Result<Array> {
    let mut h = x.contiguous(s)?;
    for i in 0..3 {
        h = resnet(&h, w, &format!("decoder.up_blocks.{index}.resnets.{i}"), s)?;
    }
    if has_upsample {
        h = upsample(&h, w, &format!("decoder.up_blocks.{index}.upsamplers.0"), s)?;
    }
    Ok(h)
}

/// `post_quant_conv`, then the decoder, from a latent to an image.
///
/// `latent_nhwc` is `[n, h, w, 4]`; the result is `[n, 8h, 8w, 3]` in the VAE's
/// own range, before any scaling the pipeline applies.
pub fn decode(latent_nhwc: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    // 1x1, so no padding.
    let h = conv(
        latent_nhwc,
        get(w, "post_quant_conv.weight")?,
        Some(get(w, "post_quant_conv.bias")?),
        0,
        s,
    )?;
    let h = conv(
        &h,
        get(w, "decoder.conv_in.weight")?,
        Some(get(w, "decoder.conv_in.bias")?),
        1,
        s,
    )?;
    let h = mid_block(&h, w, s)?;

    // 512 -> 512 -> 512 -> 256 -> 128, upsampling on all but the last.
    let mut h = h;
    for (i, has_up) in [true, true, true, false].into_iter().enumerate() {
        h = up_block(&h, w, i, has_up, s)?;
    }

    let h = h
        .group_norm(
            NORM_GROUPS,
            VAE_EPS,
            Some(get(w, "decoder.conv_norm_out.weight")?),
            Some(get(w, "decoder.conv_norm_out.bias")?),
            s,
        )?
        .silu(s)?;
    conv(
        &h,
        get(w, "decoder.conv_out.weight")?,
        Some(get(w, "decoder.conv_out.bias")?),
        1,
        s,
    )
}

/// `AutoencoderKL`'s encoder: an image in, the latent distribution's moments
/// out.
///
/// `conv_out` emits **twice** `latent_channels` — the first half is the mean,
/// the second the log-variance. Taking the wrong half loads fine and yields
/// noise.
///
/// Down blocks have `layers_per_block` resnets, one *fewer* than the decoder's
/// up blocks.
pub fn encode_moments(image_nhwc: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    let h = conv(
        image_nhwc,
        get(w, "encoder.conv_in.weight")?,
        Some(get(w, "encoder.conv_in.bias")?),
        1,
        s,
    )?;

    // 128 -> 128 -> 256 -> 512 -> 512, downsampling on all but the last.
    let mut h = h;
    for (i, has_down) in [true, true, true, false].into_iter().enumerate() {
        for j in 0..2 {
            h = resnet(&h, w, &format!("encoder.down_blocks.{i}.resnets.{j}"), s)?;
        }
        if has_down {
            h = downsample(&h, w, &format!("encoder.down_blocks.{i}.downsamplers.0"), s)?;
        }
    }

    let h = resnet(&h, w, "encoder.mid_block.resnets.0", s)?;
    let h = attention(&h, w, "encoder.mid_block.attentions.0", s)?;
    let h = resnet(&h, w, "encoder.mid_block.resnets.1", s)?;

    let h = h
        .group_norm(
            NORM_GROUPS,
            VAE_EPS,
            Some(get(w, "encoder.conv_norm_out.weight")?),
            Some(get(w, "encoder.conv_norm_out.bias")?),
            s,
        )?
        .silu(s)?;
    let h = conv(
        &h,
        get(w, "encoder.conv_out.weight")?,
        Some(get(w, "encoder.conv_out.bias")?),
        1,
        s,
    )?;
    // 1x1, so no padding.
    conv(
        &h,
        get(w, "quant_conv.weight")?,
        Some(get(w, "quant_conv.bias")?),
        0,
        s,
    )
}

/// The encoder's stride-2 downsample.
///
/// **Asymmetric padding**: one row at the bottom and one column at the right,
/// none at the top or left, which is `Downsample2D(padding=0)` followed by
/// `F.pad(x, (0, 1, 0, 1))` in diffusers. A symmetric `padding: 1` runs,
/// produces the right shape, and shifts the whole image half a pixel per
/// downsample — `docs/handoff.md` records that bug measuring 17.32.
///
/// NHWC, so the spatial axes are 1 and 2 rather than 2 and 3.
fn downsample(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let padded = x.pad(&[1, 2], &[0, 0], &[1, 1], s)?;
    super::conv_strided(
        &padded,
        get(w, &format!("{prefix}.conv.weight"))?,
        Some(get(w, &format!("{prefix}.conv.bias"))?),
        2,
        0,
        s,
    )
}
