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

/// What differs between the VAEs in this repository.
///
/// The convolutional geometry does not — `[128, 256, 512, 512]`, two layers a
/// block, 32 groups — so this is a parameterisation rather than a family of
/// models. What differs is the latent width, whether the 1x1 quant convolutions
/// exist, and how the latent is scaled.
#[derive(Debug, Clone, Copy)]
pub struct VaeConfig {
    pub latent_channels: usize,
    /// **False for Flux.** Building them anyway looks for weights that do not
    /// exist, which is loud; *not* building them when they do exist silently
    /// drops a 1x1 convolution, which is not.
    pub use_quant_conv: bool,
    pub scaling_factor: f32,
    /// **Shift first, then scale**, and the decode inverts it in the opposite
    /// order. Getting either backwards leaves the image recognisable with wrong
    /// contrast — the failure that survives eyeballing.
    pub shift_factor: f32,
}

impl VaeConfig {
    pub fn sd15() -> Self {
        Self {
            latent_channels: 4,
            use_quant_conv: true,
            scaling_factor: 0.18215,
            shift_factor: 0.0,
        }
    }

    /// SDXL's, which differs from SD 1.5's only in `scaling_factor`.
    pub fn sdxl() -> Self {
        Self {
            scaling_factor: 0.13025,
            ..Self::sd15()
        }
    }

    /// Flux: a 16-channel latent, no quant convolutions, and a shift.
    ///
    /// The wider latent is why Flux images hold fine detail SD's 4-channel one
    /// cannot represent, and it costs nothing here because the encoder and
    /// decoder are already parameterised by it.
    pub fn flux() -> Self {
        Self {
            latent_channels: 16,
            use_quant_conv: false,
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }

    /// SD 3.5: Flux's geometry with its own scale and shift.
    pub fn sd35() -> Self {
        Self {
            scaling_factor: 1.5305,
            shift_factor: 0.0609,
            ..Self::flux()
        }
    }

    /// `(x - shift) * scale` — a raw latent to the sampler's.
    pub fn scale(&self, latent: &Array, s: &Stream) -> Result<Array> {
        latent
            .sub(&Array::scalar_f32(self.shift_factor)?, s)?
            .mul(&Array::scalar_f32(self.scaling_factor)?, s)
    }

    /// `x / scale + shift` — the inverse, in the opposite order.
    pub fn unscale(&self, latent: &Array, s: &Stream) -> Result<Array> {
        latent
            .div(&Array::scalar_f32(self.scaling_factor)?, s)?
            .add(&Array::scalar_f32(self.shift_factor)?, s)
    }
}

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
    decode_with(latent_nhwc, &VaeConfig::sd15(), w, s)
}

/// [`decode`] for any of the VAEs in [`VaeConfig`].
pub fn decode_with(latent_nhwc: &Array, cfg: &VaeConfig, w: &Weights, s: &Stream) -> Result<Array> {
    // 1x1, so no padding. Absent on Flux.
    let h = if cfg.use_quant_conv {
        conv(
            latent_nhwc,
            get(w, "post_quant_conv.weight")?,
            Some(get(w, "post_quant_conv.bias")?),
            0,
            s,
        )?
    } else {
        latent_nhwc.contiguous(s)?
    };
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
    encode_moments_with(image_nhwc, &VaeConfig::sd15(), w, s)
}

/// [`encode_moments`] for any of the VAEs in [`VaeConfig`].
pub fn encode_moments_with(
    image_nhwc: &Array,
    cfg: &VaeConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
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
    // 1x1, so no padding. Absent on Flux.
    if !cfg.use_quant_conv {
        return Ok(h);
    }
    conv(
        &h,
        get(w, "quant_conv.weight")?,
        Some(get(w, "quant_conv.bias")?),
        0,
        s,
    )
}

/// The latent distribution's `(mean, logvar)`, each `[n, h/8, w/8, 4]`.
///
/// `encode_moments` emits both halves stacked on the channel axis; splitting
/// them is the caller's job everywhere else in this codebase, and getting the
/// halves backwards yields noise that loads fine.
pub fn encode_dist(image_nhwc: &Array, w: &Weights, s: &Stream) -> Result<(Array, Array)> {
    encode_dist_with(image_nhwc, &VaeConfig::sd15(), w, s)
}

/// [`encode_dist`] for any of the VAEs in [`VaeConfig`].
pub fn encode_dist_with(
    image_nhwc: &Array,
    cfg: &VaeConfig,
    w: &Weights,
    s: &Stream,
) -> Result<(Array, Array)> {
    let moments = encode_moments_with(image_nhwc, cfg, w, s)?;
    let c = moments.shape()[3];
    if c % 2 != 0 {
        return Err(Error::Msg(format!(
            "mlx: moments should have an even channel count, got {c}"
        )));
    }
    let half = c / 2;
    Ok((
        moments.narrow(3, 0, half, s)?.contiguous(s)?,
        moments.narrow(3, half, half, s)?.contiguous(s)?,
    ))
}

/// The distribution's **mean**, unscaled — what img2img encodes with.
///
/// Not a draw from the distribution. The sampler supplies all the randomness,
/// so drawing here too would add variance the seed does not control and make
/// the run irreproducible. `encode_sampled` on the candle side exists for the
/// cases that genuinely want a draw; nothing in a pipeline does.
pub fn encode(image_nhwc: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    Ok(encode_dist(image_nhwc, w, s)?.0)
}

/// [`encode`] for any of the VAEs in [`VaeConfig`], with the scaling applied.
///
/// This is the form a pipeline wants: the latent the sampler operates on, not
/// the distribution the model expresses.
pub fn encode_scaled(
    image_nhwc: &Array,
    cfg: &VaeConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let (mean, _) = encode_dist_with(image_nhwc, cfg, w, s)?;
    cfg.scale(&mean, s)
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
