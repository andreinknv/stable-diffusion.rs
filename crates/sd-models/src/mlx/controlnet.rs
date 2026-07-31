//! ControlNet on MLX — a copy of the UNet's down stack that emits corrections.
//!
//! The corrections are added into the UNet's skips and mid output, so a
//! ControlNet must be built from the *same* config as the UNet it steers.
//! Deriving the two separately is how they drift.
//!
//! **The hint is added to `conv_in`'s output, not concatenated.** A
//! ControlNet's `conv_in` takes the same four latent channels the UNet's does,
//! and the hint arrives already projected to its width by
//! `controlnet_cond_embedding` — three stride-2 convolutions, which is exactly
//! the VAE's 8x reduction reached a different way.
//!
//! Every output goes through a 1x1 "zero conv" before leaving. Those start at
//! zero in a freshly initialised ControlNet, which is what lets one be attached
//! to a trained UNet without disturbing it.

use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{Error, Result};

use super::{
    conv, conv_strided, down_block, get, mid_block, timestep_embedding, UNetConfig, Weights,
};

/// Widths of the hint pyramid. Three stride-2 steps, so 8x in total.
const CONDITIONING_CHANNELS: [usize; 4] = [16, 32, 96, 256];

/// Projects the hint image down to the latent resolution and `conv_in`'s width.
///
/// Pairs: one stride-1 convolution at the current width, then one stride-2 that
/// both widens and halves. No activation after the last convolution — it is the
/// zero conv, and its output is a correction rather than a feature map.
fn cond_embedding(hint_nhwc: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    let p = |n: &str| format!("controlnet_cond_embedding.{n}");
    let mut h = conv(
        hint_nhwc,
        get(w, &p("conv_in.weight"))?,
        w.get(&p("conv_in.bias")),
        1,
        s,
    )?
    .silu(s)?;

    let mut i = 0usize;
    for step in 0..CONDITIONING_CHANNELS.len() - 1 {
        let _ = step;
        for stride in [1usize, 2] {
            h = conv_strided(
                &h,
                get(w, &p(&format!("blocks.{i}.weight")))?,
                w.get(&p(&format!("blocks.{i}.bias"))),
                stride,
                1,
                s,
            )?
            .silu(s)?;
            i += 1;
        }
    }

    conv(
        &h,
        get(w, &p("conv_out.weight"))?,
        w.get(&p("conv_out.bias")),
        1,
        s,
    )
}

/// What a ControlNet contributes: one correction per UNet skip, plus one for
/// the mid block.
pub struct Control {
    pub down: Vec<Array>,
    pub mid: Array,
}

/// Run a ControlNet.
///
/// `scale` multiplies every correction; 0 contributes exactly nothing rather
/// than merely almost nothing.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    sample_nhwc: &Array,
    timestep: &Array,
    context: &Array,
    hint_nhwc: &Array,
    scale: f64,
    cfg: &UNetConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Control> {
    let temb = timestep_embedding(timestep, 320, w, s)?;

    // Added, not concatenated.
    let h = conv(
        sample_nhwc,
        get(w, "conv_in.weight")?,
        Some(get(w, "conv_in.bias")?),
        1,
        s,
    )?
    .add(&cond_embedding(hint_nhwc, w, s)?, s)?;

    let mut skips = vec![h.contiguous(s)?];
    let mut h = h;
    let blocks = cfg.down_has_attention.len();
    for i in 0..blocks {
        let heads = cfg.down_has_attention[i].then(|| cfg.heads[i]);
        let (out, mut block_skips) = down_block(
            &h,
            &temb,
            context,
            w,
            &format!("down_blocks.{i}"),
            cfg.layers_per_block,
            heads,
            cfg.transformer_layers[i],
            cfg.use_linear_projection,
            // A ControlNet has no IP-Adapter or grounding of its own; the base
            // UNet carries both.
            None,
            None,
            i + 1 < blocks,
            s,
        )?;
        h = out;
        skips.append(&mut block_skips);
    }
    let mid = mid_block(&h, &temb, context, cfg, None, None, w, s)?;

    let scale = Array::scalar_f32(scale as f32)?;
    let mut down = Vec::with_capacity(skips.len());
    for (i, skip) in skips.iter().enumerate() {
        // 1x1, so no padding.
        let projected = conv(
            skip,
            get(w, &format!("controlnet_down_blocks.{i}.weight"))?,
            w.get(&format!("controlnet_down_blocks.{i}.bias")),
            0,
            s,
        )?;
        down.push(projected.mul(&scale, s)?);
    }
    if down.len() != skips.len() {
        return Err(Error::Msg(format!(
            "mlx: ControlNet produced {} corrections for {} skips",
            down.len(),
            skips.len()
        )));
    }

    let mid = conv(
        &mid,
        get(w, "controlnet_mid_block.weight")?,
        w.get("controlnet_mid_block.bias"),
        0,
        s,
    )?
    .mul(&scale, s)?;

    Ok(Control { down, mid })
}
