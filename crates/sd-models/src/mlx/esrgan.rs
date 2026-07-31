//! Real-ESRGAN on MLX — 4x upscaling, RRDBNet.
//!
//! A pure convolutional network: no attention, no normalisation of any kind,
//! no diffusion. It runs *after* generation, so it composes with every
//! architecture here and knows about none of them.
//!
//! **Two residual scalings, both 0.2, both load-bearing.** Every dense block
//! returns `x5 * 0.2 + x` and every RRDB returns `inner * 0.2 + x`. The factor
//! keeps a 23-block stack of unnormalised residuals from diverging — there is
//! no GroupNorm here to rein it in — and dropping either gives a washed-out or
//! blown-out image rather than an error. They compound: 23 RRDBs of 3 dense
//! blocks each applies the scaling 92 times.
//!
//! **Dense, not sequential.** Inside a dense block each convolution sees every
//! previous output concatenated, so input widths climb 64, 96, 128, 160, 192
//! while outputs stay at 32 (except the last, back to 64). Feeding each
//! convolution only its predecessor's output loads cleanly for the first layer
//! and fails on the second's channel count.
//!
//! **Range is `[0, 1]`**, not the `[-1, 1]` images use elsewhere in this crate.

use sd_tensor::mlx::{concat, Array, Stream};
use sd_tensor::{Error, Result};

use super::{conv, get, Weights};

/// Both residual scalings.
const RESIDUAL_SCALE: f32 = 0.2;
/// LeakyReLU's slope. **0.2 here** — the usual default of 0.01 would be a
/// silent change to every activation.
const LEAK: f32 = 0.2;
/// Dense blocks per RRDB, and RRDBs in the trunk.
const DENSE_PER_RRDB: usize = 3;
const RRDB_COUNT: usize = 23;

/// `max(x, 0) + slope * min(x, 0)`, with the slope visible at the point of use.
fn leaky_relu(x: &Array, s: &Stream) -> Result<Array> {
    let zero = Array::scalar_f32(0.0)?;
    let pos = x.maximum(&zero, s)?;
    let neg = zero.maximum(&x.mul(&Array::scalar_f32(-1.0)?, s)?, s)?;
    // `neg` is -min(x, 0); subtract its scaled value rather than negate twice.
    pos.sub(&neg.mul(&Array::scalar_f32(LEAK)?, s)?, s)
}

/// Five convolutions, each seeing every previous output concatenated.
fn dense_block(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let mut parts = vec![x.contiguous(s)?];
    for i in 0..5 {
        let input = if parts.len() == 1 {
            parts[0].contiguous(s)?
        } else {
            // Channels last, so the join is on axis 3.
            let refs: Vec<&Array> = parts.iter().collect();
            concat(&refs, 3, s)?
        };
        let out = conv(
            &input,
            get(w, &format!("{prefix}.conv{}.weight", i + 1))?,
            w.get(&format!("{prefix}.conv{}.bias", i + 1)),
            1,
            s,
        )?;
        // The last convolution is **not** activated — it is the residual.
        parts.push(if i == 4 { out } else { leaky_relu(&out, s)? });
    }
    let last = parts.pop().expect("five convolutions ran");
    last.mul(&Array::scalar_f32(RESIDUAL_SCALE)?, s)?.add(x, s)
}

/// Three dense blocks, then the same 0.2 residual again.
fn rrdb(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let mut h = x.contiguous(s)?;
    for i in 0..DENSE_PER_RRDB {
        h = dense_block(&h, w, &format!("{prefix}.rdb{}", i + 1), s)?;
    }
    h.mul(&Array::scalar_f32(RESIDUAL_SCALE)?, s)?.add(x, s)
}

/// Nearest-neighbour 2x over NHWC.
///
/// **Nearest, not bilinear**: the convolution after it is what learns the
/// interpolation, and a smooth upsample changes what it receives.
fn nearest2x(x: &Array, s: &Stream) -> Result<Array> {
    let [n, h, wd, c] = x.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: esrgan upsample got {:?}",
            x.shape()
        )));
    };
    x.reshape(&[n, h, 1, wd, 1, c], s)?
        .broadcast_to(&[n, h, 2, wd, 2, c], s)?
        .contiguous(s)?
        .reshape(&[n, h * 2, wd * 2, c], s)
}

/// Upscale 4x. Takes and returns NHWC in `[0, 1]`.
pub fn upscale(image_nhwc: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    let feat = conv(
        image_nhwc,
        get(w, "conv_first.weight")?,
        w.get("conv_first.bias"),
        1,
        s,
    )?;

    let mut h = feat.contiguous(s)?;
    for i in 0..RRDB_COUNT {
        h = rrdb(&h, w, &format!("body.{i}"), s)?;
    }
    // The trunk's long skip: the body's contribution is **added** to
    // conv_first's output, not used in place of it.
    let mut h = feat.add(
        &conv(
            &h,
            get(w, "conv_body.weight")?,
            w.get("conv_body.bias"),
            1,
            s,
        )?,
        s,
    )?;

    for name in ["conv_up1", "conv_up2"] {
        let up = nearest2x(&h, s)?;
        h = leaky_relu(
            &conv(
                &up,
                get(w, &format!("{name}.weight"))?,
                w.get(&format!("{name}.bias")),
                1,
                s,
            )?,
            s,
        )?;
    }

    let hr = leaky_relu(
        &conv(&h, get(w, "conv_hr.weight")?, w.get("conv_hr.bias"), 1, s)?,
        s,
    )?;
    conv(
        &hr,
        get(w, "conv_last.weight")?,
        w.get("conv_last.bias"),
        1,
        s,
    )
}
