//! TAESD on MLX — a tiny distilled autoencoder, interchangeable with the VAE.
//!
//! About 5 MB of 3x3 convolutions: no attention, no GroupNorm, no sampling
//! head. Lossier than the VAE and cheap enough to decode every step, which is
//! what makes step previews possible at all.
//!
//! **Its latent convention is its own.** The SD VAE expects `latent / 0.18215`;
//! TAESD's `scaling_factor` is 1.0, so it takes the sampler's latent unscaled.
//! Applying the VAE's factor here multiplies the input by 5.5 and produces a
//! washed-out image — a plausible picture, no error, and nothing downstream
//! that can tell. So the scaling stays out of this module entirely.
//!
//! Three more conversions are load-bearing for the same reason: the decoder
//! soft-clamps its input with `tanh(x/3)*3` and returns `2x - 1`, and the
//! encoder takes `(x + 1) / 2`. Omitting any of them is silently wrong.

use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{Error, Result};

use super::{conv, conv_strided, get, Weights};

/// Residual blocks per stage, decoder order. The encoder's is this reversed.
const DECODER_BLOCKS: [usize; 4] = [3, 3, 3, 1];
const ENCODER_BLOCKS: [usize; 4] = [1, 3, 3, 3];
/// The soft clamp's half-range. `tanh(x/3)*3` maps all of R into `[-3, 3]`.
const CLAMP: f32 = 3.0;

/// `conv.0 -> relu -> conv.2 -> relu -> conv.4`, then `(h + x).relu()`.
///
/// The odd indices are the ReLUs, which carry no weights — hence 0, 2, 4. The
/// fuse ReLU is applied **after** the residual add, not before it.
fn block(x: &Array, w: &Weights, prefix: &str, s: &Stream) -> Result<Array> {
    let c = |i: usize, src: &Array| -> Result<Array> {
        conv(
            src,
            get(w, &format!("{prefix}.conv.{i}.weight"))?,
            w.get(&format!("{prefix}.conv.{i}.bias")),
            1,
            s,
        )
    };
    let h = c(0, x)?.relu(s)?;
    let h = c(2, &h)?.relu(s)?;
    let h = c(4, &h)?;
    h.add(x, s)?.relu(s)
}

/// Nearest-neighbour 2x over NHWC, with no convolution after it.
fn nearest2x(x: &Array, s: &Stream) -> Result<Array> {
    let [n, h, wd, c] = x.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: taesd upsample got {:?}",
            x.shape()
        )));
    };
    x.reshape(&[n, h, 1, wd, 1, c], s)?
        .broadcast_to(&[n, h, 2, wd, 2, c], s)?
        .contiguous(s)?
        .reshape(&[n, h * 2, wd * 2, c], s)
}

/// Latent to image. Takes the sampler's latent **unscaled**.
pub fn decode(latent_nhwc: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    let p = |i: usize| format!("decoder.layers.{i}");

    // Soft clamp into [-3, 3]. TAESD was distilled with it in place, so an
    // out-of-range latent the VAE would render as a bright artefact is instead
    // squashed — and removing it changes ordinary output too, because tanh is
    // not the identity anywhere.
    let clamp = Array::scalar_f32(CLAMP)?;
    let mut h = latent_nhwc.div(&clamp, s)?.tanh(s)?.mul(&clamp, s)?;

    let mut i = 0usize;
    h = conv(
        &h,
        get(w, &format!("{}.weight", p(i)))?,
        w.get(&format!("{}.bias", p(i))),
        1,
        s,
    )?
    .relu(s)?;
    i += 2; // 1 is that ReLU, which carries no weights.

    for (stage, &count) in DECODER_BLOCKS.iter().enumerate() {
        let is_final = stage == DECODER_BLOCKS.len() - 1;
        for _ in 0..count {
            h = block(&h, w, &p(i), s)?;
            i += 1;
        }
        if !is_final {
            h = nearest2x(&h, s)?;
            i += 1; // the upsample carries no weights
        }
        // Only the last convolution carries a bias, and it is the one that
        // narrows to RGB.
        h = conv(
            &h,
            get(w, &format!("{}.weight", p(i)))?,
            w.get(&format!("{}.bias", p(i))),
            1,
            s,
        )?;
        i += 1;
    }

    // The stack works in [0, 1]; images elsewhere in this crate are [-1, 1].
    h.mul(&Array::scalar_f32(2.0)?, s)?
        .sub(&Array::scalar_f32(1.0)?, s)
}

/// Image to latent. Returns the sampler's latent **unscaled**.
pub fn encode(image_nhwc: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    let p = |i: usize| format!("encoder.layers.{i}");

    // The stack works in [0, 1]; images here are [-1, 1].
    let mut h = image_nhwc
        .add(&Array::scalar_f32(1.0)?, s)?
        .div(&Array::scalar_f32(2.0)?, s)?;

    let mut i = 0usize;
    for (stage, &count) in ENCODER_BLOCKS.iter().enumerate() {
        // The first convolution reads the image; the rest halve, and only the
        // first carries a bias.
        let first = stage == 0;
        h = conv_strided(
            &h,
            get(w, &format!("{}.weight", p(i)))?,
            w.get(&format!("{}.bias", p(i))),
            if first { 1 } else { 2 },
            1,
            s,
        )?;
        i += 1;
        for _ in 0..count {
            h = block(&h, w, &p(i), s)?;
            i += 1;
        }
    }
    // The final convolution narrows to the latent channels.
    conv(
        &h,
        get(w, &format!("{}.weight", p(i)))?,
        w.get(&format!("{}.bias", p(i))),
        1,
        s,
    )
}
