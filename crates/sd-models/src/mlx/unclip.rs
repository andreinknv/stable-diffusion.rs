//! unCLIP's noise augmentation on MLX.
//!
//! The UNet conditions on a CLIP *image* embedding, and it was trained on
//! embeddings that had been deliberately noised by a stated amount. This is
//! that: whiten, mix with noise, un-whiten, then append the level's sinusoid so
//! the model knows how much it was given.
//!
//! # Three things that keep the shape and change the meaning
//!
//! - **The un-whitening matters.** The UNet's `class_embedding` was trained on
//!   embeddings in CLIP's own units, not in the normalizer's.
//! - **The two halves are the same width**, so a reversed concatenation
//!   produces exactly the right shape and conditions the model on a sinusoid
//!   where it expects a picture. Order is: embedding first, then the level.
//! - **The unconditional row is zeros of the whole `2 * dim`**, not an
//!   augmented zero embedding and not an absent argument. An unCLIP UNet always
//!   projects something into its timestep embedding, and what diffusers hands
//!   it for "no image" is a zero vector including the half that would carry the
//!   noise level.

use sd_tensor::mlx::{concat, Array, Stream};
use sd_tensor::{Error, Result};

use super::{get, sinusoid_embedding, Weights};

/// unCLIP's training schedule length.
pub const TRAIN_TIMESTEPS: usize = 1000;

/// `(x - mean) / std`, from the checkpoint's `image_normalizer`.
///
/// The tensors are the bare names `mean` and `std`, not nested under a module.
pub fn scale(embeds: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    embeds.sub(get(w, "mean")?, s)?.div(get(w, "std")?, s)
}

/// `x * std + mean`, exactly undoing [`scale`].
pub fn unscale(embeds: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    embeds.mul(get(w, "std")?, s)?.add(get(w, "mean")?, s)
}

/// Noise an image embedding and say by how much. `[b, dim]` -> `[b, 2 * dim]`.
///
/// `noise` is `[b, dim]` standard normal, supplied rather than drawn so the
/// caller owns its seed sequence. `level` indexes the schedule and is clamped
/// to it.
pub fn augment(
    image_embeds: &Array,
    level: usize,
    noise: &Array,
    alphas_cumprod: &[f64],
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let [b, dim] = image_embeds.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: an image embedding should be [b, dim], got {:?}",
            image_embeds.shape()
        )));
    };
    let level = level.min(alphas_cumprod.len().saturating_sub(1));
    let alpha = alphas_cumprod[level];

    // Whiten, mix, un-whiten.
    let scaled = scale(image_embeds, w, s)?;
    let noisy = scaled
        .mul(&Array::scalar_f32(alpha.sqrt() as f32)?, s)?
        .add(
            &noise.mul(&Array::scalar_f32((1.0 - alpha).sqrt() as f32)?, s)?,
            s,
        )?;
    let noisy = unscale(&noisy, w, s)?;

    // The same sinusoid the timestep takes, at the embedding's width — cosine
    // first, `flip_sin_to_cos=True`.
    let levels = Array::from_slice_f32(&vec![level as f32; b], &[b])?;
    let embedded = sinusoid_embedding(&levels, dim, s)?;

    concat(&[&noisy, &embedded], 1, s)
}

/// The unconditional row of a guidance batch: zeros, `[b, 2 * dim]`.
pub fn unconditional(batch: usize, dim: usize) -> Result<Array> {
    Array::from_slice_f32(&vec![0.0; batch * dim * 2], &[batch, dim * 2])
}
