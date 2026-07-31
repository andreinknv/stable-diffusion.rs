//! The txt2img loop on MLX: prompt embeddings and a latent in, an image out.
//!
//! The schedule itself is *not* reimplemented. `sd_sample::Schedule` and
//! `sigmas_for_steps` return `Vec<f64>` and touch no tensors, so they are
//! scalar mathematics rather than a backend concern — the candle path and this
//! one call the same functions and cannot drift apart. Only the per-step tensor
//! arithmetic is written here.
//!
//! Matches `Txt2ImgPipeline::denoise_inner`, whose sequence per step is:
//!
//! 1. `cat([latent, latent])`, then divide by `sqrt(sigma^2 + 1)` — the
//!    k-diffusion input scaling. Omitting it gives noisy, oversaturated output.
//! 2. UNet at the timestep nearest `sigma` in the training schedule.
//! 3. `uncond + (cond - uncond) * cfg_scale`.
//! 4. `denoised = latent - noise_pred * sigma`, which is epsilon prediction.
//! 5. One Euler-ancestral step.

use sd_tensor::mlx::{concat, Array, Stream};
use sd_tensor::Result;

/// One Euler-ancestral step, the arithmetic of
/// `sd_sample::euler_ancestral_step` on MLX arrays.
///
/// `noise` is ignored when `sigma_next == 0.0`: the last step lands on the
/// clean image and must not have noise added back. Scalar work stays in f64,
/// as it does on the candle side; only the tensors are f32.
pub fn euler_ancestral_step(
    x: &Array,
    denoised: &Array,
    sigma: f64,
    sigma_next: f64,
    noise: &Array,
    s: &Stream,
) -> Result<Array> {
    // sigma == 0 means there is nothing left to denoise, and dividing by it
    // would produce NaN rather than an error.
    if sigma == 0.0 {
        return x.contiguous(s);
    }

    // The `min` matters: without it `sigma_up` is fine for most steps and
    // wrong near the end, where sigma_next approaches zero.
    let sigma_up = sigma_next
        .min((sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2)).sqrt());
    let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).max(0.0).sqrt();

    let d = x
        .sub(denoised, s)?
        .div(&Array::scalar_f32(sigma as f32)?, s)?;
    let x = x.add(
        &d.mul(&Array::scalar_f32((sigma_down - sigma) as f32)?, s)?,
        s,
    )?;

    if sigma_next > 0.0 {
        x.add(&noise.mul(&Array::scalar_f32(sigma_up as f32)?, s)?, s)
    } else {
        Ok(x)
    }
}

/// Classifier-free guidance over a `[2, ...]` batch, unconditional row first.
pub fn guidance(batched: &Array, cfg_scale: f64, s: &Stream) -> Result<Array> {
    let uncond = batched.narrow(0, 0, 1, s)?;
    let cond = batched.narrow(0, 1, 1, s)?;
    cond.sub(&uncond, s)?
        .mul(&Array::scalar_f32(cfg_scale as f32)?, s)?
        .add(&uncond, s)
}

/// The k-diffusion input scaling, applied to the doubled guidance batch.
pub fn scale_model_input(latent: &Array, sigma: f64, s: &Stream) -> Result<Array> {
    let doubled = concat(&[latent, latent], 0, s)?;
    doubled.div(&Array::scalar_f32((sigma * sigma + 1.0).sqrt() as f32)?, s)
}

/// Epsilon prediction: `denoised = latent - output * sigma`.
///
/// SD 1.5 is an epsilon model. **The v-prediction branch is not ported yet**,
/// and a v-model fed through here produces a plausible wrong image rather than
/// an error — so the name says `epsilon` instead of leaving the caller to
/// infer which parameterisation this is. `Prediction::V` on the candle side is
/// `x/(1 + sigma^2) - v * sigma/sqrt(1 + sigma^2)`.
pub fn denoise_epsilon(latent: &Array, output: &Array, sigma: f64, s: &Stream) -> Result<Array> {
    latent.sub(&output.mul(&Array::scalar_f32(sigma as f32)?, s)?, s)
}
