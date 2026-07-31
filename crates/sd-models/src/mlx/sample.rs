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
use sd_tensor::{Error, Result};

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

/// Classifier-free guidance over a doubled batch, unconditional half first.
///
/// **Splits in half rather than taking rows 0 and 1.** For a single image the
/// two are the same thing; for an AnimateDiff clip the batch is `2 * frames`
/// and taking two rows would guide the first frame and discard the rest —
/// which returns a tensor of the wrong shape only if the caller checks, and
/// silently drops the clip if it does not.
pub fn guidance(batched: &Array, cfg_scale: f64, s: &Stream) -> Result<Array> {
    let n = batched.shape()[0];
    if n % 2 != 0 {
        return Err(Error::Msg(format!(
            "mlx: a guidance batch should be even, got {n}"
        )));
    }
    let half = n / 2;
    let uncond = batched.narrow(0, 0, half, s)?;
    let cond = batched.narrow(0, half, half, s)?;
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

/// DPM++ 2M solver state.
///
/// **Stateful and order-dependent**: each step uses the previous step's
/// `denoised`. Calling steps out of order, or reusing a solver across images
/// without [`DpmSolverPlusPlus2M::reset`], produces output that is subtly wrong
/// in a way that reads as a bad seed rather than as a bug.
#[derive(Debug, Default)]
pub struct DpmSolverPlusPlus2M {
    prev_denoised: Option<Array>,
    /// `t` from the previous step, needed for the step-size ratio.
    prev_t: Option<f64>,
}

impl DpmSolverPlusPlus2M {
    pub fn new() -> Self {
        Self::default()
    }

    /// Discard the carried state. Call between images.
    pub fn reset(&mut self) {
        self.prev_denoised = None;
        self.prev_t = None;
    }

    /// One step. Call once per step, in order.
    pub fn step(
        &mut self,
        x: &Array,
        denoised: &Array,
        sigma: f64,
        sigma_next: f64,
        s: &Stream,
    ) -> Result<Array> {
        // Final step: t_next would be -ln(0) = +inf. Branch before computing
        // rather than letting an infinity propagate — the exact result here is
        // the denoised prediction itself.
        if sigma_next == 0.0 {
            self.prev_denoised = Some(denoised.contiguous(s)?);
            self.prev_t = Some(-sigma.ln());
            return denoised.contiguous(s);
        }
        if sigma == 0.0 {
            return x.contiguous(s);
        }

        let t = -sigma.ln();
        let t_next = -sigma_next.ln();
        let h = t_next - t;

        // The denoised estimate the step actually uses: the current prediction
        // on the first step, and a second-order extrapolation thereafter.
        let d = match (&self.prev_denoised, self.prev_t) {
            (Some(prev), Some(prev_t)) => {
                let h_last = t - prev_t;
                let r = h_last / h;
                let inv = 1.0 / (2.0 * r);
                denoised
                    .mul(&Array::scalar_f32((1.0 + inv) as f32)?, s)?
                    .sub(&prev.mul(&Array::scalar_f32(inv as f32)?, s)?, s)?
            }
            // First-order fallback until there is a previous step to use.
            _ => denoised.contiguous(s)?,
        };

        let x_next = x
            .mul(&Array::scalar_f32((sigma_next / sigma) as f32)?, s)?
            .sub(
                &d.mul(&Array::scalar_f32(((-h).exp() - 1.0) as f32)?, s)?,
                s,
            )?;

        self.prev_denoised = Some(denoised.contiguous(s)?);
        self.prev_t = Some(t);
        Ok(x_next)
    }
}

// -- img2img and inpainting ------------------------------------------------
//
// `Strength` itself is not reimplemented here, for the same reason
// `sd_sample::Schedule` is not: `Strength::start_index` is arithmetic on two
// integers and touches no tensor, so the two backends call the same function
// and cannot drift. Only the tensor work is below.

/// Noise an encoded latent to the sigma the run starts at.
///
/// This is what makes strength mean something: a later start is less noise and
/// so a smaller departure from the input. `noise` must be standard normal and
/// shaped like the latent — the caller owns the draw, and with it
/// reproducibility.
pub fn noise_to_sigma(latent: &Array, noise: &Array, sigma: f64, s: &Stream) -> Result<Array> {
    latent.add(&noise.mul(&Array::scalar_f32(sigma as f32)?, s)?, s)
}

/// Reduce a pixel-resolution mask to the latent grid by 8x8 **maximum**.
///
/// **Max, not mean, and the two are not interchangeable.** One white pixel in
/// an 8x8 block means that latent cell must be free to change: a latent cell
/// is not a pixel, and averaging would give 1/64 — an almost-frozen cell,
/// producing a hard seam exactly at the mask edge, where it is most visible.
/// The cost is that repainting dilates the mask by up to one latent cell,
/// which the pixel-space composite at the end takes back.
///
/// `mask_px` is `[1, h, w, 1]` with 1 where the model may write; the result is
/// `[1, h/8, w/8, 1]`.
pub fn latent_mask(mask_px: &Array, s: &Stream) -> Result<Array> {
    let [n, h, w, c] = mask_px.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: a mask should be [n, h, w, 1], got {:?}",
            mask_px.shape()
        )));
    };
    if h % 8 != 0 || w % 8 != 0 {
        return Err(Error::Msg(format!(
            "mlx: a {h}x{w} mask does not divide into latent cells"
        )));
    }
    mask_px
        .reshape(&[n, h / 8, 8, w / 8, 8, c], s)?
        .max(&[2, 4], false, s)
}

/// Restore everything outside the mask to the original, noised to the level
/// the next step expects.
///
/// **Called inside the sampling loop, not once at the end.** That is what keeps
/// the model's context honest: it sees the true surroundings at every step, so
/// what it paints actually joins up with them. Compositing only at the end
/// produces an edit that is locally plausible and does not meet its border.
///
/// `mask` is 1 where the model may write. At the final step `sigma_next` is 0
/// and the original goes back unnoised.
pub fn restore_outside_mask(
    latent: &Array,
    init: &Array,
    mask: &Array,
    noise: &Array,
    sigma_next: f64,
    s: &Stream,
) -> Result<Array> {
    let restored = if sigma_next > 0.0 {
        noise_to_sigma(init, noise, sigma_next, s)?
    } else {
        init.contiguous(s)?
    };
    let keep = Array::scalar_f32(1.0)?.sub(mask, s)?;
    latent.mul(mask, s)?.add(&restored.mul(&keep, s)?, s)
}

// -- step caching -----------------------------------------------------------

/// Relative L1 distance, `|a - b|_1 / |b|_1`.
///
/// **Relative rather than absolute**, because the tensors involved span orders
/// of magnitude across a run and an absolute threshold would mean something
/// different at step 1 than at step 19. This is what TeaCache accumulates, and
/// what makes the numbers comparable between steps.
pub fn relative_l1(a: &Array, b: &Array, s: &Stream) -> Result<f64> {
    let diff = a
        .sub(b, s)?
        .abs(s)?
        .sum(&[], true, s)?
        .to_vec_f32(s)?
        .first()
        .copied()
        .unwrap_or(0.0) as f64;
    let base = b
        .abs(s)?
        .sum(&[], true, s)?
        .to_vec_f32(s)?
        .first()
        .copied()
        .unwrap_or(0.0) as f64;
    Ok(diff / base.max(f64::EPSILON))
}
