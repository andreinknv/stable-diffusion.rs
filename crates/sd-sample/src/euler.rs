//! Euler ancestral sampling.

use sd_tensor::{Result, Tensor};

/// One Euler-ancestral step.
///
/// `noise` must be standard normal with the same shape as `x`. It is ignored
/// when `sigma_next == 0.0` — the last step lands on the clean image and must
/// not have noise added back.
///
/// Scalar arithmetic stays in `f64`; only the tensor ops are `f32`.
pub fn euler_ancestral_step(
    x: &Tensor,
    denoised: &Tensor,
    sigma: f64,
    sigma_next: f64,
    noise: &Tensor,
) -> Result<Tensor> {
    // sigma == 0 means there is nothing left to denoise, and dividing by it
    // would produce NaN rather than an error.
    if sigma == 0.0 {
        return Ok(x.clone());
    }

    // The `min` matters: without it `sigma_up` is fine for most steps and
    // wrong near the end, where sigma_next approaches zero.
    let sigma_up = sigma_next
        .min((sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2)).sqrt());
    let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).max(0.0).sqrt();

    // d is the derivative toward the denoised prediction.
    let d = ((x - denoised)? / sigma)?;
    let x = (x + (d * (sigma_down - sigma))?)?;

    if sigma_next > 0.0 {
        x + (noise * sigma_up)?
    } else {
        Ok(x)
    }
}
