//! DPM++ 2M — a second-order multistep solver.
//!
//! Stateful and order-dependent: each step uses the previous step's
//! `denoised`. Calling steps out of order, or reusing a solver across images
//! without [`DpmSolverPlusPlus2M::reset`], produces output that is subtly
//! wrong in a way that reads as a bad seed rather than as a bug.

use sd_tensor::{Result, Tensor};

/// DPM++ 2M solver state.
#[derive(Debug, Default)]
pub struct DpmSolverPlusPlus2M {
    prev_denoised: Option<Tensor>,
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
        x: &Tensor,
        denoised: &Tensor,
        sigma: f64,
        sigma_next: f64,
    ) -> Result<Tensor> {
        // Final step: t_next would be -ln(0) = +inf. Branch before computing
        // rather than letting an infinity propagate — the exact result here is
        // the denoised prediction itself.
        if sigma_next == 0.0 {
            self.prev_denoised = Some(denoised.clone());
            self.prev_t = Some(-sigma.ln());
            return Ok(denoised.clone());
        }
        if sigma == 0.0 {
            return Ok(x.clone());
        }

        let t = -sigma.ln();
        let t_next = -sigma_next.ln();
        let h = t_next - t;

        // `d` is the denoised estimate the step actually uses: the current
        // prediction on the first step, and a second-order extrapolation from
        // the previous one thereafter.
        let d = match (&self.prev_denoised, self.prev_t) {
            (Some(prev), Some(prev_t)) => {
                let h_last = t - prev_t;
                let r = h_last / h;
                let inv = 1.0 / (2.0 * r);
                ((denoised * (1.0 + inv))? - (prev * inv)?)?
            }
            // First-order fallback until there is a previous step to use.
            _ => denoised.clone(),
        };

        let x_next = ((x * (sigma_next / sigma))? - (d * ((-h).exp() - 1.0))?)?;

        self.prev_denoised = Some(denoised.clone());
        self.prev_t = Some(t);
        Ok(x_next)
    }
}
