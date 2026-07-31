//! Noise schedules and samplers.
//!
//! Two independent choices, and conflating them is a common way to reproduce
//! someone else's recipe and get a different picture:
//!
//! - **Which sigmas to visit** — [`schedulers`]. Karras, exponential,
//!   sgm-uniform, or the discrete training ladder.
//! - **How to step between them** — the sampler, in [`crate::steps`] for the
//!   scalar coefficients and `sd_models::mlx::sample` for the tensor work.

pub mod flow;
pub mod lcm;
pub mod schedulers;
pub mod sigmas;
pub mod steps;

pub use flow::{flow_sigmas, flow_timesteps, FlowMatchConfig};
pub use lcm::{lcm_sigmas, lcm_timesteps, ORIGINAL_INFERENCE_STEPS};
pub use schedulers::{sigmas_for, Scheduler};
pub use sigmas::{sigmas_for_steps, sigmas_from};

/// Beta schedule shapes used by Stable Diffusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BetaSchedule {
    /// SD 1.x / 2.x default: linear in `sqrt(beta)`.
    ScaledLinear,
    Linear,
}

/// Discrete noise schedule.
#[derive(Debug, Clone)]
pub struct Schedule {
    pub betas: Vec<f64>,
    pub alphas_cumprod: Vec<f64>,
}

impl Schedule {
    /// Build a schedule over `train_timesteps` steps.
    ///
    /// SD 1.x uses `beta_start = 0.00085`, `beta_end = 0.012`,
    /// `train_timesteps = 1000`, [`BetaSchedule::ScaledLinear`].
    pub fn new(
        train_timesteps: usize,
        beta_start: f64,
        beta_end: f64,
        schedule: BetaSchedule,
    ) -> Self {
        let n = train_timesteps.max(1);
        let betas: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1).max(1) as f64;
                match schedule {
                    BetaSchedule::Linear => beta_start + t * (beta_end - beta_start),
                    BetaSchedule::ScaledLinear => {
                        let s = beta_start.sqrt() + t * (beta_end.sqrt() - beta_start.sqrt());
                        s * s
                    }
                }
            })
            .collect();

        let mut alphas_cumprod = Vec::with_capacity(n);
        let mut running = 1.0;
        for &b in &betas {
            running *= 1.0 - b;
            alphas_cumprod.push(running);
        }

        Self {
            betas,
            alphas_cumprod,
        }
    }

    /// SD 1.x / 2.x default schedule.
    pub fn sd15() -> Self {
        Self::new(1000, 0.00085, 0.012, BetaSchedule::ScaledLinear)
    }

    /// `sigma_t = sqrt((1 - a_t) / a_t)`, the k-diffusion parameterisation.
    pub fn sigmas(&self) -> Vec<f64> {
        self.alphas_cumprod
            .iter()
            .map(|&a| ((1.0 - a) / a).sqrt())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd15_schedule_has_expected_endpoints() {
        let s = Schedule::sd15();
        assert_eq!(s.betas.len(), 1000);
        assert!((s.betas[0] - 0.00085).abs() < 1e-9);
        assert!((s.betas[999] - 0.012).abs() < 1e-9);
    }

    #[test]
    fn alphas_cumprod_is_monotonically_decreasing() {
        let s = Schedule::sd15();
        for w in s.alphas_cumprod.windows(2) {
            assert!(w[1] < w[0], "alphas_cumprod must decrease");
        }
        assert!(*s.alphas_cumprod.last().unwrap() > 0.0);
    }

    #[test]
    fn sigmas_increase_with_timestep() {
        let s = Schedule::sd15();
        let sig = s.sigmas();
        assert!(sig[0] < sig[999]);
        assert!(sig[0] > 0.0);
    }
}
