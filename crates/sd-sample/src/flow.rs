//! Rectified flow matching, as Flux and SD 3 use it.
//!
//! A different formulation from the DDPM-derived samplers in this crate, and
//! simpler. There is no beta schedule and no `alphas_cumprod`: training
//! interpolates linearly between noise and data, `x_t = (1-t) * x_0 + t * eps`,
//! so the model predicts the constant velocity `eps - x_0` and a step is
//! nothing more than
//!
//! ```text
//! x_next = x + (sigma_next - sigma) * velocity
//! ```
//!
//! All the subtlety is in choosing the sigmas, which is where [`shift`] comes
//! in: uniform sigmas spend too many steps where the image is nearly clean, so
//! the schedule is warped toward the noisy end. Flux warps it *by resolution* —
//! a 1024x1024 image gets a different schedule from a 512x512 one, because it
//! has 4x the tokens to resolve.
//!
//! [`shift`]: FlowMatchConfig::shift

/// Parameters of `FlowMatchEulerDiscreteScheduler`.
#[derive(Debug, Clone, Copy)]
pub struct FlowMatchConfig {
    pub num_train_timesteps: usize,
    /// Static warp toward the noisy end. Ignored when
    /// [`Self::use_dynamic_shifting`] is set.
    pub shift: f64,
    /// Choose the warp from the image's token count instead of using
    /// [`Self::shift`]. Flux does; SD 3 does not.
    pub use_dynamic_shifting: bool,
    pub base_shift: f64,
    pub max_shift: f64,
    pub base_image_seq_len: usize,
    pub max_image_seq_len: usize,
}

impl FlowMatchConfig {
    /// Flux, from `scheduler/scheduler_config.json`.
    pub fn flux() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 3.0,
            use_dynamic_shifting: true,
            base_shift: 0.5,
            max_shift: 1.15,
            base_image_seq_len: 256,
            max_image_seq_len: 4096,
        }
    }

    /// SD 3: a fixed shift of 3.0, no resolution dependence.
    pub fn sd3() -> Self {
        Self {
            use_dynamic_shifting: false,
            ..Self::flux()
        }
    }

    /// Interpolate the warp exponent from the token count.
    ///
    /// Linear in sequence length between `(base_image_seq_len, base_shift)`
    /// and `(max_image_seq_len, max_shift)`, and deliberately **not** clamped
    /// to that range — diffusers extrapolates, and images outside 256..4096
    /// tokens are ordinary rather than exceptional.
    pub fn mu(&self, image_seq_len: usize) -> f64 {
        let (x0, x1) = (
            self.base_image_seq_len as f64,
            self.max_image_seq_len as f64,
        );
        let m = (self.max_shift - self.base_shift) / (x1 - x0);
        let b = self.base_shift - m * x0;
        image_seq_len as f64 * m + b
    }
}

/// Warp a sigma toward the noisy end.
///
/// `exp(mu) / (exp(mu) + (1/t - 1))`. Undefined at `t = 0` and `t = 1`, which
/// is why the schedule below is built over the open interval and the
/// terminating zero is appended afterwards rather than warped.
fn time_shift(mu: f64, t: f64) -> f64 {
    let e = mu.exp();
    e / (e + (1.0 / t - 1.0))
}

/// The sigma schedule for `steps` inference steps, descending, with a
/// terminating `0.0`. Length is `steps + 1`.
///
/// `image_seq_len` is the number of tokens the transformer will see — for Flux
/// that is `(h/16) * (w/16)`, since the VAE downsamples by 8 and the patchifier
/// takes 2x2 blocks. Ignored unless the config asks for dynamic shifting.
pub fn flow_sigmas(cfg: &FlowMatchConfig, steps: usize, image_seq_len: usize) -> Vec<f64> {
    let n = steps.max(1);
    let t = cfg.num_train_timesteps as f64;

    // diffusers walks timesteps from `t` down to 1 and divides by `t`, so the
    // schedule spans 1.0 down to 1/t rather than reaching 0. The final 0 is
    // appended, not interpolated to.
    let sigma_max = 1.0;
    let sigma_min = 1.0 / t;

    let mut sigmas: Vec<f64> = (0..n)
        .map(|i| {
            let f = if n == 1 {
                0.0
            } else {
                i as f64 / (n - 1) as f64
            };
            sigma_max + f * (sigma_min - sigma_max)
        })
        .collect();

    if cfg.use_dynamic_shifting {
        let mu = cfg.mu(image_seq_len);
        for s in sigmas.iter_mut() {
            *s = time_shift(mu, *s);
        }
    } else {
        for s in sigmas.iter_mut() {
            *s = cfg.shift * *s / (1.0 + (cfg.shift - 1.0) * *s);
        }
    }

    sigmas.push(0.0);
    sigmas
}

/// The timestep values the transformer is conditioned on.
///
/// Flux takes `sigma` scaled to the training range rather than an index, so
/// this is just `sigma * num_train_timesteps` over the non-terminal sigmas.
pub fn flow_timesteps(cfg: &FlowMatchConfig, sigmas: &[f64]) -> Vec<f64> {
    sigmas
        .iter()
        .take(sigmas.len().saturating_sub(1))
        .map(|s| s * cfg.num_train_timesteps as f64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmas_descend_from_one_to_zero() {
        let cfg = FlowMatchConfig::flux();
        let s = flow_sigmas(&cfg, 20, 4096);
        assert_eq!(s.len(), 21, "steps + 1, with a terminating zero");
        assert_eq!(*s.last().unwrap(), 0.0);
        assert!(s[0] > 0.99, "starts at pure noise, got {}", s[0]);
        for w in s.windows(2) {
            assert!(w[0] > w[1], "must descend: {} then {}", w[0], w[1]);
        }
    }

    #[test]
    fn dynamic_shift_depends_on_resolution() {
        let cfg = FlowMatchConfig::flux();
        // 512x512 is 1024 tokens, 1024x1024 is 4096.
        let small = flow_sigmas(&cfg, 20, 1024);
        let large = flow_sigmas(&cfg, 20, 4096);
        assert_ne!(
            small[10], large[10],
            "the whole point of dynamic shifting is resolution dependence"
        );
        // More tokens means a stronger warp, so the schedule sits higher —
        // more of the budget spent while the image is still noisy.
        assert!(
            large[10] > small[10],
            "expected the larger image to hold higher sigmas: {} vs {}",
            large[10],
            small[10]
        );
    }

    #[test]
    fn static_shift_ignores_resolution() {
        let cfg = FlowMatchConfig::sd3();
        assert_eq!(flow_sigmas(&cfg, 10, 256), flow_sigmas(&cfg, 10, 4096));
    }

    #[test]
    fn mu_interpolates_between_the_configured_endpoints() {
        let cfg = FlowMatchConfig::flux();
        assert!((cfg.mu(256) - cfg.base_shift).abs() < 1e-12);
        assert!((cfg.mu(4096) - cfg.max_shift).abs() < 1e-12);
        // Midpoint, to catch a slope/intercept swap that endpoints alone miss.
        assert!((cfg.mu(2176) - (cfg.base_shift + cfg.max_shift) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn timesteps_scale_sigmas_and_drop_the_terminal_zero() {
        let cfg = FlowMatchConfig::flux();
        let s = flow_sigmas(&cfg, 4, 4096);
        let t = flow_timesteps(&cfg, &s);
        assert_eq!(t.len(), 4, "one timestep per step, not per sigma");
        for (ti, si) in t.iter().zip(&s) {
            assert!((ti - si * 1000.0).abs() < 1e-9);
        }
    }
}
