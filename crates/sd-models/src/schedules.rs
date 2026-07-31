//! The scalar schedules two models share, and nothing else.
//!
//! **No tensors.** unCLIP's noise augmentation and the prior's DDPM step are
//! three tensor operations each and live with the backend; what is here is the
//! arithmetic that decides *what* those operations do, so a second
//! implementation cannot fit a different curve.

pub const TRAIN_TIMESTEPS: usize = 1000;

/// Largest beta the cosine schedule is allowed to produce.
///
/// The cosine `alpha_bar` goes to zero at the end of the ladder, so the ratio
/// that defines each beta goes to one; clamping keeps the last few steps from
/// destroying the signal entirely. 0.999 is diffusers' `max_beta`.
const MAX_BETA: f64 = 0.999;

/// Cumulative alphas for `squaredcos_cap_v2`, the schedule the image noiser
/// uses.
///
/// **`t` divides by `n`, not by `n - 1`.** SD's own beta schedules space their
/// interpolation across `n - 1` so the last entry lands exactly on `beta_end`;
/// this one integrates `alpha_bar` between consecutive `i/n` boundaries and has
/// no `beta_end` to land on. The two differ by one part in a thousand at every
/// step, which is enough to move the augmented embedding and not nearly enough
/// to look wrong.
/// Shared with [`crate::prior`], which samples on this exact ladder — the
/// prior's own scheduler is the same `squaredcos_cap_v2` over the same 1000
/// steps. That is the one place the "this is not a sampler's schedule" note
/// above stops applying: there it *is* one.
/// Public because the MLX augmentation needs the same ladder, and it is scalar
/// arithmetic over a cosine — the class of thing that must exist once so two
/// backends cannot drift apart on it.
pub fn cosine_alphas_cumprod(n: usize) -> Vec<f64> {
    // alpha_bar(t) = cos((t + 0.008) / 1.008 * pi / 2)^2
    let alpha_bar = |t: f64| {
        let x = (t + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2;
        x.cos() * x.cos()
    };
    let mut out = Vec::with_capacity(n);
    let mut running = 1.0;
    for i in 0..n {
        let t1 = i as f64 / n as f64;
        let t2 = (i + 1) as f64 / n as f64;
        let beta = (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(MAX_BETA);
        running *= 1.0 - beta;
        out.push(running);
    }
    out
}

/// One DDPM step's scalars: `mean = clamp(x0) * x0_coeff + sample *
/// sample_coeff`, then `+ noise * std` unless this is the last step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepCoefficients {
    pub x0_coeff: f64,
    pub sample_coeff: f64,
    /// The prior clamps its prediction to this. 5.0, not 1.0 — the embedding
    /// is whitened, so its natural range is several standard deviations.
    pub clip_range: f64,
    /// `None` at `t == 0`, where no variance is added at all. That is what
    /// lands the run on a definite answer rather than a sample from a
    /// distribution around it.
    pub std: Option<f64>,
}

/// The prior's own sampler: DDPM over a 768-vector.
///
/// **Nothing else in this project samples like this.** Every other sampler
/// here is k-diffusion's: continuous sigmas, a model that predicts noise, and
/// an ODE step. This one is the original DDPM formulation — discrete
/// timesteps, ancestral, with variance added back — and the model predicts the
/// **sample** rather than the noise, so there is no `x0` to compute.
#[derive(Debug)]
pub struct PriorScheduler {
    alphas_cumprod: Vec<f64>,
    /// The subset of the 1000 training steps a run visits, descending.
    timesteps: Vec<usize>,
    train_timesteps: usize,
    /// The prior clamps its prediction. 5.0, not 1.0 — the embedding is
    /// whitened, so its natural range is several standard deviations.
    clip_sample_range: f64,
}

impl PriorScheduler {
    /// `steps` is the number of denoising steps; diffusers' pipeline uses 25.
    pub fn new(steps: usize) -> Self {
        let train_timesteps = crate::unclip::TRAIN_TIMESTEPS;
        // Exactly `DDPMScheduler.set_timesteps`: evenly spaced *by integer
        // ratio*, descending, and the arithmetic is `(arange(n) * ratio)`
        // reversed — not a linspace, which would land on different integers.
        let steps = steps.clamp(1, train_timesteps);
        let ratio = train_timesteps / steps;
        let timesteps = (0..steps).map(|i| i * ratio).rev().collect();
        Self {
            alphas_cumprod: crate::unclip::cosine_alphas_cumprod(train_timesteps),
            timesteps,
            train_timesteps,
            clip_sample_range: 5.0,
        }
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// One DDPM step. `predicted` is the model's estimate of the *clean*
    /// sample; `noise` is a fresh standard normal draw of the same shape.
    ///
    /// The final step adds no variance, which is what lands the run on a
    /// definite answer rather than a sample from a distribution around it.
    /// The step's scalars, so a second backend can do the three tensor
    /// operations without reimplementing the schedule.
    ///
    /// Split out for the reason `sd_sample::Schedule` is scalar: this is
    /// closed-form arithmetic over `alphas_cumprod` and touches no tensor, so
    /// two backends calling it cannot drift apart. The DDPM formulation is the
    /// part that is easy to get subtly wrong, and it now exists once.
    pub fn coefficients(&self, timestep: usize) -> Result<StepCoefficients, String> {
        let t = timestep.min(self.train_timesteps - 1);
        let prev = self.previous_timestep(t)?;

        let alpha_prod_t = self.alphas_cumprod[t];
        let alpha_prod_prev = match prev {
            Some(p) => self.alphas_cumprod[p],
            // `set_alpha_to_one` is not set, so the step off the end of the
            // ladder uses 1.0 — a noiseless starting point.
            None => 1.0,
        };
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_prev = 1.0 - alpha_prod_prev;
        let current_alpha = alpha_prod_t / alpha_prod_prev;
        let current_beta = 1.0 - current_alpha;

        // **Guarded on `t > 0`, not on "is this the last step"**, which is
        // what diffusers does. For this schedule the two coincide — the ladder
        // always ends at 0 — but they are different conditions and copying the
        // literal one costs nothing.
        let std = (t != 0).then(|| {
            // `fixed_small_log`: the *log* of the variance is what is clamped,
            // and the standard deviation is its exponential of a half. Using
            // the variance directly gives noise too small by its own square
            // root — a quieter, blurrier result with nothing to catch it.
            let variance = (beta_prod_prev / beta_prod_t) * current_beta;
            (0.5 * variance.max(1e-20).ln()).exp()
        });

        Ok(StepCoefficients {
            x0_coeff: alpha_prod_prev.sqrt() * current_beta / beta_prod_t,
            sample_coeff: current_alpha.sqrt() * beta_prod_prev / beta_prod_t,
            clip_range: self.clip_sample_range,
            std,
        })
    }

    fn previous_timestep(&self, t: usize) -> Result<Option<usize>, String> {
        let i = self.timesteps.iter().position(|&x| x == t).ok_or_else(|| {
            format!(
                "timestep {t} is not on this {}-step schedule",
                self.timesteps.len()
            )
        })?;
        Ok(self.timesteps.get(i + 1).copied())
    }
}
