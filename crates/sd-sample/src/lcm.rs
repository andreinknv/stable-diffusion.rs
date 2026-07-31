//! Latent Consistency Model sampling.
//!
//! LCM is not another ODE solver. Euler and DPM++ integrate a trajectory from
//! noise toward the image, so each step must be small enough for the
//! integration to hold. A consistency model is trained so that a *single*
//! evaluation maps any point on that trajectory straight to its origin, which
//! is why four steps suffice where the solvers need twenty.
//!
//! The loop is correspondingly different. A solver moves along the trajectory;
//! LCM jumps to the origin and then re-noises back out to the next level:
//!
//! ```text
//!   x0     = f(x, t)                    the consistency function
//!   x_next = x0 + sigma_next * noise    fresh noise, every step
//! ```
//!
//! with the last step keeping `x0` and adding nothing.
//!
//! # Two things that make it look broken if you miss them
//!
//! **Guidance must be about 1.** The distillation folded a guidance scale in,
//! so applying another on top double-counts it. At the 7.5 that suits SD 1.5
//! the image blows out to saturated blocks — which looks like a broken
//! sampler and is not one.
//!
//! **The timesteps are a fixed subset, not an even spread.** `lcm_timesteps`
//! reproduces the schedule the distillation used; running a consistency model
//! on an evenly spaced ladder asks it for a mapping it was never trained to
//! make.

use crate::Schedule;

/// Standard deviation the consistency parameterisation assumes for data.
///
/// 0.5 in every published LCM. It appears only in the boundary conditions.
const SIGMA_DATA: f64 = 0.5;

/// Multiplier applied to the timestep inside the boundary conditions.
///
/// 10.0, matching diffusers' `LCMScheduler`. It has no meaning on its own —
/// it sets where the blend between "trust the model" and "trust the input"
/// sits along the schedule, and the trained weights expect this exact value.
const TIMESTEP_SCALING: f64 = 10.0;

/// Steps the distillation was performed over.
///
/// 50 for every published LCM adapter, and the reason the inference timesteps
/// are the subset they are.
pub const ORIGINAL_INFERENCE_STEPS: usize = 50;

/// The discrete timesteps an LCM run visits, noisiest first.
///
/// Not an even spread of `train_timesteps`. The distillation walked a ladder
/// of `original_inference_steps` rungs — `k = train/original`, giving
/// `[k-1, 2k-1, ...]` — and inference takes an evenly spaced subset *of that
/// ladder*, counting down. For SD 1.5 at four steps that is
/// `[999, 759, 519, 279]`, which is the sequence to check against diffusers if
/// this is ever suspected.
pub fn lcm_timesteps(
    train_timesteps: usize,
    original_inference_steps: usize,
    steps: usize,
) -> Vec<usize> {
    if steps == 0 || original_inference_steps == 0 || train_timesteps == 0 {
        return Vec::new();
    }
    let k = (train_timesteps / original_inference_steps).max(1);
    // The distillation ladder: k-1, 2k-1, ... ascending.
    let ladder: Vec<usize> = (1..=original_inference_steps).map(|i| i * k - 1).collect();
    // Walk it backwards, skipping evenly, and take as many as asked for. The
    // skip is floor division: with 50 rungs and 4 steps it is 12, so the run
    // visits every twelfth rung from the top rather than the top four.
    let skip = (ladder.len() / steps).max(1);
    ladder
        .iter()
        .rev()
        .step_by(skip)
        .take(steps)
        .copied()
        .collect()
}

/// Sigma ladder for an LCM run: one per timestep, plus a trailing zero.
///
/// The pipeline is parameterised by sigma throughout, so the timesteps are
/// converted here rather than threaded separately. `sigma = sqrt((1-a)/a)` for
/// `a = alphas_cumprod[t]`, which is the same relation
/// [`Schedule::sigmas`] uses.
pub fn lcm_sigmas(schedule: &Schedule, timesteps: &[usize]) -> Vec<f64> {
    let train = schedule.sigmas();
    let mut out: Vec<f64> = timesteps
        .iter()
        .map(|&t| {
            train
                .get(t.min(train.len().saturating_sub(1)))
                .copied()
                .unwrap_or(0.0)
        })
        .collect();
    out.push(0.0);
    out
}

/// The consistency function's boundary conditions at `timestep`.
///
/// `c_skip` weights the input and `c_out` the model's prediction. They are
/// built so that at `t = 0` the function is the identity — `c_skip = 1`,
/// `c_out = 0` — which is what makes it a *consistency* function rather than
/// merely a one-step denoiser. Everywhere above the very bottom of the
/// schedule `c_out` is essentially 1, so the step is dominated by the
/// prediction; the blend matters only as the trajectory lands.
pub fn boundary_conditions(timestep: f64) -> (f64, f64) {
    let scaled = timestep * TIMESTEP_SCALING;
    let denom = scaled * scaled + SIGMA_DATA * SIGMA_DATA;
    (SIGMA_DATA * SIGMA_DATA / denom, scaled / denom.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BetaSchedule;

    #[test]
    fn four_steps_visit_the_timesteps_diffusers_visits() {
        // The literal sequence LCMScheduler produces for SD 1.5 at four steps.
        // Worth pinning as a constant rather than a property: an off-by-one in
        // the ladder shifts every timestep by 20 and still produces a
        // plausible image, slightly wrong.
        assert_eq!(lcm_timesteps(1000, 50, 4), vec![999, 759, 519, 279]);
        assert_eq!(
            lcm_timesteps(1000, 50, 8),
            vec![999, 879, 759, 639, 519, 399, 279, 159]
        );
        // One step takes the top of the ladder.
        assert_eq!(lcm_timesteps(1000, 50, 1), vec![999]);
    }

    #[test]
    fn the_ladder_descends_and_never_repeats() {
        for steps in 1..=10 {
            let ts = lcm_timesteps(1000, 50, steps);
            assert_eq!(ts.len(), steps, "asked for {steps}");
            for w in ts.windows(2) {
                assert!(w[0] > w[1], "timesteps must descend: {ts:?}");
            }
            assert!(*ts.first().unwrap() < 1000, "within the training range");
        }
    }

    #[test]
    fn the_boundary_conditions_make_the_function_an_identity_at_zero() {
        // The defining property of a consistency function. If this drifts, the
        // final step stops landing on the image and starts blending noise
        // back in.
        let (c_skip, c_out) = boundary_conditions(0.0);
        assert_eq!(c_skip, 1.0);
        assert_eq!(c_out, 0.0);

        // And at the top of the schedule it is the model's prediction alone.
        let (c_skip, c_out) = boundary_conditions(999.0);
        assert!(c_skip < 1e-6, "c_skip should vanish high up, got {c_skip}");
        assert!(
            (c_out - 1.0).abs() < 1e-6,
            "c_out should be ~1, got {c_out}"
        );
    }

    #[test]
    fn sigmas_line_up_with_the_timesteps_and_end_at_zero() {
        let s = Schedule::new(1000, 0.00085, 0.012, BetaSchedule::ScaledLinear);
        let ts = lcm_timesteps(1000, 50, 4);
        let sigmas = lcm_sigmas(&s, &ts);
        assert_eq!(
            sigmas.len(),
            ts.len() + 1,
            "one boundary per step, plus the landing"
        );
        assert_eq!(*sigmas.last().unwrap(), 0.0);
        for w in sigmas.windows(2) {
            assert!(w[0] > w[1], "sigmas must descend: {sigmas:?}");
        }
    }
}
