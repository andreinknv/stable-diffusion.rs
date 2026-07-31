//! Choosing *which* sigmas to visit, given how many steps you can afford.
//!
//! A model is trained on 1000 noise levels. Sampling visits twenty of them, and
//! which twenty is a free choice that changes the picture as much as the
//! sampler does. [`sigmas_for_steps`](crate::sigmas_for_steps) makes the
//! obvious one — walk the training ladder at even index spacing — and it is
//! *not* what most published step counts assume.
//!
//! **Karras is the one that matters.** A "20 steps, DPM++ 2M" recipe from a
//! model card, a community preset, or another implementation almost always
//! means Karras spacing, so running the same sampler at the same step count on
//! the discrete ladder is not a slightly different image: it is a comparison
//! between two different schedules that neither side declared.
//!
//! Every schedule here returns `n + 1` values, **descending**, ending at
//! exactly `0.0` — the same contract `sigmas_for_steps` has, because the
//! samplers consume `(sigma, sigma_next)` pairs and a trailing zero is what
//! makes the last step land on a clean image rather than near one.

/// How the sigmas between `sigma_max` and `sigma_min` are spaced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Scheduler {
    /// Even spacing over the *training ladder's indices*, interpolating
    /// between neighbouring training sigmas. What this project has always
    /// done, and what `diffusers`' `EulerDiscreteScheduler` does by default.
    #[default]
    Discrete,
    /// Karras et al. 2022, eq. 5: even spacing in `sigma^(1/rho)`.
    Karras,
    /// Even spacing in `log(sigma)` — a geometric ladder.
    Exponential,
    /// Even spacing over the *timestep* range, converted back to sigmas.
    /// What `diffusers` calls `sgm_uniform`, and what SDXL's refiner wants.
    SgmUniform,
}

impl Scheduler {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "discrete" | "default" => Self::Discrete,
            "karras" => Self::Karras,
            "exponential" | "exp" => Self::Exponential,
            "sgm-uniform" | "sgm_uniform" => Self::SgmUniform,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Discrete => "discrete",
            Self::Karras => "karras",
            Self::Exponential => "exponential",
            Self::SgmUniform => "sgm-uniform",
        }
    }
}

/// Karras' `rho`. 7.0 in the paper and in every implementation that matters;
/// it controls how much of the budget goes to the low-noise end.
pub const KARRAS_RHO: f64 = 7.0;

/// Karras spacing between `sigma_min` and `sigma_max`.
///
/// Transcribed from `k_diffusion.sampling.get_sigmas_karras`:
///
/// ```text
/// ramp   = linspace(0, 1, n)
/// sigmas = (max^(1/rho) + ramp * (min^(1/rho) - max^(1/rho)))^rho
/// ```
///
/// **The ramp runs `0 -> 1` while the sigmas run high -> low**, which is why
/// `sigma_max` is the term that stands alone. Writing it the other way round
/// produces an ascending ladder that samples happily and returns noise.
///
/// Note that k-diffusion appends a trailing `0.0` *outside* this formula, so
/// the result is `n + 1` long and the last entry is not a Karras sigma at all.
pub fn karras_sigmas(n: usize, sigma_min: f64, sigma_max: f64, rho: f64) -> Vec<f64> {
    if n == 0 {
        return vec![0.0];
    }
    let (inv_min, inv_max) = (sigma_min.powf(1.0 / rho), sigma_max.powf(1.0 / rho));
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        // `n - 1` in the denominator, so the ramp reaches exactly 1 and the
        // last sigma is exactly `sigma_min`. With `n` it stops short and the
        // final step is larger than every other, which shows up as a residual
        // graininess that looks like too few steps.
        let ramp = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.0
        };
        out.push((inv_max + ramp * (inv_min - inv_max)).powf(rho));
    }
    out.push(0.0);
    out
}

/// Even spacing in `log(sigma)`.
///
/// `k_diffusion.sampling.get_sigmas_exponential`. Spends more of the budget at
/// high noise than Karras does, which suits samplers that take large early
/// steps well.
pub fn exponential_sigmas(n: usize, sigma_min: f64, sigma_max: f64) -> Vec<f64> {
    if n == 0 {
        return vec![0.0];
    }
    let (lo, hi) = (sigma_min.ln(), sigma_max.ln());
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let ramp = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.0
        };
        out.push((hi + ramp * (lo - hi)).exp());
    }
    out.push(0.0);
    out
}

/// Even spacing over the training *timesteps*, read back as sigmas.
///
/// **Not the same as [`Scheduler::Discrete`]**, and the difference is the last
/// step: `sgm_uniform` divides by `n` rather than `n - 1`, so its final sigma
/// is one interval above zero rather than the lowest training sigma. That is
/// deliberate — it pairs with the trailing zero to make a full-length final
/// step — and it is what SDXL's refiner and most SD 3 recipes assume.
pub fn sgm_uniform_sigmas(n: usize, train: &[f64]) -> Vec<f64> {
    if n == 0 || train.is_empty() {
        return vec![0.0];
    }
    let last = train.len() - 1;
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        // Descending: step 0 is the noisiest.
        let t = last as f64 * (1.0 - i as f64 / n as f64);
        let lo = t.floor() as usize;
        let hi = (lo + 1).min(last);
        let frac = t - lo as f64;
        out.push(train[lo] * (1.0 - frac) + train[hi] * frac);
    }
    out.push(0.0);
    out
}

/// The sigma ladder for `n` steps under `scheduler`, from a training schedule.
///
/// `sigma_min` and `sigma_max` for the continuous schedules come from the
/// training ladder's own ends rather than from constants, so a model with a
/// different beta schedule — SD 2.x, or anything v-prediction — gets *its*
/// range instead of SD 1.5's.
pub fn sigmas_for(scheduler: Scheduler, train: &[f64], n: usize) -> Vec<f64> {
    if train.is_empty() {
        return vec![0.0];
    }
    let (sigma_min, sigma_max) = (train[0], train[train.len() - 1]);
    match scheduler {
        Scheduler::Discrete => crate::sigmas::sigmas_from(train, n),
        Scheduler::Karras => karras_sigmas(n, sigma_min, sigma_max, KARRAS_RHO),
        Scheduler::Exponential => exponential_sigmas(n, sigma_min, sigma_max),
        Scheduler::SgmUniform => sgm_uniform_sigmas(n, train),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Schedule;

    fn train() -> Vec<f64> {
        Schedule::sd15().sigmas()
    }

    /// The shared contract: `n + 1` values, descending, ending at exactly zero.
    #[test]
    fn every_schedule_returns_a_descending_ladder_ending_at_zero() {
        for scheduler in [
            Scheduler::Discrete,
            Scheduler::Karras,
            Scheduler::Exponential,
            Scheduler::SgmUniform,
        ] {
            let s = sigmas_for(scheduler, &train(), 20);
            assert_eq!(s.len(), 21, "{scheduler:?}: 20 steps needs 21 boundaries");
            assert_eq!(s[20], 0.0, "{scheduler:?}: must end at exactly zero");
            for w in s[..20].windows(2) {
                assert!(
                    w[1] < w[0],
                    "{scheduler:?}: ascending at {w:?}; sampling would return noise"
                );
            }
        }
    }

    /// **Karras against the formula it is transcribed from.**
    ///
    /// Computed here rather than checked in, because the value of this test is
    /// that it pins the *expression* — the ramp direction and the `n - 1`
    /// denominator are the two things a plausible rewrite gets wrong.
    #[test]
    fn karras_matches_k_diffusions_expression() {
        let (n, lo, hi, rho) = (10, 0.0292, 14.6146, 7.0);
        let got = karras_sigmas(n, lo, hi, rho);
        for (i, &g) in got[..n].iter().enumerate() {
            let ramp = i as f64 / (n - 1) as f64;
            let want =
                (hi.powf(1.0 / rho) + ramp * (lo.powf(1.0 / rho) - hi.powf(1.0 / rho))).powf(rho);
            assert!((g - want).abs() < 1e-12, "index {i}: {g} vs {want}");
        }
        // The ends are exact, which is the property the `n - 1` denominator buys.
        assert!((got[0] - hi).abs() < 1e-9, "first sigma is sigma_max");
        assert!((got[n - 1] - lo).abs() < 1e-9, "last sigma is sigma_min");
    }

    /// **Karras is not the discrete ladder**, which is the entire reason it is
    /// here. If these ever agreed, one of them would be being ignored.
    #[test]
    fn karras_and_discrete_are_different_schedules() {
        let (k, d) = (
            sigmas_for(Scheduler::Karras, &train(), 20),
            sigmas_for(Scheduler::Discrete, &train(), 20),
        );
        assert_ne!(k, d);
        // And the difference is where it should be: Karras spends more of the
        // budget at low noise, so its middle sigmas sit below the discrete
        // ladder's.
        assert!(
            k[10] < d[10],
            "Karras should be below the discrete ladder in the middle: {} vs {}",
            k[10],
            d[10]
        );
    }

    /// Exponential spacing is geometric: the ratio between neighbours is
    /// constant. Checking the ratios rather than the values is what makes this
    /// a test of the *shape*.
    #[test]
    fn exponential_spacing_is_geometric() {
        let s = exponential_sigmas(12, 0.03, 14.6);
        let first = s[1] / s[0];
        for w in s[..12].windows(2) {
            assert!(
                ((w[1] / w[0]) - first).abs() < 1e-9,
                "ratio drifted: {} vs {first}",
                w[1] / w[0]
            );
        }
    }

    /// **`sgm_uniform` does not end at the lowest training sigma**, which is
    /// the one thing that distinguishes it from `Discrete` and the reason it
    /// exists. Pinned so a "simplification" to `n - 1` is caught.
    #[test]
    fn sgm_uniform_stops_one_interval_short() {
        let t = train();
        let (sgm, disc) = (
            sgm_uniform_sigmas(20, &t),
            sigmas_for(Scheduler::Discrete, &t, 20),
        );
        assert!(
            sgm[19] > disc[19],
            "sgm_uniform's last sigma should sit above the discrete ladder's: {} vs {}",
            sgm[19],
            disc[19]
        );
        assert!((disc[19] - t[0]).abs() < 1e-9, "discrete ends at sigma_min");
    }

    /// Degenerate inputs are answered rather than panicking.
    #[test]
    fn zero_steps_and_one_step_are_answered() {
        for scheduler in [
            Scheduler::Discrete,
            Scheduler::Karras,
            Scheduler::Exponential,
            Scheduler::SgmUniform,
        ] {
            assert_eq!(sigmas_for(scheduler, &train(), 0), vec![0.0]);
            let one = sigmas_for(scheduler, &train(), 1);
            assert_eq!(one.len(), 2, "{scheduler:?}");
            assert_eq!(one[1], 0.0);
            assert!(one[0] > 0.0, "{scheduler:?}: one step starts at some noise");
        }
    }

    #[test]
    fn names_round_trip() {
        for s in [
            Scheduler::Discrete,
            Scheduler::Karras,
            Scheduler::Exponential,
            Scheduler::SgmUniform,
        ] {
            assert_eq!(Scheduler::parse(s.name()), Some(s));
        }
        assert_eq!(Scheduler::parse("nope"), None);
    }
}
