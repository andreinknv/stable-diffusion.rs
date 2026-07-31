//! Sampler steps, as scalar coefficients.
//!
//! Every sampler here reduces to "combine these tensors with these weights",
//! and the weights are functions of `(sigma, sigma_next)` alone. Keeping them
//! here — backend-free, `f64`, unit-testable without a GPU — is what lets the
//! tensor side stay a handful of multiplies and adds, and what makes a
//! sampler's arithmetic checkable against the paper it comes from rather than
//! against an image.
//!
//! # The shared shape
//!
//! All of these consume `denoised` — the model's estimate of the clean image,
//! `x0` — rather than the raw epsilon. `denoise_epsilon` does that conversion
//! once, so a sampler never sees the prediction type and the same code serves
//! epsilon and v-prediction models.
//!
//! # Second-order samplers cost two evaluations
//!
//! Heun and DPM++ 2S ancestral each call the model **twice per step**. That is
//! not overhead to be optimised away — it is the method — so a 20-step Heun run
//! costs about what a 40-step Euler run does. The comparison that makes them
//! look good is at equal *evaluations*, not equal steps, and
//! [`SamplerKind::evaluations_per_step`] exists so a progress bar can say so.

/// Which sampler a run uses.
///
/// The `Copy` scalar half of `sd_models`' sampler dispatch: this decides *what*
/// arithmetic happens, the MLX side does it to tensors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SamplerKind {
    /// Euler ancestral. Injects fresh noise every step, so it never converges
    /// to a fixed image as steps rise — which is a feature at low step counts
    /// and the reason step caching refuses it.
    #[default]
    EulerAncestral,
    /// Plain Euler. Deterministic, and the baseline every method is compared
    /// against.
    Euler,
    /// Heun's method: an Euler step, then a correction using the derivative at
    /// where it landed. **Two model evaluations per step.**
    Heun,
    /// DPM-Solver++ (2M): second order using the *previous* step's derivative,
    /// so it costs one evaluation per step. The default for most recipes.
    DpmPlusPlus2M,
    /// DPM-Solver++ (2S) ancestral: a midpoint method with noise injection.
    /// **Two model evaluations per step.**
    DpmPlusPlus2SAncestral,
    /// DDIM with `eta = 0`. Deterministic, and what most papers report.
    Ddim,
    /// Latent consistency sampling. **Only meaningful with an LCM-distilled
    /// model or adapter**, and wants 4-8 steps at `cfg_scale` near 1 — the
    /// guidance is distilled in, so applying more on top double-counts it and
    /// blows the image out.
    Lcm,
}

impl SamplerKind {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "euler-a" | "euler_a" | "euler-ancestral" => Self::EulerAncestral,
            "euler" => Self::Euler,
            "heun" => Self::Heun,
            "dpmpp2m" | "dpmpp-2m" | "dpm++2m" => Self::DpmPlusPlus2M,
            "dpmpp2s-a" | "dpmpp_2s_a" | "dpm++2s-a" => Self::DpmPlusPlus2SAncestral,
            "ddim" => Self::Ddim,
            "lcm" => Self::Lcm,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::EulerAncestral => "euler-a",
            Self::Euler => "euler",
            Self::Heun => "heun",
            Self::DpmPlusPlus2M => "dpmpp2m",
            Self::DpmPlusPlus2SAncestral => "dpmpp2s-a",
            Self::Ddim => "ddim",
            Self::Lcm => "lcm",
        }
    }

    /// Every sampler this project offers.
    pub fn all() -> &'static [Self] {
        &[
            Self::EulerAncestral,
            Self::Euler,
            Self::Heun,
            Self::DpmPlusPlus2M,
            Self::DpmPlusPlus2SAncestral,
            Self::Ddim,
            Self::Lcm,
        ]
    }

    /// **Whether the sampler draws fresh noise each step.**
    ///
    /// The load-bearing consequence is step caching: reusing a prediction
    /// across steps assumes the trajectory is deterministic, and an ancestral
    /// sampler's is not. Asking for both is refused rather than silently
    /// producing a worse image.
    pub fn is_ancestral(self) -> bool {
        matches!(
            self,
            Self::EulerAncestral | Self::DpmPlusPlus2SAncestral | Self::Lcm
        )
    }

    /// How many model evaluations one step costs.
    ///
    /// Two for the second-order methods. A caller sizing a progress bar or
    /// comparing samplers wants this, because comparing Heun and Euler at equal
    /// *steps* compares one against half its budget.
    pub fn evaluations_per_step(self) -> usize {
        match self {
            Self::Heun | Self::DpmPlusPlus2SAncestral => 2,
            _ => 1,
        }
    }
}

/// The `(sigma_up, sigma_down)` split an ancestral step uses.
///
/// k-diffusion's `get_ancestral_step` with `eta = 1`:
///
/// ```text
/// sigma_up   = sqrt(sigma_next^2 * (sigma^2 - sigma_next^2) / sigma^2)
/// sigma_down = sqrt(sigma_next^2 - sigma_up^2)
/// ```
///
/// **`sigma_down` is where the deterministic part of the step lands**, below
/// `sigma_next`; `sigma_up` is how much fresh noise is added back to reach it.
/// Stepping to `sigma_next` directly *and* adding noise overshoots, which looks
/// like a sampler that never resolves detail.
pub fn ancestral_split(sigma: f64, sigma_next: f64, eta: f64) -> (f64, f64) {
    if sigma <= 0.0 {
        return (0.0, sigma_next);
    }
    let up = (eta
        * (sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2))
            .max(0.0)
            .sqrt())
    .min(sigma_next);
    let down = (sigma_next.powi(2) - up.powi(2)).max(0.0).sqrt();
    (up, down)
}

/// Weights for `x_next = a*x + b*denoised` — the plain Euler step.
///
/// Derived rather than asserted: the derivative is `d = (x - denoised) / sigma`
/// and the step is `x + d * (sigma_next - sigma)`, which rearranges to
/// `x * (sigma_next / sigma) + denoised * (1 - sigma_next / sigma)`.
pub fn euler_weights(sigma: f64, sigma_next: f64) -> (f64, f64) {
    if sigma <= 0.0 {
        return (0.0, 1.0);
    }
    let r = sigma_next / sigma;
    (r, 1.0 - r)
}

/// The second half of a Heun step.
///
/// After an Euler step to `x_euler` at `sigma_next`, the model is evaluated
/// again there and the two derivatives are averaged. Returns
/// `(a, b, c)` for `x_next = a*x + b*denoised + c*denoised_next`.
///
/// **When `sigma_next` is zero there is no second derivative to take** — the
/// trajectory has arrived — so this degenerates to the Euler step, which is
/// what k-diffusion does and what stops the last step dividing by zero.
pub fn heun_weights(sigma: f64, sigma_next: f64) -> (f64, f64, f64) {
    if sigma <= 0.0 {
        return (0.0, 1.0, 0.0);
    }
    if sigma_next <= 0.0 {
        let (a, b) = euler_weights(sigma, sigma_next);
        return (a, b, 0.0);
    }
    // d      = (x - denoised) / sigma
    // d_next = (x_euler - denoised_next) / sigma_next
    // x_next = x + (d + d_next)/2 * dt,  dt = sigma_next - sigma
    //
    // Substituting x_euler = x + d*dt and collecting terms gives the closed
    // form below. Written out because deriving it at the call site, in tensor
    // ops, is where a sign goes missing.
    let dt = sigma_next - sigma;
    let h = 0.5 * dt;
    // From d: coefficient on x is 1/sigma, on denoised is -1/sigma.
    // From d_next: x_euler/sigma_next = (x + d*dt)/sigma_next.
    let x_from_d = 1.0 / sigma;
    let x_from_dnext = (1.0 + dt / sigma) / sigma_next;
    let den_from_d = -1.0 / sigma;
    let den_from_dnext = -(dt / sigma) / sigma_next;
    let dennext_from_dnext = -1.0 / sigma_next;
    (
        1.0 + h * (x_from_d + x_from_dnext),
        h * (den_from_d + den_from_dnext),
        h * dennext_from_dnext,
    )
}

/// DDIM with `eta = 0`, in the sigma parameterisation.
///
/// With `eta = 0` DDIM *is* the Euler step on this ladder — the two are the
/// same update written in different coordinates. Kept as a distinct name
/// because every paper reports against "DDIM" and a user asking for it should
/// get it rather than be told it is a synonym.
pub fn ddim_weights(sigma: f64, sigma_next: f64) -> (f64, f64) {
    euler_weights(sigma, sigma_next)
}

/// DPM-Solver++ (2S) ancestral: the midpoint the first half-step lands on.
///
/// Returns `(sigma_mid, a, b)` where the midpoint estimate is
/// `x_mid = a*x + b*denoised`, evaluated at `sigma_mid`.
///
/// The midpoint is **geometric in sigma**, not arithmetic: `sigma_mid` is
/// `exp((log(sigma) + log(sigma_down)) / 2)`. DPM-Solver works in `lambda =
/// -log(sigma)`, so an arithmetic midpoint in sigma is not the midpoint of the
/// interval the method integrates over, and using one degrades the order back
/// to first.
pub fn dpmpp_2s_midpoint(sigma: f64, sigma_down: f64) -> (f64, f64, f64) {
    if sigma <= 0.0 || sigma_down <= 0.0 {
        let (a, b) = euler_weights(sigma, sigma_down);
        return (sigma_down, a, b);
    }
    let (lam, lam_down) = (-sigma.ln(), -sigma_down.ln());
    let lam_mid = 0.5 * (lam + lam_down);
    let sigma_mid = (-lam_mid).exp();
    // x_mid = (sigma_mid/sigma) * x - (exp(-(lam_mid-lam)) - 1) * denoised
    let h = lam_mid - lam;
    (sigma_mid, sigma_mid / sigma, -((-h).exp() - 1.0))
}

/// DPM-Solver++ (2S) ancestral: the full step, given the midpoint's `denoised`.
///
/// `x_next = a*x + b*denoised_mid`, landing at `sigma_down`; the caller then
/// adds `sigma_up * noise`.
pub fn dpmpp_2s_step(sigma: f64, sigma_down: f64) -> (f64, f64) {
    if sigma <= 0.0 || sigma_down <= 0.0 {
        return euler_weights(sigma, sigma_down);
    }
    let h = -sigma_down.ln() + sigma.ln();
    (sigma_down / sigma, -((-h).exp() - 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Euler's weights must sum to one at every sigma pair.**
    ///
    /// The step is a convex-ish blend of the current latent and the model's
    /// clean estimate; weights that do not sum to one rescale the image, which
    /// shows up as a run that gets progressively brighter or darker rather than
    /// as an error.
    #[test]
    fn euler_weights_sum_to_one() {
        for (sigma, next) in [(14.6, 6.7), (6.7, 3.5), (0.44, 0.029), (0.029, 0.0)] {
            let (a, b) = euler_weights(sigma, next);
            assert!((a + b - 1.0).abs() < 1e-12, "{sigma} -> {next}: {a} + {b}");
        }
    }

    /// The last step lands exactly on the model's estimate.
    #[test]
    fn the_final_step_returns_the_clean_estimate() {
        let (a, b) = euler_weights(0.029, 0.0);
        assert!(a.abs() < 1e-12, "no latent survives the last step");
        assert!((b - 1.0).abs() < 1e-12);
    }

    /// **Heun degenerates to Euler at the last step**, where there is no
    /// second derivative to take. Without this it divides by zero.
    #[test]
    fn heun_degenerates_to_euler_at_zero() {
        let (a, b, c) = heun_weights(0.029, 0.0);
        let (ea, eb) = euler_weights(0.029, 0.0);
        assert!((a - ea).abs() < 1e-12);
        assert!((b - eb).abs() < 1e-12);
        assert_eq!(c, 0.0, "no second evaluation contributes at sigma_next = 0");
        assert!(a.is_finite() && b.is_finite());
    }

    /// **Heun reduces to Euler when the two derivatives agree.**
    ///
    /// If `denoised_next == denoised`, averaging the derivatives is averaging a
    /// value with itself, so the corrected step must equal the Euler step. This
    /// is the check that catches a sign error in the closed form, which no
    /// weights-sum-to-one test would.
    #[test]
    fn heun_matches_euler_when_the_model_does_not_move() {
        for (sigma, next) in [(14.6, 6.7), (3.5, 2.06), (0.44, 0.029)] {
            let (a, b, c) = heun_weights(sigma, next);
            let (ea, eb) = euler_weights(sigma, next);
            // x_next = a*x + (b + c)*denoised when denoised_next == denoised.
            assert!(
                (a - ea).abs() < 1e-9,
                "{sigma} -> {next}: latent weight {a} vs {ea}"
            );
            assert!(
                (b + c - eb).abs() < 1e-9,
                "{sigma} -> {next}: denoised weight {} vs {eb}",
                b + c
            );
        }
    }

    /// The ancestral split conserves variance: `up^2 + down^2 == next^2`.
    #[test]
    fn the_ancestral_split_conserves_variance() {
        for (sigma, next) in [(14.6, 6.7), (3.5, 2.06), (0.44, 0.029)] {
            let (up, down) = ancestral_split(sigma, next, 1.0);
            assert!(
                (up * up + down * down - next * next).abs() < 1e-9,
                "{sigma} -> {next}: {up}^2 + {down}^2 != {next}^2"
            );
            assert!(down < next, "the deterministic part lands below sigma_next");
        }
    }

    /// **`eta = 0` makes an ancestral sampler deterministic**, which is what
    /// makes `eta` a real dial rather than decoration.
    #[test]
    fn eta_zero_removes_the_noise_injection() {
        let (up, down) = ancestral_split(3.5, 2.06, 0.0);
        assert_eq!(up, 0.0);
        assert!(
            (down - 2.06).abs() < 1e-12,
            "and lands on sigma_next itself"
        );
    }

    /// DPM++ 2S's midpoint is geometric in sigma, not arithmetic.
    #[test]
    fn the_dpmpp_2s_midpoint_is_geometric() {
        let (sigma, down) = (4.0, 1.0);
        let (mid, _, _) = dpmpp_2s_midpoint(sigma, down);
        assert!((mid - 2.0).abs() < 1e-9, "geometric mean of 4 and 1 is 2");
        assert!(
            (mid - 2.5).abs() > 0.1,
            "an arithmetic midpoint would be 2.5"
        );
    }

    /// DPM++ 2S's weights also sum to one, for the same reason Euler's must.
    #[test]
    fn dpmpp_2s_weights_sum_to_one() {
        for (sigma, down) in [(14.6, 6.0), (3.5, 1.9), (0.44, 0.02)] {
            let (_, a, b) = dpmpp_2s_midpoint(sigma, down);
            assert!((a + b - 1.0).abs() < 1e-9, "midpoint: {a} + {b}");
            let (c, d) = dpmpp_2s_step(sigma, down);
            assert!((c + d - 1.0).abs() < 1e-9, "step: {c} + {d}");
        }
    }

    /// **The second-order samplers declare their true cost.**
    #[test]
    fn second_order_samplers_cost_two_evaluations() {
        assert_eq!(SamplerKind::Heun.evaluations_per_step(), 2);
        assert_eq!(
            SamplerKind::DpmPlusPlus2SAncestral.evaluations_per_step(),
            2
        );
        assert_eq!(SamplerKind::Euler.evaluations_per_step(), 1);
        assert_eq!(SamplerKind::DpmPlusPlus2M.evaluations_per_step(), 1);
    }

    /// **Which samplers draw noise** decides which can be step-cached.
    #[test]
    fn the_ancestral_samplers_are_the_ones_that_draw_noise() {
        assert!(SamplerKind::EulerAncestral.is_ancestral());
        assert!(SamplerKind::DpmPlusPlus2SAncestral.is_ancestral());
        assert!(SamplerKind::Lcm.is_ancestral());
        assert!(!SamplerKind::Euler.is_ancestral());
        assert!(!SamplerKind::Heun.is_ancestral());
        assert!(!SamplerKind::DpmPlusPlus2M.is_ancestral());
        assert!(!SamplerKind::Ddim.is_ancestral());
    }

    #[test]
    fn every_sampler_name_round_trips() {
        for &s in SamplerKind::all() {
            assert_eq!(SamplerKind::parse(s.name()), Some(s), "{s:?}");
        }
        assert_eq!(SamplerKind::parse("nope"), None);
    }
}
