//! Generation settings, and the error a pipeline returns.
//!
//! **Backend-free by construction**: nothing here holds a tensor. That is what
//! lets one set of settings describe a run whatever executes it, and it is why
//! these live apart from the pipeline that consumes them rather than inside it.

use std::path::PathBuf;

/// Which sampler a run uses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SamplerKind {
    #[default]
    EulerAncestral,
    DpmPlusPlus2M,
    /// Latent consistency sampling. **Only meaningful with an LCM-distilled
    /// model or adapter**, and wants 4-8 steps at `cfg_scale` near 1 — the
    /// guidance is distilled in, so applying more on top double-counts it and
    /// blows the image out.
    Lcm,
}

/// How much of the schedule an img2img run replaces.
///
/// `1.0` ignores the input entirely; `0.0` returns it unchanged. The value
/// selects where in the sigma ladder to start: at strength `s` with `n` steps,
/// the run begins at index `n - round(n*s)` and executes the remaining steps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Strength(f64);

impl Strength {
    /// Clamped to `[0, 1]`; anything outside is a caller error, not a mode.
    pub fn new(v: f64) -> Self {
        Self(v.clamp(0.0, 1.0))
    }

    pub fn get(self) -> f64 {
        self.0
    }

    /// Index into a `steps + 1` sigma ladder at which to begin.
    ///
    /// Public because it is the whole meaning of the parameter: `steps -
    /// start_index(steps)` is how much work a run will actually do, and a
    /// caller sizing a progress bar needs it.
    pub fn start_index(self, steps: usize) -> usize {
        let run = (steps as f64 * self.0).round() as usize;
        steps.saturating_sub(run.min(steps))
    }
}

impl Default for Strength {
    fn default() -> Self {
        Self(0.75)
    }
}

/// Everything a single generation needs.
#[derive(Debug, Clone)]
pub struct Txt2ImgConfig {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub cfg_scale: f64,
    pub seed: u64,
    pub sampler: SamplerKind,
}

impl Default for Txt2ImgConfig {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 512,
            height: 512,
            steps: 20,
            cfg_scale: 7.5,
            seed: 0,
            sampler: SamplerKind::default(),
        }
    }
}

/// The polynomial that turns "how far the timestep embedding moved" into "how
/// far the model's output is estimated to move".
///
/// **Per model.** These coefficients describe SD 1.5's schedule and its
/// embedding widths; SDXL or SD 2.x need their own, which is one command
/// (`--example cache_fit`). Using these on another architecture is not
/// catastrophic — the accumulator is monotone either way — but the threshold
/// stops meaning what it says.
const CACHE_RESCALE: [f64; 5] = [
    5.036842e-2,
    1.022504e-1,
    -4.397247e-1,
    5.716702e-1,
    -1.481600e-1,
];

/// Evaluate [`CACHE_RESCALE`], **clamped at zero**.
///
/// A least-squares polynomial is free to go negative where the data does not
/// constrain it, and a negative contribution would let the accumulator *fall*
/// — reusing a prediction for longer the further the model moved. Clamping is
/// what makes the accumulator monotone, which is what makes the threshold a
/// bound rather than a suggestion.
pub fn cache_rescale(moved: f64) -> f64 {
    CACHE_RESCALE
        .iter()
        .enumerate()
        .map(|(p, c)| c * moved.powi(p as i32))
        .sum::<f64>()
        .max(0.0)
}

/// What a pipeline returns when it cannot proceed.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error(
        "missing model file: {0}\n\
         Expected the standard diffusers layout under the model directory."
    )]
    MissingFile(PathBuf),
    #[error("steps must be at least 1")]
    NoSteps,
    #[error("tokenizer: {0}")]
    Tokenize(#[from] sd_models::clip::TokenizeError),
    #[error("{0}")]
    Tensor(#[from] sd_tensor::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Strength is where the run *starts*, not how many steps it runs.**
    #[test]
    fn strength_selects_where_in_the_ladder_to_begin() {
        assert_eq!(Strength::new(1.0).start_index(20), 0, "1.0 runs everything");
        assert_eq!(Strength::new(0.0).start_index(20), 20, "0.0 runs nothing");
        assert_eq!(Strength::new(0.5).start_index(20), 10);
        // Out of range is clamped rather than being a separate mode.
        assert_eq!(Strength::new(5.0).start_index(20), 0);
        assert_eq!(Strength::new(-1.0).start_index(20), 20);
    }

    /// **The cache accumulator must be monotone**, which is what makes the
    /// threshold a bound. A raw polynomial fit goes negative outside its data.
    #[test]
    fn the_cache_rescale_never_contributes_a_negative() {
        for i in 0..2000 {
            let moved = i as f64 / 100.0;
            assert!(
                cache_rescale(moved) >= 0.0,
                "cache_rescale({moved}) went negative"
            );
        }
        // And it is not identically zero, or caching would never trigger.
        assert!(cache_rescale(0.1) > 0.0);
    }
}
