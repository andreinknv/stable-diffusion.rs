//! Generation settings, and the error a pipeline returns.
//!
//! **Backend-free by construction**: nothing here holds a tensor. That is what
//! lets one set of settings describe a run whatever executes it, and it is why
//! these live apart from the pipeline that consumes them rather than inside it.

use std::path::PathBuf;

/// Which sampler a run uses, and how the sigmas between steps are spaced.
///
/// **Re-exported rather than redefined.** These used to be declared here as
/// well as in `sd_sample`, which is how the two would come to disagree about
/// what `dpmpp2m` means — and a sampler mismatch produces a worse image with
/// no error, which is this project's most-repeated failure shape.
pub use sd_sample::schedulers::Scheduler;
pub use sd_sample::steps::SamplerKind;

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

/// How many CLIP layers to discard from the end of the text encoder.
///
/// **1 means "use the last hidden state", which is what SD 1.5 was trained
/// with.** 2 means the penultimate layer, and a large fraction of community
/// checkpoints — most anime and illustration finetunes — were trained that way
/// and expect it. Running one of those at 1 is not an error and does not look
/// broken; it produces a flatter, less on-model picture, which is exactly the
/// kind of silent wrongness worth making explicit.
///
/// SDXL and SD 3 read the penultimate layer by architecture rather than by
/// this setting, so it does not apply there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipSkip(usize);

impl ClipSkip {
    /// Clamped to at least 1: skipping every layer is not a mode.
    pub fn new(n: usize) -> Self {
        Self(n.max(1))
    }

    pub fn get(self) -> usize {
        self.0
    }

    /// How many layers of the encoder to actually run, out of `total`.
    ///
    /// Saturating rather than wrapping, and at least one layer always runs.
    pub fn layers_of(self, total: usize) -> usize {
        total.saturating_sub(self.0.saturating_sub(1)).max(1)
    }
}

impl Default for ClipSkip {
    fn default() -> Self {
        Self(1)
    }
}

/// The precision the diffusion model runs at.
///
/// **Not a quality dial with an obvious better end.** Measured on SD 1.5 at
/// 768x768, 20 steps: f16 is 1.10x faster and 1.15 GB smaller, and its image
/// differs from the f32 one at PSNR 36.8 dB — a different sample of comparable
/// quality rather than a degraded one. f32 is the default because every golden
/// test in this project is verified at f32 tolerances.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Precision {
    #[default]
    F32,
    F16,
}

impl Precision {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "f32" | "fp32" => Self::F32,
            "f16" | "fp16" | "half" => Self::F16,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

/// Everything a single generation needs.
///
/// **Backend-free**: nothing here holds a tensor, so one value describes a run
/// whatever executes it. Construct with `..Default::default()` — the struct
/// gains fields, and spelling every one at each call site is how a new setting
/// comes to be silently ignored at half of them.
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
    /// Which sigmas the steps visit. **Karras is what most published recipes
    /// assume**, so a step count copied from a model card wants it.
    pub scheduler: Scheduler,
    /// See [`ClipSkip`]. Applies to SD 1.x and 2.x.
    pub clip_skip: ClipSkip,
    /// How many images to generate. Seeds run `seed`, `seed + 1`, ... so a
    /// batch is reproducible per image rather than only as a whole.
    pub batch_count: usize,
    pub precision: Precision,
    /// Make the image tile: every convolution wraps at the edge, so the left
    /// and right edges agree and so do the top and bottom.
    ///
    /// **Applies to the decoder as well as the UNet.** A latent that tiles
    /// decoded through a zero-padded VAE has a seam again, which is a
    /// confusing thing to see after the sampler did its part correctly.
    pub seamless: bool,
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
            scheduler: Scheduler::default(),
            clip_skip: ClipSkip::default(),
            batch_count: 1,
            precision: Precision::default(),
            seamless: false,
        }
    }
}

impl Txt2ImgConfig {
    /// The seed for image `i` of a batch.
    ///
    /// **Wrapping, not saturating**: a seed near `u64::MAX` should roll over
    /// rather than give every remaining image of the batch the same one.
    pub fn seed_for(&self, i: usize) -> u64 {
        self.seed.wrapping_add(i as u64)
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
