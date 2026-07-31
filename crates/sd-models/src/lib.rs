//! The models, on MLX.
//!
//! Every architecture this project runs — SD 1.5, SD 2.x, SDXL, unCLIP, SD 3.5,
//! Flux, plus the encoders, adapters and upscalers around them — lives under
//! [`mlx`], each gated against `diffusers` or `transformers` at a measured
//! tolerance by the `mlx_golden_*` tests.
//!
//! What is *not* under `mlx` is the work that never touches a tensor:
//!
//! - [`clip`] and [`t5`]'s tokenizers, which turn text into ids.
//! - [`schedules`], the scalar ladders unCLIP's noise augmentation and the
//!   prior's DDPM step are computed from.
//!
//! They are apart because a second copy of a schedule is how two
//! implementations come to disagree about what step 7 of 20 means.

/// CLIP's tokenizer.
pub mod clip;
/// The models.
///
/// Gated because it is the whole MLX-dependent half of this crate. Without the
/// feature the tokenizers and the scalar schedules still build, which is what
/// lets a machine with no MLX check that the crate graph is intact.
#[cfg(feature = "mlx")]
pub mod mlx;
/// Scalar schedules, shared and tensor-free.
pub mod schedules;
/// T5's tokenizer.
pub mod t5;

/// Kept as `prior` because callers name it that; the contents are scalar.
pub mod prior {
    pub use crate::schedules::{PriorScheduler, StepCoefficients};
}

/// Kept as `unclip` for the same reason.
pub mod unclip {
    pub use crate::schedules::{cosine_alphas_cumprod, TRAIN_TIMESTEPS};
}
