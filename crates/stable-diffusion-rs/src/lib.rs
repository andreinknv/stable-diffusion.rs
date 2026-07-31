//! Diffusion model inference in pure Rust.
//!
//! This is the public API surface. It re-exports the pieces users need and
//! keeps the internal crate split an implementation detail.
//!
//! The crate name is long because `sd` is taken on crates.io. Alias it:
//!
//! ```
//! use stable_diffusion_rs as sd;
//! ```
//!
//! Runs on MLX. `sd-tensor` is the only crate that names a backend, and
//! `scripts/check-seam.sh` enforces it.

pub use sd_loader as loader;
pub use sd_models as models;
pub use sd_sample as sample;
pub use sd_tensor as tensor;

#[cfg(feature = "mlx")]
pub mod canny;
/// Generation settings, backend-free.
pub mod config;
/// The generation pipelines.
#[cfg(feature = "mlx")]
pub mod mlx;

/// Kept as `pipeline` because that is what callers import; the settings are
/// backend-free and live in [`config`].
pub mod pipeline {
    pub use crate::config::*;
}

/// Crate version, for `--version` and bug reports.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
