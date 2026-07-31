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
//! Milestone 1 is VAE decoding; text-to-image lands once CLIP and the UNet
//! are verified against the golden harness. See `docs/roadmap.md`.

pub use sd_loader as loader;
pub use sd_models as models;
pub use sd_sample as sample;
pub use sd_tensor as tensor;

pub mod canny;
pub mod image_io;
/// The generation pipeline on MLX.
///
/// Parallel to [`pipeline`] rather than generic over the backend: `sd-tensor`
/// presents MLX with its own shape rather than emulating candle's API, so one
/// pipeline cannot serve both. This is what replaces [`pipeline`] when candle
/// goes.
#[cfg(feature = "mlx")]
pub mod mlx;
pub mod pipeline;

/// Crate version, for `--version` and bug reports.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
