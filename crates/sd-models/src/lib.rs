//! Diffusion model architectures.
//!
//! Parameter names follow HuggingFace `diffusers` conventions so that
//! pretrained weights load without a conversion table. Where a model ships in
//! the original CompVis/LDM layout, conversion belongs in `sd-loader`, not
//! here.
//!
//! # Build order
//!
//! 1. [`vae`] — decoder first. It produces a *visible* result from a known
//!    latent, which validates conv2d, group norm, attention and upsampling
//!    before anything depends on them.
//! 2. [`clip`] — text encoder.
//! 3. [`unet`] — the large one. Verify block by block, never whole-model.

pub mod clip;
pub mod unet;
pub mod vae;
