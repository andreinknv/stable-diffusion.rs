//! Variational autoencoder (`AutoencoderKL`).
//!
//! Milestone 1 implements the **decoder** only: latents in, image out. That is
//! deliberate — decoding a reference latent gives a visible, checkable result
//! and exercises most of the op surface the UNet will later need.

mod decoder;
mod encoder;

pub use decoder::{Decoder, DecoderConfig};
pub use encoder::{Encoder, EncoderConfig};

use sd_tensor::nn::{conv2d, Conv2d, Conv2dConfig};
use sd_tensor::{Result, Tensor, VarBuilder};

/// Configuration for `AutoencoderKL`.
///
/// Defaults match the SD 1.x / 2.x VAE. SDXL uses the same geometry and
/// differs only in `scaling_factor`.
#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub latent_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
    /// Latents are stored scaled; decoding divides by this first.
    pub scaling_factor: f64,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self::sd15()
    }
}

impl VaeConfig {
    /// Stable Diffusion 1.x / 2.x.
    pub fn sd15() -> Self {
        Self {
            latent_channels: 4,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            norm_eps: 1e-6,
            scaling_factor: 0.18215,
        }
    }

    /// SDXL. Identical geometry; different latent scaling.
    pub fn sdxl() -> Self {
        Self {
            scaling_factor: 0.13025,
            ..Self::sd15()
        }
    }
}

/// The decode half of `AutoencoderKL`: `post_quant_conv` followed by [`Decoder`].
#[derive(Debug)]
pub struct AutoencoderKlDecoder {
    post_quant_conv: Conv2d,
    decoder: Decoder,
    scaling_factor: f64,
}

impl AutoencoderKlDecoder {
    /// Build from weights. `vb` should be rooted at the VAE itself, so that
    /// `post_quant_conv` and `decoder.*` resolve directly beneath it.
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let post_quant_conv = conv2d(
            cfg.latent_channels,
            cfg.latent_channels,
            1,
            Conv2dConfig::default(),
            vb.pp("post_quant_conv"),
        )?;
        let decoder = Decoder::new(&DecoderConfig::from(cfg), vb.pp("decoder"))?;
        Ok(Self {
            post_quant_conv,
            decoder,
            scaling_factor: cfg.scaling_factor,
        })
    }

    /// Decode *already unscaled* latents `[b, 4, h, w]` to `[b, 3, h*8, w*8]`
    /// in roughly `[-1, 1]`.
    pub fn decode_raw(&self, latents: &Tensor) -> Result<Tensor> {
        let xs = self.post_quant_conv.forward_t(latents)?;
        self.decoder.forward(&xs)
    }

    /// Decode latents as produced by the sampler, applying `scaling_factor`.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        self.decode_raw(&(latents / self.scaling_factor)?)
    }
}

/// The encode half of `AutoencoderKL`: [`Encoder`] followed by `quant_conv`.
///
/// Needed for img2img, which starts from a real image rather than from noise.
#[derive(Debug)]
pub struct AutoencoderKlEncoder {
    encoder: Encoder,
    quant_conv: Conv2d,
    latent_channels: usize,
    scaling_factor: f64,
}

impl AutoencoderKlEncoder {
    /// `vb` should be rooted at the VAE itself, so `encoder.*` and
    /// `quant_conv` resolve directly beneath it.
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(&EncoderConfig::from(cfg), vb.pp("encoder"))?;
        // Operates on the concatenated (mean, logvar), hence 2x on both sides.
        let quant_conv = conv2d(
            2 * cfg.latent_channels,
            2 * cfg.latent_channels,
            1,
            Conv2dConfig::default(),
            vb.pp("quant_conv"),
        )?;
        Ok(Self {
            encoder,
            quant_conv,
            latent_channels: cfg.latent_channels,
            scaling_factor: cfg.scaling_factor,
        })
    }

    /// Encode `[b, 3, h, w]` in `[-1, 1]` to the latent distribution's
    /// `(mean, logvar)`, each `[b, latent_channels, h/8, w/8]`.
    ///
    /// Unscaled — this is the distribution as the model expresses it.
    pub fn encode_dist(&self, image: &Tensor) -> Result<(Tensor, Tensor)> {
        let moments = self.quant_conv.forward_t(&self.encoder.forward(image)?)?;
        let mean = moments.narrow(1, 0, self.latent_channels)?;
        let logvar = moments.narrow(1, self.latent_channels, self.latent_channels)?;
        Ok((mean.contiguous()?, logvar.contiguous()?))
    }

    /// Encode to a latent ready for the sampler, applying `scaling_factor`.
    ///
    /// Uses the distribution's **mean** rather than sampling from it. For
    /// img2img that is what you want: the sampler adds its own noise, so
    /// drawing here too would add variance the seed does not control and make
    /// the result irreproducible.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let (mean, _) = self.encode_dist(image)?;
        mean * self.scaling_factor
    }

    /// Sample from the latent distribution using externally supplied noise.
    ///
    /// `noise` must be standard normal and shaped like the latent. Kept
    /// separate from [`Self::encode`] so the caller owns the randomness, and
    /// with it reproducibility.
    pub fn encode_sampled(&self, image: &Tensor, noise: &Tensor) -> Result<Tensor> {
        let (mean, logvar) = self.encode_dist(image)?;
        // diffusers clamps logvar to [-30, 20] before exponentiating; without
        // it a degenerate checkpoint can produce an infinite std.
        let std = (logvar.clamp(-30.0, 20.0)? * 0.5)?.exp()?;
        let sample = (mean + (std * noise)?)?;
        sample * self.scaling_factor
    }

    /// Latent channel count, for sizing the noise [`Self::encode_sampled`]
    /// expects.
    pub fn latent_channels(&self) -> usize {
        self.latent_channels
    }
}

trait ForwardT {
    fn forward_t(&self, xs: &Tensor) -> Result<Tensor>;
}

impl ForwardT for Conv2d {
    fn forward_t(&self, xs: &Tensor) -> Result<Tensor> {
        use sd_tensor::Module;
        self.forward(xs)
    }
}
