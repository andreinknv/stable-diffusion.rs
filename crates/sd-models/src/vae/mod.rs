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

    /// [`Self::decode`], in overlapping tiles.
    ///
    /// A whole-image decode allocates a conv im2col of `cin * 9` values per
    /// output position — 9.66 GB at 1024px, which does not fit in GPU memory
    /// on a 36 GiB Mac. Tiling caps that at whatever one tile needs, so the
    /// cost becomes linear in area instead of a single enormous allocation.
    ///
    /// The result is *close to* but not identical to a whole-image decode:
    /// convolutions at a tile edge see padding where they would otherwise see
    /// neighbouring pixels, which the overlap blend hides rather than
    /// eliminates. Small latents that already fit are decoded whole, so this
    /// is safe to call unconditionally.
    pub fn decode_tiled(&self, latents: &Tensor) -> Result<Tensor> {
        self.decode_raw_tiled(&(latents / self.scaling_factor)?)
    }

    /// [`Self::decode_raw`], in overlapping tiles.
    pub fn decode_raw_tiled(&self, latents: &Tensor) -> Result<Tensor> {
        let (_, _, lh, lw) = latents.dims4()?;
        if lh <= TILE_LATENT_EDGE && lw <= TILE_LATENT_EDGE {
            return self.decode_raw(latents);
        }

        let tile = TILE_LATENT_EDGE;
        let stride = ((tile as f64) * (1.0 - TILE_OVERLAP)) as usize;
        let stride = stride.max(1);
        // In output pixels: the decoder upsamples by 8.
        let scale = 8;
        let blend_extent = (tile - stride) * scale;
        let keep = stride * scale;

        let mut rows: Vec<Vec<Tensor>> = Vec::new();
        let mut y = 0;
        while y < lh {
            let h = tile.min(lh - y);
            let mut row = Vec::new();
            let mut x = 0;
            while x < lw {
                let w = tile.min(lw - x);
                let patch = latents.narrow(2, y, h)?.narrow(3, x, w)?.contiguous()?;
                row.push(self.decode_raw(&patch)?);
                if x + w >= lw {
                    break;
                }
                x += stride;
            }
            rows.push(row);
            if y + h >= lh {
                break;
            }
            y += stride;
        }

        // Blend each tile into its upper and left neighbours, then trim to the
        // stride so the kept regions tile exactly.
        let mut out_rows = Vec::with_capacity(rows.len());
        for i in 0..rows.len() {
            let mut out_row: Vec<Tensor> = Vec::with_capacity(rows[i].len());
            for j in 0..rows[i].len() {
                let mut t = rows[i][j].clone();
                if i > 0 {
                    t = blend(&rows[i - 1][j], &t, blend_extent, 2)?;
                }
                if j > 0 {
                    // The *untrimmed* left neighbour, not the trimmed result
                    // already pushed to `out_row` — those differ in height on
                    // any row whose tiles were cut short, and blending across
                    // that mismatch is a shape error.
                    t = blend(&rows[i][j - 1], &t, blend_extent, 3)?;
                }
                let last_col = j + 1 == rows[i].len();
                let last_row = i + 1 == rows.len();
                let th = t.dims()[2];
                let tw = t.dims()[3];
                let h = if last_row { th } else { keep.min(th) };
                let w = if last_col { tw } else { keep.min(tw) };
                out_row.push(t.narrow(2, 0, h)?.narrow(3, 0, w)?.contiguous()?);
            }
            out_rows.push(Tensor::cat(&out_row, 3)?);
        }
        Tensor::cat(&out_rows, 2)
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

/// Latent edge of one decode tile. 64 latent = 512px, the size SD 1.5 was
/// trained at and comfortably inside GPU memory.
pub const TILE_LATENT_EDGE: usize = 64;

/// Fraction of a tile that overlaps its neighbour.
///
/// The decoder is not shift-invariant — its convolutions see different
/// padding at a tile edge than they would mid-image — so tiles must overlap
/// and be blended, or the seams are visible as hard lines.
const TILE_OVERLAP: f64 = 0.25;

/// Linear ramp from 0 to 1 along `dim`, shaped to broadcast over `[b, c, h, w]`.
fn ramp(extent: usize, dim: usize, device: &sd_tensor::Device) -> Result<Tensor> {
    let values: Vec<f32> = (0..extent)
        .map(|i| (i as f32 + 0.5) / extent as f32)
        .collect();
    let t = Tensor::from_vec(values, extent, device)?;
    match dim {
        2 => t.reshape((1, 1, extent, 1)),
        _ => t.reshape((1, 1, 1, extent)),
    }
}

/// Cross-fade `b` into the trailing `extent` of `a` along `dim`.
///
/// Returns `b` with its leading `extent` replaced by the blend. Tensors are
/// immutable here, so this rebuilds rather than writing in place.
fn blend(a: &Tensor, b: &Tensor, extent: usize, dim: usize) -> Result<Tensor> {
    let a_len = a.dims()[dim];
    let b_len = b.dims()[dim];
    let extent = extent.min(a_len).min(b_len);
    if extent == 0 {
        return Ok(b.clone());
    }
    let w = ramp(extent, dim, a.device())?;
    let a_tail = a.narrow(dim, a_len - extent, extent)?;
    let b_head = b.narrow(dim, 0, extent)?;
    // a fades out as b fades in.
    let mixed = (a_tail.broadcast_mul(&(1.0 - &w)?)? + b_head.broadcast_mul(&w)?)?;
    if b_len == extent {
        return Ok(mixed);
    }
    let rest = b.narrow(dim, extent, b_len - extent)?;
    Tensor::cat(&[&mixed, &rest], dim)
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
