//! TAESD — a tiny distilled autoencoder, interchangeable with the VAE.
//!
//! `madebyollin/taesd` replaces the 330 MB VAE with about 5 MB of 3x3
//! convolutions: no attention, no GroupNorm, no sampling head. It is lossier,
//! and it is cheap enough to decode every step of a run rather than only the
//! last one — which is what makes step previews possible at all.
//!
//! # Its latent convention is its own
//!
//! The SD VAE expects `latent / 0.18215`; TAESD's `scaling_factor` is **1.0**,
//! so it takes the sampler's latent unscaled. Applying the VAE's factor here
//! multiplies the input by 5.5 and produces a washed-out image — a plausible
//! picture, no error, and nothing downstream that can tell. The scaling is
//! therefore *inside* this module rather than left to the caller, and the
//! golden test compares against `AutoencoderTiny.decode` end to end so the
//! convention is covered rather than assumed.
//!
//! Two more conversions live in the forward passes for the same reason:
//! the decoder soft-clamps its input with `tanh(x/3)*3` and returns
//! `2x - 1`, and the encoder takes `(x + 1) / 2`. Every one of them is a
//! silent-wrongness bug if omitted.

use sd_tensor::nn::{conv2d, conv2d_no_bias, Conv2d, Conv2dConfig};
use sd_tensor::{DType, Module, Result, Tensor, VarBuilder};

/// Width of every hidden layer. TAESD is uniform: 64 channels throughout.
const CHANNELS: usize = 64;
/// Residual blocks per stage, decoder order. The encoder's is this reversed.
const DECODER_BLOCKS: [usize; 4] = [3, 3, 3, 1];
const ENCODER_BLOCKS: [usize; 4] = [1, 3, 3, 3];
/// The soft clamp's half-range. `tanh(x/3)*3` maps all of R into `[-3, 3]`.
const CLAMP: f64 = 3.0;

fn conv3x3(in_c: usize, out_c: usize, stride: usize, bias: bool, vb: VarBuilder) -> Result<Conv2d> {
    let cfg = Conv2dConfig {
        padding: 1,
        stride,
        ..Default::default()
    };
    if bias {
        conv2d(in_c, out_c, 3, cfg, vb)
    } else {
        conv2d_no_bias(in_c, out_c, 3, cfg, vb)
    }
}

/// Three convolutions with ReLU between them, added to the input.
///
/// The skip is the identity, not a projection: every block in TAESD has equal
/// input and output width, so the checkpoint carries no `skip` weight at all.
/// A projection built here would fail to load rather than run wrong, which is
/// the good direction.
#[derive(Debug)]
struct TinyBlock {
    conv0: Conv2d,
    conv1: Conv2d,
    conv2: Conv2d,
}

impl TinyBlock {
    fn new(vb: VarBuilder) -> Result<Self> {
        // 0, 2, 4 — the odd indices are the ReLUs, which carry no weights.
        let vb = vb.pp("conv");
        Ok(Self {
            conv0: conv3x3(CHANNELS, CHANNELS, 1, true, vb.pp("0"))?,
            conv1: conv3x3(CHANNELS, CHANNELS, 1, true, vb.pp("2"))?,
            conv2: conv3x3(CHANNELS, CHANNELS, 1, true, vb.pp("4"))?,
        })
    }
}

impl Module for TinyBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.conv0.forward(xs)?.relu()?;
        let h = self.conv1.forward(&h)?.relu()?;
        let h = self.conv2.forward(&h)?;
        // The fuse ReLU is applied *after* the residual add, not before it.
        (h + xs)?.relu()
    }
}

/// One step in the layer stack, so both stacks can be a flat `Vec`.
///
/// Flat because the checkpoint is flat: `layers.0`, `layers.2`, ... including
/// the activation and upsample positions that carry no weights. Modelling it
/// as nested stages would mean maintaining a separate index mapping, and that
/// mapping is exactly what goes wrong.
#[derive(Debug)]
enum Layer {
    Conv(Conv2d),
    Block(TinyBlock),
    Upsample,
    /// The decoder's `layers.1`. It carries no weights, so it loads fine
    /// whether or not it is here — and omitting it left the encoder exact and
    /// the decoder off by 2.5, which is how it was found.
    Relu,
}

impl Layer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Layer::Conv(c) => c.forward(xs),
            Layer::Block(b) => b.forward(xs),
            Layer::Upsample => {
                let (_, _, h, w) = xs.dims4()?;
                xs.upsample_nearest2d(h * 2, w * 2)
            }
            Layer::Relu => xs.relu(),
        }
    }
}

/// TAESD's decoder: latent to image.
#[derive(Debug)]
pub struct TinyDecoder {
    layers: Vec<Layer>,
    dtype: DType,
}

impl TinyDecoder {
    pub fn new(latent_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("decoder").pp("layers");
        let mut layers = Vec::new();
        let mut i = 0usize;

        layers.push(Layer::Conv(conv3x3(
            latent_channels,
            CHANNELS,
            1,
            true,
            vb.pp(i.to_string()),
        )?));
        layers.push(Layer::Relu);
        i += 2; // 1 is that ReLU, which carries no weights

        for (stage, &count) in DECODER_BLOCKS.iter().enumerate() {
            let is_final = stage == DECODER_BLOCKS.len() - 1;
            for _ in 0..count {
                layers.push(Layer::Block(TinyBlock::new(vb.pp(i.to_string()))?));
                i += 1;
            }
            if !is_final {
                layers.push(Layer::Upsample);
                i += 1;
            }
            // The last stage's convolution is the one that carries a bias and
            // narrows to RGB; the others widen nothing and have none.
            let out = if is_final { out_channels } else { CHANNELS };
            layers.push(Layer::Conv(conv3x3(
                CHANNELS,
                out,
                1,
                is_final,
                vb.pp(i.to_string()),
            )?));
            i += 1;
        }

        Ok(Self {
            layers,
            dtype: vb.dtype(),
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Decode `[b, 4, h, w]` to `[b, 3, 8h, 8w]` in `[-1, 1]`.
    ///
    /// Takes the sampler's latent **unscaled** — no `/ 0.18215`. See the module
    /// docs; this is the mistake that produces a washed-out image and no error.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        // Soft clamp into [-3, 3]. TAESD was distilled with it in place, so an
        // out-of-range latent that the VAE would render as a bright artefact is
        // instead squashed — and removing it changes ordinary output too,
        // because tanh is not the identity anywhere.
        let mut h = ((latent.to_dtype(self.dtype)? / CLAMP)?.tanh()? * CLAMP)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        // The stack works in [0, 1]; images elsewhere in this crate are
        // [-1, 1].
        (h * 2.0)? - 1.0
    }
}

/// TAESD's encoder: image to latent.
#[derive(Debug)]
pub struct TinyEncoder {
    layers: Vec<Layer>,
    dtype: DType,
}

impl TinyEncoder {
    pub fn new(in_channels: usize, latent_channels: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("encoder").pp("layers");
        let mut layers = Vec::new();
        let mut i = 0usize;

        for (stage, &count) in ENCODER_BLOCKS.iter().enumerate() {
            // The first convolution reads the image; the rest halve, and only
            // the first carries a bias.
            let conv = if stage == 0 {
                conv3x3(in_channels, CHANNELS, 1, true, vb.pp(i.to_string()))?
            } else {
                conv3x3(CHANNELS, CHANNELS, 2, false, vb.pp(i.to_string()))?
            };
            layers.push(Layer::Conv(conv));
            i += 1;
            for _ in 0..count {
                layers.push(Layer::Block(TinyBlock::new(vb.pp(i.to_string()))?));
                i += 1;
            }
        }
        layers.push(Layer::Conv(conv3x3(
            CHANNELS,
            latent_channels,
            1,
            true,
            vb.pp(i.to_string()),
        )?));

        Ok(Self {
            layers,
            dtype: vb.dtype(),
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Encode `[b, 3, h, w]` in `[-1, 1]` to `[b, 4, h/8, w/8]`.
    ///
    /// Deterministic: unlike the VAE this has no sampling head, so there is no
    /// distribution to take a mean of.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let mut h = ((image.to_dtype(self.dtype)? + 1.0)? * 0.5)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        Ok(h)
    }
}

/// Both halves, from one checkpoint.
#[derive(Debug)]
pub struct TinyAutoencoder {
    pub encoder: TinyEncoder,
    pub decoder: TinyDecoder,
}

impl TinyAutoencoder {
    /// Build from `madebyollin/taesd` (SD 1.5/2.x) or `taesdxl` — the two are
    /// the same architecture with different weights.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            encoder: TinyEncoder::new(3, 4, vb.clone())?,
            decoder: TinyDecoder::new(4, 3, vb)?,
        })
    }
}
