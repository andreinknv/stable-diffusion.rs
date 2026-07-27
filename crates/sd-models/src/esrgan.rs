//! Real-ESRGAN — 4x upscaling, RRDBNet.
//!
//! A pure convolutional network: no attention, no normalisation of any kind,
//! and no diffusion. It runs *after* generation, taking a finished image and
//! returning one four times the size, so it composes with every architecture
//! here and knows about none of them.
//!
//! # Two residual scalings, both 0.2, both load-bearing
//!
//! Every [`ResidualDenseBlock`] returns `x5 * 0.2 + x`, and every [`Rrdb`]
//! returns `inner * 0.2 + x`. The factor keeps a 23-block stack of unnormalised
//! residuals from diverging — there is no GroupNorm here to rein it in — and
//! dropping either one produces a washed-out or blown-out image rather than an
//! error. They compound: with 23 RRDBs of 3 dense blocks each, the scaling is
//! applied 92 times.
//!
//! # Dense, not sequential
//!
//! Inside a dense block each convolution sees *every* previous output
//! concatenated, so the input widths climb 64, 96, 128, 160, 192 while the
//! outputs stay at 32 (except the last, back to 64). Feeding each convolution
//! only its predecessor's output is the natural misreading, and it loads
//! cleanly for the first layer before failing on the second's channel count.
//!
//! # Range
//!
//! `[0, 1]`, not the `[-1, 1]` that images use elsewhere in this crate.
//! [`RealEsrgan::upscale`] takes and returns `[0, 1]`; the caller converts.

use sd_tensor::nn::{conv2d, Conv2d, Conv2dConfig};
use sd_tensor::{DType, Module, Result, Tensor, VarBuilder};

/// Feature width throughout the trunk.
const FEATURES: usize = 64;
/// Width each dense convolution contributes.
const GROWTH: usize = 32;
/// RRDB blocks in the x4 model.
const BLOCKS: usize = 23;
/// The residual scaling, applied at both levels.
const RESIDUAL_SCALE: f64 = 0.2;
/// The scale factor. Two nearest-neighbour doublings.
const SCALE: usize = 4;
/// Input-space tile edge for [`RealEsrgan::upscale_tiled`].
const TILE: usize = 384;
/// Context kept around each tile, discarded after upscaling.
///
/// 16 input pixels. The theoretical receptive field of 345 padded 3x3
/// convolutions is far larger, but the *effective* one is small — this is the
/// same order of overlap the reference implementation uses, and the seam it
/// leaves is not visible.
const TILE_PAD: usize = 16;

/// LeakyReLU's slope. 0.2, not the 0.01 that is `LeakyReLU`'s default
/// elsewhere — a different slope changes every activation in the network.
const LEAK: f64 = 0.2;

fn conv3x3(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Conv2d> {
    conv2d(
        in_c,
        out_c,
        3,
        Conv2dConfig {
            padding: 1,
            ..Default::default()
        },
        vb,
    )
}

/// `max(x, 0) + slope * min(x, 0)`.
///
/// Written out rather than reached for in `candle_nn` so the slope is visible
/// at the point of use: it is 0.2 here and the default 0.01 would be a silent
/// change to every activation.
fn leaky_relu(xs: &Tensor) -> Result<Tensor> {
    xs.maximum(&xs.zeros_like()?)? + (xs.minimum(&xs.zeros_like()?)? * LEAK)?
}

/// Five convolutions, each seeing every previous output.
#[derive(Debug)]
pub struct ResidualDenseBlock {
    convs: Vec<Conv2d>,
}

impl ResidualDenseBlock {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let mut convs = Vec::with_capacity(5);
        for i in 0..5 {
            // Input grows by `GROWTH` per layer, because each one is handed
            // the concatenation of everything before it.
            let in_c = FEATURES + i * GROWTH;
            let out_c = if i == 4 { FEATURES } else { GROWTH };
            convs.push(conv3x3(in_c, out_c, vb.pp(format!("conv{}", i + 1)))?);
        }
        Ok(Self { convs })
    }
}

impl Module for ResidualDenseBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // `parts` accumulates the input and every activation so far; each
        // convolution takes their concatenation along the channel axis.
        let mut parts: Vec<Tensor> = vec![xs.clone()];
        for (i, conv) in self.convs.iter().enumerate() {
            let input = if parts.len() == 1 {
                parts[0].clone()
            } else {
                Tensor::cat(&parts, 1)?
            };
            let out = conv.forward(&input)?;
            // The last convolution is *not* activated — it is the residual.
            parts.push(if i == 4 { out } else { leaky_relu(&out)? });
        }
        let last = parts.pop().expect("five convolutions ran");
        (last * RESIDUAL_SCALE)? + xs
    }
}

/// Three dense blocks, themselves residual.
#[derive(Debug)]
pub struct Rrdb {
    blocks: [ResidualDenseBlock; 3],
}

impl Rrdb {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            blocks: [
                ResidualDenseBlock::new(vb.pp("rdb1"))?,
                ResidualDenseBlock::new(vb.pp("rdb2"))?,
                ResidualDenseBlock::new(vb.pp("rdb3"))?,
            ],
        })
    }
}

impl Module for Rrdb {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        (h * RESIDUAL_SCALE)? + xs
    }
}

/// Real-ESRGAN x4.
#[derive(Debug)]
pub struct RealEsrgan {
    conv_first: Conv2d,
    body: Vec<Rrdb>,
    conv_body: Conv2d,
    conv_up1: Conv2d,
    conv_up2: Conv2d,
    conv_hr: Conv2d,
    conv_last: Conv2d,
    dtype: DType,
}

impl RealEsrgan {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let vb_body = vb.pp("body");
        let body = (0..BLOCKS)
            .map(|i| Rrdb::new(vb_body.pp(i.to_string())))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_first: conv3x3(3, FEATURES, vb.pp("conv_first"))?,
            body,
            conv_body: conv3x3(FEATURES, FEATURES, vb.pp("conv_body"))?,
            conv_up1: conv3x3(FEATURES, FEATURES, vb.pp("conv_up1"))?,
            conv_up2: conv3x3(FEATURES, FEATURES, vb.pp("conv_up2"))?,
            conv_hr: conv3x3(FEATURES, FEATURES, vb.pp("conv_hr"))?,
            conv_last: conv3x3(FEATURES, 3, vb.pp("conv_last"))?,
            dtype: vb.dtype(),
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Whether an output of this size survives candle's Metal convolution.
    ///
    /// **It silently does not, above `i32::MAX` im2col elements.** A 3x3
    /// convolution over `out_h * out_w` positions at [`FEATURES`] channels
    /// builds an im2col matrix of `out_h * out_w * 64 * 9` elements; past
    /// `i32::MAX` the kernel returns a dark, horizontally banded image with no
    /// error and no failed command buffer.
    ///
    /// Measured exactly rather than inferred: a 1928 px output is correct and
    /// 1936 px is not, and `sqrt(i32::MAX / (64*9))` is 1930. CPU is
    /// unaffected — it renders 2048 px correctly, which is what identified the
    /// fault as candle's Metal path rather than this port.
    fn fits_on_metal(out_h: usize, out_w: usize) -> bool {
        out_h.saturating_mul(out_w).saturating_mul(FEATURES * 9) <= i32::MAX as usize
    }

    /// Upscale, splitting into tiles when the whole image would not survive.
    ///
    /// One pass whenever it fits — so nothing changes for small images, and
    /// the tiled path introduces no seams where it is not needed. On CPU it is
    /// always one pass; the limit is Metal's.
    pub fn upscale_tiled(&self, image: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = image.dims4()?;
        if image.device().is_cpu() || Self::fits_on_metal(h * SCALE, w * SCALE) {
            return self.upscale(image);
        }

        self.upscale_in_tiles(image, TILE, TILE_PAD)
    }

    /// Upscale tile by tile, with an explicit tile size and overlap.
    ///
    /// Separate from [`Self::upscale_tiled`] so the tiling itself is testable
    /// without a 2000 px image: give it a `pad` at least as large as the image
    /// and every tile sees full context, so the result must be *identical* to
    /// one pass — which is what pins the crop offsets and the concatenation
    /// order.
    pub fn upscale_in_tiles(&self, image: &Tensor, tile: usize, pad: usize) -> Result<Tensor> {
        let (_, _, h, w) = image.dims4()?;
        let tile = tile.max(1);
        let mut rows = Vec::new();
        let mut y = 0;
        while y < h {
            let th = tile.min(h - y);
            let mut cols = Vec::new();
            let mut x = 0;
            while x < w {
                let tw = tile.min(w - x);
                // Expand by the overlap, clamped to the image. The tile is
                // upscaled with that context and then cropped back, so every
                // output pixel was computed with real neighbours rather than
                // with an edge.
                let (y0, x0) = (y.saturating_sub(pad), x.saturating_sub(pad));
                let y1 = (y + th + pad).min(h);
                let x1 = (x + tw + pad).min(w);

                let region = image.narrow(2, y0, y1 - y0)?.narrow(3, x0, x1 - x0)?;
                let up = self.upscale(&region)?;
                let cropped = up.narrow(2, (y - y0) * SCALE, th * SCALE)?.narrow(
                    3,
                    (x - x0) * SCALE,
                    tw * SCALE,
                )?;
                cols.push(cropped);
                x += tile;
            }
            rows.push(Tensor::cat(&cols, 3)?);
            y += tile;
        }
        Tensor::cat(&rows, 2)
    }

    /// Upscale `[b, 3, h, w]` in `[0, 1]` to `[b, 3, 4h, 4w]`.
    ///
    /// **`[0, 1]`, not `[-1, 1]`.** Every other image in this crate uses the
    /// signed range; this network was trained on the unsigned one, and feeding
    /// it signed values returns a washed-out image with no error.
    pub fn upscale(&self, image: &Tensor) -> Result<Tensor> {
        let feat = self.conv_first.forward(&image.to_dtype(self.dtype)?)?;

        let mut h = feat.clone();
        for block in &self.body {
            h = block.forward(&h)?;
        }
        // The trunk's long skip: the body's contribution is added to
        // `conv_first`'s output, not used in place of it.
        let mut h = (feat + self.conv_body.forward(&h)?)?;

        // Two nearest-neighbour doublings, each followed by a convolution.
        // Nearest, not bilinear: the convolution after it is what learns the
        // interpolation, and a smooth upsample changes what it receives.
        for conv in [&self.conv_up1, &self.conv_up2] {
            let (_, _, height, width) = h.dims4()?;
            h = leaky_relu(&conv.forward(&h.upsample_nearest2d(height * 2, width * 2)?)?)?;
        }

        self.conv_last
            .forward(&leaky_relu(&self.conv_hr.forward(&h)?)?)
    }
}
