//! `AutoencoderKL` encoder — image in, latent distribution out.
//!
//! Layout and parameter names follow `diffusers`
//! (`diffusers/models/autoencoders/vae.py::Encoder`).
//!
//! Graph, for `block_out_channels = [128, 256, 512, 512]`:
//!
//! ```text
//! conv_in (3 -> 128)
//! down_blocks[0]: 2x resnet (128 -> 128) + downsample  -> 1/2
//! down_blocks[1]: 2x resnet (128 -> 256) + downsample  -> 1/4
//! down_blocks[2]: 2x resnet (256 -> 512) + downsample  -> 1/8
//! down_blocks[3]: 2x resnet (512 -> 512)               (no downsample)
//! mid_block: resnet -> attention -> resnet             (512)
//! conv_norm_out -> silu -> conv_out (512 -> 8)
//! ```
//!
//! `conv_out` emits **twice** `latent_channels`: the first half is the mean of
//! the latent distribution and the second the log-variance. Emitting 4 instead
//! of 8 fails to load; taking the wrong half loads fine and yields noise.
//!
//! Down blocks have `layers_per_block` resnets — one *fewer* than the
//! decoder's up blocks, which have `layers_per_block + 1`.

use sd_tensor::nn::{conv2d, group_norm, Conv2d, Conv2dConfig, GroupNorm};
use sd_tensor::{ops, Module, Result, Tensor, VarBuilder};

use super::decoder::{MidBlock, ResnetBlock};
use super::VaeConfig;

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub in_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
}

impl From<&VaeConfig> for EncoderConfig {
    fn from(c: &VaeConfig) -> Self {
        Self {
            // The VAE is symmetric: it consumes what the decoder emits.
            in_channels: c.out_channels,
            latent_channels: c.latent_channels,
            block_out_channels: c.block_out_channels.clone(),
            layers_per_block: c.layers_per_block,
            norm_num_groups: c.norm_num_groups,
            norm_eps: c.norm_eps,
        }
    }
}

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

/// Stride-2 conv that halves the spatial dims.
///
/// The padding here is **asymmetric** — one row at the bottom and one column
/// at the right, none at the top or left — which is what diffusers does
/// (`Downsample2D` with `padding=0`, then `F.pad(x, (0, 1, 0, 1))`). A
/// symmetric `padding: 1` runs, produces the right shape, and shifts the whole
/// image half a pixel per downsample.
#[derive(Debug)]
struct Downsample2D {
    conv: Conv2d,
}

impl Downsample2D {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: conv2d(
                channels,
                channels,
                3,
                Conv2dConfig {
                    stride: 2,
                    padding: 0,
                    ..Default::default()
                },
                vb.pp("conv"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // (left, right) on width, then on height.
        let xs = xs.pad_with_zeros(3, 0, 1)?;
        let xs = xs.pad_with_zeros(2, 0, 1)?;
        self.conv.forward(&xs)
    }
}

/// `DownEncoderBlock2D`: N resnets, then an optional downsample.
#[derive(Debug)]
struct DownEncoderBlock {
    resnets: Vec<ResnetBlock>,
    downsampler: Option<Downsample2D>,
}

impl DownEncoderBlock {
    fn new(
        in_c: usize,
        out_c: usize,
        layers: usize,
        downsample: bool,
        groups: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_resnets = vb.pp("resnets");
        let mut resnets = Vec::with_capacity(layers);
        for i in 0..layers {
            let from = if i == 0 { in_c } else { out_c };
            resnets.push(ResnetBlock::new(
                from,
                out_c,
                groups,
                eps,
                vb_resnets.pp(i.to_string()),
            )?);
        }
        let downsampler = if downsample {
            Some(Downsample2D::new(out_c, vb.pp("downsamplers").pp("0"))?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsampler,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }
        match &self.downsampler {
            Some(d) => d.forward(&h),
            None => Ok(h),
        }
    }
}

/// The encoder.
#[derive(Debug)]
pub struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<DownEncoderBlock>,
    mid_block: MidBlock,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Encoder {
    pub fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let channels = &cfg.block_out_channels;
        let first = *channels.first().ok_or_else(|| {
            sd_tensor::Error::Msg("block_out_channels must not be empty".to_string())
        })?;
        let last = *channels.last().expect("checked non-empty");

        let conv_in = conv3x3(cfg.in_channels, first, vb.pp("conv_in"))?;

        let vb_down = vb.pp("down_blocks");
        let mut down_blocks = Vec::with_capacity(channels.len());
        let mut prev = first;
        for (i, &out_c) in channels.iter().enumerate() {
            let is_final = i == channels.len() - 1;
            down_blocks.push(DownEncoderBlock::new(
                prev,
                out_c,
                cfg.layers_per_block,
                !is_final,
                cfg.norm_num_groups,
                cfg.norm_eps,
                vb_down.pp(i.to_string()),
            )?);
            prev = out_c;
        }

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block: MidBlock::new(last, cfg.norm_num_groups, cfg.norm_eps, vb.pp("mid_block"))?,
            conv_norm_out: group_norm(
                cfg.norm_num_groups,
                last,
                cfg.norm_eps,
                vb.pp("conv_norm_out"),
            )?,
            // Twice the latent channels: mean and log-variance, concatenated.
            conv_out: conv3x3(last, 2 * cfg.latent_channels, vb.pp("conv_out"))?,
        })
    }

    /// `[b, 3, h, w]` -> `[b, 2 * latent_channels, h/8, w/8]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(xs)?;
        for block in &self.down_blocks {
            h = block.forward(&h)?;
        }
        let h = self.mid_block.forward(&h)?;
        let h = self.conv_norm_out.forward(&h)?;
        let h = ops::silu(&h)?;
        self.conv_out.forward(&h)
    }
}
