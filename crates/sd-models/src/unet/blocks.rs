//! The four kinds of block the UNet is assembled from.
//!
//! Composition only — the resnets come from `resnet.rs` and the transformers
//! from `attention.rs`, unchanged.
//!
//! The asymmetry to keep in mind: **down blocks hold `layers_per_block`
//! resnets, up blocks hold `layers_per_block + 1`.** It is real, it is what
//! the checkpoint contains, and using the same count for both fails to load
//! `up_blocks.0.resnets.2`.

use sd_tensor::nn::{conv2d, Conv2d, Conv2dConfig};
use sd_tensor::{Module, Result, Tensor, VarBuilder};

use super::attention::Transformer2DModel;
use super::resnet::ResnetBlock2D;

/// Strided conv that halves the spatial dims.
#[derive(Debug)]
pub struct Downsample2D {
    conv: Conv2d,
}

impl Downsample2D {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: conv2d(
                channels,
                channels,
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("conv"),
            )?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv.forward(xs)
    }
}

/// Nearest-neighbour 2x upsample then a 3x3 conv — not a transposed conv.
#[derive(Debug)]
pub struct Upsample2D {
    conv: Conv2d,
}

impl Upsample2D {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: conv2d(
                channels,
                channels,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("conv"),
            )?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        let xs = xs.upsample_nearest2d(h * 2, w * 2)?;
        self.conv.forward(&xs)
    }
}

/// A down block: `layers_per_block` resnets, optionally each followed by a
/// transformer, optionally ending in a downsampler.
#[derive(Debug)]
pub struct DownBlock2D {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<Transformer2DModel>,
    downsampler: Option<Downsample2D>,
}

/// Cross-attention settings for a block that has it.
#[derive(Debug, Clone, Copy)]
pub struct AttentionSpec {
    /// Head *count*. `dim_head` is derived as `channels / heads`.
    pub heads: usize,
    pub cross_dim: usize,
    /// Transformer blocks per attention module. 1 throughout SD 1.5; SDXL
    /// uses [1, 2, 10], so the deepest block carries ten.
    pub depth: usize,
}

/// How a block is configured. Grouped into a struct because passing eight
/// positional numbers to two constructors is how the head count and the head
/// dim get swapped.
pub struct BlockConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub temb_channels: usize,
    pub num_layers: usize,
    pub groups: usize,
    pub eps: f64,
    /// `Some(spec)` adds a transformer after every resnet.
    pub attention: Option<AttentionSpec>,
    pub resample: bool,
}

impl DownBlock2D {
    pub fn new(cfg: &BlockConfig, vb: VarBuilder) -> Result<Self> {
        let vb_resnets = vb.pp("resnets");
        let vb_attn = vb.pp("attentions");
        let mut resnets = Vec::with_capacity(cfg.num_layers);
        let mut attentions = Vec::new();

        for i in 0..cfg.num_layers {
            let in_c = if i == 0 {
                cfg.in_channels
            } else {
                cfg.out_channels
            };
            resnets.push(ResnetBlock2D::new(
                in_c,
                cfg.out_channels,
                cfg.temb_channels,
                cfg.groups,
                cfg.eps,
                vb_resnets.pp(i.to_string()),
            )?);
            if let Some(a) = cfg.attention {
                attentions.push(Transformer2DModel::new(
                    cfg.out_channels,
                    a.heads,
                    cfg.out_channels / a.heads,
                    a.depth,
                    a.cross_dim,
                    vb_attn.pp(i.to_string()),
                )?);
            }
        }

        let downsampler = if cfg.resample {
            Some(Downsample2D::new(
                cfg.out_channels,
                vb.pp("downsamplers").pp("0"),
            )?)
        } else {
            None
        };

        Ok(Self {
            resnets,
            attentions,
            downsampler,
        })
    }

    /// Runs the block, pushing every intermediate onto `skips`.
    ///
    /// One push per resnet(+attention) pair, plus one for the downsampler
    /// output. Missing either shifts the whole stack and every up block then
    /// concatenates the wrong tensor — at shapes that stay valid for several
    /// of them, so it does not fail where the mistake is.
    pub fn forward(
        &self,
        xs: &Tensor,
        temb: &Tensor,
        context: &Tensor,
        skips: &mut Vec<Tensor>,
    ) -> Result<Tensor> {
        let mut h = xs.clone();
        for (i, resnet) in self.resnets.iter().enumerate() {
            h = resnet.forward(&h, temb)?;
            if let Some(attn) = self.attentions.get(i) {
                h = attn.forward(&h, context)?;
            }
            skips.push(h.clone());
        }
        if let Some(down) = &self.downsampler {
            h = down.forward(&h)?;
            skips.push(h.clone());
        }
        Ok(h)
    }
}

/// An up block: `layers_per_block + 1` resnets, each consuming one skip.
#[derive(Debug)]
pub struct UpBlock2D {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<Transformer2DModel>,
    upsampler: Option<Upsample2D>,
}

impl UpBlock2D {
    /// `skip_channels` is the channel count of the skip each resnet consumes,
    /// in the order they are popped. Derived from the down pass rather than
    /// hardcoded, because an up resnet's input width is
    /// `prev_channels + skip_channels` and that is not obvious from the config.
    pub fn new(cfg: &BlockConfig, skip_channels: &[usize], vb: VarBuilder) -> Result<Self> {
        let vb_resnets = vb.pp("resnets");
        let vb_attn = vb.pp("attentions");
        let mut resnets = Vec::with_capacity(cfg.num_layers);
        let mut attentions = Vec::new();

        for (i, &skip) in skip_channels.iter().enumerate().take(cfg.num_layers) {
            let prev = if i == 0 {
                cfg.in_channels
            } else {
                cfg.out_channels
            };
            resnets.push(ResnetBlock2D::new(
                prev + skip,
                cfg.out_channels,
                cfg.temb_channels,
                cfg.groups,
                cfg.eps,
                vb_resnets.pp(i.to_string()),
            )?);
            if let Some(a) = cfg.attention {
                attentions.push(Transformer2DModel::new(
                    cfg.out_channels,
                    a.heads,
                    cfg.out_channels / a.heads,
                    a.depth,
                    a.cross_dim,
                    vb_attn.pp(i.to_string()),
                )?);
            }
        }

        let upsampler = if cfg.resample {
            Some(Upsample2D::new(
                cfg.out_channels,
                vb.pp("upsamplers").pp("0"),
            )?)
        } else {
            None
        };

        Ok(Self {
            resnets,
            attentions,
            upsampler,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        temb: &Tensor,
        context: &Tensor,
        skips: &mut Vec<Tensor>,
    ) -> Result<Tensor> {
        let mut h = xs.clone();
        for (i, resnet) in self.resnets.iter().enumerate() {
            let skip = skips.pop().ok_or_else(|| {
                sd_tensor::Error::Msg("up block ran out of skip connections".to_string())
            })?;
            // [h, skip] along the channel axis, never [skip, h]. The channel
            // count works out either way and the numbers are wrong.
            h = Tensor::cat(&[&h, &skip], 1)?;
            h = resnet.forward(&h, temb)?;
            if let Some(attn) = self.attentions.get(i) {
                h = attn.forward(&h, context)?;
            }
        }
        if let Some(up) = &self.upsampler {
            h = up.forward(&h)?;
        }
        Ok(h)
    }
}

/// The bottleneck: resnet, transformer, resnet.
#[derive(Debug)]
pub struct MidBlock2DCrossAttn {
    resnet_0: ResnetBlock2D,
    attention: Transformer2DModel,
    resnet_1: ResnetBlock2D,
}

impl MidBlock2DCrossAttn {
    pub fn new(
        channels: usize,
        temb_channels: usize,
        attention: AttentionSpec,
        groups: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_resnets = vb.pp("resnets");
        Ok(Self {
            resnet_0: ResnetBlock2D::new(
                channels,
                channels,
                temb_channels,
                groups,
                eps,
                vb_resnets.pp("0"),
            )?,
            attention: Transformer2DModel::new(
                channels,
                attention.heads,
                channels / attention.heads,
                attention.depth,
                attention.cross_dim,
                vb.pp("attentions").pp("0"),
            )?,
            resnet_1: ResnetBlock2D::new(
                channels,
                channels,
                temb_channels,
                groups,
                eps,
                vb_resnets.pp("1"),
            )?,
        })
    }

    pub fn forward(&self, xs: &Tensor, temb: &Tensor, context: &Tensor) -> Result<Tensor> {
        let h = self.resnet_0.forward(xs, temb)?;
        let h = self.attention.forward(&h, context)?;
        self.resnet_1.forward(&h, temb)
    }
}
