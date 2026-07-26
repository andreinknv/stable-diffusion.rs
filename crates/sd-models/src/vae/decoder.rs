//! `AutoencoderKL` decoder.
//!
//! Layout and parameter names follow `diffusers`
//! (`diffusers/models/autoencoders/vae.py::Decoder`) so SD 1.x/2.x/XL weights
//! load unmodified.
//!
//! Graph, for `block_out_channels = [128, 256, 512, 512]`:
//!
//! ```text
//! conv_in (4 -> 512)
//! mid_block: resnet -> attention -> resnet          (512)
//! up_blocks[0]: 3x resnet (512 -> 512) + upsample   -> 2x
//! up_blocks[1]: 3x resnet (512 -> 512) + upsample   -> 4x
//! up_blocks[2]: 3x resnet (512 -> 256) + upsample   -> 8x
//! up_blocks[3]: 3x resnet (256 -> 128)              (no upsample)
//! conv_norm_out -> silu -> conv_out (128 -> 3)
//! ```
//!
//! Note the up-block channel counts are the *reverse* of
//! `block_out_channels`, and each up block has `layers_per_block + 1` resnets
//! — one more than the encoder. Getting either wrong yields weights that load
//! cleanly and produce garbage.

use sd_tensor::nn::{conv2d, group_norm, linear, Conv2d, Conv2dConfig, GroupNorm, Linear};
use sd_tensor::{ops, DType, Module, Result, Tensor, VarBuilder};

use super::VaeConfig;

#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub latent_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
}

impl DecoderConfig {
    /// Bytes in the largest single activation tensor a decode of this latent
    /// size will allocate.
    ///
    /// Attention used to be the allocation that mattered, and refusing an
    /// oversized one is what kept a too-large decode from wedging the machine.
    /// `ops::chunked_attention` bounds attention by construction, which moves
    /// the largest allocation here: an up block at full resolution. At a 384
    /// latent that is 9.0 GiB in a single tensor — wired on Metal and
    /// unreclaimable — so it needs the same refusal attention used to provide.
    ///
    /// This is on the config rather than the built decoder so a caller can
    /// cost a decode *before* constructing one. `None` means the count
    /// overflowed, which is itself a refusal.
    pub fn peak_activation_bytes(
        &self,
        batch: usize,
        latent_h: usize,
        latent_w: usize,
        dtype: DType,
    ) -> Option<u64> {
        let (b, h, w) = (batch as u64, latent_h as u64, latent_w as u64);
        // Up blocks run in reverse order, and every one but the last doubles
        // the spatial scale.
        let reversed = self.block_out_channels.iter().rev();
        let last = self.block_out_channels.len().saturating_sub(1);

        // conv_in and mid_block run at latent resolution, on the widest layer.
        let mut peak = b
            .checked_mul(*self.block_out_channels.last()? as u64)?
            .checked_mul(h)?
            .checked_mul(w)?;

        let mut scale = 1u64;
        for (i, &channels) in reversed.enumerate() {
            if i != last {
                scale = scale.checked_mul(2)?;
            }
            // The post-upsample output is the largest tensor a block holds.
            let elems = b
                .checked_mul(channels as u64)?
                .checked_mul(h.checked_mul(scale)?)?
                .checked_mul(w.checked_mul(scale)?)?;
            peak = peak.max(elems);
        }
        peak.checked_mul(dtype.size_in_bytes() as u64)
    }
}

impl From<&VaeConfig> for DecoderConfig {
    fn from(c: &VaeConfig) -> Self {
        Self {
            latent_channels: c.latent_channels,
            out_channels: c.out_channels,
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

/// `ResnetBlock2D` without time embedding (the VAE variant).
#[derive(Debug)]
pub(super) struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    /// 1x1 projection, present only when `in_channels != out_channels`.
    conv_shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    pub(super) fn new(
        in_c: usize,
        out_c: usize,
        groups: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = group_norm(groups, in_c, eps, vb.pp("norm1"))?;
        let conv1 = conv3x3(in_c, out_c, vb.pp("conv1"))?;
        let norm2 = group_norm(groups, out_c, eps, vb.pp("norm2"))?;
        let conv2 = conv3x3(out_c, out_c, vb.pp("conv2"))?;
        let conv_shortcut = if in_c != out_c {
            Some(conv2d(
                in_c,
                out_c,
                1,
                Conv2dConfig::default(),
                vb.pp("conv_shortcut"),
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
        })
    }

    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(xs)?;
        let h = ops::silu(&h)?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h)?;
        let h = ops::silu(&h)?;
        let h = self.conv2.forward(&h)?;
        let residual = match &self.conv_shortcut {
            Some(c) => c.forward(xs)?,
            None => xs.clone(),
        };
        h + residual
    }
}

/// Single-head spatial self-attention over `[b, c, h, w]`.
///
/// diffusers stores this as `Attention` with `to_q/to_k/to_v/to_out.0` and a
/// `group_norm`. `residual_connection = true`, `rescale_output_factor = 1.0`.
#[derive(Debug)]
struct AttentionBlock {
    group_norm: GroupNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
}

impl AttentionBlock {
    fn new(channels: usize, groups: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            group_norm: group_norm(groups, channels, eps, vb.pp("group_norm"))?,
            to_q: linear(channels, channels, vb.pp("to_q"))?,
            to_k: linear(channels, channels, vb.pp("to_k"))?,
            to_v: linear(channels, channels, vb.pp("to_v"))?,
            to_out: linear(channels, channels, vb.pp("to_out").pp("0"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        let residual = xs;

        // GroupNorm over channels; flattening spatial dims first or after is
        // equivalent, so normalise in NCHW then reshape to [b, hw, c].
        let ys = self.group_norm.forward(xs)?;
        let ys = ys.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?;

        let q = self.to_q.forward(&ys)?;
        let k = self.to_k.forward(&ys)?;
        let v = self.to_v.forward(&ys)?;

        // Single head: insert a head axis so the shared attention helper
        // (which expects [b, heads, seq, dim]) applies unchanged.
        let q = q.unsqueeze(1)?;
        let k = k.unsqueeze(1)?;
        let v = v.unsqueeze(1)?;
        let ys = ops::scaled_dot_product_attention(&q, &k, &v)?.squeeze(1)?;

        let ys = self.to_out.forward(&ys)?;
        let ys = ys.transpose(1, 2)?.reshape((b, c, h, w))?;
        ys + residual
    }
}

/// Nearest-neighbour 2x upsample followed by a 3x3 conv.
#[derive(Debug)]
struct Upsample2D {
    conv: Conv2d,
}

impl Upsample2D {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: conv3x3(channels, channels, vb.pp("conv"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        let xs = xs.upsample_nearest2d(h * 2, w * 2)?;
        self.conv.forward(&xs)
    }
}

/// `UNetMidBlock2D`: resnet, attention, resnet.
#[derive(Debug)]
pub(super) struct MidBlock {
    resnet_in: ResnetBlock,
    attention: AttentionBlock,
    resnet_out: ResnetBlock,
}

impl MidBlock {
    pub(super) fn new(channels: usize, groups: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let resnets = vb.pp("resnets");
        Ok(Self {
            resnet_in: ResnetBlock::new(channels, channels, groups, eps, resnets.pp("0"))?,
            attention: AttentionBlock::new(channels, groups, eps, vb.pp("attentions").pp("0"))?,
            resnet_out: ResnetBlock::new(channels, channels, groups, eps, resnets.pp("1"))?,
        })
    }

    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.resnet_in.forward(xs)?;
        let h = self.attention.forward(&h)?;
        self.resnet_out.forward(&h)
    }
}

/// `UpDecoderBlock2D`: N resnets, then an optional upsample.
#[derive(Debug)]
struct UpDecoderBlock {
    resnets: Vec<ResnetBlock>,
    upsampler: Option<Upsample2D>,
}

impl UpDecoderBlock {
    fn new(
        in_c: usize,
        out_c: usize,
        num_layers: usize,
        add_upsample: bool,
        groups: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_res = vb.pp("resnets");
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            // Only the first resnet changes channel count.
            let block_in = if i == 0 { in_c } else { out_c };
            resnets.push(ResnetBlock::new(
                block_in,
                out_c,
                groups,
                eps,
                vb_res.pp(i.to_string()),
            )?);
        }
        let upsampler = if add_upsample {
            Some(Upsample2D::new(out_c, vb.pp("upsamplers").pp("0"))?)
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        match &self.upsampler {
            Some(u) => u.forward(&h),
            None => Ok(h),
        }
    }
}

/// The decoder.
#[derive(Debug)]
pub struct Decoder {
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpDecoderBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    /// Kept for `peak_activation_bytes`, which gates oversized decodes.
    cfg: DecoderConfig,
}

impl Decoder {
    pub fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let groups = cfg.norm_num_groups;
        let eps = cfg.norm_eps;

        let reversed: Vec<usize> = cfg.block_out_channels.iter().copied().rev().collect();
        let first = *reversed.first().ok_or_else(|| {
            sd_tensor::Error::Msg("block_out_channels must not be empty".to_string())
        })?;
        let last = *cfg.block_out_channels.first().expect("checked non-empty");

        let conv_in = conv3x3(cfg.latent_channels, first, vb.pp("conv_in"))?;
        let mid_block = MidBlock::new(first, groups, eps, vb.pp("mid_block"))?;

        let vb_up = vb.pp("up_blocks");
        let mut up_blocks = Vec::with_capacity(reversed.len());
        let mut prev_c = first;
        for (i, &out_c) in reversed.iter().enumerate() {
            let is_final = i == reversed.len() - 1;
            up_blocks.push(UpDecoderBlock::new(
                prev_c,
                out_c,
                cfg.layers_per_block + 1,
                !is_final,
                groups,
                eps,
                vb_up.pp(i.to_string()),
            )?);
            prev_c = out_c;
        }

        let conv_norm_out = group_norm(groups, last, eps, vb.pp("conv_norm_out"))?;
        let conv_out = conv3x3(last, cfg.out_channels, vb.pp("conv_out"))?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            cfg: cfg.clone(),
        })
    }

    /// `[b, latent_channels, h, w]` -> `[b, out_channels, h*8, w*8]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, _, lh, lw) = xs.dims4()?;
        ops::check_alloc_budget(
            self.cfg.peak_activation_bytes(b, lh, lw, xs.dtype()),
            &format!("VAE decode activation for a {lh}x{lw} latent"),
        )?;

        let h = self.conv_in.forward(xs)?;
        let mut h = self.mid_block.forward(&h)?;
        for block in &self.up_blocks {
            h = block.forward(&h)?;
        }
        let h = self.conv_norm_out.forward(&h)?;
        let h = ops::silu(&h)?;
        self.conv_out.forward(&h)
    }
}
