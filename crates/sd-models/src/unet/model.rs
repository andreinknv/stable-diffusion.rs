//! `UNet2DConditionModel` — the assembly.
//!
//! No new math here. The skip connections are the whole difficulty: the stack
//! must be pushed and popped in exactly the right order, and the up-block
//! resnets are wider than the config suggests because each one consumes a skip
//! alongside its input.

use sd_tensor::nn::{conv2d, group_norm, Conv2d, Conv2dConfig, GroupNorm};
use sd_tensor::{ops, Module, Result, Tensor, VarBuilder};

use super::blocks::{BlockConfig, DownBlock2D, MidBlock2DCrossAttn, UpBlock2D};
use super::embeddings::{timestep_embedding, TimestepEmbedding};

/// Geometry of the denoiser.
#[derive(Debug, Clone)]
pub struct UNetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    /// **The number of attention heads, despite the name.** SD 1.5's config
    /// calls this `attention_head_dim` and diffusers reads it as a head count,
    /// so at 320 channels this is 8 heads of 40 — not heads of width 8.
    pub attention_head_dim: usize,
    pub cross_attention_dim: usize,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
}

impl UNetConfig {
    pub fn sd15() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![320, 640, 1280, 1280],
            layers_per_block: 2,
            attention_head_dim: 8,
            cross_attention_dim: 768,
            norm_num_groups: 32,
            norm_eps: 1e-5,
        }
    }

    /// Channel width of every skip the down pass pushes, in push order.
    ///
    /// Derived rather than hardcoded because each up resnet's input is
    /// `prev_channels + skip_channels`, and getting that wrong fails to load
    /// with a shape error several blocks away from the mistake.
    ///
    /// For SD 1.5 this is 12 entries: one for `conv_in`, then per down block
    /// one per resnet plus one for the downsampler where present.
    pub fn skip_channels(&self) -> Vec<usize> {
        let mut skips = vec![self.block_out_channels[0]];
        let last = self.block_out_channels.len().saturating_sub(1);
        for (i, &out_c) in self.block_out_channels.iter().enumerate() {
            for _ in 0..self.layers_per_block {
                skips.push(out_c);
            }
            if i != last {
                skips.push(out_c);
            }
        }
        skips
    }
}

/// The denoiser.
#[derive(Debug)]
pub struct UNet2DConditionModel {
    conv_in: Conv2d,
    time_embedding: TimestepEmbedding,
    down_blocks: Vec<DownBlock2D>,
    mid_block: MidBlock2DCrossAttn,
    up_blocks: Vec<UpBlock2D>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    /// `block_out_channels[0]`, the width of the raw sinusoid.
    freq_dim: usize,
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

impl UNet2DConditionModel {
    pub fn new(cfg: &UNetConfig, vb: VarBuilder) -> Result<Self> {
        let channels = &cfg.block_out_channels;
        let first = channels[0];
        let last = *channels.last().ok_or_else(|| {
            sd_tensor::Error::Msg("block_out_channels must not be empty".to_string())
        })?;
        let temb_channels = first * 4;
        let heads = cfg.attention_head_dim;

        let conv_in = conv3x3(cfg.in_channels, first, vb.pp("conv_in"))?;
        let time_embedding = TimestepEmbedding::new(first, temb_channels, vb.pp("time_embedding"))?;

        // -- down --------------------------------------------------------
        let vb_down = vb.pp("down_blocks");
        let mut down_blocks = Vec::with_capacity(channels.len());
        let mut prev = first;
        for (i, &out_c) in channels.iter().enumerate() {
            let is_final = i == channels.len() - 1;
            down_blocks.push(DownBlock2D::new(
                &BlockConfig {
                    in_channels: prev,
                    out_channels: out_c,
                    temb_channels,
                    num_layers: cfg.layers_per_block,
                    groups: cfg.norm_num_groups,
                    eps: cfg.norm_eps,
                    // The deepest down block has neither attention nor a
                    // downsampler.
                    attention: (!is_final).then_some((heads, cfg.cross_attention_dim)),
                    resample: !is_final,
                },
                vb_down.pp(i.to_string()),
            )?);
            prev = out_c;
        }

        let mid_block = MidBlock2DCrossAttn::new(
            last,
            temb_channels,
            heads,
            cfg.cross_attention_dim,
            cfg.norm_num_groups,
            cfg.norm_eps,
            vb.pp("mid_block"),
        )?;

        // -- up ----------------------------------------------------------
        //
        // Up blocks consume the skip stack from the end, `layers_per_block + 1`
        // at a time, so block `i` takes the last chunk that remains.
        let mut remaining = cfg.skip_channels();
        let up_layers = cfg.layers_per_block + 1;
        let reversed: Vec<usize> = channels.iter().copied().rev().collect();

        let vb_up = vb.pp("up_blocks");
        let mut up_blocks = Vec::with_capacity(reversed.len());
        let mut prev = last;
        for (i, &out_c) in reversed.iter().enumerate() {
            let split = remaining.len().saturating_sub(up_layers);
            let mut chunk = remaining.split_off(split);
            // Popped last-first, so the chunk is consumed in reverse.
            chunk.reverse();

            up_blocks.push(UpBlock2D::new(
                &BlockConfig {
                    in_channels: prev,
                    out_channels: out_c,
                    temb_channels,
                    num_layers: up_layers,
                    groups: cfg.norm_num_groups,
                    eps: cfg.norm_eps,
                    // The first up block has no attention; the last has no
                    // upsampler.
                    attention: (i != 0).then_some((heads, cfg.cross_attention_dim)),
                    resample: i != reversed.len() - 1,
                },
                &chunk,
                vb_up.pp(i.to_string()),
            )?);
            prev = out_c;
        }

        Ok(Self {
            conv_in,
            time_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out: group_norm(
                cfg.norm_num_groups,
                first,
                cfg.norm_eps,
                vb.pp("conv_norm_out"),
            )?,
            conv_out: conv3x3(first, cfg.out_channels, vb.pp("conv_out"))?,
            freq_dim: first,
        })
    }

    /// `sample`: `[b, 4, h, w]`, `timestep`: `[b]`, `context`: `[b, 77, 768]`.
    pub fn forward(&self, sample: &Tensor, timestep: &Tensor, context: &Tensor) -> Result<Tensor> {
        self.forward_with_skips(sample, timestep, context)
            .map(|(out, _, _)| out)
    }

    /// [`Self::forward`], also returning the skip stack and the mid-block
    /// output.
    ///
    /// For the golden test. With 25 blocks between input and output, a single
    /// final number cannot say whether the down pass or the up pass is at
    /// fault; the skips can.
    pub fn forward_with_skips(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
    ) -> Result<(Tensor, Vec<Tensor>, Tensor)> {
        // `timestep` is [b], not a scalar: a scalar yields [1, 1280] where
        // [b, 1280] is needed, and that only surfaces deep inside a resnet.
        let temb = timestep_embedding(timestep, self.freq_dim)?;
        let temb = self.time_embedding.forward(&temb)?;

        let mut h = self.conv_in.forward(sample)?;
        // conv_in's output is the first skip. Omitting it shifts the entire
        // stack by one, and the up blocks then concatenate the wrong tensor at
        // shapes that stay valid for several of them.
        let mut skips = vec![h.clone()];

        for block in &self.down_blocks {
            h = block.forward(&h, &temb, context, &mut skips)?;
        }

        let mid = self.mid_block.forward(&h, &temb, context)?;

        let captured = skips.clone();
        let mut h = mid.clone();
        for block in &self.up_blocks {
            h = block.forward(&h, &temb, context, &mut skips)?;
        }

        let h = self.conv_norm_out.forward(&h)?;
        let h = ops::silu(&h)?;
        let out = self.conv_out.forward(&h)?;
        Ok((out, captured, mid))
    }
}
