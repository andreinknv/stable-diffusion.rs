//! ControlNet — spatial conditioning for the UNet.
//!
//! A ControlNet is a **copy of the UNet's down and mid stack**, trained to read
//! a control image — Canny edges, a depth map, a pose skeleton — and emit one
//! correction per skip connection. The UNet then runs unchanged except that
//! each skip, and the mid output, has its correction added before the up pass
//! consumes it. Nothing about the up blocks, the text conditioning or the
//! sampler changes.
//!
//! That structure is why this module is short: [`DownBlock2D`],
//! [`MidBlock2DCrossAttn`] and [`TimestepEmbedding`] are the UNet's own, reused
//! verbatim. What is new is only the hint encoder and the zero convolutions.
//!
//! # The zero convolutions, and why an untrained ControlNet is invisible
//!
//! Every output passes through a 1x1 convolution initialised to zero
//! (`controlnet_down_blocks`, `controlnet_mid_block`). At the start of training
//! they emit zeros, so the corrections are zero and the UNet behaves exactly as
//! it did without them — which is what let ControlNet be trained on a frozen
//! base without destroying it in the first few steps. In a trained checkpoint
//! they are ordinary weights and this is only history, but it explains the
//! layer that otherwise looks redundant.
//!
//! # The hint encoder is not a VAE
//!
//! The control image enters at **full pixel resolution** and is reduced to
//! latent resolution by three stride-2 convolutions inside
//! [`ConditioningEmbedding`] — not by the VAE. Encoding the hint with the VAE
//! instead is a natural-looking mistake that produces a plausible image
//! ignoring the control entirely, because the numbers reaching `conv_in` are
//! then in the wrong space.

use sd_tensor::nn::{conv2d, Conv2d, Conv2dConfig};
use sd_tensor::{ops, DType, Module, Result, Tensor, VarBuilder};

use crate::unet::{
    timestep_embedding, AdditionEmbedding, BlockConfig, DownBlock2D, MidBlock2DCrossAttn,
    TimestepEmbedding, UNetConfig, UnetAttentionSpec,
};

/// Channel widths inside the hint encoder, before the projection to the UNet's
/// first block width. Every published ControlNet uses these.
const CONDITIONING_CHANNELS: [usize; 4] = [16, 32, 96, 256];

/// The corrections a ControlNet produces for one denoising step.
///
/// `down` has one entry per skip the UNet pushes — 12 for SD 1.5 — in push
/// order, and `mid` corrects the mid block's output.
#[derive(Debug, Clone)]
pub struct Control {
    pub down: Vec<Tensor>,
    pub mid: Tensor,
}

/// Maps a control image at pixel resolution to the UNet's first block width at
/// latent resolution.
///
/// Three stride-2 convolutions, so exactly the VAE's 8x reduction, reached a
/// different way. Interleaved with SiLU, and ending in the zero convolution.
#[derive(Debug)]
pub struct ConditioningEmbedding {
    conv_in: Conv2d,
    blocks: Vec<Conv2d>,
    conv_out: Conv2d,
}

fn conv3x3(in_c: usize, out_c: usize, stride: usize, vb: VarBuilder) -> Result<Conv2d> {
    conv2d(
        in_c,
        out_c,
        3,
        Conv2dConfig {
            padding: 1,
            stride,
            ..Default::default()
        },
        vb,
    )
}

impl ConditioningEmbedding {
    pub fn new(conditioning_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let first = CONDITIONING_CHANNELS[0];
        let conv_in = conv3x3(conditioning_channels, first, 1, vb.pp("conv_in"))?;

        // Pairs: one stride-1 conv at the current width, then one stride-2
        // conv that both widens and halves. Three pairs, so 8x in total.
        let vb_blocks = vb.pp("blocks");
        let mut blocks = Vec::with_capacity((CONDITIONING_CHANNELS.len() - 1) * 2);
        for i in 0..CONDITIONING_CHANNELS.len() - 1 {
            let c_in = CONDITIONING_CHANNELS[i];
            let c_out = CONDITIONING_CHANNELS[i + 1];
            blocks.push(conv3x3(
                c_in,
                c_in,
                1,
                vb_blocks.pp(blocks.len().to_string()),
            )?);
            blocks.push(conv3x3(
                c_in,
                c_out,
                2,
                vb_blocks.pp(blocks.len().to_string()),
            )?);
        }

        let last = CONDITIONING_CHANNELS[CONDITIONING_CHANNELS.len() - 1];
        Ok(Self {
            conv_in,
            blocks,
            conv_out: conv3x3(last, out_channels, 1, vb.pp("conv_out"))?,
        })
    }
}

impl Module for ConditioningEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = ops::silu(&self.conv_in.forward(xs)?)?;
        for block in &self.blocks {
            h = ops::silu(&block.forward(&h)?)?;
        }
        // No activation after the last convolution: it is the zero conv, and
        // its output is a correction rather than a feature map.
        self.conv_out.forward(&h)
    }
}

/// A ControlNet.
#[derive(Debug)]
pub struct ControlNet {
    conv_in: Conv2d,
    cond_embedding: ConditioningEmbedding,
    time_embedding: TimestepEmbedding,
    down_blocks: Vec<DownBlock2D>,
    mid_block: MidBlock2DCrossAttn,
    /// One 1x1 zero convolution per skip, in push order.
    down_projections: Vec<Conv2d>,
    mid_projection: Conv2d,
    freq_dim: usize,
    dtype: DType,
    /// SDXL only. A ControlNet copies the base's *down* stack, so it inherits
    /// the base's conditioning too — an SDXL ControlNet is
    /// `addition_embed_type: "text_time"` and is given the same pooled
    /// embedding and time ids the UNet gets. Its own `add_embedding` weights,
    /// not the UNet's.
    add_embedding: Option<(TimestepEmbedding, AdditionEmbedding)>,
}

impl ControlNet {
    /// Build from a UNet config — the same one the base model uses.
    ///
    /// A ControlNet is architecturally determined by its base: it must produce
    /// a correction of exactly the right shape for every skip, so its down
    /// stack has to match. Taking the base's config rather than a separate one
    /// makes a mismatch impossible to express.
    pub fn new(cfg: &UNetConfig, vb: VarBuilder) -> Result<Self> {
        let channels = &cfg.block_out_channels;
        let first = channels[0];
        let last = *channels.last().ok_or_else(|| {
            sd_tensor::Error::Msg("block_out_channels must not be empty".to_string())
        })?;
        let temb_channels = first * 4;

        let conv_in = conv3x3(cfg.in_channels, first, 1, vb.pp("conv_in"))?;
        let cond_embedding =
            ConditioningEmbedding::new(3, first, vb.pp("controlnet_cond_embedding"))?;
        let time_embedding = TimestepEmbedding::new(first, temb_channels, vb.pp("time_embedding"))?;

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
                    attention: cfg.attention_spec(i),
                    resample: !is_final,
                },
                vb_down.pp(i.to_string()),
            )?);
            prev = out_c;
        }

        let mid_block = MidBlock2DCrossAttn::new(
            last,
            temb_channels,
            UnetAttentionSpec {
                heads: *cfg.attention_head_dim.last().ok_or_else(|| {
                    sd_tensor::Error::Msg("attention_head_dim must not be empty".to_string())
                })?,
                cross_dim: cfg.cross_attention_dim,
                depth: *cfg.transformer_layers_per_block.last().ok_or_else(|| {
                    sd_tensor::Error::Msg(
                        "transformer_layers_per_block must not be empty".to_string(),
                    )
                })?,
                linear_projection: cfg.use_linear_projection,
            },
            cfg.norm_num_groups,
            cfg.norm_eps,
            vb.pp("mid_block"),
        )?;

        // One projection per skip, at that skip's width. `skip_channels` is the
        // same list the up blocks are sized from, so the two cannot disagree.
        let vb_proj = vb.pp("controlnet_down_blocks");
        let down_projections = cfg
            .skip_channels()
            .into_iter()
            .enumerate()
            .map(|(i, c)| conv2d(c, c, 1, Conv2dConfig::default(), vb_proj.pp(i.to_string())))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            conv_in,
            cond_embedding,
            time_embedding,
            down_blocks,
            mid_block,
            down_projections,
            mid_projection: conv2d(
                last,
                last,
                1,
                Conv2dConfig::default(),
                vb.pp("controlnet_mid_block"),
            )?,
            freq_dim: first,
            dtype: vb.dtype(),
            add_embedding: match cfg.addition {
                Some(add) => Some((
                    TimestepEmbedding::new(
                        add.projection_input_dim,
                        temb_channels,
                        vb.pp("add_embedding"),
                    )?,
                    add,
                )),
                None => None,
            },
        })
    }

    /// Whether this ControlNet expects SDXL's micro-conditioning.
    pub fn takes_micro_conditioning(&self) -> bool {
        self.add_embedding.is_some()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Run the control pass for one denoising step.
    ///
    /// `hint` is the control image as `[b, 3, h, w]` in `[0, 1]` at **pixel**
    /// resolution — not `[-1, 1]` as images are elsewhere in this crate, and
    /// not latent resolution. `scale` multiplies every correction; 1.0 is the
    /// published strength and 0.0 makes the run identical to no ControlNet.
    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        hint: &Tensor,
        scale: f64,
    ) -> Result<Control> {
        self.forward_with(sample, timestep, context, hint, scale, None)
    }

    /// [`Self::forward`], with SDXL's micro-conditioning.
    ///
    /// `added` is `(pooled, time_ids)`, exactly what
    /// `UNet2DConditionModel::forward_sdxl` takes — and it must be the *same*
    /// pair, because the corrections are added to a UNet that saw it.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_sdxl(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        hint: &Tensor,
        scale: f64,
        pooled: &Tensor,
        time_ids: &Tensor,
    ) -> Result<Control> {
        self.forward_with(
            sample,
            timestep,
            context,
            hint,
            scale,
            Some((pooled, time_ids)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        hint: &Tensor,
        scale: f64,
        added: Option<(&Tensor, &Tensor)>,
    ) -> Result<Control> {
        let temb = timestep_embedding(timestep, self.freq_dim)?.to_dtype(self.dtype)?;
        let temb = self.time_embedding.forward(&temb)?;

        // The same slot, and the same arithmetic, as the UNet's — see
        // `unet::model`. A ControlNet whose base has micro-conditioning must
        // receive it too, or its corrections are computed at a different
        // timestep embedding than the block they correct.
        let temb = match (&self.add_embedding, added) {
            (Some((embed, add_cfg)), Some((pooled, time_ids))) => {
                let (b, n) = time_ids.dims2()?;
                let flat = time_ids.flatten_all()?;
                let sinusoid =
                    timestep_embedding(&flat, add_cfg.time_embed_dim)?.to_dtype(self.dtype)?;
                let sinusoid = sinusoid.reshape((b, n * add_cfg.time_embed_dim))?;
                // Pooled first, then the sinusoid — both orders sum to the
                // same width and load, and the reversed one conditions on
                // nonsense.
                let combined = Tensor::cat(&[pooled, &sinusoid], 1)?;
                (temb + embed.forward(&combined)?)?
            }
            (Some(_), None) => {
                return Err(sd_tensor::Error::Msg(
                    "this ControlNet expects SDXL micro-conditioning; use forward_sdxl".to_string(),
                ))
            }
            (None, Some(_)) => {
                return Err(sd_tensor::Error::Msg(
                    "micro-conditioning supplied to a ControlNet that has no add_embedding"
                        .to_string(),
                ))
            }
            (None, None) => temb,
        };

        // The hint is added to conv_in's output, not concatenated: a
        // ControlNet's conv_in takes the same 4 latent channels the UNet's
        // does, and the hint arrives already projected to its width.
        let h = (self.conv_in.forward(sample)? + self.cond_embedding.forward(hint)?)?;

        let mut skips = vec![h.clone()];
        let mut h = h;
        for block in &self.down_blocks {
            h = block.forward(&h, &temb, context, &mut skips)?;
        }
        let mid = self.mid_block.forward(&h, &temb, context)?;

        if skips.len() != self.down_projections.len() {
            return Err(sd_tensor::Error::Msg(format!(
                "ControlNet produced {} skips for {} projections",
                skips.len(),
                self.down_projections.len()
            )));
        }
        let down = skips
            .iter()
            .zip(&self.down_projections)
            .map(|(s, p)| p.forward(s)? * scale)
            .collect::<Result<Vec<_>>>()?;

        Ok(Control {
            down,
            mid: (self.mid_projection.forward(&mid)? * scale)?,
        })
    }
}
