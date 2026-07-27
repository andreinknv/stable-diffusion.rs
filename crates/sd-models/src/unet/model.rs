//! `UNet2DConditionModel` — the assembly.
//!
//! No new math here. The skip connections are the whole difficulty: the stack
//! must be pushed and popped in exactly the right order, and the up-block
//! resnets are wider than the config suggests because each one consumes a skip
//! alongside its input.

use sd_tensor::nn::{conv2d, group_norm, Conv2d, Conv2dConfig, GroupNorm};
use sd_tensor::{ops, DType, Module, Result, Tensor, VarBuilder};

use super::blocks::{AttentionSpec, BlockConfig, DownBlock2D, MidBlock2DCrossAttn, UpBlock2D};
use super::embeddings::{timestep_embedding, TimestepEmbedding};

/// Geometry of the denoiser.
#[derive(Debug, Clone)]
pub struct UNetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    /// **Head counts, despite the name.** Both configs call this
    /// `attention_head_dim` and diffusers reads it as a head count, so SD 1.5's
    /// 8 at 320 channels means 8 heads of 40 — not heads of width 8. One entry
    /// per block.
    pub attention_head_dim: Vec<usize>,
    /// Transformer blocks per attention module, one entry per block. All 1 in
    /// SD 1.5; SDXL uses [1, 2, 10].
    pub transformer_layers_per_block: Vec<usize>,
    /// Which down blocks carry cross-attention. Up blocks mirror this in
    /// reverse. SD 1.5 attends on all but the deepest; SDXL on all but the
    /// *shallowest*, which is the opposite end.
    pub down_block_has_attention: Vec<bool>,
    pub cross_attention_dim: usize,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
    /// SDXL projects the spatial transformer in and out with `Linear`;
    /// SD 1.5 uses 1x1 convolutions. The stored weights differ in rank, so
    /// getting it wrong fails to load rather than running wrong.
    pub use_linear_projection: bool,
    /// SDXL's micro-conditioning. `None` for SD 1.5, which has none.
    pub addition: Option<AdditionEmbedding>,
}

/// SDXL's extra conditioning: image size and crop offsets, sinusoidally
/// embedded and concatenated with the pooled text embedding.
#[derive(Debug, Clone, Copy)]
pub struct AdditionEmbedding {
    /// Width of the sinusoid applied to each of the six time ids.
    pub time_embed_dim: usize,
    /// Input width of `add_embedding`: `6 * time_embed_dim + pooled_dim`.
    pub projection_input_dim: usize,
}

impl UNetConfig {
    pub fn sd15() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![320, 640, 1280, 1280],
            layers_per_block: 2,
            attention_head_dim: vec![8; 4],
            transformer_layers_per_block: vec![1; 4],
            down_block_has_attention: vec![true, true, true, false],
            cross_attention_dim: 768,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            use_linear_projection: false,
            addition: None,
        }
    }

    /// SDXL base.
    ///
    /// Three blocks rather than four, attention on the *last two* rather than
    /// the first three, a much deeper transformer at the bottom, and the
    /// text_time micro-conditioning.
    pub fn sdxl() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![320, 640, 1280],
            layers_per_block: 2,
            // 320/5 = 640/10 = 1280/20 = 64 wide throughout.
            attention_head_dim: vec![5, 10, 20],
            transformer_layers_per_block: vec![1, 2, 10],
            down_block_has_attention: vec![false, true, true],
            cross_attention_dim: 2048,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            use_linear_projection: true,
            addition: Some(AdditionEmbedding {
                time_embed_dim: 256,
                // 6 ids * 256 = 1536, plus the 1280-wide pooled embedding.
                projection_input_dim: 2816,
            }),
        }
    }

    /// Attention spec for block `i`, or `None` when that block has none.
    ///
    /// Public because a ControlNet is a copy of this down stack and must be
    /// built from the same answer; deriving it twice is how the two drift.
    pub fn attention_spec(&self, i: usize) -> Option<AttentionSpec> {
        if !*self.down_block_has_attention.get(i)? {
            return None;
        }
        Some(AttentionSpec {
            heads: *self.attention_head_dim.get(i)?,
            cross_dim: self.cross_attention_dim,
            depth: *self.transformer_layers_per_block.get(i)?,
            linear_projection: self.use_linear_projection,
        })
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
    /// The weights' dtype. `timestep_embedding` computes in f32 for accuracy —
    /// the frequencies span several orders of magnitude — and the result is
    /// cast here. Without that, f16 weights meet an f32 embedding and the
    /// first linear fails on a dtype mismatch.
    dtype: DType,
    /// SDXL only. Projects the micro-conditioning into the time embedding.
    add_embedding: Option<(TimestepEmbedding, AdditionEmbedding)>,
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
                    attention: cfg.attention_spec(i),
                    // The deepest block never downsamples, whichever
                    // architecture this is.
                    resample: !is_final,
                },
                vb_down.pp(i.to_string()),
            )?);
            prev = out_c;
        }

        // The mid block always attends, at the deepest block's width and depth.
        let deepest = channels.len() - 1;
        let mid_block = MidBlock2DCrossAttn::new(
            last,
            temb_channels,
            AttentionSpec {
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
        let _ = deepest;

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
                    // Up blocks mirror the down pass, so block `i` here
                    // corresponds to down block `len - 1 - i`.
                    attention: cfg.attention_spec(reversed.len() - 1 - i),
                    // The last up block never upsamples.
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

    /// The dtype this UNet's weights are in.
    ///
    /// `sample` and `context` must arrive in it, and the output comes back in
    /// it. The pipeline keeps the sampler in f32 and casts at this boundary.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// `sample`: `[b, 4, h, w]`, `timestep`: `[b]`, `context`: `[b, 77, dim]`.
    pub fn forward(&self, sample: &Tensor, timestep: &Tensor, context: &Tensor) -> Result<Tensor> {
        self.forward_with_skips(sample, timestep, context, None)
            .map(|(out, _, _)| out)
    }

    /// [`Self::forward`], with a ControlNet's corrections applied.
    ///
    /// `down` is one correction per skip, in push order, and `mid` corrects the
    /// mid block. Deliberately typed as plain tensors rather than a ControlNet
    /// type: the UNet does not need to know what produced them, and keeping it
    /// ignorant is what lets a ControlNet be added without touching this
    /// model's definition.
    pub fn forward_controlled(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        down: &[Tensor],
        mid: &Tensor,
    ) -> Result<Tensor> {
        self.run(sample, timestep, context, None, Some((down, mid)))
            .map(|(out, _, _)| out)
    }

    /// SDXL's forward: the same, plus micro-conditioning.
    ///
    /// `pooled` is the second text encoder's projected embedding `[b, 1280]`,
    /// and `time_ids` is `[b, 6]` — original height and width, crop top and
    /// left, then target height and width. Both are required by an SDXL UNet
    /// and rejected by an SD 1.5 one, which has nowhere to put them.
    pub fn forward_sdxl(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        pooled: &Tensor,
        time_ids: &Tensor,
    ) -> Result<Tensor> {
        self.forward_with_skips(sample, timestep, context, Some((pooled, time_ids)))
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
        added: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Vec<Tensor>, Tensor)> {
        self.run(sample, timestep, context, added, None)
    }

    fn run(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        added: Option<(&Tensor, &Tensor)>,
        control: Option<(&[Tensor], &Tensor)>,
    ) -> Result<(Tensor, Vec<Tensor>, Tensor)> {
        // `timestep` is [b], not a scalar: a scalar yields [1, 1280] where
        // [b, 1280] is needed, and that only surfaces deep inside a resnet.
        let temb = timestep_embedding(timestep, self.freq_dim)?.to_dtype(self.dtype)?;
        let temb = self.time_embedding.forward(&temb)?;

        let temb = match (&self.add_embedding, added) {
            (Some((embed, cfg)), Some((pooled, time_ids))) => {
                // Each of the six ids gets its own sinusoid, then they are
                // flattened and concatenated with the pooled text embedding.
                let (b, n) = time_ids.dims2()?;
                let flat = time_ids.flatten_all()?;
                let sinusoid =
                    timestep_embedding(&flat, cfg.time_embed_dim)?.to_dtype(self.dtype)?;
                let sinusoid = sinusoid.reshape((b, n * cfg.time_embed_dim))?;
                // Pooled text embedding **first**, then the time sinusoid.
                // The halves are 1280 and 1536, so either order sums to 2816
                // and loads and runs — the reversed one just conditions on
                // nonsense.
                let combined = Tensor::cat(&[pooled, &sinusoid], 1)?;
                // Added to the timestep embedding, not concatenated with it.
                (temb + embed.forward(&combined)?)?
            }
            (Some(_), None) => {
                return Err(sd_tensor::Error::Msg(
                    "this UNet expects SDXL micro-conditioning; use forward_sdxl".to_string(),
                ))
            }
            (None, Some(_)) => {
                return Err(sd_tensor::Error::Msg(
                    "micro-conditioning supplied to a UNet that has no add_embedding".to_string(),
                ))
            }
            (None, None) => temb,
        };

        let mut h = self.conv_in.forward(sample)?;
        // conv_in's output is the first skip. Omitting it shifts the entire
        // stack by one, and the up blocks then concatenate the wrong tensor at
        // shapes that stay valid for several of them.
        let mut skips = vec![h.clone()];

        for block in &self.down_blocks {
            h = block.forward(&h, &temb, context, &mut skips)?;
        }

        let mid = self.mid_block.forward(&h, &temb, context)?;

        // ControlNet corrects what the up pass consumes, not what the down
        // pass computes: the down blocks have already run above, unchanged.
        let (skips, mid) = match control {
            Some((down, mid_res)) => {
                if down.len() != skips.len() {
                    return Err(sd_tensor::Error::Msg(format!(
                        "ControlNet gave {} corrections for {} skips",
                        down.len(),
                        skips.len()
                    )));
                }
                let corrected = skips
                    .iter()
                    .zip(down)
                    .map(|(s, c)| s + c)
                    .collect::<Result<Vec<_>>>()?;
                (corrected, (mid + mid_res)?)
            }
            None => (skips, mid),
        };
        let mut skips = skips;

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
