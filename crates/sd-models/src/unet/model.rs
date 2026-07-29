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
    /// unCLIP's image conditioning: the width of the vector `class_embedding`
    /// projects into the timestep embedding. `Some(2048)` for
    /// stable-diffusion-2-1-unclip, `None` for everything else here.
    ///
    /// diffusers spells this `class_embed_type: "projection"` plus
    /// `projection_class_embeddings_input_dim`; the type is an `Option` because
    /// "projection" is the only variant any checkpoint here uses, and a bool
    /// plus a width would let the two disagree.
    pub class_projection: Option<usize>,
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
            class_projection: None,
        }
    }

    /// InstructPix2Pix.
    ///
    /// SD 1.5 exactly, except that `conv_in` takes **8 channels**: the noisy
    /// latent concatenated with the *source image's* latent. That is the whole
    /// architectural difference — the edit instruction rides on the ordinary
    /// text conditioning, and everything downstream of `conv_in` is unchanged.
    ///
    /// The extra four channels are why this cannot be detected from the cross
    /// attention like SD 2.x is: `conv_in`'s own shape is the tell.
    pub fn instruct_pix2pix() -> Self {
        Self {
            in_channels: 8,
            ..Self::sd15()
        }
    }

    /// SD 2.x.
    ///
    /// SD 1.5's block geometry with a wider text encoder behind it: the cross
    /// attention takes 1024 rather than 768, the head counts are SDXL-style
    /// (`[5, 10, 20, 20]`, all 64 wide), and the spatial transformers project
    /// with `Linear` as SDXL's do rather than SD 1.5's 1x1 convolutions. That
    /// last one changes the stored weights' rank, so getting it wrong fails to
    /// load rather than running wrong.
    pub fn sd2() -> Self {
        Self {
            attention_head_dim: vec![5, 10, 20, 20],
            cross_attention_dim: 1024,
            use_linear_projection: true,
            ..Self::sd15()
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
            class_projection: None,
        }
    }

    /// unCLIP — `stabilityai/stable-diffusion-2-1-unclip` and its open mirrors.
    ///
    /// **SD 2.x exactly**, plus one thing: a `class_embedding` that projects a
    /// CLIP *image* embedding into the timestep embedding. Every block, every
    /// width and the text encoder behind it are unchanged, which is why this is
    /// two lines rather than an architecture.
    ///
    /// 2048 because the vector is the 1024-wide image embedding concatenated
    /// with a 1024-wide sinusoid of the noise level it was augmented at — see
    /// [`crate::unclip`]. Handing it the bare 1024 embedding fails to load,
    /// which is the good direction.
    pub fn unclip() -> Self {
        Self {
            class_projection: Some(2048),
            ..Self::sd2()
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
    /// unCLIP only. Projects the augmented image embedding into the same slot.
    ///
    /// A separate field from `add_embedding` rather than a shared one because
    /// they are separate tensors in the checkpoint (`class_embedding` against
    /// `add_embedding`) and separate arguments in diffusers. No checkpoint
    /// carries both, but collapsing them would make that a silent assumption
    /// rather than an observation.
    class_embedding: Option<TimestepEmbedding>,
    in_channels: usize,
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
    /// Build with an IP-Adapter's decoupled cross-attention attached.
    ///
    /// `ip_vb` should be rooted at `ip_adapter`. The weights are installed for
    /// the duration of this call and each cross-attention pulls its own as it
    /// is built — see [`crate::unet::ip`] for why that beats threading a
    /// parameter through every block type, and for the index mapping, which is
    /// *not* this UNet's construction order.
    ///
    /// Refuses if the entries and the layers do not match exactly. Consuming
    /// too few would leave the deepest layers unconditioned and still render.
    pub fn new_with_ip(
        cfg: &UNetConfig,
        vb: VarBuilder,
        ip_vb: VarBuilder,
        tokens: usize,
    ) -> Result<Self> {
        let source = super::ip::IpSource::new(ip_vb, super::ip::IpSource::sd15_order(), tokens);
        // Safety: the guard drops at the end of this scope, before `ip_vb`'s
        // borrow ends, so the erased lifetime never escapes.
        let guard = unsafe { super::ip::install(source) };
        let built = Self::new(cfg, vb);
        let complete = super::ip::fully_consumed();
        drop(guard);
        let model = built?;
        complete?;
        Ok(model)
    }

    /// Build with an AnimateDiff motion adapter attached.
    ///
    /// `motion_vb` is rooted at the adapter's top level. The modules are
    /// installed for the duration of this call and each block pulls its own —
    /// see [`crate::unet::motion`]. Refuses unless all 21 are consumed.
    ///
    /// The frame count is *not* set here: it belongs to a generation, not to a
    /// model. Wrap the run in [`crate::unet::motion::with_frames`].
    pub fn new_with_motion(
        cfg: &UNetConfig,
        vb: VarBuilder,
        motion_vb: VarBuilder,
    ) -> Result<Self> {
        let source =
            super::motion::MotionSource::new(motion_vb, super::motion::MotionSource::sd15_names());
        // Safety: the guard drops before `motion_vb`'s borrow ends.
        let guard = unsafe { super::motion::install(source) };
        let built = Self::new(cfg, vb);
        let complete = super::motion::fully_consumed();
        drop(guard);
        let model = built?;
        complete?;
        Ok(model)
    }

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
            in_channels: cfg.in_channels,
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
            class_embedding: match cfg.class_projection {
                Some(dim) => Some(TimestepEmbedding::new(
                    dim,
                    temb_channels,
                    vb.pp("class_embedding"),
                )?),
                None => None,
            },
        })
    }

    /// Input channels. 4 for every architecture here except InstructPix2Pix,
    /// which takes 8 — the noisy latent concatenated with the source image's.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// The dtype this UNet's weights are in.
    ///
    /// `sample` and `context` must arrive in it, and the output comes back in
    /// it. The pipeline keeps the sampler in f32 and casts at this boundary.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Whether this UNet conditions on a CLIP image embedding.
    ///
    /// Read from the weights it was built with, so a caller can tell an unCLIP
    /// checkpoint from an ordinary SD 2.x one without reopening the file.
    pub fn takes_class_labels(&self) -> bool {
        self.class_embedding.is_some()
    }

    /// The timestep embedding this UNet would use, without running it.
    ///
    /// `[b, block_out_channels[0] * 4]`. Two sinusoid-and-MLP calls, which is
    /// nothing beside a forward pass — that is the point. **Step caching needs
    /// to predict how much the output will change before paying for it**, and
    /// TeaCache's finding is that the timestep embedding's own movement is a
    /// far better predictor of that than the latent's.
    ///
    /// Note this depends on the timestep *alone*, not on the latent. For a
    /// DiT, where modulation multiplies into the token stream, the equivalent
    /// quantity is content-dependent; in a UNet the embedding is added to
    /// resnet activations and carries no content. So this is a schedule
    /// predictor, and whether that is enough is a question for measurement,
    /// not for the paper.
    pub fn timestep_features(&self, timestep: &Tensor) -> Result<Tensor> {
        let temb = timestep_embedding(timestep, self.freq_dim)?.to_dtype(self.dtype)?;
        self.time_embedding.forward(&temb)
    }

    /// `sample`: `[b, 4, h, w]`, `timestep`: `[b]`, `context`: `[b, 77, dim]`.
    pub fn forward(&self, sample: &Tensor, timestep: &Tensor, context: &Tensor) -> Result<Tensor> {
        self.forward_with_skips(sample, timestep, context, None)
            .map(|(out, _, _)| out)
    }

    /// unCLIP's forward: the same, plus the augmented image embedding.
    ///
    /// `class_labels` is `[b, 2048]` — the CLIP image embedding, noised at a
    /// chosen level, with that level's sinusoid appended. Build it with
    /// [`crate::unclip::NoiseAugmentor::augment`] rather than by hand: the
    /// halves are the same width, so a swapped concatenation runs.
    ///
    /// The unconditional row of a guidance batch is **zeros of this shape**,
    /// not an absent argument — an unCLIP UNet has nowhere to put "no image".
    pub fn forward_unclip(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        class_labels: &Tensor,
    ) -> Result<Tensor> {
        self.run(sample, timestep, context, None, None, Some(class_labels))
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
        self.run(sample, timestep, context, None, Some((down, mid)), None)
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
        self.run(sample, timestep, context, added, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        added: Option<(&Tensor, &Tensor)>,
        control: Option<(&[Tensor], &Tensor)>,
        class_labels: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>, Tensor)> {
        // `timestep` is [b], not a scalar: a scalar yields [1, 1280] where
        // [b, 1280] is needed, and that only surfaces deep inside a resnet.
        let temb = timestep_embedding(timestep, self.freq_dim)?.to_dtype(self.dtype)?;
        let temb = self.time_embedding.forward(&temb)?;

        // unCLIP's image conditioning lands in the *same slot* as SDXL's
        // micro-conditioning — added to the timestep embedding, before the
        // blocks see it — so nothing downstream of here knows it exists. That
        // is also why getting it wrong is silent: every block runs unchanged
        // on a temb that means something else.
        let temb = match (&self.class_embedding, class_labels) {
            (Some(embed), Some(labels)) => {
                (temb + embed.forward(&labels.to_dtype(self.dtype)?)?)?
            }
            (Some(_), None) => {
                return Err(sd_tensor::Error::Msg(
                    "this UNet expects an image embedding; use forward_unclip".to_string(),
                ))
            }
            (None, Some(_)) => {
                return Err(sd_tensor::Error::Msg(
                    "class labels supplied to a UNet that has no class_embedding".to_string(),
                ))
            }
            (None, None) => temb,
        };

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
