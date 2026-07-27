//! AnimateDiff motion modules: attention over *time*.
//!
//! A motion module is inserted after every resnet in the UNet — 21 of them for
//! SD 1.5 — and each is a small transformer whose attention runs across frames
//! rather than across pixels. That is the whole difference between "N images
//! that happen to share a seed" and "frames of one motion".
//!
//! # The reshape is the mechanism
//!
//! Everything else here is an ordinary transformer block, and our own
//! [`Attention`] and [`FeedForward`] are reused verbatim — the checkpoint's
//! names match theirs exactly. What makes it *temporal* is which axis becomes
//! the sequence:
//!
//! ```text
//!   [b*f, c, h, w]                       the UNet's ordinary activation
//!   -> [b*f, h*w, c]                     pixels as sequence  (spatial)
//!   -> [b*h*w, f, c]                     frames as sequence  (temporal)
//! ```
//!
//! Attention then mixes across `f`, so each pixel sees itself at every other
//! frame and at no other pixel. Getting this permute wrong leaves a module that
//! runs, has the right shapes throughout, and blurs across space instead of
//! time — which looks like a weak motion module rather than a broken one.
//!
//! # `num_frames` divides the batch
//!
//! The UNet has no frame axis; frames ride on the batch. So a motion module
//! needs to be told how many there are, and the count is ambient for the same
//! reason the IP-Adapter's weights are — it must reach 21 modules and is
//! uniform. [`with_frames`] sets it for a generation.
//!
//! With one frame, temporal attention over a sequence of length 1 is the
//! identity up to the projections, so a motion module left installed for a
//! still image is nearly — but not exactly — a no-op. It should not be.

use std::cell::Cell;

use sd_tensor::nn::{
    group_norm, layer_norm, linear, GroupNorm, LayerNorm, LayerNormConfig, Linear,
};
use sd_tensor::{Module, Result, Tensor, VarBuilder};

use super::attention::{Attention, FeedForward};

/// Positional encoding length the published adapters carry.
const MAX_FRAMES: usize = 32;
/// GroupNorm groups, per the adapter config.
const NORM_GROUPS: usize = 32;
/// Heads in every motion module, per the adapter config.
pub const HEADS: usize = 8;

thread_local! {
    /// Frames per batch entry. 1 means "a still image".
    static FRAMES: Cell<usize> = const { Cell::new(1) };
}

/// How many frames the batch currently holds.
pub fn frames() -> usize {
    FRAMES.with(Cell::get)
}

/// Restores the previous frame count when dropped.
#[must_use = "the frame count reverts when this guard is dropped"]
pub struct FramesGuard(usize);

impl Drop for FramesGuard {
    fn drop(&mut self) {
        FRAMES.with(|f| f.set(self.0));
    }
}

/// Declare the batch's frame count for the duration of a generation.
pub fn with_frames(n: usize) -> FramesGuard {
    FramesGuard(FRAMES.with(|f| f.replace(n.max(1))))
}

/// One temporal transformer block.
#[derive(Debug)]
struct MotionBlock {
    /// `[1, 32, c]`, a stored buffer rather than a computed sinusoid — the
    /// checkpoint ships it, so it is loaded rather than derived.
    pos_embed: Tensor,
    norm1: LayerNorm,
    attn1: Attention,
    norm2: LayerNorm,
    attn2: Attention,
    norm3: LayerNorm,
    ff: FeedForward,
}

impl MotionBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let norm_cfg = LayerNormConfig::default();
        Ok(Self {
            pos_embed: vb.pp("pos_embed").get((1, MAX_FRAMES, channels), "pe")?,
            norm1: layer_norm(channels, norm_cfg, vb.pp("norm1"))?,
            // Both attentions are *self*-attention over time: the adapter's
            // config sets `motion_cross_attention_dim: null`, so attn2 attends
            // over frames too rather than over the text.
            attn1: Attention::new(channels, None, HEADS, channels / HEADS, vb.pp("attn1"))?,
            norm2: layer_norm(channels, norm_cfg, vb.pp("norm2"))?,
            attn2: Attention::new(channels, None, HEADS, channels / HEADS, vb.pp("attn2"))?,
            norm3: layer_norm(channels, norm_cfg, vb.pp("norm3"))?,
            ff: FeedForward::new(channels, 4, vb.pp("ff"))?,
        })
    }

    /// `xs` is `[b*h*w, f, c]` — already temporal. Returns the same shape.
    ///
    /// The permute that makes it temporal happens once, in
    /// [`MotionModule::forward`], not here. An earlier version did it per
    /// block and was wrong by about 3 while keeping every shape valid.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let num_frames = xs.dim(1)?;
        let pe = self.pos_embed.narrow(1, 0, num_frames)?;
        let mut h = xs.clone();

        // The positional encoding goes on the **normed** states inside each
        // attention path — not once onto the residual stream — and it is
        // applied **twice**, before each attention. Adding it once to the
        // stream instead runs, keeps every shape, and is wrong by about 3.
        let normed = self.norm1.forward(&h)?.broadcast_add(&pe)?;
        h = (self.attn1.forward(&normed, None)? + &h)?;

        let normed = self.norm2.forward(&h)?.broadcast_add(&pe)?;
        h = (self.attn2.forward(&normed, None)? + &h)?;

        self.ff.forward(&self.norm3.forward(&h)?)? + &h
    }
}

/// A motion module: the temporal transformer plus its projections.
#[derive(Debug)]
pub struct MotionModule {
    norm: GroupNorm,
    proj_in: Linear,
    blocks: Vec<MotionBlock>,
    proj_out: Linear,
}

impl MotionModule {
    pub fn new(channels: usize, depth: usize, vb: VarBuilder) -> Result<Self> {
        let vb_blocks = vb.pp("transformer_blocks");
        Ok(Self {
            // 1e-6, not the UNet's 1e-5. The adapter's own GroupNorm uses it.
            norm: group_norm(NORM_GROUPS, channels, 1e-6, vb.pp("norm"))?,
            proj_in: linear(channels, channels, vb.pp("proj_in"))?,
            blocks: (0..depth)
                .map(|i| MotionBlock::new(channels, vb_blocks.pp(i.to_string())))
                .collect::<Result<Vec<_>>>()?,
            proj_out: linear(channels, channels, vb.pp("proj_out"))?,
        })
    }

    /// `xs` is `[b*f, c, h, w]`; returns the same shape.
    ///
    /// Residual around the whole module, so a zero-initialised `proj_out`
    /// would make it an exact identity — which is how these are trained
    /// against a frozen base, and why an untrained one does nothing.
    pub fn forward(&self, xs: &Tensor, num_frames: usize) -> Result<Tensor> {
        let (bf, c, height, width) = xs.dims4()?;
        let b = bf / num_frames;
        let residual = xs;

        // **The normalisation spans frames.** `[b*f, c, h, w]` is regrouped to
        // `[b, c, f, h, w]` first, so each group's statistics are taken over
        // the whole clip rather than one frame. Normalising per frame instead
        // runs, keeps every shape, and is wrong by about 3 — which is how this
        // was found.
        let grouped = xs
            .reshape((b, num_frames, c, height, width))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        let normed = self.norm.forward(&grouped)?;

        // -> [b*h*w, f, c]: pixels become the batch, frames the sequence.
        let h = normed.permute((0, 3, 4, 2, 1))?.contiguous()?.reshape((
            b * height * width,
            num_frames,
            c,
        ))?;

        let mut h = self.proj_in.forward(&h)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let h = self.proj_out.forward(&h)?;

        // Back: the flat buffer is `[b, h, w, f, c]` at this point, because
        // that is the order the temporal reshape left it in.
        let h = h
            .reshape((b, height, width, num_frames, c))?
            .permute((0, 3, 4, 1, 2))?
            .contiguous()?
            .reshape((bf, c, height, width))?;
        h + residual
    }
}
