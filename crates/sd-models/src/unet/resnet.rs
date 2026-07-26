//! The UNet's time-conditioned residual block.
//!
//! This is deliberately *not* the VAE's `ResnetBlock`, and the two must not be
//! merged: this one takes a timestep embedding, and it norms with `eps = 1e-5`
//! where the VAE uses `1e-6`.

use sd_tensor::nn::{conv2d, group_norm, linear, Conv2d, Conv2dConfig, GroupNorm, Linear};
use sd_tensor::{ops, Module, Result, Tensor, VarBuilder};

/// Residual block with timestep conditioning.
#[derive(Debug)]
pub struct ResnetBlock2D {
    norm1: GroupNorm,
    conv1: Conv2d,
    time_emb_proj: Linear,
    norm2: GroupNorm,
    conv2: Conv2d,
    /// Present only when `in_channels != out_channels`. Building it
    /// unconditionally makes weight loading fail on blocks that do not have
    /// one, with a "cannot find tensor" error that does not point here.
    conv_shortcut: Option<Conv2d>,
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

impl ResnetBlock2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        temb_channels: usize,
        groups: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_shortcut = if in_channels == out_channels {
            None
        } else {
            Some(conv2d(
                in_channels,
                out_channels,
                1,
                Conv2dConfig::default(),
                vb.pp("conv_shortcut"),
            )?)
        };

        Ok(Self {
            norm1: group_norm(groups, in_channels, eps, vb.pp("norm1"))?,
            conv1: conv3x3(in_channels, out_channels, vb.pp("conv1"))?,
            time_emb_proj: linear(temb_channels, out_channels, vb.pp("time_emb_proj"))?,
            norm2: group_norm(groups, out_channels, eps, vb.pp("norm2"))?,
            conv2: conv3x3(out_channels, out_channels, vb.pp("conv2"))?,
            conv_shortcut,
        })
    }

    /// `xs`: `[b, in_channels, h, w]`, `temb`: `[b, temb_channels]`.
    /// Returns `[b, out_channels, h, w]`.
    pub fn forward(&self, xs: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(xs)?;
        let h = ops::silu(&h)?;
        let h = self.conv1.forward(&h)?;

        // silu *before* the projection, and the result added *after* conv1 and
        // *before* norm2. Both orderings are easy to invert and both inversions
        // run cleanly.
        let t = ops::silu(temb)?;
        let t = self.time_emb_proj.forward(&t)?;
        // [b, out] -> [b, out, 1, 1] so it broadcasts over the spatial dims
        // rather than along the wrong axis.
        let t = t.unsqueeze(2)?.unsqueeze(3)?;
        let h = h.broadcast_add(&t)?;

        let h = self.norm2.forward(&h)?;
        let h = ops::silu(&h)?;
        let h = self.conv2.forward(&h)?;

        let shortcut = match &self.conv_shortcut {
            Some(conv) => conv.forward(xs)?,
            None => xs.clone(),
        };
        h + shortcut
    }
}
