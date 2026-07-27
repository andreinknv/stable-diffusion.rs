//! IP-Adapter: conditioning on a reference image.
//!
//! Three pieces, of which this file holds one. The [`crate::clip`] vision
//! tower turns an image into an embedding; [`ImageProjModel`] here turns that
//! embedding into a short run of tokens; and the UNet's cross-attention layers
//! attend over those tokens alongside the text — see [`crate::unet::ip`].
//!
//! # It takes the projected embedding
//!
//! `ClipVisionEncoder::image_embeds`, which is 1024 wide for ViT-H, **not**
//! `pooled`, which is 1280. The widths differ, so the wrong one fails to load.

use sd_tensor::nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear};
use sd_tensor::{Module, Result, Tensor, VarBuilder};

/// Image tokens the base SD 1.5 adapter produces. The `plus` variants use 16
/// and read the whole patch sequence rather than the pooled embedding.
pub const NUM_TOKENS: usize = 4;

/// Projects one image embedding into a short run of context tokens.
#[derive(Debug)]
pub struct ImageProjModel {
    proj: Linear,
    norm: LayerNorm,
    tokens: usize,
    cross_dim: usize,
}

impl ImageProjModel {
    /// `vb` should be rooted at `image_proj`.
    pub fn new(embed_dim: usize, cross_dim: usize, tokens: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            // One Linear producing every token at once, then reshaped — which
            // is why its output is `tokens * cross_dim` rather than `cross_dim`.
            proj: linear(embed_dim, tokens * cross_dim, vb.pp("proj"))?,
            norm: layer_norm(cross_dim, LayerNormConfig::default(), vb.pp("norm"))?,
            tokens,
            cross_dim,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// `[b, embed_dim]` -> `[b, tokens, cross_dim]`.
    pub fn forward(&self, image_embeds: &Tensor) -> Result<Tensor> {
        let b = image_embeds.dim(0)?;
        let projected =
            self.proj
                .forward(image_embeds)?
                .reshape((b, self.tokens, self.cross_dim))?;
        self.norm.forward(&projected)
    }
}
