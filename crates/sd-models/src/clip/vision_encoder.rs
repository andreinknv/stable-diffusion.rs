//! CLIP's vision tower — the image half of the encoder.
//!
//! Structurally the text tower with a different front end: the transformer
//! stack, attention and MLP are literally [`ClipEncoderLayer`], reused. What
//! differs is how tokens are made and one thing that is easy to miss.
//!
//! # No causal mask
//!
//! Text attends causally; an image has no order to respect, so every patch
//! sees every other. The layer takes a mask parameter either way, so this
//! passes zeros — which is the additive identity for attention logits, not a
//! special case in the kernel.
//!
//! Reusing the text tower's mask here would be a quiet disaster: each patch
//! would see only those before it in raster order, the model would still emit
//! an embedding of the right shape, and the image conditioning would simply be
//! wrong.
//!
//! # Tokens are patches, plus one
//!
//! A stride-`patch` convolution cuts the image into non-overlapping patches —
//! 224/14 = 16 across, so 256 — and a learned **class embedding** is prepended
//! as token 0. That token is what the pooled output reads, so it must be first
//! and it must be prepended rather than appended.
//!
//! # `pre_layrnorm`
//!
//! Spelled that way in the checkpoint — a typo upstream that is now load-
//! bearing, since the tensor is named after it. Do not correct it.

use sd_tensor::nn::{
    conv2d_no_bias, layer_norm, linear_no_bias, Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig,
    Linear,
};
use sd_tensor::{DType, Device, Module, Result, Tensor, VarBuilder};

use super::text_encoder::ClipEncoderLayer;
use super::{ClipActivation, ClipTextConfig};
use crate::image::UnitImage;

/// Geometry of the vision tower.
#[derive(Debug, Clone)]
pub struct ClipVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub layer_norm_eps: f64,
    pub activation: ClipActivation,
    /// Width of `visual_projection`, when the checkpoint carries one.
    ///
    /// **1024 for ViT-H, where the tower itself is 1280.** IP-Adapter consumes
    /// the *projected* embedding, not the pooled hidden state, and the two are
    /// different widths — so getting this wrong fails to load rather than
    /// running wrong, which is the good direction.
    pub projection_dim: Option<usize>,
}

impl ClipVisionConfig {
    /// `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, which is the image encoder
    /// IP-Adapter ships for SD 1.5.
    pub fn vit_h_14() -> Self {
        Self {
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            image_size: 224,
            patch_size: 14,
            layer_norm_eps: 1e-5,
            // OpenCLIP, so plain gelu — the quick approximation is OpenAI's.
            activation: ClipActivation::Gelu,
            projection_dim: Some(1024),
        }
    }

    /// Patches across one edge; the sequence is this squared, plus the class
    /// token.
    pub fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }

    pub fn sequence_length(&self) -> usize {
        self.grid() * self.grid() + 1
    }

    /// The transformer stack is the text tower's, so it is configured with the
    /// text config. Only the fields the layers read are meaningful here.
    fn as_text_config(&self) -> ClipTextConfig {
        ClipTextConfig {
            vocab_size: 1,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            max_position_embeddings: self.sequence_length(),
            layer_norm_eps: self.layer_norm_eps,
            activation: self.activation,
            projection_dim: None,
        }
    }
}

/// The vision tower.
#[derive(Debug)]
pub struct ClipVisionEncoder {
    patch_embedding: Conv2d,
    class_embedding: Tensor,
    position_embedding: Tensor,
    pre_layernorm: LayerNorm,
    layers: Vec<ClipEncoderLayer>,
    post_layernorm: LayerNorm,
    visual_projection: Option<Linear>,
    /// Zeros, of the sequence's shape: attention with no masking.
    no_mask: Tensor,
    config: ClipVisionConfig,
    dtype: DType,
}

impl ClipVisionEncoder {
    pub fn new(cfg: &ClipVisionConfig, vb: VarBuilder) -> Result<Self> {
        let vb_model = vb.pp("vision_model");
        let vb_embed = vb_model.pp("embeddings");
        let seq = cfg.sequence_length();

        let patch_embedding = conv2d_no_bias(
            3,
            cfg.hidden_size,
            cfg.patch_size,
            Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb_embed.pp("patch_embedding"),
        )?;
        let class_embedding = vb_embed.get(cfg.hidden_size, "class_embedding")?;
        let position_embedding = vb_embed
            .pp("position_embedding")
            .get((seq, cfg.hidden_size), "weight")?;

        let norm_cfg = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        // `pre_layrnorm` — the checkpoint's spelling. See the module docs.
        let pre_layernorm = layer_norm(cfg.hidden_size, norm_cfg, vb_model.pp("pre_layrnorm"))?;
        let post_layernorm = layer_norm(cfg.hidden_size, norm_cfg, vb_model.pp("post_layernorm"))?;

        let text_cfg = cfg.as_text_config();
        let vb_layers = vb_model.pp("encoder").pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| ClipEncoderLayer::new(&text_cfg, vb_layers.pp(i.to_string())))
            .collect::<Result<Vec<_>>>()?;

        let visual_projection = cfg
            .projection_dim
            .map(|dim| linear_no_bias(cfg.hidden_size, dim, vb.pp("visual_projection")))
            .transpose()?;

        let dtype = vb.dtype();
        Ok(Self {
            patch_embedding,
            class_embedding,
            position_embedding,
            pre_layernorm,
            layers,
            post_layernorm,
            visual_projection,
            no_mask: Tensor::zeros((seq, seq), dtype, vb.device())?,
            config: cfg.clone(),
            dtype,
        })
    }

    pub fn config(&self) -> &ClipVisionConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        self.class_embedding.device()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Encode `[b, 3, image_size, image_size]` to `[b, seq, hidden]`.
    ///
    /// The image must already be resized and normalised with CLIP's own mean
    /// and standard deviation — see [`preprocess`]. Feeding it a `[-1, 1]`
    /// image, which is what the rest of this crate uses, produces an embedding
    /// of the right shape describing the wrong picture.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let xs = pixel_values.to_dtype(self.dtype)?;
        let (b, _, _, _) = xs.dims4()?;

        // [b, hidden, grid, grid] -> [b, grid*grid, hidden]
        let patches = self.patch_embedding.forward(&xs)?;
        let (_, hidden, gh, gw) = patches.dims4()?;
        let patches = patches
            .reshape((b, hidden, gh * gw))?
            .transpose(1, 2)?
            .contiguous()?;

        // The class token goes *first*: the pooled output reads position 0.
        let class_token = self
            .class_embedding
            .reshape((1, 1, hidden))?
            .broadcast_as((b, 1, hidden))?
            .contiguous()?;
        let xs = Tensor::cat(&[&class_token, &patches], 1)?;

        let xs = xs.broadcast_add(&self.position_embedding.unsqueeze(0)?)?;
        let mut xs = self.pre_layernorm.forward(&xs)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &self.no_mask)?;
        }
        Ok(xs)
    }

    /// The pooled image embedding: the class token, post-normed. `[b, hidden]`.
    ///
    /// This is what IP-Adapter's base model projects. The `plus` variants take
    /// the full patch sequence instead, which is why both are exposed.
    pub fn pooled(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let hidden = self.forward(pixel_values)?;
        let class_token = hidden.narrow(1, 0, 1)?.squeeze(1)?;
        self.post_layernorm.forward(&class_token)
    }
}

impl ClipVisionEncoder {
    /// The projected image embedding: `[b, projection_dim]`.
    ///
    /// **This is what IP-Adapter consumes**, not [`Self::pooled`]. The
    /// projection narrows ViT-H's 1280 to 1024, and the adapter's own
    /// projection expects that width — feeding it the pooled 1280 fails to
    /// load, which is the loud failure and is why this is a separate method
    /// rather than a flag.
    pub fn image_embeds(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let pooled = self.pooled(pixel_values)?;
        match &self.visual_projection {
            Some(proj) => proj.forward(&pooled),
            None => Err(sd_tensor::Error::Msg(
                "this vision encoder has no visual_projection".to_string(),
            )),
        }
    }
}

/// CLIP's channel means and standard deviations, in RGB order.
///
/// Not ImageNet's, though they are close enough that swapping them produces a
/// working but subtly degraded embedding rather than an error.
pub const CLIP_MEAN: [f32; 3] = [0.481_454_67, 0.457_827_5, 0.408_210_73];
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Normalise an image for the vision tower.
///
/// Takes [`UnitImage`] rather than a bare tensor: `[0, 1]`, not the `[-1, 1]`
/// used everywhere a VAE is involved. The two are the same shape and dtype, so
/// this used to be a comment — and a signed image handed here is *accepted*,
/// producing an embedding of the right shape describing the wrong picture.
/// Now the caller has to say which range it has.
pub fn preprocess(image: &UnitImage) -> Result<Tensor> {
    let image = image.tensor();
    let device = image.device();
    let mean = Tensor::from_slice(&CLIP_MEAN, (1, 3, 1, 1), device)?;
    let std = Tensor::from_slice(&CLIP_STD, (1, 3, 1, 1), device)?;
    image
        .to_dtype(DType::F32)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}
