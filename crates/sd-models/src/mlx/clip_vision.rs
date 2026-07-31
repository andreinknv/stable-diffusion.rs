//! CLIP's vision tower on MLX — the image half of the encoder.
//!
//! Structurally the text tower with a different front end: the transformer
//! stack is the same layer, reused. What differs is how tokens are made, and
//! three things that are easy to miss.
//!
//! # No causal mask
//!
//! Text attends causally; an image has no order to respect, so every patch sees
//! every other. On MLX this is `sdpa` rather than `sdpa_causal` — the text
//! tower's call reads almost identically and would be a quiet disaster: each
//! patch would see only those before it in raster order, and the model would
//! still emit an embedding of the right shape.
//!
//! # Tokens are patches, plus one
//!
//! A stride-`patch` convolution cuts the image into non-overlapping patches —
//! 224/14 = 16 across, so 256 — and a learned **class embedding** is prepended
//! as token 0. That token is what the pooled output reads, so it must be first
//! and prepended rather than appended.
//!
//! # `pre_layrnorm`
//!
//! Spelled that way in the checkpoint — a typo upstream that is now
//! load-bearing, since the tensor is named after it. Do not correct it.

use sd_tensor::mlx::{concat, Array, Stream};
use sd_tensor::{Error, Result};

use super::clip::{Activation, CLIP_EPS};
use super::{conv_strided, get, linear, Weights};

/// Geometry of the vision tower.
#[derive(Debug, Clone, Copy)]
pub struct VisionConfig {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub activation: Activation,
    /// True when the checkpoint carries a `visual_projection`.
    ///
    /// **1024 for ViT-H, where the tower itself is 1280.** IP-Adapter consumes
    /// the *projected* embedding, not the pooled hidden state, and the two are
    /// different widths.
    pub projection: bool,
}

impl VisionConfig {
    /// `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, the image encoder IP-Adapter
    /// ships for SD 1.5.
    pub fn vit_h_14() -> Self {
        Self {
            hidden: 1280,
            heads: 16,
            layers: 32,
            image_size: 224,
            patch_size: 14,
            // OpenCLIP, so plain gelu — the quick approximation is OpenAI's.
            activation: Activation::Gelu,
            projection: true,
        }
    }

    /// Patches across one edge.
    pub fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// The grid squared, plus the class token.
    pub fn sequence_length(&self) -> usize {
        self.grid() * self.grid() + 1
    }
}

/// One encoder layer. The text tower's, except that the attention is **not**
/// causal.
fn encoder_layer(
    x: &Array,
    cfg: &VisionConfig,
    w: &Weights,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let p = |name: &str| format!("{prefix}.{name}");
    let [n, seq, hidden] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: vision layer got {:?}", x.shape())));
    };
    let head_dim = hidden / cfg.heads;

    let y = x.layer_norm(
        Some(get(w, &p("layer_norm1.weight"))?),
        Some(get(w, &p("layer_norm1.bias"))?),
        CLIP_EPS,
        s,
    )?;
    let proj = |name: &str, src: &Array| -> Result<Array> {
        linear(
            src,
            get(w, &p(&format!("self_attn.{name}.weight")))?,
            w.get(&p(&format!("self_attn.{name}.bias"))),
            s,
        )
    };
    let split = |t: &Array| -> Result<Array> {
        t.reshape(&[n, seq, cfg.heads, head_dim], s)?
            .transpose(&[0, 2, 1, 3], s)
    };
    // `sdpa`, not `sdpa_causal`. See the module docs.
    let attended = split(&proj("q_proj", &y)?)?.sdpa(
        &split(&proj("k_proj", &y)?)?,
        &split(&proj("v_proj", &y)?)?,
        1.0 / (head_dim as f32).sqrt(),
        s,
    )?;
    let merged = attended
        .transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[n, seq, hidden], s)?;
    let x = x.add(&proj("out_proj", &merged)?, s)?;

    let y = x.layer_norm(
        Some(get(w, &p("layer_norm2.weight"))?),
        Some(get(w, &p("layer_norm2.bias"))?),
        CLIP_EPS,
        s,
    )?;
    let y = linear(
        &y,
        get(w, &p("mlp.fc1.weight"))?,
        w.get(&p("mlp.fc1.bias")),
        s,
    )?;
    let y = match cfg.activation {
        Activation::QuickGelu => y.quick_gelu(s)?,
        Activation::Gelu => y.gelu(s)?,
    };
    let y = linear(
        &y,
        get(w, &p("mlp.fc2.weight"))?,
        w.get(&p("mlp.fc2.bias")),
        s,
    )?;
    x.add(&y, s)
}

/// Patch embedding, class token, position embedding, `pre_layrnorm`, then the
/// stack. `[n, seq, hidden]`.
///
/// `pixels_nhwc` is `[n, image_size, image_size, 3]`, already normalised with
/// CLIP's own mean and standard deviation. Handing it a `[-1, 1]` image — what
/// the rest of this crate uses — produces an embedding of the right shape
/// describing the wrong picture.
pub fn hidden_states(
    pixels_nhwc: &Array,
    cfg: &VisionConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let [n, h, wd, _] = pixels_nhwc.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: pixels should be [n, h, w, 3], got {:?}",
            pixels_nhwc.shape()
        )));
    };
    if h != cfg.image_size || wd != cfg.image_size {
        return Err(Error::Msg(format!(
            "mlx: this tower wants {0}x{0}, got {h}x{wd}",
            cfg.image_size
        )));
    }

    // A stride-`patch` convolution with no padding and no bias: the patches
    // are non-overlapping by construction rather than by arithmetic.
    let patches = conv_strided(
        pixels_nhwc,
        get(w, "vision_model.embeddings.patch_embedding.weight")?,
        None,
        cfg.patch_size,
        0,
        s,
    )?;
    let grid = cfg.grid();
    // NHWC already has channels last, so flattening the grid needs no
    // transpose — the candle path permutes here and this does not.
    let patches = patches.reshape(&[n, grid * grid, cfg.hidden], s)?;

    // The class token goes *first*: the pooled output reads position 0.
    let class_token = get(w, "vision_model.embeddings.class_embedding")?
        .reshape(&[1, 1, cfg.hidden], s)?
        .broadcast_to(&[n, 1, cfg.hidden], s)?
        .contiguous(s)?;
    let x = concat(&[&class_token, &patches], 1, s)?;

    let positions = get(w, "vision_model.embeddings.position_embedding.weight")?;
    let x = x.add(
        &positions.reshape(&[1, cfg.sequence_length(), cfg.hidden], s)?,
        s,
    )?;

    // `pre_layrnorm`, the checkpoint's spelling.
    let mut x = x.layer_norm(
        Some(get(w, "vision_model.pre_layrnorm.weight")?),
        Some(get(w, "vision_model.pre_layrnorm.bias")?),
        CLIP_EPS,
        s,
    )?;
    for i in 0..cfg.layers {
        x = encoder_layer(&x, cfg, w, &format!("vision_model.encoder.layers.{i}"), s)?;
    }
    Ok(x)
}

/// The pooled image embedding: the class token, post-normed. `[n, hidden]`.
///
/// This is what IP-Adapter's base model projects. The `plus` variants take the
/// full patch sequence instead, which is why both are exposed.
pub fn pooled(pixels_nhwc: &Array, cfg: &VisionConfig, w: &Weights, s: &Stream) -> Result<Array> {
    pool(&hidden_states(pixels_nhwc, cfg, w, s)?, w, s)
}

/// [`pooled`] on a sequence this tower has already produced.
pub fn pool(hidden: &Array, w: &Weights, s: &Stream) -> Result<Array> {
    let [n, _, dim] = hidden.shape()[..] else {
        return Err(Error::Msg(format!(
            "mlx: vision hidden should be [n, seq, hidden], got {:?}",
            hidden.shape()
        )));
    };
    let class_token = hidden.narrow(1, 0, 1, s)?.reshape(&[n, dim], s)?;
    class_token.layer_norm(
        Some(get(w, "vision_model.post_layernorm.weight")?),
        Some(get(w, "vision_model.post_layernorm.bias")?),
        CLIP_EPS,
        s,
    )
}

/// The projected image embedding: `[n, projection_dim]`.
///
/// **This is what IP-Adapter consumes**, not [`pooled`]. The projection narrows
/// ViT-H's 1280 to 1024, and the adapter's own projection expects that width.
pub fn image_embeds(
    pixels_nhwc: &Array,
    cfg: &VisionConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    if !cfg.projection {
        return Err(Error::Msg(
            "mlx: this vision tower has no visual_projection".into(),
        ));
    }
    let pooled = pooled(pixels_nhwc, cfg, w, s)?;
    // `transformers` stores it without a bias.
    linear(&pooled, get(w, "visual_projection.weight")?, None, s)
}

/// CLIP's channel means and standard deviations, in RGB order.
///
/// Not ImageNet's, though they are close enough that swapping them produces a
/// working but subtly degraded embedding rather than an error.
pub const CLIP_MEAN: [f32; 3] = [0.481_454_67, 0.457_827_5, 0.408_210_73];
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Normalise a `[0, 1]` NHWC image for the vision tower.
///
/// **`[0, 1]`, not the `[-1, 1]` used everywhere a VAE is involved.** The two
/// are the same shape and dtype, so a signed image handed here is accepted and
/// produces an embedding of the right shape describing the wrong picture.
pub fn preprocess(unit_nhwc: &Array, s: &Stream) -> Result<Array> {
    let mean = Array::from_slice_f32(&CLIP_MEAN, &[1, 1, 1, 3])?;
    let std = Array::from_slice_f32(&CLIP_STD, &[1, 1, 1, 3])?;
    unit_nhwc.sub(&mean, s)?.div(&std, s)
}
