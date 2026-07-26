//! The spatial transformer that injects text conditioning into the UNet.
//!
//! Three `eps` values are in play across this file and its neighbours, and
//! they are genuinely inconsistent in the reference implementation:
//! [`Transformer2DModel`]'s GroupNorm is `1e-6`, the LayerNorms inside
//! [`BasicTransformerBlock`] are `1e-5`, and the resnets in `resnet.rs` are
//! `1e-5`. Unifying them would be tidier and wrong.

use sd_tensor::nn::{
    conv2d, group_norm, layer_norm, linear, linear_no_bias, Conv2d, Conv2dConfig, GroupNorm,
    LayerNorm, LayerNormConfig, Linear,
};
use sd_tensor::{ops, Module, Result, Tensor, VarBuilder, D};

/// LayerNorm epsilon inside a transformer block.
const BLOCK_EPS: f64 = 1e-5;
/// GroupNorm epsilon on the spatial wrapper. Not the same as `BLOCK_EPS`.
const SPATIAL_NORM_EPS: f64 = 1e-6;
const SPATIAL_NORM_GROUPS: usize = 32;

/// Multi-head attention, self or cross.
#[derive(Debug)]
pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
    dim_head: usize,
}

impl Attention {
    /// `cross_dim = None` is self-attention; `Some(768)` attends over text.
    pub fn new(
        query_dim: usize,
        cross_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = heads * dim_head;
        let kv_dim = cross_dim.unwrap_or(query_dim);
        Ok(Self {
            // q/k/v carry no bias; to_out does. Getting this wrong fails at
            // load with "cannot find tensor to_q.bias", which means exactly
            // what it says: that bias should not exist.
            to_q: linear_no_bias(query_dim, inner_dim, vb.pp("to_q"))?,
            to_k: linear_no_bias(kv_dim, inner_dim, vb.pp("to_k"))?,
            to_v: linear_no_bias(kv_dim, inner_dim, vb.pp("to_v"))?,
            to_out: linear(inner_dim, query_dim, vb.pp("to_out").pp("0"))?,
            heads,
            dim_head,
        })
    }

    /// `[b, s, inner]` -> `[b, heads, s, dim_head]`.
    fn split_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        xs.reshape((b, s, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// `context` is `None` for self-attention, else `[b, seq_kv, cross_dim]`.
    ///
    /// `seq_kv` need not equal `seq_q` — in SD 1.5 cross-attention it is 77
    /// against 256.
    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_q, _) = xs.dims3()?;
        let kv = context.unwrap_or(xs);

        let q = self.split_heads(&self.to_q.forward(xs)?)?;
        let k = self.split_heads(&self.to_k.forward(kv)?)?;
        let v = self.split_heads(&self.to_v.forward(kv)?)?;

        // Unmasked: the UNet's transformer attends over everything.
        let out = ops::scaled_dot_product_attention(&q, &k, &v)?;

        let inner_dim = self.heads * self.dim_head;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq_q, inner_dim))?;
        self.to_out.forward(&out)
    }
}

/// GEGLU feed-forward.
#[derive(Debug)]
pub struct FeedForward {
    proj: Linear,
    out: Linear,
    inner: usize,
}

impl FeedForward {
    /// `mult` is 4 for SD 1.5.
    pub fn new(dim: usize, mult: usize, vb: VarBuilder) -> Result<Self> {
        let inner = dim * mult;
        Ok(Self {
            // The projection emits twice `inner`: half value, half gate.
            proj: linear(dim, inner * 2, vb.pp("net").pp("0").pp("proj"))?,
            // `net.1` is dropout and has no parameters, hence the jump to 2.
            // The gap is real; renumbering breaks weight loading.
            out: linear(inner, dim, vb.pp("net").pp("2"))?,
            inner,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.proj.forward(xs)?;
        // Hidden is the first half, gate the second. Swapped, this produces
        // plausible garbage rather than an error.
        let hidden = h.narrow(D::Minus1, 0, self.inner)?;
        let gate = h.narrow(D::Minus1, self.inner, self.inner)?;
        // The erf gelu, not the tanh approximation.
        let h = (hidden * ops::gelu(&gate)?)?;
        self.out.forward(&h)
    }
}

/// Self-attention, cross-attention, feed-forward — each pre-normed with its
/// own residual.
#[derive(Debug)]
pub struct BasicTransformerBlock {
    norm1: LayerNorm,
    attn1: Attention,
    norm2: LayerNorm,
    attn2: Attention,
    norm3: LayerNorm,
    ff: FeedForward,
}

impl BasicTransformerBlock {
    pub fn new(
        dim: usize,
        heads: usize,
        dim_head: usize,
        cross_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm_cfg = LayerNormConfig {
            eps: BLOCK_EPS,
            ..Default::default()
        };
        Ok(Self {
            norm1: layer_norm(dim, norm_cfg, vb.pp("norm1"))?,
            attn1: Attention::new(dim, None, heads, dim_head, vb.pp("attn1"))?,
            norm2: layer_norm(dim, norm_cfg, vb.pp("norm2"))?,
            attn2: Attention::new(dim, Some(cross_dim), heads, dim_head, vb.pp("attn2"))?,
            norm3: layer_norm(dim, norm_cfg, vb.pp("norm3"))?,
            ff: FeedForward::new(dim, 4, vb.pp("ff"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor, context: &Tensor) -> Result<Tensor> {
        let xs = (self.attn1.forward(&self.norm1.forward(xs)?, None)? + xs)?;
        let xs = (self
            .attn2
            .forward(&self.norm2.forward(&xs)?, Some(context))?
            + &xs)?;
        let ys = self.ff.forward(&self.norm3.forward(&xs)?)?;
        ys + xs
    }
}

/// Spatial wrapper: `[b, c, h, w]` in and out, transformer in the middle.
#[derive(Debug)]
pub struct Transformer2DModel {
    norm: GroupNorm,
    proj_in: Conv2d,
    blocks: Vec<BasicTransformerBlock>,
    proj_out: Conv2d,
    inner: usize,
}

impl Transformer2DModel {
    pub fn new(
        channels: usize,
        heads: usize,
        dim_head: usize,
        depth: usize,
        cross_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner = heads * dim_head;
        let vb_blocks = vb.pp("transformer_blocks");
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            blocks.push(BasicTransformerBlock::new(
                inner,
                heads,
                dim_head,
                cross_dim,
                vb_blocks.pp(i.to_string()),
            )?);
        }

        Ok(Self {
            // 1e-6 here, unlike the 1e-5 LayerNorms inside the blocks.
            norm: group_norm(
                SPATIAL_NORM_GROUPS,
                channels,
                SPATIAL_NORM_EPS,
                vb.pp("norm"),
            )?,
            proj_in: conv2d(
                channels,
                inner,
                1,
                Conv2dConfig::default(),
                vb.pp("proj_in"),
            )?,
            blocks,
            proj_out: conv2d(
                inner,
                channels,
                1,
                Conv2dConfig::default(),
                vb.pp("proj_out"),
            )?,
            inner,
        })
    }

    /// `xs`: `[b, channels, h, w]`, `context`: `[b, 77, 768]`.
    pub fn forward(&self, xs: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (b, _, h, w) = xs.dims4()?;
        let residual = xs;

        let ys = self.norm.forward(xs)?;
        let ys = self.proj_in.forward(&ys)?;

        // permute *then* reshape, with contiguous between. A bare reshape from
        // [b, c, h, w] to [b, h*w, c] interleaves channels with spatial
        // positions: right shape, wrong numbers.
        let ys = ys
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((b, h * w, self.inner))?;

        let mut ys = ys;
        for block in &self.blocks {
            ys = block.forward(&ys, context)?;
        }

        let ys = ys
            .reshape((b, h, w, self.inner))?
            .permute((0, 3, 1, 2))?
            .contiguous()?;
        let ys = self.proj_out.forward(&ys)?;
        ys + residual
    }
}
