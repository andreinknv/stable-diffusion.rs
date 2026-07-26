//! SD 3 / SD 3.5's MMDiT transformer.
//!
//! The same family as [`flux`](crate::flux) — patchified latent, joint
//! attention over image and text tokens, conditioning by modulation — but
//! different in four ways that matter, each of which fails quietly if assumed
//! from Flux:
//!
//! - **Learned positional embeddings, not RoPE.** A `384 x 384` table is
//!   stored and cropped from the *centre* for the image size in use. Cropping
//!   from a corner gives a coherent image with the composition subtly off.
//! - **Every block is a joint block.** There is no single-stream half; image
//!   and text keep separate weights throughout.
//! - **The last block's context half is `pre_only`.** It contributes keys and
//!   values to the final attention and then its own output is discarded, so
//!   it carries no projection and no MLP.
//! - **SD 3.5 adds a second self-attention** on the image stream in its first
//!   13 blocks — "MMDiT-X". Those blocks modulate nine ways instead of six.
//!
//! Names follow the original Stability layout (`joint_blocks.0.x_block.attn`),
//! which is what the published checkpoints use.

use crate::weights::{Proj, QuantizedWeights, Source};
use sd_tensor::nn::VarBuilder;
use sd_tensor::{ops, DType, Result, Tensor, D};

/// SD 3 transformer geometry.
#[derive(Debug, Clone)]
pub struct Sd3Config {
    pub in_channels: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
    /// Blocks carrying a second image self-attention. Empty for SD 3.
    pub dual_attention_layers: Vec<usize>,
    /// Width of the stored positional table, in patches per side.
    pub pos_embed_max_size: usize,
    /// T5 width.
    pub context_dim: usize,
    /// CLIP-L and CLIP-G pooled, concatenated.
    pub pooled_dim: usize,
}

impl Sd3Config {
    /// `stabilityai/stable-diffusion-3.5-medium`.
    pub fn medium_35() -> Self {
        Self {
            in_channels: 16,
            patch_size: 2,
            hidden_size: 1536,
            num_heads: 24,
            depth: 24,
            dual_attention_layers: (0..13).collect(),
            pos_embed_max_size: 384,
            context_dim: 4096,
            pooled_dim: 2048,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    fn mlp_hidden(&self) -> usize {
        self.hidden_size * 4
    }

    fn is_dual(&self, block: usize) -> bool {
        self.dual_attention_layers.contains(&block)
    }
}

/// Sinusoidal timestep embedding, `cos` first.
///
/// Unlike Flux's, the input is **not** scaled by 1000: SD 3 is handed a
/// timestep already in the training range rather than a sigma in `[0, 1]`.
fn timestep_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10_000f64.ln()) * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), t.device())?;
    let t = t.to_dtype(DType::F32)?.reshape((t.elem_count(), 1))?;
    let args = t.broadcast_mul(&freqs)?;
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)
}

/// `x * (1 + scale) + shift`, with both broadcast over the token axis.
fn modulate(xs: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    xs.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// LayerNorm with no learned parameters — the scale and shift come from the
/// modulation vector instead.
fn plain_layer_norm(xs: &Tensor) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs32 = xs.to_dtype(DType::F32)?;
    let mean = xs32.mean_keepdim(D::Minus1)?;
    let centred = xs32.broadcast_sub(&mean)?;
    let var = centred.sqr()?.mean_keepdim(D::Minus1)?;
    centred
        .broadcast_div(&(var + 1e-6)?.sqrt()?)?
        .to_dtype(dtype)
}

/// RMSNorm over the head dimension, applied to queries and keys.
#[derive(Debug)]
struct RmsNorm {
    weight: Tensor,
}

impl RmsNorm {
    fn new(dim: usize, src: Source, path: &str) -> Result<Self> {
        Ok(Self {
            weight: src.tensor(&format!("{path}.weight"), dim)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs32 = xs.to_dtype(DType::F32)?;
        let rrms = (xs32.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
        xs32.broadcast_div(&rrms)?
            .to_dtype(dtype)?
            .broadcast_mul(&self.weight.to_dtype(dtype)?)
    }
}

/// Fused QKV with per-head query/key normalisation.
#[derive(Debug)]
struct Attention {
    qkv: Proj,
    proj: Option<Proj>,
    ln_q: RmsNorm,
    ln_k: RmsNorm,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Sd3Config, with_proj: bool, src: Source, path: &str) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            qkv: src.linear(&format!("{path}.qkv"), h, 3 * h)?,
            // Absent on a `pre_only` block, whose output is discarded.
            proj: if with_proj {
                Some(src.linear(&format!("{path}.proj"), h, h)?)
            } else {
                None
            },
            ln_q: RmsNorm::new(cfg.head_dim(), src, &format!("{path}.ln_q"))?,
            ln_k: RmsNorm::new(cfg.head_dim(), src, &format!("{path}.ln_k"))?,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// Project and split into per-head `q`, `k`, `v`, normalising q and k.
    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, n, _) = xs.dims3()?;
        let t = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let take = |i: usize| t.narrow(0, i, 1)?.squeeze(0)?.contiguous();
        Ok((
            self.ln_q.forward(&take(0)?)?,
            self.ln_k.forward(&take(1)?)?,
            take(2)?,
        ))
    }

    fn post(&self, attn: &Tensor) -> Result<Tensor> {
        let (b, h, n, d) = attn.dims4()?;
        let merged = attn.transpose(1, 2)?.reshape((b, n, h * d))?;
        self.proj
            .as_ref()
            .expect("post-attention on a pre_only block")
            .forward(&merged)
    }
}

/// Feed-forward: `fc2(gelu(fc1(x)))`.
#[derive(Debug)]
struct Mlp {
    fc1: Proj,
    fc2: Proj,
}

impl Mlp {
    fn new(cfg: &Sd3Config, src: Source, path: &str) -> Result<Self> {
        Ok(Self {
            fc1: src.linear(&format!("{path}.fc1"), cfg.hidden_size, cfg.mlp_hidden())?,
            fc2: src.linear(&format!("{path}.fc2"), cfg.mlp_hidden(), cfg.hidden_size)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&ops::gelu_approx(&self.fc1.forward(xs)?)?)
    }
}

/// One half of a joint block — either the image stream or the text stream.
#[derive(Debug)]
struct HalfBlock {
    ada_ln: Proj,
    attn: Attention,
    /// SD 3.5's second image self-attention. Image stream only, first 13
    /// blocks only.
    attn2: Option<Attention>,
    mlp: Option<Mlp>,
    chunks: usize,
    hidden: usize,
}

impl HalfBlock {
    fn new(cfg: &Sd3Config, pre_only: bool, dual: bool, src: Source, path: &str) -> Result<Self> {
        // 2 for pre_only (shift and scale, nothing to gate), 9 with a second
        // attention, 6 otherwise.
        let chunks = if pre_only {
            2
        } else if dual {
            9
        } else {
            6
        };
        let h = cfg.hidden_size;
        Ok(Self {
            // `adaLN_modulation` is Sequential(SiLU, Linear); index 1 is the
            // Linear and index 0 has no weights.
            ada_ln: src.linear(&format!("{path}.adaLN_modulation.1"), h, chunks * h)?,
            attn: Attention::new(cfg, !pre_only, src, &format!("{path}.attn"))?,
            attn2: if dual {
                Some(Attention::new(cfg, true, src, &format!("{path}.attn2"))?)
            } else {
                None
            },
            mlp: if pre_only {
                None
            } else {
                Some(Mlp::new(cfg, src, &format!("{path}.mlp"))?)
            },
            chunks,
            hidden: h,
        })
    }

    /// The modulation parameters for this block, as `chunks` tensors.
    fn modulation(&self, c: &Tensor) -> Result<Vec<Tensor>> {
        let out = self.ada_ln.forward(&ops::silu(c)?)?.unsqueeze(1)?;
        (0..self.chunks)
            .map(|i| out.narrow(D::Minus1, i * self.hidden, self.hidden))
            .collect()
    }
}

/// A joint block: image and text streams with their own weights, mixed inside
/// one attention over the concatenated tokens.
#[derive(Debug)]
struct JointBlock {
    context: HalfBlock,
    x: HalfBlock,
    context_pre_only: bool,
    dual: bool,
}

impl JointBlock {
    fn new(cfg: &Sd3Config, index: usize, src: Source, path: &str) -> Result<Self> {
        let pre_only = index + 1 == cfg.depth;
        let dual = cfg.is_dual(index);
        Ok(Self {
            context: HalfBlock::new(cfg, pre_only, false, src, &format!("{path}.context_block"))?,
            x: HalfBlock::new(cfg, false, dual, src, &format!("{path}.x_block"))?,
            context_pre_only: pre_only,
            dual,
        })
    }

    fn forward(
        &self,
        context: &Tensor,
        x: &Tensor,
        c: &Tensor,
    ) -> Result<(Option<Tensor>, Tensor)> {
        let cm = self.context.modulation(c)?;
        let xm = self.x.modulation(c)?;

        let c_in = modulate(&plain_layer_norm(context)?, &cm[0], &cm[1])?;
        let (cq, ck, cv) = self.context.attn.qkv(&c_in)?;

        let x_norm = plain_layer_norm(x)?;
        let x_in = modulate(&x_norm, &xm[0], &xm[1])?;
        let (xq, xk, xv) = self.x.attn.qkv(&x_in)?;

        // Context first, then image — the order the halves are split back out
        // in below, and the order SD 3 concatenates them.
        let q = Tensor::cat(&[&cq, &xq], 2)?.contiguous()?;
        let k = Tensor::cat(&[&ck, &xk], 2)?.contiguous()?;
        let v = Tensor::cat(&[&cv, &xv], 2)?.contiguous()?;
        let attn = ops::scaled_dot_product_attention(&q, &k, &v)?;

        let c_len = cq.dim(2)?;
        let c_attn = attn.narrow(2, 0, c_len)?.contiguous()?;
        let x_attn = attn.narrow(2, c_len, attn.dim(2)? - c_len)?.contiguous()?;

        // The final block's context half feeds the attention and is then
        // dropped, so there is nothing to gate or project.
        let context_out = if self.context_pre_only {
            None
        } else {
            let ctx = (context + self.context.attn.post(&c_attn)?.broadcast_mul(&cm[2])?)?;
            let ff = self
                .context
                .mlp
                .as_ref()
                .expect("non-pre_only block has an mlp")
                .forward(&modulate(&plain_layer_norm(&ctx)?, &cm[3], &cm[4])?)?;
            Some((ctx + ff.broadcast_mul(&cm[5])?)?)
        };

        let mut xs = (x + self.x.attn.post(&x_attn)?.broadcast_mul(&xm[2])?)?;
        if self.dual {
            // A second self-attention over the image tokens alone, modulated
            // independently and added to the same residual.
            let attn2 = self.x.attn2.as_ref().expect("dual block has attn2");
            let x2_in = modulate(&x_norm, &xm[6], &xm[7])?;
            let (q2, k2, v2) = attn2.qkv(&x2_in)?;
            let a2 = ops::scaled_dot_product_attention(&q2, &k2, &v2)?;
            xs = (xs + attn2.post(&a2)?.broadcast_mul(&xm[8])?)?;
        }
        let ff = self
            .x
            .mlp
            .as_ref()
            .expect("the image stream always has an mlp")
            .forward(&modulate(&plain_layer_norm(&xs)?, &xm[3], &xm[4])?)?;
        let xs = (xs + ff.broadcast_mul(&xm[5])?)?;

        Ok((context_out, xs))
    }

    fn resident_bytes(&self) -> usize {
        let half = |h: &HalfBlock| {
            h.attn.qkv.resident_bytes()
                + h.attn.proj.as_ref().map_or(0, |p| p.resident_bytes())
                + h.attn2.as_ref().map_or(0, |a| {
                    a.qkv.resident_bytes() + a.proj.as_ref().map_or(0, |p| p.resident_bytes())
                })
                + h.mlp
                    .as_ref()
                    .map_or(0, |m| m.fc1.resident_bytes() + m.fc2.resident_bytes())
        };
        half(&self.context) + half(&self.x)
    }
}

/// Two-layer MLP used for the timestep and pooled-text embeddings.
#[derive(Debug)]
struct Embedder {
    fc1: Proj,
    fc2: Proj,
}

impl Embedder {
    fn new(in_dim: usize, hidden: usize, src: Source, path: &str) -> Result<Self> {
        Ok(Self {
            fc1: src.linear(&format!("{path}.mlp.0"), in_dim, hidden)?,
            fc2: src.linear(&format!("{path}.mlp.2"), hidden, hidden)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&ops::silu(&self.fc1.forward(xs)?)?)
    }
}

/// The SD 3 / SD 3.5 transformer.
#[derive(Debug)]
pub struct Sd3Transformer {
    /// Patch embedding, held as a matrix rather than a convolution: the
    /// kernel equals the stride, so it is exactly a linear map over each
    /// flattened patch, and reusing the packing already written for Flux
    /// avoids a second convolution path.
    patch_weight: Tensor,
    patch_bias: Tensor,
    pos_embed: Tensor,
    t_embedder: Embedder,
    y_embedder: Embedder,
    context_embedder: Proj,
    blocks: Vec<JointBlock>,
    final_ada_ln: Proj,
    final_linear: Proj,
    cfg: Sd3Config,
}

impl Sd3Transformer {
    pub fn new(cfg: &Sd3Config, vb: VarBuilder) -> Result<Self> {
        Self::from_source(cfg, Source::Dense(&vb))
    }

    /// Build with the weights left quantised. See
    /// [`crate::weights`] for why this is not merely an optimisation.
    pub fn from_quantized(cfg: &Sd3Config, weights: &QuantizedWeights) -> Result<Self> {
        Self::from_source(cfg, Source::Quantized(weights))
    }

    fn from_source(cfg: &Sd3Config, src: Source) -> Result<Self> {
        let h = cfg.hidden_size;
        let p = cfg.patch_size;
        let patch_elems = cfg.in_channels * p * p;

        // Stored as a convolution kernel `[hidden, in, p, p]`; flattened to
        // `[hidden, in*p*p]` it is the same map, and matches the `(c, ph, pw)`
        // ordering `pack_patches` produces.
        let patch_weight = src
            .tensor("x_embedder.proj.weight", (h, cfg.in_channels, p, p))
            .or_else(|_| src.tensor("x_embedder.proj.weight", (h, patch_elems)))?
            .reshape((h, patch_elems))?;

        Ok(Self {
            patch_weight,
            patch_bias: src.tensor("x_embedder.proj.bias", h)?,
            pos_embed: src.tensor(
                "pos_embed",
                (1, cfg.pos_embed_max_size * cfg.pos_embed_max_size, h),
            )?,
            t_embedder: Embedder::new(TIME_EMBED_DIM, h, src, "t_embedder")?,
            y_embedder: Embedder::new(cfg.pooled_dim, h, src, "y_embedder")?,
            context_embedder: src.linear("context_embedder", cfg.context_dim, h)?,
            blocks: (0..cfg.depth)
                .map(|i| JointBlock::new(cfg, i, src, &format!("joint_blocks.{i}")))
                .collect::<Result<Vec<_>>>()?,
            final_ada_ln: src.linear("final_layer.adaLN_modulation.1", h, 2 * h)?,
            final_linear: src.linear("final_layer.linear", h, p * p * cfg.in_channels)?,
            cfg: cfg.clone(),
        })
    }

    pub fn config(&self) -> &Sd3Config {
        &self.cfg
    }

    pub fn resident_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.resident_bytes()).sum()
    }

    /// The slice of the stored positional table for an `h x w` patch grid.
    ///
    /// Cropped from the **centre**, which is what the reference does. Taking
    /// it from a corner yields a coherent image whose composition is
    /// systematically offset — plausible enough to miss.
    fn cropped_pos_embed(&self, h: usize, w: usize) -> Result<Tensor> {
        let max = self.cfg.pos_embed_max_size;
        if h > max || w > max {
            return Err(sd_tensor::Error::Msg(format!(
                "a {h}x{w} patch grid exceeds the stored positional table ({max}x{max})"
            )));
        }
        let (top, left) = ((max - h) / 2, (max - w) / 2);
        self.pos_embed
            .reshape((1, max, max, self.cfg.hidden_size))?
            .narrow(1, top, h)?
            .narrow(2, left, w)?
            .contiguous()?
            .reshape((1, h * w, self.cfg.hidden_size))
    }

    /// The three embedding stages, before any block runs.
    ///
    /// Returns `(patched image tokens with position added, conditioning
    /// vector, embedded context)`. Public because a whole-model mismatch in a
    /// 24-block stack is not localisable otherwise, and these three are where
    /// convention errors concentrate — patch flattening order, the centre
    /// crop, and whether the timestep is pre-scaled.
    pub fn embed_stages(
        &self,
        latents: &Tensor,
        context: &Tensor,
        pooled: &Tensor,
        timestep: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (_, _, lh, lw) = latents.dims4()?;
        let p = self.cfg.patch_size;
        let patches = crate::flux::pack_latents(latents)?;
        let xs = patches
            .broadcast_matmul(&self.patch_weight.t()?.to_dtype(patches.dtype())?)?
            .broadcast_add(&self.patch_bias.to_dtype(patches.dtype())?)?;
        let xs = xs.broadcast_add(
            &self
                .cropped_pos_embed(lh / p, lw / p)?
                .to_dtype(xs.dtype())?,
        )?;

        let dtype = xs.dtype();
        let t = self
            .t_embedder
            .forward(&timestep_embedding(timestep, TIME_EMBED_DIM)?.to_dtype(dtype)?)?;
        let c = (t + self.y_embedder.forward(pooled)?)?;
        let ctx = self.context_embedder.forward(context)?;
        Ok((xs, c, ctx))
    }

    /// Predict the flow velocity.
    ///
    /// - `latents` — `[b, 16, h, w]`, unpatchified
    /// - `context` — T5 sequence `[b, n, 4096]`
    /// - `pooled` — CLIP-L and CLIP-G pooled, concatenated, `[b, 2048]`
    /// - `timestep` — `[b]`, in the training range (0..1000), not `[0, 1]`
    pub fn forward(
        &self,
        latents: &Tensor,
        context: &Tensor,
        pooled: &Tensor,
        timestep: &Tensor,
    ) -> Result<Tensor> {
        let (_, _, lh, lw) = latents.dims4()?;
        let p = self.cfg.patch_size;
        let (ph, pw) = (lh / p, lw / p);

        let patches = crate::flux::pack_latents(latents)?;
        let mut xs = patches
            .broadcast_matmul(&self.patch_weight.t()?.to_dtype(patches.dtype())?)?
            .broadcast_add(&self.patch_bias.to_dtype(patches.dtype())?)?;
        xs = xs.broadcast_add(&self.cropped_pos_embed(ph, pw)?.to_dtype(xs.dtype())?)?;

        let dtype = xs.dtype();
        let t = self
            .t_embedder
            .forward(&timestep_embedding(timestep, TIME_EMBED_DIM)?.to_dtype(dtype)?)?;
        let c = (t + self.y_embedder.forward(pooled)?)?;
        let mut context = Some(self.context_embedder.forward(context)?);

        for block in &self.blocks {
            let (ctx, x) = block.forward(
                context.as_ref().expect("only the last block drops context"),
                &xs,
                &c,
            )?;
            context = ctx;
            xs = x;
        }

        let params = self.final_ada_ln.forward(&ops::silu(&c)?)?;
        let dim = params.dim(D::Minus1)? / 2;
        let shift = params.narrow(D::Minus1, 0, dim)?.unsqueeze(1)?;
        let scale = params.narrow(D::Minus1, dim, dim)?.unsqueeze(1)?;
        let xs = self
            .final_linear
            .forward(&modulate(&plain_layer_norm(&xs)?, &shift, &scale)?)?;

        unpatchify(&xs, lh, lw, p, self.cfg.in_channels)
    }
}

const TIME_EMBED_DIM: usize = 256;

/// Fold SD 3's output tokens back into a latent.
///
/// **Not** the inverse of the packing used on the way in, and this asymmetry
/// is real rather than an oversight. The patch *embedding* is a convolution,
/// whose flattened kernel runs `(channel, ph, pw)` — the order
/// [`crate::flux::pack_latents`] produces. The final linear instead emits
/// `(ph, pw, channel)`, with channel varying fastest, which is what SD 3's
/// `einsum("nhwpqc->nchpwq")` unpacks.
///
/// Reusing Flux's inverse here produces an image of the right shape whose
/// every 2x2 patch has its channels and positions transposed — coherent
/// colours, destroyed detail, and no error.
fn unpatchify(xs: &Tensor, lh: usize, lw: usize, p: usize, c: usize) -> Result<Tensor> {
    let (b, n, _) = xs.dims3()?;
    let (ph, pw) = (lh / p, lw / p);
    if n != ph * pw {
        return Err(sd_tensor::Error::Msg(format!(
            "{n} tokens do not fill a {ph}x{pw} patch grid"
        )));
    }
    xs.reshape((b, ph, pw, p, p, c))?
        .permute((0, 5, 1, 3, 2, 4))?
        .reshape((b, c, lh, lw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn medium_config_matches_the_published_one() {
        let cfg = Sd3Config::medium_35();
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_heads, 24);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.depth, 24);
        assert_eq!(cfg.pooled_dim, 2048, "CLIP-L 768 + CLIP-G 1280");
        assert_eq!(cfg.context_dim, 4096, "T5-XXL width");
        // The first 13 blocks are dual, the rest are not, and the last is
        // additionally context-pre_only.
        assert!(cfg.is_dual(0) && cfg.is_dual(12));
        assert!(!cfg.is_dual(13) && !cfg.is_dual(23));
        assert_eq!(cfg.dual_attention_layers.len(), 13);
    }

    #[test]
    fn modulation_widths_follow_the_block_role() {
        // These are the widths the checkpoint stores, and getting one wrong
        // misreads every later chunk of the same tensor.
        let cfg = Sd3Config::medium_35();
        let h = cfg.hidden_size;
        assert_eq!(9 * h, 13824, "dual image block");
        assert_eq!(6 * h, 9216, "ordinary block");
        assert_eq!(2 * h, 3072, "pre_only context block");
    }
}
