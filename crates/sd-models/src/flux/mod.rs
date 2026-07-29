//! Flux's MMDiT transformer.
//!
//! Not a UNet. There is no downsampling, no skip connections, and no
//! convolution: the latent is cut into 2x2 patches, flattened to a token
//! sequence, and pushed through transformer blocks at constant width.
//! Position comes from explicit rotary embeddings ([`rope`]) rather than from
//! convolutional locality.
//!
//! The stack has two halves. **Double-stream** blocks keep image and text as
//! separate residual streams with their own weights, joining them only inside
//! attention so each modality keeps its own representation. **Single-stream**
//! blocks concatenate the two and run one shared stream, with attention and
//! the feed-forward fused into a single pair of matrices. flux-mini has 5 and
//! 10 of them; full Flux has 19 and 38.
//!
//! Conditioning is by modulation rather than cross-attention. A vector built
//! from the timestep, the guidance scale, and CLIP's pooled embedding is
//! projected per block into `(shift, scale, gate)` triples that scale the
//! normalised activations and gate each residual.
//!
//! Names follow the original black-forest-labs checkpoint layout
//! (`double_blocks.0.img_attn.qkv`), not the `diffusers` renaming, because
//! that is what the published weights use.

pub mod rope;

use crate::weights::{Proj, QuantizedWeights, Source};
use sd_tensor::nn::VarBuilder;
use sd_tensor::{ops, DType, Result, Tensor, D};

/// Flux transformer geometry.
#[derive(Debug, Clone)]
pub struct FluxConfig {
    /// Channels per token entering the stack: `latent_channels * patch^2`.
    pub in_channels: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
    pub depth_single_blocks: usize,
    /// Feed-forward width as a multiple of `hidden_size`.
    pub mlp_ratio: f64,
    /// Head-dimension split across the `(t, h, w)` rotary axes. Must sum to
    /// `hidden_size / num_heads`.
    pub axes_dim: Vec<usize>,
    pub theta: f64,
    /// Width of the CLIP pooled vector fed to `vector_in`.
    pub vec_in_dim: usize,
    /// Width of the T5 sequence fed to `txt_in`.
    pub context_in_dim: usize,
    /// Whether the model takes a distilled guidance scale. Flux dev and
    /// flux-mini do; schnell does not.
    pub guidance_embed: bool,
}

impl FluxConfig {
    /// `TencentARC/flux-mini`: full Flux width, 5 double and 10 single blocks
    /// instead of 19 and 38.
    pub fn mini() -> Self {
        Self {
            in_channels: 64,
            hidden_size: 3072,
            num_heads: 24,
            depth: 5,
            depth_single_blocks: 10,
            mlp_ratio: 4.0,
            axes_dim: vec![16, 56, 56],
            theta: 10_000.0,
            vec_in_dim: 768,
            context_in_dim: 4096,
            guidance_embed: true,
        }
    }

    /// `FLUX.1-dev`.
    pub fn dev() -> Self {
        Self {
            depth: 19,
            depth_single_blocks: 38,
            ..Self::mini()
        }
    }

    /// `FLUX.1-schnell`: same shape as dev, but no guidance embedding.
    pub fn schnell() -> Self {
        Self {
            guidance_embed: false,
            ..Self::dev()
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    fn mlp_hidden(&self) -> usize {
        (self.hidden_size as f64 * self.mlp_ratio) as usize
    }

    fn validate(&self) -> Result<()> {
        let sum: usize = self.axes_dim.iter().sum();
        if sum != self.head_dim() {
            return Err(sd_tensor::Error::Msg(format!(
                "axes_dim sums to {sum} but the head dimension is {}",
                self.head_dim()
            )));
        }
        Ok(())
    }
}

/// Sinusoidal timestep embedding.
///
/// `cos` first, then `sin` — the opposite of the SD UNet's ordering, and
/// swapping them costs nothing structurally while producing a model that is
/// conditioned on the wrong thing. The input is scaled by 1000 because Flux's
/// timesteps arrive as sigmas in `[0, 1]` rather than integer indices.
pub fn timestep_embedding(t: &Tensor, dim: usize, max_period: f64) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.0;
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-max_period.ln() * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), t.device())?;

    let t = (t.to_dtype(DType::F32)?.reshape((t.elem_count(), 1))? * TIME_FACTOR)?;
    let args = t.broadcast_mul(&freqs)?;
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)
}

/// Two-layer MLP with SiLU between, used for every conditioning input.
#[derive(Debug)]
struct MlpEmbedder {
    in_layer: Proj,
    out_layer: Proj,
}

impl MlpEmbedder {
    fn new(in_dim: usize, hidden: usize, src: Source, path: &str) -> Result<Self> {
        Ok(Self {
            in_layer: src.linear(&format!("{path}.in_layer"), in_dim, hidden)?,
            out_layer: src.linear(&format!("{path}.out_layer"), hidden, hidden)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.out_layer
            .forward(&ops::silu(&self.in_layer.forward(xs)?)?)
    }
}

/// RMSNorm applied per attention head, to queries and keys before attention.
///
/// Stabilises attention logits at this depth. The learned scale is
/// `head_dim`-wide, so it normalises within a head rather than across the
/// whole hidden state.
#[derive(Debug)]
struct RmsNorm {
    scale: Tensor,
}

impl RmsNorm {
    fn new(dim: usize, src: Source, path: &str) -> Result<Self> {
        Ok(Self {
            scale: src.tensor(&format!("{path}.scale"), dim)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        ops::rms_norm(xs, &self.scale, 1e-6)
    }
}

/// Query/key normalisation pair.
#[derive(Debug)]
struct QkNorm {
    query_norm: RmsNorm,
    key_norm: RmsNorm,
}

impl QkNorm {
    fn new(dim: usize, src: Source, path: &str) -> Result<Self> {
        Ok(Self {
            query_norm: RmsNorm::new(dim, src, &format!("{path}.query_norm"))?,
            key_norm: RmsNorm::new(dim, src, &format!("{path}.key_norm"))?,
        })
    }
}

/// One `(shift, scale, gate)` triple.
#[derive(Debug, Clone)]
struct Mod {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

/// Projects the conditioning vector into modulation parameters.
///
/// Six for a double-stream block (attention and feed-forward), three for a
/// single-stream one.
#[derive(Debug)]
struct Modulation {
    lin: Proj,
    double: bool,
}

impl Modulation {
    fn new(dim: usize, double: bool, src: Source, path: &str) -> Result<Self> {
        let multiplier = if double { 6 } else { 3 };
        Ok(Self {
            lin: src.linear(&format!("{path}.lin"), dim, multiplier * dim)?,
            double,
        })
    }

    /// Returns `(attention mod, feed-forward mod)`; the second is `None` for
    /// single-stream blocks.
    fn forward(&self, vec: &Tensor) -> Result<(Mod, Option<Mod>)> {
        // SiLU *before* the projection, not after.
        let out = self.lin.forward(&ops::silu(vec)?)?.unsqueeze(1)?;
        let dim = out.dim(D::Minus1)? / if self.double { 6 } else { 3 };
        let chunk = |i: usize| out.narrow(D::Minus1, i * dim, dim);
        let first = Mod {
            shift: chunk(0)?,
            scale: chunk(1)?,
            gate: chunk(2)?,
        };
        if !self.double {
            return Ok((first, None));
        }
        Ok((
            first,
            Some(Mod {
                shift: chunk(3)?,
                scale: chunk(4)?,
                gate: chunk(5)?,
            }),
        ))
    }
}

/// `(1 + scale) * norm(x) + shift`.
///
/// The `1 +` is what makes an untrained modulation the identity. Dropping it
/// leaves a model that trains but does not match published weights.
fn modulate(xs: &Tensor, m: &Mod) -> Result<Tensor> {
    xs.broadcast_mul(&(&m.scale + 1.0)?)?
        .broadcast_add(&m.shift)
}

/// LayerNorm with no learned parameters at all.
///
/// Every norm inside these blocks is `elementwise_affine=False` — the scale
/// and shift come from the modulation vector instead, which is the whole
/// mechanism by which Flux is conditioned. candle's `layer_norm` always reads
/// a `weight` even when told `affine: false` (that flag only drops the bias),
/// so it cannot express this and the arithmetic is done directly.
#[derive(Debug, Clone, Copy)]
struct PlainLayerNorm {
    eps: f64,
}

impl PlainLayerNorm {
    fn new() -> Self {
        Self { eps: 1e-6 }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        ops::plain_layer_norm(xs, self.eps)
    }
}

/// Split a fused `qkv` projection into per-head `q`, `k`, `v`.
fn split_qkv(qkv: &Tensor, heads: usize, head_dim: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, n, _) = qkv.dims3()?;
    // [b, n, 3*heads*head_dim] -> [3, b, heads, n, head_dim]
    let t = qkv
        .reshape((b, n, 3, heads, head_dim))?
        .permute((2, 0, 3, 1, 4))?;
    let take = |i: usize| t.narrow(0, i, 1)?.squeeze(0)?.contiguous();
    Ok((take(0)?, take(1)?, take(2)?))
}

fn merge_heads(xs: &Tensor) -> Result<Tensor> {
    let (b, h, n, d) = xs.dims4()?;
    xs.transpose(1, 2)?.reshape((b, n, h * d))
}

/// A double-stream block: separate image and text streams, joined in attention.
#[derive(Debug)]
struct DoubleStreamBlock {
    img_mod: Modulation,
    img_norm1: PlainLayerNorm,
    img_qkv: Proj,
    img_qk_norm: QkNorm,
    img_proj: Proj,
    img_norm2: PlainLayerNorm,
    img_mlp_in: Proj,
    img_mlp_out: Proj,

    txt_mod: Modulation,
    txt_norm1: PlainLayerNorm,
    txt_qkv: Proj,
    txt_qk_norm: QkNorm,
    txt_proj: Proj,
    txt_norm2: PlainLayerNorm,
    txt_mlp_in: Proj,
    txt_mlp_out: Proj,

    num_heads: usize,
    head_dim: usize,
}

impl DoubleStreamBlock {
    fn new(cfg: &FluxConfig, src: Source, path: &str) -> Result<Self> {
        let (h, mlp) = (cfg.hidden_size, cfg.mlp_hidden());
        let hd = cfg.head_dim();
        Ok(Self {
            img_mod: Modulation::new(h, true, src, &format!("{path}.img_mod"))?,
            img_norm1: PlainLayerNorm::new(),
            img_qkv: src.linear(&format!("{path}.img_attn.qkv"), h, 3 * h)?,
            img_qk_norm: QkNorm::new(hd, src, &format!("{path}.img_attn.norm"))?,
            img_proj: src.linear(&format!("{path}.img_attn.proj"), h, h)?,
            img_norm2: PlainLayerNorm::new(),
            // `img_mlp` is a Sequential: 0 is the projection, 1 the GELU,
            // 2 the output — hence the index-named weights.
            img_mlp_in: src.linear(&format!("{path}.img_mlp.0"), h, mlp)?,
            img_mlp_out: src.linear(&format!("{path}.img_mlp.2"), mlp, h)?,

            txt_mod: Modulation::new(h, true, src, &format!("{path}.txt_mod"))?,
            txt_norm1: PlainLayerNorm::new(),
            txt_qkv: src.linear(&format!("{path}.txt_attn.qkv"), h, 3 * h)?,
            txt_qk_norm: QkNorm::new(hd, src, &format!("{path}.txt_attn.norm"))?,
            txt_proj: src.linear(&format!("{path}.txt_attn.proj"), h, h)?,
            txt_norm2: PlainLayerNorm::new(),
            txt_mlp_in: src.linear(&format!("{path}.txt_mlp.0"), h, mlp)?,
            txt_mlp_out: src.linear(&format!("{path}.txt_mlp.2"), mlp, h)?,

            num_heads: cfg.num_heads,
            head_dim: hd,
        })
    }

    fn resident_bytes(&self) -> usize {
        [
            &self.img_qkv,
            &self.img_proj,
            &self.img_mlp_in,
            &self.img_mlp_out,
            &self.txt_qkv,
            &self.txt_proj,
            &self.txt_mlp_in,
            &self.txt_mlp_out,
        ]
        .iter()
        .map(|p| p.resident_bytes())
        .sum()
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec: &Tensor,
        pe: &rope::Rope,
    ) -> Result<(Tensor, Tensor)> {
        let (img_mod1, img_mod2) = self.img_mod.forward(vec)?;
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(vec)?;
        let (img_mod2, txt_mod2) = (
            img_mod2.expect("double block"),
            txt_mod2.expect("double block"),
        );

        let img_in = modulate(&self.img_norm1.forward(img)?, &img_mod1)?;
        let (img_q, img_k, img_v) = split_qkv(
            &self.img_qkv.forward(&img_in)?,
            self.num_heads,
            self.head_dim,
        )?;
        let img_q = self.img_qk_norm.query_norm.forward(&img_q)?;
        let img_k = self.img_qk_norm.key_norm.forward(&img_k)?;

        let txt_in = modulate(&self.txt_norm1.forward(txt)?, &txt_mod1)?;
        let (txt_q, txt_k, txt_v) = split_qkv(
            &self.txt_qkv.forward(&txt_in)?,
            self.num_heads,
            self.head_dim,
        )?;
        let txt_q = self.txt_qk_norm.query_norm.forward(&txt_q)?;
        let txt_k = self.txt_qk_norm.key_norm.forward(&txt_k)?;

        // Text first, then image — the same order the position ids were
        // concatenated in, which is what makes the rotary embedding line up.
        let q = Tensor::cat(&[&txt_q, &img_q], 2)?;
        let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
        let v = Tensor::cat(&[&txt_v, &img_v], 2)?.contiguous()?;

        let (q, k) = rope::apply_rope_fused(&q, &k, pe)?;
        let attn = ops::scaled_dot_product_attention(&q.contiguous()?, &k.contiguous()?, &v)?;
        let attn = merge_heads(&attn)?;

        let txt_len = txt.dim(1)?;
        let txt_attn = attn.narrow(1, 0, txt_len)?.contiguous()?;
        let img_attn = attn
            .narrow(1, txt_len, attn.dim(1)? - txt_len)?
            .contiguous()?;

        let img = (img
            + self
                .img_proj
                .forward(&img_attn)?
                .broadcast_mul(&img_mod1.gate)?)?;
        let img_ff = self.img_mlp_out.forward(&ops::gelu_approx(
            &self
                .img_mlp_in
                .forward(&modulate(&self.img_norm2.forward(&img)?, &img_mod2)?)?,
        )?)?;
        let img = (img + img_ff.broadcast_mul(&img_mod2.gate)?)?;

        let txt = (txt
            + self
                .txt_proj
                .forward(&txt_attn)?
                .broadcast_mul(&txt_mod1.gate)?)?;
        let txt_ff = self.txt_mlp_out.forward(&ops::gelu_approx(
            &self
                .txt_mlp_in
                .forward(&modulate(&self.txt_norm2.forward(&txt)?, &txt_mod2)?)?,
        )?)?;
        let txt = (txt + txt_ff.broadcast_mul(&txt_mod2.gate)?)?;

        Ok((img, txt))
    }
}

/// A single-stream block: one sequence, with attention and feed-forward fused
/// into `linear1` / `linear2`.
#[derive(Debug)]
struct SingleStreamBlock {
    modulation: Modulation,
    pre_norm: PlainLayerNorm,
    linear1: Proj,
    linear2: Proj,
    qk_norm: QkNorm,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    mlp_hidden: usize,
}

impl SingleStreamBlock {
    fn new(cfg: &FluxConfig, src: Source, path: &str) -> Result<Self> {
        let (h, mlp) = (cfg.hidden_size, cfg.mlp_hidden());
        Ok(Self {
            modulation: Modulation::new(h, false, src, &format!("{path}.modulation"))?,
            pre_norm: PlainLayerNorm::new(),
            // One projection producing qkv *and* the feed-forward input.
            linear1: src.linear(&format!("{path}.linear1"), h, 3 * h + mlp)?,
            // And one consuming attention output *and* the gated MLP.
            linear2: src.linear(&format!("{path}.linear2"), h + mlp, h)?,
            qk_norm: QkNorm::new(cfg.head_dim(), src, &format!("{path}.norm"))?,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
            hidden_size: h,
            mlp_hidden: mlp,
        })
    }

    fn resident_bytes(&self) -> usize {
        self.linear1.resident_bytes() + self.linear2.resident_bytes()
    }

    fn forward(&self, xs: &Tensor, vec: &Tensor, pe: &rope::Rope) -> Result<Tensor> {
        let (m, _) = self.modulation.forward(vec)?;
        let x_mod = modulate(&self.pre_norm.forward(xs)?, &m)?;

        let projected = self.linear1.forward(&x_mod)?;
        let qkv = projected.narrow(D::Minus1, 0, 3 * self.hidden_size)?;
        let mlp = projected.narrow(D::Minus1, 3 * self.hidden_size, self.mlp_hidden)?;

        let (q, k, v) = split_qkv(&qkv.contiguous()?, self.num_heads, self.head_dim)?;
        let q = self.qk_norm.query_norm.forward(&q)?;
        let k = self.qk_norm.key_norm.forward(&k)?;
        let (q, k) = rope::apply_rope_fused(&q, &k, pe)?;
        let attn = ops::scaled_dot_product_attention(&q.contiguous()?, &k.contiguous()?, &v)?;
        let attn = merge_heads(&attn)?;

        let joined = Tensor::cat(&[&attn, &ops::gelu_approx(&mlp.contiguous()?)?], D::Minus1)?;
        let out = self.linear2.forward(&joined.contiguous()?)?;
        xs + out.broadcast_mul(&m.gate)?
    }
}

/// The output head: modulate, then project back to patch channels.
#[derive(Debug)]
struct LastLayer {
    norm_final: PlainLayerNorm,
    ada_ln: Proj,
    linear: Proj,
}

impl LastLayer {
    fn new(cfg: &FluxConfig, src: Source, path: &str) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            norm_final: PlainLayerNorm::new(),
            // `adaLN_modulation` is Sequential(SiLU, Linear); index 1 is the
            // Linear, and index 0 has no weights.
            ada_ln: src.linear(&format!("{path}.adaLN_modulation.1"), h, 2 * h)?,
            linear: src.linear(&format!("{path}.linear"), h, cfg.in_channels)?,
        })
    }

    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let params = self.ada_ln.forward(&ops::silu(vec)?)?;
        let dim = params.dim(D::Minus1)? / 2;
        // shift comes first here, unlike `Modulation`, which yields
        // (shift, scale, gate) — the ordering is per-module, not global.
        let shift = params.narrow(D::Minus1, 0, dim)?.unsqueeze(1)?;
        let scale = params.narrow(D::Minus1, dim, dim)?.unsqueeze(1)?;
        let xs = self
            .norm_final
            .forward(xs)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
        self.linear.forward(&xs)
    }
}

/// The Flux transformer.
#[derive(Debug)]
pub struct FluxTransformer {
    img_in: Proj,
    txt_in: Proj,
    time_in: MlpEmbedder,
    vector_in: MlpEmbedder,
    guidance_in: Option<MlpEmbedder>,
    blocks: Blocks,
    final_layer: LastLayer,
    cfg: FluxConfig,
}

/// Where the transformer's blocks keep their weights.
///
/// The blocks are all of it — 6.78 GB of a 6.8 GB quantised schnell — so this
/// is the only part worth streaming. The embedders and the output head are a
/// few hundred MB and stay put.
#[derive(Debug)]
enum Blocks {
    /// Built once, resident on the compute device for the whole run.
    Resident {
        double: Vec<DoubleStreamBlock>,
        single: Vec<SingleStreamBlock>,
    },
    /// Weights held in host memory; each block is copied to the compute
    /// device as it is reached and released after it has run.
    ///
    /// **What this buys, and what it costs.** Peak weight residency on the
    /// accelerator drops from the whole transformer to one block — 6.78 GB to
    /// 192 MB for schnell — which is the difference between a 12 GB card
    /// running Flux and not running it. Measured on an M4 Max the copy runs at
    /// 19.8 GB/s, so a schnell step pays 19 double blocks at 9.68 ms plus 38
    /// single at 4.18 ms, about 343 ms per step, or roughly 6.5% on a 4-step
    /// run. A discrete card over PCIe should expect around double that and the
    /// same conclusion.
    ///
    /// Only the quantised path can do this. `quantized::to_device` copies the
    /// quantised block bytes verbatim, so it is cheap and bit-exact; a dense
    /// checkpoint would have to move 4x the bytes, and there is no lossless
    /// route for it that is any cheaper.
    ///
    /// **Where the overhead actually goes is not where it looks.** Per step,
    /// over the 19 double blocks (`SD_STREAM_PROFILE=1`):
    ///
    /// ```text
    ///   copy   354 ms      build  1019 ms      run  244 ms
    /// ```
    ///
    /// Rebuilding the block costs three times the copy and four times the
    /// arithmetic. It is GPU-call overhead rather than data: `QMatMul::from_arc`
    /// dequantises any F32/F16 tensor it is handed, and `Source::linear`
    /// dequantises every bias, so a block build issues roughly a dozen small
    /// device operations — 468 of this checkpoint's 776 tensors are F32
    /// biases and norm scales.
    ///
    /// So prefetching the copy, the obvious next move, would hide the
    /// *smallest* of the three. Caching the dequantised biases and norm
    /// scales instead — they are tiny, and constant across steps — is where
    /// the second is. Measure before assuming otherwise; that is what the
    /// profile hook is for.
    Streamed {
        weights: QuantizedWeights,
        /// Biases and norm scales, dequantised once and kept on the device.
        /// 127 MB against the weights' 6.66 GB, and the difference between a
        /// block rebuild costing a dozen device operations and costing none.
        dense: crate::weights::DenseCache,
        device: sd_tensor::Device,
    },
}

impl FluxTransformer {
    pub fn new(cfg: &FluxConfig, vb: VarBuilder) -> Result<Self> {
        Self::from_source(cfg, Source::Dense(&vb))
    }

    /// Build with the weights left quantised.
    ///
    /// This is what makes full-size Flux reachable at all: dev and schnell are
    /// 12B parameters, or 48 GB at F32, against roughly 6.8 GB held as Q4_K.
    /// It is also the only way the model runs on this machine at any size,
    /// since F16 — the obvious way to halve F32 — produces NaN velocities.
    pub fn from_quantized(cfg: &FluxConfig, weights: &QuantizedWeights) -> Result<Self> {
        Self::from_source(cfg, Source::Quantized(weights))
    }

    /// Build with the blocks streamed rather than resident.
    ///
    /// `weights` stay wherever the caller loaded them — host memory is the
    /// point — and each block is copied to `device` as it is reached. See
    /// [`Blocks::Streamed`] for what that costs and buys.
    ///
    /// Everything outside the blocks is built resident on `device`: the
    /// embedders and output head are a few hundred MB against the blocks'
    /// 6.78 GB, and they are touched every step, so streaming them would be
    /// all cost.
    pub fn from_quantized_streaming(
        cfg: &FluxConfig,
        weights: &QuantizedWeights,
        device: &sd_tensor::Device,
    ) -> Result<Self> {
        // The embedders and output head run every step, so they are moved to
        // the device once and stay. The blocks are the 6.78 GB and are what
        // stays behind.
        let mut resident = QuantizedWeights::new();
        for (name, weight) in weights.iter() {
            if !name.starts_with("double_blocks.") && !name.starts_with("single_blocks.") {
                resident.insert(
                    name.clone(),
                    sd_tensor::quantized::to_device(weight, device)?,
                );
            }
        }
        let dense =
            crate::weights::dense_cache(weights, &["double_blocks.", "single_blocks."], device)?;
        Self::from_source_with_blocks(
            cfg,
            Source::Quantized(&resident),
            Blocks::Streamed {
                weights: weights.clone(),
                dense,
                device: device.clone(),
            },
        )
    }

    fn from_source(cfg: &FluxConfig, src: Source) -> Result<Self> {
        let blocks = Blocks::Resident {
            double: (0..cfg.depth)
                .map(|i| DoubleStreamBlock::new(cfg, src, &format!("double_blocks.{i}")))
                .collect::<Result<Vec<_>>>()?,
            single: (0..cfg.depth_single_blocks)
                .map(|i| SingleStreamBlock::new(cfg, src, &format!("single_blocks.{i}")))
                .collect::<Result<Vec<_>>>()?,
        };
        Self::from_source_with_blocks(cfg, src, blocks)
    }

    /// Everything but the blocks, which the caller supplies.
    ///
    /// Split out so the streaming constructor can build the embedders and the
    /// output head from device-resident weights while the blocks stay in host
    /// memory — the two halves genuinely live in different places.
    fn from_source_with_blocks(cfg: &FluxConfig, src: Source, blocks: Blocks) -> Result<Self> {
        cfg.validate()?;
        let h = cfg.hidden_size;
        Ok(Self {
            img_in: src.linear("img_in", cfg.in_channels, h)?,
            txt_in: src.linear("txt_in", cfg.context_in_dim, h)?,
            time_in: MlpEmbedder::new(TIME_EMBED_DIM, h, src, "time_in")?,
            vector_in: MlpEmbedder::new(cfg.vec_in_dim, h, src, "vector_in")?,
            guidance_in: if cfg.guidance_embed {
                Some(MlpEmbedder::new(TIME_EMBED_DIM, h, src, "guidance_in")?)
            } else {
                None
            },
            blocks,
            final_layer: LastLayer::new(cfg, src, "final_layer")?,
            cfg: cfg.clone(),
        })
    }

    /// Weight bytes held *on the compute device* by the quantised projections.
    ///
    /// Zero when dense, and — deliberately — excludes streamed blocks, whose
    /// weights are in host memory and only ever one block at a time on the
    /// device. Reporting them here would say the opposite of what streaming
    /// achieves.
    pub fn resident_bytes(&self) -> usize {
        let block: usize = match &self.blocks {
            Blocks::Resident { double, single } => double
                .iter()
                .map(|b| b.resident_bytes())
                .chain(single.iter().map(|b| b.resident_bytes()))
                .sum(),
            Blocks::Streamed { .. } => 0,
        };
        block + self.img_in.resident_bytes() + self.txt_in.resident_bytes()
    }

    pub fn config(&self) -> &FluxConfig {
        &self.cfg
    }

    /// Predict the flow velocity.
    ///
    /// - `img` — packed latent `[b, img_len, in_channels]`
    /// - `img_ids` / `txt_ids` — `[b, len, 3]` coordinates from [`rope`]
    /// - `txt` — T5 sequence `[b, txt_len, context_in_dim]`
    /// - `timesteps` — `[b]`, in `[0, 1]`
    /// - `y` — CLIP pooled `[b, vec_in_dim]`
    /// - `guidance` — `[b]` distilled guidance scale, required iff the config
    ///   asks for it
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut img = self.img_in.forward(img)?;
        let txt = self.txt_in.forward(txt)?;

        // `timestep_embedding` works in f32 deliberately — its frequencies
        // span several orders of magnitude and f16 loses the low ones — so it
        // is cast back to the weights' dtype here rather than being computed
        // narrow throughout.
        let dtype = img.dtype();
        let embed = |t: &Tensor| -> Result<Tensor> {
            timestep_embedding(t, TIME_EMBED_DIM, self.cfg.theta)?.to_dtype(dtype)
        };

        let mut vec = self.time_in.forward(&embed(timesteps)?)?;
        match (&self.guidance_in, guidance) {
            (Some(g), Some(scale)) => {
                vec = (vec + g.forward(&embed(scale)?)?)?;
            }
            (Some(_), None) => {
                return Err(sd_tensor::Error::Msg(
                    "this checkpoint has a guidance embedding and needs a guidance scale".into(),
                ))
            }
            (None, Some(_)) => {
                return Err(sd_tensor::Error::Msg(
                    "this checkpoint takes no guidance scale (schnell is not distilled on one)"
                        .into(),
                ))
            }
            (None, None) => {}
        }
        vec = (vec + self.vector_in.forward(y)?)?;

        // Text ids precede image ids, matching the concatenation order inside
        // every block.
        let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
        let pe = rope::embed_nd_cos_sin(&ids, &self.cfg.axes_dim, self.cfg.theta)?;

        let mut txt = txt;
        match &self.blocks {
            Blocks::Resident { double, .. } => {
                for block in double {
                    let (i, t) = block.forward(&img, &txt, &vec, &pe)?;
                    img = i;
                    txt = t;
                }
            }
            Blocks::Streamed {
                weights,
                dense,
                device,
            } => {
                // `SD_STREAM_PROFILE=1` splits a streamed step three ways. It
                // is here because the split is not what anyone guesses: the
                // copy is the small part. See `Blocks::Streamed`.
                let dbg = std::env::var("SD_STREAM_PROFILE").is_ok();
                let sync_every = crate::weights::stream_sync_every();
                let (mut t_copy, mut t_build, mut t_run) = (0.0f64, 0.0f64, 0.0f64);
                for i in 0..self.cfg.depth {
                    let path = format!("double_blocks.{i}");
                    let t0 = std::time::Instant::now();
                    let resident = crate::weights::block_weights(weights, &path, device)?;
                    let t1 = std::time::Instant::now();
                    let block = DoubleStreamBlock::new(
                        &self.cfg,
                        Source::QuantizedCached(&resident, dense),
                        &path,
                    )?;
                    let t2 = std::time::Instant::now();
                    t_copy += (t1 - t0).as_secs_f64();
                    t_build += (t2 - t1).as_secs_f64();
                    let (a, t) = block.forward(&img, &txt, &vec, &pe)?;
                    t_run += t2.elapsed().as_secs_f64();
                    img = a;
                    txt = t;
                    // Release this block's device memory before making the
                    // next copy. Dropping is not enough on Metal: candle
                    // pools its buffers and only hands them back inside
                    // `drop_unused_buffers`, which runs on synchronise. Without
                    // this the pool grows by a block per iteration — measured
                    // going from 354 ms of copy per step to 62 seconds as the
                    // machine started swapping.
                    drop(block);
                    drop(resident);
                    if (i + 1) % sync_every == 0 {
                        device.synchronize()?;
                    }
                }
                device.synchronize()?;
                if dbg {
                    eprintln!(
                        "  [stream] double: copy {:.0} ms, build {:.0} ms, run {:.0} ms",
                        t_copy * 1e3,
                        t_build * 1e3,
                        t_run * 1e3
                    );
                }
            }
        }

        // The single-stream half runs on the concatenation, then the text
        // half is dropped — it has done its work as context by this point.
        let txt_len = txt.dim(1)?;
        let mut xs = Tensor::cat(&[&txt, &img], 1)?.contiguous()?;
        match &self.blocks {
            Blocks::Resident { single, .. } => {
                for block in single {
                    xs = block.forward(&xs, &vec, &pe)?;
                }
            }
            Blocks::Streamed {
                weights,
                dense,
                device,
            } => {
                for i in 0..self.cfg.depth_single_blocks {
                    let path = format!("single_blocks.{i}");
                    let resident = crate::weights::block_weights(weights, &path, device)?;
                    let block = SingleStreamBlock::new(
                        &self.cfg,
                        Source::QuantizedCached(&resident, dense),
                        &path,
                    )?;
                    xs = block.forward(&xs, &vec, &pe)?;
                    drop(block);
                    drop(resident);
                    device.synchronize()?;
                }
            }
        }
        let img = xs.narrow(1, txt_len, xs.dim(1)? - txt_len)?.contiguous()?;

        self.final_layer.forward(&img, &vec)
    }
}

/// Width of the sinusoidal timestep embedding before its MLP.
const TIME_EMBED_DIM: usize = 256;

/// Fold a latent `[b, c, h, w]` into patch tokens `[b, (h/2)*(w/2), c*4]`.
///
/// Flux's "patchify". The channel order within a token is `(c, ph, pw)` —
/// getting it wrong scrambles each 2x2 patch internally, which the model can
/// partly absorb, so it degrades quality rather than failing.
pub fn pack_latents(latents: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = latents.dims4()?;
    if h % 2 != 0 || w % 2 != 0 {
        return Err(sd_tensor::Error::Msg(format!(
            "latent {h}x{w} must have even sides to pack into 2x2 patches"
        )));
    }
    latents
        .reshape((b, c, h / 2, 2, w / 2, 2))?
        .permute((0, 2, 4, 1, 3, 5))?
        .reshape((b, (h / 2) * (w / 2), c * 4))
}

/// The inverse of [`pack_latents`].
pub fn unpack_latents(tokens: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let (b, n, cc) = tokens.dims3()?;
    let c = cc / 4;
    if n != (h / 2) * (w / 2) {
        return Err(sd_tensor::Error::Msg(format!(
            "{n} tokens do not fill a {h}x{w} latent"
        )));
    }
    tokens
        .reshape((b, h / 2, w / 2, c, 2, 2))?
        .permute((0, 3, 1, 4, 2, 5))?
        .reshape((b, c, h, w))
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use sd_tensor::gguf::{GgmlDType, QTensor};
    use sd_tensor::Device;
    use std::sync::Arc;

    fn fake(name: &str, weights: &mut QuantizedWeights) {
        let t = Tensor::zeros((256, 256), DType::F32, &Device::Cpu).unwrap();
        let q = QTensor::quantize(&t, GgmlDType::Q4K).unwrap();
        weights.insert(name.to_string(), Arc::new(q));
    }

    #[test]
    fn a_block_selects_its_own_weights_and_not_a_longer_numbered_sibling() {
        // `double_blocks.1` must not sweep up `double_blocks.10..19`, which is
        // what a prefix match without the trailing dot does. Flux schnell has
        // 19 double and 38 single blocks, so every single-digit index has
        // two-digit siblings and the bug would be silent: the block would
        // simply be built from whichever duplicate name won.
        let mut all = QuantizedWeights::new();
        for i in [1usize, 10, 11, 19] {
            fake(&format!("double_blocks.{i}.img_attn.qkv.weight"), &mut all);
            fake(&format!("double_blocks.{i}.img_mlp.0.weight"), &mut all);
        }
        let one = crate::weights::block_weights(&all, "double_blocks.1", &Device::Cpu).unwrap();
        assert_eq!(
            one.len(),
            2,
            "block 1 has two weights, not block 10's as well"
        );
        assert!(one.keys().all(|k| k.starts_with("double_blocks.1.")));

        let ten = crate::weights::block_weights(&all, "double_blocks.10", &Device::Cpu).unwrap();
        assert_eq!(ten.len(), 2);
        assert!(ten.keys().all(|k| k.starts_with("double_blocks.10.")));
    }

    #[test]
    fn a_missing_block_is_an_error_rather_than_an_empty_one() {
        // Silently returning nothing would build a block whose every weight
        // lookup fails later, reported as a missing tensor deep in the model
        // rather than as the wrong index here.
        let mut all = QuantizedWeights::new();
        fake("double_blocks.0.img_attn.qkv.weight", &mut all);
        let err = crate::weights::block_weights(&all, "double_blocks.7", &Device::Cpu)
            .expect_err("block 7 does not exist");
        assert!(err.to_string().contains("double_blocks.7."), "{err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sd_tensor::Device;

    #[test]
    fn packing_round_trips() {
        let dev = Device::Cpu;
        let x = Tensor::rand(-1f32, 1f32, (2, 16, 8, 6), &dev).unwrap();
        let packed = pack_latents(&x).unwrap();
        assert_eq!(packed.dims(), &[2, 12, 64], "16 channels x 2x2 = 64");
        let back = unpack_latents(&packed, 8, 6).unwrap();
        let err = (&x - &back)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(err, 0.0, "packing must be exactly invertible");
    }

    #[test]
    fn packing_keeps_spatial_neighbours_together() {
        // A round trip alone would pass for any consistent permutation,
        // including one that scrambles each patch. Check that token k really
        // holds the 2x2 block at (row, col).
        let dev = Device::Cpu;
        let v: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x = Tensor::from_vec(v, (1, 1, 4, 4), &dev).unwrap();
        let packed = pack_latents(&x).unwrap();
        let got = packed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // First token is the top-left 2x2 of a row-major 4x4: 0,1,4,5.
        assert_eq!(&got[..4], &[0.0, 1.0, 4.0, 5.0]);
        // Second token is the next 2x2 across: 2,3,6,7.
        assert_eq!(&got[4..8], &[2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn odd_latents_are_rejected() {
        let dev = Device::Cpu;
        let x = Tensor::rand(-1f32, 1f32, (1, 16, 5, 4), &dev).unwrap();
        assert!(pack_latents(&x).is_err(), "odd height cannot patch evenly");
    }

    #[test]
    fn timestep_embedding_puts_cos_first() {
        // The SD UNet emits sin first. Flux is the other way round, and the
        // shapes are identical either way.
        let dev = Device::Cpu;
        let t = Tensor::from_vec(vec![0f32], 1, &dev).unwrap();
        let e = timestep_embedding(&t, 8, 10_000.0).unwrap();
        let v = e.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v.len(), 8);
        // At t = 0 every angle is 0, so cos = 1 and sin = 0.
        assert!(v[..4].iter().all(|x| (x - 1.0).abs() < 1e-6), "cos half");
        assert!(v[4..].iter().all(|x| x.abs() < 1e-6), "sin half");
    }

    #[test]
    fn config_axes_must_fill_the_head() {
        let mut cfg = FluxConfig::mini();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.head_dim(), 128);
        cfg.axes_dim = vec![16, 56, 55];
        assert!(
            cfg.validate().is_err(),
            "axes that do not sum to the head dimension must be rejected"
        );
    }
}
