//! The compute seam for stable-diffusion.rs.
//!
//! # Why this crate exists
//!
//! Every model, sampler and loader in this workspace talks to tensors *only*
//! through this crate. `sd-tensor` is the single place that names `candle`.
//!
//! That buys us one thing: the ability to change our mind. candle is pre-1.0,
//! maintained largely by one person, and — like ggml before it — optimised for
//! language models rather than diffusion. If it stalls, or if a specific kernel
//! turns out to be the bottleneck, we replace it *here* instead of rewriting
//! every model in the workspace.
//!
//! # The rule
//!
//! No crate other than `sd-tensor` may `use candle_core` or `candle_nn`.
//! This is enforced in CI by `scripts/check-seam.sh`. If you find yourself
//! wanting to reach past the seam, add the missing thing to this crate instead.
//!
//! Keep the seam *thin*. It is a re-export surface plus the handful of ops
//! candle does not provide — not an abstraction layer with its own opinions.

pub use candle_core::{
    safetensors, DType, Device, Error, IndexOp, Module, Result, Shape, Tensor, D,
};
pub use candle_nn::VarBuilder;

/// Layers we build models out of. Re-exported so model crates never name candle.
pub mod nn {
    pub use candle_nn::{
        conv2d, conv2d_no_bias, embedding, group_norm, layer_norm, linear, linear_no_bias, Conv2d,
        Conv2dConfig, Embedding, GroupNorm, LayerNorm, LayerNormConfig, Linear, VarBuilder, VarMap,
    };
}

/// Elementwise and reduction ops.
///
/// Most forward to candle. The ones that do not are marked, and are the first
/// candidates for a native implementation if we ever move off candle.
pub mod ops {
    use super::{DType, Error, Result, Tensor, D};

    pub use candle_nn::ops::{silu, softmax, softmax_last_dim};

    /// SiLU / swish: `x * sigmoid(x)`.
    pub fn swish(xs: &Tensor) -> Result<Tensor> {
        silu(xs)
    }

    /// Exact GELU (erf-based), matching PyTorch's default `nn.GELU()`.
    ///
    /// Note this is *not* the tanh approximation. Diffusion models are
    /// sensitive to the difference; using the wrong one produces images that
    /// look plausible but drift from the reference.
    pub fn gelu(xs: &Tensor) -> Result<Tensor> {
        xs.gelu_erf()
    }

    /// Tanh-approximate GELU. Used by some text encoders (e.g. CLIP's
    /// `quick_gelu` is different again — see [`quick_gelu`]).
    pub fn gelu_approx(xs: &Tensor) -> Result<Tensor> {
        xs.gelu()
    }

    /// CLIP's activation: `x * sigmoid(1.702 * x)`.
    pub fn quick_gelu(xs: &Tensor) -> Result<Tensor> {
        xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?)?
    }

    /// Which implementation served an attention call.
    ///
    /// Returned by [`attention_with_path`] so callers — tests especially — can
    /// assert *which* path ran. Without it, a test that compares the
    /// dispatcher against [`naive_attention`] silently degrades into comparing
    /// the naive path with itself: it passes, and proves nothing.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum AttentionPath {
        /// candle's fused kernel. The score matrix is never materialised.
        Fused,
        /// [`chunked_attention`]: the score matrix exists one tile at a time.
        Chunked,
        /// [`naive_attention`], bounded by [`check_attention_budget`]. Also
        /// what a chunked call degenerates to when one chunk already covers
        /// the whole query axis.
        Naive,
    }

    /// Environment override for the naive-attention memory budget, in bytes.
    pub const ATTENTION_BUDGET_ENV: &str = "SD_ATTENTION_BUDGET_BYTES";

    /// Ceiling on any single allocation the models make.
    ///
    /// 4 GiB, and the figure is set by convolution rather than by attention.
    /// A 512px decode — one tile, and SD 1.5's whole image — allocates a
    /// 2.42 GB conv im2col, so anything below that refuses ordinary work. An
    /// untiled 1024px decode needs 9.66 GB and is refused, which is correct:
    /// tiling exists to bring it back under this line.
    ///
    /// It was 2 GiB while the estimate counted only activations. That
    /// estimate under-reported by 18x, so the two numbers were wrong together
    /// and looked consistent. See `DecoderConfig::peak_alloc_bytes`.
    pub const DEFAULT_ATTENTION_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

    /// Bytes one attention score matrix needs.
    ///
    /// `None` means the count overflowed 64 bits, which is itself a refusal —
    /// no machine runs that shape.
    ///
    /// Note the `batch` and `heads` factors: the score matrix is
    /// `[batch, heads, seq_q, seq_k]`, so a per-head figure understates a
    /// multi-head call by `batch * heads`.
    pub fn attention_score_bytes(
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_k: usize,
        dtype: DType,
    ) -> Option<u64> {
        (batch as u64)
            .checked_mul(heads as u64)?
            .checked_mul(seq_q as u64)?
            .checked_mul(seq_k as u64)?
            .checked_mul(dtype.size_in_bytes() as u64)
    }

    /// Parse a budget override. `None` (unset) keeps the default.
    pub fn parse_attention_budget(raw: Option<&str>) -> Result<u64> {
        let Some(raw) = raw else {
            return Ok(DEFAULT_ATTENTION_BUDGET_BYTES);
        };
        raw.trim().parse().map_err(|_| {
            Error::Msg(format!(
                "{ATTENTION_BUDGET_ENV} must be a byte count, got {raw:?}"
            ))
        })
    }

    /// The active budget, honouring [`ATTENTION_BUDGET_ENV`].
    pub fn attention_budget_bytes() -> Result<u64> {
        match std::env::var(ATTENTION_BUDGET_ENV) {
            Ok(raw) => parse_attention_budget(Some(&raw)),
            Err(std::env::VarError::NotPresent) => Ok(DEFAULT_ATTENTION_BUDGET_BYTES),
            Err(std::env::VarError::NotUnicode(_)) => Err(Error::Msg(format!(
                "{ATTENTION_BUDGET_ENV} is not valid UTF-8"
            ))),
        }
    }

    /// Refuse a naive-attention call whose score matrix exceeds the budget.
    ///
    /// This is a hard error rather than a warning because of how the failure
    /// actually presents. On a Metal build the score matrix is wired GPU
    /// memory: the GPU cannot take page faults, so those pages are pinned,
    /// unswappable, and invisible to jetsam. Exceeding physical RAM therefore
    /// does not kill the process — the kernel runs out of reclaimable pages
    /// and the machine panics.
    ///
    /// On 2026-07-25 a VAE decode at a 384x384 latent projected 81 GiB on a
    /// 36 GiB Mac and took the whole machine down with a watchdog timeout. The
    /// process never got an allocation failure to report. That is why the
    /// check happens here, before anything is allocated, instead of being left
    /// to the allocator — and why it lives in the seam rather than in one
    /// benchmark, since every caller allocates through this path.
    ///
    /// The returned figure is one score matrix. Peak usage is **at least
    /// twice** that: scaling and softmax each produce a separate allocation of
    /// the same size, and a mask adds another.
    ///
    /// Cost here is `O(n^4)` in the latent edge `n`, because `seq = n*n` and
    /// the matrix is `seq x seq`. One step up a doubling sweep costs 16x.
    pub fn check_attention_budget(
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_k: usize,
        dtype: DType,
    ) -> Result<u64> {
        let bytes = attention_score_bytes(batch, heads, seq_q, seq_k, dtype);
        check_alloc_budget(
            bytes,
            &format!("attention score matrix [{batch}, {heads}, {seq_q}, {seq_k}] ({dtype:?})"),
        )?;
        Ok(bytes.expect("check_alloc_budget rejects the overflow case"))
    }

    /// Refuse any single allocation over the budget.
    ///
    /// `bytes` is `None` when the size computation overflowed, which is itself
    /// a refusal. `what` describes the allocation and appears in the error.
    ///
    /// This is the general form of [`check_attention_budget`]. Attention is no
    /// longer the only thing large enough to matter: with
    /// [`chunked_attention`] bounding the score matrix, the biggest allocation
    /// in a VAE decode becomes the full-resolution activation tensors, so
    /// those are checked too. See the crash note on
    /// [`check_attention_budget`] for why this is a hard error.
    pub fn check_alloc_budget(bytes: Option<u64>, what: &str) -> Result<()> {
        let budget = attention_budget_bytes()?;
        let Some(bytes) = bytes else {
            return Err(Error::Msg(format!(
                "{what} overflows a 64-bit byte count; there is no machine that runs it"
            )));
        };
        if bytes > budget {
            return Err(Error::Msg(format!(
                "refusing to allocate: {what} = {} for a single call, over the {} budget, and \
                 peak use is at least double that.\n\n\
                 For a VAE decode, attention is O(n^4) in the latent edge and activations are \
                 O(n^2), so the next size up costs 16x and 4x respectively. On a Metal build \
                 these are wired GPU memory the OS cannot reclaim or swap, so overshooting \
                 physical RAM panics the machine instead of failing this process. See \
                 docs/backends.md.\n\n\
                 Use a smaller size, or raise the budget deliberately with \
                 {ATTENTION_BUDGET_ENV}=<bytes>.",
                human_bytes(bytes),
                human_bytes(budget),
            )));
        }
        Ok(())
    }

    /// Format a byte count in units a human can act on.
    pub fn human_bytes(bytes: u64) -> String {
        const UNITS: [(&str, u64); 4] = [
            ("GiB", 1 << 30),
            ("MiB", 1 << 20),
            ("KiB", 1 << 10),
            ("B", 1),
        ];
        for (suffix, scale) in UNITS {
            if bytes >= scale {
                return format!("{:.1} {suffix}", bytes as f64 / scale as f64);
            }
        }
        format!("{bytes} B")
    }

    /// Reference attention: materialises the full `seq_q x seq_k` score matrix.
    ///
    /// `q`, `k`, `v` are `[batch, heads, seq, head_dim]`. `k`/`v` may have a
    /// different sequence length than `q` (cross-attention).
    ///
    /// Kept as the correctness oracle for [`scaled_dot_product_attention`] and
    /// as the fallback wherever candle's fused kernel declines the shape.
    ///
    /// Refuses oversized calls via [`check_attention_budget`] before
    /// allocating anything. At a 64x64 latent (`seq = 4096`) the score matrix
    /// is 4096x4096 f32 — 64 MiB per call — and it is measurably the dominant
    /// cost at production resolution. See docs/backends.md.
    pub fn naive_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, heads, seq_q, dim) = q.dims4()?;
        let seq_k = k.dim(D::Minus2)?;
        check_attention_budget(batch, heads, seq_q, seq_k, q.dtype())?;

        let scale = 1f64 / (dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * scale)?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let weights = softmax_last_dim(&scores)?;
        weights.matmul(&v.contiguous()?)
    }

    /// Environment override for the per-chunk score budget, in bytes.
    pub const ATTENTION_CHUNK_ENV: &str = "SD_ATTENTION_CHUNK_BYTES";

    /// Target size for one chunk's score matrix.
    ///
    /// 64 MiB is exactly the score matrix for SD 1.5 at 512x512, so that decode
    /// — the common case — stays on the single-chunk path and pays nothing for
    /// chunking existing. Anything larger splits and stays bounded near this
    /// figure instead of growing as `n^4`.
    ///
    /// This is reasoned, not measured. Chunking cannot beat not-chunking on
    /// speed: it performs the same arithmetic with more kernel launches and a
    /// concatenation at the end, so the only question was how much it costs,
    /// and that is what sets "don't chunk until you must". An attempt to
    /// measure the cost at latent 64 on an M4 Max produced 9.1/11.8/9.6 s
    /// unchunked against 17.3/12.3/8.0 s at 8 MiB chunks — distributions that
    /// overlap entirely, on a machine too noisy to separate them. If you want a
    /// tuned value, measure on a quiet box and override with
    /// [`ATTENTION_CHUNK_ENV`]; do not trust a single run.
    pub const DEFAULT_ATTENTION_CHUNK_BYTES: u64 = 64 * 1024 * 1024;

    fn attention_chunk_bytes() -> Result<u64> {
        match std::env::var(ATTENTION_CHUNK_ENV) {
            Ok(raw) => raw.trim().parse().map_err(|_| {
                Error::Msg(format!(
                    "{ATTENTION_CHUNK_ENV} must be a byte count, got {raw:?}"
                ))
            }),
            Err(std::env::VarError::NotPresent) => Ok(DEFAULT_ATTENTION_CHUNK_BYTES),
            Err(std::env::VarError::NotUnicode(_)) => Err(Error::Msg(format!(
                "{ATTENTION_CHUNK_ENV} is not valid UTF-8"
            ))),
        }
    }

    /// How many query rows fit in one chunk's score budget.
    ///
    /// Always at least 1: a single query row is the smallest unit that can be
    /// computed, so an enormous `seq_k` produces a slow call rather than a
    /// refused one.
    pub fn attention_chunk_rows(
        batch: usize,
        heads: usize,
        seq_k: usize,
        dtype: DType,
        target_bytes: u64,
    ) -> usize {
        let per_row = attention_score_bytes(batch, heads, 1, seq_k, dtype).unwrap_or(u64::MAX);
        if per_row == 0 {
            return 1;
        }
        (target_bytes / per_row).max(1) as usize
    }

    /// Attention computed in query chunks, so the full `seq_q x seq_k` score
    /// matrix is never materialised.
    ///
    /// Each chunk sees the entire key axis, so the softmax denominator is the
    /// same one the unchunked path computes — this is exact, not an
    /// approximation, and needs no running-maximum bookkeeping. Peak score
    /// memory drops from `seq_q x seq_k` to `chunk x seq_k`.
    ///
    /// What this does *not* do is fuse softmax into the matmul. The score
    /// matrix for each chunk still round-trips through memory, so this is a
    /// memory win first and a speed win only insofar as smaller tiles cache
    /// better. Real flash attention needs a fused kernel, which candle does
    /// not expose for these shapes — see docs/backends.md.
    ///
    /// Degenerates to a single chunk (i.e. exactly [`naive_attention`]) when
    /// the whole score matrix already fits the target, so small shapes pay
    /// nothing for this path existing.
    pub fn chunked_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        chunked_attention_with_path(q, k, v, mask).map(|(t, _)| t)
    }

    fn chunked_attention_with_path(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, AttentionPath)> {
        chunked_attention_sized(q, k, v, mask, attention_chunk_bytes()?)
    }

    /// [`chunked_attention`] with an explicit chunk target.
    ///
    /// Separate from the env-reading entry point so tests can force a chunk
    /// size without mutating global state, which is what makes the
    /// chunked-vs-naive equivalence check reliable under a parallel test
    /// runner.
    pub(crate) fn chunked_attention_sized(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        target_bytes: u64,
    ) -> Result<(Tensor, AttentionPath)> {
        let (batch, heads, seq_q, dim) = q.dims4()?;
        let seq_k = k.dim(D::Minus2)?;
        let rows = attention_chunk_rows(batch, heads, seq_k, q.dtype(), target_bytes);
        if rows >= seq_q {
            // One chunk is the whole thing; take the simple path unchanged.
            return Ok((naive_attention(q, k, v, mask)?, AttentionPath::Naive));
        }
        // Bound what a single chunk allocates. `rows` is derived from the
        // target, so this only trips if one query row is itself over budget.
        check_attention_budget(batch, heads, rows, seq_k, q.dtype())?;

        let scale = 1f64 / (dim as f64).sqrt();
        let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let v = v.contiguous()?;
        // A mask may be broadcast over the query axis (`causal_mask` is not —
        // it carries a real row per query, so it has to be sliced alongside).
        let mask_rows = mask.map(|m| m.dim(D::Minus2)).transpose()?;

        let mut out = Vec::with_capacity(seq_q.div_ceil(rows));
        let mut start = 0;
        while start < seq_q {
            let len = rows.min(seq_q - start);
            let qc = q.narrow(D::Minus2, start, len)?.contiguous()?;
            let scores = (qc.matmul(&kt)? * scale)?;
            let scores = match (mask, mask_rows) {
                (Some(m), Some(1)) => scores.broadcast_add(m)?,
                (Some(m), Some(_)) => {
                    scores.broadcast_add(&m.narrow(D::Minus2, start, len)?.contiguous()?)?
                }
                _ => scores,
            };
            out.push(softmax_last_dim(&scores)?.matmul(&v)?);
            start += len;
        }
        Ok((Tensor::cat(&out, D::Minus2)?, AttentionPath::Chunked))
    }

    /// Attention, plus which implementation served it.
    ///
    /// candle 0.11 implements SDPA for Metal only — its `cpu_fwd` bails
    /// outright — so on any other device we go straight to
    /// [`naive_attention`] rather than paying for a guaranteed failure.
    ///
    /// Even on Metal the fused kernel declines shapes, and as of candle 0.11
    /// it declines every shape in this workspace: f32 at `head_dim = 512` (the
    /// VAE attention block) is explicitly excluded, and a mask must be
    /// `[batch, heads, seq_q, seq_k]` while [`causal_mask`] is `[1, 1, s, s]`.
    /// So [`AttentionPath::Fused`] is currently unreachable here. Treat it as
    /// an optimisation that may arrive, not one already banked.
    ///
    /// Everything else goes to [`chunked_attention`], which reports `Chunked`
    /// when it actually splits and `Naive` when one chunk already covers the
    /// query axis. A declined fused shape therefore falls back to a path whose
    /// allocation is bounded by construction; it cannot quietly turn into the
    /// allocation that wedges the machine.
    pub fn attention_with_path(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, AttentionPath)> {
        if q.device().is_metal() {
            let dim = q.dim(D::Minus1)?;
            let scale = (1f64 / (dim as f64).sqrt()) as f32;
            let qc = q.contiguous()?;
            let kc = k.contiguous()?;
            let vc = v.contiguous()?;
            // `softcapping = 1.0` disables the tanh softcap path.
            if let Ok(t) = candle_nn::ops::sdpa(&qc, &kc, &vc, mask, false, scale, 1.0) {
                return Ok((t, AttentionPath::Fused));
            }
        }
        chunked_attention_with_path(q, k, v, mask)
    }

    /// Scaled dot-product attention, unmasked.
    ///
    /// `q`, `k`, `v` are `[batch, heads, seq, head_dim]`. `k`/`v` may have a
    /// different sequence length than `q` (cross-attention).
    pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        attention_with_path(q, k, v, None).map(|(t, _)| t)
    }

    /// Scaled dot-product attention with an additive mask.
    ///
    /// `mask` is added to the scores before softmax, so masked positions hold
    /// a large negative value (`f32::NEG_INFINITY`) and visible positions
    /// `0.0`. Build one with [`causal_mask`].
    pub fn scaled_dot_product_attention_masked(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        attention_with_path(q, k, v, Some(mask)).map(|(t, _)| t)
    }

    /// Additive causal mask of shape `[1, 1, seq, seq]`.
    ///
    /// Position `(i, j)` is `0.0` when `j <= i` and `f32::NEG_INFINITY`
    /// otherwise, ready to pass to
    /// [`scaled_dot_product_attention_masked`].
    ///
    /// The leading `1, 1` broadcasts over batch and heads, which the naive
    /// path handles. candle's fused kernel instead requires a mask materialised
    /// to `[batch, heads, seq_q, seq_k]`, so passing this one keeps us on the
    /// naive path — see [`attention_with_path`].
    pub fn causal_mask(seq: usize, device: &super::Device) -> Result<Tensor> {
        let mut data = Vec::with_capacity(seq * seq);
        for i in 0..seq {
            for j in 0..seq {
                data.push(if j <= i { 0f32 } else { f32::NEG_INFINITY });
            }
        }
        Tensor::from_vec(data, (1, 1, seq, seq), device)
    }
}

pub mod sysmem;

/// Device selection.
pub mod device {
    use super::{Device, Result};

    /// Pick the best available accelerator, falling back to CPU.
    ///
    /// Honours the enabled cargo features; a build without `cuda` or `metal`
    /// always returns CPU.
    pub fn best() -> Result<Device> {
        #[cfg(feature = "cuda")]
        if let Ok(d) = Device::new_cuda(0) {
            return Ok(d);
        }
        #[cfg(feature = "metal")]
        if let Ok(d) = Device::new_metal(0) {
            return Ok(d);
        }
        Ok(Device::Cpu)
    }

    /// Always CPU. Use this for golden tests: correctness first, one variable
    /// at a time. Debugging a wrong kernel and a wrong architecture
    /// simultaneously is how ports stall.
    pub fn cpu() -> Device {
        Device::Cpu
    }
}

/// Deterministic, device-independent random noise.
///
/// candle's `Device::set_seed` does not work on CPU (it errors with "cannot
/// seed the CPU rng"), and its GPU RNG would not match CPU output anyway. Both
/// make `--seed 42` mean different things on different machines, which is not
/// acceptable for a tool whose output people share and reproduce.
///
/// So we generate noise ourselves and upload it. Same seed produces bit-
/// identical *noise* on every device and every candle version. It costs one
/// host-to-device copy per image, which is nothing next to a denoise loop.
///
/// # What this does and does not guarantee
///
/// The noise is bit-identical across devices. The **image is not**, because
/// the arithmetic between them is not: f32 reduction order differs per
/// backend, and twenty sequential UNet evaluations compound the difference.
///
/// Measured on 2026-07-26, same seed and prompt at 256x256, CPU against
/// Metal: mean absolute difference 0.9/255, max 35/255, and only 27% of
/// pixels exactly equal. The two images are indistinguishable by eye and not
/// interchangeable byte-for-byte.
///
/// So: same seed on the same device and build reproduces exactly — that is
/// what `--seed` is for. Across devices it reproduces the *picture*, not the
/// file. Do not build a cache key or a regression test on cross-device
/// byte equality.
///
/// This deliberately does *not* try to match PyTorch's `randn`. Matching torch
/// bit-for-bit is a separate problem and not worth solving to make our own
/// output reproducible.
pub mod rng {
    use super::{DType, Device, Result, Tensor};

    /// splitmix64 — small, fast, and good enough for sampling noise.
    #[derive(Debug, Clone)]
    pub struct SeededRng {
        state: u64,
    }

    impl SeededRng {
        pub fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in `(0, 1]`. Never returns 0, so `ln()` below is safe.
        fn next_f64(&mut self) -> f64 {
            // 53 significant bits, shifted off zero.
            let bits = self.next_u64() >> 11;
            (bits as f64 + 1.0) / (9007199254740992.0 + 1.0)
        }

        /// Standard normal values via Box-Muller.
        pub fn normals(&mut self, n: usize) -> Vec<f32> {
            let mut out = Vec::with_capacity(n);
            while out.len() < n {
                let u1 = self.next_f64();
                let u2 = self.next_f64();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = std::f64::consts::TAU * u2;
                out.push((r * theta.cos()) as f32);
                if out.len() < n {
                    out.push((r * theta.sin()) as f32);
                }
            }
            out
        }

        /// A tensor of standard normal noise on `device`.
        pub fn randn<S: Into<super::Shape>>(
            &mut self,
            shape: S,
            device: &Device,
        ) -> Result<Tensor> {
            let shape = shape.into();
            let data = self.normals(shape.elem_count());
            Tensor::from_vec(data, shape, device)?.to_dtype(DType::F32)
        }
    }
}

/// Assertions for the golden-tensor harness.
pub mod testing {
    use super::{DType, Result, Tensor};

    /// Maximum absolute difference between two tensors.
    pub fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f64> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?;
        let d = (a - b)?.abs()?.max(0)?;
        d.to_scalar::<f32>().map(|v| v as f64)
    }

    /// Report describing how far two tensors are apart.
    #[derive(Debug, Clone)]
    pub struct Closeness {
        pub max_abs: f64,
        pub mean_abs: f64,
        pub shape_a: Vec<usize>,
        pub shape_b: Vec<usize>,
    }

    impl std::fmt::Display for Closeness {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "shapes {:?} vs {:?}, max_abs={:.3e}, mean_abs={:.3e}",
                self.shape_a, self.shape_b, self.max_abs, self.mean_abs
            )
        }
    }

    /// Compare two tensors elementwise.
    ///
    /// Returns `Err` if the shapes differ, otherwise a [`Closeness`] report.
    pub fn closeness(a: &Tensor, b: &Tensor) -> Result<Closeness> {
        let shape_a = a.dims().to_vec();
        let shape_b = b.dims().to_vec();
        if shape_a != shape_b {
            return Ok(Closeness {
                max_abs: f64::INFINITY,
                mean_abs: f64::INFINITY,
                shape_a,
                shape_b,
            });
        }
        let af = a.to_dtype(DType::F32)?.flatten_all()?;
        let bf = b.to_dtype(DType::F32)?.flatten_all()?;
        let diff = (af - bf)?.abs()?;
        let max_abs = diff.max(0)?.to_scalar::<f32>()? as f64;
        let mean_abs = diff.mean(0)?.to_scalar::<f32>()? as f64;
        Ok(Closeness {
            max_abs,
            mean_abs,
            shape_a,
            shape_b,
        })
    }

    /// Default tolerance for f16-origin weights run in f32.
    ///
    /// Tighter than this and you chase phantom failures from accumulation
    /// order; looser and real bugs slip through.
    pub const DEFAULT_ATOL: f64 = 1e-4;
    pub const DEFAULT_RTOL: f64 = 1e-3;

    /// Panic with a useful message unless `a` and `b` agree within `atol`.
    pub fn assert_close(a: &Tensor, b: &Tensor, atol: f64, what: &str) -> Result<()> {
        let c = closeness(a, b)?;
        assert!(
            c.max_abs <= atol,
            "{what}: tensors diverge beyond atol={atol:.3e}\n  {c}\n\
             Hint: check axis order and parameter naming before suspecting the kernel."
        );
        Ok(())
    }
}

/// The attention memory budget that keeps an oversized decode from wedging the
/// machine.
///
/// These live in the library, not in the benchmark that first triggered the
/// crash, for two reasons: every caller allocates through
/// [`ops::naive_attention`], and unit tests in an `examples/` target are not
/// run by `cargo test` — a guard tested only there is a guard with no
/// regression cover at all.
#[cfg(test)]
mod attention_budget_tests {
    use super::ops::*;
    use super::{DType, Device, Tensor};

    /// One square f32 score matrix for a VAE decode at latent edge `n`.
    fn latent_bytes(n: usize) -> Option<u64> {
        let seq = n.checked_mul(n)?;
        attention_score_bytes(1, 1, seq, seq, DType::F32)
    }

    #[test]
    fn score_matrix_grows_as_the_fourth_power_of_the_latent_edge() {
        // Quadrupling the latent edge is a 256x memory cost, which is exactly
        // why eyeballing "one size up" is not safe here.
        assert_eq!(latent_bytes(16), Some(256 * 1024));
        assert_eq!(latent_bytes(64), Some(64 * 1024 * 1024));
        assert_eq!(latent_bytes(128), Some(1024 * 1024 * 1024));
        assert_eq!(latent_bytes(256), Some(16 * 1024 * 1024 * 1024));
        assert_eq!(latent_bytes(384), Some(86_973_087_744));
    }

    #[test]
    fn batch_and_heads_multiply_the_cost() {
        // The score matrix is [batch, heads, seq_q, seq_k]. Costing a single
        // head and calling it the total understates a real UNet call by an
        // order of magnitude.
        let one = attention_score_bytes(1, 1, 4096, 4096, DType::F32).unwrap();
        assert_eq!(
            attention_score_bytes(2, 8, 4096, 4096, DType::F32),
            Some(one * 16)
        );
        // Dtype counts too: f16 halves it.
        assert_eq!(
            attention_score_bytes(1, 1, 4096, 4096, DType::F16),
            Some(one / 2)
        );
    }

    #[test]
    fn the_documented_bench_sweep_still_runs() {
        for n in [16usize, 32, 64] {
            let seq = n * n;
            assert!(
                check_attention_budget(1, 1, seq, seq, DType::F32).is_ok(),
                "docs/backends.md benchmarks latent {n}; the budget must not block it"
            );
        }
    }

    #[test]
    fn sdxl_at_1024_is_not_collateral_damage() {
        // A 128 latent is 1 GiB — real, supported work. A budget that refuses
        // it would be protecting the machine by breaking the product.
        assert!(check_attention_budget(1, 1, 128 * 128, 128 * 128, DType::F32).is_ok());
    }

    #[test]
    fn the_run_that_panicked_the_machine_is_refused() {
        // 2026-07-25: a decode at a 384 latent on a 36 GiB Mac projected
        // 81 GiB of wired Metal memory and took the kernel down with a
        // watchdog timeout.
        let seq = 384 * 384;
        let err = check_attention_budget(1, 1, seq, seq, DType::F32)
            .expect_err("384 must not be runnable under the default budget");
        let msg = err.to_string();
        assert!(
            msg.contains("81.0 GiB"),
            "should state the real cost: {msg}"
        );
        assert!(
            msg.contains(ATTENTION_BUDGET_ENV),
            "should name the override: {msg}"
        );
    }

    #[test]
    fn the_first_size_over_budget_is_refused() {
        // 128 is 1 GiB and allowed; 192 is 5 GiB and is not. The boundary
        // matters more than the extremes — it is the step someone actually
        // takes next after a successful run.
        assert!(check_attention_budget(1, 1, 128 * 128, 128 * 128, DType::F32).is_ok());
        assert!(check_attention_budget(1, 1, 192 * 192, 192 * 192, DType::F32).is_err());
    }

    #[test]
    fn overflow_is_refused_rather_than_wrapped() {
        // Wrapping would turn an impossible shape into a small byte count and
        // wave it straight through the budget check.
        assert_eq!(
            attention_score_bytes(1, 1, usize::MAX, usize::MAX, DType::F32),
            None
        );
        let seq = (1usize << 20) * (1usize << 20);
        assert!(check_attention_budget(1, 1, seq, seq, DType::F32).is_err());
    }

    #[test]
    fn the_budget_override_is_explicit_and_validated() {
        assert_eq!(
            parse_attention_budget(None).unwrap(),
            DEFAULT_ATTENTION_BUDGET_BYTES
        );
        assert_eq!(parse_attention_budget(Some(" 4096 ")).unwrap(), 4096);
        assert!(parse_attention_budget(Some("2GiB")).is_err());
        assert!(parse_attention_budget(Some("-1")).is_err());
    }

    #[test]
    fn sizes_are_reported_in_units_a_human_can_act_on() {
        assert_eq!(human_bytes(86_973_087_744), "81.0 GiB");
        assert_eq!(human_bytes(64 * 1024 * 1024), "64.0 MiB");
        // Zero is below every scale and falls through to the plain form.
        assert_eq!(human_bytes(0), "0 B");
    }

    #[test]
    fn the_unchunked_path_still_refuses_an_oversized_call() -> super::Result<()> {
        // The inputs here are ~1 MB each; the score matrix they imply is
        // 4.4 GiB, just over budget. The dispatcher now *serves* this shape by
        // splitting it, which is the point of chunking — but `naive_attention`
        // is the one that would allocate the matrix whole, and it must still
        // refuse before the matmul rather than after. If the guard were
        // missing or ran too late, this test would allocate 4.4 GiB.
        let dev = Device::Cpu;
        // sqrt(4 GiB / 4 bytes) is 32768, so this is the first round number
        // past it. Tied to DEFAULT_ATTENTION_BUDGET_BYTES: if that changes,
        // this has to move with it.
        let seq = 33_000;
        let q = Tensor::zeros((1, 1, seq, 8), DType::F32, &dev)?;
        let err = naive_attention(&q, &q, &q, None)
            .expect_err("a 4.4 GiB score matrix is over the default budget");
        assert!(
            err.to_string().contains("refusing to allocate"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    #[test]
    fn chunking_agrees_with_the_unchunked_reference() -> super::Result<()> {
        // Each chunk sees the whole key axis, so this should be exact rather
        // than merely close. The tolerance is for matmul batching differences,
        // not for an approximation — a running-softmax bug would blow past it.
        let dev = Device::Cpu;
        for (b, h, sq, sk, d) in [(1usize, 1, 64usize, 64usize, 8usize), (2, 3, 40, 57, 16)] {
            let q = Tensor::randn(0f32, 1f32, (b, h, sq, d), &dev)?;
            let k = Tensor::randn(0f32, 1f32, (b, h, sk, d), &dev)?;
            let v = Tensor::randn(0f32, 1f32, (b, h, sk, d), &dev)?;

            // A target of 1 byte forces one query row per chunk — the most
            // fragmented arrangement, and the one most likely to expose a
            // slicing or concatenation error.
            let (got, path) = chunked_attention_sized(&q, &k, &v, None, 1)?;
            assert_eq!(path, AttentionPath::Chunked);
            let want = naive_attention(&q, &k, &v, None)?;
            assert_eq!(got.dims(), want.dims());
            super::testing::assert_close(&got, &want, 1e-6, "chunked vs naive")?;
        }
        Ok(())
    }

    #[test]
    fn chunking_slices_a_causal_mask_alongside_the_queries() -> super::Result<()> {
        // The mask carries one row per query, so a chunk covering queries
        // [i..j] must take mask rows [i..j]. Reusing row 0 for every chunk
        // would still produce plausible numbers, so compare against the
        // unchunked result rather than eyeballing shapes.
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 48usize, 16usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?;
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?;
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?;
        let mask = causal_mask(s, &dev)?;

        let (got, path) = chunked_attention_sized(&q, &k, &v, Some(&mask), 1)?;
        assert_eq!(path, AttentionPath::Chunked);
        let want = naive_attention(&q, &k, &v, Some(&mask))?;
        super::testing::assert_close(&got, &want, 1e-6, "chunked masked vs naive")?;
        Ok(())
    }

    #[test]
    fn a_shape_that_already_fits_is_not_chunked() -> super::Result<()> {
        // Chunking a small call would add kernel launches and a concat for
        // nothing. SD 1.5 at 512x512 must stay on the single-chunk path.
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 1, 64, 8), DType::F32, &dev)?;
        let (_, path) = chunked_attention_sized(&q, &q, &q, None, DEFAULT_ATTENTION_CHUNK_BYTES)?;
        assert_eq!(path, AttentionPath::Naive);
        Ok(())
    }

    #[test]
    fn chunk_rows_track_the_target_and_never_reach_zero() {
        // 4096 keys of f32, single head: 16 KiB per query row.
        assert_eq!(
            attention_chunk_rows(1, 1, 4096, DType::F32, 8 * 1024 * 1024),
            512
        );
        // Heads and batch divide the row count, because they multiply the row.
        assert_eq!(
            attention_chunk_rows(2, 8, 4096, DType::F32, 8 * 1024 * 1024),
            32
        );
        // A row bigger than the whole target still yields one row, so an
        // enormous key axis produces a slow call rather than a division by
        // zero or an empty chunk loop.
        assert_eq!(attention_chunk_rows(1, 1, usize::MAX, DType::F32, 1), 1);
        assert_eq!(attention_chunk_rows(1, 1, 4096, DType::F32, 0), 1);
    }

    #[test]
    fn chunking_makes_a_previously_refused_size_allocatable() {
        // The whole point. A 384 latent needs an 81 GiB score matrix in one
        // piece, which is refused; in 8 MiB chunks the peak allocation is the
        // chunk, which is not.
        let seq = 384 * 384;
        assert!(check_attention_budget(1, 1, seq, seq, DType::F32).is_err());
        let rows = attention_chunk_rows(1, 1, seq, DType::F32, DEFAULT_ATTENTION_CHUNK_BYTES);
        assert!(check_attention_budget(1, 1, rows, seq, DType::F32).is_ok());
    }

    #[test]
    fn attention_reports_the_naive_path_off_metal() -> super::Result<()> {
        // candle 0.11's SDPA is Metal-only, so on a CPU test runner the fused
        // path is unreachable. Asserting this is what stops a "fused agrees
        // with naive" test from quietly becoming naive-agrees-with-itself.
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 2, 8, 32), DType::F32, &dev)?;
        let (_, path) = attention_with_path(&q, &q, &q, None)?;
        assert_eq!(path, AttentionPath::Naive);
        Ok(())
    }
}
