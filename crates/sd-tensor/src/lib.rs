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
    safetensors, CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Device, Error, IndexOp,
    Layout, Module, Result, Shape, Tensor, D,
};
pub use candle_nn::VarBuilder;

/// Memory refusals — the guard declining, as opposed to something being wrong.
///
/// The distinction is load-bearing: "this machine is busy" and "this model is
/// broken" are different answers, and the GPU smoke test skips on the first
/// and fails on the second.
///
/// It cannot be an [`Error`] variant, because `Error` is candle's and this
/// workspace does not fork it. So the marker lives here, next to the only two
/// places that produce one, with a test binding the constructor to the
/// predicate. **The point is that it is in one place**: before this, four call
/// sites in three crates each matched a substring of a message defined
/// somewhere else, and editing that message would have quietly turned every
/// memory skip into a hard failure.
pub mod refusal {
    use super::Error;

    /// Every refusal message begins with this.
    pub const MARKER: &str = "refusing to";

    /// Build a refusal. `detail` continues the sentence — `refuse("allocate:
    /// ...")` reads "refusing to allocate: ...".
    pub fn refuse(detail: impl std::fmt::Display) -> Error {
        Error::Msg(format!("{MARKER} {detail}"))
    }

    /// Whether an error is the memory guard declining.
    ///
    /// Matches anywhere in the chain's display, because a refusal is usually
    /// seen after being wrapped by a caller's own error type.
    pub fn is_refusal(e: &Error) -> bool {
        e.to_string().contains(MARKER)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_refusal_is_recognised_as_one() {
            assert!(is_refusal(&refuse("start: needs 9 GB")));
            assert!(!is_refusal(&Error::Msg("shape mismatch".into())));
        }
    }
}

/// Read one tensor's shape from a safetensors file without loading any data.
///
/// For deciding *which* model a checkpoint is before building it. The
/// alternative is parsing `config.json`, which means a JSON dependency and
/// trusting a file that need not be present — a `.safetensors` always carries
/// its own shapes, and they are the thing that has to match.
///
/// `Ok(None)` means the file is readable and has no such tensor, which is the
/// answer a caller probing for an architecture wants; only an unreadable or
/// malformed file is an error.
pub fn tensor_shape(path: &std::path::Path, name: &str) -> Result<Option<Vec<usize>>> {
    // Safety: candle's own loader mmaps these files the same way.
    let mapped = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
    Ok(mapped
        .tensors()
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, view)| view.shape().to_vec()))
}

/// Layers we build models out of. Re-exported so model crates never name candle.
pub mod nn {
    pub use candle_nn::{
        embedding, layer_norm, linear, linear_no_bias, Conv2dConfig, Embedding, LayerNorm,
        LayerNormConfig, Linear, VarBuilder, VarMap,
    };
    // Shadowing candle's, so every model picks up circular padding without
    // changing. See `crate::conv`.
    pub use crate::conv::{conv2d, conv2d_no_bias, Conv2d};
    // Also shadowing candle's, for the same reason in a different direction:
    // candle's `GroupNorm` is a composition of about ten ops running at 1.7 to
    // 5.8 GB/s, and it is 23.5% of an SD 1.5 step. Same constructor, same
    // weight names; see `crate::fused`.
    pub use crate::fused::{group_norm, GroupNorm};
}

/// Convolution with optional circular padding, for seamless tiling.
pub mod conv;
pub mod fused;
/// The MLX backend. See the module docs and docs/handoff.md.
#[cfg(feature = "mlx")]
pub mod mlx;
/// A GGUF reader that does not go through candle. See the module docs.
#[cfg(feature = "mlx")]
pub mod mlx_gguf;
#[cfg(feature = "metal")]
pub mod mps;

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

    /// RMS normalisation: `xs / sqrt(mean(xs^2) + eps) * alpha`.
    ///
    /// One copy for T5, Flux and SD 3, which had three identical hand-written
    /// ones. **Deliberately not `candle_nn::ops::rms_norm`**, which is the
    /// obvious thing to reach for and is worse on both axes that matter.
    ///
    /// candle's fused kernel sums the row with a plain sequential
    /// `.sum::<f32>()`, where `mean_keepdim` reduces in blocks. Sequential
    /// error grows with row length, and these rows are long — T5's `d_model`
    /// is 4096. Measured against an f64 reference:
    ///
    /// ```text
    ///   shape [1, 154, 4096]        max abs error    relative
    ///     this implementation        9.695e-7        7.9e-8
    ///     candle_nn fused            9.627e-6        7.9e-7     10x worse
    ///   shape [1, 77, 768]
    ///     this implementation        1.258e-6        9.4e-8
    ///     candle_nn fused            4.775e-6        3.6e-7     3.8x worse
    /// ```
    ///
    /// That is not academic: swapping T5 to the fused kernel moved
    /// `golden_t5` from passing to 3.891e-3, past a 3e-3 bound that was itself
    /// set by measuring the reference's own f32-vs-f64 spread.
    ///
    /// And the speed it buys back is small and not even one-directional —
    /// 2.1x faster at `[1, 154, 4096]`, 2.7x *slower* at `[1, 77, 768]` — on
    /// an op that is a rounding error of any real run. There was no trade to
    /// make.
    ///
    /// The f32 normalisation is required rather than tidy: T5's activations
    /// reach ~200,000 against f16's 65,504 ceiling, so the reciprocal has to
    /// be formed in f32 or the row silently becomes zero. `alpha` is applied
    /// in the input dtype, matching what the three copies did and what
    /// transformers does.
    pub fn rms_norm(xs: &Tensor, alpha: &Tensor, eps: f64) -> Result<Tensor> {
        if prefer_fused_norm(xs) && alpha.dtype() == DType::F32 {
            return candle_nn::ops::rms_norm(xs, alpha, eps as f32);
        }
        let dtype = xs.dtype();
        let xs32 = xs.to_dtype(DType::F32)?;
        let rrms = (xs32.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
        xs32.broadcast_div(&rrms)?
            .to_dtype(dtype)?
            .broadcast_mul(&alpha.to_dtype(dtype)?)
    }

    /// LayerNorm with no learned scale or shift: `(x - mean) / sqrt(var + eps)`.
    ///
    /// Flux and SD 3 both need this and had a byte-identical copy each. It is
    /// not what `candle_nn::layer_norm` gives you: that always reads a
    /// `weight`, even told `affine: false` — the flag only drops the bias — so
    /// it cannot express a norm with no parameters at all. In these models the
    /// scale and shift arrive from the modulation vector instead, which *is*
    /// the conditioning mechanism, so a norm that quietly applied a learned
    /// weight would be conditioning the model twice.
    ///
    /// `candle_nn::ops::layer_norm` is the fused kernel and is not used here,
    /// for the reason given on [`rms_norm`]: its sibling sums each row
    /// sequentially where `mean_keepdim` reduces in blocks, and these rows are
    /// 3072 wide. The same measurement should be made before adopting it.
    ///
    /// Normalised in f32 whatever comes in, then narrowed back.
    pub fn plain_layer_norm(xs: &Tensor, eps: f64) -> Result<Tensor> {
        if prefer_fused_norm(xs) {
            // The fused kernel wants explicit affine parameters where this
            // form has none, so it gets a unit scale and a zero shift.
            // Allocating those two `[width]` vectors per call is measurable and
            // small: the path still wins 3.2x at 1536 tokens and 4.8x at 4608.
            let width = xs.dim(D::Minus1)?;
            let ones = Tensor::ones(width, DType::F32, xs.device())?;
            let zeros = Tensor::zeros(width, DType::F32, xs.device())?;
            return candle_nn::ops::layer_norm(xs, &ones, &zeros, eps as f32);
        }
        let dtype = xs.dtype();
        let xs32 = xs.to_dtype(DType::F32)?;
        let mean = xs32.mean_keepdim(D::Minus1)?;
        let centred = xs32.broadcast_sub(&mean)?;
        let var = centred.sqr()?.mean_keepdim(D::Minus1)?;
        centred
            .broadcast_div(&(var + eps)?.sqrt()?)?
            .to_dtype(dtype)
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
        /// candle's CPU flash kernel — see [`flash_attention_cpu`]. Also never
        /// materialises the score matrix.
        FlashCpu,
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
            return Err(crate::refusal::refuse(format!(
                "allocate: {what} = {} for a single call, over the {} budget, and \
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

    /// Whether candle's CPU flash kernel can serve this call.
    ///
    /// Checked up front rather than discovered from an `Err`, because the
    /// kernel reads `q`/`k` through raw strides with no fallback for a
    /// non-unit last stride: a shape it cannot handle is not always a shape it
    /// refuses. Everything below is a real constraint of the kernel, sourced
    /// from `candle_nn::attention::cpu_flash`:
    ///
    /// - **f32 only.** The kernel accumulates in f32 and always returns f32,
    ///   so any other input dtype would silently change the output's. That
    ///   costs nothing here: CPU activations are f32 throughout — see the F16
    ///   note in docs/roadmap.md.
    /// - **Matching head counts.** It supports grouped-query attention by
    ///   `heads / kv_heads`, which is integer division: an unequal split would
    ///   round rather than fail. No model here uses GQA.
    /// - **Matching head dims.** It takes `head_dim` from `q` and uses it to
    ///   index `v`, so a `v` with a different width reads out of its row.
    /// - **A mask whose last two axes are exactly `[seq_q, seq_k]`, with every
    ///   leading axis 1.** The kernel indexes the mask flat as
    ///   `q_pos * seq_k + kv_pos`, taking `seq_k` from `k` and never looking
    ///   at the mask's own shape; [`causal_mask`]'s `[1, 1, s, s]` flattens to
    ///   exactly that. This is checked per axis rather than by element count
    ///   on purpose: `seq_q * seq_k` is symmetric, so a transposed mask has
    ///   the right count and the wrong layout, and would be read
    ///   row-for-column without erroring. A mask that varies per batch element
    ///   or per head is rejected for the same reason — T5's
    ///   `[batch, heads, n, n]` relative-position bias is the live example.
    pub fn flash_cpu_supported(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> bool {
        let (Ok((b, h, seq_q, d)), Ok(kd), Ok(vd)) = (q.dims4(), k.dims4(), v.dims4()) else {
            return false;
        };
        // The kernel reads CPU storage directly and errors on anything else.
        if !q.device().is_cpu() || !k.device().is_cpu() || !v.device().is_cpu() {
            return false;
        }
        if q.dtype() != DType::F32 || k.dtype() != DType::F32 || v.dtype() != DType::F32 {
            return false;
        }
        // (batch, heads, head_dim) must agree; only the key axis may differ,
        // which is what makes cross-attention work.
        if (kd.0, kd.1, kd.3) != (b, h, d) || (vd.0, vd.1, vd.3) != (b, h, d) || kd.2 != vd.2 {
            return false;
        }
        match mask {
            None => true,
            Some(m) => {
                if m.dtype() != DType::F32 {
                    return false;
                }
                // Check the axes, not the element count. `seq_q * seq_k` is
                // symmetric, so a transposed `[1, 1, seq_k, seq_q]` mask has
                // exactly the count a correct one does — it would pass a
                // count check and then be read row-for-column, silently.
                let dims = m.dims();
                let (Some((&last, rest)), true) = (dims.split_last(), dims.len() >= 2) else {
                    return false;
                };
                let (&next_to_last, leading) = rest
                    .split_last()
                    .expect("len >= 2 leaves at least one dim after split_last");
                last == kd.2 && next_to_last == seq_q && leading.iter().all(|&d| d == 1)
            }
        }
    }

    /// Environment override for [`DEFAULT_FLASH_CPU_MAX_SEQ`], in tokens.
    ///
    /// `0` disables the CPU flash path entirely, which is the quickest way to
    /// tell whether a numerical difference came from it.
    pub const FLASH_CPU_MAX_SEQ_ENV: &str = "SD_FLASH_CPU_MAX_SEQ";

    /// Longest sequence for which the CPU flash kernel beats [`chunked_attention`].
    ///
    /// **Flash attention on CPU is not a uniform win, and this constant is
    /// where that fact lives.** candle's kernel streams one output row at a
    /// time, so it never materialises the score matrix — but it also gets no
    /// register blocking across query rows, and it re-reads the whole key axis
    /// per row. Against that, [`naive_attention`]'s two matmuls run through a
    /// tuned gemm. Which wins is a question about sequence length: short
    /// sequences make the gemms too small to amortise their blocking, long
    /// ones let the gemm pull ahead of the streaming loop.
    ///
    /// 512 is where those cross, measured on an M4 Max (16 cores) over
    /// `head_dim` in {40, 64, 80, 128, 160}, `heads` in {8, 12, 20, 24, 64} and
    /// batch in {1, 2}. Flash is faster at every configuration at or below it,
    /// and slower above it at all but one:
    ///
    /// ```text
    ///   seq_q     64    128    256    512    768   1024   4096
    ///   speedup  2.9x   5.5x   2.4x   1.2x   1.0x   0.9x   1.0x   (h=24, d=128, b=1)
    ///            1.5x   4.6x   2.4x   1.1x   0.7x   0.6x   0.5x   (h=8,  d=40,  b=1)
    /// ```
    ///
    /// The top row is not monotone, and the exception is real rather than
    /// noise: at `head_dim = 128` flash sags to 0.85x around 1024 and then
    /// climbs back, reaching 1.15-1.24x at Flux's 4608. Two mechanisms are
    /// competing — gemm blocking, which favours long sequences, and score
    /// matrix traffic, which eventually punishes them. Capturing that second
    /// crossing would take a rule with two disjoint intervals fitted to one
    /// `head_dim` on one machine, so this does not try; the shape it would
    /// win, Flux at 1024 on CPU, is not one anyone runs.
    ///
    /// Which real shapes the limit covers: CLIP at 77 tokens, and SD 1.5 and
    /// SDXL's UNet blocks at the 16x16 and 8x8 levels. Which it excludes:
    /// SD 3.5 (1178) and Flux (1536+) joint attention, and the UNet's two
    /// largest levels — the shapes that dominate a denoise step, and where the
    /// streaming kernel loses by up to 2x.
    ///
    /// **T5 is not on that list, despite its 154 tokens sitting well under the
    /// limit.** Its relative-position bias is a full `[batch, heads, n, n]`
    /// tensor, and the kernel indexes a mask flat with no head axis, so
    /// [`flash_cpu_supported`] refuses it. `--example attention_path` times
    /// T5's shape *unmasked* and reports 5-8x, which is not a speedup anything
    /// here can collect; the dispatch is pinned by
    /// `the_real_text_encoder_shapes_take_the_paths_we_think_they_do` in
    /// sd-models' `api_contract` so the two cannot drift apart again.
    ///
    /// **That distribution caps what this is worth end to end, to roughly
    /// nothing.** For SD 1.5 the eligible calls total about 0.4 s of a
    /// generation; for SD 3.5, whose only eligible attention is CLIP-L and
    /// CLIP-G, about 13 ms. Neither separates from noise: SD 1.5 at 512x512,
    /// 20 steps ran 113.3 s with this path and 114.4 s without, and four
    /// alternating SD 3.5 runs gave 245.2 / 216.2 / 228.7 / 230.4 s — a spread
    /// within one configuration wider than the gap between them. The shapes it
    /// wins are real; they are not where the time goes. Keep it anyway: it is
    /// free, it is verified against the naive path, and it removes the score
    /// matrix from the calls it does serve.
    ///
    /// Reproduce with `--example attention_path`, which prints both paths side
    /// by side. Read the noise note there before trusting a single run: an
    /// earlier version of that benchmark reported figures 10x apart on
    /// back-to-back runs of the same binary.
    pub const DEFAULT_FLASH_CPU_MAX_SEQ: usize = 512;

    /// The active sequence limit, honouring [`FLASH_CPU_MAX_SEQ_ENV`].
    pub fn flash_cpu_max_seq() -> Result<usize> {
        match std::env::var(FLASH_CPU_MAX_SEQ_ENV) {
            Ok(raw) => raw.trim().parse().map_err(|_| {
                Error::Msg(format!(
                    "{FLASH_CPU_MAX_SEQ_ENV} must be a token count, got {raw:?}"
                ))
            }),
            Err(std::env::VarError::NotPresent) => Ok(DEFAULT_FLASH_CPU_MAX_SEQ),
            Err(std::env::VarError::NotUnicode(_)) => Err(Error::Msg(format!(
                "{FLASH_CPU_MAX_SEQ_ENV} is not valid UTF-8"
            ))),
        }
    }

    /// Whether the CPU flash path is both able to serve this call and faster
    /// than [`chunked_attention`] at it.
    ///
    /// Both axes are bounded, not just `seq_q`. The mechanism is about `seq_q`
    /// — it sets how much work each gemm gets — and a long query axis against
    /// a short key axis was measured to lose (0.83x at `seq_q = 4096`,
    /// `seq_k = 77`). The reverse, a short query axis against a long key axis,
    /// occurs in none of the five architectures here and so was never
    /// measured; bounding both keeps the untested case on the tested path.
    pub fn flash_cpu_preferred(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<bool> {
        if !flash_cpu_supported(q, k, v, mask) {
            return Ok(false);
        }
        let limit = flash_cpu_max_seq()?;
        Ok(q.dim(D::Minus2)?.max(k.dim(D::Minus2)?) <= limit)
    }

    /// Attention through candle's CPU flash kernel.
    ///
    /// Like the Metal fused path, the `seq_q x seq_k` score matrix is never
    /// materialised: each output row streams the key axis under a running
    /// softmax maximum. So there is no [`check_attention_budget`] call here —
    /// there is no score matrix to bound, and peak memory is the output, which
    /// is the size of `v`.
    ///
    /// Call [`flash_cpu_supported`] first; this returns an error, or worse a
    /// wrong answer, on a shape the kernel does not handle. Use
    /// [`flash_cpu_preferred`] to decide whether it is also the *fast* choice —
    /// this function does not consult [`DEFAULT_FLASH_CPU_MAX_SEQ`], so that
    /// the benchmark can time it on shapes the dispatcher would not send here.
    ///
    /// **Batches are run one element at a time, deliberately.** candle
    /// dispatches `batch > 1` to a "varlen" kernel that repacks q, k and v into
    /// a single packed sequence, and that repack costs more than it saves: at
    /// batch 2, `heads = 24`, `head_dim = 128` it was 1.6x slower than this
    /// loop at 320 tokens and 4.1x slower at 1024, and the gap grows with
    /// sequence length. Looping the batch-1 kernel makes
    /// batch 2 behave like two batch-1 calls, which is the behaviour
    /// [`DEFAULT_FLASH_CPU_MAX_SEQ`] is calibrated against. It also lifts
    /// candle's refusal to combine an explicit mask with `batch > 1`, since
    /// every call this makes is batch 1.
    ///
    /// The kernel's layout is `[batch, seq, heads, head_dim]` where this
    /// workspace uses `[batch, heads, seq, head_dim]`, so the inputs are
    /// transposed on the way in. It reads through strides, so the transpose of
    /// a contiguous tensor is a view and costs nothing; the `contiguous()`
    /// calls exist to guarantee the unit last stride the kernel assumes, and
    /// are no-ops when the caller already passes contiguous tensors. That
    /// transposed view is also the *better* layout for it — one head's keys
    /// end up contiguous, so the inner loop walks memory forwards. Its output
    /// is `[batch, heads, seq, head_dim]` already.
    pub fn flash_attention_cpu(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let batch = q.dim(0)?;
        if batch == 1 {
            return flash_attention_cpu_single(q, k, v, mask);
        }
        let mut out = Vec::with_capacity(batch);
        for i in 0..batch {
            // The mask is shared across the batch: `flash_cpu_supported`
            // accepts only a `seq_q * seq_k` mask, which by construction has no
            // batch axis to slice.
            out.push(flash_attention_cpu_single(
                &q.narrow(0, i, 1)?,
                &k.narrow(0, i, 1)?,
                &v.narrow(0, i, 1)?,
                mask,
            )?);
        }
        Tensor::cat(&out, 0)
    }

    /// One batch element through candle's batch-1 CPU flash kernel.
    fn flash_attention_cpu_single(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        use candle_nn::attention::{flash_attn, AttnMask};

        let dim = q.dim(D::Minus1)?;
        let scale = (1f64 / (dim as f64).sqrt()) as f32;
        let qt = q.contiguous()?.transpose(1, 2)?;
        let kt = k.contiguous()?.transpose(1, 2)?;
        let vt = v.contiguous()?.transpose(1, 2)?;
        let attn_mask = match mask {
            Some(m) => AttnMask::Mask(m.contiguous()?),
            None => AttnMask::None,
        };
        flash_attn::<f32>(&qt, &kt, &vt, scale, attn_mask, None, None)
    }

    /// Attention, plus which implementation served it.
    ///
    /// candle 0.11 implements SDPA for Metal only — its `cpu_fwd` bails
    /// outright — so on any other device we go straight to
    /// [`naive_attention`] rather than paying for a guaranteed failure.
    ///
    /// On Metal, candle 0.11's fused kernel accepts head dimensions of 32,
    /// 64, 72, 80, 96, 128, 256 and 512, and it takes every unmasked shape in
    /// this workspace except SD 1.5's UNet, whose 40- and 160-wide heads are
    /// not in that set. Measured against the chunked path it is worth having:
    ///
    /// ```text
    ///                              CPU        Metal
    ///   SDXL UNet 1024        453.7 ms    14.8 ms  fused
    ///   Flux 1024             957.0 ms    43.8 ms  fused
    ///   SD 3.5 512             37.1 ms     2.7 ms  fused
    ///   SD 1.5 UNet 512       110.7 ms    16.8 ms  chunked (d=40)
    /// ```
    ///
    /// It still declines two things worth knowing about: f32 at
    /// `head_dim = 512`, which exceeds Metal's 32 KB of threadgroup memory,
    /// and a mask that is not `[batch, heads, seq_q, seq_k]` — [`causal_mask`]
    /// is `[1, 1, s, s]`, so CLIP's masked attention stays on the chunked
    /// path. Reproduce with `--example attention_path`.
    ///
    /// On CPU there is a second fused option — candle's own CPU flash kernel,
    /// via [`flash_attention_cpu`]. It is taken only for short sequences,
    /// because unlike the Metal kernel it is not a uniform win:
    /// [`DEFAULT_FLASH_CPU_MAX_SEQ`] carries the measurements and the
    /// reasoning. In practice that means the text encoders and the deeper
    /// UNet blocks take it and the large image-attention shapes do not.
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
        if flash_cpu_preferred(q, k, v, mask)? {
            return Ok((flash_attention_cpu(q, k, v, mask)?, AttentionPath::FlashCpu));
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

    /// Whether to hand this tensor to candle's fused norm kernels instead of
    /// the compositions below.
    ///
    /// The two are not interchangeable, and which one is better depends on the
    /// backend, because they do not share a reduction. Relative error against
    /// an f64 reference, from `--example norm_accuracy`:
    ///
    /// ```text
    ///                        CPU                     Metal
    ///   [1,154,4096]  ours 1.3e-7  candle 8.8e-7    ours 1.4e-7  candle 9.0e-8
    ///   [1,1536,3072] ours 9.4e-8  candle 7.5e-7    ours 1.3e-7  candle 9.3e-8
    ///   [1,77,768]    ours 7.1e-8  candle 4.4e-7    ours 1.1e-7  candle 7.6e-8
    /// ```
    ///
    /// candle's CPU kernel sums each row with a sequential `.sum::<f32>()`
    /// where `mean_keepdim` reduces in blocks, and error in a sequential sum
    /// grows with row length — 6 to 9x worse here, enough that routing T5 onto
    /// it moves `golden_t5` to 3.891e-3 past a 3e-3 bound. Its Metal kernel
    /// reduces across a threadgroup in a tree, which is at least as accurate as
    /// blocks, and 4.5x faster besides.
    ///
    /// So: fused on Metal, the composition on CPU. CUDA is deliberately not
    /// included — candle reduces in a tree there too and it would very likely
    /// behave like Metal, but there is no CUDA machine here to measure it on,
    /// and this file does not carry claims that were not measured.
    ///
    /// Half precision stays on the composition on every backend: it reduces in
    /// f32 whatever the input dtype, which `t5::RmsNorm` needs rather than
    /// prefers, since at d_model 4096 a sum of squares overflows f16.
    fn prefer_fused_norm(xs: &Tensor) -> bool {
        xs.dtype() == DType::F32 && xs.device().is_metal()
    }

    /// candle's fused layer norm, exposed only so `--example norm_path` can
    /// measure it. Not used in any model path: see the note on
    /// [`plain_layer_norm`].
    pub fn fused_layer_norm(
        xs: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
        eps: f64,
    ) -> Result<Tensor> {
        candle_nn::ops::layer_norm(xs, alpha, beta, eps as f32)
    }

    /// candle's fused RMS norm, exposed only for the same measurement.
    pub fn fused_rms_norm(xs: &Tensor, alpha: &Tensor, eps: f64) -> Result<Tensor> {
        candle_nn::ops::rms_norm(xs, alpha, eps as f32)
    }

    /// candle's fused rotary embedding, interleaved variant.
    ///
    /// `xs` is `[b, heads, seq, head_dim]`; `cos` and `sin` are `[seq, dim/2]`
    /// or `[b, seq, dim/2]`. Rotates **adjacent pairs** — `(x0, x1)`,
    /// `(x2, x3)` — which is Flux's convention. `rope` (without the `_i`)
    /// splits the head in half instead and is a different function on the
    /// same shapes.
    pub fn rope_interleaved(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        candle_nn::rotary_emb::rope_i(xs, cos, sin)
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

/// GGUF, the container almost every quantised community model ships in.
///
/// Re-exported rather than reimplemented. candle already parses GGUF and
/// dequantises the k-quant families, in safe Rust, and a second hand-written
/// parser for a format with this much variation in the wild would be more
/// risk than the seam is worth. This is the same bargain the project makes
/// for tensor math: use it, but only from here.
pub mod gguf {
    pub use candle_core::quantized::gguf_file::{write, Content, TensorInfo, Value, ValueType};
    pub use candle_core::quantized::{GgmlDType, QTensor};
}

pub mod quantized;

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

    /// Whether two handles name the same physical device.
    ///
    /// Lives here because deciding it means matching on candle's `Device`
    /// variants and reading the backend's own identifiers, which is exactly
    /// what the seam exists to keep out of the model and pipeline crates.
    /// candle's `Device` is not `PartialEq` and its own `same_device` is
    /// private, so without this every caller invents its own — and the
    /// plausible-looking version, comparing `is_cpu()`/`is_metal()`, calls two
    /// different CUDA cards equal.
    pub fn same(a: &Device, b: &Device) -> bool {
        // `same_device` comes from candle's `BackendDevice`, which is only in
        // scope here — another reason this belongs in the seam.
        #[allow(unused_imports)]
        use candle_core::backend::BackendDevice;
        match (a, b) {
            (Device::Cpu, Device::Cpu) => true,
            #[cfg(feature = "metal")]
            (Device::Metal(x), Device::Metal(y)) => x.same_device(y),
            #[cfg(feature = "cuda")]
            (Device::Cuda(x), Device::Cuda(y)) => x.same_device(y),
            _ => false,
        }
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

    /// A standard-normal draw as an MLX array, `[n, c, h, w]` in **NHWC**.
    ///
    /// The transpose happens here rather than at every call site because the
    /// draw order is what a seed pins: `normals` fills NCHW-major, and
    /// re-ordering afterwards is what keeps an MLX image identical to a candle
    /// one from the same seed.
    #[cfg(feature = "mlx")]
    pub fn randn_nhwc(
        rng: &mut SeededRng,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<crate::mlx::Array> {
        let v = rng.normals(n * c * h * w);
        let mut out = vec![0.0f32; v.len()];
        for bi in 0..n {
            for ci in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        out[((bi * h + y) * w + x) * c + ci] = v[((bi * c + ci) * h + y) * w + x];
                    }
                }
            }
        }
        crate::mlx::Array::from_slice_f32(&out, &[n, h, w, c])
    }

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
/// Skip a test for want of reference data.
///
/// Takes the same arguments as `eprintln!`. Prints a uniform `SKIP:` line, and
/// **panics instead** when `SD_REQUIRE_FIXTURES` is set — see
/// [`testing::skip_without_fixtures`] for why that switch exists.
///
/// Use it only for *missing data*. Environmental skips — no GPU, a memory
/// refusal, an unset `SD_TEST_*` path — stay plain `eprintln!`, because those
/// are not something generating fixtures would fix.
#[macro_export]
macro_rules! skip_missing_fixture {
    ($($arg:tt)*) => {
        $crate::testing::skip_without_fixtures(&format!($($arg)*))
    };
}

pub mod testing {
    use super::{DType, Result, Tensor};

    /// Set this to turn every "no reference data" skip into a failure.
    pub const REQUIRE_FIXTURES_ENV: &str = "SD_REQUIRE_FIXTURES";

    /// Skip a test for want of reference data — loudly, and not at all when
    /// [`REQUIRE_FIXTURES_ENV`] is set.
    ///
    /// # Why this exists
    ///
    /// Golden data is generated locally and gitignored, so every numerical
    /// test returns early when it is absent. An early return is a **pass**,
    /// and the measurement that prompted this is stark: renaming
    /// `tests/golden` aside and running the suite gives *the same 362 passing
    /// tests* as running it with every fixture in place. Nothing in the
    /// output distinguishes "verified" from "did nothing", so a fresh clone,
    /// a CI run, or a colleague who forgot to run the dumper all get a
    /// perfect green board that checked no numbers at all.
    ///
    /// That is not hypothetical. Three tests in this workspace silently
    /// stopped running when a symlink was renamed, and reported `ok`
    /// throughout.
    ///
    /// So: skipping still works by default — the fixtures really are too
    /// large to commit — but it is now *choosable*. `SD_REQUIRE_FIXTURES=1
    /// cargo test` is the run that means something.
    /// Backs [`crate::skip_missing_fixture`]; call the macro, not this.
    #[doc(hidden)]
    pub fn skip_without_fixtures(message: &str) {
        assert!(
            std::env::var_os(REQUIRE_FIXTURES_ENV).is_none(),
            "{message}\n\n\
             This test needs reference data and {REQUIRE_FIXTURES_ENV} is set, so \
             skipping is a failure. Generate the fixtures, or unset the variable to \
             go back to skipping."
        );
        eprintln!("SKIP: {message}");
    }

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

    /// Worst violation of `|a - b| <= rtol * |b|`, i.e. `max(|a-b| - rtol*|b|)`.
    ///
    /// Compare the result against an `atol`. Use this instead of
    /// [`closeness`] whenever the tensor carries values far from order 1,
    /// which for text encoders is the normal case rather than the exception:
    /// CLIP peaks at 851, and T5 at ~40,000. f32 cannot hold 1e-4 absolute at
    /// that magnitude, so an absolute bound reports arithmetic noise as a
    /// failure and tells you nothing about correctness.
    ///
    /// A negative result means every element was inside the relative
    /// allowance with room to spare.
    pub fn allclose_excess(a: &Tensor, b: &Tensor, rtol: f64) -> Result<f64> {
        let af = a.to_dtype(DType::F32)?.flatten_all()?;
        let bf = b.to_dtype(DType::F32)?.flatten_all()?;
        let diff = (&af - &bf)?.abs()?;
        let allowance = (bf.abs()? * rtol)?;
        Ok((diff - allowance)?.max(0)?.to_scalar::<f32>()? as f64)
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
        assert!(crate::refusal::is_refusal(&err), "unexpected error: {err}");
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
    fn flash_cpu_agrees_with_the_naive_reference() -> super::Result<()> {
        // Flash rebuilds the softmax incrementally under a running maximum
        // instead of normalising a materialised score matrix. That is exact in
        // exact arithmetic and very close in f32, but it is a different
        // summation order, so this is the test that says the reordering is
        // sound. Cross-attention (sq != sk) is included because the kernel
        // takes its key length from `k` and its head dim from `q`, and mixing
        // those up is the obvious way to get this wrong.
        let dev = Device::Cpu;
        for (b, h, sq, sk, d) in [
            (1usize, 1, 64usize, 64usize, 8usize),
            (1, 3, 40, 57, 16),
            (1, 8, 33, 77, 40),
            (2, 4, 24, 24, 64),
        ] {
            let q = Tensor::randn(0f32, 1f32, (b, h, sq, d), &dev)?;
            let k = Tensor::randn(0f32, 1f32, (b, h, sk, d), &dev)?;
            let v = Tensor::randn(0f32, 1f32, (b, h, sk, d), &dev)?;
            assert!(
                flash_cpu_supported(&q, &k, &v, None),
                "{b},{h},{sq},{sk},{d}"
            );

            let got = flash_attention_cpu(&q, &k, &v, None)?;
            let want = naive_attention(&q, &k, &v, None)?;
            assert_eq!(got.dims(), want.dims());
            assert_eq!(got.dtype(), want.dtype());
            super::testing::assert_close(&got, &want, 1e-5, "flash vs naive")?;
        }
        Ok(())
    }

    #[test]
    fn flash_cpu_takes_a_causal_mask() -> super::Result<()> {
        // The kernel indexes the mask flat as `q_pos * seq_k + kv_pos`, with no
        // batch or head axis, and `causal_mask` is `[1, 1, s, s]`. Those agree
        // only because the leading axes are 1 — if this ever compares equal
        // while the mask is transposed, the result is a model that attends
        // forwards in time and still produces plausible text embeddings.
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 48usize, 16usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?;
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?;
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?;
        let mask = causal_mask(s, &dev)?;
        assert!(flash_cpu_supported(&q, &k, &v, Some(&mask)));

        let got = flash_attention_cpu(&q, &k, &v, Some(&mask))?;
        let want = naive_attention(&q, &k, &v, Some(&mask))?;
        super::testing::assert_close(&got, &want, 1e-5, "flash masked vs naive")?;
        Ok(())
    }

    #[test]
    fn flash_cpu_declines_what_it_cannot_serve() -> super::Result<()> {
        // Each of these would otherwise be served wrongly rather than refused:
        // the kernel reads through raw strides and does not validate.
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 4, 16, 32), DType::F32, &dev)?;

        // f16 in, f32 out — the kernel always returns f32.
        let h = Tensor::zeros((1, 4, 16, 32), DType::F16, &dev)?;
        assert!(!flash_cpu_supported(&h, &h, &h, None));

        // Fewer kv heads than q heads: candle would treat it as GQA.
        let gqa = Tensor::zeros((1, 2, 16, 32), DType::F32, &dev)?;
        assert!(!flash_cpu_supported(&q, &gqa, &gqa, None));

        // A `v` narrower than `q`: the kernel takes head_dim from `q`.
        let narrow = Tensor::zeros((1, 4, 16, 8), DType::F32, &dev)?;
        assert!(!flash_cpu_supported(&q, &q, &narrow, None));

        // A mask that is not seq_q x seq_k. A per-batch or per-head mask would
        // be read as if it were the flat one, silently.
        let wrong = Tensor::zeros((1, 1, 16, 8), DType::F32, &dev)?;
        assert!(!flash_cpu_supported(&q, &q, &q, Some(&wrong)));
        let per_head = Tensor::zeros((1, 4, 16, 16), DType::F32, &dev)?;
        assert!(!flash_cpu_supported(&q, &q, &q, Some(&per_head)));

        // A *transposed* mask is the dangerous one: `seq_q * seq_k` is
        // symmetric, so it has exactly the element count a correct mask has.
        // A guard that counted elements would accept it and the kernel would
        // read it row-for-column, with no error anywhere.
        let k = Tensor::zeros((1, 4, 8, 32), DType::F32, &dev)?;
        let right = Tensor::zeros((1, 1, 16, 8), DType::F32, &dev)?;
        let flipped = Tensor::zeros((1, 1, 8, 16), DType::F32, &dev)?;
        assert_eq!(right.elem_count(), flipped.elem_count());
        assert!(flash_cpu_supported(&q, &k, &k, Some(&right)));
        assert!(!flash_cpu_supported(&q, &k, &k, Some(&flipped)));

        // Rank below 2 has no seq_q axis to check at all.
        let flat = Tensor::zeros(16 * 8, DType::F32, &dev)?;
        assert!(!flash_cpu_supported(&q, &k, &k, Some(&flat)));
        Ok(())
    }

    #[test]
    fn flash_cpu_runs_a_batch_one_element_at_a_time() -> super::Result<()> {
        // The batch loop exists to avoid candle's varlen repack, and it is the
        // reason a mask works at batch > 1 at all — candle refuses that
        // combination outright, so a regression to the varlen path would show
        // up here as an error rather than as a wrong number.
        let dev = Device::Cpu;
        let (b, h, s, d) = (3usize, 2usize, 24usize, 16usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?;
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?;
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)?;
        let mask = causal_mask(s, &dev)?;
        assert!(flash_cpu_supported(&q, &k, &v, Some(&mask)));

        let got = flash_attention_cpu(&q, &k, &v, Some(&mask))?;
        let want = naive_attention(&q, &k, &v, Some(&mask))?;
        assert_eq!(got.dims(), want.dims());
        super::testing::assert_close(&got, &want, 1e-5, "flash batched masked vs naive")?;

        // Element i of the batch must be element i of the output. Recomputing
        // one element alone and comparing catches a concatenation in the wrong
        // order, which averaging over the whole tensor would hide.
        let solo = flash_attention_cpu(
            &q.narrow(0, 2, 1)?,
            &k.narrow(0, 2, 1)?,
            &v.narrow(0, 2, 1)?,
            Some(&mask),
        )?;
        super::testing::assert_close(&got.narrow(0, 2, 1)?, &solo, 1e-6, "batch element 2")?;
        Ok(())
    }

    #[test]
    fn flash_cpu_is_preferred_only_for_short_sequences() -> super::Result<()> {
        // The whole value of the CPU flash path is that it is *not* taken for
        // the big image-attention shapes, where it loses by up to 2x. If this
        // test goes green with the limit ignored, every large model gets
        // slower and nothing else fails.
        let dev = Device::Cpu;
        let limit = DEFAULT_FLASH_CPU_MAX_SEQ;
        let short = Tensor::zeros((1, 4, limit, 64), DType::F32, &dev)?;
        assert!(flash_cpu_preferred(&short, &short, &short, None)?);

        let long = Tensor::zeros((1, 4, limit + 1, 64), DType::F32, &dev)?;
        assert!(!flash_cpu_preferred(&long, &long, &long, None)?);

        // A long query axis against a short key axis was measured to lose, so
        // it must be the maximum of the two that decides, not the minimum.
        assert!(!flash_cpu_preferred(&long, &short, &short, None)?);
        assert!(!flash_cpu_preferred(&short, &long, &long, None)?);
        Ok(())
    }

    #[test]
    fn attention_never_reports_the_metal_path_off_metal() -> super::Result<()> {
        // candle 0.11's SDPA is Metal-only, so on a CPU test runner the fused
        // path is unreachable. Asserting this is what stops a "fused agrees
        // with naive" test from quietly becoming naive-agrees-with-itself.
        //
        // Which of the CPU paths runs is a length question, so check both
        // sides of the limit: short sequences take candle's CPU flash kernel,
        // long ones stay on the chunked path.
        let dev = Device::Cpu;
        let short = Tensor::zeros((1, 2, 8, 32), DType::F32, &dev)?;
        let (_, path) = attention_with_path(&short, &short, &short, None)?;
        assert_eq!(path, AttentionPath::FlashCpu);

        let long = Tensor::zeros((1, 2, DEFAULT_FLASH_CPU_MAX_SEQ + 1, 32), DType::F32, &dev)?;
        let (_, path) = attention_with_path(&long, &long, &long, None)?;
        assert_eq!(path, AttentionPath::Naive);
        Ok(())
    }
}
