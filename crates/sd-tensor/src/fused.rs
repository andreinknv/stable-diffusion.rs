//! Kernels this project writes itself, dispatched onto candle's Metal device.
//!
//! candle is the tensor backend, not a ceiling. Where a hot path is several
//! candle ops in a row, each one reads its input from memory and writes its
//! output back, and the arithmetic between those trips is trivial — these are
//! memory-bound, so the cost *is* the round trips. One kernel doing the same
//! work makes one trip.
//!
//! Nothing here forks candle or adds anything to the build graph.
//! `candle-metal-kernels` already compiles as part of every `--features metal`
//! build; this asks for it by name so we can reach three things it exposes:
//! `Device::new_library_with_source` to compile our own Metal source at
//! runtime, `MetalDevice::command_encoder` to encode a dispatch onto the same
//! command stream candle is already using, and `MetalStorage` to hand the
//! result back as an ordinary `Tensor`.
//!
//! Every kernel here is written against the composition it replaces, and the
//! test beside it compares the two. A kernel that is fast and slightly
//! different is not an optimisation, it is a bug with a benchmark.

use crate::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor, D};

/// Fused GEGLU: `hidden * gelu(gate)`, where the two halves are the first and
/// second halves of the last axis.
///
/// The composition this replaces runs `narrow`, `narrow`, `gelu`, `mul`. The
/// narrows are free — they are views — but `gelu` reads `inner` and writes
/// `inner`, then `mul` reads two lots of `inner` and writes `inner` again.
/// Five trips over the data where two would do.
///
/// Called once per transformer block, so sixteen times per SD 1.5 forward, on
/// a tensor eight times the model width.
pub struct Geglu {
    /// Half the size of the last axis: the width of the output.
    pub inner: usize,
}

impl Geglu {
    /// The exact same arithmetic as [`crate::ops::gelu`], for the fallback and
    /// for the test that pins the kernel to it.
    fn gelu_erf(x: f32) -> f32 {
        // The same arrangement as the Metal kernel, and for the same reason:
        // `1 + erf(u)` cancels to exactly zero for u below about -4, so the
        // negative branch reads erfc off directly instead. See `SOURCE`.
        let u = x * std::f32::consts::FRAC_1_SQRT_2;
        let erfc_a = erfc_abs(u.abs());
        let one_plus_erf = if u >= 0.0 { 2.0 - erfc_a } else { erfc_a };
        0.5 * x * one_plus_erf
    }
}

/// `erfc` for a non-negative argument, Abramowitz and Stegun 7.1.26.
///
/// Neither `erf` nor `erfc` is in Rust's standard library. This is the same
/// polynomial the Metal kernel uses, returned before the `1 -` that would turn
/// it into `erf` — which is the whole point, since that subtraction is what
/// loses the tail.
// The coefficients are quoted at the precision Abramowitz and Stegun print
// them, which is more than f32 holds. Rounding them by hand here would save
// nothing — the compiler does exactly that — and would make them no longer
// searchable against the source they came from.
#[allow(clippy::excessive_precision)]
fn erfc_abs(a: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
        + 0.254_829_592)
        * t
        * (-a * a).exp()
}

impl CustomOp1 for Geglu {
    fn name(&self) -> &'static str {
        "fused-geglu"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (src, rows, stride) = cpu_input(storage, layout, self.inner)?;
        let mut out = vec![0f32; rows * self.inner];
        for r in 0..rows {
            let base = r * stride;
            for c in 0..self.inner {
                let value = src[base + c];
                let gate = src[base + self.inner + c];
                out[r * self.inner + c] = value * Self::gelu_erf(gate);
            }
        }
        let mut dims = layout.shape().dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = self.inner;
        Ok((CpuStorage::F32(out), Shape::from(dims)))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle_core::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::geglu(self, storage, layout)
    }
}

/// Shared validation for both backends.
fn check(layout: &Layout, inner: usize) -> Result<(usize, usize)> {
    let dims = layout.shape().dims();
    let last = *dims.last().ok_or_else(|| {
        candle_core::Error::Msg("fused geglu: input must have at least one axis".into())
    })?;
    if last != inner * 2 {
        return Err(candle_core::Error::Msg(format!(
            "fused geglu: last axis is {last}, expected {} (2 x inner)",
            inner * 2
        )));
    }
    if !layout.is_contiguous() {
        // The kernel indexes rows by a fixed stride. A non-contiguous input
        // would be read as though it were contiguous, which is silent.
        return Err(candle_core::Error::Msg(
            "fused geglu: input must be contiguous".into(),
        ));
    }
    let rows = dims.iter().product::<usize>() / last;
    Ok((rows, last))
}

fn cpu_input<'a>(
    storage: &'a CpuStorage,
    layout: &Layout,
    inner: usize,
) -> Result<(&'a [f32], usize, usize)> {
    let (rows, stride) = check(layout, inner)?;
    let CpuStorage::F32(src) = storage else {
        return Err(candle_core::Error::Msg(
            "fused geglu: only f32 is implemented".into(),
        ));
    };
    Ok((&src[layout.start_offset()..], rows, stride))
}

/// `SD_FUSED_KERNELS=0` sends everything back to the compositions.
///
/// Not a tuning knob. It exists so the kernels can be measured against what
/// they replace **in one session, alternating**, rather than against a number
/// written down earlier on a machine in a different state — which is how this
/// project has previously convinced itself of a speedup that was not there.
/// It also gives a one-variable answer to "is the kernel doing this?" when
/// something downstream looks wrong.
pub(crate) fn kernel_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SD_FUSED_KERNELS").as_deref() != Ok("0"))
}

/// `hidden * gelu(gate)` over the last axis, split in half.
///
/// Falls back to the composition on any backend without a kernel, and on any
/// input the kernel declines, so callers do not branch.
pub fn geglu(h: &Tensor, inner: usize) -> Result<Tensor> {
    if kernel_enabled()
        && h.device().is_metal()
        && h.dtype() == crate::DType::F32
        && h.is_contiguous()
    {
        return h.apply_op1_no_bwd(&Geglu { inner });
    }
    let hidden = h.narrow(D::Minus1, 0, inner)?;
    let gate = h.narrow(D::Minus1, inner, inner)?;
    hidden * crate::ops::gelu(&gate)?
}

/// Adaptive layer norm: `norm(x) * (1 + scale) + shift`.
///
/// The conditioning mechanism of every diffusion transformer. `x` is
/// normalised with no learned affine at all — the scale and shift arrive from
/// the timestep and prompt instead, which is how those steer the network.
///
/// As four ops this is a norm, an add of 1, a broadcast multiply and a
/// broadcast add: seven trips over an activation that is 50 MB at Flux's
/// 1024x1024. As one kernel it is two, because the row is held in threadgroup
/// memory across both reductions and the write.
///
/// `scale` and `shift` are `[b, 1, width]`. They arrive from a `narrow` of the
/// modulation projection and so are usually not contiguous; they are made so
/// here, which copies 12 KB against an activation three thousand times larger.
pub struct AdaLayerNorm {
    pub eps: f64,
}

impl AdaLayerNorm {
    /// Above this the row no longer fits in threadgroup memory and the kernel
    /// declines. Apple GPUs give 32 KB; Flux and SD 3 are 3072 wide and T5-XXL
    /// is 4096, so all of them fit with room to spare.
    pub const MAX_WIDTH: usize = 8192;
}

impl crate::CustomOp3 for AdaLayerNorm {
    fn name(&self) -> &'static str {
        "fused-adaln"
    }

    fn cpu_fwd(
        &self,
        xs: &CpuStorage,
        xl: &Layout,
        scale: &CpuStorage,
        sl: &Layout,
        shift: &CpuStorage,
        fl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (rows, width, tokens) = adaln_dims(xl, sl, fl)?;
        let (CpuStorage::F32(x), CpuStorage::F32(sc), CpuStorage::F32(sh)) = (xs, scale, shift)
        else {
            return Err(candle_core::Error::Msg(
                "fused adaln: only f32 is implemented".into(),
            ));
        };
        let (x, sc, sh) = (
            &x[xl.start_offset()..],
            &sc[sl.start_offset()..],
            &sh[fl.start_offset()..],
        );
        let mut out = vec![0f32; rows * width];
        for r in 0..rows {
            let base = r * width;
            let cbase = (r / tokens) * width;
            let row = &x[base..base + width];
            // Reduced in f64 on this path. It is the reference the kernel is
            // tested against, so it should be the more accurate of the two.
            let mean = row.iter().map(|v| *v as f64).sum::<f64>() / width as f64;
            let var = row
                .iter()
                .map(|v| {
                    let d = *v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / width as f64;
            let rstd = 1.0 / (var + self.eps).sqrt();
            for c in 0..width {
                let n = ((row[c] as f64 - mean) * rstd) as f32;
                out[base + c] = n * (1.0 + sc[cbase + c]) + sh[cbase + c];
            }
        }
        Ok((CpuStorage::F32(out), xl.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        xs: &candle_core::MetalStorage,
        xl: &Layout,
        scale: &candle_core::MetalStorage,
        sl: &Layout,
        shift: &candle_core::MetalStorage,
        fl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::adaln(self, xs, xl, scale, sl, shift, fl)
    }
}

/// `(rows, width, tokens)`, with every shape requirement checked once for both
/// backends.
fn adaln_dims(xl: &Layout, sl: &Layout, fl: &Layout) -> Result<(usize, usize, usize)> {
    let dims = xl.shape().dims();
    if dims.len() != 3 {
        return Err(candle_core::Error::Msg(format!(
            "fused adaln: expected [b, tokens, width], got {dims:?}"
        )));
    }
    let (b, tokens, width) = (dims[0], dims[1], dims[2]);
    for (name, l) in [("scale", sl), ("shift", fl)] {
        let d = l.shape().dims();
        if d != [b, 1, width] {
            return Err(candle_core::Error::Msg(format!(
                "fused adaln: {name} is {d:?}, expected {:?}",
                [b, 1, width]
            )));
        }
        if !l.is_contiguous() {
            return Err(candle_core::Error::Msg(format!(
                "fused adaln: {name} must be contiguous"
            )));
        }
    }
    if !xl.is_contiguous() {
        return Err(candle_core::Error::Msg(
            "fused adaln: input must be contiguous".into(),
        ));
    }
    Ok((b * tokens, width, tokens))
}

/// `norm(x) * (1 + scale) + shift`, fused where there is a kernel for it.
///
/// Falls back to the composition otherwise, so callers do not branch.
pub fn ada_layer_norm(xs: &Tensor, scale: &Tensor, shift: &Tensor, eps: f64) -> Result<Tensor> {
    let usable = kernel_enabled()
        && xs.device().is_metal()
        && xs.dtype() == crate::DType::F32
        && xs.rank() == 3
        && xs.dim(D::Minus1)? <= AdaLayerNorm::MAX_WIDTH
        && xs.is_contiguous();
    if usable {
        // Cheap on both counts: these are [b, 1, width], and `contiguous` is a
        // no-op when they already are.
        let scale = scale.contiguous()?;
        let shift = shift.contiguous()?;
        if scale.dims() == [xs.dim(0)?, 1, xs.dim(2)?] {
            return xs.apply_op3_no_bwd(&scale, &shift, &AdaLayerNorm { eps });
        }
    }
    crate::ops::plain_layer_norm(xs, eps)?
        .broadcast_mul(&(scale + 1.0)?)?
        .broadcast_add(shift)
}

/// Group normalisation with its affine, as one kernel.
///
/// candle implements this as roughly ten ops. At SD 1.5's shapes that measures
/// between 1.7 and 5.8 GB/s — against 71.6 GB/s for the adaLN kernel on the
/// same machine — and it is 23.5% of a step, so it is the largest single thing
/// in the UNet that is not a convolution or a matmul.
pub struct GroupNormOp {
    pub groups: usize,
    pub eps: f64,
}

/// The shape facts both backends need.
struct GroupNormDims {
    /// One per (batch, group).
    rows: usize,
    /// Elements in a group row: `channels_per_group * hw`.
    n: usize,
    hw: usize,
    cpg: usize,
}

fn group_norm_dims(xl: &Layout, wl: &Layout, bl: &Layout, groups: usize) -> Result<GroupNormDims> {
    let dims = xl.shape().dims();
    if dims.len() != 4 {
        return Err(candle_core::Error::Msg(format!(
            "fused group_norm: expected [b, c, h, w], got {dims:?}"
        )));
    }
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    if c % groups != 0 {
        return Err(candle_core::Error::Msg(format!(
            "fused group_norm: {groups} groups do not divide {c} channels"
        )));
    }
    for (name, l) in [("weight", wl), ("bias", bl)] {
        if l.shape().dims() != [c] {
            return Err(candle_core::Error::Msg(format!(
                "fused group_norm: {name} is {:?}, expected [{c}]",
                l.shape().dims()
            )));
        }
        if !l.is_contiguous() {
            return Err(candle_core::Error::Msg(format!(
                "fused group_norm: {name} must be contiguous"
            )));
        }
    }
    if !xl.is_contiguous() {
        return Err(candle_core::Error::Msg(
            "fused group_norm: input must be contiguous".into(),
        ));
    }
    let cpg = c / groups;
    let hw = h * w;
    Ok(GroupNormDims {
        rows: b * groups,
        n: cpg * hw,
        hw,
        cpg,
    })
}

impl crate::CustomOp3 for GroupNormOp {
    fn name(&self) -> &'static str {
        "fused-group-norm"
    }

    fn cpu_fwd(
        &self,
        xs: &CpuStorage,
        xl: &Layout,
        weight: &CpuStorage,
        wl: &Layout,
        bias: &CpuStorage,
        bl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let d = group_norm_dims(xl, wl, bl, self.groups)?;
        let (CpuStorage::F32(x), CpuStorage::F32(w), CpuStorage::F32(b)) = (xs, weight, bias)
        else {
            return Err(candle_core::Error::Msg(
                "fused group_norm: only f32 is implemented".into(),
            ));
        };
        let (x, w, b) = (
            &x[xl.start_offset()..],
            &w[wl.start_offset()..],
            &b[bl.start_offset()..],
        );
        let mut out = vec![0f32; xl.shape().elem_count()];
        for row in 0..d.rows {
            let base = row * d.n;
            let cbase = (row % self.groups) * d.cpg;
            // f64 on this path: it is the reference the kernel is tested
            // against, so it should be the more accurate of the two.
            let mean = x[base..base + d.n].iter().map(|v| *v as f64).sum::<f64>() / d.n as f64;
            let var = x[base..base + d.n]
                .iter()
                .map(|v| {
                    let t = *v as f64 - mean;
                    t * t
                })
                .sum::<f64>()
                / d.n as f64;
            let rstd = 1.0 / (var + self.eps).sqrt();
            for lc in 0..d.cpg {
                let (wc, bc) = (w[cbase + lc] as f64, b[cbase + lc] as f64);
                let o = base + lc * d.hw;
                for j in 0..d.hw {
                    out[o + j] = ((x[o + j] as f64 - mean) * rstd * wc + bc) as f32;
                }
            }
        }
        Ok((CpuStorage::F32(out), xl.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        xs: &candle_core::MetalStorage,
        xl: &Layout,
        weight: &candle_core::MetalStorage,
        wl: &Layout,
        bias: &candle_core::MetalStorage,
        bl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::group_norm(self, xs, xl, weight, wl, bias, bl)
    }
}

/// Group normalisation, shadowing `candle_nn::GroupNorm`.
///
/// Constructed exactly as candle's is — the same parameter names and the same
/// defaults — so checkpoint loading is unchanged and this is only a question
/// of which arithmetic runs.
#[derive(Debug)]
pub struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    groups: usize,
    eps: f64,
    /// candle's composition, for every input the kernel declines.
    inner: candle_nn::GroupNorm,
}

impl GroupNorm {
    pub fn new(
        weight: Tensor,
        bias: Tensor,
        channels: usize,
        groups: usize,
        eps: f64,
    ) -> Result<Self> {
        let inner = candle_nn::GroupNorm::new(weight.clone(), bias.clone(), channels, groups, eps)?;
        Ok(Self {
            weight,
            bias,
            groups,
            eps,
            inner,
        })
    }
}

impl crate::Module for GroupNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let usable = kernel_enabled()
            && xs.device().is_metal()
            && xs.dtype() == crate::DType::F32
            && xs.rank() == 4
            && xs.is_contiguous();
        if usable {
            return xs.apply_op3_no_bwd(
                &self.weight,
                &self.bias,
                &GroupNormOp {
                    groups: self.groups,
                    eps: self.eps,
                },
            );
        }
        crate::Module::forward(&self.inner, xs)
    }
}

/// candle's group norm, exposed so `--example group_norm_kernel` can measure
/// what it replaced. Not used by any model path.
pub fn candle_group_norm(
    weight: &Tensor,
    bias: &Tensor,
    channels: usize,
    groups: usize,
    eps: f64,
) -> Result<impl crate::Module> {
    candle_nn::GroupNorm::new(weight.clone(), bias.clone(), channels, groups, eps)
}

/// Same signature and same weight names as `candle_nn::group_norm`.
pub fn group_norm(
    groups: usize,
    channels: usize,
    eps: f64,
    vb: crate::nn::VarBuilder,
) -> Result<GroupNorm> {
    let weight = vb.get_with_hints(channels, "weight", candle_nn::Init::Const(1.))?;
    let bias = vb.get_with_hints(channels, "bias", candle_nn::Init::Const(0.))?;
    GroupNorm::new(weight, bias, channels, groups, eps)
}

#[cfg(feature = "metal")]
mod metal {
    use super::{check, Geglu};
    use crate::{Result, Shape};
    use candle_core::backend::BackendStorage;
    use candle_core::{DType, MetalStorage};
    use candle_metal_kernels::metal::{Device, Library};
    use objc2_metal::MTLSize;
    use std::cell::RefCell;

    /// Our own source. One thread per `float4` of output: two 16-byte loads
    /// and one 16-byte store, against the composition's five trips.
    ///
    /// `inner` is the model width times four, so 1280, 2560 or 5120 in SD 1.5
    /// — always a multiple of four, and both halves therefore start on a
    /// 16-byte boundary, which is what makes the vector loads legal.
    ///
    /// The grid is two-dimensional so that a row and a column arrive as
    /// `thread_position_in_grid` directly. Indexing a flat grid would need an
    /// integer divide and modulo per thread to recover them, and integer
    /// division is slow enough on a GPU to show up in a kernel this thin.
    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// The erf GELU, written to survive its own tail.
//
// Metal has no `erf`, so it has to be approximated here. The approximation is
// Abramowitz and Stegun 7.1.26, which is the usual choice and is what candle
// uses; what differs is the arrangement.
//
// A&S computes erf(a) for a >= 0 as `1 - poly(t) * exp(-a*a)`. GELU then wants
// `1 + erf(u)`. For negative u that is `1 - (1 - erfc)`, and the subtraction
// destroys the answer: once erfc falls below f32's epsilon the `1 - erfc`
// rounds to exactly 1, the outer `1 -` gives exactly 0, and every input below
// about -6 returns precisely zero instead of a small negative number.
//
// But `poly(t) * exp(-a*a)` *is* erfc(a), before any subtraction happens. So
// the negative branch reads it off directly and never cancels. Same
// polynomial, same op count, no dead tail.
inline float4 gelu_erf4(float4 x) {
    float4 u = x * 0.70710678118654752440f;   // x / sqrt(2)
    float4 a = fabs(u);
    float4 t = 1.0f / (1.0f + 0.3275911f * a);
    float4 erfc_a = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t
                      - 0.284496736f) * t + 0.254829592f) * t * exp(-a * a);
    // 1 + erf(u): two minus erfc above zero, erfc itself below it.
    float4 one_plus_erf = select(erfc_a, 2.0f - erfc_a, u >= 0.0f);
    return 0.5f * x * one_plus_erf;
}

// Sum across a threadgroup, in a tree.
//
// `simd_sum` reduces 32 lanes at a time in hardware; the per-simdgroup totals
// then go through one short serial pass. Both halves are blocked rather than
// sequential over the row, which is the difference that made candle's CPU norm
// 6 to 9x less accurate than the composition it replaced here.
inline float tg_sum(float v, threadgroup float *scratch, uint lane, uint lanes) {
    float s = simd_sum(v);
    if ((lane & 31u) == 0u) { scratch[lane >> 5] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float total = 0.0f;
        uint n = (lanes + 31u) >> 5;
        for (uint i = 0u; i < n; ++i) { total += scratch[i]; }
        scratch[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = scratch[0];
    // Before the caller reuses `scratch` for the second reduction.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// norm(x) * (1 + scale) + shift, one threadgroup per row.
//
// The row is read from device memory once, into threadgroup memory, and both
// reductions and the write then run against that copy. Two device trips where
// the four-op composition makes seven.
// Group norm plus its affine, one threadgroup per (batch, group).
//
// candle implements this as about ten ops — two reductions and five full
// passes over the tensor — which measures at 1.7 to 5.8 GB/s. This makes two
// passes: one to reduce, one to write.
//
// A group row is up to 40,960 elements at SD 1.5's widest, far past the 32 KB
// of threadgroup memory the adaLN kernel relies on, so the row cannot be
// cached and is read twice instead.
//
// The accumulators are shifted by the row's first element. Summing x and x*x
// directly and forming `E[x*x] - E[x]^2` cancels badly when the mean is large
// next to the spread, which is exactly what a post-convolution activation
// looks like; subtracting any constant first leaves the variance unchanged and
// keeps both sums near zero.
kernel void group_norm_f32(
    device const float *x       [[buffer(0)]],
    device const float *weight  [[buffer(1)]],
    device const float *bias    [[buffer(2)]],
    device float       *out     [[buffer(3)]],
    constant uint      &n       [[buffer(4)]],
    constant uint      &hw      [[buffer(5)]],
    constant uint      &cpg     [[buffer(6)]],
    constant uint      &groups  [[buffer(7)]],
    constant float     &eps     [[buffer(8)]],
    threadgroup float  *scratch [[threadgroup(0)]],
    uint row   [[threadgroup_position_in_grid]],
    uint lane  [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]])
{
    uint base  = row * n;
    uint cbase = (row % groups) * cpg;
    float x0 = x[base];

    float s = 0.0f, ss = 0.0f;
    for (uint i = lane; i < n; i += lanes) {
        float d = x[base + i] - x0;
        s += d;
        ss += d * d;
    }
    s  = tg_sum(s,  scratch, lane, lanes);
    ss = tg_sum(ss, scratch, lane, lanes);

    float inv = 1.0f / (float)n;
    float m    = s * inv;
    float var  = ss * inv - m * m;
    float mean = x0 + m;
    // The shifted form can land a hair below zero on a constant row.
    float rstd = rsqrt(max(var, 0.0f) + eps);

    // Channel-major within the group, so the affine pair is loaded once per
    // channel rather than divided out per element.
    for (uint lc = 0; lc < cpg; ++lc) {
        float w = weight[cbase + lc];
        float b = bias[cbase + lc];
        uint  o = base + lc * hw;
        for (uint j = lane; j < hw; j += lanes) {
            out[o + j] = (x[o + j] - mean) * rstd * w + b;
        }
    }
}

kernel void adaln_f32(
    device const float *x       [[buffer(0)]],
    device const float *scale   [[buffer(1)]],
    device const float *shift   [[buffer(2)]],
    device float       *out     [[buffer(3)]],
    constant uint      &width   [[buffer(4)]],
    constant uint      &tokens  [[buffer(5)]],
    constant float     &eps     [[buffer(6)]],
    threadgroup float  *tile    [[threadgroup(0)]],
    threadgroup float  *scratch [[threadgroup(1)]],
    uint row   [[threadgroup_position_in_grid]],
    uint lane  [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]])
{
    uint base = row * width;
    // scale and shift are [b, 1, width]: one row of conditioning per batch
    // element, shared by every token in it.
    uint cbase = (row / tokens) * width;

    float s = 0.0f;
    for (uint i = lane; i < width; i += lanes) {
        float v = x[base + i];
        tile[i] = v;
        s += v;
    }
    float mean = tg_sum(s, scratch, lane, lanes) / (float)width;

    float ss = 0.0f;
    for (uint i = lane; i < width; i += lanes) {
        float d = tile[i] - mean;
        ss += d * d;
    }
    // eps inside the sqrt, after the divide — matching ops::plain_layer_norm.
    float rstd = rsqrt(tg_sum(ss, scratch, lane, lanes) / (float)width + eps);

    for (uint i = lane; i < width; i += lanes) {
        float n = (tile[i] - mean) * rstd;
        out[base + i] = n * (1.0f + scale[cbase + i]) + shift[cbase + i];
    }
}

kernel void geglu_f32(
    device const float4 *h      [[buffer(0)]],
    device float4       *out    [[buffer(1)]],
    constant uint       &inner4 [[buffer(2)]],
    uint2                gid    [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    if (col >= inner4) { return; }

    // Row r of the input holds 2*inner floats: the value half, then the gate.
    uint base  = row * inner4 * 2u;
    float4 value = h[base + col];
    float4 gate  = h[base + inner4 + col];

    out[row * inner4 + col] = value * gelu_erf4(gate);
}
"#;

    thread_local! {
        /// Compiling the source costs milliseconds; doing it per call would
        /// cost more than the kernel saves. `ComputePipeline` wraps a
        /// retained Objective-C object and is not `Sync`, so the cache is
        /// per thread rather than global — each worker compiles once.
        static LIBRARY: RefCell<Option<Library>> = const { RefCell::new(None) };
    }

    fn library(device: &Device) -> Result<Library> {
        LIBRARY.with(|slot| {
            let mut slot = slot.borrow_mut();
            if let Some(lib) = slot.as_ref() {
                return Ok(lib.clone());
            }
            let lib = device
                .new_library_with_source(super::metal::SOURCE, None)
                .map_err(|e| candle_core::Error::Msg(format!("fused geglu: compiling: {e}")))?;
            *slot = Some(lib.clone());
            Ok(lib)
        })
    }

    /// The compiled pipeline for one kernel by name.
    fn pipeline(
        device: &candle_core::MetalDevice,
        name: &'static str,
    ) -> Result<candle_metal_kernels::metal::ComputePipeline> {
        let lib = library(device.device())?;
        let func = lib
            .get_function(name, None)
            .map_err(|e| candle_core::Error::Msg(format!("fused {name}: get_function: {e}")))?;
        device
            .device()
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("fused {name}: pipeline: {e}")))
    }

    pub(super) fn group_norm(
        op: &super::GroupNormOp,
        xs: &MetalStorage,
        xl: &crate::Layout,
        weight: &MetalStorage,
        wl: &crate::Layout,
        bias: &MetalStorage,
        bl: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let d = super::group_norm_dims(xl, wl, bl, op.groups)?;
        let device = xs.device().clone();
        let pipeline = pipeline(&device, "group_norm_f32")?;

        let count = xl.shape().elem_count();
        let out = device.new_buffer(count, DType::F32, "fused-group-norm")?;
        let esz = DType::F32.size_in_bytes();
        let (n, hw, cpg, groups, eps) = (
            d.n as u32,
            d.hw as u32,
            d.cpg as u32,
            op.groups as u32,
            op.eps as f32,
        );

        {
            let guard = device.command_encoder()?;
            guard.set_compute_pipeline_state(&pipeline);
            let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
            encoder.set_input_buffer(0, Some(xs.buffer()), xl.start_offset() * esz);
            encoder.set_input_buffer(1, Some(weight.buffer()), wl.start_offset() * esz);
            encoder.set_input_buffer(2, Some(bias.buffer()), bl.start_offset() * esz);
            encoder.set_output_buffer(3, Some(&out), 0);
            encoder.set_bytes(4, &n);
            encoder.set_bytes(5, &hw);
            encoder.set_bytes(6, &cpg);
            encoder.set_bytes(7, &groups);
            encoder.set_bytes(8, &eps);
            encoder.set_threadgroup_memory_length(0, 64 * esz);

            let lanes = pipeline.max_total_threads_per_threadgroup().min(256) / 32 * 32;
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: d.rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: lanes.max(32),
                    height: 1,
                    depth: 1,
                },
            );
        }

        Ok((
            MetalStorage::new(out, device, count, DType::F32),
            xl.shape().clone(),
        ))
    }

    pub(super) fn adaln(
        op: &super::AdaLayerNorm,
        xs: &MetalStorage,
        xl: &crate::Layout,
        scale: &MetalStorage,
        sl: &crate::Layout,
        shift: &MetalStorage,
        fl: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (rows, width, tokens) = super::adaln_dims(xl, sl, fl)?;
        let device = xs.device().clone();
        let pipeline = pipeline(&device, "adaln_f32")?;

        let count = rows * width;
        let out = device.new_buffer(count, DType::F32, "fused-adaln")?;
        let (w, tk, eps) = (width as u32, tokens as u32, op.eps as f32);
        let esz = DType::F32.size_in_bytes();

        {
            let guard = device.command_encoder()?;
            guard.set_compute_pipeline_state(&pipeline);
            let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
            encoder.set_input_buffer(0, Some(xs.buffer()), xl.start_offset() * esz);
            encoder.set_input_buffer(1, Some(scale.buffer()), sl.start_offset() * esz);
            encoder.set_input_buffer(2, Some(shift.buffer()), fl.start_offset() * esz);
            encoder.set_output_buffer(3, Some(&out), 0);
            encoder.set_bytes(4, &w);
            encoder.set_bytes(5, &tk);
            encoder.set_bytes(6, &eps);
            // The row, and one slot per simdgroup for the reduction.
            encoder.set_threadgroup_memory_length(0, width * esz);
            encoder.set_threadgroup_memory_length(1, 64 * esz);

            // A multiple of 32, so no simdgroup is partly idle, and capped by
            // what the pipeline will actually take.
            let lanes = pipeline.max_total_threads_per_threadgroup().min(256) / 32 * 32;
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: lanes.max(32),
                    height: 1,
                    depth: 1,
                },
            );
        }

        Ok((
            MetalStorage::new(out, device, count, DType::F32),
            xl.shape().clone(),
        ))
    }

    pub(super) fn geglu(
        op: &Geglu,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (rows, _) = check(layout, op.inner)?;
        if op.inner % 4 != 0 {
            return Err(candle_core::Error::Msg(format!(
                "fused geglu: inner {} is not a multiple of 4",
                op.inner
            )));
        }
        let device = storage.device().clone();
        let pipeline = pipeline(&device, "geglu_f32")?;

        let count = rows * op.inner;
        let out = device.new_buffer(count, DType::F32, "fused-geglu")?;
        let inner4 = (op.inner / 4) as u32;

        {
            let guard = device.command_encoder()?;
            guard.set_compute_pipeline_state(&pipeline);
            // The guard owns the command stream; the encoder is how buffers
            // and the dispatch are attached to it.
            let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
            // The input offset is in elements of `float4`, and `start_offset`
            // is in elements of `f32` — a contiguous tensor from a `narrow`
            // can carry one, and dividing it silently would read the wrong
            // rows. `check` already required contiguity; require the offset to
            // be expressible too rather than assume it is zero.
            let byte_offset = layout.start_offset() * DType::F32.size_in_bytes();
            encoder.set_input_buffer(0, Some(storage.buffer()), byte_offset);
            encoder.set_output_buffer(1, Some(&out), 0);
            encoder.set_bytes(2, &inner4);

            // One thread per float4 of output. The threadgroup width is capped
            // by the pipeline rather than assumed: 1024 is not guaranteed.
            let width = pipeline.max_total_threads_per_threadgroup().min(256);
            encoder.dispatch_threads(
                MTLSize {
                    width: inner4 as usize,
                    height: rows,
                    depth: 1,
                },
                MTLSize {
                    width,
                    height: 1,
                    depth: 1,
                },
            );
        }

        let mut dims = layout.shape().dims().to_vec();
        *dims.last_mut().expect("rank >= 1") = op.inner;
        Ok((
            MetalStorage::new(out, device, count, DType::F32),
            Shape::from(dims),
        ))
    }
}
