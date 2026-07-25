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
    use super::{Result, Tensor, D};

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

    /// Scaled dot-product attention without a mask.
    ///
    /// `q`, `k`, `v` are `[batch, heads, seq, head_dim]`.
    ///
    /// Deliberately naive: it materialises the full `seq x seq` score matrix.
    /// For 512x512 VAE attention that is fine; for large UNet cross-attention
    /// it is not. Replacing this with a fused/flash kernel is the single
    /// highest-value optimisation available behind the seam, and it can be done
    /// without touching a line of model code.
    pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let dim = q.dim(D::Minus1)?;
        let scale = 1f64 / (dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * scale)?;
        let weights = softmax_last_dim(&scores)?;
        weights.matmul(&v.contiguous()?)
    }
}

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
