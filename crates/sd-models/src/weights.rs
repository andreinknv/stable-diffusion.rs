//! Building a model from either dense or quantised weights.
//!
//! Two models now need this — T5 and Flux — and for the same reason, which is
//! worth stating because it is not the obvious one. Quantised residency is
//! usually a memory optimisation. Here it is also a *correctness* requirement:
//! both models carry activations far outside F16's range (T5 peaks near
//! 200,000 against F16's ceiling of 65,504), so dequantising to F16 at load
//! produces NaN partway up the stack, and dequantising to F32 costs more
//! memory than the machine has. Holding the blocks and expanding per matmul
//! avoids both.
//!
//! One [`Source`] serves both paths so the dense and quantised constructions
//! cannot drift apart — which matters, because only the dense one has a
//! golden reference to check against.

use std::collections::HashMap;
use std::sync::Arc;

use sd_tensor::gguf::QTensor;
use sd_tensor::nn::{linear, linear_no_bias, Linear, VarBuilder};
use sd_tensor::quantized::QLinear;
use sd_tensor::{Module, Result, Tensor};

/// Quantised tensors keyed by the name the model asks for.
pub type QuantizedWeights = HashMap<String, Arc<QTensor>>;

/// A projection, dense or quantised.
#[derive(Debug)]
pub enum Proj {
    Dense(Linear),
    Quantized(QLinear),
}

impl Proj {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(xs),
            Self::Quantized(q) => q.forward(xs),
        }
    }

    /// Weight bytes actually held. Zero for the dense path, where the tensor
    /// is owned by the `VarBuilder` rather than by this layer.
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Dense(_) => 0,
            Self::Quantized(q) => q.resident_bytes(),
        }
    }
}

/// One transformer block's weights, copied to `device` and still quantised.
///
/// The selection primitive for streamed residency: a model too large for an
/// accelerator keeps its blocks in host memory and calls this as each is
/// reached. Shared between Flux and SD 3 because the naming convention is —
/// both carry the original upstream `<stack>.<i>.<...>` layout.
///
/// The trailing dot is load-bearing. Without it `double_blocks.1` also matches
/// `double_blocks.10` through `19`, and Flux schnell has 19 double and 38
/// single blocks, so every single-digit index has two-digit siblings. The
/// block would be built from whichever duplicate name won, silently.
pub fn block_weights(
    all: &QuantizedWeights,
    path: &str,
    device: &sd_tensor::Device,
) -> Result<QuantizedWeights> {
    let prefix = format!("{path}.");
    let mut out = QuantizedWeights::new();
    for (name, weight) in all.iter() {
        if name.starts_with(&prefix) {
            out.insert(
                name.clone(),
                sd_tensor::quantized::to_device(weight, device)?,
            );
        }
    }
    if out.is_empty() {
        return Err(sd_tensor::Error::Msg(format!(
            "no quantised weights under {prefix}"
        )));
    }
    Ok(out)
}

/// Tensors already dequantised onto the compute device, keyed by GGUF name.
///
/// Biases and norm scales, essentially. In a quantised Flux checkpoint they
/// are 472 of the 776 tensors and **127 MB against the weights' 6.66 GB** —
/// small enough to hold permanently, numerous enough that dequantising them
/// repeatedly dominates everything else.
pub type DenseCache = HashMap<String, Tensor>;

/// Dequantise every dense tensor under `prefixes` onto `device`, once.
///
/// The companion to [`block_weights`] for streamed residency: the quantised
/// matrices stream, and everything dense — biases, norm scales — is prepared
/// here and reused for the life of the model. Only F32/F16/BF16 tensors are
/// taken, which is exactly the small ones; a quantised tensor left out of the
/// cache is a weight, and weights are what streaming is for.
pub fn dense_cache(
    all: &QuantizedWeights,
    prefixes: &[&str],
    device: &sd_tensor::Device,
) -> Result<DenseCache> {
    use sd_tensor::gguf::GgmlDType;
    let mut out = DenseCache::new();
    for (name, weight) in all.iter() {
        if !prefixes.iter().any(|p| name.starts_with(p)) {
            continue;
        }
        if !matches!(
            weight.dtype(),
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        ) {
            continue;
        }
        out.insert(name.clone(), weight.dequantize(device)?);
    }
    Ok(out)
}

/// How many streamed blocks run between releases of device memory.
///
/// **This is the memory/speed dial, and it is the whole trade streaming
/// makes.** Dropping a block frees nothing on Metal by itself: candle pools
/// its buffers and returns the unreferenced ones only inside
/// `drop_unused_buffers`, which runs on synchronise. So the interval between
/// synchronises *is* the peak residency — one block at 1, nineteen blocks at
/// 19.
///
/// Measured on Flux schnell, 512, 4 steps, Metal, against 20.9 s resident:
///
/// ```text
///   sync every  1    29.5 s    ~1 block on the device   (191 MB)
///   sync every  4    26.6 s
///   sync every  8    25.2 s
///   sync every 19    25.3 s    the whole stack pools    (3.6 GB)
/// ```
///
/// The default is 1, because the reason to stream at all is the memory; a
/// caller who wants the seconds back and has the room can raise it. Output is
/// bit-identical at every setting — this changes when buffers are returned,
/// not what is computed.
///
/// Leaving it out entirely is not an option, and that is worth knowing: with
/// no synchronise the pool grows unboundedly and a run that took 25 s
/// degraded to over 60 s per step as the machine began swapping.
pub fn stream_sync_every() -> usize {
    std::env::var("SD_STREAM_SYNC_EVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(1)
}

/// Where a model's weights come from.
#[derive(Clone, Copy)]
pub enum Source<'a> {
    Dense(&'a VarBuilder<'a>),
    Quantized(&'a QuantizedWeights),
    /// Quantised weights, plus dense ones already on the device.
    ///
    /// For streamed residency, where a block is rebuilt on every step. Without
    /// the cache each rebuild dequantises every bias and norm scale afresh —
    /// a dozen small device operations per block — and that, not the weight
    /// copy, is where a streamed step spends its time: measured at 1019 ms of
    /// build against 354 ms of copy and 244 ms of actual arithmetic, per step
    /// over Flux's 19 double blocks.
    ///
    /// Falls back to dequantising when a name is absent, so a partial cache is
    /// a performance question rather than a correctness one.
    QuantizedCached(&'a QuantizedWeights, &'a DenseCache),
}

impl<'a> Source<'a> {
    /// Descend a dotted path, since callers name weights the way the
    /// checkpoint does rather than by chaining `pp`.
    fn at(vb: &VarBuilder<'a>, path: &str) -> VarBuilder<'a> {
        let mut sub = vb.clone();
        for part in path.split('.') {
            if !part.is_empty() {
                sub = sub.pp(part);
            }
        }
        sub
    }

    fn quantized(w: &QuantizedWeights, key: &str) -> Result<Arc<QTensor>> {
        w.get(key)
            .cloned()
            .ok_or_else(|| sd_tensor::Error::Msg(format!("quantised weights are missing {key}")))
    }

    /// A dense tensor, from the cache when it is there and by dequantising
    /// when it is not.
    fn dense(w: &QuantizedWeights, cache: Option<&DenseCache>, key: &str) -> Result<Tensor> {
        if let Some(t) = cache.and_then(|c| c.get(key)) {
            return Ok(t.clone());
        }
        let q = Self::quantized(w, key)?;
        q.dequantize(&q.device())
    }

    /// The quantised weights and dense cache behind this source, if any.
    fn parts(&self) -> Option<(&'a QuantizedWeights, Option<&'a DenseCache>)> {
        match self {
            Self::Dense(_) => None,
            Self::Quantized(w) => Some((w, None)),
            Self::QuantizedCached(w, c) => Some((w, Some(c))),
        }
    }

    /// A projection with a bias.
    pub fn linear(&self, path: &str, in_dim: usize, out_dim: usize) -> Result<Proj> {
        let Some((w, cache)) = self.parts() else {
            let Self::Dense(vb) = self else {
                unreachable!("parts covers the rest")
            };
            return Ok(Proj::Dense(linear(in_dim, out_dim, Self::at(vb, path))?));
        };
        let weight = Self::quantized(w, &format!("{path}.weight"))?;
        // The bias stays dense. It is one value per output feature —
        // negligible beside the weight — and quantising it costs accuracy for
        // no meaningful saving.
        let bias = Self::dense(w, cache, &format!("{path}.bias"))?;
        let q = QLinear::new(weight, Some(bias))?;
        Ok(Proj::Quantized(match runtime_lora::delta_for(path) {
            Some(delta) => q.with_lora(delta),
            None => q,
        }))
    }

    /// A projection with no bias.
    pub fn linear_no_bias(&self, path: &str, in_dim: usize, out_dim: usize) -> Result<Proj> {
        match self {
            Self::Dense(vb) => Ok(Proj::Dense(linear_no_bias(
                in_dim,
                out_dim,
                Self::at(vb, path),
            )?)),
            Self::Quantized(w) | Self::QuantizedCached(w, _) => {
                let q = QLinear::new(Self::quantized(w, &format!("{path}.weight"))?, None)?;
                Ok(Proj::Quantized(match runtime_lora::delta_for(path) {
                    Some(delta) => q.with_lora(delta),
                    None => q,
                }))
            }
        }
    }

    /// A plain tensor, dequantised if necessary.
    ///
    /// For norm scales and embeddings: the former are stored dense in every
    /// checkpoint anyway, and the latter is a lookup rather than a matmul, so
    /// there is nothing for a quantised kernel to do.
    pub fn tensor<S: Into<sd_tensor::Shape>>(&self, path: &str, shape: S) -> Result<Tensor> {
        // `path` names the tensor itself, so the last segment is its leaf.
        let (prefix, leaf) = path.rsplit_once('.').unwrap_or(("", path));
        match self {
            Self::Dense(vb) => Self::at(vb, prefix).get(shape, leaf),
            Self::Quantized(w) => Self::dense(w, None, path),
            Self::QuantizedCached(w, cache) => Self::dense(w, Some(cache), path),
        }
    }
}

/// A LoRA applied at runtime to quantised layers.
///
/// Installed for the duration of a model construction, so each layer can look
/// up its own correction by the path it already knows — the same shape as
/// GLIGEN's fusers, and for the same reason: there is no index to get wrong.
///
/// Ambient rather than a parameter because `Source` is threaded through every
/// model constructor in the crate, and widening it would touch all of them for
/// something only quantised linear layers care about.
pub mod runtime_lora {
    use std::cell::RefCell;

    use sd_tensor::quantized::LoraDelta;

    thread_local! {
        static INSTALLED: RefCell<Option<(sd_loader::Lora, f64)>> = const { RefCell::new(None) };
    }

    /// Removes the adapter when dropped.
    #[must_use = "the adapter is removed when this guard is dropped"]
    pub struct Guard;

    impl Drop for Guard {
        fn drop(&mut self) {
            INSTALLED.with(|s| *s.borrow_mut() = None);
        }
    }

    /// Apply `lora` to quantised layers built while the guard lives.
    pub fn install(lora: sd_loader::Lora, multiplier: f64) -> Guard {
        INSTALLED.with(|s| *s.borrow_mut() = Some((lora, multiplier)));
        Guard
    }

    /// This layer's correction, if the installed adapter has one.
    pub(super) fn delta_for(path: &str) -> Option<LoraDelta> {
        INSTALLED.with(|s| {
            let slot = s.borrow();
            let (lora, multiplier) = slot.as_ref()?;
            let (down, up, scale) = lora.delta_for(path)?;
            Some(LoraDelta {
                down: down.clone(),
                up: up.clone(),
                scale: scale * multiplier,
            })
        })
    }

    /// How many layers the installed adapter covers, for a caller to check
    /// against what it expected — a LoRA that matches nothing applies silently.
    pub fn len() -> usize {
        INSTALLED.with(|s| s.borrow().as_ref().map_or(0, |(l, _)| l.len()))
    }
}
