//! Loading a checkpoint **quantised at rest**.
//!
//! This is what makes a 12B-parameter transformer runnable on a 36 GB machine.
//! Flux schnell is 47.6 GB dense in f32 and 23.8 GB in f16 — neither fits
//! beside the activations. Held in MLX's own 4-bit affine quantisation with
//! group-64 scales it is about 6.7 GB, and the dequantisation happens inside
//! the matmul kernel a tile at a time rather than materialising the weight.
//!
//! # The peak is one tensor, not one model
//!
//! `Gguf::open` reads only the header and the tensor directory. Each tensor is
//! then dequantised from GGUF, immediately requantised into MLX's scheme, and
//! the dense form dropped before the next one is read. So the largest
//! allocation is the largest single tensor — a few hundred megabytes — and not
//! the model.
//!
//! **This is the whole reason the loader is written as a loop over the
//! directory rather than as `mlx_gguf::load` followed by a conversion.** The
//! eager form is one line shorter and needs 47.6 GB.
//!
//! # Quantising from a GGUF stacks two lossy steps
//!
//! The values reaching `quantize` here have already been through Q4_K, so the
//! error compounds: `docs/handoff.md` measures a cosine of ~0.995 against the
//! GGUF values, which is itself against the original f16. **Quantise from the
//! original checkpoint where one is available** — `from_safetensors` does that
//! — and take this path only when the GGUF is all there is.
//!
//! # What stays dense, and why
//!
//! Only 2-D weights are quantised. Norm scales, biases and embeddings are 1-D
//! or small, quantising them saves nothing measurable, and MLX's kernel wants a
//! group size that divides the input width — which a 1-D tensor does not have.
//! A weight whose input width is not a multiple of the group size also stays
//! dense rather than being padded, because padding changes the arithmetic.

use std::collections::HashMap;
use std::path::Path;

use sd_tensor::mlx::{Array, QuantizedArray, Stream};
use sd_tensor::{mlx_gguf, Error, Result};

use super::Weights;

/// MLX's default group size. Must divide the weight's input width.
pub const GROUP_SIZE: usize = 64;
/// Bits per weight. 4 is what keeps Flux schnell resident; 8 is near-lossless
/// and roughly doubles the footprint.
pub const DEFAULT_BITS: usize = 4;

/// A checkpoint where the large weights are quantised and the rest are not.
///
/// Both maps are consulted by name, so a caller asks for a tensor without
/// knowing which side it landed on.
pub struct QuantizedWeights {
    /// 2-D weights held in MLX's 4-bit affine form.
    pub quantized: HashMap<String, QuantizedArray>,
    /// Everything else: norms, biases, embeddings, and any weight whose input
    /// width does not divide by [`GROUP_SIZE`].
    pub dense: Weights,
}

impl QuantizedWeights {
    /// Total resident bytes, quantised and dense together.
    ///
    /// The honest figure: packed weights *plus* their scales and biases, which
    /// at group 64 are a real 6 % on top of the bits.
    pub fn resident_bytes(&self) -> usize {
        let q: usize = self.quantized.values().map(|a| a.resident_bytes()).sum();
        let d: usize = self.dense.values().map(|a| a.elem_count() * 4).sum();
        q + d
    }

    pub fn len(&self) -> usize {
        self.quantized.len() + self.dense.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether a name resolves at all, on either side.
    pub fn contains_key(&self, name: &str) -> bool {
        self.quantized.contains_key(name) || self.dense.contains_key(name)
    }
}

/// Tensors that need more bits than the rest.
///
/// **Not every parameter tolerates 4 bits equally.** A modulation layer emits
/// the shift, scale and gate that *multiply* the residual stream, so an error
/// there is multiplied by everything downstream rather than added to it. The
/// input and output projections sit outside the residual and are read once
/// each, with nothing after them to average the error away.
///
/// Measured on flux-mini, whole-transformer cosine against the dense run, with
/// resident size as a fraction of dense f32:
///
/// ```text
///   4-bit everywhere            0.9334   15.6 %
///   8-bit everywhere            0.9993   28.1 %
///   these layers dense, rest 4  0.9924   39.0 %   <- worse than 8-bit
///   these layers 8, rest 4      0.9919   19.1 %   <- the default
/// ```
///
/// The third row is why the policy assigns a *bit width* rather than a
/// boolean: keeping these dense recovers the accuracy and costs more than
/// simply using 8 bits throughout, which would make the policy pointless. At 8
/// bits they cost 3.5 points of size and buy back nearly six points of cosine.
pub fn sensitive(name: &str) -> bool {
    name.contains("_mod.lin")
        || name.contains("modulation")
        || name.contains("_embedder")
        || name.starts_with("img_in")
        || name.starts_with("txt_in")
        || name.starts_with("final_layer")
}

/// The default policy: [`sensitive`] layers at 8 bits, everything else at
/// `bits`.
pub fn mixed(bits: usize) -> impl Fn(&str) -> usize {
    move |name: &str| if sensitive(name) { 8 } else { bits }
}

/// Is this tensor worth quantising?
///
/// 2-D, and its input width a multiple of the group size. Everything else stays
/// dense — see the module docs.
fn quantizable(shape: &[usize]) -> bool {
    shape.len() == 2 && shape[1] % GROUP_SIZE == 0
}

/// Load a GGUF checkpoint into MLX, quantising the large weights as they are
/// read.
///
/// The dense form of the whole model is never allocated: each tensor is
/// dequantised from GGUF, requantised, and dropped.
pub fn from_gguf(path: &Path, bits: usize, s: &Stream) -> Result<QuantizedWeights> {
    let mut g = mlx_gguf::Gguf::open(path)?;
    let infos = g.tensors.clone();
    let mut quantized = HashMap::new();
    let mut dense: Weights = HashMap::new();

    for info in &infos {
        // One tensor's dense values, alive only until the end of this
        // iteration. This is the peak.
        let values = g.dequantize(info)?;
        let array = Array::from_slice_f32(&values, &info.shape)?;
        drop(values);

        let want = if sensitive(&info.name) { 8 } else { bits };
        if quantizable(&info.shape) {
            let q = QuantizedArray::quantize(&array, GROUP_SIZE, want, s)?;
            // **Evaluate before dropping the dense source.** MLX is lazy, so
            // without this the quantisation is a graph node holding a
            // reference to every dense tensor in the file — which is the 47.6
            // GB this exists to avoid, arrived at by a different route.
            sd_tensor::mlx::eval(&[&q.weight, &q.scales, &q.biases])?;
            quantized.insert(info.name.clone(), q);
        } else {
            dense.insert(info.name.clone(), array.contiguous(s)?);
        }
    }
    if quantized.is_empty() && dense.is_empty() {
        return Err(Error::Msg(format!(
            "gguf: {} carries no tensors",
            path.display()
        )));
    }
    Ok(QuantizedWeights { quantized, dense })
}

/// Load a GGUF whose names need translating, quantising as it reads.
///
/// `rename` maps a file name to the name the model asks for, and returns
/// `None` for a tensor that is not part of this model. `sd_loader::t5_key` is
/// one such mapping and `sd_loader::ldm` holds the others — they are pure
/// string logic, so both backends call the same ones and cannot come to
/// disagree about which tensor is which.
pub fn from_gguf_renamed(
    path: &Path,
    bits: usize,
    rename: impl Fn(&str) -> Option<String>,
    s: &Stream,
) -> Result<QuantizedWeights> {
    let mut g = mlx_gguf::Gguf::open(path)?;
    let infos = g.tensors.clone();
    let mut quantized = HashMap::new();
    let mut dense: Weights = HashMap::new();

    for info in &infos {
        let Some(name) = rename(&info.name) else {
            continue;
        };
        let values = g.dequantize(info)?;
        let array = Array::from_slice_f32(&values, &info.shape)?;
        drop(values);

        let want = if sensitive(&name) { 8 } else { bits };
        if want >= 2 && quantizable(&info.shape) {
            let q = QuantizedArray::quantize(&array, GROUP_SIZE, want, s)?;
            sd_tensor::mlx::eval(&[&q.weight, &q.scales, &q.biases])?;
            quantized.insert(name, q);
        } else {
            dense.insert(name, array.contiguous(s)?);
        }
    }
    if quantized.is_empty() && dense.is_empty() {
        return Err(Error::Msg(format!(
            "gguf: {} carries no tensors this model recognises",
            path.display()
        )));
    }
    Ok(QuantizedWeights { quantized, dense })
}

/// Quantise an already-loaded dense map.
///
/// **Preferred over [`from_gguf`] when the original checkpoint is available**,
/// because it quantises once rather than on top of GGUF's own loss. It does
/// require the dense weights to fit, which is the trade.
pub fn from_dense(w: &Weights, bits: usize, s: &Stream) -> Result<QuantizedWeights> {
    from_dense_with(w, mixed(bits), s)
}

/// [`from_dense`] with an explicit per-tensor bit width.
///
/// `bits_for` returns the width for a tensor; **0 keeps it dense**. See
/// [`mixed`] for the default and [`sensitive`] for why it is not uniform.
pub fn from_dense_with(
    w: &Weights,
    bits_for: impl Fn(&str) -> usize,
    s: &Stream,
) -> Result<QuantizedWeights> {
    let mut quantized = HashMap::new();
    let mut dense: Weights = HashMap::new();
    for (name, array) in w {
        let bits = bits_for(name);
        if bits >= 2 && quantizable(&array.shape()) {
            let q = QuantizedArray::quantize(array, GROUP_SIZE, bits, s)?;
            sd_tensor::mlx::eval(&[&q.weight, &q.scales, &q.biases])?;
            quantized.insert(name.clone(), q);
        } else {
            dense.insert(name.clone(), array.contiguous(s)?);
        }
    }
    Ok(QuantizedWeights { quantized, dense })
}

/// A linear layer over weights that may be on either side of the split.
///
/// The quantised path is `mlx_quantized_matmul`, which dequantises inside the
/// kernel; the dense path is the ordinary one. A caller writes one call and
/// does not care which.
pub fn linear(
    x: &Array,
    name: &str,
    bias: Option<&Array>,
    w: &QuantizedWeights,
    s: &Stream,
) -> Result<Array> {
    let y = if let Some(q) = w.quantized.get(name) {
        q.matmul(x, s)?
    } else {
        let dense = w
            .dense
            .get(name)
            .ok_or_else(|| Error::Msg(format!("mlx: {name} is in neither map")))?;
        super::linear(x, dense, None, s)?
    };
    match bias {
        Some(b) => y.add(b, s),
        None => Ok(y),
    }
}

/// Where a model's weights come from, dense or quantised.
///
/// One trait so a forward pass is written once. The alternative — a second copy
/// of `flux::forward` differing only in which `linear` it calls — is how the
/// two would come to disagree about, say, which projection carries a bias.
pub trait WeightSource {
    /// `x @ w[name].T + bias`, by whatever kernel suits how `name` is held.
    fn linear(&self, x: &Array, name: &str, bias: Option<&Array>, s: &Stream) -> Result<Array>;

    /// A tensor that is always dense: a norm scale, a bias, an embedding.
    ///
    /// **Not for weights.** Asking for a quantised one here is an error rather
    /// than a silent dequantisation, because a caller that reaches past
    /// [`Self::linear`] is the caller that materialises the model.
    fn dense(&self, name: &str) -> Result<&Array>;

    /// `dense`, but absent is `None` rather than an error — for optional
    /// biases.
    fn optional(&self, name: &str) -> Option<&Array>;
}

impl WeightSource for Weights {
    fn linear(&self, x: &Array, name: &str, bias: Option<&Array>, s: &Stream) -> Result<Array> {
        let w = self
            .get(name)
            .ok_or_else(|| Error::Msg(format!("mlx: missing tensor {name}")))?;
        super::linear(x, w, bias, s)
    }

    fn dense(&self, name: &str) -> Result<&Array> {
        self.get(name)
            .ok_or_else(|| Error::Msg(format!("mlx: missing tensor {name}")))
    }

    fn optional(&self, name: &str) -> Option<&Array> {
        self.get(name)
    }
}

impl WeightSource for QuantizedWeights {
    fn linear(&self, x: &Array, name: &str, bias: Option<&Array>, s: &Stream) -> Result<Array> {
        linear(x, name, bias, self, s)
    }

    fn dense(&self, name: &str) -> Result<&Array> {
        self.dense.get(name).ok_or_else(|| {
            if self.quantized.contains_key(name) {
                Error::Msg(format!(
                    "mlx: {name} is quantised; reach it through `linear` rather than \
                     dequantising it"
                ))
            } else {
                Error::Msg(format!("mlx: missing tensor {name}"))
            }
        })
    }

    fn optional(&self, name: &str) -> Option<&Array> {
        self.dense.get(name)
    }
}
