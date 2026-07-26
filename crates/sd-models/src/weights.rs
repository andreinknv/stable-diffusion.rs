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

/// Where a model's weights come from.
#[derive(Clone, Copy)]
pub enum Source<'a> {
    Dense(&'a VarBuilder<'a>),
    Quantized(&'a QuantizedWeights),
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

    /// A projection with a bias.
    pub fn linear(&self, path: &str, in_dim: usize, out_dim: usize) -> Result<Proj> {
        match self {
            Self::Dense(vb) => Ok(Proj::Dense(linear(in_dim, out_dim, Self::at(vb, path))?)),
            Self::Quantized(w) => {
                let weight = Self::quantized(w, &format!("{path}.weight"))?;
                // The bias stays dense. It is one value per output feature —
                // negligible beside the weight — and quantising it costs
                // accuracy for no meaningful saving.
                let bias = Self::quantized(w, &format!("{path}.bias"))?;
                let bias = bias.dequantize(&bias.device())?;
                Ok(Proj::Quantized(QLinear::new(weight, Some(bias))?))
            }
        }
    }

    /// A projection with no bias.
    pub fn linear_no_bias(&self, path: &str, in_dim: usize, out_dim: usize) -> Result<Proj> {
        match self {
            Self::Dense(vb) => Ok(Proj::Dense(linear_no_bias(
                in_dim,
                out_dim,
                Self::at(vb, path),
            )?)),
            Self::Quantized(w) => Ok(Proj::Quantized(QLinear::new(
                Self::quantized(w, &format!("{path}.weight"))?,
                None,
            )?)),
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
            Self::Quantized(w) => {
                let t = Self::quantized(w, path)?;
                t.dequantize(&t.device())
            }
        }
    }
}
