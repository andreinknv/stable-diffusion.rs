//! Linear layers that keep their weights quantised.
//!
//! Everywhere else in this workspace a GGUF weight is dequantised to f32 at
//! load, which means quantisation buys disk and nothing else — SD 1.5 occupies
//! the same 4.26 GB of RAM whether the file is Q4_0 or f32. That is affordable
//! for SD 1.5 and disqualifying for anything larger: Flux is 12B parameters,
//! or 48 GB of f32, which no amount of quantisation in the *file* would fix.
//!
//! [`QLinear`] holds the quantised block data and dequantises per matmul
//! instead, so residency tracks the file rather than the parameter count.
//!
//! This is deliberately only the linear layer. Convolutions are not here
//! because block quantisation does not apply to them — a 3x3 kernel's fastest
//! axis is 3 wide, so every published checkpoint stores conv weights at F16
//! already. Transformer architectures (MMDiT, T5) are almost entirely linear,
//! which is why this is the piece that unblocks them.

use std::sync::Arc;

use candle_core::quantized::{QMatMul, QTensor};
use candle_core::Module;

use crate::{Result, Tensor};

/// A linear layer whose weight stays in its quantised form.
///
/// Construct from a [`QTensor`] read out of a GGUF file. The weight is
/// expected in `[out_features, in_features]`, matching both candle's `Linear`
/// and the layout GGUF stores.
pub struct QLinear {
    weight: QMatMul,
    bias: Option<Tensor>,
    resident: usize,
}

impl QLinear {
    /// Wrap a quantised weight, optionally with a bias.
    ///
    /// The bias stays dense: it is one value per output feature, so it is
    /// negligible next to the weight and quantising it costs accuracy for no
    /// meaningful saving. Published checkpoints disagree on this — the stock
    /// SD 1.5 Q4_0 file does quantise biases — but nothing forces us to.
    pub fn new(weight: Arc<QTensor>, bias: Option<Tensor>) -> Result<Self> {
        let resident = weight.storage_size_in_bytes();
        Ok(Self {
            weight: QMatMul::from_arc(weight)?,
            bias,
            resident,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.weight.forward(xs)?;
        match &self.bias {
            // Broadcast: the bias is [out], the activation [.., out].
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }

    /// Bytes of weight data actually held in memory.
    ///
    /// This is the number the whole module exists to move, so it is worth
    /// being able to assert on rather than infer from a process RSS reading.
    pub fn resident_bytes(&self) -> usize {
        self.resident
    }
}

/// What the same weight would cost dequantised, for comparison.
pub fn dequantised_bytes(weight: &QTensor) -> usize {
    weight.shape().elem_count() * std::mem::size_of::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Device};
    use candle_core::quantized::GgmlDType;
    use candle_core::IndexOp;

    /// Shapes divisible by 256 so k-quants apply — see the roadmap for why
    /// SD 1.5's 320-channel blocks do not get this luxury.
    fn weight(dev: &Device) -> Tensor {
        Tensor::rand(-1f32, 1f32, (512, 256), dev).unwrap()
    }

    #[test]
    fn quantised_linear_matches_a_dense_matmul() {
        let dev = Device::Cpu;
        let w = weight(&dev);
        let xs = Tensor::rand(-1f32, 1f32, (4, 256), &dev).unwrap();

        let dense = xs.matmul(&w.t().unwrap()).unwrap();
        let q = QLinear::new(
            Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0).unwrap()),
            None,
        )
        .unwrap();
        let got = q.forward(&xs).unwrap();

        assert_eq!(got.dims(), dense.dims());

        // Compare relative to the output's own scale. A 256-long dot product
        // accumulates quantisation noise from every term, so absolute error
        // grows with the contraction length and an absolute bound would just
        // encode this shape. The same reasoning as the rtol comparison in
        // `golden_clip_encoder.rs`.
        let mean_abs = |t: &Tensor| {
            t.abs()
                .unwrap()
                .mean_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        let relative = mean_abs(&(&got - &dense).unwrap()) / mean_abs(&dense);
        assert!(
            relative < 0.02,
            "quantised matmul diverged: {:.4} relative error",
            relative
        );

        // The bound above tolerates noise, so on its own it would also accept
        // a systematically wrong result of the right magnitude. Correlation
        // is what rules out a transposed axis: it collapses toward zero for a
        // wrong orientation however plausible the scale.
        let a = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = dense.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let n = a.len() as f32;
        let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
        let cov: f32 = a.iter().zip(&b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let va: f32 = a.iter().map(|x| (x - ma).powi(2)).sum::<f32>().sqrt();
        let vb: f32 = b.iter().map(|y| (y - mb).powi(2)).sum::<f32>().sqrt();
        let corr = cov / (va * vb);
        assert!(corr > 0.999, "quantised matmul lost structure: corr {corr}");
    }

    #[test]
    fn bias_is_applied() {
        let dev = Device::Cpu;
        let w = weight(&dev);
        let xs = Tensor::rand(-1f32, 1f32, (4, 256), &dev).unwrap();
        let b = Tensor::rand(-1f32, 1f32, 512, &dev).unwrap();

        let qw = Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0).unwrap());
        let without = QLinear::new(qw.clone(), None)
            .unwrap()
            .forward(&xs)
            .unwrap();
        let with = QLinear::new(qw, Some(b.clone()))
            .unwrap()
            .forward(&xs)
            .unwrap();

        let delta = (&with - &without).unwrap();
        // Every row must differ from the no-bias result by exactly the bias.
        let err = (&delta.i(0).unwrap() - &b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(err < 1e-5, "bias not added along the feature axis: {err}");
    }

    #[test]
    fn residency_tracks_the_quantisation_not_the_parameter_count() {
        let dev = Device::Cpu;
        let w = weight(&dev);
        let dense = w.elem_count() * std::mem::size_of::<f32>();

        for (dtype, max_ratio) in [(GgmlDType::Q4K, 0.20), (GgmlDType::Q8_0, 0.30)] {
            let qt = QTensor::quantize(&w, dtype).unwrap();
            let q = QLinear::new(Arc::new(qt), None).unwrap();
            let ratio = q.resident_bytes() as f64 / dense as f64;
            assert!(
                ratio < max_ratio,
                "{dtype:?} held {} of {dense} bytes ({ratio:.3}) — that is the \
                 dequantise-on-load behaviour this type exists to avoid",
                q.resident_bytes()
            );
        }
    }

    #[test]
    fn f16_weights_are_dequantised_because_no_quantised_kernel_applies() {
        // candle materialises F16/F32 QTensors rather than pretending they
        // have block structure. Worth pinning: it means a checkpoint whose
        // tensors all fell back to F16 gets no residency win, which is
        // exactly the SD 1.5 k-quant case.
        let dev = Device::Cpu;
        let w = weight(&dev);
        let qt = QTensor::quantize(&w.to_dtype(DType::F16).unwrap(), GgmlDType::F16).unwrap();
        let q = QLinear::new(Arc::new(qt), None).unwrap();
        assert!(matches!(q.weight, QMatMul::Tensor(_)));
    }
}
