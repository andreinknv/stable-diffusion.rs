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

use candle_core::quantized::{QMatMul, QStorage, QTensor};
use candle_core::Module;

use crate::{Device, Result, Tensor};

/// Copy a tensor into a fresh buffer if it starts partway into someone else's.
///
/// **This works around a real miscomputation, not a style preference.**
/// candle 0.11's Metal quantised matmul passes the activation buffer to the
/// kernel without adding the layout's `start_offset`, so a tensor that is a
/// view into the middle of a larger allocation is read *from the beginning of
/// that allocation*. No error is raised: the shapes are right, the arithmetic
/// runs, and the answer is the product of the wrong rows.
///
/// The trap is that `contiguous()` does not save you. `narrow` along anything
/// but the last axis of a contiguous tensor produces a layout candle considers
/// contiguous — the elements *are* consecutive, they merely start late — so
/// `contiguous()` is a no-op and the offset survives it. `force_contiguous`
/// is the one that always copies.
///
/// Flux hit this in every double-stream block. Attention runs on the text and
/// image tokens joined, then splits them again:
///
/// ```text
///   txt_attn = attn.narrow(1, 0, 512)          offset 0     -> correct
///   img_attn = attn.narrow(1, 512, 1024)       offset 512*3072 -> read the text rows
/// ```
///
/// Every image-attention projection in all 19 blocks was therefore computed
/// from the text half. The rendered image was a flat orange field. Localising
/// it took a bisection down to one op because every *isolated* check passes —
/// the op is correct whenever its input happens to own its buffer, which is
/// what a freshly constructed test tensor always does.
///
/// Gated on the device because the CPU backend honours the offset correctly
/// and the copy is pure cost there. CUDA is untested here and is included with
/// the non-CPU backends deliberately: an unnecessary copy is cheap, and a
/// silently wrong matmul is what this function exists to prevent.
fn without_storage_offset(xs: &Tensor) -> Result<Tensor> {
    if xs.device().is_cpu() {
        return Ok(xs.clone());
    }
    // Scoped: `storage_and_layout` holds a read lock that `force_contiguous`
    // would deadlock against.
    let offset = {
        let (_storage, layout) = xs.storage_and_layout();
        layout.start_offset()
    };
    if offset == 0 {
        return Ok(xs.clone());
    }
    xs.force_contiguous()
}

/// A linear layer whose weight stays in its quantised form.
///
/// Construct from a [`QTensor`] read out of a GGUF file. The weight is
/// expected in `[out_features, in_features]`, matching both candle's `Linear`
/// and the layout GGUF stores.
pub struct QLinear {
    weight: QMatMul,
    bias: Option<Tensor>,
    resident: usize,
    /// A LoRA applied at *runtime* rather than merged.
    ///
    /// Merging into a quantised weight means dequantising, adding, and
    /// requantising — lossy, and it throws away the compression that made the
    /// model fit in the first place. Adding the correction to the *output*
    /// instead leaves the quantised weight untouched and costs two small dense
    /// matmuls: `x @ down^T` is `[.., rank]` and `@ up^T` brings it back, so
    /// nothing of size `[in, out]` is ever formed.
    lora: Option<LoraDelta>,
}

/// A low-rank correction held as its factors.
#[derive(Debug, Clone)]
pub struct LoraDelta {
    /// `[rank, in]`.
    pub down: Tensor,
    /// `[out, rank]`.
    pub up: Tensor,
    /// `alpha/rank * multiplier`, already folded.
    pub scale: f64,
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
            lora: None,
        })
    }

    /// Attach a runtime LoRA. Consuming, so a layer either has one from
    /// construction or never does.
    pub fn with_lora(mut self, delta: LoraDelta) -> Self {
        self.lora = Some(delta);
        self
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = without_storage_offset(xs)?;
        let out = self.weight.forward(&xs)?;
        let out = match &self.lora {
            None => out,
            Some(delta) => {
                // The factors are applied in order, so the widest intermediate
                // is `[.., rank]` — 4 to 128 wide, against `in` or `out`.
                let dtype = out.dtype();
                let x = xs.to_dtype(delta.down.dtype())?;
                let low = x.broadcast_matmul(&delta.down.t()?)?;
                let correction = low.broadcast_matmul(&delta.up.t()?)?;
                (out + (correction * delta.scale)?.to_dtype(dtype)?)?
            }
        };
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

impl std::fmt::Debug for QLinear {
    // QMatMul holds opaque block storage and is not Debug. Report what is
    // actually useful when a model is printed: how much it costs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QLinear")
            .field("resident_bytes", &self.resident)
            .field("bias", &self.bias.is_some())
            .finish()
    }
}

/// Move a quantised weight to another device, still quantised.
///
/// The enabling primitive for streaming weights: a model too large to sit on
/// an accelerator can keep its blocks in host memory and copy each one across
/// as it is needed. Without this the only route between devices is
/// `dequantize` and re-`quantize`, which is both far more expensive and
/// **lossy** — requantising already-quantised values rounds them a second
/// time.
///
/// This copies the quantised block bytes verbatim and rebuilds the tensor
/// around them, so the result is bit-identical to the source. That also makes
/// it cheap in the way that matters: a Flux block moves about 120 MB rather
/// than the 1 GB it would occupy dequantised.
///
/// Returns the input untouched when it is already on `device`, so callers can
/// use it unconditionally.
pub fn to_device(weight: &Arc<QTensor>, device: &Device) -> Result<Arc<QTensor>> {
    if crate::device::same(&weight.device(), device) {
        return Ok(weight.clone());
    }
    let storage = QStorage::from_data(weight.data()?, device, weight.dtype())?;
    Ok(Arc::new(QTensor::new(storage, weight.shape().clone())?))
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
    fn moving_a_quantised_weight_between_devices_is_lossless() {
        // The whole point of moving block bytes rather than dequantising and
        // requantising: the second route rounds already-rounded values and
        // would drift a little more on every hop. CPU-to-CPU is the only trip
        // a test runner can make, and it still exercises the copy path when
        // the devices differ; when they do not, the tensor must come back
        // untouched rather than be rebuilt.
        let dev = Device::Cpu;
        let w = weight(&dev);
        let q = Arc::new(QTensor::quantize(&w, GgmlDType::Q4K).unwrap());

        let same = to_device(&q, &Device::Cpu).unwrap();
        assert!(Arc::ptr_eq(&q, &same), "a no-op move must not copy");

        // Rebuild through the byte path explicitly and check it is identical.
        let rebuilt = Arc::new(
            QTensor::new(
                QStorage::from_data(q.data().unwrap(), &dev, q.dtype()).unwrap(),
                q.shape().clone(),
            )
            .unwrap(),
        );
        let a = q.dequantize(&dev).unwrap();
        let b = rebuilt.dequantize(&dev).unwrap();
        let diff = (&a - &b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0, "moving quantised bytes must be bit-exact");
        assert_eq!(rebuilt.dtype(), q.dtype());
        assert_eq!(rebuilt.shape(), q.shape());
    }

    #[test]
    fn a_narrowed_activation_gives_the_same_answer_as_its_own_copy() {
        // The Flux/Metal corruption in one assertion. `narrow` along anything
        // but the last axis yields a layout candle calls contiguous — the
        // elements are consecutive, they just start late — so `contiguous()`
        // is a no-op and the tensor keeps a non-zero `start_offset`. candle
        // 0.11's Metal quantised matmul does not add that offset, so it reads
        // from the beginning of the buffer and returns the product of the
        // wrong rows, with no error anywhere. See `without_storage_offset`.
        //
        // On a CPU runner both sides take the same path and this only pins the
        // contract. It bites on `--features metal`, and `--example
        // metal_check` runs the same comparison across devices.
        let dev = Device::Cpu;
        let w = weight(&dev);
        let q = QLinear::new(
            Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0).unwrap()),
            None,
        )
        .unwrap();

        let big = Tensor::rand(-1f32, 1f32, (1, 12, 256), &dev).unwrap();
        // Offset deliberately non-zero: rows 4..12 start 4*256 elements in.
        let view = big.narrow(1, 4, 8).unwrap().contiguous().unwrap();
        let copied = view.force_contiguous().unwrap();

        // The premise: these hold the same numbers.
        let diff = (&view - &copied)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            diff, 0.0,
            "the two inputs must be the same tensor of values"
        );

        let from_view = q.forward(&view).unwrap();
        let from_copy = q.forward(&copied).unwrap();
        let out_diff = (&from_view - &from_copy)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            out_diff, 0.0,
            "a quantised matmul must not depend on where its input sits in memory"
        );
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

#[cfg(test)]
mod lora_tests {
    use super::*;
    use crate::{Device, Tensor};

    /// A quantised layer and the same weight dense, so the two can be compared.
    fn pair(in_dim: usize, out_dim: usize) -> (QLinear, Tensor, Device) {
        let dev = Device::Cpu;
        let dense = Tensor::randn(0f32, 0.05, (out_dim, in_dim), &dev).unwrap();
        let q = candle_core::quantized::QTensor::quantize(
            &dense,
            candle_core::quantized::GgmlDType::Q8_0,
        )
        .unwrap();
        let layer = QLinear::new(std::sync::Arc::new(q), None).unwrap();
        (layer, dense, dev)
    }

    #[test]
    fn a_runtime_lora_matches_merging_it_into_the_dense_weight() {
        // The property the whole runtime path exists for: applying the
        // correction to the *output* must equal applying it to the *weight*,
        // so nothing is given up by not dequantising.
        //
        // Compared against a dense merge rather than against nothing, because
        // "it changed the output" would pass for any wrong scale or transpose.
        let (in_dim, out_dim, rank) = (64usize, 32usize, 4usize);
        let (layer, dense, dev) = pair(in_dim, out_dim);

        let down = Tensor::randn(0f32, 0.1, (rank, in_dim), &dev).unwrap();
        let up = Tensor::randn(0f32, 0.1, (out_dim, rank), &dev).unwrap();
        let scale = 0.75f64;

        let layer_ref = QLinear::new(
            std::sync::Arc::new(
                candle_core::quantized::QTensor::quantize(
                    &dense,
                    candle_core::quantized::GgmlDType::Q8_0,
                )
                .unwrap(),
            ),
            None,
        )
        .unwrap();
        let with_lora = layer.with_lora(LoraDelta {
            down: down.clone(),
            up: up.clone(),
            scale,
        });
        let xs = Tensor::randn(0f32, 1.0, (2, in_dim), &dev).unwrap();
        let got = with_lora.forward(&xs).unwrap();

        // The same correction merged into the dense weight, then run densely.
        let delta = (up.matmul(&down).unwrap() * scale).unwrap();
        let merged = (&dense + &delta).unwrap();
        let want = xs.matmul(&merged.t().unwrap()).unwrap();

        // The quantiser's own error, measured rather than assumed: the same
        // layer with no LoRA at all, against the same dense weight. Whatever
        // that is, the LoRA path must not exceed it — the correction is
        // computed in f32 and never touches the quantiser, so it should add
        // nothing.
        let plain = layer_ref.forward(&xs).unwrap();
        let plain_want = xs.matmul(&dense.t().unwrap()).unwrap();
        let worst = |a: &Tensor, b: &Tensor| {
            let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            a.iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        let floor = worst(&plain, &plain_want);
        let with = worst(&got, &want);
        println!("quantiser noise {floor:.3e}, with runtime LoRA {with:.3e}");
        assert!(
            with <= floor * 1.5,
            "the LoRA path added error beyond the quantiser: {with:.3e} against a \
             {floor:.3e} floor"
        );
    }

    #[test]
    fn scale_zero_is_bit_identical_to_no_lora() {
        // What makes the strength safe to expose, and a check that the
        // correction is *added* rather than replacing the quantised output.
        let (in_dim, out_dim, rank) = (64usize, 32usize, 4usize);
        let (layer, _dense, dev) = pair(in_dim, out_dim);
        let xs = Tensor::randn(0f32, 1.0, (2, in_dim), &dev).unwrap();
        let plain = layer.forward(&xs).unwrap();

        let zeroed = layer.with_lora(LoraDelta {
            down: Tensor::randn(0f32, 0.1, (rank, in_dim), &dev).unwrap(),
            up: Tensor::randn(0f32, 0.1, (out_dim, rank), &dev).unwrap(),
            scale: 0.0,
        });
        let got = zeroed.forward(&xs).unwrap();
        assert_eq!(
            plain.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
    }

    #[test]
    fn the_widest_intermediate_is_the_rank_not_the_layer() {
        // The reason this is worth doing at all: never forming `up @ down`.
        // A 4096x4096 layer with rank 8 would need 64 MB for the product and
        // needs 128 KB for the factors.
        let (in_dim, out_dim, rank) = (4096usize, 4096usize, 8usize);
        let factors = (rank * in_dim + out_dim * rank) * 4;
        let product = in_dim * out_dim * 4;
        assert!(
            product / factors > 200,
            "the saving should be large: {product} vs {factors}"
        );
    }
}
