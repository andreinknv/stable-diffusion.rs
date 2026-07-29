//! Apple's own convolution, via MetalPerformanceShadersGraph.
//!
//! Measurement code, not a model path. candle's Metal `conv2d` is im2col plus
//! a matmul: for a 3x3 it materialises a buffer nine times the input — 283 MB
//! at `[2, 960, 64, 64]` — writes it and reads it straight back, at about
//! 25 GB/s. A direct convolution never builds it. This exists to find out what
//! that is worth before anything is built on the answer.
//!
//! MPSGraph ships in macOS; `objc2-metal-performance-shaders-graph` is the
//! Rust declaration of a framework already on the machine, not new code.

use crate::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};
use candle_core::backend::BackendStorage;
use candle_core::{DType, MetalStorage};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::{NSArray, NSMutableDictionary, NSNumber};
use objc2_metal::{MTLBuffer, MTLDevice};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphConvolution2DOpDescriptor, MPSGraphPaddingStyle, MPSGraphTensor,
    MPSGraphTensorData, MPSGraphTensorNamedDataLayout,
};

/// `[b, ci, h, w]` convolved with `[co, ci, k, k]`, stride 1.
pub struct MpsConv2d {
    pub padding: usize,
}

fn shape(dims: &[usize]) -> Retained<NSArray<NSNumber>> {
    let ns: Vec<Retained<NSNumber>> = dims.iter().map(|d| NSNumber::new_usize(*d)).collect();
    NSArray::from_retained_slice(&ns)
}

impl CustomOp2 for MpsConv2d {
    fn name(&self) -> &'static str {
        "mps-conv2d"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "mps conv2d: Metal only — this is a measurement against candle's im2col".into(),
        ))
    }

    fn metal_fwd(
        &self,
        xs: &MetalStorage,
        xl: &Layout,
        ws: &MetalStorage,
        wl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (b, ci, h, w) = xl.shape().dims4()?;
        let (co, wci, kh, kw) = wl.shape().dims4()?;
        if ci != wci {
            return Err(candle_core::Error::Msg(format!(
                "mps conv2d: input has {ci} channels, weights expect {wci}"
            )));
        }
        if !xl.is_contiguous() || !wl.is_contiguous() {
            return Err(candle_core::Error::Msg(
                "mps conv2d: both inputs must be contiguous".into(),
            ));
        }
        let oh = h + 2 * self.padding - kh + 1;
        let ow = w + 2 * self.padding - kw + 1;

        let device = xs.device().clone();
        let out = device.new_buffer(b * co * oh * ow, DType::F32, "mps-conv2d")?;

        // MPSGraph runs on its own command queue, so anything candle has
        // merely *enqueued* into the input buffers has not necessarily
        // executed yet. Nothing orders the two queues against each other, and
        // reading a buffer whose producing kernel is still in flight returns
        // whatever was there before — silently, and only once the tensors are
        // large enough for the work to still be pending.
        device.wait_until_completed()?;

        // SAFETY: every pointer handed to MPSGraph below is a live Metal
        // buffer owned by `xs`, `ws` or `out`, each outliving this call; the
        // shapes passed match the buffers' actual extents, which is what the
        // dims4 checks above establish.
        unsafe {
            let raw = device.device().as_ref();
            let queue = raw
                .newCommandQueue()
                .ok_or_else(|| candle_core::Error::Msg("mps conv2d: no command queue".into()))?;

            let graph = MPSGraph::new();
            let src = graph.placeholderWithShape_dataType_name(
                Some(&shape(&[b, ci, h, w])),
                MPSDataType::Float32,
                None,
            );
            let wt = graph.placeholderWithShape_dataType_name(
                Some(&shape(&[co, ci, kh, kw])),
                MPSDataType::Float32,
                None,
            );
            let desc = MPSGraphConvolution2DOpDescriptor::
                descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
                    1, 1, 1, 1, 1,
                    self.padding, self.padding, self.padding, self.padding,
                    MPSGraphPaddingStyle::Explicit,
                    MPSGraphTensorNamedDataLayout::NCHW,
                    MPSGraphTensorNamedDataLayout::OIHW,
                ).ok_or_else(|| candle_core::Error::Msg("mps conv2d: descriptor".into()))?;
            let res = graph.convolution2DWithSourceTensor_weightsTensor_descriptor_name(
                &src, &wt, &desc, None,
            );

            let feed = |t: &MetalStorage, dims: &[usize]| {
                MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                    MPSGraphTensorData::alloc(),
                    t.buffer().as_ref(),
                    &shape(dims),
                    MPSDataType::Float32,
                )
            };
            let feeds = NSMutableDictionary::<MPSGraphTensor, MPSGraphTensorData>::new();
            feeds.setObject_forKey(&feed(xs, &[b, ci, h, w]), ProtocolObject::from_ref(&*src));
            feeds.setObject_forKey(&feed(ws, &[co, ci, kh, kw]), ProtocolObject::from_ref(&*wt));

            // The result lands straight in our own buffer, so nothing is
            // copied back out and the timing is the convolution alone.
            // `out` is an `Arc<Buffer>`, so the conversion has to be spelled
            // out or `as_ref` resolves to the `Arc`'s.
            let out_buf: &ProtocolObject<dyn MTLBuffer> = (*out).as_ref();
            let out_data = MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                MPSGraphTensorData::alloc(),
                out_buf,
                &shape(&[b, co, oh, ow]),
                MPSDataType::Float32,
            );
            let results = NSMutableDictionary::<MPSGraphTensor, MPSGraphTensorData>::new();
            results.setObject_forKey(&out_data, ProtocolObject::from_ref(&*res));

            graph.runWithMTLCommandQueue_feeds_targetOperations_resultsDictionary(
                ProtocolObject::from_ref(&*queue),
                &feeds,
                None,
                &results,
            );
        }

        Ok((
            MetalStorage::new(out, device, b * co * oh * ow, DType::F32),
            Shape::from((b, co, oh, ow)),
        ))
    }
}

/// Convolve with Apple's implementation. Metal and f32 only.
pub fn conv2d(xs: &Tensor, weight: &Tensor, padding: usize) -> Result<Tensor> {
    xs.apply_op2_no_bwd(weight, &MpsConv2d { padding })
}
