//! Is `candle_nn::ops::layer_norm` worth adopting for `ops::plain_layer_norm`?
//!
//! The same question that was asked of `rms_norm` and answered "no": candle's
//! fused kernel sums each row with a sequential `.sum::<f32>()` where
//! `mean_keepdim` reduces in blocks, so it trades accuracy for speed, and the
//! trade gets worse as rows get longer. This measures both against an f64
//! reference computed from the same inputs, at the shapes Flux and SD 3
//! actually use.
//!
//! Run: `cargo run --release -p sd-tensor --example layer_norm_bench`

use candle_core::{DType, Device, Tensor, D};

const EPS: f64 = 1e-6;

/// The port's implementation, mirroring `ops::plain_layer_norm`.
fn ours(xs: &Tensor) -> candle_core::Result<Tensor> {
    let xs32 = xs.to_dtype(DType::F32)?;
    let mean = xs32.mean_keepdim(D::Minus1)?;
    let centred = xs32.broadcast_sub(&mean)?;
    let var = centred.sqr()?.mean_keepdim(D::Minus1)?;
    centred.broadcast_div(&(var + EPS)?.sqrt()?)
}

/// The same arithmetic in f64: neither implementation's noise floor, just the
/// answer both are trying to reach.
fn reference(xs: &Tensor) -> candle_core::Result<Tensor> {
    let xs64 = xs.to_dtype(DType::F64)?;
    let mean = xs64.mean_keepdim(D::Minus1)?;
    let centred = xs64.broadcast_sub(&mean)?;
    let var = centred.sqr()?.mean_keepdim(D::Minus1)?;
    centred.broadcast_div(&(var + EPS)?.sqrt()?)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> candle_core::Result<f64> {
    let a = a.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
    let b = b.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
    Ok(a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max))
}

fn main() -> candle_core::Result<()> {
    // Accuracy is measured on CPU because Metal has no f64, so there would be
    // no reference to measure against. Timing is measured on both, because the
    // default device is Metal and a CPU-only speed answer would be about a
    // path most runs do not take.
    let dev = Device::Cpu;
    // (batch, tokens, width), named for where they occur.
    let shapes: [(usize, usize, usize, &str); 4] = [
        (1, 1536, 3072, "Flux MMDiT @512"),
        (1, 1170, 1536, "SD 3.5 MMDiT @512"),
        (1, 154, 4096, "T5 encoder"),
        (1, 77, 768, "CLIP encoder"),
    ];

    println!(
        "{:<22} {:>12} {:>12}   {:>9} {:>9} {:>7}",
        "shape", "ours", "candle", "ours", "candle", "speed"
    );
    for (b, n, w, name) in shapes {
        let xs = Tensor::randn(0f32, 1.0, (b, n, w), &dev)?;
        let alpha = Tensor::ones(w, DType::F32, &dev)?;
        let beta = Tensor::zeros(w, DType::F32, &dev)?;

        let want = reference(&xs)?;
        let a = ours(&xs)?;
        let c = candle_nn::ops::layer_norm(&xs, &alpha, &beta, EPS as f32)?;
        let (ea, ec) = (max_abs_diff(&a, &want)?, max_abs_diff(&c, &want)?);

        // Time a handful of calls each; these are cheap, so repeat enough to
        // clear timer noise.
        let reps = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(ours(&xs)?);
        }
        let ta = t0.elapsed().as_secs_f64() / reps as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(candle_nn::ops::layer_norm(&xs, &alpha, &beta, EPS as f32)?);
        }
        let tc = t1.elapsed().as_secs_f64() / reps as f64;

        println!(
            "{name:<22} {ea:>12.3e} {ec:>12.3e}   {:>8.2}ms {:>8.2}ms {:>6.2}x",
            ta * 1e3,
            tc * 1e3,
            ta / tc
        );
    }
    println!("\nerror columns are max |x - f64_reference|; speed is ours/candle (>1 means candle is faster)");

    let Ok(gpu) = Device::new_metal(0) else {
        return Ok(());
    };
    println!("\n-- timing on Metal (no f64 there, so no accuracy column) --");
    println!(
        "{:<22} {:>9} {:>9} {:>7}",
        "shape", "ours", "candle", "speed"
    );
    for (b, n, w, name) in shapes {
        let xs = Tensor::randn(0f32, 1.0, (b, n, w), &gpu)?;
        let alpha = Tensor::ones(w, DType::F32, &gpu)?;
        let beta = Tensor::zeros(w, DType::F32, &gpu)?;
        let reps = 50;
        // Warm up: the first call compiles kernels and allocates buffers.
        std::hint::black_box(ours(&xs)?);
        std::hint::black_box(candle_nn::ops::layer_norm(&xs, &alpha, &beta, EPS as f32)?);
        gpu.synchronize()?;

        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(ours(&xs)?);
        }
        gpu.synchronize()?;
        let ta = t0.elapsed().as_secs_f64() / reps as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(candle_nn::ops::layer_norm(&xs, &alpha, &beta, EPS as f32)?);
        }
        gpu.synchronize()?;
        let tc = t1.elapsed().as_secs_f64() / reps as f64;
        println!(
            "{name:<22} {:>8.3}ms {:>8.3}ms {:>6.2}x",
            ta * 1e3,
            tc * 1e3,
            ta / tc
        );
    }
    Ok(())
}
