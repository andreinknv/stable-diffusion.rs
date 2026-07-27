//! Does each op agree between CPU and Metal? Fast is worthless if wrong.
use sd_tensor::gguf::{GgmlDType, QTensor};
use sd_tensor::quantized::QLinear;
use sd_tensor::{DType, Device, Tensor};
use std::sync::Arc;

fn cmp(label: &str, a: &Tensor, b: &Tensor) {
    let a = a
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let b = b
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let c = sd_tensor::testing::closeness(&a, &b).unwrap();
    let scale = b
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let verdict = if c.max_abs < 1e-2 * scale.max(1.0) as f64 {
        "ok"
    } else {
        "MISMATCH"
    };
    println!(
        "  {label:28} max_abs {:.3e}  mean {:.3e}  (scale {:.2})  {verdict}",
        c.max_abs, c.mean_abs, scale
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let m = Device::new_metal(0)?;
    let c = Device::Cpu;

    // 1. Fused attention, Flux-shaped.
    let mk = |d: &Device| Tensor::rand(-1f32, 1f32, (1, 24, 512, 128), d).unwrap();
    let (q, k, v) = (mk(&c), mk(&c), mk(&c));
    let (qm, km, vm) = (q.to_device(&m)?, k.to_device(&m)?, v.to_device(&m)?);
    let (rc, pc) = sd_tensor::ops::attention_with_path(&q, &k, &v, None)?;
    let (rm, pm) = sd_tensor::ops::attention_with_path(&qm, &km, &vm, None)?;
    println!("attention (cpu {pc:?} vs metal {pm:?}):");
    cmp("attention", &rm, &rc);

    // 2. Quantised matmul, which is how every large model's weights are held.
    for dt in [
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q8_0,
        GgmlDType::F16,
    ] {
        let w = Tensor::rand(-1f32, 1f32, (1536, 1536), &c)?;
        let xs = Tensor::rand(-1f32, 1f32, (1, 64, 1536), &c)?;
        let lc = QLinear::new(Arc::new(QTensor::quantize(&w, dt)?), None)?;
        let wm = QTensor::quantize(&w.to_device(&m)?, dt)?;
        let lm = QLinear::new(Arc::new(wm), None)?;
        cmp(
            &format!("QLinear {dt:?}"),
            &lm.forward(&xs.to_device(&m)?)?,
            &lc.forward(&xs)?,
        );
    }

    // 3. Plain matmul and conv, as controls.
    let a = Tensor::rand(-1f32, 1f32, (1, 512, 1536), &c)?;
    let b = Tensor::rand(-1f32, 1f32, (1, 1536, 512), &c)?;
    cmp(
        "matmul",
        &a.to_device(&m)?.matmul(&b.to_device(&m)?)?,
        &a.matmul(&b)?,
    );

    // 4. A quantised matmul on a tensor that does not own its buffer.
    //
    // This is the one that mattered, and everything above passed while it was
    // broken. candle 0.11's Metal quantised matmul ignored the activation's
    // `start_offset`, so a view into the middle of a larger tensor was read
    // from the start of it: every double-stream block projected the text half
    // of the attention output in place of the image half, and Flux rendered a
    // flat orange field. `sd_tensor::quantized::without_storage_offset` is the
    // workaround.
    //
    // Note the comparison is against the *same layer on the same device* with
    // the input copied, not against the CPU. A device-vs-device check would
    // have caught it too, but this form says exactly what is wrong: a matmul
    // that depends on where its input happens to live.
    println!("offset-sensitivity (a narrowed activation vs its own copy):");
    let w = Tensor::rand(-0.1f32, 0.1f32, (1536, 1536), &c)?;
    for dt in [GgmlDType::Q4K, GgmlDType::Q8_0] {
        let ql = QLinear::new(Arc::new(QTensor::quantize(&w.to_device(&m)?, dt)?), None)?;
        // [1, 96, 1536] with the last 64 rows taken: offset 32*1536, and
        // `contiguous()` will not move it because candle already calls that
        // layout contiguous.
        let big = Tensor::rand(-1f32, 1f32, (1, 96, 1536), &m)?;
        let view = big.narrow(1, 32, 64)?.contiguous()?;
        let copied = view.force_contiguous()?;
        cmp(
            &format!("QLinear {dt:?} view vs copy"),
            &ql.forward(&view)?,
            &ql.forward(&copied)?,
        );
        // And against the CPU, which honours the offset, as the outer check.
        let ql_c = QLinear::new(Arc::new(QTensor::quantize(&w, dt)?), None)?;
        cmp(
            &format!("QLinear {dt:?} narrowed, metal vs cpu"),
            &ql.forward(&view)?,
            &ql_c.forward(&view.to_device(&c)?)?,
        );
    }
    Ok(())
}
