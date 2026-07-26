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
    Ok(())
}
