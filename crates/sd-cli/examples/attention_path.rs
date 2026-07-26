//! Which attention path each model's shapes actually take, and how fast.
use sd_tensor::ops::{attention_with_path, AttentionPath};
use sd_tensor::{DType, Device, Tensor};

fn bench(label: &str, dev: &Device, b: usize, h: usize, n: usize, d: usize) {
    let mk = || Tensor::rand(-1f32, 1f32, (b, h, n, d), dev).unwrap();
    let (q, k, v) = (mk(), mk(), mk());
    // Warm up: the first Metal call compiles pipelines.
    let (_, path) = match attention_with_path(&q, &k, &v, None) {
        Ok(r) => r,
        Err(e) => {
            println!("  {label:34} ERROR {e}");
            return;
        }
    };
    let reps = 5;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        let (out, _) = attention_with_path(&q, &k, &v, None).unwrap();
        out.sum_all().unwrap().to_scalar::<f32>().unwrap(); // force completion
    }
    let per = t0.elapsed().as_secs_f64() / reps as f64;
    let tag = match path {
        AttentionPath::Fused => "FUSED",
        AttentionPath::Chunked => "chunked",
        AttentionPath::Naive => "naive",
    };
    println!("  {label:34} {tag:8} {:8.1} ms", per * 1000.0);
}

fn main() {
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("SD1.5 UNet 512 (h=8,d=40)", 8, 4096, 40),
        ("SDXL UNet 1024 (h=20,d=64)", 20, 4096, 64),
        ("Flux 512 (h=24,d=128)", 24, 1536, 128),
        ("Flux 1024 (h=24,d=128)", 24, 4608, 128),
        ("SD3.5 512 (h=24,d=64)", 24, 1178, 64),
        ("T5-XXL 154tok (h=64,d=64)", 64, 154, 64),
    ];
    for (name, dev) in [
        ("CPU", Device::Cpu),
        ("Metal", Device::new_metal(0).unwrap_or(Device::Cpu)),
    ] {
        if name == "Metal" && !dev.is_metal() {
            println!("\n(no Metal device)");
            continue;
        }
        println!("\n{name}:");
        for (label, h, n, d) in shapes {
            bench(label, &dev, 1, *h, *n, *d);
        }
    }
    let _ = DType::F32;
}
