//! Which attention path each model's shapes actually take, and how fast.
//!
//! On CPU it also times the path that *wasn't* chosen, because "the dispatcher
//! picked flash" is only good news if flash is actually faster at that shape.
//! Two columns make a bad choice visible; one column hides it.
use sd_tensor::ops::{
    attention_with_path, chunked_attention, flash_attention_cpu, flash_cpu_supported, AttentionPath,
};
use sd_tensor::{DType, Device, Tensor};

/// Fastest of `reps` runs, in milliseconds, after one untimed warm-up.
///
/// **Minimum, not mean or median.** Noise on this machine is one-sided — a
/// scheduler preemption, a page fault or a thermal excursion only ever makes a
/// run slower, never faster — so the minimum is the least contaminated
/// estimate of how fast the code can go. A mean over 5 reps here moved by 10x
/// between two back-to-back runs of this very example, which is more than any
/// effect it is trying to measure.
fn best_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
    }
    best
}

/// Time two paths against each other, alternating between them.
///
/// Interleaving matters: measuring all of A then all of B lets any drift over
/// the run — a background build finishing, the fans spinning up — land
/// entirely on one of them and read as a difference between the paths.
fn compare_ms(reps: usize, mut a: impl FnMut(), mut b: impl FnMut()) -> (f64, f64) {
    a();
    b();
    let (mut best_a, mut best_b) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        a();
        best_a = best_a.min(t0.elapsed().as_secs_f64() * 1000.0);
        let t1 = std::time::Instant::now();
        b();
        best_b = best_b.min(t1.elapsed().as_secs_f64() * 1000.0);
    }
    (best_a, best_b)
}

fn bench(label: &str, dev: &Device, b: usize, h: usize, n_q: usize, n_k: usize, d: usize) {
    let mk = |n| Tensor::rand(-1f32, 1f32, (b, h, n, d), dev).unwrap();
    let (q, k, v) = (mk(n_q), mk(n_k), mk(n_k));
    let (_, path) = match attention_with_path(&q, &k, &v, None) {
        Ok(r) => r,
        Err(e) => {
            println!("  {label:34} ERROR {e}");
            return;
        }
    };
    let reps = 5;
    let per = best_ms(reps, || {
        let (out, _) = attention_with_path(&q, &k, &v, None).unwrap();
        out.sum_all().unwrap().to_scalar::<f32>().unwrap(); // force completion
    });
    let tag = match path {
        AttentionPath::Fused => "FUSED",
        AttentionPath::FlashCpu => "flash",
        AttentionPath::Chunked => "chunked",
        AttentionPath::Naive => "naive",
    };
    let other = if flash_cpu_supported(&q, &k, &v, None) {
        let (chunked, flash) = compare_ms(
            reps,
            || {
                chunked_attention(&q, &k, &v, None).unwrap();
            },
            || {
                flash_attention_cpu(&q, &k, &v, None).unwrap();
            },
        );
        let ratio = chunked / flash;
        // The dispatcher is supposed to take flash exactly when it is faster.
        // Say so out loud, rather than leaving it to be read off two columns.
        //
        // The 5% deadband is not politeness: repeat runs of this benchmark
        // move by about that much, so flagging a 1.02x difference would print
        // a complaint that flips sign between runs and means nothing.
        let verdict = if ratio > 1.05 && path != AttentionPath::FlashCpu {
            "  <-- flash is faster here"
        } else if ratio < 0.95 && path == AttentionPath::FlashCpu {
            "  <-- chunked is faster here"
        } else {
            ""
        };
        format!("   chunked {chunked:8.1}  flash {flash:8.1}  {ratio:5.2}x{verdict}")
    } else {
        String::new()
    };
    println!("  {label:34} {tag:8} {per:8.1} ms{other}");
}

fn main() {
    // (label, heads, seq_q, seq_k, head_dim). Cross-attention shapes are here
    // because they are half of every UNet block and they are where the two
    // paths differ most: a 77-token key axis makes the score matmul too small
    // to amortise, which is exactly where a streaming kernel wins.
    //
    // **These are all unmasked, and one of them is therefore a lie.** T5's
    // real call carries a `[batch, heads, n, n]` relative-position bias, which
    // `flash_cpu_supported` refuses because the kernel indexes a mask flat.
    // The 5-8x below is a speedup nothing here can collect. It is kept because
    // it is the clearest illustration of *where* the streaming kernel wins,
    // but do not read a row of this table as a claim about a model without
    // checking whether that model's call is masked.
    let shapes: &[(&str, usize, usize, usize, usize)] = &[
        ("SD1.5 UNet 512 self (h=8,d=40)", 8, 4096, 4096, 40),
        ("SD1.5 UNet 512 cross (h=8,d=40)", 8, 4096, 77, 40),
        ("SDXL UNet 1024 self (h=20,d=64)", 20, 4096, 4096, 64),
        ("SDXL UNet 1024 cross (h=20,d=64)", 20, 4096, 77, 64),
        ("Flux 512 (h=24,d=128)", 24, 1536, 1536, 128),
        ("Flux 1024 (h=24,d=128)", 24, 4608, 4608, 128),
        ("SD3.5 512 (h=24,d=64)", 24, 1178, 1178, 64),
        ("T5-XXL 154tok (h=64,d=64) UNMASKED", 64, 154, 154, 64),
        ("CLIP-L 77tok (h=12,d=64)", 12, 77, 77, 64),
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
        for (label, h, n_q, n_k, d) in shapes {
            bench(label, &dev, 1, *h, *n_q, *n_k, *d);
        }
    }
    let _ = DType::F32;
}
