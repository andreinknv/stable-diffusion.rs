//! The candle-free GGUF reader against candle's, tensor for tensor.
//!
//! candle's GGUF path is already gated by `golden_gguf.rs`, so agreeing with it
//! is agreeing with the reference — and it is the only way to check a
//! dequantiser, because a wrong block layout produces plausible numbers of the
//! right shape.
//!
//! Every quantisation type this project's checkpoints use is covered by
//! choosing files that carry them: Q4_0 and Q8_0 from SD 1.5's GGUFs, Q4_K and
//! Q5_K from SD 3.5's, Q6_K from T5's.
//!
//! ```bash
//! cargo test -p sd-tensor --features mlx --test mlx_gguf_agrees_with_candle -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::{Path, PathBuf};

use sd_tensor::{mlx_gguf, Device};

fn cache() -> PathBuf {
    PathBuf::from("/Volumes/AI MODELS/huggingface/hub")
}

fn find(pattern: &str) -> Option<PathBuf> {
    let root = cache();
    if !root.exists() {
        return None;
    }
    fn walk(dir: &Path, pattern: &str, out: &mut Option<PathBuf>, depth: usize) {
        if out.is_some() || depth > 6 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, pattern, out, depth + 1);
            } else if p
                .file_name()
                .is_some_and(|n| n.to_string_lossy() == pattern)
            {
                *out = Some(p);
                return;
            }
        }
    }
    let mut found = None;
    walk(&root, pattern, &mut found, 0);
    found
}

/// Compare every tensor of one file, or skip if it is not on this machine.
fn compare(file: &Path, label: &str) {
    let dev = Device::Cpu;

    let mut ours = mlx_gguf::Gguf::open(file).expect("our reader");
    let infos = ours.tensors.clone();

    let mut f = std::fs::File::open(file).expect("open");
    let content = sd_tensor::gguf::Content::read(&mut f).expect("candle reader");

    assert_eq!(
        infos.len(),
        content.tensor_infos.len(),
        "{label}: tensor counts differ"
    );

    let mut checked = 0usize;
    let mut worst = 0.0f32;
    let mut worst_name = String::new();
    for info in &infos {
        let q = content
            .tensor(&mut f, &info.name, &dev)
            .unwrap_or_else(|e| panic!("{label}: candle could not read {}: {e}", info.name));
        let want = q
            .dequantize(&dev)
            .expect("dequantize")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got = ours.dequantize(info).expect("our dequantize");

        assert_eq!(
            got.len(),
            want.len(),
            "{label}: {} element count ({:?} against candle)",
            info.name,
            info.shape
        );
        for (a, b) in got.iter().zip(&want) {
            let d = (a - b).abs();
            if d > worst {
                worst = d;
                worst_name = info.name.clone();
            }
        }
        checked += 1;
    }
    eprintln!("{label:<26} {checked} tensors, worst {worst:.3e} at {worst_name}");
    // Both sides dequantise the same integers with the same scales, so this is
    // exact rather than close. A tolerance here would hide a wrong block
    // layout that happens to land nearby.
    assert_eq!(
        worst, 0.0,
        "{label}: dequantisation differs from candle at {worst_name}"
    );
}

#[test]
fn q4_0_and_q8_0_match_candle() {
    for (name, label) in [
        (
            "stable-diffusion-v1-5-pruned-emaonly-Q4_0.gguf",
            "sd15 Q4_0",
        ),
        (
            "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
            "sd15 Q8_0",
        ),
    ] {
        let Some(p) = find(name) else {
            sd_tensor::skip_missing_fixture!("SKIP: {name} is not on this machine");
            continue;
        };
        compare(&p, label);
    }
}

#[test]
fn q4_k_and_q5_k_match_candle() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/sd35/sd35-medium-q4_k_m.gguf");
    if !path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no sd35 gguf");
        return;
    }
    // This is the file MLX's own loader refuses: it carries 48 Q5_K tensors and
    // fails with `gguf_tensor_to_f16 failed`, which is why this module exists.
    compare(&path, "sd35 Q4_K + Q5_K");
}

#[test]
fn q6_k_matches_candle() {
    let Some(p) = find("t5-v1_1-xxl-encoder-Q4_K_S.gguf") else {
        sd_tensor::skip_missing_fixture!("SKIP: no t5 gguf");
        return;
    };
    compare(&p, "t5 Q4_K + Q5_K + Q6_K");
}
