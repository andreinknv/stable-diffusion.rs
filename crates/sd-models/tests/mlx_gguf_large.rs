//! Full-size quantised GGUF checkpoints, read on the MLX path.
//!
//! # What this checks, and what it deliberately does not
//!
//! `Gguf::open` reads the header and the tensor directory and touches no
//! tensor data, so the *geometry* of a 12B checkpoint is free to verify: all
//! 19 double and 38 single blocks resolve by name, at the shapes the config
//! predicts. That is worth pinning on its own — a checkpoint whose block count
//! disagrees with the config fails deep inside a forward pass naming one
//! arbitrary tensor.
//!
//! **It does not run them.** `sd_models::mlx::gguf` dequantises to f32, and
//! Flux schnell is 12B parameters — 48 GB dense, which does not fit on this
//! machine at all. The candle path keeps those weights *quantised at rest*
//! (`FluxTransformer::from_quantized`, `resident_bytes`) and dequantises per
//! operation; MLX has its own quantisation scheme and no equivalent has been
//! built here yet.
//!
//! So this is the honest boundary of the port, written as a test rather than
//! as a promise: **quantised-at-rest inference on MLX does not exist, and full
//! -size Flux and T5-XXL cannot leave candle until it does.** `docs/handoff.md`
//! records what it would cost — requantising into MLX's own 4-bit scheme, at a
//! cosine of about 0.995 against the GGUF values.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_gguf_large -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashSet;
use std::path::PathBuf;

use sd_tensor::mlx_gguf::Gguf;

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(name)
}

/// Count the distinct block indices under a prefix, e.g. `double_blocks.`.
fn blocks(names: &[String], prefix: &str) -> usize {
    let mut seen = HashSet::new();
    for n in names {
        let Some(rest) = n.strip_prefix(prefix) else {
            continue;
        };
        if let Some((idx, _)) = rest.split_once('.') {
            if let Ok(i) = idx.parse::<usize>() {
                seen.insert(i);
            }
        }
    }
    seen.len()
}

fn tensor_names(path: &std::path::Path) -> Vec<String> {
    let g = Gguf::open(path).expect("opening the gguf");
    g.tensors.iter().map(|t| t.name.clone()).collect()
}

/// Schnell's geometry, read from the file rather than assumed from the config.
#[test]
fn schnell_geometry_is_read_from_the_file() {
    let path = golden("flux/flux-schnell-q4_k_s.gguf");
    if !path.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no Flux schnell gguf; fetch city96/FLUX.1-schnell-gguf"
        );
        return;
    }
    let names = tensor_names(&path);
    let (double, single) = (
        blocks(&names, "double_blocks."),
        blocks(&names, "single_blocks."),
    );
    eprintln!(
        "schnell: {double} double, {single} single, {} tensors",
        names.len()
    );
    assert_eq!(
        (double, single),
        (19, 38),
        "schnell is 19 double and 38 single blocks"
    );

    // The names are the ones the MLX Flux asks for — the GGUF spelling, not
    // the diffusers renaming. This is what makes the two meet without a
    // translation table.
    for required in [
        "double_blocks.0.img_attn.qkv.weight",
        "double_blocks.0.txt_attn.qkv.weight",
        "single_blocks.0.linear1.weight",
    ] {
        assert!(
            names.iter().any(|n| n == required),
            "{required} is missing; the MLX Flux reads GGUF names directly"
        );
    }
}

/// T5-XXL's encoder geometry.
#[test]
fn t5_xxl_geometry_is_read_from_the_file() {
    let path = golden("flux/t5-xxl-q4_k_s.gguf");
    if !path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no T5-XXL gguf");
        return;
    }
    let names = tensor_names(&path);
    let layers = blocks(&names, "enc.blk.");
    eprintln!("t5-xxl: {layers} encoder blocks, {} tensors", names.len());
    assert_eq!(layers, 24, "T5 v1.1 XXL's encoder is 24 blocks");
}

/// **The dense cost, stated rather than discovered at the end of a run.**
///
/// This is the whole reason quantised-at-rest matters, and it is arithmetic on
/// the tensor directory — no data is read.
#[test]
fn dequantising_schnell_would_not_fit_and_the_test_says_so() {
    let path = golden("flux/flux-schnell-q4_k_s.gguf");
    if !path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no Flux schnell gguf");
        return;
    }
    let g = Gguf::open(&path).expect("open");
    let elements: usize = g.tensors.iter().map(|t| t.elem_count()).sum();
    let dense_gb = (elements * 4) as f64 / 1e9;
    eprintln!(
        "schnell: {:.2}B parameters, {dense_gb:.1} GB dequantised to f32",
        elements as f64 / 1e9
    );

    // 36 GB of unified memory, shared with everything else. This is a fact
    // about the checkpoint, asserted so the boundary is visible in the suite
    // rather than only in prose.
    assert!(
        dense_gb > 36.0,
        "schnell dequantises to {dense_gb:.1} GB, which would fit — if that is true \
         then `sd_models::mlx::gguf::load` can run it and this limitation is stale"
    );
}
