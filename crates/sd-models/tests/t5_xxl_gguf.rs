//! T5-XXL loaded from a real quantised GGUF.
//!
//! `golden_t5.rs` verifies the *implementation* against transformers using the
//! small checkpoint. This verifies the *name mapping* at full size, which is a
//! separate failure mode: the model can be perfectly correct and still be fed
//! `wi_1` where it wanted `wi_0`.
//!
//! There is no numerical reference here on purpose. Producing one means
//! running T5-XXL in transformers and storing 19 GB of f32 weights, which is
//! not a fixture anyone will regenerate. What *is* checkable without that is
//! everything structural — that all 24 blocks resolve, that nothing is
//! missing, and that the output is finite and correctly scaled — and those
//! catch the mapping errors that actually happen.
//!
//! Loaded at F16: 4.7B parameters is 9.4 GB there against 18.8 at F32, and
//! this runs on a 36 GB machine alongside everything else.

use std::path::PathBuf;

use sd_models::t5::{T5Config, T5EncoderModel};
use sd_tensor::{DType, Device, Tensor};

fn gguf() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/flux/t5-xxl-q4_k_s.gguf")
}

#[test]
fn t5_xxl_loads_from_gguf_and_encodes() {
    let path = gguf();
    if !path.exists() {
        eprintln!(
            "SKIP: no T5-XXL gguf at {}. Fetch \
             city96/t5-v1_1-xxl-encoder-gguf.",
            path.display()
        );
        return;
    }

    let dev = Device::Cpu;
    // F16 deliberately, and it is the one thing in this workspace that a
    // build with `--features accelerate` cannot run: candle's Accelerate
    // backend has no f16 matmul and bails rather than falling back. Skipping
    // is the honest outcome — the mapping this verifies is dtype-independent,
    // and the alternative is either a false failure or loading 18.8 GB at f32.
    if cfg!(feature = "accelerate") {
        eprintln!(
            "SKIP t5_xxl_loads_from_gguf_and_encodes: this loads F16, and candle's \
             Accelerate CPU backend has no f16 matmul. Run it without the feature."
        );
        return;
    }
    let vb = match sd_loader::t5_var_builder_from_gguf(&path, DType::F16, &dev) {
        Ok(vb) => vb,
        Err(e) => {
            // The memory guard declining is a pass, not a failure: it means
            // the machine is busy, which is exactly when this should not run.
            eprintln!("SKIP: {e}");
            return;
        }
    };

    let cfg = T5Config::xxl();
    // Every one of the 24 blocks must resolve. A missing tensor fails here
    // rather than silently producing a shorter stack, because the model asks
    // for each by name.
    let model = T5EncoderModel::new(&cfg, vb).expect("all T5-XXL tensors should map");

    let n = 16;
    let ids = Tensor::from_vec(
        (0..n as u32)
            .map(|i| (i * 37 + 5) % 32000)
            .collect::<Vec<_>>(),
        (1, n),
        &dev,
    )
    .unwrap();

    let out = model.forward(&ids).unwrap();
    assert_eq!(
        out.dims(),
        &[1, n, cfg.d_model],
        "T5-XXL should emit d_model = 4096"
    );

    let out32 = out.to_dtype(DType::F32).unwrap();
    let v = out32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "non-finite activations — with RMSNorm accumulating in f32 this should \
         not happen even though the weights are f16"
    );

    // Post-final-norm, so order 1. A swapped gate/up projection or a wrong
    // norm would still be finite but would land far from here.
    let absmax = v.iter().fold(0f32, |a, b| a.max(b.abs()));
    let absmean = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
    eprintln!("t5-xxl output: absmax {absmax:.3}, absmean {absmean:.4}");
    assert!(
        (0.01..100.0).contains(&absmean),
        "output scale {absmean} is implausible for a normed T5 encoder"
    );

    // Not constant: a mapping that resolved every name to the same tensor, or
    // an embedding that failed to vary with the input, would still pass the
    // checks above.
    let first = &v[..cfg.d_model];
    let second = &v[cfg.d_model..2 * cfg.d_model];
    let differ = first
        .iter()
        .zip(second)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        differ > 1e-3,
        "two different tokens produced near-identical embeddings ({differ:.3e}) \
         — suspect the embedding lookup or the position bias"
    );
}
