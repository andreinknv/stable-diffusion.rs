//! CLIP's text tower on MLX, against `tests/golden/clip_encoder`.
//!
//! The same fixture `golden_clip_encoder.rs` uses, dumped from
//! `transformers`. Compared layer by layer as well as at the end, so a failure
//! names the layer.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_clip -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::clip;
use sd_tensor::mlx::{load_safetensors, Array, Stream};

/// The combined criterion `golden_clip_encoder.rs` uses, and for its reason:
/// **CLIP carries massive activations.** The reference peaks at 851.03 at
/// `[0, 0, 681]` and holds it through all twelve layers, where an f32 ULP is
/// 6.1e-5 — so `DEFAULT_ATOL` alone would fail for any implementation, correct
/// or not. `|a - b| <= atol + rtol * |b|` is what numpy and torch use, and on
/// order-1 values it is still ~atol, so nothing is relaxed where accuracy is
/// actually testable.
const ATOL: f32 = 1e-4; // sd_tensor::testing::DEFAULT_ATOL
const RTOL: f32 = 1e-3; // sd_tensor::testing::DEFAULT_RTOL

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/clip_encoder")
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let refs = golden_dir().join("reference.safetensors");
    let clip = golden_dir().join("clip.safetensors");
    if !refs.exists() || !clip.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no clip_encoder fixture.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py clip_encoder --output tests/golden\n"
        );
        return None;
    }
    Some((
        load_safetensors(&refs).expect("loading reference"),
        load_safetensors(&clip).expect("loading CLIP weights"),
    ))
}

/// Worst violation of `|a - b| <= atol + rtol * |b|`, in the same form
/// `golden_clip_encoder.rs::allclose_excess` computes. Compare against `ATOL`.
fn compare(got: &Array, want: &Array, s: &Stream, what: &str) -> f32 {
    let g = got.to_vec_f32(s).expect("mlx result");
    let wv = want.to_vec_f32(s).expect("reference");
    assert_eq!(g.len(), wv.len(), "{what}: element count");
    let (mut worst, mut peak, mut excess) = (0.0f32, 0.0f32, 0.0f32);
    for (a, b) in g.iter().zip(&wv) {
        let d = (a - b).abs();
        worst = worst.max(d);
        peak = peak.max(b.abs());
        excess = excess.max(d - RTOL * b.abs());
    }
    let excess = excess.max(0.0);
    eprintln!(
        "{what:<20} peak {peak:>8.3}  max_abs {worst:.3e}  excess {excess:.3e}   atol {ATOL:.0e}"
    );
    excess
}

/// The fixture's token ids are int64; MLX indexes with int32, so they are
/// narrowed here. SD 1.5's vocabulary is 49408, far inside i32.
fn token_ids(refs: &HashMap<String, Array>, s: &Stream) -> Array {
    let ids = refs.get("token_ids").expect("token_ids");
    let as_f32 = ids.to_f32(s).expect("ids to f32").to_vec_f32(s).unwrap();
    let as_i32: Vec<i32> = as_f32.iter().map(|&v| v as i32).collect();
    Array::from_slice_i32(&as_i32, &ids.shape()).expect("ids")
}

#[test]
fn the_text_tower_matches_transformers() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();
    let ids = token_ids(&refs, &s);

    let (final_state, layers) = clip::text_encoder_layers(&ids, &w, &s).unwrap();

    // Embeddings first: a wrong position lookup shows here and nowhere earlier.
    let emb = clip::embeddings(&ids, &w, &s).unwrap();
    assert!(
        compare(&emb, refs.get("embeddings").unwrap(), &s, "embeddings") <= ATOL,
        "embeddings"
    );

    let mut first_bad = None;
    for (i, got) in layers.iter().enumerate() {
        let name = format!("layer_{i:02}");
        let worst = compare(got, refs.get(&name).unwrap(), &s, &name);
        if worst > ATOL && first_bad.is_none() {
            first_bad = Some((i, worst));
        }
    }
    if let Some((i, worst)) = first_bad {
        panic!("first bad layer is {i} at {worst:.3e}, beyond rtol={RTOL:.0e} + atol={ATOL:.0e}");
    }

    let worst = compare(
        &final_state,
        refs.get("last_hidden_state").unwrap(),
        &s,
        "last_hidden_state",
    );
    assert!(
        worst <= ATOL,
        "the text tower is {worst:.3e} from transformers, past atol {ATOL:.0e}"
    );
}
