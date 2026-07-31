//! A Qwen3-family decoder as a text encoder, against `transformers`.
//!
//! The shared foundation for Qwen-Image, Z-Image and FLUX.2 — all three
//! condition on an LLM rather than on T5 or CLIP, and all three are the same
//! transformer at different widths. Verified at 0.6B so that loading a 4B or
//! 7B checkpoint later separates "is the forward right" from "is the name
//! mapping right".
//!
//! ```bash
//! .venv/bin/python xtask/golden/dump_reference.py llm --output tests/golden
//! cargo test -p sd-models --features mlx --test mlx_golden_llm -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::llm::{self, LlmConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

/// Qwen3's hidden states reach into the hundreds — the RMSNorm scales are
/// large and 28 residual additions accumulate — so this is a relative bound,
/// as `golden_clip_encoder` is for the same reason.
const RTOL: f32 = 2e-3;
const ATOL: f32 = 2e-3;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/llm")
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let (refs, w) = (
        golden().join("reference.safetensors"),
        golden().join("llm.safetensors"),
    );
    if !refs.exists() || !w.exists() {
        return None;
    }
    Some((
        load_safetensors(&refs).expect("reference"),
        load_safetensors(&w).expect("weights"),
    ))
}

/// Worst excess over `atol + rtol*|want|`, and the peak magnitude alongside it
/// so a tolerance can be judged rather than guessed at.
fn compare(got: &Array, want: &Array, s: &Stream, what: &str) -> f32 {
    let g = got.to_vec_f32(s).expect("got");
    let w = want.to_vec_f32(s).expect("want");
    assert_eq!(g.len(), w.len(), "{what}: element count");
    let (mut worst, mut peak, mut exc) = (0.0f32, 0.0f32, 0.0f32);
    for (a, b) in g.iter().zip(&w) {
        let d = (a - b).abs();
        worst = worst.max(d);
        peak = peak.max(b.abs());
        exc = exc.max(d - RTOL * b.abs());
    }
    let exc = exc.max(0.0);
    eprintln!("{what:<14} peak {peak:>9.2}  max_abs {worst:.3e}  excess {exc:.3e}");
    exc
}

fn token_ids(refs: &HashMap<String, Array>, s: &Stream) -> Array {
    // The fixture writes int64; MLX takes i32 indices.
    let raw = refs.get("token_ids").expect("token_ids");
    let v: Vec<i32> = raw
        .to_f32(s)
        .expect("to f32")
        .to_vec_f32(s)
        .expect("ids")
        .iter()
        .map(|&x| x as i32)
        .collect();
    Array::from_slice_i32(&v, &raw.shape()).expect("ids")
}

/// **Every layer, not just the output.**
///
/// A 28-layer stack that disagrees at the end tells you nothing about where.
/// The reference captures each hidden state for exactly this reason, and the
/// first layer to exceed tolerance is the one to look at.
#[test]
fn every_layer_matches_transformers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no llm fixture. See the module docs.");
        return;
    };
    let s = Stream::gpu();
    let cfg = LlmConfig::qwen3_0_6b();
    let ids = token_ids(&refs, &s);

    let states = llm::hidden_states(&ids, &cfg, &w, &s).expect("forward");
    assert_eq!(
        states.len(),
        cfg.layers + 1,
        "one state per layer, plus the embedding"
    );

    for (i, got) in states.iter().enumerate() {
        let want = refs
            .get(&format!("hidden_{i}"))
            .unwrap_or_else(|| panic!("reference has no hidden_{i}"));
        let excess = compare(got, want, &s, &format!("hidden_{i}"));
        assert!(
            excess <= ATOL,
            "layer {i} is {excess:.3e} outside tolerance — the first layer to \
             fail is the one that is wrong, not the ones after it"
        );
    }
}

/// The output a diffusion model would actually condition on.
#[test]
fn the_encoded_state_matches_transformers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no llm fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = LlmConfig::qwen3_0_6b();
    let ids = token_ids(&refs, &s);

    let got = llm::encode(&ids, &cfg, &w, &s).expect("encode");
    assert_eq!(got.shape(), vec![1, 16, cfg.hidden]);
    let excess = compare(
        &got,
        refs.get("last_hidden_state").expect("last_hidden_state"),
        &s,
        "encoded",
    );
    assert!(excess <= ATOL, "the encoder is {excess:.3e} out");
}

/// **The attention must be causal.**
///
/// A decoder run bidirectionally produces an embedding of exactly the right
/// shape from a model that never saw one — no error anywhere. The check:
/// changing the *last* token cannot alter the first token's state, because
/// nothing may attend forwards.
#[test]
fn no_token_sees_the_ones_after_it() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no llm fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = LlmConfig::qwen3_0_6b();
    let ids = token_ids(&refs, &s);
    let base = llm::encode(&ids, &cfg, &w, &s).expect("encode");

    // Swap the final token for a different one.
    let mut v: Vec<i32> = ids
        .to_f32(&s)
        .unwrap()
        .to_vec_f32(&s)
        .unwrap()
        .iter()
        .map(|&x| x as i32)
        .collect();
    let last = v.len() - 1;
    v[last] = (v[last] + 1234) % 10000;
    let poked = Array::from_slice_i32(&v, &ids.shape()).unwrap();
    let after = llm::encode(&poked, &cfg, &w, &s).expect("encode");

    let (a, b) = (base.to_vec_f32(&s).unwrap(), after.to_vec_f32(&s).unwrap());
    // The first token's state occupies the first `hidden` values.
    let first_drift = a[..cfg.hidden]
        .iter()
        .zip(&b[..cfg.hidden])
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    // ...and the last token's must move, or the input was ignored entirely.
    let start = (v.len() - 1) * cfg.hidden;
    let last_drift = a[start..]
        .iter()
        .zip(&b[start..])
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("changing the last token moves: first {first_drift:.3e}, last {last_drift:.3e}");

    assert!(
        first_drift < 1e-4,
        "the first token moved by {first_drift:.3e} when a later token changed; \
         the attention is bidirectional and this is not the model it loaded"
    );
    assert!(
        last_drift > 1e-3,
        "the last token barely moved ({last_drift:.3e}); the ids are being ignored"
    );
}

/// **Grouped-query attention pairs each query with its own group's key.**
///
/// Qwen3-0.6B has 16 query heads over 8 kv heads, so each kv head serves two
/// queries. Expanding by repeating the whole tensor rather than broadcasting a
/// group axis interleaves them — every query then attends to the wrong key,
/// and the output is the right shape. Covered by the layer comparison above;
/// pinned separately because the two expansions differ only in axis order.
#[test]
fn the_config_declares_its_grouping() {
    let cfg = LlmConfig::qwen3_0_6b();
    assert_eq!(cfg.group_size(), 2, "16 query heads over 8 kv heads");
    // The head dimension is *not* hidden/heads here, which is the trap.
    assert_ne!(
        cfg.head_dim,
        cfg.hidden / cfg.heads,
        "Qwen3-0.6B's head_dim is 128 while hidden/heads is 64; deriving it by \
         division gives a projection that does not match its weight"
    );
    assert_eq!(
        cfg.heads * cfg.head_dim,
        2048,
        "the q projection is wider than the model"
    );

    // The other configurations this module serves.
    assert_eq!(
        LlmConfig::qwen2_5_vl_7b().hidden,
        3584,
        "Qwen-Image's joint_attention_dim"
    );
    assert_eq!(LlmConfig::qwen3_4b().hidden, 2560, "Z-Image's cap_feat_dim");
    assert!(LlmConfig::qwen2_5_vl_7b().qkv_bias, "Qwen2.5 has qkv bias");
    assert!(
        !LlmConfig::mistral_small_3_2().qk_norm,
        "Mistral has no qk-norm"
    );
}
