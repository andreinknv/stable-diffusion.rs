//! SDXL's second text encoder on MLX — OpenCLIP ViT-bigG.
//!
//! Three things separate it from the tower `mlx_golden_clip` covers, and each
//! one loads, runs, and produces the right shape when wrong:
//!
//! - **Plain `gelu`, not `quick_gelu`.** The two differ by about 1e-2 — too
//!   small to look like a bug, too large to be right.
//! - **SDXL conditions on the penultimate layer, raw**, without
//!   `final_layer_norm`.
//! - **The pooled vector is the EOS hidden state, projected**, and EOS is found
//!   by argmax over the token ids rather than at a fixed index.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_sdxl_text_encoder -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::clip::{self, ClipConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

/// `golden_sdxl_text_encoder.rs`'s form — the *excess* over `DEFAULT_RTOL`,
/// not a raw absolute difference — but not its `DEFAULT_ATOL`, and the
/// difference is measured rather than assumed.
///
/// Measured on this fixture: the penultimate state's excess is 1.909e-4 and the
/// final state's 2.226e-4, both above `DEFAULT_ATOL`. `diagnose_the_residual`
/// says why that is accumulation and not a defect: **one element in 98,560**
/// violates, and its reference value is 0.038 — a cancellation, where 32 layers
/// of f32 leave an absolute floor around 2.4e-4 regardless of how small the
/// result is. Every element with `|ref| > 1` agrees to 3.077e-4 relative, well
/// inside `RTOL`.
///
/// So the bound is the measured floor with room, not `DEFAULT_ATOL` widened
/// until the test passed. candle reaches `DEFAULT_ATOL` here because it reduces
/// on CPU in a different order — the same class of difference `docs/handoff.md`
/// records for the UNet, where `DEFAULT_ATOL` "is below what the candle port
/// itself achieves on this fixture, so it would fail a correct implementation."
const ATOL: f32 = 5e-4;
const RTOL: f32 = 1e-3;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/sdxl_text_encoder_2")
}

/// Worst violation of `|a - b| <= atol + rtol * |b|`, which is the form
/// `testing::allclose_excess` computes on the candle side.
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
    eprintln!("{what:<16} peak {peak:>8.3}  max_abs {worst:.3e}  excess {exc:.3e}");
    exc
}

fn fixtures() -> Option<(
    std::collections::HashMap<String, Array>,
    std::collections::HashMap<String, Array>,
)> {
    let (refs, w) = (
        golden().join("reference.safetensors"),
        golden().join("text_encoder_2.safetensors"),
    );
    if !refs.exists() || !w.exists() {
        return None;
    }
    Some((
        load_safetensors(&refs).expect("reference"),
        load_safetensors(&w).expect("weights"),
    ))
}

/// Token ids arrive as floats in the fixture; CLIP indexes with integers.
fn token_ids(refs: &std::collections::HashMap<String, Array>, s: &Stream) -> Array {
    let ids = refs.get("token_ids").expect("token_ids");
    let f = ids.to_f32(s).unwrap().to_vec_f32(s).unwrap();
    let v: Vec<i32> = f.iter().map(|&x| x as i32).collect();
    Array::from_slice_i32(&v, &ids.shape()).unwrap()
}

#[test]
fn the_bigg_tower_matches_transformers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no sdxl_text_encoder_2 fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = ClipConfig::sdxl_2();
    let ids = token_ids(&refs, &s);

    let got = clip::text_encoder_with(&ids, &cfg, &w, &s).expect("forward");
    let worst = compare(
        &got,
        refs.get("last_hidden_state").expect("last_hidden_state"),
        &s,
        "last_hidden",
    );
    assert!(
        worst <= ATOL,
        "the bigG tower is {worst:.3e} from transformers"
    );
}

/// The penultimate layer, which is what SDXL actually conditions on.
#[test]
fn the_penultimate_layer_matches_transformers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no sdxl_text_encoder_2 fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = ClipConfig::sdxl_2();
    let ids = token_ids(&refs, &s);

    let got = clip::penultimate(&ids, &cfg, &w, &s).expect("penultimate");
    let worst = compare(
        &got,
        refs.get("penultimate").expect("penultimate"),
        &s,
        "penultimate",
    );
    assert!(worst <= ATOL, "the penultimate layer is {worst:.3e} out");

    // **And it is not the final layer.** Both are [1, 77, 1280] and the two
    // are trivially swappable, so the difference is asserted rather than
    // assumed.
    let last = refs.get("last_hidden_state").expect("last_hidden_state");
    let (a, b) = (got.to_vec_f32(&s).unwrap(), last.to_vec_f32(&s).unwrap());
    let spread = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        spread > 1.0,
        "the penultimate and final states differ by only {spread:.3e}; one is being \
         substituted for the other"
    );
}

/// The pooled vector: EOS hidden state, then `text_projection`.
#[test]
fn the_pooled_embedding_matches_transformers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no sdxl_text_encoder_2 fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = ClipConfig::sdxl_2();
    let ids = token_ids(&refs, &s);

    let (context, pooled) = clip::sdxl_conditioning(&ids, &cfg, &w, &s).expect("conditioning");
    assert_eq!(pooled.shape(), vec![1, 1280]);
    assert_eq!(context.shape(), vec![1, 77, 1280]);

    let worst = compare(&pooled, refs.get("pooled").expect("pooled"), &s, "pooled");
    assert!(worst <= ATOL, "the pooled embedding is {worst:.3e} out");

    // One forward, two outputs: the sequence must be the penultimate layer,
    // not a second encode.
    let alone = clip::penultimate(&ids, &cfg, &w, &s).expect("penultimate");
    assert_eq!(
        context.to_vec_f32(&s).unwrap(),
        alone.to_vec_f32(&s).unwrap(),
        "sdxl_conditioning's sequence must be exactly the penultimate layer"
    );
}

/// **EOS is the first highest id, not the last.**
///
/// Invisible for SDXL's own tokenizer, which pads with `!` (id 0) and so has
/// exactly one 49407 — either rule finds it. It is *not* invisible for a
/// CLIP-L sequence, which pads with EOS itself: a 10-token prompt has 68
/// copies and the two rules are 67 positions apart. `docs/handoff.md` records
/// that this cost 1.72 on the candle side, in every caller that pools.
#[test]
fn pooling_takes_the_first_eos_not_the_last() {
    let s = Stream::gpu();
    // Two rows of five, EOS-padded the way SD 1.5's tokenizer pads.
    let ids = Array::from_slice_i32(&[1, 2, 9, 9, 9, 4, 9, 9, 9, 9], &[2, 5]).unwrap();
    // Hidden state whose value *is* the position, so the pooled row says which
    // index was taken.
    let hidden: Vec<f32> = (0..10).map(|i| (i % 5) as f32).collect();
    let hidden = Array::from_slice_f32(&hidden, &[2, 5, 1]).unwrap();

    let pooled = clip::pool(&hidden, &ids, &s).unwrap();
    assert_eq!(pooled.shape(), vec![2, 1]);
    assert_eq!(
        pooled.to_vec_f32(&s).unwrap(),
        vec![2.0, 1.0],
        "pooling must take the first occurrence of the highest id; the last would \
         give [4, 4]"
    );
}

/// Diagnostic: where the residual error lives.
#[test]
#[ignore]
fn diagnose_the_residual() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();
    let cfg = ClipConfig::sdxl_2();
    let ids = token_ids(&refs, &s);
    let got = clip::penultimate(&ids, &cfg, &w, &s).unwrap();
    let g = got.to_vec_f32(&s).unwrap();
    let b = refs.get("penultimate").unwrap().to_vec_f32(&s).unwrap();
    let mut over = 0usize;
    let mut worst_rel = 0.0f32;
    let mut mags = vec![];
    for (x, y) in g.iter().zip(&b) {
        let d = (x - y).abs();
        if d - 1e-3 * y.abs() > 1e-4 {
            over += 1;
            mags.push(y.abs());
        }
        if y.abs() > 1.0 {
            worst_rel = worst_rel.max(d / y.abs());
        }
    }
    mags.sort_by(|a, c| a.partial_cmp(c).unwrap());
    eprintln!(
        "elements {} over {} ({:.4}%)",
        g.len(),
        over,
        100.0 * over as f32 / g.len() as f32
    );
    if !mags.is_empty() {
        eprintln!(
            "their |ref| ranges {:.3} .. {:.3}, median {:.3}",
            mags[0],
            mags[mags.len() - 1],
            mags[mags.len() / 2]
        );
    }
    eprintln!("worst relative error where |ref| > 1: {worst_rel:.3e}");
}
