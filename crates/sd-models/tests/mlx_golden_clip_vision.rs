//! CLIP's vision tower on MLX, against `tests/golden/clip_vision`.
//!
//! The tower IP-Adapter and unCLIP condition on. The thing this catches that
//! inspection cannot is the mask: an image has no order to respect, so
//! attention here is *not* causal — and reusing the text tower's causal call
//! produces an embedding of exactly the right shape from a model where each
//! patch saw only those before it in raster order.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_clip_vision -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::clip_vision::{self, VisionConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

/// The same form and reasoning as `mlx_golden_sdxl_text_encoder`: this is the
/// same 32-layer OpenCLIP stack at the same width, so the same f32 floor
/// applies. Measured below rather than assumed.
const ATOL: f32 = 5e-4;
const RTOL: f32 = 1e-3;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/clip_vision")
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let (refs, w) = (
        golden().join("reference.safetensors"),
        golden().join("image_encoder.safetensors"),
    );
    if !refs.exists() || !w.exists() {
        return None;
    }
    Some((
        load_safetensors(&refs).expect("reference"),
        load_safetensors(&w).expect("weights"),
    ))
}

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
    eprintln!("{what:<14} peak {peak:>8.3}  max_abs {worst:.3e}  excess {exc:.3e}");
    exc
}

/// The fixture's pixels are NCHW and already CLIP-normalised.
fn pixels(refs: &HashMap<String, Array>, s: &Stream) -> Array {
    refs.get("pixels")
        .expect("pixels")
        .transpose(&[0, 2, 3, 1], s)
        .expect("NCHW -> NHWC")
}

#[test]
fn the_vision_tower_matches_transformers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no clip_vision fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = VisionConfig::vit_h_14();
    let px = pixels(&refs, &s);

    let got = clip_vision::hidden_states(&px, &cfg, &w, &s).expect("forward");
    // 16x16 patches plus the class token.
    assert_eq!(got.shape(), vec![1, 257, 1280]);
    assert_eq!(cfg.sequence_length(), 257);

    let worst = compare(&got, refs.get("hidden").expect("hidden"), &s, "hidden");
    assert!(
        worst <= ATOL,
        "the vision tower is {worst:.3e} from transformers"
    );
}

/// The pooled embedding: token 0, post-normed.
#[test]
fn the_pooled_embedding_matches_transformers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no clip_vision fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = VisionConfig::vit_h_14();
    let px = pixels(&refs, &s);

    let got = clip_vision::pooled(&px, &cfg, &w, &s).expect("pooled");
    assert_eq!(got.shape(), vec![1, 1280]);
    let worst = compare(&got, refs.get("pooled").expect("pooled"), &s, "pooled");
    assert!(worst <= ATOL, "the pooled embedding is {worst:.3e} out");
}

/// **The attention must not be causal.**
///
/// With a causal mask, token 0 — the class token, which is what the pooled
/// output reads — attends to nothing but itself, so perturbing any patch would
/// leave it untouched. Without one, every patch reaches it. This distinguishes
/// the two calls, which are one identifier apart and produce identical shapes.
#[test]
fn every_patch_reaches_the_class_token() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no clip_vision fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = VisionConfig::vit_h_14();
    let px = pixels(&refs, &s);

    let base = clip_vision::pooled(&px, &cfg, &w, &s).expect("pooled");

    // Perturb the bottom-right corner, which is the *last* patch in raster
    // order and so the one a causal mask hides from everything before it.
    let mut data = px.to_vec_f32(&s).unwrap();
    let sz = cfg.image_size;
    for y in sz - 14..sz {
        for x in sz - 14..sz {
            for c in 0..3 {
                data[(y * sz + x) * 3 + c] += 2.0;
            }
        }
    }
    let poked = Array::from_slice_f32(&data, &px.shape()).unwrap();
    let after = clip_vision::pooled(&poked, &cfg, &w, &s).expect("pooled");

    let (a, b) = (base.to_vec_f32(&s).unwrap(), after.to_vec_f32(&s).unwrap());
    let delta = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("last patch moves the class token by {delta:.3e}");
    assert!(
        delta > 1e-3,
        "perturbing the last patch moved the class token by only {delta:.3e}; the \
         attention is masked and the tower is reading an image in raster prefix order"
    );
}

/// The projected embedding is 1024 wide, not the tower's 1280 — and it is what
/// IP-Adapter consumes.
#[test]
fn the_projection_narrows_to_the_adapters_width() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no clip_vision fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = VisionConfig::vit_h_14();
    let px = pixels(&refs, &s);

    let embeds = clip_vision::image_embeds(&px, &cfg, &w, &s).expect("image_embeds");
    assert_eq!(
        embeds.shape(),
        vec![1, 1024],
        "ViT-H projects 1280 down to 1024; the adapter expects the projected width"
    );

    // A tower declared without a projection must refuse rather than silently
    // return the pooled 1280.
    let none = VisionConfig {
        projection: false,
        ..cfg
    };
    assert!(
        clip_vision::image_embeds(&px, &none, &w, &s).is_err(),
        "a tower with no visual_projection must refuse to produce image embeds"
    );
}

/// **`preprocess` takes `[0, 1]`, not `[-1, 1]`.**
///
/// The two are the same shape and dtype, so the wrong range is accepted and
/// describes the wrong picture. Pinned as arithmetic rather than trusted.
#[test]
fn preprocess_normalises_with_clips_own_statistics() {
    let s = Stream::gpu();
    // A flat mid-grey image in [0, 1].
    let px = Array::from_slice_f32(&[0.5, 0.5, 0.5], &[1, 1, 1, 3]).unwrap();
    let got = clip_vision::preprocess(&px, &s)
        .unwrap()
        .to_vec_f32(&s)
        .unwrap();
    for (c, &g) in got.iter().enumerate().take(3) {
        let want = (0.5 - clip_vision::CLIP_MEAN[c]) / clip_vision::CLIP_STD[c];
        assert!((g - want).abs() < 1e-6, "channel {c}: {g} vs {want}");
    }
}
