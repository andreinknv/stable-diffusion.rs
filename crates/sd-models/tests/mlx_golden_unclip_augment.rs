//! unCLIP's noise augmentation on MLX, against `tests/golden/unclip`.
//!
//! The UNet conditions on a CLIP image embedding that has been noised by a
//! stated amount, and told how much. Two levels are gated because level 0 is
//! nearly the identity — a port that ignored `level` entirely would match it —
//! and 250 is where the mixing is visible.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_unclip_augment -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::unclip;
use sd_tensor::mlx::{load_safetensors, Array, Stream};

/// `golden_unclip.rs`'s `AUGMENT_TOL`, and not a tighter one.
///
/// This looked like closed-form arithmetic on order-1 numbers, which would
/// justify 1e-5. It is not: the augmented embedding peaks at **7.3**, so
/// 1.514e-5 at level 250 is a relative error of 2e-6 — f32 accumulation
/// through a whiten, a mix and an un-whiten. The candle port measures the same
/// class of difference and holds to 5e-5 for the same reason.
const TOL: f32 = 5e-5;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unclip")
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let (refs, w) = (
        golden().join("reference.safetensors"),
        golden().join("image_normalizer.safetensors"),
    );
    if !refs.exists() || !w.exists() {
        return None;
    }
    Some((
        load_safetensors(&refs).expect("reference"),
        load_safetensors(&w).expect("normalizer"),
    ))
}

fn worst(got: &Array, want: &Array, s: &Stream) -> f32 {
    let (a, b) = (got.to_vec_f32(s).unwrap(), want.to_vec_f32(s).unwrap());
    assert_eq!(a.len(), b.len(), "element count");
    a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn the_augmentation_matches_diffusers_at_two_levels() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no unCLIP fixture.");
        return;
    };
    let s = Stream::gpu();
    let alphas = sd_models::unclip::cosine_alphas_cumprod(unclip::TRAIN_TIMESTEPS);
    let embeds = refs.get("image_embeds").expect("image_embeds");
    let noise = refs.get("noise").expect("noise");

    for (level, key) in [(0usize, "noised_0"), (250, "noised_250")] {
        let got = unclip::augment(embeds, level, noise, &alphas, &w, &s).expect("augment");
        assert_eq!(got.shape(), vec![1, 2048], "the embedding and its level");
        let d = worst(&got, refs.get(key).expect(key), &s);
        eprintln!("augment level {level:<4} max_abs {d:.3e}  tol {TOL:.0e}");
        assert!(d <= TOL, "level {level} is {d:.3e} out");
    }
}

/// **The level must change the answer**, or the comparison above proves only
/// that one of the two matches.
#[test]
fn the_level_changes_the_augmentation() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no unCLIP fixture.");
        return;
    };
    let s = Stream::gpu();
    let alphas = sd_models::unclip::cosine_alphas_cumprod(unclip::TRAIN_TIMESTEPS);
    let embeds = refs.get("image_embeds").unwrap();
    let noise = refs.get("noise").unwrap();

    let a = unclip::augment(embeds, 0, noise, &alphas, &w, &s).unwrap();
    let b = unclip::augment(embeds, 250, noise, &alphas, &w, &s).unwrap();
    let spread = worst(&a, &b, &s);
    eprintln!("levels 0 and 250 differ by {spread:.3}");
    assert!(spread > 0.1, "the level is being ignored ({spread:.3e})");
}

/// **The halves are in the right order**, which no shape check can see: both
/// are 1024 wide, so a reversed concatenation is exactly the right size.
///
/// The first half is the noised embedding, so at level 0 it must still
/// resemble the input; the second is a sinusoid, which is bounded by 1.
#[test]
fn the_embedding_comes_before_the_level() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no unCLIP fixture.");
        return;
    };
    let s = Stream::gpu();
    let alphas = sd_models::unclip::cosine_alphas_cumprod(unclip::TRAIN_TIMESTEPS);
    let embeds = refs.get("image_embeds").unwrap();
    let noise = refs.get("noise").unwrap();

    let got = unclip::augment(embeds, 0, noise, &alphas, &w, &s).unwrap();
    let v = got.to_vec_f32(&s).unwrap();
    let (first, second) = (&v[..1024], &v[1024..]);

    // A sinusoid embedding is exactly bounded by 1; a CLIP image embedding is
    // not, and this fixture's is not.
    let peak = |x: &[f32]| x.iter().fold(0.0f32, |m, y| m.max(y.abs()));
    let (pe, ps) = (peak(first), peak(second));
    eprintln!("first half peak {pe:.3}, second half peak {ps:.3}");
    assert!(
        ps <= 1.0 + 1e-6,
        "the second half peaks at {ps:.3}; a sinusoid cannot exceed 1, so the halves \
         look reversed"
    );
    assert!(
        pe > 1.0,
        "the first half peaks at {pe:.3}; that is within a sinusoid's range, so the \
         halves look reversed"
    );
}

/// The unconditional row is zeros of the **whole** width, including the half
/// that would carry the noise level.
#[test]
fn the_unconditional_row_is_entirely_zero() {
    let s = Stream::gpu();
    let z = unclip::unconditional(2, 1024).unwrap();
    assert_eq!(z.shape(), vec![2, 2048]);
    assert!(
        z.to_vec_f32(&s).unwrap().iter().all(|&v| v == 0.0),
        "the unconditional row must be zeros, not an augmented zero embedding"
    );
}

/// `scale` and `unscale` are inverses.
///
/// **The published unCLIP weights carry `mean = 0` and `std = 1`**, so the
/// golden comparison above runs entirely through an identity and cannot see
/// these swapped or missing. This uses statistics of its own.
#[test]
fn scale_and_unscale_are_inverses_with_real_statistics() {
    let s = Stream::gpu();
    let mut w: HashMap<String, Array> = HashMap::new();
    w.insert(
        "mean".into(),
        Array::from_slice_f32(&[1.0, -2.0, 0.5], &[1, 3]).unwrap(),
    );
    w.insert(
        "std".into(),
        Array::from_slice_f32(&[2.0, 0.5, 4.0], &[1, 3]).unwrap(),
    );
    let x = Array::from_slice_f32(&[3.0, 1.0, -1.0], &[1, 3]).unwrap();

    let scaled = unclip::scale(&x, &w, &s).unwrap();
    // (3-1)/2 = 1, (1+2)/0.5 = 6, (-1-0.5)/4 = -0.375
    let got = scaled.to_vec_f32(&s).unwrap();
    for (a, b) in got.iter().zip([1.0f32, 6.0, -0.375]) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    let round = unclip::unscale(&scaled, &w, &s).unwrap();
    for (a, b) in round
        .to_vec_f32(&s)
        .unwrap()
        .iter()
        .zip([3.0f32, 1.0, -1.0])
    {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}
