//! Flux's transformer on MLX, against `tests/golden/flux_transformer`.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_flux -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::flux::{self, FluxConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

fn golden(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(sub)
}

fn relative(got: &Array, want: &Array, s: &Stream, what: &str) -> f32 {
    let g = got.to_vec_f32(s).expect("mlx result");
    let w = want.to_vec_f32(s).expect("reference");
    assert_eq!(g.len(), w.len(), "{what}: element count");
    let (mut worst, mut peak) = (0.0f32, 0.0f32);
    for (a, b) in g.iter().zip(&w) {
        worst = worst.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    let rel = worst / peak.max(f32::MIN_POSITIVE);
    eprintln!("{what:<12} peak {peak:>9.3}  max_abs {worst:.3e}  relative {rel:.2e}");
    rel
}

#[test]
fn the_flux_transformer_matches_the_reference() {
    let refs_path = golden("flux_transformer/reference.safetensors");
    let w_path = golden("flux/flux-mini.safetensors");
    if !refs_path.exists() || !w_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no flux fixture.");
        return;
    }
    let refs = load_safetensors(&refs_path).expect("reference");
    let s = Stream::gpu();
    let cfg = FluxConfig::mini();

    let raw = load_safetensors(&w_path).expect("weights");
    let mut w = std::collections::HashMap::new();
    for (name, t) in &raw {
        w.insert(name.clone(), t.to_f32(&s).expect("f32"));
    }

    let scalar = |k: &str| -> usize { refs.get(k).unwrap().to_vec_f32(&s).unwrap()[0] as usize };
    let (lat_h, lat_w) = (scalar("latent_h"), scalar("latent_w"));
    let ids = flux::image_ids(lat_h, lat_w);

    let got = flux::forward(
        refs.get("hidden_states").expect("hidden_states"),
        &ids,
        refs.get("encoder_hidden_states")
            .expect("encoder_hidden_states"),
        refs.get("timestep").expect("timestep"),
        refs.get("pooled_projections").expect("pooled_projections"),
        Some(refs.get("guidance").expect("guidance")),
        &cfg,
        &w,
        &s,
    )
    .unwrap();

    let want = refs.get("output").expect("output");
    assert_eq!(got.shape(), want.shape(), "output shape");
    let rel = relative(&got, want, &s, "output");
    assert!(rel <= 1e-3, "the Flux transformer is {rel:.3e} relative");
}

/// A checkpoint distilled on a guidance scale must be given one, and one that
/// was not must refuse it. Both mistakes otherwise render a plausible wrong
/// image.
#[test]
fn guidance_is_required_exactly_when_the_checkpoint_has_it() {
    let refs_path = golden("flux_transformer/reference.safetensors");
    let w_path = golden("flux/flux-mini.safetensors");
    if !refs_path.exists() || !w_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no flux fixture.");
        return;
    }
    let refs = load_safetensors(&refs_path).expect("reference");
    let s = Stream::gpu();
    let raw = load_safetensors(&w_path).expect("weights");
    let mut w = std::collections::HashMap::new();
    for (name, t) in &raw {
        w.insert(name.clone(), t.to_f32(&s).expect("f32"));
    }
    let ids = flux::image_ids(16, 16);
    let call = |cfg: &FluxConfig, g: Option<&Array>| {
        flux::forward(
            refs.get("hidden_states").unwrap(),
            &ids,
            refs.get("encoder_hidden_states").unwrap(),
            refs.get("timestep").unwrap(),
            refs.get("pooled_projections").unwrap(),
            g,
            cfg,
            &w,
            &s,
        )
    };
    assert!(
        call(&FluxConfig::mini(), None).is_err(),
        "mini needs guidance"
    );
    assert!(
        call(&FluxConfig::schnell(), refs.get("guidance")).is_err(),
        "schnell takes none"
    );
}

/// **Pins the rotary convention as interleaved.**
///
/// `rope.rs` warns that a transposed or half-split rotation is still a
/// rotation, so it yields a coherent image with the geometry subtly wrong
/// rather than an error. The golden test above already rules that out — a
/// wrong convention is an O(1) divergence, not 3.5e-6 — but it rules it out
/// implicitly, across a whole model. This checks the arithmetic directly, so a
/// future edit to `apply_rope` fails here with a name rather than there with a
/// number.
///
/// With one frequency and a 2-wide head, the rotation is exactly
/// `[[cos, -sin], [sin, cos]]` applied to `(x0, x1)`.
#[test]
fn rope_is_interleaved_and_rotates_the_right_pairs() {
    let s = Stream::gpu();
    // theta irrelevant at dim 2: omega[0] = 1, so the angle is the position.
    let axes = [2usize];
    // One token at position 1 radian.
    let pe = flux::embed_nd(&[1.0], 1, &axes, 10_000.0).unwrap();
    let cos = pe.cos.to_vec_f32(&s).unwrap();
    let sin = pe.sin.to_vec_f32(&s).unwrap();
    assert!((cos[0] - 1.0f32.cos()).abs() < 1e-6, "cos of the position");
    assert!((sin[0] - 1.0f32.sin()).abs() < 1e-6, "sin of the position");

    // x = (3, 4) as one head of width 2.
    let x = Array::from_slice_f32(&[3.0, 4.0], &[1, 1, 1, 2]).unwrap();
    let got = flux::rotate(&x, &pe, &s).unwrap().to_vec_f32(&s).unwrap();

    let (c, sn) = (1.0f32.cos(), 1.0f32.sin());
    let want = [3.0 * c - 4.0 * sn, 3.0 * sn + 4.0 * c];
    for i in 0..2 {
        assert!(
            (got[i] - want[i]).abs() < 1e-5,
            "element {i}: {} != {} — the rotation is not the interleaved \
             [[cos, -sin], [sin, cos]] on (x0, x1)",
            got[i],
            want[i]
        );
    }

    // And it is *not* the split-half convention, which would pair x0 with x1
    // as halves rather than as a couple. At width 2 the two coincide, so use a
    // 4-wide head where they differ.
    let axes4 = [4usize];
    let pe4 = flux::embed_nd(&[1.0], 1, &axes4, 10_000.0).unwrap();
    let x4 = Array::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]).unwrap();
    let g4 = flux::rotate(&x4, &pe4, &s).unwrap().to_vec_f32(&s).unwrap();
    let c4 = pe4.cos.to_vec_f32(&s).unwrap();
    let s4 = pe4.sin.to_vec_f32(&s).unwrap();
    // Interleaved: pairs are (0,1) and (2,3).
    let expect = [
        1.0 * c4[0] - 2.0 * s4[0],
        1.0 * s4[0] + 2.0 * c4[0],
        3.0 * c4[1] - 4.0 * s4[1],
        3.0 * s4[1] + 4.0 * c4[1],
    ];
    // Split-half would pair (0,2) and (1,3) instead.
    let split_half_first = 1.0 * c4[0] - 3.0 * s4[0];
    for i in 0..4 {
        assert!((g4[i] - expect[i]).abs() < 1e-5, "interleaved element {i}");
    }
    assert!(
        (g4[0] - split_half_first).abs() > 1e-6,
        "interleaved and split-half must differ at width 4, or this test proves nothing"
    );
}
