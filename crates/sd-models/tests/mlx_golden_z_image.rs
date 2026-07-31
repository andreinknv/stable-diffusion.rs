//! Z-Image's transformer against `diffusers`.
//!
//! ```bash
//! .venv/bin/python xtask/golden/dump_reference.py z_image --output tests/golden
//! cargo test -p sd-models --features mlx --test mlx_golden_z_image -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::z_image::{self, ZImageConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

const ATOL: f32 = 2e-4;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/z_image")
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let (r, w) = (
        golden().join("reference.safetensors"),
        golden().join("z_image.safetensors"),
    );
    if !r.exists() || !w.exists() {
        return None;
    }
    Some((
        load_safetensors(&r).expect("reference"),
        load_safetensors(&w).expect("weights"),
    ))
}

/// The fixture's geometry.
fn config() -> ZImageConfig {
    ZImageConfig {
        dim: 64,
        layers: 2,
        refiner_layers: 1,
        heads: 4,
        kv_heads: 4,
        in_channels: 16,
        cap_feat_dim: 48,
        norm_eps: 1e-5,
        rope_theta: 256.0,
        axes_dims: vec![8, 4, 4],
        patch_size: 2,
    }
}

#[test]
fn the_transformer_matches_diffusers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no z_image fixture. See the module docs.");
        return;
    };
    let s = Stream::gpu();
    let cfg = config();

    let latent = refs.get("latent").expect("latent");
    let cap = refs.get("cap_feats").expect("cap_feats");
    let timestep = refs.get("timestep").expect("timestep");

    // **Every stage, not just the output.** Three stacks run in sequence and a
    // mismatch at the end says nothing about which one drifted.
    let stages = z_image::forward_stages(latent, cap, timestep, &cfg, &w, &s).expect("forward");
    for (name, got) in [
        ("after_noise_refiner", &stages.after_noise_refiner),
        ("unified_input", &stages.unified_input),
        ("after_context_refiner", &stages.after_context_refiner),
        ("after_layer0", &stages.after_layer0),
        ("after_layers", &stages.after_layers),
    ] {
        let want = refs.get(name).unwrap_or_else(|| panic!("no {name}"));
        assert_eq!(got.shape(), want.shape(), "{name}: shape");
        let (g, e) = (
            got.to_vec_f32(&s).expect("got"),
            want.to_vec_f32(&s).expect("want"),
        );
        let worst = g
            .iter()
            .zip(&e)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("  {name:<24} max_abs {worst:.3e}");
        assert!(
            worst <= ATOL,
            "{name} is {worst:.3e} out — this is the first stage to fail, so it \
             is the one that is wrong, not the ones after it"
        );
    }

    let got = stages.output;
    let want = refs.get("output").expect("output");
    assert_eq!(got.shape(), want.shape(), "output shape");

    let (g, e) = (
        got.to_vec_f32(&s).expect("got"),
        want.to_vec_f32(&s).expect("want"),
    );
    let (mut worst, mut peak) = (0.0f32, 0.0f32);
    for (a, b) in g.iter().zip(&e) {
        worst = worst.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    eprintln!("z-image output  peak {peak:.3}  max_abs {worst:.3e}");
    assert!(worst <= ATOL, "the Z-Image transformer is {worst:.3e} out");
}

/// **Both streams pad to a multiple of 32**, with their own learned tokens.
#[test]
fn sequences_pad_to_a_multiple_of_thirty_two() {
    assert_eq!(z_image::padded_len(1), 32);
    assert_eq!(
        z_image::padded_len(32),
        32,
        "an exact multiple is unchanged"
    );
    assert_eq!(z_image::padded_len(33), 64);
    assert_eq!(z_image::padded_len(7), 32, "the fixture's caption");
    // A 8x6 latent at patch 2 is 4x3 = 12 image tokens, padded to 32.
    assert_eq!(z_image::padded_len(12), 32);
}

/// **The image's `t` coordinate starts after the caption's.**
///
/// Both streams share one rotary axis, so an image token at `t = 0` would sit
/// exactly on the first caption token. The reference offsets the image by the
/// padded caption length plus one, and this pins that the two never collide.
#[test]
fn the_two_streams_do_not_share_a_position() {
    let cap = z_image::caption_ids(32);
    assert_eq!(cap.len(), 32 * 3);
    // **One-based**: the caption walks the t axis from 1, leaving position 0
    // to the image's padding tokens, which all sit at the origin.
    assert_eq!(&cap[..3], &[1, 0, 0]);
    assert_eq!(&cap[3..6], &[2, 0, 0]);
    assert_eq!(&cap[cap.len() - 3..], &[32, 0, 0], "and ends at 32, not 31");

    let img = z_image::image_ids(1, 4, 3);
    assert_eq!(img.len(), 12 * 3);
    // Image walks h and w, and its raw t is 0 — the offset is applied by
    // `forward`, which is what keeps this test honest about where it happens.
    assert_eq!(&img[..3], &[0, 0, 0]);
    assert_eq!(&img[3..6], &[0, 0, 1], "second token is one column across");
    assert_eq!(
        &img[9..12],
        &[0, 1, 0],
        "fourth token wraps to the next row"
    );
}

/// Patchify and unpatchify are inverses, and fold channel-*last*.
#[test]
fn patchify_round_trips() {
    let s = Stream::cpu();
    let (c, f, h, w) = (16usize, 1usize, 8usize, 6usize);
    let v: Vec<f32> = (0..c * f * h * w).map(|i| i as f32).collect();
    let x = Array::from_slice_f32(&v, &[c, f, h, w]).unwrap();

    let patched = z_image::patchify(&x, 2, &s).unwrap();
    // 1 frame, 4 rows, 3 columns of tokens; each carries 2*2*16 features.
    assert_eq!(patched.shape(), vec![12, 64]);

    let back = z_image::unpatchify(&patched, f, h, w, 2, c, &s).unwrap();
    assert_eq!(back.shape(), x.shape());
    assert_eq!(
        back.to_vec_f32(&s).unwrap(),
        x.to_vec_f32(&s).unwrap(),
        "patchify and unpatchify are not inverses"
    );
}
