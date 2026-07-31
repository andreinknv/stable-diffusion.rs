//! MLX against `tests/golden/unet_full`, the fixture `golden_unet.rs` uses.
//!
//! This is the migration's real gate. `docs/handoff.md` rule 1: a feature
//! without a number against diffusers is not done, and the number has to come
//! from the same reference the candle path is held to — not from candle, and
//! not from inspection.
//!
//! ```bash
//! cargo test -p sd-tensor --features mlx --test mlx_golden_unet
//! ```
//!
//! Built up one block at a time, in the order `down_pass_skips_match_diffusers`
//! reads them, because with 25 blocks between input and output a single final
//! number says only that something is wrong. Each layer that lands here stays
//! here.
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_tensor::mlx::{load_safetensors, Array, Stream};

/// `sd_tensor::testing::DEFAULT_ATOL`, the bound `full_unet_matches_diffusers`
/// holds the candle port to.
const ATOL: f32 = 1e-4;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unet_full")
}

/// The fixture and the checkpoint, or `None` with the same message the candle
/// golden tests print, so a machine without them skips rather than fails.
fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let refs = golden_dir().join("reference.safetensors");
    let unet = golden_dir().join("unet.safetensors");
    if !refs.exists() || !unet.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no unet_full fixture.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py unet_full --output tests/golden\n"
        );
        return None;
    }
    Some((
        load_safetensors(&refs).expect("loading reference"),
        load_safetensors(&unet).expect("loading UNet weights"),
    ))
}

/// max_abs against the reference, in the reference's own NCHW layout.
fn compare(got_nhwc: &Array, want_nchw: &Array, stream: &Stream, what: &str) -> f32 {
    let got = got_nhwc
        .transpose(&[0, 3, 1, 2], stream)
        .expect("NHWC -> NCHW")
        .to_vec_f32(stream)
        .expect("reading mlx result");
    let want = want_nchw.to_vec_f32(stream).expect("reading reference");
    assert_eq!(got.len(), want.len(), "{what}: element count");
    let worst = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{what}: max_abs {worst:.3e} against atol {ATOL:.0e}");
    worst
}

/// `conv_in`, which is entry 0 of the skip stack.
///
/// It is the smallest thing that exercises every part of the layout decision at
/// once: NCHW fixture in, NHWC convolution, `(out, kh, kw, in)` weights, and
/// the bias broadcasting over the last axis rather than a middle one. A
/// transpose mistake anywhere here shows up as a large number rather than a
/// subtle one, which is why it is the first gate rather than a later one.
#[test]
fn conv_in_matches_diffusers() {
    let Some((refs, weights)) = fixtures() else {
        return;
    };
    let s = Stream::gpu();

    let sample = refs.get("sample").expect("sample");
    let want = refs.get("down_00").expect("down_00");
    let w = weights.get("conv_in.weight").expect("conv_in.weight");
    let b = weights.get("conv_in.bias").expect("conv_in.bias");

    assert_eq!(sample.shape(), vec![1, 4, 32, 32], "fixture sample is NCHW");
    assert_eq!(w.shape(), vec![320, 4, 3, 3], "diffusers weights are OIHW");

    let x = sample.transpose(&[0, 2, 3, 1], &s).unwrap(); // NCHW -> NHWC
    let k = w.transpose(&[0, 2, 3, 1], &s).unwrap(); // OIHW -> OHWI
    let got = x
        .conv2d(&k, (1, 1), (1, 1), (1, 1), 1, &s)
        .unwrap()
        // NHWC puts channels last, so the per-channel bias broadcasts with no
        // reshape — the one place channels-last is simpler than candle.
        .add(b, &s)
        .unwrap();

    assert_eq!(got.shape(), vec![1, 32, 32, 320], "NHWC output");
    let worst = compare(&got, want, &s, "conv_in");
    assert!(
        worst <= ATOL,
        "conv_in is {worst:.3e} from diffusers, past atol {ATOL:.0e}"
    );
}

/// The checkpoint really is the one `golden_unet.rs` uses, and MLX reads every
/// tensor in it. Cheap, and it fails loudly if the symlink ever goes stale.
#[test]
fn the_checkpoint_loads_whole() {
    let Some((_, weights)) = fixtures() else {
        return;
    };
    assert!(
        weights.len() > 600,
        "SD 1.5's UNet has ~686 tensors, got {}",
        weights.len()
    );
    for key in [
        "conv_in.weight",
        "conv_out.weight",
        "time_embedding.linear_1.weight",
        "down_blocks.0.resnets.0.norm1.weight",
        "mid_block.attentions.0.proj_in.weight",
    ] {
        assert!(weights.contains_key(key), "missing {key}");
    }
}
