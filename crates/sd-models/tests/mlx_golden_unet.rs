//! SD 1.5's UNet blocks on MLX, against `tests/golden/unet_full`.
//!
//! Same fixture, same reference and same bound as `golden_unet.rs` holds the
//! candle path to. Built up in the order `down_pass_skips_match_diffusers`
//! reads the skip stack, because with 25 blocks between input and output one
//! final number says only that something is wrong.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_unet -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::{down_pass, resnet_block, timestep_embedding, transformer_2d};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

/// `sd_tensor::testing::DEFAULT_ATOL`.
const ATOL: f32 = 1e-4;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unet_full")
}

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

fn max_abs(got_nhwc: &Array, want_nchw: &Array, s: &Stream, what: &str) -> f32 {
    let got = got_nhwc
        .transpose(&[0, 3, 1, 2], s)
        .expect("NHWC -> NCHW")
        .to_vec_f32(s)
        .expect("mlx result");
    let want = want_nchw.to_vec_f32(s).expect("reference");
    assert_eq!(got.len(), want.len(), "{what}: element count");
    let worst = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{what:<12} max_abs {worst:.3e}   atol {ATOL:.0e}");
    worst
}

/// conv_in, then the first resnet, then the first transformer — which is
/// `down_01`, the entry where the whole UNet first diverged when the
/// `mlx-examples` epsilon bug was still in play.
#[test]
fn down_block_0_matches_diffusers() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();

    let sample = refs.get("sample").expect("sample");
    let context = refs.get("context").expect("context");
    let timestep = refs.get("timestep").expect("timestep");

    // conv_in -> down_00
    let x = sample.transpose(&[0, 2, 3, 1], &s).unwrap();
    let k = w
        .get("conv_in.weight")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let h = x
        .conv2d(&k, (1, 1), (1, 1), (1, 1), 1, &s)
        .unwrap()
        .add(w.get("conv_in.bias").unwrap(), &s)
        .unwrap();
    let worst = max_abs(&h, refs.get("down_00").unwrap(), &s, "down_00");
    assert!(worst <= ATOL, "conv_in: {worst:.3e}");

    // SD 1.5's first block is 320 channels wide.
    let temb = timestep_embedding(timestep, 320, &w, &s).unwrap();

    let h = resnet_block(&h, &temb, &w, "down_blocks.0.resnets.0", &s).unwrap();
    // 8 heads, one transformer layer, as UNetConfig::sd15 says.
    let h = transformer_2d(&h, context, 8, 1, &w, "down_blocks.0.attentions.0", &s).unwrap();

    let worst = max_abs(&h, refs.get("down_01").unwrap(), &s, "down_01");
    assert!(
        worst <= ATOL,
        "resnet0 + attn0 is {worst:.3e} from diffusers, past atol {ATOL:.0e}"
    );
}

/// The whole down pass, compared entry by entry.
///
/// `down_pass_skips_match_diffusers` explains why this is not one number at the
/// end: with 25 blocks between input and output, a single figure says only that
/// something is wrong. The index of the first bad skip says where — 0 is
/// conv_in, 1-3 is down block 0, and all-green through 11 means the fault is
/// downstream.
#[test]
fn the_whole_down_pass_matches_diffusers() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();

    let sample = refs.get("sample").expect("sample");
    let context = refs.get("context").expect("context");
    let temb = timestep_embedding(refs.get("timestep").expect("timestep"), 320, &w, &s).unwrap();

    let x = sample.transpose(&[0, 2, 3, 1], &s).unwrap();
    let (_deepest, skips) = down_pass(&x, &temb, context, &w, &s).unwrap();

    assert_eq!(skips.len(), 12, "skip stack must have 12 entries");

    let mut first_bad = None;
    for (i, got) in skips.iter().enumerate() {
        let name = format!("down_{i:02}");
        let want = refs
            .get(&name)
            .unwrap_or_else(|| panic!("reference has no {name}"));
        let worst = max_abs(got, want, &s, &name);
        if worst > ATOL && first_bad.is_none() {
            first_bad = Some((i, worst));
        }
    }
    if let Some((i, worst)) = first_bad {
        panic!(
            "first bad skip is index {i} at {worst:.3e}, past atol {ATOL:.0e}. \
             0 is conv_in; 1-3 is down block 0; all-green means the fault is \
             downstream of the down pass."
        );
    }
}
