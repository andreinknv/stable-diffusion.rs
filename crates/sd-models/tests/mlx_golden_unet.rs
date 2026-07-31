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

use sd_models::mlx::{down_pass, resnet_block, timestep_embedding, transformer_2d, UNetConfig};
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
    let h = transformer_2d(
        &h,
        context,
        8,
        1,
        false,
        &w,
        "down_blocks.0.attentions.0",
        &s,
    )
    .unwrap();

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
    let cfg = UNetConfig::sd15();

    let sample = refs.get("sample").expect("sample");
    let context = refs.get("context").expect("context");
    let temb = timestep_embedding(refs.get("timestep").expect("timestep"), 320, &w, &s).unwrap();

    let x = sample.transpose(&[0, 2, 3, 1], &s).unwrap();
    let (_deepest, skips) = down_pass(&x, &temb, context, &cfg, &w, &s).unwrap();

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

/// The whole UNet, against `full_unet_matches_diffusers`' bound.
///
/// `golden_unet.rs` holds the candle port to `DEFAULT_ATOL` here and measures
/// 1.1e-5, noting the accumulated error stays in the deep 1280-channel blocks
/// and `conv_out` projects back down to 4 channels. The same should hold on
/// MLX, and this is where that is checked rather than assumed.
#[test]
fn the_whole_unet_matches_diffusers() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();
    let cfg = UNetConfig::sd15();

    let x = refs
        .get("sample")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let got = sd_models::mlx::unet_forward(
        &x,
        refs.get("timestep").unwrap(),
        refs.get("context").unwrap(),
        &cfg,
        &w,
        &s,
    )
    .unwrap();

    let worst = max_abs(&got, refs.get("output").unwrap(), &s, "output");
    assert!(
        worst <= ATOL,
        "the UNet is {worst:.3e} from diffusers, past atol {ATOL:.0e}; \
         candle measures 1.1e-5 on this fixture"
    );
}

/// The mid block, on the looser bound `golden_unet.rs` uses for intermediates.
///
/// Not 1e-4: `reference_precision.py` measured diffusers missing its *own* f64
/// by 1.108e-4 on `mid_output`, so 1e-4 there would be testing float32 rather
/// than this port. `UNET_ATOL` is 1e-3 for exactly this reason.
#[test]
fn the_mid_block_matches_diffusers() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();
    let cfg = UNetConfig::sd15();
    const MID_ATOL: f32 = 1e-3;

    let x = refs
        .get("sample")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let temb = timestep_embedding(refs.get("timestep").unwrap(), 320, &w, &s).unwrap();
    let (deepest, _skips) =
        down_pass(&x, &temb, refs.get("context").unwrap(), &cfg, &w, &s).unwrap();
    let mid =
        sd_models::mlx::mid_block(&deepest, &temb, refs.get("context").unwrap(), &cfg, &w, &s)
            .unwrap();

    let worst = max_abs(&mid, refs.get("mid_output").unwrap(), &s, "mid_output");
    assert!(
        worst <= MID_ATOL,
        "mid_block is {worst:.3e}, past {MID_ATOL:.0e}"
    );
}

/// SD 2.x, against `tests/golden/unet_full_cross1024`.
///
/// Same block shapes as SD 1.5 and three differences, all in `UNetConfig`:
/// 64-wide heads throughout rather than 40 at the first block, cross-attention
/// at 1024, and **linear projections** in the transformer where SD 1.5 uses
/// 1x1 convolutions. The last is the one that matters most here — the weights
/// differ in rank, so choosing wrongly fails to load rather than rendering
/// wrongly, which is the good failure.
#[test]
fn the_sd2_unet_matches_diffusers() {
    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unet_full_cross1024");
    let (refs_path, unet_path) = (
        dir.join("reference.safetensors"),
        dir.join("unet.safetensors"),
    );
    if !refs_path.exists() || !unet_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no unet_full_cross1024 fixture.");
        return;
    }
    let refs = load_safetensors(&refs_path).expect("reference");
    let w = load_safetensors(&unet_path).expect("weights");
    let s = Stream::gpu();
    let cfg = UNetConfig::sd2();

    let context = refs.get("context").expect("context");
    assert_eq!(
        context.shape(),
        vec![1, 77, 1024],
        "SD 2.x cross-attends at 1024"
    );

    let x = refs
        .get("sample")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let got =
        sd_models::mlx::unet_forward(&x, refs.get("timestep").unwrap(), context, &cfg, &w, &s)
            .unwrap();

    let worst = max_abs(&got, refs.get("output").unwrap(), &s, "sd2 output");
    assert!(
        worst <= ATOL,
        "the SD 2.x UNet is {worst:.3e} from diffusers, past atol {ATOL:.0e}"
    );
}

/// SDXL base, against `tests/golden/sdxl_unet`.
///
/// Three blocks rather than four, attention on the **last two** rather than the
/// first three, transformer depths `[1, 2, 10]`, and the text_time
/// micro-conditioning: six time ids sinusoidally embedded at 256 each, then
/// concatenated with the 1280-wide pooled text embedding.
///
/// **Pooled first.** 1280 + 1536 = 2816 either way, so the reversed order loads
/// and runs and simply conditions on nonsense.
#[test]
fn the_sdxl_unet_matches_diffusers() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/sdxl_unet");
    let (refs_path, unet_path) = (
        dir.join("reference.safetensors"),
        dir.join("unet.safetensors"),
    );
    if !refs_path.exists() || !unet_path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no sdxl_unet fixture.");
        return;
    }
    let refs = load_safetensors(&refs_path).expect("reference");
    let w = load_safetensors(&unet_path).expect("weights");
    let s = Stream::gpu();
    let cfg = UNetConfig::sdxl();

    let context = refs.get("context").expect("context");
    assert_eq!(
        context.shape(),
        vec![1, 77, 2048],
        "SDXL cross-attends at 2048"
    );
    let pooled = refs.get("pooled").expect("pooled");
    let time_ids = refs.get("time_ids").expect("time_ids");
    assert_eq!(pooled.shape(), vec![1, 1280]);
    assert_eq!(time_ids.shape(), vec![1, 6]);

    let x = refs
        .get("sample")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let got = sd_models::mlx::unet_forward_with(
        &x,
        refs.get("timestep").unwrap(),
        context,
        Some((pooled, time_ids)),
        &cfg,
        &w,
        &s,
    )
    .unwrap();

    let worst = max_abs(&got, refs.get("output").unwrap(), &s, "sdxl output");
    assert!(
        worst <= ATOL,
        "the SDXL UNet is {worst:.3e} from diffusers, past atol {ATOL:.0e}"
    );
}

/// A UNet with micro-conditioning refuses to run without it, and one without
/// refuses to accept it. Both mistakes otherwise render a plausible wrong
/// image.
#[test]
fn micro_conditioning_is_required_when_the_config_declares_it() {
    let Some((refs, w)) = fixtures() else { return };
    let s = Stream::gpu();
    let x = refs
        .get("sample")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let (t, ctx) = (refs.get("timestep").unwrap(), refs.get("context").unwrap());

    // SDXL config, no conditioning supplied.
    assert!(
        sd_models::mlx::unet_forward_with(&x, t, ctx, None, &UNetConfig::sdxl(), &w, &s).is_err(),
        "an SDXL config must refuse to run without micro-conditioning"
    );
    // SD 1.5 config, conditioning supplied.
    assert!(
        sd_models::mlx::unet_forward_with(
            &x,
            t,
            ctx,
            Some((ctx, ctx)),
            &UNetConfig::sd15(),
            &w,
            &s
        )
        .is_err(),
        "SD 1.5 must refuse micro-conditioning it cannot use"
    );
}
