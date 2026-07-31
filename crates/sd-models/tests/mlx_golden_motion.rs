//! AnimateDiff motion modules on MLX, against `tests/golden/motion`.
//!
//! The fixture's `hidden` is `[4, 320, 8, 8]` — four frames of one clip — and
//! `output` is the module applied to it. That single module is the whole
//! mechanism: get the temporal permute wrong and it still runs, keeps every
//! shape, and blurs across space instead of time.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_motion -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::{motion, unet_forward_adapters, Adapters, Motion, UNetConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

const ATOL: f32 = 1e-4;

/// The whole-UNet bounds, copied from `mlx_golden_unet.rs` with its reasoning:
/// the reference's own noise floor on this fixture is 2.757e-4, so `ATOL`
/// alone would fail a correct implementation.
const UNET_RTOL: f32 = 1e-3;
const UNET_TOL: f32 = 1e-3;

#[test]
fn a_motion_module_matches_diffusers() {
    let g = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/motion");
    let (refs_p, adapter_p) = (
        g.join("reference.safetensors"),
        g.join("motion_adapter.safetensors"),
    );
    if !refs_p.exists() || !adapter_p.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no motion fixture.");
        return;
    }
    let refs = load_safetensors(&refs_p).expect("reference");
    let w = load_safetensors(&adapter_p).expect("adapter");
    let s = Stream::gpu();

    let hidden = refs.get("hidden").expect("hidden");
    let frames = hidden.shape()[0];
    let x = hidden.transpose(&[0, 2, 3, 1], &s).unwrap();

    let got = motion::forward(&x, frames, &w, "down_blocks.0.motion_modules.0", &s)
        .unwrap()
        .transpose(&[0, 3, 1, 2], &s)
        .unwrap();

    let want = refs.get("output").expect("output");
    assert_eq!(got.shape(), want.shape(), "shape survives the round trip");

    let g_ = got.to_vec_f32(&s).unwrap();
    let w_ = want.to_vec_f32(&s).unwrap();
    let (mut worst, mut peak) = (0.0f32, 0.0f32);
    for (a, b) in g_.iter().zip(&w_) {
        worst = worst.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    eprintln!("motion  peak {peak:.3}  max_abs {worst:.3e}   atol {ATOL:.0e}");
    assert!(
        worst <= ATOL,
        "the motion module is {worst:.3e} from diffusers"
    );
}

/// **The mixing must run along the frame axis, not the spatial one.**
///
/// A perturbation at one pixel of one frame must reach *that same pixel in the
/// other frames* far more strongly than it reaches other pixels. Get the
/// temporal permute backwards and the module still runs, keeps every shape, and
/// fails exactly this: the response spreads across space instead of time.
///
/// The two are not perfectly isolated — GroupNorm's statistics span the whole
/// clip, so every element leaks into every other through the mean and variance.
/// That leak is what the ratio below measures against, not zero.
#[test]
fn the_mixing_runs_along_the_frame_axis() {
    let g = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/motion");
    let adapter_p = g.join("motion_adapter.safetensors");
    if !adapter_p.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no motion fixture.");
        return;
    }
    let w = load_safetensors(&adapter_p).expect("adapter");
    let s = Stream::gpu();
    let prefix = "down_blocks.0.motion_modules.0";

    // Four frames of an 8x8 image, 320 channels: [f, h, w, c] in NHWC. The grid
    // has to be the real one — at 2x2 a single poked pixel is a sixteenth of
    // each GroupNorm group's population, and that leak alone swamps the signal.
    let (frames, hw, c) = (4usize, 8usize, 320usize);
    let per_pixel = c;
    let per_frame = hw * hw * c;
    let mut base = vec![0f32; frames * per_frame];
    for (i, v) in base.iter_mut().enumerate() {
        // Deterministic and varied; no RNG needed to make the point.
        *v = ((i % 37) as f32 - 18.0) * 0.05;
    }

    // Poke pixel (0,0) of frame 0 only.
    let mut poked = base.clone();
    for v in poked[0..per_pixel].iter_mut() {
        *v += 1.0;
    }

    let run = |data: &[f32]| -> Vec<f32> {
        let x = Array::from_slice_f32(data, &[frames, hw, hw, c]).unwrap();
        motion::forward(&x, frames, &w, prefix, &s)
            .unwrap()
            .to_vec_f32(&s)
            .unwrap()
    };
    let (a, b) = (run(&base), run(&poked));

    // The largest response at a given (frame, pixel).
    let at = |frame: usize, pixel: usize| -> f32 {
        let off = frame * per_frame + pixel * per_pixel;
        (0..per_pixel)
            .map(|i| (a[off + i] - b[off + i]).abs())
            .fold(0.0f32, f32::max)
    };

    // Same pixel, the frames that were not poked.
    let across_time = (1..frames).map(|f| at(f, 0)).fold(0.0f32, f32::max);
    // Other pixels, every frame — including the poked one.
    let across_space = (0..frames)
        .flat_map(|f| (1..hw * hw).map(move |p| (f, p)))
        .map(|(f, p)| at(f, p))
        .fold(0.0f32, f32::max);

    eprintln!("perturbation  across_time {across_time:.3e}  across_space {across_space:.3e}");
    assert!(
        across_time > across_space * 10.0,
        "a poke at one pixel of one frame reached other pixels ({across_space:.3e}) about as \
         strongly as it reached the same pixel in other frames ({across_time:.3e}); the \
         attention is mixing across space, not time"
    );
}

// -- the wiring, not just the module ---------------------------------------

/// Twenty-one modules go in, one after each resnet, and *where* each lands is
/// invisible to a per-module check — every insertion order keeps every shape
/// valid. Only an end-to-end comparison says the order is right.
///
/// Held to the same `rtol` the rest of the UNet is, for the same reason:
/// `golden_unclip.rs` measures the reference's own noise floor at 2.757e-4, so
/// `DEFAULT_ATOL` alone would fail a correct implementation.
#[test]
fn a_unet_with_motion_modules_matches_diffusers() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let g = root.join("tests/golden/motion");
    let unet_p = root.join("tests/golden/unet_full/unet.safetensors");
    let (refs_p, adapter_p) = (
        g.join("reference.safetensors"),
        g.join("motion_adapter.safetensors"),
    );
    if !refs_p.exists() || !adapter_p.exists() || !unet_p.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no motion or UNet fixture.");
        return;
    }
    let refs = load_safetensors(&refs_p).expect("reference");
    let Some(want) = refs.get("unet_output") else {
        sd_tensor::skip_missing_fixture!("SKIP: the fixture predates the whole-UNet dump.");
        return;
    };
    let adapter = load_safetensors(&adapter_p).expect("adapter");
    let w = load_safetensors(&unet_p).expect("unet");
    let s = Stream::gpu();

    let sample = refs.get("unet_sample").expect("unet_sample");
    // Frames ride on the batch axis; the fixture is one clip.
    let frames = sample.shape()[0];
    let x = sample.transpose(&[0, 2, 3, 1], &s).unwrap();

    let m = Motion {
        weights: &adapter,
        frames,
    };
    let ad = Adapters {
        motion: Some(&m),
        ..Default::default()
    };
    let got = unet_forward_adapters(
        &x,
        refs.get("unet_timestep").expect("timestep"),
        refs.get("unet_text").expect("text"),
        None,
        None,
        &ad,
        &UNetConfig::sd15(),
        &w,
        &s,
    )
    .unwrap()
    .transpose(&[0, 3, 1, 2], &s)
    .unwrap();

    assert_eq!(got.shape(), want.shape());
    let g_ = got.to_vec_f32(&s).unwrap();
    let w_ = want.to_vec_f32(&s).unwrap();
    let (mut peak, mut worst, mut exc) = (0.0f32, 0.0f32, 0.0f32);
    for (a, b) in g_.iter().zip(&w_) {
        let d = (a - b).abs();
        worst = worst.max(d);
        peak = peak.max(b.abs());
        exc = exc.max(d - UNET_RTOL * b.abs());
    }
    let exc = exc.max(0.0);
    eprintln!("motion unet  peak {peak:.3}  max_abs {worst:.3e}  excess {exc:.3e}");
    assert!(
        exc <= UNET_TOL,
        "the animated UNet is {exc:.3e} past the reference noise floor"
    );
}

/// **Without the adapter, the same UNet must give a different answer.** A
/// motion pass that silently found no modules would match the plain UNet
/// exactly and pass nothing but this test.
#[test]
fn the_adapter_actually_changes_the_unet() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let g = root.join("tests/golden/motion");
    let unet_p = root.join("tests/golden/unet_full/unet.safetensors");
    let (refs_p, adapter_p) = (
        g.join("reference.safetensors"),
        g.join("motion_adapter.safetensors"),
    );
    if !refs_p.exists() || !adapter_p.exists() || !unet_p.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no motion or UNet fixture.");
        return;
    }
    let refs = load_safetensors(&refs_p).expect("reference");
    let adapter = load_safetensors(&adapter_p).expect("adapter");
    let w = load_safetensors(&unet_p).expect("unet");
    let s = Stream::gpu();

    let sample = refs.get("unet_sample").expect("unet_sample");
    let frames = sample.shape()[0];
    let x = sample.transpose(&[0, 2, 3, 1], &s).unwrap();
    let (t, text) = (
        refs.get("unet_timestep").expect("timestep"),
        refs.get("unet_text").expect("text"),
    );
    let cfg = UNetConfig::sd15();

    let m = Motion {
        weights: &adapter,
        frames,
    };
    let run = |ad: &Adapters| -> Vec<f32> {
        unet_forward_adapters(&x, t, text, None, None, ad, &cfg, &w, &s)
            .unwrap()
            .to_vec_f32(&s)
            .unwrap()
    };
    let animated = run(&Adapters {
        motion: Some(&m),
        ..Default::default()
    });
    let plain = run(&Adapters::default());

    let delta = animated
        .iter()
        .zip(&plain)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("adapter changes the output by {delta:.3e}");
    assert!(
        delta > 1e-3,
        "the adapter moved the output by {delta:.3e}; the modules are not being found"
    );
}
