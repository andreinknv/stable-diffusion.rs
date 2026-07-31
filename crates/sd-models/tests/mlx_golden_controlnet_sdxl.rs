//! An SDXL ControlNet on MLX, against `tests/golden/controlnet_sdxl`.
//!
//! **An SDXL ControlNet is `addition_embed_type: "text_time"`** and is
//! conditioned on the pooled embedding and the six time ids, exactly as the
//! SDXL UNet is. That is what this file exists to check, and the failure it
//! guards against would otherwise be silent: without the micro-conditioning a
//! ControlNet still emits nine corrections of exactly the right shapes,
//! computed at a timestep embedding that means something else, and those get
//! added into a UNet that *was* conditioned — which reads as a weak ControlNet
//! rather than a bug.
//!
//! It is not silent here, because `conditioned_temb` refuses the mismatch in
//! both directions. The test asserts that guard rather than the failure,
//! and separately shows the conditioning is load-bearing: zeroing the pooled
//! embedding moves the corrections by 1.664.
//!
//! Compared correction by correction, for the same reason the SD 1.5 reference
//! is: a ControlNet has no image of its own, so those ten tensors *are* its
//! whole observable behaviour, and the index of the first bad one localises the
//! fault.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_controlnet_sdxl -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::{controlnet, UNetConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

/// `golden_controlnet_sdxl.rs`'s bounds, and its reason: these activations are
/// far from order-1, so a purely absolute tolerance would ask for more
/// significant digits than f32 has.
const RTOL: f32 = 1e-3;
const TOL: f32 = 1e-3;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/controlnet_sdxl")
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let (refs, w) = (
        golden().join("reference.safetensors"),
        golden().join("controlnet.safetensors"),
    );
    if !refs.exists() || !w.exists() {
        return None;
    }
    Some((
        load_safetensors(&refs).expect("reference"),
        load_safetensors(&w).expect("weights"),
    ))
}

/// Worst violation of `|a - b| <= atol + rtol * |b|`. `got` is NHWC.
fn excess(got: &Array, want: &Array, s: &Stream, what: &str) -> f32 {
    let g = got
        .transpose(&[0, 3, 1, 2], s)
        .expect("NHWC -> NCHW")
        .to_vec_f32(s)
        .expect("got");
    let w = want.to_vec_f32(s).expect("want");
    assert_eq!(g.len(), w.len(), "{what}: element count");
    let (mut peak, mut worst, mut exc) = (0.0f32, 0.0f32, 0.0f32);
    for (a, b) in g.iter().zip(&w) {
        let d = (a - b).abs();
        worst = worst.max(d);
        peak = peak.max(b.abs());
        exc = exc.max(d - RTOL * b.abs());
    }
    let exc = exc.max(0.0);
    eprintln!("{what:<8} peak {peak:>9.3}  max_abs {worst:.3e}  excess {exc:.3e}");
    exc
}

fn run(
    refs: &HashMap<String, Array>,
    w: &HashMap<String, Array>,
    s: &Stream,
) -> controlnet::Control {
    let cfg = UNetConfig::sdxl();
    let sample = refs
        .get("sample")
        .expect("sample")
        .transpose(&[0, 2, 3, 1], s)
        .unwrap();
    let hint = refs
        .get("hint")
        .expect("hint")
        .transpose(&[0, 2, 3, 1], s)
        .unwrap();
    let added = Some((
        refs.get("pooled").expect("pooled"),
        refs.get("time_ids").expect("time_ids"),
    ));
    controlnet::forward_with(
        &sample,
        refs.get("timestep").expect("timestep"),
        refs.get("context").expect("context"),
        &hint,
        added,
        1.0,
        &cfg,
        w,
        s,
    )
    .expect("controlnet")
}

#[test]
fn the_sdxl_controlnet_matches_diffusers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no controlnet_sdxl fixture.");
        return;
    };
    let s = Stream::gpu();
    let control = run(&refs, &w, &s);

    // Nine down corrections: SDXL's down stack is three blocks, not four.
    assert_eq!(control.down.len(), 9, "one correction per skip");

    let mut first_bad = None;
    for (i, got) in control.down.iter().enumerate() {
        let key = format!("down_{i:02}");
        let e = excess(got, refs.get(&key).expect(&key), &s, &key);
        if e > TOL && first_bad.is_none() {
            first_bad = Some((i, e));
        }
    }
    let mid = excess(&control.mid, refs.get("mid").expect("mid"), &s, "mid");

    if let Some((i, e)) = first_bad {
        panic!("the first bad correction is down_{i:02} at {e:.3e}, past {TOL:.0e}");
    }
    assert!(mid <= TOL, "the mid correction is {mid:.3e} out");
}

/// **The conditioning is refused when missing, and it matters when present.**
///
/// Two halves, because the first alone would not show that this tensor does any
/// work. The guard is what makes the silent failure impossible: an SDXL
/// ControlNet run without micro-conditioning would emit nine corrections of
/// exactly the right shapes computed at a timestep embedding that means
/// something else, and adding those into a UNet that *was* conditioned reads as
/// a weak ControlNet rather than a bug. So it errors instead — in both
/// directions, since supplying conditioning to a ControlNet with no
/// `add_embedding` is the same mistake mirrored.
#[test]
fn the_micro_conditioning_is_refused_when_missing_and_matters_when_present() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no controlnet_sdxl fixture.");
        return;
    };
    let s = Stream::gpu();
    let sample = refs
        .get("sample")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let hint = refs
        .get("hint")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let (timestep, context) = (refs.get("timestep").unwrap(), refs.get("context").unwrap());
    let go = |added: Option<(&Array, &Array)>, cfg: &UNetConfig| {
        controlnet::forward_with(&sample, timestep, context, &hint, added, 1.0, cfg, &w, &s)
    };

    // An SDXL ControlNet with nothing to condition on.
    assert!(
        go(None, &UNetConfig::sdxl()).is_err(),
        "an SDXL ControlNet must refuse to run without its micro-conditioning"
    );
    // And the mirror: conditioning handed to a ControlNet that has no
    // add_embedding to put it in.
    assert!(
        go(
            Some((refs.get("pooled").unwrap(), refs.get("time_ids").unwrap())),
            &UNetConfig::sd15()
        )
        .is_err(),
        "micro-conditioning supplied to a ControlNet with no add_embedding must fail"
    );

    // And it is load-bearing: zeroing it — the nearest legal thing to omitting
    // it — moves every correction.
    let pooled = refs.get("pooled").unwrap();
    let time_ids = refs.get("time_ids").unwrap();
    let zeros = vec![0.0f32; pooled.shape().iter().product()];
    let zero_pooled = Array::from_slice_f32(&zeros, &pooled.shape()).unwrap();
    let real = go(Some((pooled, time_ids)), &UNetConfig::sdxl()).expect("conditioned");
    let zeroed = go(Some((&zero_pooled, time_ids)), &UNetConfig::sdxl()).expect("zeroed");

    let mut worst = 0.0f32;
    for (a, b) in real.down.iter().zip(&zeroed.down) {
        assert_eq!(a.shape(), b.shape(), "the shapes are identical either way");
        let (x, y) = (a.to_vec_f32(&s).unwrap(), b.to_vec_f32(&s).unwrap());
        worst = worst.max(
            x.iter()
                .zip(&y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max),
        );
    }
    eprintln!("zeroing the pooled embedding moves the corrections by {worst:.3e}");
    assert!(
        worst > 1e-2,
        "the corrections moved by only {worst:.3e}; the pooled embedding is not \
         reaching the timestep embedding"
    );
}

/// Scale 0 contributes exactly nothing, rather than almost nothing.
#[test]
fn scale_zero_contributes_exactly_zero() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no controlnet_sdxl fixture.");
        return;
    };
    let s = Stream::gpu();
    let sample = refs
        .get("sample")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let hint = refs
        .get("hint")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let control = controlnet::forward_with(
        &sample,
        refs.get("timestep").unwrap(),
        refs.get("context").unwrap(),
        &hint,
        Some((refs.get("pooled").unwrap(), refs.get("time_ids").unwrap())),
        0.0,
        &UNetConfig::sdxl(),
        &w,
        &s,
    )
    .expect("controlnet");

    for (i, d) in control.down.iter().enumerate() {
        for v in d.to_vec_f32(&s).unwrap() {
            assert_eq!(v, 0.0, "down_{i:02} is not exactly zero at scale 0");
        }
    }
    for v in control.mid.to_vec_f32(&s).unwrap() {
        assert_eq!(v, 0.0, "the mid correction is not exactly zero at scale 0");
    }
}
