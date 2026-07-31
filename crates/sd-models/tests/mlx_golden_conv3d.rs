//! 3D convolution against PyTorch.
//!
//! The primitive every video VAE needs and no image model does. Verified on
//! its own, before anything is built on it: a convolution with a transposed
//! axis or a padding on the wrong side produces output of exactly the right
//! shape.
//!
//! ```bash
//! .venv/bin/python xtask/golden/dump_reference.py conv3d --output tests/golden
//! cargo test -p sd-models --features mlx --test mlx_golden_conv3d -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::{causal_conv3d_nchw, conv3d_nchw};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

const ATOL: f32 = 1e-4;

fn fixtures() -> Option<HashMap<String, Array>> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/conv3d/reference.safetensors");
    p.exists().then(|| load_safetensors(&p).expect("reference"))
}

fn worst(got: &Array, want: &Array, s: &Stream) -> f32 {
    let (g, e) = (
        got.to_vec_f32(s).expect("got"),
        want.to_vec_f32(s).expect("want"),
    );
    assert_eq!(g.len(), e.len(), "element count");
    g.iter()
        .zip(&e)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn a_plain_3d_convolution_matches_pytorch() {
    let Some(refs) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no conv3d fixture. See the module docs.");
        return;
    };
    let s = Stream::gpu();
    let (x, w, b) = (
        refs.get("input").expect("input"),
        refs.get("weight").expect("weight"),
        refs.get("bias").expect("bias"),
    );

    let got = conv3d_nchw(x, w, Some(b), (1, 1, 1), (1, 1, 1), &s).expect("conv3d");
    let want = refs.get("plain").expect("plain");
    assert_eq!(got.shape(), want.shape(), "shape");
    let d = worst(&got, want, &s);
    eprintln!("conv3d plain    max_abs {d:.3e}");
    assert!(d <= ATOL, "the 3D convolution is {d:.3e} out");
}

/// **Causal in time.** The temporal axis is padded only at the front, so a
/// frame never sees its successors.
#[test]
fn a_causal_3d_convolution_never_looks_forward() {
    let Some(refs) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no conv3d fixture.");
        return;
    };
    let s = Stream::gpu();
    let (x, w, b) = (
        refs.get("input").expect("input"),
        refs.get("weight").expect("weight"),
        refs.get("bias").expect("bias"),
    );

    let got = causal_conv3d_nchw(x, w, Some(b), (1, 1, 1), (1, 1, 1), &s).expect("causal");
    let want = refs.get("causal").expect("causal");
    assert_eq!(got.shape(), want.shape(), "shape");
    let d = worst(&got, want, &s);
    eprintln!("conv3d causal   max_abs {d:.3e}");
    assert!(d <= ATOL, "the causal 3D convolution is {d:.3e} out");

    // And it is genuinely different from the symmetric one — otherwise this
    // test would pass against a non-causal implementation.
    let sym = conv3d_nchw(x, w, Some(b), (1, 1, 1), (1, 1, 1), &s).expect("plain");
    assert!(
        worst(&got, &sym, &s) > 1e-3,
        "causal and symmetric padding gave the same answer; the causality is \
         not being applied"
    );
}

/// The strided case, which is how the VAE downsamples space while leaving
/// time alone.
#[test]
fn a_strided_3d_convolution_matches_pytorch() {
    let Some(refs) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no conv3d fixture.");
        return;
    };
    let s = Stream::gpu();
    let (x, w, b) = (
        refs.get("input").expect("input"),
        refs.get("weight").expect("weight"),
        refs.get("bias").expect("bias"),
    );

    let got = conv3d_nchw(x, w, Some(b), (1, 2, 2), (1, 1, 1), &s).expect("strided");
    let want = refs.get("strided").expect("strided");
    assert_eq!(got.shape(), want.shape(), "shape: time kept, space halved");
    let d = worst(&got, want, &s);
    eprintln!("conv3d strided  max_abs {d:.3e}");
    assert!(d <= ATOL, "the strided 3D convolution is {d:.3e} out");
}
