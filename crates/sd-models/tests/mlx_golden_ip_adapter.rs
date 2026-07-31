//! IP-Adapter on MLX, against `tests/golden/ip_adapter`.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_ip_adapter -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::{ip::IpAdapter, unet_forward_with, UNetConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

/// The UNet's own bound — see `golden_unet.rs`.
const ATOL: f32 = 1e-4;

fn max_abs(got_nhwc: &Array, want_nchw: &Array, s: &Stream, what: &str) -> f32 {
    let got = got_nhwc
        .transpose(&[0, 3, 1, 2], s)
        .expect("NHWC -> NCHW")
        .to_vec_f32(s)
        .expect("mlx");
    let want = want_nchw.to_vec_f32(s).expect("reference");
    assert_eq!(got.len(), want.len(), "{what}: element count");
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{what:<16} max_abs {worst:.3e}   atol {ATOL:.0e}");
    worst
}

type Weights = std::collections::HashMap<String, Array>;

/// The reference, the adapter, and the base UNet.
fn fixtures() -> Option<(Weights, Weights, Weights)> {
    let g = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden");
    let refs = g.join("ip_adapter/reference.safetensors");
    let adapter = g.join("ip_adapter/ip-adapter_sd15.safetensors");
    let unet = g.join("unet_full/unet.safetensors");
    if !refs.exists() || !adapter.exists() || !unet.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: needs the ip_adapter and unet_full fixtures.");
        return None;
    }
    Some((
        load_safetensors(&refs).expect("reference"),
        load_safetensors(&adapter).expect("adapter"),
        load_safetensors(&unet).expect("unet"),
    ))
}

#[test]
fn the_adapter_matches_diffusers() {
    let Some((refs, adapter_w, unet_w)) = fixtures() else {
        return;
    };
    let s = Stream::gpu();
    let cfg = UNetConfig::sd15();

    let x = refs
        .get("sample")
        .expect("sample")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let t = refs.get("timestep").expect("timestep");
    let text = refs.get("text").expect("text");

    // The fixture supplies the tokens already through `image_proj`, so this
    // exercises the decoupled attention rather than the projector.
    let tokens = refs.get("image_tokens").expect("image_tokens");
    let dims = tokens.shape();
    let tokens = tokens
        .reshape(&[dims[0], dims[2], dims[3]], &s)
        .expect("[b, tokens, dim]");

    let ip = IpAdapter::new(&adapter_w, tokens, 1.0);
    let got =
        unet_forward_with(&x, t, text, None, None, Some(&ip), None, &cfg, &unet_w, &s).unwrap();
    let worst = max_abs(&got, refs.get("output").unwrap(), &s, "ip output");
    assert!(
        worst <= ATOL,
        "the IP-Adapter is {worst:.3e} from diffusers"
    );
}

/// **A strength of 0 must reproduce the unadapted image, not merely approach
/// it.** The fixture carries that run separately, so it is checked rather than
/// assumed.
#[test]
fn a_zero_scale_reproduces_the_unadapted_run() {
    let Some((refs, adapter_w, unet_w)) = fixtures() else {
        return;
    };
    let s = Stream::gpu();
    let cfg = UNetConfig::sd15();

    let x = refs
        .get("sample")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let t = refs.get("timestep").unwrap();
    let text = refs.get("text").unwrap();
    let dims = refs.get("image_tokens").unwrap().shape();
    let tokens = refs
        .get("image_tokens")
        .unwrap()
        .reshape(&[dims[0], dims[2], dims[3]], &s)
        .unwrap();

    let ip = IpAdapter::new(&adapter_w, tokens, 0.0);
    let got =
        unet_forward_with(&x, t, text, None, None, Some(&ip), None, &cfg, &unet_w, &s).unwrap();
    let worst = max_abs(&got, refs.get("output_scale0").unwrap(), &s, "ip scale 0");
    assert!(worst <= ATOL, "at scale 0 the adapter is {worst:.3e}");

    // And it must equal a run with no adapter at all, bit for bit.
    let plain = unet_forward_with(&x, t, text, None, None, None, None, &cfg, &unet_w, &s).unwrap();
    assert_eq!(
        got.to_vec_f32(&s).unwrap(),
        plain.to_vec_f32(&s).unwrap(),
        "scale 0 must be identical to no adapter, not merely close"
    );
}
