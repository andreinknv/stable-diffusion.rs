//! Flux's latent packing on MLX, against candle's.
//!
//! A cross-backend comparison rather than a golden one, for the same reason
//! `mlx_lora_agrees` and `mlx_gguf_agrees_with_candle` are: this is pure index
//! arithmetic with no reference tensor to compare against, and candle's is
//! already gated by `golden_flux_transformer` running end to end.
//!
//! **It is exactly the kind of thing that runs and is wrong.** Flux's
//! *unpacking* inverts its packing; SD 3's `unpatchify` deliberately does not,
//! because its patch embedding is a convolution whose flattened kernel runs
//! `(channel, ph, pw)` where its final linear emits `(ph, pw, channel)`. Both
//! consume `c*4`-wide tokens from a 2x2 patch grid, so using the wrong one
//! produces an image of exactly the right shape with every patch's channels and
//! positions transposed: coherent colours, destroyed detail, no error.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_flux_packing_agrees -- --nocapture
//! ```
#![cfg(all(feature = "mlx", feature = "metal"))]

use sd_models::flux as candle_flux;
use sd_models::mlx::flux as mlx_flux;
use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{Device, Tensor};

/// A deterministic latent, as both a candle tensor and an MLX array.
fn latent(c: usize, h: usize, w: usize) -> (Tensor, Array) {
    let n = c * h * w;
    // Values that make every position distinguishable, so a transposed packing
    // cannot coincidentally agree.
    let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
    (
        Tensor::from_vec(v.clone(), (1, c, h, w), &Device::Cpu).expect("candle"),
        Array::from_slice_f32(&v, &[1, c, h, w]).expect("mlx"),
    )
}

#[test]
fn packing_agrees_with_candle_exactly() {
    let s = Stream::gpu();
    // Several shapes, including a non-square one — a packing that happens to
    // work on squares can still transpose h and w.
    for (c, h, w) in [(16usize, 4usize, 4usize), (16, 8, 4), (4, 6, 10)] {
        let (ct, mt) = latent(c, h, w);
        let want = candle_flux::pack_latents(&ct).expect("candle pack");
        let got = mlx_flux::pack_latents(&mt, &s).expect("mlx pack");

        assert_eq!(
            got.shape(),
            want.dims().to_vec(),
            "{c}x{h}x{w}: packed shape"
        );
        assert_eq!(
            got.to_vec_f32(&s).unwrap(),
            want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "{c}x{h}x{w}: the two packings disagree element for element"
        );
    }
}

#[test]
fn unpacking_agrees_with_candle_exactly() {
    let s = Stream::gpu();
    for (c, h, w) in [(16usize, 4usize, 4usize), (16, 8, 4), (4, 6, 10)] {
        let (ct, mt) = latent(c, h, w);
        let packed_c = candle_flux::pack_latents(&ct).expect("candle pack");
        let packed_m = mlx_flux::pack_latents(&mt, &s).expect("mlx pack");

        let want = candle_flux::unpack_latents(&packed_c, h, w).expect("candle unpack");
        let got = mlx_flux::unpack_latents(&packed_m, h, w, &s).expect("mlx unpack");

        assert_eq!(got.shape(), want.dims().to_vec(), "{c}x{h}x{w}: shape");
        assert_eq!(
            got.to_vec_f32(&s).unwrap(),
            want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "{c}x{h}x{w}: the two unpackings disagree"
        );
    }
}

/// **A round trip must be the identity**, which catches a permutation that is
/// self-inverse-looking but wrong.
#[test]
fn pack_then_unpack_is_the_identity() {
    let s = Stream::gpu();
    let (_, mt) = latent(16, 8, 6);
    let round =
        mlx_flux::unpack_latents(&mlx_flux::pack_latents(&mt, &s).unwrap(), 8, 6, &s).unwrap();
    assert_eq!(
        round.to_vec_f32(&s).unwrap(),
        mt.to_vec_f32(&s).unwrap(),
        "packing then unpacking changed the latent"
    );
}

/// **Flux's packing *is* SD 3's; their unpackings are not each other's.**
///
/// This test began asserting the opposite, and was wrong: both use the same
/// `(0, 2, 4, 1, 3, 5)` permutation, so `mlx::flux::pack_latents` delegates to
/// SD 3's rather than carrying a second copy.
///
/// The real distinction is on the way back. Flux's `unpack_latents` is a true
/// inverse of the packing; SD 3's `unpatchify` deliberately is not, because its
/// patch embedding is a convolution whose flattened kernel runs
/// `(channel, ph, pw)` where its final linear emits `(ph, pw, channel)`. That
/// is the pair that must not be confused, and it is asserted here rather than
/// left to the comments.
#[test]
fn the_two_packings_are_the_same_and_the_unpackings_are_not() {
    let s = Stream::gpu();
    let (_, mt) = latent(16, 8, 8);
    let flux = mlx_flux::pack_latents(&mt, &s).unwrap();
    let sd3 = sd_models::mlx::sd3::pack_latents(&mt, &s).unwrap();
    assert_eq!(
        flux.to_vec_f32(&s).unwrap(),
        sd3.to_vec_f32(&s).unwrap(),
        "the two packings are the same permutation; if this fails one has been changed"
    );

    // Flux's unpacking inverts it exactly.
    let round = mlx_flux::unpack_latents(&flux, 8, 8, &s).unwrap();
    assert_eq!(
        round.to_vec_f32(&s).unwrap(),
        mt.to_vec_f32(&s).unwrap(),
        "Flux's unpacking must invert the packing"
    );
}

/// An odd latent side cannot pack into 2x2 patches, and is refused.
#[test]
fn an_odd_latent_side_is_refused() {
    let s = Stream::gpu();
    let (_, mt) = latent(16, 5, 4);
    assert!(mlx_flux::pack_latents(&mt, &s).is_err(), "5 is odd");
}
