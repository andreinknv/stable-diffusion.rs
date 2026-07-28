//! Tiled VAE decoding.
//!
//! A whole-image decode allocates a conv im2col of `cin * 9` values per
//! output position — 9.66 GB at 1024px, which does not fit in GPU memory on a
//! 36 GiB Mac. Tiling caps that at one tile's worth.
//!
//! The tiled result is deliberately *close to* rather than identical to a
//! whole-image decode: convolutions at a tile edge see padding where they
//! would otherwise see neighbouring pixels. The overlap blend hides that; it
//! does not remove it. So these tests pin two things — that tiling agrees
//! closely with the whole-image decode where both are possible, and that the
//! seams are not visible as discontinuities.

use std::path::PathBuf;

use sd_models::vae::{AutoencoderKlDecoder, AutoencoderKlEncoder, VaeConfig, TILE_LATENT_EDGE};
use sd_tensor::{testing, DType, Device, Tensor};

fn decoder(dev: &Device) -> Option<AutoencoderKlDecoder> {
    let w = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/vae_decoder/vae.safetensors");
    if !w.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no VAE weights; run xtask/golden/dump_reference.py vae"
        );
        return None;
    }
    let vb = sd_loader::safetensors_var_builder(&[&w], DType::F32, dev).expect("weights");
    Some(AutoencoderKlDecoder::new(&VaeConfig::sd15(), vb).expect("decoder"))
}

#[test]
fn a_latent_that_already_fits_is_not_tiled() {
    // Below the tile size there is nothing to gain and a blend would only
    // introduce error, so `decode_tiled` must be exactly `decode`.
    let dev = Device::Cpu;
    let Some(dec) = decoder(&dev) else { return };
    let z = sd_tensor::rng::SeededRng::new(0)
        .randn((1, 4, TILE_LATENT_EDGE, TILE_LATENT_EDGE), &dev)
        .unwrap();

    let whole = dec.decode(&z).expect("decode");
    let tiled = dec.decode_tiled(&z).expect("decode_tiled");
    let c = testing::closeness(&whole, &tiled).expect("comparing");
    assert_eq!(
        c.max_abs, 0.0,
        "small latents must take the untiled path: {c}"
    );
}

#[test]
fn tiling_agrees_closely_with_a_whole_image_decode() {
    let dev = Device::Cpu;
    let Some(dec) = decoder(&dev) else { return };
    // 80 latent = 640px. Two constraints meet here: it must exceed one tile
    // so tiling actually engages, and the whole-image *reference* must itself
    // be permitted — at 88 latent the untiled decode allocates 4.57 GB and the
    // budget refuses it, which is the guard working, not a test failure.
    let z = sd_tensor::rng::SeededRng::new(1)
        .randn((1, 4, 80, 80), &dev)
        .unwrap();

    let whole = dec.decode(&z).expect("whole decode");
    let tiled = dec.decode_tiled(&z).expect("tiled decode");
    assert_eq!(
        whole.dims(),
        tiled.dims(),
        "tiling must not change the shape"
    );

    let c = testing::closeness(&whole, &tiled).expect("comparing");
    eprintln!("tiled vs whole: {c}");
    // Not 1e-4: the edge-padding difference is real and this is an
    // approximation by construction. On a [-1, 1] image a mean of a few
    // thousandths is imperceptible, and the max is dominated by the few
    // pixels nearest a seam.
    assert!(
        c.mean_abs < 0.02,
        "tiled decode drifts too far from the whole-image decode: {c}"
    );
}

#[test]
fn tile_seams_are_not_visible_as_discontinuities() {
    // The failure mode tiling introduces is a hard line at a seam. Compare
    // the gradient across each seam column against the gradient elsewhere: if
    // blending works, the seam is unremarkable.
    let dev = Device::Cpu;
    let Some(dec) = decoder(&dev) else { return };
    // No whole-image reference here, so this is free to exceed the untiled
    // budget — each tile is decoded separately and every one fits.
    let z = sd_tensor::rng::SeededRng::new(2)
        .randn((1, 4, 96, 96), &dev)
        .unwrap();
    let img: Tensor = dec.decode_tiled(&z).expect("tiled decode");

    let v = img.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (_, c, h, w) = img.dims4().unwrap();
    let at = |ch: usize, y: usize, x: usize| v[ch * h * w + y * w + x];

    // Horizontal step between adjacent columns, averaged over rows/channels.
    let col_step = |x: usize| -> f32 {
        let mut acc = 0.0;
        for ch in 0..c {
            for y in 0..h {
                acc += (at(ch, y, x) - at(ch, y, x - 1)).abs();
            }
        }
        acc / (c * h) as f32
    };

    let steps: Vec<f32> = (1..w).map(col_step).collect();
    let mean = steps.iter().sum::<f32>() / steps.len() as f32;
    let worst = steps.iter().cloned().fold(0f32, f32::max);
    eprintln!(
        "column step: mean {mean:.5}, worst {worst:.5}, ratio {:.2}",
        worst / mean
    );

    // A hard seam shows up as one column an order of magnitude above the rest.
    assert!(
        worst < mean * 6.0,
        "a column step of {worst:.4} against a mean of {mean:.4} looks like a visible seam"
    );
}

// -- the encode half -------------------------------------------------------
//
// img2img runs both, so an untiled encode pays the same `cin * 9` im2col
// blowup the decode does — twice in one run. These mirror the decode tests
// and are kept deliberately small: the point is the tiling logic, not scale.

fn encoder(dev: &Device) -> Option<AutoencoderKlEncoder> {
    let w = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/vae_decoder/vae.safetensors");
    if !w.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no VAE weights; run xtask/golden/dump_reference.py vae"
        );
        return None;
    }
    let vb = sd_loader::safetensors_var_builder(&[&w], DType::F32, dev).expect("weights");
    Some(AutoencoderKlEncoder::new(&VaeConfig::sd15(), vb).expect("encoder"))
}

#[test]
fn an_image_that_already_fits_is_not_tiled() {
    let dev = Device::Cpu;
    let Some(enc) = encoder(&dev) else { return };
    // Exactly one tile: 64 latent x 8.
    let px = TILE_LATENT_EDGE * 8;
    let img = sd_tensor::rng::SeededRng::new(3)
        .randn((1, 3, px, px), &dev)
        .unwrap();

    let whole = enc.encode(&img).expect("encode");
    let tiled = enc.encode_tiled(&img).expect("encode_tiled");
    let c = testing::closeness(&whole, &tiled).expect("comparing");
    assert_eq!(
        c.max_abs, 0.0,
        "an image within one tile must not be tiled: {c}"
    );
}

#[test]
fn tiled_encode_agrees_closely_with_a_whole_image_encode() {
    let dev = Device::Cpu;
    let Some(enc) = encoder(&dev) else { return };
    // 576px: past one tile so tiling engages, small enough to stay cheap.
    let img = sd_tensor::rng::SeededRng::new(4)
        .randn((1, 3, 576, 576), &dev)
        .unwrap();

    let whole = enc.encode(&img).expect("whole encode");
    let tiled = enc.encode_tiled(&img).expect("tiled encode");
    assert_eq!(
        whole.dims(),
        tiled.dims(),
        "tiling must not change the latent shape"
    );
    assert_eq!(tiled.dims(), &[1, 4, 72, 72], "576px -> 72 latent");

    let c = testing::closeness(&whole, &tiled).expect("comparing");
    eprintln!("tiled encode vs whole: {c}");
    // Same reasoning as the decode: convolutions at a tile edge see padding
    // where they would otherwise see neighbours, so this is close, not equal.
    // Latents are order-1, so a mean of a few hundredths is small.
    assert!(
        c.mean_abs < 0.05,
        "tiled encode drifts too far from the whole-image encode: {c}"
    );
}

#[test]
fn a_tiled_encode_round_trips_through_a_tiled_decode() {
    // The img2img path end to end, at a size where both halves tile. If the
    // two tilings disagreed about geometry — a stride, a trim, an edge — the
    // round trip would not come back the right shape, let alone the right
    // image.
    let dev = Device::Cpu;
    let (Some(enc), Some(dec)) = (encoder(&dev), decoder(&dev)) else {
        return;
    };
    // A smooth gradient, not noise. A VAE cannot reproduce noise — that is
    // what the latent bottleneck is for — so a round-trip assertion against
    // noise would fail for a correct encoder and prove nothing.
    let (h, w) = (576usize, 576usize);
    let mut data = Vec::with_capacity(3 * h * w);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let fy = y as f32 / h as f32;
                let fx = x as f32 / w as f32;
                // Distinct per channel so a channel swap would show up.
                data.push(match c {
                    0 => fx * 2.0 - 1.0,
                    1 => fy * 2.0 - 1.0,
                    _ => (fx * fy) * 2.0 - 1.0,
                });
            }
        }
    }
    let img = Tensor::from_vec(data, (1, 3, h, w), &dev).expect("gradient");

    let latent = enc.encode_tiled(&img).expect("tiled encode");
    assert_eq!(latent.dims(), &[1, 4, 72, 72]);
    let out = dec.decode_tiled(&latent).expect("tiled decode");
    assert_eq!(out.dims(), img.dims(), "round trip must preserve the shape");

    let c = testing::closeness(&out, &img).expect("comparing");
    eprintln!("tiled round trip vs source: {c}");
    // A VAE round trip is lossy by design; this only asserts the image
    // survives recognisably rather than becoming noise.
    assert!(
        c.mean_abs < 0.25,
        "a tiled round trip should still resemble its input: {c}"
    );
}
