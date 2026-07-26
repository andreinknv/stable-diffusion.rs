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

use sd_models::vae::{AutoencoderKlDecoder, VaeConfig, TILE_LATENT_EDGE};
use sd_tensor::{testing, DType, Device, Tensor};

fn decoder(dev: &Device) -> Option<AutoencoderKlDecoder> {
    let w = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/vae_decoder/vae.safetensors");
    if !w.exists() {
        eprintln!("SKIP: no VAE weights; run xtask/golden/dump_reference.py vae");
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
    // 96 latent = 768px: bigger than one tile, small enough that the
    // whole-image reference is still affordable on CPU.
    let z = sd_tensor::rng::SeededRng::new(1)
        .randn((1, 4, 96, 96), &dev)
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
