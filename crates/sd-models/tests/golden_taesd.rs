//! Golden verification for TAESD.
//!
//! Compared against `diffusers.AutoencoderTiny` **through its public
//! `decode`/`encode`**, not just the convolution stack. TAESD's failure mode is
//! not a wrong convolution — the architecture is trivially simple — it is a
//! wrong *convention*: the SD VAE's `/ 0.18215` applied where TAESD wants none,
//! or a missing `tanh(x/3)*3`, or a missing `[0,1] -> [-1,1]`. Each of those
//! produces a plausible image and no error, and each is only visible if the
//! test covers the wrapper rather than the layers.

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::vae::{TinyAutoencoder, TinyDecoder, TinyEncoder};
use sd_tensor::nn::{VarBuilder, VarMap};
use sd_tensor::{testing, DType, Device, Tensor};

/// TAESD is a distilled model of modest dynamic range — its output is an image
/// in `[-1, 1]` — so a plain absolute bound is meaningful here in a way it is
/// not for the UNet's mid-block activations.
const TOL: f64 = 1e-4;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/taesd")
}

fn refs() -> Option<HashMap<String, Tensor>> {
    let path = golden_dir().join("reference.safetensors");
    if !path.exists() {
        eprintln!(
            "SKIP: no reference data.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py taesd --output tests/golden\n"
        );
        return None;
    }
    Some(sd_tensor::safetensors::load(&path, &Device::Cpu).expect("loading reference"))
}

fn real_taesd(dev: &Device) -> Option<TinyAutoencoder> {
    let path = golden_dir().join("taesd.safetensors");
    if !path.exists() {
        eprintln!("SKIP: no taesd.safetensors");
        return None;
    }
    let vb = sd_loader::safetensors_var_builder(&[&path], DType::F32, dev).expect("loading TAESD");
    Some(TinyAutoencoder::new(vb).expect("building TAESD"))
}

// -- structural -----------------------------------------------------------

#[test]
fn the_decoder_upsamples_by_exactly_eight() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let dec = TinyDecoder::new(4, 3, vb).expect("builds");

    let latent = Tensor::zeros((1, 4, 8, 8), DType::F32, &dev).unwrap();
    let out = dec.decode(&latent).expect("decode");
    assert_eq!(out.dims(), &[1, 3, 64, 64]);
}

#[test]
fn the_encoder_downsamples_by_exactly_eight() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let enc = TinyEncoder::new(3, 4, vb).expect("builds");

    let image = Tensor::zeros((1, 3, 64, 64), DType::F32, &dev).unwrap();
    let out = enc.encode(&image).expect("encode");
    assert_eq!(out.dims(), &[1, 4, 8, 8]);
}

#[test]
fn an_extreme_latent_is_soft_clamped_rather_than_blowing_up() {
    // The `tanh(x/3)*3` on the way in. Without it a latent that has drifted --
    // which happens at high guidance -- reaches the convolutions at full
    // magnitude.
    //
    // This needs the *real* weights. An earlier version built the decoder from
    // `VarMap::new()` and asserted the same bound; a random fifteen-layer
    // convolution stack amplifies, so it returned 1352 and the failure said
    // nothing about the clamp. diffusers on the same input gives 2.066.
    let dev = Device::Cpu;
    let Some(tae) = real_taesd(&dev) else { return };

    let wild = (Tensor::ones((1, 4, 8, 8), DType::F32, &dev).unwrap() * 1000.0).unwrap();
    let out = tae.decoder.decode(&wild).expect("decode");
    let max = out
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(max.is_finite() && max < 10.0, "output blew up to {max}");
}

// -- golden ---------------------------------------------------------------

#[test]
fn decode_matches_autoencoder_tiny() {
    let dev = Device::Cpu;
    let Some(refs) = refs() else { return };
    let Some(tae) = real_taesd(&dev) else { return };

    let got = tae.decoder.decode(&refs["latent"]).expect("decode");
    let want = &refs["decoded"];
    assert_eq!(got.dims(), want.dims());
    let excess = testing::allclose_excess(&got, want, 0.0).expect("compare");
    assert!(excess <= TOL, "decode: max diff {excess:.3e}");
    println!("decode max diff {excess:.3e}");
}

#[test]
fn encode_matches_autoencoder_tiny() {
    let dev = Device::Cpu;
    let Some(refs) = refs() else { return };
    let Some(tae) = real_taesd(&dev) else { return };

    let got = tae.encoder.encode(&refs["image"]).expect("encode");
    let want = &refs["encoded"];
    assert_eq!(got.dims(), want.dims());
    let excess = testing::allclose_excess(&got, want, 0.0).expect("compare");
    assert!(excess <= TOL, "encode: max diff {excess:.3e}");
    println!("encode max diff {excess:.3e}");
}

/// A smooth gradient with a bright square in it: `[1, 3, 256, 256]` in
/// `[-1, 1]`.
fn structured_image(dev: &Device) -> Tensor {
    let n = 256usize;
    let mut data = vec![0f32; 3 * n * n];
    for y in 0..n {
        for x in 0..n {
            let (fx, fy) = (
                x as f32 / (n - 1) as f32 * 2.0 - 1.0,
                y as f32 / (n - 1) as f32 * 2.0 - 1.0,
            );
            let inside = (64..192).contains(&x) && (64..192).contains(&y);
            for (c, v) in [fx, fy, fx * fy].into_iter().enumerate() {
                data[c * n * n + y * n + x] = if inside { 1.0 } else { v };
            }
        }
    }
    Tensor::from_vec(data, (1, 3, n, n), dev).unwrap()
}

fn correlation(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut cov, mut va, mut vb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(&b) {
        cov += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    cov / (va.sqrt() * vb.sqrt())
}

#[test]
fn a_round_trip_reconstructs_a_structured_image() {
    // What this pins is that the two halves share a latent convention: an
    // encoder scaled differently from the decoder still round-trips to
    // *something*, but not to the input.
    //
    // The image has to be structured. An earlier version used the reference's
    // uniform-noise image and asserted r > 0.3; it returned 0.014 -- and so
    // does diffusers, on the same input, because per-pixel white noise is
    // simply not reconstructable through an 8x bottleneck. The port was right
    // and the test was wrong. On this image diffusers gives 0.997.
    let dev = Device::Cpu;
    let Some(tae) = real_taesd(&dev) else { return };

    let image = structured_image(&dev);
    let latent = tae.encoder.encode(&image).expect("encode");
    let back = tae.decoder.decode(&latent).expect("decode");
    assert_eq!(back.dims(), image.dims());

    let r = correlation(&image, &back);
    assert!(r > 0.99, "round-trip correlation {r:.3}, expected > 0.99");
    println!("round-trip correlation {r:.3}");
}

#[test]
fn white_noise_does_not_survive_the_bottleneck() {
    // The companion to the test above, and the reason it needs a structured
    // image. Recorded as a fact about TAESD rather than left as a trap for
    // whoever next writes a round-trip test: diffusers returns 0.014 here too.
    let dev = Device::Cpu;
    let Some(refs) = refs() else { return };
    let Some(tae) = real_taesd(&dev) else { return };

    let image = &refs["image"]; // uniform noise
    let latent = tae.encoder.encode(image).expect("encode");
    let back = tae.decoder.decode(&latent).expect("decode");
    let r = correlation(image, &back);
    assert!(r.abs() < 0.1, "noise round-tripped at {r:.3}, expected ~0");
}
