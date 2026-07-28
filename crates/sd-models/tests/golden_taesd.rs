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

/// Decoder tolerance, and encoder tolerance — they differ by 10x, measured.
///
/// `xtask/golden/reference_precision.py taesd` runs each checkpoint against
/// **itself** in f64, same weights and inputs, so neither run has a bug and
/// the gap between them is float32's own noise floor:
///
/// ```text
///              reference f32 vs its own f64      this port vs reference f32
///              encoded        decoded            encoded      decoded
///   taesd      4.574e-6       1.114e-5           1.240e-5     1.860e-5
///   taesdxl    6.257e-5       9.946e-6           7.033e-5     1.609e-5
///   taesd3     2.661e-6       5.528e-6           3.278e-6     7.153e-6
///   taef1      2.156e-4       3.977e-6           1.996e-4     3.934e-6
/// ```
///
/// **`taef1`'s encoder is 80x noisier in f32 than `taesd3`'s despite being the
/// same architecture**, and the port sits *below* that floor — 1.996e-4
/// against 2.156e-4. An earlier version of this test used a flat 1e-4 and
/// failed on exactly that entry, which was the tolerance measuring float32
/// rather than the port. Its `mean_abs` is 8.876e-7, so the number is one
/// outlier element and not a systematic shift.
///
/// So: `ENCODE_TOL` is 4.6x the worst encoder floor, `DECODE_TOL` 9x the worst
/// decoder floor — the same margins the UNet's bound uses. A real porting bug
/// is nowhere near either; the missing decoder ReLU found during this port
/// showed 2.5.
const DECODE_TOL: f64 = 1e-4;
const ENCODE_TOL: f64 = 1e-3;

/// The published checkpoints, and their latent width.
///
/// Same architecture throughout — only the weights and the channel count
/// differ. Each base model needs its own: loading `taesd` for an SDXL run
/// produces a plausible image in the wrong colours rather than an error, and
/// the 4/16 split at least fails loudly.
const CHECKPOINTS: [(&str, usize); 4] = [
    ("taesd", 4),   // SD 1.5, SD 2.x
    ("taesdxl", 4), // SDXL
    ("taesd3", 16), // SD 3.x
    ("taef1", 16),  // Flux
];

fn golden_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(name)
}

fn refs_for(name: &str) -> Option<HashMap<String, Tensor>> {
    let path = golden_dir(name).join("reference.safetensors");
    if !path.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no reference data.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py taesd --model-id madebyollin/{name} \
             --output tests/golden\n"
        );
        return None;
    }
    Some(sd_tensor::safetensors::load(&path, &Device::Cpu).expect("loading reference"))
}

fn refs() -> Option<HashMap<String, Tensor>> {
    refs_for("taesd")
}

fn real_for(dev: &Device, name: &str, channels: usize) -> Option<TinyAutoencoder> {
    let path = golden_dir(name).join("weights.safetensors");
    if !path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no {name} weights");
        return None;
    }
    let vb = sd_loader::safetensors_var_builder(&[&path], DType::F32, dev).expect("loading TAESD");
    Some(TinyAutoencoder::with_channels(channels, vb).expect("building TAESD"))
}

fn real_taesd(dev: &Device) -> Option<TinyAutoencoder> {
    real_for(dev, "taesd", 4)
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
    for (name, channels) in CHECKPOINTS {
        let Some(refs) = refs_for(name) else { continue };
        let Some(tae) = real_for(&dev, name, channels) else {
            continue;
        };

        let got = tae.decoder.decode(&refs["latent"]).expect("decode");
        let want = &refs["decoded"];
        assert_eq!(got.dims(), want.dims());
        let excess = testing::allclose_excess(&got, want, 0.0).expect("compare");
        assert!(excess <= DECODE_TOL, "{name} decode: max diff {excess:.3e}");
        println!("{name} decode max diff {excess:.3e}");
    }
}

#[test]
fn encode_matches_autoencoder_tiny() {
    let dev = Device::Cpu;
    for (name, channels) in CHECKPOINTS {
        let Some(refs) = refs_for(name) else { continue };
        let Some(tae) = real_for(&dev, name, channels) else {
            continue;
        };

        let got = tae.encoder.encode(&refs["image"]).expect("encode");
        let want = &refs["encoded"];
        assert_eq!(got.dims(), want.dims());
        let excess = testing::allclose_excess(&got, want, 0.0).expect("compare");
        assert!(excess <= ENCODE_TOL, "{name} encode: max diff {excess:.3e}");
        println!("{name} encode max diff {excess:.3e}");
    }
}

#[test]
fn the_two_checkpoints_are_genuinely_different_weights() {
    // They share an architecture, so `taesd` loads happily for an SDXL run and
    // produces a plausible image in the wrong colours. This pins that they are
    // not interchangeable, which is why `--taesd` has to be told which is which.
    let dev = Device::Cpu;
    let (Some(a), Some(b)) = (real_for(&dev, "taesd", 4), real_for(&dev, "taesdxl", 4)) else {
        return;
    };
    let Some(refs) = refs_for("taesd") else {
        return;
    };

    let out_a = a.decoder.decode(&refs["latent"]).expect("decode");
    let out_b = b.decoder.decode(&refs["latent"]).expect("decode");
    let excess = testing::allclose_excess(&out_a, &out_b, 0.0).expect("compare");
    assert!(
        excess > 0.1,
        "the two checkpoints decoded the same latent to within {excess:.3e}"
    );
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
