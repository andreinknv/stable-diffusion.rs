//! The MLX pipeline through its public API, against a real checkpoint.
//!
//! `mlx_end_to_end` and `mlx_img2img` hand-assemble the sampling loop, which
//! proves the models compose and proves nothing about what a caller gets. This
//! goes through `MlxPipeline` — load a model directory, call `txt2img`, get
//! bytes — which is the surface that has to work before candle can go.
//!
//! Needs a real SD 1.5 checkpoint:
//!
//! ```bash
//! SD_TEST_MODEL_DIR=$(pwd)/models/sd15 \
//!   cargo test -p stable-diffusion-rs --features mlx --test mlx_pipeline -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use stable_diffusion_rs::mlx::MlxPipeline;
use stable_diffusion_rs::pipeline::{SamplerKind, Strength, Txt2ImgConfig};
use stable_diffusion_rs::tensor::mlx::Array;

/// Few steps: this checks that the pipeline runs and produces an image, not
/// that twenty steps look better than four. The numerical quality of every
/// model underneath is already gated against diffusers.
const STEPS: usize = 4;

fn model_dir() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("SD_TEST_MODEL_DIR").ok()?);
    p.is_dir().then_some(p)
}

fn config() -> Txt2ImgConfig {
    Txt2ImgConfig {
        prompt: "a photograph of an astronaut riding a horse on mars".into(),
        negative_prompt: String::new(),
        width: 256,
        height: 256,
        steps: STEPS,
        cfg_scale: 7.5,
        seed: 42,
        sampler: SamplerKind::EulerAncestral,
        ..Default::default()
    }
}

/// An image, not noise. Two cheap properties that noise fails: the values
/// occupy a real range, and neighbouring pixels correlate.
fn looks_like_an_image(w: usize, h: usize, bytes: &[u8]) -> (f64, f64, f64) {
    assert_eq!(bytes.len(), w * h * 3, "RGB bytes");
    let mean = bytes.iter().map(|&b| b as f64).sum::<f64>() / bytes.len() as f64;
    let sd = (bytes
        .iter()
        .map(|&b| (b as f64 - mean).powi(2))
        .sum::<f64>()
        / bytes.len() as f64)
        .sqrt();
    // Mean absolute step between horizontally adjacent pixels. Noise is ~85 on
    // a uniform byte range; an image is far smoother.
    let mut steps = 0.0f64;
    let mut n = 0usize;
    for y in 0..h {
        for x in 1..w {
            for c in 0..3 {
                let i = (y * w + x) * 3 + c;
                steps += (bytes[i] as f64 - bytes[i - 3] as f64).abs();
                n += 1;
            }
        }
    }
    (mean, sd, steps / n as f64)
}

#[test]
fn txt2img_through_the_public_api() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR to an SD 1.5 checkpoint.");
        return;
    };
    let pipe = MlxPipeline::load(&dir).expect("loading the pipeline");
    let (w, h, bytes) = pipe.txt2img(&config()).expect("txt2img");

    assert_eq!((w, h), (256, 256));
    let (mean, sd, step) = looks_like_an_image(w, h, &bytes);
    eprintln!("txt2img  mean {mean:.1}  sd {sd:.1}  neighbour step {step:.1}");
    assert!(
        (5.0..250.0).contains(&mean),
        "a mean of {mean:.1} is a blank image, not a picture"
    );
    assert!(sd > 10.0, "a standard deviation of {sd:.1} is a flat field");
    assert!(
        step < 40.0,
        "neighbouring pixels differ by {step:.1} on average; that is noise, not an image"
    );
}

/// **The same seed must give the same image, byte for byte.**
///
/// The whole reason noise is drawn through `SeededRng` on the CPU rather than
/// on the GPU. Reproducibility is a promise the CLI makes.
#[test]
fn the_same_seed_gives_the_same_image() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR to an SD 1.5 checkpoint.");
        return;
    };
    let pipe = MlxPipeline::load(&dir).expect("loading the pipeline");
    let cfg = config();
    let (_, _, a) = pipe.txt2img(&cfg).expect("first");
    let (_, _, b) = pipe.txt2img(&cfg).expect("second");
    assert_eq!(a, b, "the same seed produced two different images");

    // And a different seed must not.
    let other = Txt2ImgConfig {
        seed: 43,
        ..cfg.clone()
    };
    let (_, _, c) = pipe.txt2img(&other).expect("third");
    assert_ne!(
        a, c,
        "two seeds produced the same image; the seed is ignored"
    );
}

/// The deterministic sampler runs and gives a different image from the
/// ancestral one — they are different algorithms, not different spellings.
#[test]
fn the_samplers_are_not_the_same_algorithm() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR to an SD 1.5 checkpoint.");
        return;
    };
    let pipe = MlxPipeline::load(&dir).expect("loading the pipeline");
    let euler = config();
    let dpm = Txt2ImgConfig {
        sampler: SamplerKind::DpmPlusPlus2M,
        ..euler.clone()
    };

    let (w, h, a) = pipe.txt2img(&euler).expect("euler");
    let (_, _, b) = pipe.txt2img(&dpm).expect("dpm");
    assert_ne!(a, b, "the two samplers produced identical images");

    // Both must still be images.
    for (name, bytes) in [("euler_a", &a), ("dpmpp2m", &b)] {
        let (mean, sd, step) = looks_like_an_image(w, h, bytes);
        eprintln!("{name:<8} mean {mean:.1}  sd {sd:.1}  neighbour step {step:.1}");
        assert!((5.0..250.0).contains(&mean), "{name}: mean {mean:.1}");
        assert!(step < 40.0, "{name}: neighbour step {step:.1}");
    }
}

/// A size that does not divide into latent cells is refused, rather than
/// silently rounded.
#[test]
fn a_size_that_is_not_a_multiple_of_eight_is_refused() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR to an SD 1.5 checkpoint.");
        return;
    };
    let pipe = MlxPipeline::load(&dir).expect("loading the pipeline");
    let cfg = Txt2ImgConfig {
        width: 250,
        ..config()
    };
    assert!(pipe.txt2img(&cfg).is_err(), "250 is not a multiple of 8");
}

/// img2img at strength 0 returns the input through the VAE and nothing else.
#[test]
fn img2img_at_strength_zero_is_the_round_trip() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR to an SD 1.5 checkpoint.");
        return;
    };
    let pipe = MlxPipeline::load(&dir).expect("loading the pipeline");
    let cfg = config();

    // Generate one image, then feed it back. Using the pipeline's own output
    // means the source is on the VAE's manifold, which is what makes a round
    // trip meaningful — `mlx_img2img` records why a `randn` source is not.
    let (w, h, bytes) = pipe.txt2img(&cfg).expect("txt2img");
    let px: Vec<f32> = bytes.iter().map(|&b| b as f32 / 127.5 - 1.0).collect();
    let image = Array::from_slice_f32(&px, &[1, h, w, 3]).unwrap();

    let (rw, rh, round) = pipe
        .img2img(&cfg, &image, Strength::new(0.0))
        .expect("img2img");
    assert_eq!((rw, rh), (w, h));

    let drift = bytes
        .iter()
        .zip(&round)
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
        .max()
        .unwrap_or(0);
    let mean_drift = bytes
        .iter()
        .zip(&round)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / bytes.len() as f64;
    eprintln!("strength 0 round trip: mean {mean_drift:.2}/255, worst {drift}/255");
    // The VAE is lossy by design; what this rules out is a denoise having run.
    assert!(
        mean_drift < 12.0,
        "strength 0 moved the image by {mean_drift:.2}/255 on average; steps ran that \
         should not have"
    );
}

/// Higher strength departs further from the source. The ordering is the whole
/// meaning of the parameter.
#[test]
fn img2img_strength_orders_the_departure() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR to an SD 1.5 checkpoint.");
        return;
    };
    let pipe = MlxPipeline::load(&dir).expect("loading the pipeline");
    let cfg = config();
    let (w, h, source) = pipe.txt2img(&cfg).expect("txt2img");
    let px: Vec<f32> = source.iter().map(|&b| b as f32 / 127.5 - 1.0).collect();
    let image = Array::from_slice_f32(&px, &[1, h, w, 3]).unwrap();

    let distance = |out: &[u8]| -> f64 {
        source
            .iter()
            .zip(out)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .sum::<f64>()
            / source.len() as f64
    };

    let (_, _, gentle) = pipe
        .img2img(&cfg, &image, Strength::new(0.25))
        .expect("gentle");
    let (_, _, heavy) = pipe
        .img2img(&cfg, &image, Strength::new(0.95))
        .expect("heavy");
    let (near, far) = (distance(&gentle), distance(&heavy));
    eprintln!("departure: 0.25 -> {near:.2}/255, 0.95 -> {far:.2}/255");
    assert!(
        near < far,
        "strength 0.25 departed {near:.2} and 0.95 departed {far:.2}; the start index is \
         not being honoured"
    );
}

/// Inpainting leaves the region outside the mask alone.
#[test]
fn inpaint_bounds_the_edit_to_the_mask() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR to an SD 1.5 checkpoint.");
        return;
    };
    let pipe = MlxPipeline::load(&dir).expect("loading the pipeline");
    let cfg = config();
    let (w, h, source) = pipe.txt2img(&cfg).expect("txt2img");
    let px: Vec<f32> = source.iter().map(|&b| b as f32 / 127.5 - 1.0).collect();
    let image = Array::from_slice_f32(&px, &[1, h, w, 3]).unwrap();

    // White (writeable) in the top-left quadrant only.
    let mut mask = vec![0f32; h * w];
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            mask[y * w + x] = 1.0;
        }
    }
    let mask = Array::from_slice_f32(&mask, &[1, h, w, 1]).unwrap();

    let (_, _, painted) = pipe
        .inpaint(&cfg, &image, Strength::new(0.95), &mask)
        .expect("inpaint");

    let (mut inside, mut outside, mut n_in, mut n_out) = (0.0f64, 0.0f64, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let i = (y * w + x) * 3 + c;
                let d = (source[i] as f64 - painted[i] as f64).abs();
                if y < h / 2 && x < w / 2 {
                    inside += d;
                    n_in += 1;
                } else {
                    outside += d;
                    n_out += 1;
                }
            }
        }
    }
    let (inside, outside) = (inside / n_in as f64, outside / n_out as f64);
    eprintln!("inpaint: inside {inside:.2}/255, outside {outside:.2}/255");
    assert!(
        inside > outside * 2.0,
        "the edit moved the outside ({outside:.2}) nearly as much as the inside \
         ({inside:.2}); the mask is not bounding the run"
    );
}
