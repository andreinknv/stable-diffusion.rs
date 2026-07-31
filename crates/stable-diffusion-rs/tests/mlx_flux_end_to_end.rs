//! Full-size Flux schnell, prompt to pixels, on MLX.
//!
//! **This is what quantised-at-rest was for.** schnell is 11.89B parameters —
//! 47.6 GB dense in f32 — and T5-XXL is another 18.8 GB. Neither fits on a
//! 36 GB machine, let alone both. Held under the default mixed policy the pair
//! is about 13 GB and the run completes.
//!
//! The pieces are assembled explicitly rather than from a `diffusers`
//! directory, because no such directory exists here: the transformer is a GGUF,
//! T5 is a separate GGUF in llama.cpp's naming, and CLIP-L and the VAE come
//! from the golden fixtures. `FluxPipeline::from_gguf` is that path.
//!
//! ```bash
//! cargo test --release -p stable-diffusion-rs --features mlx \
//!   --test mlx_flux_end_to_end -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::flux::FluxConfig;
use stable_diffusion_rs::mlx::{FluxPipeline, FluxRunConfig};

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(name)
}

#[test]
fn schnell_generates_an_image_from_a_prompt() {
    let parts = [
        golden("flux/flux-schnell-q4_k_s.gguf"),
        golden("flux/t5-xxl-q4_k_s.gguf"),
        golden("flux/clip-l.safetensors"),
        golden("flux_vae/vae.safetensors"),
        golden("flux/clip-tokenizer.json"),
        golden("flux/t5-tokenizer.json"),
    ];
    if !parts.iter().all(|p| p.exists()) {
        sd_tensor::skip_missing_fixture!("SKIP: Flux schnell needs its gguf and tokenizers.");
        return;
    }

    let pipe = FluxPipeline::from_gguf(
        &parts[0],
        &parts[1],
        &parts[2],
        &parts[3],
        &parts[4],
        &parts[5],
        FluxConfig::schnell(),
        sd_models::mlx::quantized::DEFAULT_BITS,
        sd_tensor::mlx::Device::Gpu,
    )
    .expect("assembling schnell from gguf");

    let gb = pipe.resident_bytes() as f64 / 1e9;
    eprintln!("schnell + T5 resident: {gb:.2} GB");
    // 36 GB of unified memory, shared with everything else. Dense f32 for the
    // pair is 66 GB — `mlx_gguf_large` asserts the transformer's half alone
    // does not fit.
    assert!(
        gb < 20.0,
        "the pair came to {gb:.2} GB, which is not the saving this exists for"
    );

    // 256px keeps the activations small; the model is the same at any size and
    // this test is about whether it runs, not how it looks at four steps.
    let cfg = FluxRunConfig {
        prompt: "a photograph of an astronaut riding a horse on mars".into(),
        width: 256,
        height: 256,
        steps: 2,
        guidance: 3.5,
        seed: 42,
    };
    let (w, h, bytes) = pipe.txt2img(&cfg).expect("schnell txt2img");
    assert_eq!((w, h), (256, 256));
    assert_eq!(bytes.len(), w * h * 3);

    // An image, not noise: the values occupy a real range and neighbouring
    // pixels correlate.
    let mean = bytes.iter().map(|&b| b as f64).sum::<f64>() / bytes.len() as f64;
    let sd = (bytes
        .iter()
        .map(|&b| (b as f64 - mean).powi(2))
        .sum::<f64>()
        / bytes.len() as f64)
        .sqrt();
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
    let step = steps / n as f64;
    eprintln!("schnell image: mean {mean:.1}, sd {sd:.1}, neighbour step {step:.1}");
    assert!(
        (5.0..250.0).contains(&mean),
        "a mean of {mean:.1} is a blank image"
    );
    assert!(sd > 5.0, "a standard deviation of {sd:.1} is a flat field");
    assert!(
        step < 45.0,
        "neighbouring pixels differ by {step:.1}; that is noise, not an image"
    );
}

// -- SD 3.5 -----------------------------------------------------------------

/// **SD 3.5 medium, prompt to pixels.** The same story as schnell: three text
/// encoders and a transformer that do not fit dense together.
///
/// Assembled from parts because the published directory ships no
/// `text_encoder_3` and no tokenizers, and names its encoders
/// `model.fp16.safetensors`.
#[test]
fn sd35_generates_an_image_from_a_prompt() {
    use stable_diffusion_rs::mlx::{Sd3Pipeline, Sd3RunConfig};

    let parts = [
        golden("sd35/sd35-medium.safetensors"),
        golden("sd35/vae.safetensors"),
        golden("sd35/clip-l.safetensors"),
        golden("sd35/clip-g.safetensors"),
        golden("flux/t5-xxl-q4_k_s.gguf"),
        golden("flux/clip-tokenizer.json"),
        golden("flux/t5-tokenizer.json"),
    ];
    if !parts.iter().all(|p| p.exists()) {
        sd_tensor::skip_missing_fixture!("SKIP: SD 3.5 needs its checkpoint and tokenizers.");
        return;
    }

    let pipe = Sd3Pipeline::from_parts(
        &parts[0],
        &parts[1],
        &parts[2],
        &parts[3],
        &parts[4],
        &parts[5],
        &parts[6],
        sd_models::mlx::quantized::DEFAULT_BITS,
        sd_tensor::mlx::Device::Gpu,
    )
    .expect("assembling SD 3.5 from parts");

    let gb = pipe.resident_bytes() as f64 / 1e9;
    eprintln!("sd35 resident: {gb:.2} GB");
    assert!(gb < 20.0, "SD 3.5 came to {gb:.2} GB");

    let cfg = Sd3RunConfig {
        prompt: "a photograph of an astronaut riding a horse on mars".into(),
        negative_prompt: String::new(),
        width: 256,
        height: 256,
        steps: 2,
        cfg_scale: 4.5,
        seed: 42,
        ..Default::default()
    };
    let (w, h, bytes) = pipe.txt2img(&cfg).expect("sd35 txt2img");
    assert_eq!((w, h), (256, 256));

    let mean = bytes.iter().map(|&b| b as f64).sum::<f64>() / bytes.len() as f64;
    let sd = (bytes
        .iter()
        .map(|&b| (b as f64 - mean).powi(2))
        .sum::<f64>()
        / bytes.len() as f64)
        .sqrt();
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
    let step = steps / n as f64;
    eprintln!("sd35 image: mean {mean:.1}, sd {sd:.1}, neighbour step {step:.1}");
    assert!((5.0..250.0).contains(&mean), "a mean of {mean:.1} is blank");
    assert!(sd > 5.0, "a standard deviation of {sd:.1} is a flat field");
    assert!(step < 45.0, "neighbour step {step:.1} is noise");
}
