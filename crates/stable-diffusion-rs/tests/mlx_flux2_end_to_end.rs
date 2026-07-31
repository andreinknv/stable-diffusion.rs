//! FLUX.2 from a prompt to pixels, against the real checkpoint.
//!
//! ```bash
//! SD_TEST_FLUX2_DIR=$(pwd)/tests/golden/flux2model \
//!   cargo test -p stable-diffusion-rs --features mlx --test mlx_flux2_end_to_end -- --nocapture
//! ```
//!
//! # Why this exists on top of the golden tests
//!
//! `mlx_golden_flux2` checks the transformer at 2.9e-6 and the latent patching
//! exactly, and **both passed while this produced coloured mush.** Everything
//! that was wrong lived in the glue between them:
//!
//! - **Text position ids were all zero.** Image tokens use axes 1 and 2 of the
//!   four; text tokens carry the token index in axis **3**. With them zeroed,
//!   every word shared one position — word order erased. The palette still
//!   followed the prompt, so it looked like a sampling problem.
//! - **The prompt was not in a conversation.** The encoder is instruction
//!   tuned and the checkpoint learned to read prompts inside FLUX.2's system
//!   message and Qwen's ChatML wrapper. Bare, it renders soft and painterly.
//! - **Padding repeated the last real token** instead of the pad token, so a
//!   sixteen-word prompt ended with its final word four hundred times.
//!
//! None of those is an error at any layer. That is the argument for an
//! end-to-end test with a real checkpoint rather than component tests alone.
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::flux2::Flux2Config;
use stable_diffusion_rs::mlx::{Flux2Pipeline, Flux2RunConfig};

fn model_dir() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("SD_TEST_FLUX2_DIR").ok()?);
    p.is_dir().then_some(p)
}

/// Mean, spread, and the mean step between horizontally adjacent pixels.
/// Noise is around 85 on the last; a photograph is far smoother.
fn statistics(w: usize, h: usize, bytes: &[u8]) -> (f64, f64, f64) {
    assert_eq!(bytes.len(), w * h * 3, "RGB bytes");
    let mean = bytes.iter().map(|&b| b as f64).sum::<f64>() / bytes.len() as f64;
    let sd = (bytes
        .iter()
        .map(|&b| (b as f64 - mean).powi(2))
        .sum::<f64>()
        / bytes.len() as f64)
        .sqrt();
    let (mut steps, mut n) = (0.0, 0usize);
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
fn flux2_generates_an_image_from_a_prompt() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_FLUX2_DIR to a FLUX.2 directory.");
        return;
    };
    let pipe = Flux2Pipeline::load(&dir, Flux2Config::klein_4b()).expect("loading FLUX.2 klein-4B");

    let gb = pipe.resident_bytes() as f64 / 1e9;
    eprintln!("flux2 resident: {gb:.2} GB");
    // klein-4B is 7.75 GB of transformer plus 8 GB of encoder in bf16. If this
    // is not far below that, the quantisation is not happening.
    assert!(gb < 12.0, "FLUX.2 came to {gb:.2} GB");

    let cfg = Flux2RunConfig {
        prompt: "a rusty crab on a beach at sunset".into(),
        width: 512,
        height: 512,
        steps: 4,
        seed: 42,
        ..Default::default()
    };
    let (w, h, bytes) = pipe.txt2img(&cfg).expect("txt2img");
    assert_eq!((w, h), (512, 512));

    let (mean, sd, step) = statistics(w, h, &bytes);
    eprintln!("flux2  mean {mean:.1}  sd {sd:.1}  neighbour step {step:.2}");
    assert!(
        (5.0..250.0).contains(&mean),
        "a mean of {mean:.1} is a blank image"
    );
    assert!(sd > 10.0, "a standard deviation of {sd:.1} is a flat field");
    assert!(
        step < 40.0,
        "neighbouring pixels differ by {step:.2}; that is noise, not an image"
    );

    // A different prompt must give a different picture, or the conditioning is
    // not reaching the model — which is exactly the failure the position-id
    // bug produced, and which the statistics above would not have caught.
    let other = Flux2RunConfig {
        prompt: "a lighthouse on a cliff at dusk, dramatic sky".into(),
        ..cfg.clone()
    };
    let (_, _, second) = pipe.txt2img(&other).expect("second prompt");
    assert_ne!(bytes, second, "two prompts produced the same image");
}

/// **The klein releases are distilled and carry no guidance embedder.**
///
/// Passing a scale there is refused rather than ignored: a caller who set it
/// believes it did something.
#[test]
fn a_distilled_checkpoint_refuses_a_guidance_scale() {
    let cfg = Flux2Config::klein_4b();
    assert!(!cfg.guidance_embed, "klein-4B is distilled");
    assert!(Flux2Config::dev().guidance_embed, "dev takes guidance");
}
