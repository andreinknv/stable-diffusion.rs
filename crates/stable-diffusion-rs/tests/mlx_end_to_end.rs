//! txt2img on MLX, prompt to pixels.
//!
//! Everything below the sampler is already gated against diffusers —
//! `mlx_golden_clip`, `mlx_golden_unet`, `mlx_golden_vae`. This is the first
//! test where MLX does the whole job, and what it checks is that the pieces
//! compose: a real schedule, real guidance, twenty steps, and an image that is
//! an image rather than noise.
//!
//! ```bash
//! cargo test -p stable-diffusion-rs --features mlx --test mlx_end_to_end -- --nocapture
//! ```
//!
//! The schedule is `sd_sample::Schedule` and `sigmas_for_steps`, unchanged and
//! shared with the candle path — they return `Vec<f64>` and touch no tensors,
//! so a divergence there is impossible by construction rather than by test.
#![cfg(feature = "mlx")]

use std::path::{Path, PathBuf};

use sd_models::mlx::{clip, sample, unet_forward, vae};
use sd_sample::{sigmas_for_steps, Schedule};
use sd_tensor::mlx::{load_safetensors, Array, Stream};
use sd_tensor::rng::SeededRng;
use sd_tensor::{Device, Tensor};

/// The SD VAE's convention. TAESD's is 1.0; using the wrong one is a plausible
/// image in wrong colours.
const VAE_SCALE: f32 = 0.18215;
const STEPS: usize = 20;
const CFG_SCALE: f64 = 7.5;
/// `<|startoftext|>` and `<|endoftext|>` in SD 1.5's vocabulary.
const BOS: i32 = 49406;
const EOS: i32 = 49407;

fn golden(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(sub)
}

fn have(paths: &[PathBuf]) -> bool {
    paths.iter().all(|p| p.exists())
}

/// Noise drawn through `SeededRng`, the same generator the candle pipeline
/// uses, then handed to MLX as plain data.
///
/// This is deliberate: it makes the two backends see *identical* draws, so a
/// difference between their images is the models and not the dice.
fn seeded_noise(rng: &mut SeededRng, shape: (usize, usize, usize, usize)) -> Vec<f32> {
    let t: Tensor = rng.randn(shape, &Device::Cpu).expect("randn");
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// NCHW `[1,4,h,w]` values into MLX's NHWC.
fn nchw_to_nhwc(data: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for ci in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[(y * w + x) * c + ci] = data[ci * h * w + y * w + x];
            }
        }
    }
    out
}

#[test]
fn txt2img_produces_an_image() {
    let unet_dir = golden("unet_full");
    let vae_dir = golden("vae_decoder");
    let clip_dir = golden("clip_encoder");
    let needed = [
        unet_dir.join("unet.safetensors"),
        vae_dir.join("vae.safetensors"),
        clip_dir.join("clip.safetensors"),
        clip_dir.join("reference.safetensors"),
    ];
    if !have(&needed) {
        sd_tensor::skip_missing_fixture!(
            "SKIP: txt2img needs the unet_full, vae_decoder and clip_encoder fixtures."
        );
        return;
    }

    let s = Stream::gpu();
    let unet_w = load_safetensors(&needed[0]).expect("unet");
    let vae_w = load_safetensors(&needed[1]).expect("vae");
    let clip_w = load_safetensors(&needed[2]).expect("clip");
    let clip_refs = load_safetensors(&needed[3]).expect("clip reference");

    // The conditional prompt is the fixture's tokenisation; the unconditional
    // one is the empty string, which CLIP pads with EOS.
    let cond_ids = {
        let ids = clip_refs.get("token_ids").expect("token_ids");
        let f = ids.to_f32(&s).unwrap().to_vec_f32(&s).unwrap();
        let v: Vec<i32> = f.iter().map(|&x| x as i32).collect();
        Array::from_slice_i32(&v, &ids.shape()).unwrap()
    };
    let mut uncond = vec![EOS; clip::MAX_POSITION];
    uncond[0] = BOS;
    let uncond_ids = Array::from_slice_i32(&uncond, &[1, clip::MAX_POSITION]).unwrap();

    let cond = clip::text_encoder(&cond_ids, &clip_w, &s).expect("cond");
    let uncond = clip::text_encoder(&uncond_ids, &clip_w, &s).expect("uncond");
    // Unconditional row first, matching the guidance batch the UNet is fed.
    let context = sd_tensor::mlx::concat(&[&uncond, &cond], 0, &s).expect("context");
    assert_eq!(context.shape(), vec![2, 77, 768]);

    // 64x64 latent -> 512x512, SD 1.5's native resolution. Below it the model
    // is out of distribution and the image degrades — `docs/handoff.md` records
    // the same for SDXL. A 32x32 latent here produced recognisable structure in
    // oversaturated blocks, which is that effect and not a porting bug.
    let (lh, lw) = (64usize, 64usize);
    let schedule = Schedule::sd15();
    let sigmas = sigmas_for_steps(&schedule, STEPS);
    assert_eq!(sigmas.len(), STEPS + 1, "n steps need n+1 boundaries");
    assert_eq!(*sigmas.last().unwrap(), 0.0, "the ladder ends at zero");

    let mut rng = SeededRng::new(42);
    let init = seeded_noise(&mut rng, (1, 4, lh, lw));
    // Sampling starts at maximum noise, so the latent is scaled by the first
    // sigma rather than used raw.
    let scaled: Vec<f32> = init.iter().map(|v| v * sigmas[0] as f32).collect();
    let mut latent =
        Array::from_slice_f32(&nchw_to_nhwc(&scaled, 4, lh, lw), &[1, lh, lw, 4]).unwrap();

    let train_sigmas = schedule.sigmas();
    for i in 0..STEPS {
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);

        let latent_in = sample::scale_model_input(&latent, sigma, &s).unwrap();
        // The UNet takes a discrete training timestep, not a continuous sigma.
        let t = train_sigmas
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (*a - sigma)
                    .abs()
                    .partial_cmp(&(*b - sigma).abs())
                    .expect("finite")
            })
            .map(|(i, _)| i as f32)
            .expect("non-empty schedule");
        let timestep = Array::from_slice_f32(&[t, t], &[2]).unwrap();

        let out = unet_forward(&latent_in, &timestep, &context, &unet_w, &s).expect("unet");
        let noise_pred = sample::guidance(&out, CFG_SCALE, &s).expect("cfg");
        let denoised = sample::denoise_epsilon(&latent, &noise_pred, sigma, &s).expect("denoise");

        let step_noise = seeded_noise(&mut rng, (1, 4, lh, lw));
        let step_noise =
            Array::from_slice_f32(&nchw_to_nhwc(&step_noise, 4, lh, lw), &[1, lh, lw, 4]).unwrap();
        latent =
            sample::euler_ancestral_step(&latent, &denoised, sigma, sigma_next, &step_noise, &s)
                .expect("euler step");
    }

    let scaled = latent
        .div(&Array::scalar_f32(VAE_SCALE).unwrap(), &s)
        .unwrap();
    let image = vae::decode(&scaled, &vae_w, &s).expect("decode");
    assert_eq!(image.shape(), vec![1, 512, 512, 3]);

    const W: usize = 512;
    let pixels = image.to_vec_f32(&s).unwrap();

    // The VAE emits roughly [-1, 1]; the pipeline maps that to bytes.
    let bytes: Vec<u8> = pixels
        .iter()
        .map(|&v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();

    // An image, not noise. Two cheap properties that noise fails: the values
    // occupy a real range, and neighbouring pixels correlate.
    let mean = bytes.iter().map(|&b| b as f64).sum::<f64>() / bytes.len() as f64;
    let var = bytes
        .iter()
        .map(|&b| (b as f64 - mean).powi(2))
        .sum::<f64>()
        / bytes.len() as f64;
    eprintln!("image mean {mean:.1}, sd {:.1}", var.sqrt());
    assert!(
        (5.0..250.0).contains(&mean),
        "a saturated or empty image: mean {mean:.1}"
    );
    assert!(var.sqrt() > 3.0, "no contrast: sd {:.1}", var.sqrt());

    // Horizontal neighbour difference. Independent noise averages ~85 for
    // uniform bytes; a photograph is far smoother.
    let mut diff = 0.0f64;
    let mut count = 0usize;
    for y in 0..512 {
        for x in 0..511 {
            let a = bytes[(y * 512 + x) * 3] as f64;
            let b = bytes[(y * 512 + x + 1) * 3] as f64;
            diff += (a - b).abs();
            count += 1;
        }
    }
    let smoothness = diff / count as f64;
    eprintln!("mean |neighbour difference| {smoothness:.2}");
    assert!(
        smoothness < 25.0,
        "neighbouring pixels do not correlate ({smoothness:.2}); this is noise, not an image"
    );

    // CARGO_TARGET_TMPDIR rather than the repository root: a test that leaves
    // an untracked file behind makes every later `git status` noisier.
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("mlx-txt2img.raw");
    std::fs::write(&out, &bytes).expect("writing pixels");
    eprintln!(
        "wrote {} ({} bytes, {W}x{W} RGB8)",
        out.display(),
        bytes.len()
    );
}
