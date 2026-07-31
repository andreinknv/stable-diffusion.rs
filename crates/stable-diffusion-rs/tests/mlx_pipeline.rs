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

use stable_diffusion_rs::mlx::{
    FluxPaths, FluxPipeline, GroundedBox, MlxPipeline, Sd3Paths, Sd3Pipeline, SdxlPipeline,
};
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

// -- SDXL -------------------------------------------------------------------

fn sdxl_dir() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("SD_TEST_SDXL_DIR").ok()?);
    p.is_dir().then_some(p)
}

/// SDXL through its own pipeline, at its **native 1024**.
///
/// Not 256 like the SD 1.5 tests above: `docs/handoff.md` records that SDXL
/// below its native resolution is out of distribution and produces mush, so a
/// small run here would fail the "is this an image" check for a reason that is
/// the model's and not the port's.
/// Measured at **24 s** in a release build on this machine, so it is not
/// ignored — an unrun test is not a gate. A debug build is far slower; the
/// project's verification command uses `--release` for exactly this reason.
#[test]
fn sdxl_txt2img_through_the_public_api() {
    let Some(dir) = sdxl_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_SDXL_DIR to an SDXL checkpoint.");
        return;
    };
    let pipe = SdxlPipeline::load(&dir).expect("loading SDXL");
    let cfg = Txt2ImgConfig {
        width: 1024,
        height: 1024,
        steps: 4,
        sampler: SamplerKind::DpmPlusPlus2M,
        ..config()
    };
    let (w, h, bytes) = pipe.txt2img(&cfg).expect("sdxl txt2img");
    assert_eq!((w, h), (1024, 1024));

    let (mean, sd, step) = looks_like_an_image(w, h, &bytes);
    eprintln!("sdxl  mean {mean:.1}  sd {sd:.1}  neighbour step {step:.1}");
    assert!((5.0..250.0).contains(&mean), "mean {mean:.1}");
    assert!(sd > 10.0, "a standard deviation of {sd:.1} is a flat field");
    assert!(step < 40.0, "neighbour step {step:.1} is noise");
}

/// **The conditioning is 2048 wide, and CLIP-L's half comes first.**
///
/// Cheap enough to run always: it loads the towers and encodes one prompt
/// without touching the UNet. The order is invisible to a shape check — 768 +
/// 1280 and 1280 + 768 are both 2048 — so the assembled context is compared
/// against the two halves obtained separately. Each tower is itself gated by
/// `mlx_golden_clip` and `mlx_golden_sdxl_text_encoder`; what this adds is the
/// composition.
#[test]
fn sdxl_conditioning_puts_clip_l_first() {
    let Some(dir) = sdxl_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_SDXL_DIR to an SDXL checkpoint.");
        return;
    };
    let pipe = SdxlPipeline::load(&dir).expect("loading SDXL");
    let s = pipe.stream();
    let (context, pooled) = pipe.encode_for_test("a red apple").expect("encode");
    let (l, g, g_pooled) = pipe.encode_halves("a red apple").expect("halves");

    assert_eq!(context.shape(), vec![1, 77, 2048], "768 + 1280");
    assert_eq!(l.shape(), vec![1, 77, 768], "CLIP-L");
    assert_eq!(g.shape(), vec![1, 77, 1280], "OpenCLIP bigG");
    assert_eq!(pooled.shape(), vec![1, 1280], "the pooled vector is bigG's");
    assert_eq!(
        pooled.to_vec_f32(s).unwrap(),
        g_pooled.to_vec_f32(s).unwrap(),
        "the pooled vector must come from the second tower, not the first"
    );

    let (ctx, lv, gv) = (
        context.to_vec_f32(s).unwrap(),
        l.to_vec_f32(s).unwrap(),
        g.to_vec_f32(s).unwrap(),
    );
    for pos in 0..77 {
        let row = &ctx[pos * 2048..(pos + 1) * 2048];
        assert_eq!(
            &row[..768],
            &lv[pos * 768..(pos + 1) * 768],
            "position {pos}: the first 768 features are not CLIP-L's"
        );
        assert_eq!(
            &row[768..],
            &gv[pos * 1280..(pos + 1) * 1280],
            "position {pos}: the last 1280 features are not bigG's"
        );
    }

    // The two halves are genuinely different tensors, so the check above is
    // not satisfied trivially. CLIP-L's activations are the larger here —
    // `mlx_golden_clip` records its peak at 851 against bigG's 66.
    let peak = |v: &[f32]| v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    eprintln!(
        "sdxl context: CLIP-L peak {:.2}, bigG peak {:.2}",
        peak(&lv),
        peak(&gv)
    );
}

// -- attachments ------------------------------------------------------------

fn controlnet_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("SD_TEST_CONTROLNET").ok()?);
    p.is_file().then_some(p)
}

fn lora_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/lora/lcm-lora-sdv1-5.safetensors")
}

/// **A ControlNet must steer, and a scale of 0 must not.**
///
/// The hint is a hard vertical edge down the middle — structure a Canny
/// ControlNet has something to say about, and which the prompt does not
/// mention, so any effect on the image is the ControlNet's.
#[test]
fn a_controlnet_steers_the_image_and_scale_zero_does_not() {
    let (Some(dir), Some(net)) = (model_dir(), controlnet_path()) else {
        sd_tensor::skip_missing_fixture!(
            "SKIP: needs SD_TEST_MODEL_DIR and SD_TEST_CONTROLNET (a .safetensors file)."
        );
        return;
    };
    let cfg = config();
    let (w, h) = (cfg.width, cfg.height);

    // A white vertical line on black, in [-1, 1].
    let mut px = vec![-1.0f32; h * w * 3];
    for y in 0..h {
        for c in 0..3 {
            px[(y * w + w / 2) * 3 + c] = 1.0;
        }
    }
    let hint = Array::from_slice_f32(&px, &[1, h, w, 3]).unwrap();

    let plain = MlxPipeline::load(&dir).expect("plain");
    let (_, _, without) = plain.txt2img(&cfg).expect("txt2img");

    let mut steered = MlxPipeline::load(&dir).expect("steered");
    steered.attach_controlnet(&net, 1.0).expect("attach");
    assert_eq!(steered.controlnet_count(), 1);
    let (_, _, with) = steered
        .txt2img_controlled(&cfg, Some(&hint))
        .expect("controlled");

    let drift = |a: &[u8], b: &[u8]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .sum::<f64>()
            / a.len() as f64
    };
    let moved = drift(&without, &with);
    eprintln!("controlnet at scale 1 moved the image by {moved:.2}/255");
    assert!(
        moved > 2.0,
        "the ControlNet moved the image by only {moved:.2}/255; it is not being applied"
    );

    // **Scale 0 must be exactly the unsteered run**, not merely close to it.
    let mut off = MlxPipeline::load(&dir).expect("off");
    off.attach_controlnet(&net, 0.0).expect("attach");
    let (_, _, zero) = off.txt2img_controlled(&cfg, Some(&hint)).expect("scale 0");
    assert_eq!(
        zero, without,
        "a ControlNet at scale 0 changed the image; its corrections are not exactly zero"
    );
}

/// A ControlNet with nothing to read is refused, rather than steering toward a
/// blank image.
#[test]
fn a_controlnet_without_a_map_is_refused() {
    let (Some(dir), Some(net)) = (model_dir(), controlnet_path()) else {
        sd_tensor::skip_missing_fixture!("SKIP: needs SD_TEST_MODEL_DIR and SD_TEST_CONTROLNET.");
        return;
    };
    let mut pipe = MlxPipeline::load(&dir).expect("pipeline");
    pipe.attach_controlnet(&net, 1.0).expect("attach");
    assert!(
        pipe.txt2img(&config()).is_err(),
        "a ControlNet with no control map must be refused"
    );
}

/// **A LoRA must map completely, or not at all.**
///
/// A half-applied adapter still renders a plausible image, so partial coverage
/// is an error rather than a warning.
#[test]
fn a_lora_merges_completely_and_changes_the_image() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR.");
        return;
    };
    if !lora_path().exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no LCM LoRA fixture.");
        return;
    }
    let cfg = config();
    let plain = MlxPipeline::load(&dir).expect("plain");
    let (_, _, before) = plain.txt2img(&cfg).expect("txt2img");

    let mut adapted = MlxPipeline::load(&dir).expect("adapted");
    let merged = adapted.attach_lora(&lora_path(), 1.0).expect("attach");
    assert_eq!(merged, 278, "lcm-lora-sdv1-5 corrects 278 layers");
    let (_, _, after) = adapted.txt2img(&cfg).expect("txt2img");
    assert_ne!(before, after, "the LoRA changed nothing");

    // And a multiplier of 0 is bit-identical to no adapter at all.
    let mut zeroed = MlxPipeline::load(&dir).expect("zeroed");
    zeroed.attach_lora(&lora_path(), 0.0).expect("attach");
    let (_, _, none) = zeroed.txt2img(&cfg).expect("txt2img");
    assert_eq!(
        none, before,
        "a LoRA at multiplier 0 changed the image; it must be an exact no-op"
    );
}

/// **GLIGEN puts things where the boxes say.**
///
/// Two runs from the same seed, differing only in where the box is. If the
/// grounding reaches the UNet, the two images differ most in the two box
/// regions and least elsewhere; if it does not, they are identical.
#[test]
fn gligen_boxes_move_the_content() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gligen");
    if !dir.is_dir() {
        sd_tensor::skip_missing_fixture!("SKIP: no GLIGEN checkpoint at models/gligen.");
        return;
    }
    let pipe = MlxPipeline::load(&dir).expect("loading GLIGEN");
    let cfg = config();

    let left = pipe
        .generate(
            &cfg,
            None,
            None,
            &[GroundedBox {
                bbox: [0.05, 0.4, 0.45, 0.95],
                phrase: "a red apple".into(),
            }],
        )
        .expect("left");
    let right = pipe
        .generate(
            &cfg,
            None,
            None,
            &[GroundedBox {
                bbox: [0.55, 0.4, 0.95, 0.95],
                phrase: "a red apple".into(),
            }],
        )
        .expect("right");

    assert_ne!(left.2, right.2, "moving the box changed nothing");

    // And an ungrounded run differs from both.
    let plain = pipe.txt2img(&cfg).expect("plain");
    assert_ne!(plain.2, left.2, "grounding changed nothing");
}

/// An ordinary SD 1.5 UNet has no fuser layers, so boxes are refused rather
/// than dropped.
#[test]
fn grounded_boxes_are_refused_by_a_plain_unet() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR.");
        return;
    };
    let pipe = MlxPipeline::load(&dir).expect("pipeline");
    let err = pipe.generate(
        &config(),
        None,
        None,
        &[GroundedBox {
            bbox: [0.1, 0.1, 0.5, 0.5],
            phrase: "a cat".into(),
        }],
    );
    assert!(err.is_err(), "a plain UNet must refuse grounded boxes");
}

/// **The IP-Adapter conditions on a picture**, and scale 0 does not.
#[test]
fn an_ip_adapter_conditions_on_the_reference_image() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR.");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let adapter = root.join("tests/golden/ip_adapter/ip-adapter_sd15.safetensors");
    let vision = root.join("tests/golden/clip_vision/image_encoder.safetensors");
    if !adapter.exists() || !vision.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no IP-Adapter or image encoder.");
        return;
    }
    let cfg = config();

    // A reference image in CLIP's [0, 1] — a simple colour field is enough to
    // move the output measurably.
    let mut px = vec![0.0f32; 224 * 224 * 3];
    for y in 0..224 {
        for x in 0..224 {
            px[(y * 224 + x) * 3] = 0.9;
            px[(y * 224 + x) * 3 + 1] = 0.2;
            px[(y * 224 + x) * 3 + 2] = 0.1;
        }
    }
    let reference = Array::from_slice_f32(&px, &[1, 224, 224, 3]).unwrap();

    let plain = MlxPipeline::load(&dir).expect("plain");
    let (_, _, without) = plain.txt2img(&cfg).expect("txt2img");

    let mut adapted = MlxPipeline::load(&dir).expect("adapted");
    adapted
        .attach_ip_adapter(&adapter, &vision, 1.0)
        .expect("attach");
    let (_, _, with) = adapted
        .generate(&cfg, None, Some(&reference), &[])
        .expect("adapted run");
    assert_ne!(without, with, "the IP-Adapter changed nothing");

    // Scale 0 must be exactly the unadapted run.
    let mut off = MlxPipeline::load(&dir).expect("off");
    off.attach_ip_adapter(&adapter, &vision, 0.0)
        .expect("attach");
    let (_, _, zero) = off
        .generate(&cfg, None, Some(&reference), &[])
        .expect("scale 0");
    assert_eq!(
        zero, without,
        "an IP-Adapter at scale 0 changed the image; it must contribute exactly nothing"
    );
}

// -- SD 3.5 -----------------------------------------------------------------

/// **SD 3.5's pipeline is written but not end-to-end verified here**, because
/// T5-XXL is not on this machine in safetensors — only as a 4-bit GGUF, which
/// dequantises to 18.8 GB and does not fit beside the rest.
///
/// The candle `Sd3Pipeline` has no end-to-end test either; it is reachable only
/// through the CLI. So what is pinned on both sides is the same thing: the
/// models underneath (`mlx_golden_sd3`, `mlx_golden_t5`, `mlx_golden_flux_vae`
/// for the 16-channel latent) and the layout the pipeline assembles them into.
#[test]
fn sd3_paths_name_every_piece_including_both_t5_shards() {
    let root = PathBuf::from("/models/sd35");
    let paths = Sd3Paths::in_dir(&root);
    assert!(paths
        .transformer
        .ends_with("transformer/diffusion_pytorch_model.safetensors"));
    assert!(paths.clip_l.ends_with("text_encoder/model.safetensors"));
    assert!(paths.clip_g.ends_with("text_encoder_2/model.safetensors"));
    // **Two shards.** T5-XXL does not fit in one file, and a single-file
    // assumption drops half the encoder and surfaces as a missing tensor
    // naming one arbitrary layer.
    assert_eq!(paths.t5.len(), 2, "T5-XXL ships as two shards");
    assert!(paths.t5[0].ends_with("model-00001-of-00002.safetensors"));
    assert!(paths.t5[1].ends_with("model-00002-of-00002.safetensors"));
    // T5's tokenizer is a sentencepiece model, not CLIP's tokenizer.json.
    assert!(paths.t5_tokenizer.ends_with("tokenizer_3/spiece.model"));
}

/// A missing checkpoint is refused by name, not by whichever file is loaded
/// first.
#[test]
fn sd3_refuses_a_directory_that_is_not_there() {
    let err = Sd3Pipeline::load(&PathBuf::from("/nonexistent/sd35"));
    assert!(err.is_err(), "a missing SD 3.5 directory must be refused");
}

/// **Flow matching's ladder is resolution-dependent**, which is the trap that
/// distinguishes it from the k-diffusion schedules elsewhere in this project.
///
/// A ladder computed for a different image size still has the right length and
/// still descends to zero, so only comparing two sizes catches it.
#[test]
fn the_flow_ladder_depends_on_the_image_size() {
    use sd_sample::flow::{flow_sigmas, FlowMatchConfig};
    let cfg = FlowMatchConfig::sd3();
    // 1024 and 512 give 64x64 and 32x32 latents, so 32x32 and 16x16 patches.
    let big = flow_sigmas(&cfg, 28, 32 * 32);
    let small = flow_sigmas(&cfg, 28, 16 * 16);

    assert_eq!(big.len(), 29, "steps + 1");
    assert_eq!(*big.last().unwrap(), 0.0, "the ladder ends at zero");
    assert_eq!(small.len(), big.len(), "the same length either way");
    assert_ne!(
        big, small,
        "the two resolutions gave the same ladder; the sequence length is being ignored"
    );
}

// -- Flux -------------------------------------------------------------------

/// **`guidance_embed` is a property of the checkpoint, not the caller.**
///
/// schnell is not distilled on a guidance scale and rejects one; dev and
/// flux-mini require one. Driven by the config so a setting cannot be silently
/// discarded, nor a required one silently omitted.
#[test]
fn flux_variants_differ_on_whether_guidance_is_distilled() {
    use sd_models::mlx::flux::FluxConfig;
    assert!(
        !FluxConfig::schnell().guidance_embed,
        "schnell is not distilled on a guidance scale"
    );
    assert!(FluxConfig::dev().guidance_embed, "dev is");
    assert!(FluxConfig::mini().guidance_embed, "flux-mini is");
    // And they are not the same model otherwise.
    assert_ne!(
        FluxConfig::schnell().depth,
        FluxConfig::mini().depth,
        "schnell and mini have different depths"
    );
}

/// Flux's paths name every shard, because both the transformer and T5-XXL ship
/// sharded and a single-file assumption drops most of the model.
#[test]
fn flux_paths_read_the_directory_rather_than_guessing_a_shard_count() {
    let paths = FluxPaths::in_dir(&PathBuf::from("/nonexistent/flux"));
    // Nothing on disk, so nothing found — the point is that it does not
    // fabricate a filename it then fails to open with a confusing message.
    assert!(
        paths.transformer.is_empty(),
        "no transformer shards should be invented for a directory that is not there"
    );
    assert!(paths.t5.is_empty());
    // The unsharded pieces are still named.
    assert!(paths
        .vae
        .ends_with("vae/diffusion_pytorch_model.safetensors"));
    assert!(paths.clip.ends_with("text_encoder/model.safetensors"));
    // T5's tokenizer is sentencepiece, CLIP's is a tokenizer.json.
    assert!(paths.t5_tokenizer.ends_with("tokenizer_2/spiece.model"));
    assert!(paths.clip_tokenizer.ends_with("tokenizer/tokenizer.json"));
}

/// A missing checkpoint is refused rather than half-loaded.
#[test]
fn flux_refuses_a_directory_that_is_not_there() {
    use sd_models::mlx::flux::FluxConfig;
    let err = FluxPipeline::load(&PathBuf::from("/nonexistent/flux"), FluxConfig::schnell());
    assert!(err.is_err(), "a missing Flux directory must be refused");
}

/// **Flux's ladder is resolution-dependent too**, and its `mu` shift differs
/// from SD 3's — the two configs are not interchangeable.
#[test]
fn the_flux_and_sd3_flow_configs_are_not_the_same() {
    use sd_sample::flow::{flow_sigmas, FlowMatchConfig};
    let (flux, sd3) = (FlowMatchConfig::flux(), FlowMatchConfig::sd3());
    let a = flow_sigmas(&flux, 4, 64 * 64);
    let b = flow_sigmas(&sd3, 4, 64 * 64);
    assert_eq!(a.len(), b.len());
    assert_ne!(
        a, b,
        "the two flow configurations gave the same ladder; one is being ignored"
    );
}

// -- AnimateDiff ------------------------------------------------------------

fn motion_adapter() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/motion/motion_adapter.safetensors")
}

/// **A clip is frames, not one image repeated.**
///
/// The motion modules attend across the frame axis, so a run with an adapter
/// must produce `frames()` distinct images — and a run without one must
/// produce exactly the same single image it did before, because attaching
/// nothing must cost nothing.
#[test]
fn a_motion_adapter_produces_a_clip_of_distinct_frames() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR.");
        return;
    };
    if !motion_adapter().exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no motion adapter fixture.");
        return;
    }
    let cfg = config();
    let mut pipe = MlxPipeline::load(&dir).expect("pipeline");
    assert_eq!(pipe.frames(), 1, "no adapter means one frame");

    const FRAMES: usize = 3;
    pipe.attach_motion(&motion_adapter(), FRAMES)
        .expect("attach");
    assert_eq!(pipe.frames(), FRAMES);

    let (w, h, bytes) = pipe.txt2img(&cfg).expect("animated txt2img");
    let per = w * h * 3;
    assert_eq!(
        bytes.len(),
        per * FRAMES,
        "a clip is {FRAMES} images back to back"
    );

    // Every frame must be an image, and no two may be identical.
    for f in 0..FRAMES {
        let frame = &bytes[f * per..(f + 1) * per];
        let (mean, sd, step) = looks_like_an_image(w, h, frame);
        eprintln!("frame {f}: mean {mean:.1}  sd {sd:.1}  neighbour step {step:.1}");
        assert!((5.0..250.0).contains(&mean), "frame {f}: mean {mean:.1}");
        assert!(step < 40.0, "frame {f}: neighbour step {step:.1} is noise");
    }
    for f in 1..FRAMES {
        assert_ne!(
            &bytes[..per],
            &bytes[f * per..(f + 1) * per],
            "frame {f} is identical to frame 0; the clip is one image repeated"
        );
    }

    // And consecutive frames must be *more* alike than a frame is to an
    // unrelated image — that is what the motion modules are for.
    let mean_abs = |a: &[u8], b: &[u8]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .sum::<f64>()
            / a.len() as f64
    };
    let neighbouring = mean_abs(&bytes[..per], &bytes[per..2 * per]);
    let unrelated = {
        let plain = MlxPipeline::load(&dir).expect("plain");
        let (_, _, other) = plain
            .txt2img(&Txt2ImgConfig {
                seed: 99,
                ..cfg.clone()
            })
            .expect("unrelated");
        mean_abs(&bytes[..per], &other)
    };
    eprintln!("consecutive frames {neighbouring:.2}/255, unrelated image {unrelated:.2}/255");
    assert!(
        neighbouring < unrelated,
        "consecutive frames differ by {neighbouring:.2} and an unrelated image by \
         {unrelated:.2}; the frames are not a clip"
    );
}

/// A checkpoint with no motion modules is refused, rather than installing
/// nothing and rendering unrelated images.
#[test]
fn an_adapter_without_motion_modules_is_refused() {
    let Some(dir) = model_dir() else {
        sd_tensor::skip_missing_fixture!("SKIP: set SD_TEST_MODEL_DIR.");
        return;
    };
    let mut pipe = MlxPipeline::load(&dir).expect("pipeline");
    // The VAE is a real checkpoint and carries no motion modules.
    let not_an_adapter = dir.join("vae/diffusion_pytorch_model.safetensors");
    assert!(
        pipe.attach_motion(&not_an_adapter, 4).is_err(),
        "a checkpoint with no motion modules must be refused"
    );
    // And a clip of zero frames is a caller error, not a mode.
    assert!(pipe.attach_motion(&motion_adapter(), 0).is_err());
}
