//! Property tests for the txt2img pipeline.
//!
//! There is no golden reference here: end-to-end output depends on our own
//! RNG, which deliberately does not match PyTorch. So these pin the
//! properties that must hold regardless — determinism, the sigma-to-timestep
//! mapping, and input validation — and the end-to-end test asserts the image
//! is finite and in range without claiming it depicts anything.
//!
//! A test suite cannot tell you the picture is a crab. Look at it.

use stable_diffusion_rs::pipeline::{sigma_to_timestep, SamplerKind, Txt2ImgConfig};
use stable_diffusion_rs::sample::{sigmas_for_steps, Schedule};
use stable_diffusion_rs::tensor::rng::SeededRng;
use stable_diffusion_rs::tensor::{DType, Device, Tensor};

/// Serialises tests that load a pipeline.
///
/// Each one is about 6 GB resident, and `cargo test` runs the file's tests in
/// parallel — enough of them at once got the binary SIGKILLed by the OS. The
/// pure tests still run concurrently; only the heavy ones queue.
fn heavy() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A poisoned lock here means another test panicked while holding it, which
    // is not a reason to fail every later test as well.
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn config_defaults_are_sane() {
    let cfg = Txt2ImgConfig::default();
    assert_eq!(cfg.width, 512);
    assert_eq!(cfg.height, 512);
    assert_eq!(cfg.steps, 20);
    assert_eq!(cfg.cfg_scale, 7.5);
    assert_eq!(cfg.sampler, SamplerKind::EulerAncestral);
    assert!(cfg.negative_prompt.is_empty());
    // Latents are 1/8 scale, so the defaults must divide by 8.
    assert_eq!(cfg.width % 8, 0);
    assert_eq!(cfg.height % 8, 0);
}

#[test]
fn sigma_to_timestep_is_monotonic() {
    // Larger sigma means more noise means a later training timestep. A
    // non-monotonic mapping would run the UNet at the wrong point in the
    // schedule and show up as noise rather than as an error.
    let schedule = Schedule::sd15();
    let sigmas = sigmas_for_steps(&schedule, 20);

    let timesteps: Vec<f64> = sigmas
        .iter()
        .map(|&s| sigma_to_timestep(&schedule, s))
        .collect();
    for pair in timesteps.windows(2) {
        assert!(
            pair[0] >= pair[1],
            "timesteps must descend with sigma, got {pair:?}"
        );
    }
}

#[test]
fn sigma_to_timestep_maps_max_sigma_near_999() {
    let schedule = Schedule::sd15();
    let train = schedule.sigmas();

    // The largest training sigma is the last index.
    let t = sigma_to_timestep(&schedule, train[train.len() - 1]);
    assert_eq!(t, (train.len() - 1) as f64);

    // And sigma 0 maps to the start of the schedule.
    assert_eq!(sigma_to_timestep(&schedule, 0.0), 0.0);

    // An out-of-range index here is the classic cause of NaN at step 1.
    for &s in &[0.0, 0.5, 14.6, 1e6] {
        let t = sigma_to_timestep(&schedule, s);
        assert!(
            (0.0..train.len() as f64).contains(&t),
            "sigma {s} mapped out of range: {t}"
        );
    }
}

/// The pipeline draws its initial latent from `SeededRng`, scaled by the first
/// sigma. These two tests pin that reproducibility without needing weights.
fn initial_latent(seed: u64) -> Tensor {
    let schedule = Schedule::sd15();
    let sigmas = sigmas_for_steps(&schedule, 20);
    let mut rng = SeededRng::new(seed);
    (rng.randn((1, 4, 64, 64), &Device::Cpu).unwrap() * sigmas[0]).unwrap()
}

#[test]
fn same_seed_gives_identical_latents() {
    let a = initial_latent(42);
    let b = initial_latent(42);
    let c = stable_diffusion_rs::tensor::testing::closeness(&a, &b).unwrap();
    assert_eq!(c.max_abs, 0.0, "same seed must be bit-identical: {c}");
}

#[test]
fn different_seeds_give_different_latents() {
    let a = initial_latent(42);
    let b = initial_latent(43);
    let c = stable_diffusion_rs::tensor::testing::closeness(&a, &b).unwrap();
    assert!(c.max_abs > 0.0, "different seeds must differ");
}

#[test]
fn end_to_end_produces_finite_image_in_range() {
    let _heavy = heavy();
    let Ok(dir) = std::env::var("SD_TEST_MODEL_DIR") else {
        eprintln!(
            "SKIP end_to_end_produces_finite_image_in_range: set SD_TEST_MODEL_DIR \
             to a diffusers-layout SD 1.5 directory to run it."
        );
        return;
    };

    use stable_diffusion_rs::pipeline::Txt2ImgPipeline;
    let dev = Device::Cpu;
    let pipeline =
        Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev).expect("loading pipeline");

    // 256x256 at the default 20 steps: about 30 seconds, and a converged
    // latent rather than noise.
    //
    // The step count is load-bearing for the range assertion below, which is
    // why it is the full default rather than a token 2. Measured max abs by
    // step count: 2 steps gives 2.39, 8 gives 1.51, 20 gives 1.38. The first
    // two fail the [-1.5, 1.5] bound, not because the pipeline is wrong but
    // because the sampler has not finished. An output-range assertion only
    // means anything once it has.
    let cfg = Txt2ImgConfig {
        prompt: "a rusty crab on a beach".to_string(),
        width: 256,
        height: 256,
        steps: 20,
        seed: 42,
        ..Default::default()
    };
    let img = pipeline.run(&cfg).expect("running pipeline");

    assert_eq!(img.dims(), &[1, 3, 256, 256]);
    let values = img.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        values.iter().all(|v| v.is_finite()),
        "image contains NaN or inf"
    );
    let max = values.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(max <= 1.5, "values outside [-1.5, 1.5]: max abs {max}");
    eprintln!("end-to-end ok: {:?}, max abs {max:.4}", img.dims());
}

// -- img2img ---------------------------------------------------------------

#[test]
fn strength_selects_where_in_the_schedule_to_start() {
    let _heavy = heavy();
    use stable_diffusion_rs::pipeline::Strength;

    // 1.0 replaces everything: start at 0 and run all 20 steps, which is the
    // same work txt2img does.
    assert_eq!(Strength::new(1.0).get(), 1.0);
    // 0.0 keeps the input: nothing left to run.
    assert_eq!(Strength::new(0.0).get(), 0.0);
    // Out of range is a caller error, not a mode — clamp rather than wrap or
    // panic, since a negative strength has no sensible meaning.
    assert_eq!(Strength::new(-1.0).get(), 0.0);
    assert_eq!(Strength::new(7.0).get(), 1.0);
    // The documented default.
    assert_eq!(Strength::default().get(), 0.75);
}

#[test]
fn strength_maps_to_the_number_of_steps_actually_run() {
    use stable_diffusion_rs::pipeline::Strength;

    // This mapping *is* the feature: strength is only meaningful as "how many
    // of the steps get replaced", and an off-by-one here shows up as an image
    // that is subtly too close to, or too far from, the input.
    let steps = 20;
    assert_eq!(Strength::new(1.0).start_index(steps), 0, "full run");
    assert_eq!(
        Strength::new(0.0).start_index(steps),
        steps,
        "nothing to run"
    );
    assert_eq!(Strength::new(0.75).start_index(steps), 5, "15 of 20 steps");
    assert_eq!(Strength::new(0.5).start_index(steps), 10);

    // Monotonic: more strength never means less work.
    let mut prev = usize::MAX;
    for i in 0..=10 {
        let idx = Strength::new(i as f64 / 10.0).start_index(steps);
        assert!(idx <= prev, "start index must not increase with strength");
        prev = idx;
    }

    // Never runs past the end of the ladder, whatever the step count.
    for steps in [1usize, 3, 20, 50] {
        for s in [0.0, 0.01, 0.5, 0.99, 1.0] {
            assert!(Strength::new(s).start_index(steps) <= steps);
        }
    }
}

#[test]
fn img2img_round_trips_an_image_through_the_encoder() {
    let _heavy = heavy();
    let Ok(dir) = std::env::var("SD_TEST_MODEL_DIR") else {
        eprintln!("SKIP img2img_round_trips_an_image_through_the_encoder: set SD_TEST_MODEL_DIR.");
        return;
    };
    let Ok(init) = std::env::var("SD_TEST_INIT_IMAGE") else {
        eprintln!("SKIP: set SD_TEST_INIT_IMAGE to a source image.");
        return;
    };

    use stable_diffusion_rs::pipeline::{Img2ImgConfig, Strength, Txt2ImgPipeline};
    let dev = Device::Cpu;
    let pipeline =
        Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev).expect("loading pipeline");

    let cfg = Img2ImgConfig {
        base: Txt2ImgConfig {
            prompt: "a watercolour painting of a crab".to_string(),
            width: 256,
            height: 256,
            steps: 12,
            seed: 42,
            ..Default::default()
        },
        init_image: std::path::PathBuf::from(&init),
        strength: Strength::new(0.6),
    };
    let img = pipeline.run_img2img(&cfg).expect("running img2img");

    assert_eq!(img.dims(), &[1, 3, 256, 256]);
    let values = img.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(values.iter().all(|v| v.is_finite()), "img2img produced NaN");
    let max = values.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(max <= 1.5, "values outside [-1.5, 1.5]: max abs {max}");
    eprintln!("img2img ok: {:?}, max abs {max:.4}", img.dims());
}

// -- SDXL ------------------------------------------------------------------

#[test]
fn sdxl_end_to_end_produces_finite_image_in_range() {
    let _heavy = heavy();
    let Ok(dir) = std::env::var("SD_TEST_SDXL_DIR") else {
        eprintln!(
            "SKIP sdxl_end_to_end_produces_finite_image_in_range: set SD_TEST_SDXL_DIR \
             to a diffusers-layout SDXL directory to run it."
        );
        return;
    };

    use stable_diffusion_rs::pipeline::SdxlPipeline;
    let dev = Device::Cpu;
    let pipeline = SdxlPipeline::load(std::path::Path::new(&dir), &dev).expect("loading SDXL");

    // 512 rather than SDXL's native 1024: this checks the plumbing, and the
    // CPU cost scales with the square. The picture will be poor at this size,
    // which is a property of the model, not of the pipeline.
    let cfg = Txt2ImgConfig {
        prompt: "a rusty crab on a beach".to_string(),
        width: 512,
        height: 512,
        steps: 4,
        seed: 42,
        ..Default::default()
    };
    let img = pipeline.run(&cfg).expect("running SDXL");

    assert_eq!(img.dims(), &[1, 3, 512, 512]);
    let values = img.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(values.iter().all(|v| v.is_finite()), "SDXL produced NaN");
    eprintln!("sdxl ok: {:?}", img.dims());
}

// -- latent in/out, and determinism ---------------------------------------

/// A small config for the tests below. 128px and 2 steps: these assert
/// *equality between runs*, which does not need a converged image.
#[cfg(test)]
fn tiny_config(seed: u64) -> stable_diffusion_rs::pipeline::Txt2ImgConfig {
    stable_diffusion_rs::pipeline::Txt2ImgConfig {
        prompt: "a crab".into(),
        negative_prompt: String::new(),
        width: 128,
        height: 128,
        steps: 2,
        cfg_scale: 7.5,
        seed,
        sampler: Default::default(),
        cancel: None,
    }
}

#[test]
fn supplying_the_initial_latent_reproduces_the_seeded_run_exactly() {
    let _heavy = heavy();
    // The contract that makes `initial_latent` useful: it must be the *same*
    // latent the seeded path would have drawn, so a caller can take it,
    // perturb it, and know that an unperturbed round trip changes nothing.
    // Bit-identical, not close — anything less would hide a divergence in the
    // sampler's own noise sequence, which is the part that is easy to get
    // wrong when the initial draw is skipped.
    let Ok(dir) = std::env::var("SD_TEST_MODEL_DIR") else {
        eprintln!("SKIP supplying_the_initial_latent_reproduces_the_seeded_run_exactly");
        return;
    };
    use stable_diffusion_rs::pipeline::Txt2ImgPipeline;
    let dev = Device::Cpu;
    let pipeline =
        Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev).expect("loading pipeline");
    let cfg = tiny_config(7);

    let seeded = pipeline.run(&cfg).expect("seeded run");
    let start = pipeline.initial_latent(&cfg).expect("initial latent");
    let (explicit, _) = pipeline
        .run_with_latent(&cfg, Some(&start), &mut |_| {})
        .expect("explicit run");

    let a = seeded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = explicit.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a, b, "initial_latent must reproduce the seeded run exactly");
}

#[test]
fn a_different_initial_latent_gives_a_different_image() {
    let _heavy = heavy();
    // The other half: if the supplied latent were quietly ignored, the test
    // above would still pass. This one fails in that case.
    let Ok(dir) = std::env::var("SD_TEST_MODEL_DIR") else {
        eprintln!("SKIP a_different_initial_latent_gives_a_different_image");
        return;
    };
    use stable_diffusion_rs::pipeline::Txt2ImgPipeline;
    let dev = Device::Cpu;
    let pipeline =
        Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev).expect("loading pipeline");
    let cfg = tiny_config(7);

    let mine = pipeline
        .initial_latent(&tiny_config(1234))
        .expect("other latent");
    let (image, _) = pipeline
        .run_with_latent(&cfg, Some(&mine), &mut |_| {})
        .expect("explicit run");
    let seeded = pipeline.run(&cfg).expect("seeded run");

    let a = seeded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = image.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_ne!(a, b, "the supplied latent was ignored");
}

#[test]
fn generation_is_deterministic_across_runs() {
    let _heavy = heavy();
    // Same seed and parameters must give byte-identical output. Callers record
    // generation parameters as provenance and promise their users the asset can
    // be reproduced later, so this is a guarantee rather than an accident —
    // and an accident is what it would be without a test, since summation
    // order is the sort of thing that changes under a dependency bump.
    let Ok(dir) = std::env::var("SD_TEST_MODEL_DIR") else {
        eprintln!("SKIP generation_is_deterministic_across_runs");
        return;
    };
    use stable_diffusion_rs::pipeline::Txt2ImgPipeline;
    let dev = Device::Cpu;
    let pipeline =
        Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev).expect("loading pipeline");
    let cfg = tiny_config(99);

    let first = pipeline.run(&cfg).expect("first run");
    let second = pipeline.run(&cfg).expect("second run");
    let a = first.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = second.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a, b, "two runs of the same seed diverged");

    // And across pipeline instances, which is the case that would catch a
    // load path that is not bit-stable.
    //
    // The first pipeline is dropped before the second loads. Holding both is
    // 12 GB, and with the rest of this file running in parallel the memory
    // guard refuses the second — correctly. An earlier version of this test
    // did exactly that and looked like a determinism failure, which it was
    // not.
    drop(pipeline);
    let reloaded =
        Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev).expect("reloading pipeline");
    let third = reloaded.run(&cfg).expect("third run");
    let c = third.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a, c, "a reloaded pipeline diverged from the first");
}

#[test]
fn control_maps_must_match_the_attached_controlnets() {
    let _heavy = heavy();
    // With several ControlNets bound, a caller passing the wrong number of
    // maps would otherwise get them zipped to the shorter list — every shape
    // still valid, the wrong ControlNet reading the wrong hint, and a
    // plausible image out. Refused instead.
    let (Ok(dir), Ok(cnet)) = (
        std::env::var("SD_TEST_MODEL_DIR"),
        std::env::var("SD_TEST_CONTROLNET"),
    ) else {
        eprintln!("SKIP control_maps_must_match_the_attached_controlnets");
        return;
    };
    use stable_diffusion_rs::pipeline::{Control, ControlConfig, Txt2ImgPipeline};
    let dev = Device::Cpu;
    let pipeline = Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev)
        .expect("loading pipeline")
        .with_controlnet(std::path::Path::new(&cnet))
        .expect("attaching a ControlNet");
    assert_eq!(pipeline.controlnet_count(), 1);

    let base = tiny_config(1);
    let hint = Tensor::zeros((1, 3, base.height, base.width), DType::F32, &dev).unwrap();

    // Two maps for one ControlNet.
    let too_many = ControlConfig {
        base: base.clone(),
        controls: vec![
            Control {
                hint: hint.clone(),
                scale: 1.0,
            },
            Control {
                hint: hint.clone(),
                scale: 1.0,
            },
        ],
    };
    assert!(
        pipeline.run_control(&too_many).is_err(),
        "two maps for one net"
    );

    // And none at all.
    let none = ControlConfig {
        base: base.clone(),
        controls: Vec::new(),
    };
    assert!(pipeline.run_control(&none).is_err(), "no maps for one net");

    // A latent-resolution map, which is the other easy mistake: control maps
    // are at pixel resolution.
    let latent_sized = ControlConfig {
        base,
        controls: vec![Control {
            hint: Tensor::zeros((1, 3, 16, 16), DType::F32, &dev).unwrap(),
            scale: 1.0,
        }],
    };
    assert!(
        pipeline.run_control(&latent_sized).is_err(),
        "latent-sized map"
    );
}

#[test]
fn one_conditioning_selected_every_step_is_the_ordinary_run() {
    let _heavy = heavy();
    // The equivalence that makes `run_conditioned` safe to reach for: a
    // single-entry set with a constant selector must reproduce the plain run
    // bit-identically. Anything less would mean the conditioned path takes a
    // different code route, and the two would drift apart silently.
    let Ok(dir) = std::env::var("SD_TEST_MODEL_DIR") else {
        eprintln!("SKIP one_conditioning_selected_every_step_is_the_ordinary_run");
        return;
    };
    use stable_diffusion_rs::pipeline::Txt2ImgPipeline;
    let dev = Device::Cpu;
    let pipeline =
        Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev).expect("loading pipeline");
    let cfg = tiny_config(3);

    let plain = pipeline.run(&cfg).expect("plain run");
    let cond = pipeline
        .encode_conditioning(&cfg.prompt, &cfg.negative_prompt)
        .expect("encode");
    let (conditioned, _) = pipeline
        .run_conditioned(&cfg, &[cond], &mut |_, _| 0, None, &mut |_| {})
        .expect("conditioned run");

    let a = plain.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = conditioned.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a, b, "the conditioned path diverged from the plain one");
}

#[test]
fn gating_the_negative_prompt_to_a_window_changes_the_image() {
    let _heavy = heavy();
    // The technique the hook exists for: a negative applied only during part
    // of the schedule. If the selector were ignored this would be identical to
    // applying it throughout, so this is the test that proves per-step
    // conditioning actually reaches the model.
    let Ok(dir) = std::env::var("SD_TEST_MODEL_DIR") else {
        eprintln!("SKIP gating_the_negative_prompt_to_a_window_changes_the_image");
        return;
    };
    use stable_diffusion_rs::pipeline::Txt2ImgPipeline;
    let dev = Device::Cpu;
    let pipeline =
        Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev).expect("loading pipeline");
    let mut cfg = tiny_config(5);
    cfg.steps = 4;

    let with_negative = pipeline
        .encode_conditioning(&cfg.prompt, "red")
        .expect("encode");
    let without = pipeline
        .encode_conditioning(&cfg.prompt, "")
        .expect("encode");

    // Negative on for the whole run.
    let (always, _) = pipeline
        .run_conditioned(
            &cfg,
            std::slice::from_ref(&with_negative),
            &mut |_, _| 0,
            None,
            &mut |_| {},
        )
        .expect("always");
    // Negative only in the middle of the schedule.
    let (windowed, _) = pipeline
        .run_conditioned(
            &cfg,
            &[without, with_negative],
            &mut |step, total| usize::from(step * 4 > total && step * 4 <= total * 3),
            None,
            &mut |_| {},
        )
        .expect("windowed");

    let a = always.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = windowed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_ne!(a, b, "the per-step selector was ignored");
}

#[test]
fn a_cancelled_run_stops_and_says_where() {
    let _heavy = heavy();
    use stable_diffusion_rs::pipeline::{Cancel, PipelineError, Txt2ImgPipeline};
    let Ok(dir) = std::env::var("SD_TEST_MODEL_DIR") else {
        eprintln!("SKIP a_cancelled_run_stops_and_says_where");
        return;
    };
    let dev = Device::Cpu;
    let pipeline =
        Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev).expect("loading pipeline");
    let mut cfg = tiny_config(11);
    cfg.steps = 6;
    let cancel = Cancel::new();
    cfg.cancel = Some(cancel.clone());

    // Cancel from the progress callback, which is where a GUI would do it.
    let err = pipeline
        .run_with_progress(&cfg, &mut |p| {
            if p.step == 2 {
                cancel.cancel();
            }
        })
        .expect_err("should have been cancelled");

    match err {
        PipelineError::Cancelled { completed, total } => {
            // Checked at the top of the step, so cancelling during step 2
            // stops before step 3 runs.
            assert_eq!(completed, 2);
            assert_eq!(total, 6);
        }
        other => panic!("expected Cancelled, got {other}"),
    }
}

#[test]
fn a_textual_inversion_changes_the_prompt_it_appears_in() {
    let _heavy = heavy();
    // The whole feature in one assertion: a trigger word with an embedding
    // behind it must condition differently from the same word without one.
    // If the splice silently did nothing, the trigger would tokenise as an
    // ordinary word and the two would match.
    let (Ok(dir), Ok(emb)) = (
        std::env::var("SD_TEST_MODEL_DIR"),
        std::env::var("SD_TEST_EMBEDDING"),
    ) else {
        eprintln!("SKIP a_textual_inversion_changes_the_prompt_it_appears_in");
        return;
    };
    use stable_diffusion_rs::pipeline::Txt2ImgPipeline;
    let dev = Device::Cpu;
    let plain = Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev).expect("loading pipeline");
    let stem = std::path::Path::new(&emb)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap()
        .to_string();

    let mut cfg = tiny_config(21);
    cfg.steps = 2;
    cfg.prompt = format!("a painting of a cat in the style of {stem}");

    let without = plain.run(&cfg).expect("without the embedding");

    let with = Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev)
        .expect("loading pipeline")
        .with_embedding(std::path::Path::new(&emb))
        .expect("loading the embedding");
    assert_eq!(with.embedding_names(), vec![stem.as_str()]);
    let with_out = with.run(&cfg).expect("with the embedding");

    let a = without.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = with_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_ne!(a, b, "the embedding was not spliced into the prompt");
}

#[test]
fn a_prompt_without_the_trigger_is_unaffected() {
    let _heavy = heavy();
    // The other half: registering an embedding must not change prompts that
    // do not name it. Without this, the test above would pass even if the
    // splice were overwriting arbitrary positions.
    let (Ok(dir), Ok(emb)) = (
        std::env::var("SD_TEST_MODEL_DIR"),
        std::env::var("SD_TEST_EMBEDDING"),
    ) else {
        eprintln!("SKIP a_prompt_without_the_trigger_is_unaffected");
        return;
    };
    use stable_diffusion_rs::pipeline::Txt2ImgPipeline;
    let dev = Device::Cpu;
    let mut cfg = tiny_config(22);
    cfg.steps = 2;
    cfg.prompt = "a painting of a cat".into();

    let plain = Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev).expect("loading pipeline");
    let without = plain.run(&cfg).expect("plain");
    drop(plain);

    let with = Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev)
        .expect("loading pipeline")
        .with_embedding(std::path::Path::new(&emb))
        .expect("loading the embedding");
    let with_out = with.run(&cfg).expect("with an unused embedding");

    let a = without.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = with_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a, b, "an unused embedding changed the result");
}

#[test]
fn an_embedding_of_the_wrong_width_is_refused() {
    let _heavy = heavy();
    // 1024-wide is SD 2.x's; in an SD 1.5 prompt it would otherwise surface as
    // a shape error from inside the transformer.
    let (Ok(dir), Ok(emb)) = (
        std::env::var("SD_TEST_MODEL_DIR"),
        std::env::var("SD_TEST_EMBEDDING_WRONG_WIDTH"),
    ) else {
        eprintln!("SKIP an_embedding_of_the_wrong_width_is_refused");
        return;
    };
    use stable_diffusion_rs::pipeline::Txt2ImgPipeline;
    let dev = Device::Cpu;
    let stem = std::path::Path::new(&emb)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap()
        .to_string();
    let mut cfg = tiny_config(23);
    cfg.steps = 1;
    cfg.prompt = format!("a cat, {stem}");

    let pipeline = Txt2ImgPipeline::load(std::path::Path::new(&dir), &dev)
        .expect("loading pipeline")
        .with_embedding(std::path::Path::new(&emb))
        .expect("loading is fine; using it is not");
    assert!(pipeline.run(&cfg).is_err(), "wrong width was accepted");
}
