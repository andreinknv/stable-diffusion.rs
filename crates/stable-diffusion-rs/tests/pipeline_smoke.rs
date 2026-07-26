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
use stable_diffusion_rs::tensor::{Device, Tensor};

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
