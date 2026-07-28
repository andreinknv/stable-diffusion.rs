//! One forward per architecture, on the GPU.
//!
//! **The gap this fills is structural.** Every golden test in this workspace
//! runs on CPU, so a dtype or op that exists on one backend only is invisible
//! to all of them — GLIGEN shipped a `to_dtype(F64)` that works on CPU and
//! fails at *load* on Metal, which is the default device, and none of the 337
//! tests could have caught it.
//!
//! So this asserts almost nothing about the numbers: only that each
//! architecture loads and produces finite values of the right shape on the
//! GPU. Correctness is the golden suite's job. This is the smoke alarm.
//!
//! Deliberately small — 128 px, one step. The point is to touch every kernel
//! each architecture uses, not to make an image, and a big run would make this
//! too slow to keep in the default test pass.

#![cfg(feature = "metal")]

use std::path::PathBuf;

use stable_diffusion_rs::pipeline::{Txt2ImgConfig, Txt2ImgPipeline};
use stable_diffusion_rs::tensor::{Device, Tensor};

/// Model directories to try, by the name they are linked under. Missing ones
/// skip: this must stay green on a machine that has not downloaded them.
const DIFFUSERS_LAYOUT: [&str; 4] = ["sd15", "sd21", "sdxl", "gligen"];

/// Models that cannot be driven from text alone, and the test that covers
/// each instead. Listed so the coverage check below can tell "not exercised"
/// from "exercised elsewhere".
const NON_TXT2IMG: [&str; 3] = ["ip2p", "unclip", "unclip-t2i"];

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn gpu() -> Option<Device> {
    match Device::new_metal(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("SKIP: no Metal device ({e})");
            None
        }
    }
}

/// Whether an error is the memory guard declining rather than a real fault.
///
/// The distinction matters: "this machine is busy" and "this model is broken
/// on the GPU" are different answers, and conflating them makes the smoke
/// test noisy enough to be ignored — which is worse than not having it.
fn refused_for_memory(e: &stable_diffusion_rs::pipeline::PipelineError) -> bool {
    e.is_memory_refusal()
}

fn tiny(seed: u64) -> Txt2ImgConfig {
    Txt2ImgConfig {
        prompt: "a crab".into(),
        negative_prompt: String::new(),
        width: 128,
        height: 128,
        steps: 1,
        cfg_scale: 7.5,
        seed,
        sampler: Default::default(),
        frames: 1,
        cache_threshold: 0.0,
        cancel: None,
    }
}

/// Finite, right shape, and not uniformly zero — a dead kernel returning zeros
/// would otherwise pass every other check here.
fn assert_plausible(image: &Tensor, label: &str) {
    let dims = image.dims().to_vec();
    assert_eq!(
        dims.len(),
        4,
        "{label}: expected [b, c, h, w], got {dims:?}"
    );
    assert_eq!(dims[1], 3, "{label}: expected 3 channels, got {dims:?}");

    let v = image.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "{label}: output contains NaN or infinity"
    );
    let spread = {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for x in &v {
            lo = lo.min(*x);
            hi = hi.max(*x);
        }
        hi - lo
    };
    assert!(
        spread > 1e-3,
        "{label}: output is flat (spread {spread:.2e})"
    );
}

#[test]
fn every_diffusers_layout_model_runs_on_the_gpu() {
    let Some(dev) = gpu() else { return };
    let mut ran = 0;
    for (i, name) in DIFFUSERS_LAYOUT.iter().enumerate() {
        let dir = models_dir().join(name);
        if !dir.join("unet").exists() {
            sd_tensor::skip_missing_fixture!("SKIP {name}: not present");
            continue;
        }
        let pipeline = match Txt2ImgPipeline::load(&dir, &dev) {
            Ok(p) => p,
            Err(e) if refused_for_memory(&e) => {
                // Not a failure. The guard declining on a busy machine says
                // nothing about whether this model works on the GPU, and a
                // smoke test that goes red when something else is running
                // teaches people to ignore it.
                eprintln!("  {name}: SKIP, not enough free memory right now");
                continue;
            }
            Err(e) => panic!("{name} failed to load on Metal: {e}"),
        };
        let image = match pipeline.run(&tiny(i as u64)) {
            Ok(image) => image,
            Err(e) if refused_for_memory(&e) => {
                eprintln!("  {name}: SKIP, not enough free memory to run");
                continue;
            }
            Err(e) => panic!("{name} failed to run on Metal: {e}"),
        };
        assert_plausible(&image, name);
        eprintln!("  {name}: ok");
        ran += 1;
    }
    eprintln!(
        "{ran} of {} models exercised on the GPU",
        DIFFUSERS_LAYOUT.len()
    );
}

#[test]
fn the_gligen_grounding_path_runs_on_the_gpu() {
    // Its own test because grounding takes a *different* route through the
    // UNet — the fusers, and the scalar gate reads that were the original bug.
    let Some(dev) = gpu() else { return };
    let dir = models_dir().join("gligen");
    if !dir.join("unet").exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no gligen model");
        return;
    }
    use stable_diffusion_rs::pipeline::{GroundedBox, GroundingConfig};
    let pipeline = match Txt2ImgPipeline::load(&dir, &dev) {
        Ok(p) => p,
        Err(e) if refused_for_memory(&e) => {
            eprintln!("SKIP: not enough free memory for gligen right now");
            return;
        }
        Err(e) => panic!("gligen failed to load on Metal: {e}"),
    };
    let image = pipeline
        .run_grounded(&GroundingConfig {
            base: tiny(9),
            boxes: vec![GroundedBox {
                bbox: [0.1, 0.1, 0.5, 0.5],
                phrase: "a crab".into(),
            }],
            grounding_fraction: 0.5,
        })
        .expect("grounded run on Metal");
    assert_plausible(&image, "gligen grounded");
}

#[test]
fn the_quantised_path_runs_on_the_gpu() {
    // Quantised kernels are a separate code path from dense ones, and the
    // Metal bug that produced a flat orange Flux image lived in exactly this
    // one. Uses whichever GGUF checkpoint is linked for the golden tests.
    let Some(dev) = gpu() else { return };
    let flux = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/flux");
    if !flux.join("flux-schnell-q4_k_s.gguf").exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no quantised Flux checkpoint");
        return;
    }
    let paths = stable_diffusion_rs::pipeline::paths_in(&flux);
    let cfg = stable_diffusion_rs::models::flux::FluxConfig::mini();
    let (d, s) = stable_diffusion_rs::loader::flux_block_counts(&paths.transformer)
        .expect("reading block counts");
    let guidance =
        stable_diffusion_rs::loader::flux_has_guidance(&paths.transformer).expect("guidance flag");
    let cfg = stable_diffusion_rs::models::flux::FluxConfig {
        depth: d,
        depth_single_blocks: s,
        guidance_embed: guidance,
        ..cfg
    };
    let placement = stable_diffusion_rs::pipeline::Placement::on(&dev);
    let pipe = match stable_diffusion_rs::pipeline::FluxPipeline::load_with_placement(
        &paths, &cfg, &placement,
    ) {
        Ok(p) => p,
        Err(e) if refused_for_memory(&e) => {
            eprintln!("SKIP: not enough free memory for quantised Flux right now");
            return;
        }
        Err(e) => panic!("quantised Flux failed to load on Metal: {e}"),
    };
    let image = pipe
        .run(&stable_diffusion_rs::pipeline::FluxConfigRun {
            prompt: "a crab".into(),
            width: 128,
            height: 128,
            steps: 1,
            guidance: 3.5,
            seed: 1,
        })
        .expect("quantised Flux runs on Metal");
    assert_plausible(&image, "flux quantised");
}

#[test]
fn the_unclip_path_runs_on_the_gpu() {
    // Its own test because unCLIP reaches the GPU through two paths nothing
    // else here does at once: a ViT-H vision tower, and a projection added
    // into every timestep embedding. It also cannot be driven from text.
    let Some(dev) = gpu() else { return };
    let dir = models_dir().join("unclip");
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/crab-512-dpmpp2m-seed42.png");
    if !dir.join("unet").exists() || !source.exists() {
        eprintln!("SKIP: no unclip model or source image");
        return;
    }
    let pipeline = match Txt2ImgPipeline::load(&dir, &dev) {
        Ok(p) => p,
        Err(e) if refused_for_memory(&e) => {
            eprintln!("SKIP: not enough free memory for unclip right now");
            return;
        }
        Err(e) => panic!("unclip failed to load on Metal: {e}"),
    };
    assert!(pipeline.is_unclip(), "the checkpoint was not detected");

    // Text alone must be refused with a message that names the fix. Without
    // this the run would succeed on a zero image embedding — the guidance
    // batch's unconditional row — and return a plausible wrong image.
    let refused = pipeline.run(&tiny(11));
    assert!(
        refused.is_err(),
        "text-only on an unCLIP UNet should be refused"
    );

    let image = pipeline
        .run_unclip(&unclip_cfg(11, Some(source.clone())))
        .expect("unclip run on Metal");
    assert_plausible(&image, "unclip");

    // More than a smoke check, and deliberately so: nothing else covers the
    // step between "the UNet handles class labels" — which the golden suite
    // verifies — and "the pipeline builds them from the reference image". If
    // the embedding were dropped, zeroed or computed once and cached, every
    // assertion above would still pass and every run would return the same
    // picture whatever it was shown.
    let other =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/hires-on-512to1024.png");
    if !other.exists() {
        eprintln!("SKIP the reference-changes-the-image check: no second image");
        return;
    }
    let elsewhere = pipeline
        .run_unclip(&unclip_cfg(11, Some(other)))
        .expect("second unclip run on Metal");
    let a = image.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = elsewhere.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_ne!(a, b, "a different reference image changed nothing");
}

#[test]
fn the_unclip_prior_path_runs_on_the_gpu() {
    // The other half of unCLIP, and its own checkpoint: `-t2i-l`, whose image
    // side is 768-wide where the image-variation model's is 1024. It reaches
    // the GPU through a transformer nothing else here uses — 20 self-attention
    // blocks over 81 tokens, with an additive mask.
    let Some(dev) = gpu() else { return };
    let dir = models_dir().join("unclip-t2i");
    if !dir.join("prior").exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no unclip-t2i model");
        return;
    }
    let pipeline = match Txt2ImgPipeline::load(&dir, &dev) {
        Ok(p) => p,
        Err(e) if refused_for_memory(&e) => {
            eprintln!("SKIP: not enough free memory for unclip-t2i right now");
            return;
        }
        Err(e) => panic!("unclip-t2i failed to load on Metal: {e}"),
    };

    // Without the prior attached, a run with no reference image must say so
    // rather than producing something from a zero embedding.
    let refused = pipeline.run_unclip(&unclip_cfg(3, None));
    assert!(
        refused.is_err(),
        "a prompt-only unCLIP run without a prior should be refused"
    );

    let pipeline = match pipeline.with_prior(&dir) {
        Ok(p) => p,
        Err(e) if refused_for_memory(&e) => {
            eprintln!("SKIP: not enough free memory for the prior right now");
            return;
        }
        Err(e) => panic!("attaching the prior on Metal: {e}"),
    };
    assert!(pipeline.has_prior());

    let mut cfg = unclip_cfg(3, None);
    cfg.base.prompt = "a crab".into();
    // Two steps rather than 25: this is a smoke test, and the prior's
    // correctness is the golden suite's job.
    cfg.prior_steps = 2;
    let image = pipeline.run_unclip(&cfg).expect("prior run on Metal");
    assert_plausible(&image, "unclip prior");
}

fn unclip_cfg(
    seed: u64,
    init_image: Option<PathBuf>,
) -> stable_diffusion_rs::pipeline::UnclipConfig {
    stable_diffusion_rs::pipeline::UnclipConfig {
        base: tiny(seed),
        init_image,
        prior_steps: 25,
        prior_guidance: 4.0,
        noise_level: 0,
    }
}

/// Kept last: a note for whoever adds an architecture.
///
/// Add it to [`DIFFUSERS_LAYOUT`] if it loads through `Txt2ImgPipeline`, or
/// give it its own test if it takes a different route. The cost of forgetting
/// is a backend-specific failure that ships.
#[test]
fn the_smoke_list_covers_what_the_repo_links() {
    let dir = models_dir();
    if !dir.exists() {
        eprintln!("SKIP: no models directory");
        return;
    }
    let linked: Vec<String> = std::fs::read_dir(&dir)
        .expect("reading models/")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("unet").exists())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    for name in &linked {
        assert!(
            DIFFUSERS_LAYOUT.contains(&name.as_str()) || NON_TXT2IMG.contains(&name.as_str()),
            "models/{name} is linked but not in the GPU smoke list — add it, \
             or the next backend-specific bug there will ship"
        );
    }
}

#[test]
fn the_instruct_path_runs_on_the_gpu() {
    // Its own test because an InstructPix2Pix UNet takes 8 input channels and
    // cannot be driven from text alone — plain txt2img on it is now a clear
    // error rather than a convolution shape mismatch, which is what this
    // smoke test surfaced.
    let Some(dev) = gpu() else { return };
    let dir = models_dir().join("ip2p");
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/crab-512-dpmpp2m-seed42.png");
    if !dir.join("unet").exists() || !source.exists() {
        eprintln!("SKIP: no ip2p model or source image");
        return;
    }
    use stable_diffusion_rs::pipeline::InstructConfig;
    let pipeline = match Txt2ImgPipeline::load(&dir, &dev) {
        Ok(p) => p,
        Err(e) if refused_for_memory(&e) => {
            eprintln!("SKIP: not enough free memory for ip2p right now");
            return;
        }
        Err(e) => panic!("ip2p failed to load on Metal: {e}"),
    };

    // And the mistake it guards is worth pinning: text alone must be refused
    // with a message that names the fix.
    let refused = pipeline.run(&tiny(7));
    assert!(
        refused.is_err(),
        "text-only on an 8-channel UNet should be refused"
    );

    let image = pipeline
        .run_instruct(&InstructConfig {
            base: tiny(7),
            init_image: source,
            image_guidance: 1.5,
        })
        .expect("instruct run on Metal");
    assert_plausible(&image, "ip2p");
}
