//! Compile-time proof that the MLX surface is what the port claims.
//!
//! The counterpart to `api_contract.rs`, and for the same reason: a list of
//! "these functions exist, use only these" that drifts from reality costs
//! whoever reads it a session guessing at signatures.
//!
//! Every call below is on a tiny array, so this runs in well under a second
//! and needs no fixture. **It proves the shape of the API, not its numbers** —
//! the numbers are what the `mlx_golden_*` files are for. A function that
//! exists here and is wrong there is caught there.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_api_contract
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;

use sd_models::mlx::{
    clip, clip_vision, gguf, gligen, ip, motion, prior, sample, vae, Adapters, Motion, UNetConfig,
    Weights,
};
use sd_tensor::mlx::{concat, Array, Stream};

fn tiny(v: &[f32], shape: &[usize]) -> Array {
    Array::from_slice_f32(v, shape).expect("array")
}

/// The UNet configurations the port ships, and the fields that distinguish
/// them.
///
/// Pinned because each is a claim about a checkpoint's architecture, and a
/// wrong one loads a plausible number of plausible tensors.
#[test]
fn the_unet_configs_say_what_they_are() {
    let sd15 = UNetConfig::sd15();
    let sd2 = UNetConfig::sd2();
    let sdxl = UNetConfig::sdxl();
    let unclip = UNetConfig::unclip();

    assert_eq!(
        sd15.down_has_attention.len(),
        4,
        "SD 1.5 has four down blocks"
    );
    assert_eq!(sdxl.down_has_attention.len(), 3, "SDXL has three");
    assert!(sdxl.addition.is_some(), "SDXL is text_time conditioned");
    assert!(sd15.addition.is_none(), "SD 1.5 is not");
    assert!(
        unclip.class_projection,
        "unCLIP conditions on an image embedding"
    );
    assert!(!sd15.class_projection);
    assert!(sd2.use_linear_projection, "SD 2.x projects linearly");
    assert!(!sd15.use_linear_projection, "SD 1.5 uses 1x1 convolutions");
    assert_eq!(
        sdxl.transformer_layers,
        vec![1, 2, 10],
        "SDXL's transformer depth is not uniform"
    );
}

/// `Adapters` defaults to nothing attached, and every slot is optional.
#[test]
fn adapters_default_to_nothing_attached() {
    let ad = Adapters::default();
    assert!(ad.ip.is_none());
    assert!(ad.objs.is_none());
    assert!(ad.motion.is_none());
    assert!(ad.control.is_none());
}

/// The sampler arithmetic, on numbers small enough to check by hand.
#[test]
fn the_sampler_functions_have_the_documented_shapes() {
    let s = Stream::gpu();
    let latent = tiny(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
    let denoised = tiny(&[0.5, 1.0, 1.5, 2.0], &[1, 2, 2, 1]);
    let noise = tiny(&[0.0; 4], &[1, 2, 2, 1]);

    // The guidance batch is doubled on the batch axis.
    let doubled = sample::scale_model_input(&latent, 1.0, &s).unwrap();
    assert_eq!(doubled.shape(), vec![2, 2, 2, 1]);

    // Guidance halves it again.
    let batched = concat(&[&latent, &denoised], 0, &s).unwrap();
    let guided = sample::guidance(&batched, 7.5, &s).unwrap();
    assert_eq!(guided.shape(), vec![1, 2, 2, 1]);

    // Epsilon prediction, and one step of each sampler.
    let eps = sample::denoise_epsilon(&latent, &denoised, 0.5, &s).unwrap();
    assert_eq!(eps.shape(), latent.shape());
    let stepped = sample::euler_ancestral_step(&latent, &denoised, 1.0, 0.5, &noise, &s).unwrap();
    assert_eq!(stepped.shape(), latent.shape());
    let mut dpm = sample::DpmSolverPlusPlus2M::new();
    let stepped = dpm.step(&latent, &denoised, 1.0, 0.5, &s).unwrap();
    assert_eq!(stepped.shape(), latent.shape());
    dpm.reset();

    // img2img and inpainting.
    let noised = sample::noise_to_sigma(&latent, &noise, 2.0, &s).unwrap();
    assert_eq!(noised.shape(), latent.shape());
    let mask_px = tiny(&vec![1.0; 8 * 8], &[1, 8, 8, 1]);
    let mask = sample::latent_mask(&mask_px, &s).unwrap();
    assert_eq!(
        mask.shape(),
        vec![1, 1, 1, 1],
        "8x8 pixels is one latent cell"
    );
    let restored =
        sample::restore_outside_mask(&latent, &denoised, &mask, &noise, 0.5, &s).unwrap();
    assert_eq!(restored.shape(), latent.shape());
}

/// **`sigma == 0` must not divide by zero.** The last step lands there, and a
/// NaN would propagate silently through the decode into a blank image.
#[test]
fn a_zero_sigma_is_handled_rather_than_divided_by() {
    let s = Stream::gpu();
    let latent = tiny(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
    let denoised = tiny(&[0.5, 1.0, 1.5, 2.0], &[1, 2, 2, 1]);
    let noise = tiny(&[1.0; 4], &[1, 2, 2, 1]);

    let out = sample::euler_ancestral_step(&latent, &denoised, 0.0, 0.0, &noise, &s).unwrap();
    for v in out.to_vec_f32(&s).unwrap() {
        assert!(v.is_finite(), "a zero sigma produced {v}");
    }
    let mut dpm = sample::DpmSolverPlusPlus2M::new();
    let out = dpm.step(&latent, &denoised, 0.0, 0.0, &s).unwrap();
    for v in out.to_vec_f32(&s).unwrap() {
        assert!(v.is_finite(), "a zero sigma produced {v}");
    }
}

/// The VAE configurations, and that `scale`/`unscale` are inverses.
#[test]
fn the_vae_configs_and_their_parameterisation() {
    let s = Stream::gpu();
    for cfg in [
        vae::VaeConfig::sd15(),
        vae::VaeConfig::sdxl(),
        vae::VaeConfig::flux(),
        vae::VaeConfig::sd35(),
    ] {
        let x = tiny(&[-2.0, 0.0, 3.5], &[3]);
        let round = cfg.unscale(&cfg.scale(&x, &s).unwrap(), &s).unwrap();
        for (a, b) in round
            .to_vec_f32(&s)
            .unwrap()
            .iter()
            .zip(x.to_vec_f32(&s).unwrap())
        {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}

/// The CLIP configurations, text and vision.
#[test]
fn the_clip_configs_say_what_they_are() {
    let (sd15, sd2, sdxl2) = (
        clip::ClipConfig::sd15(),
        clip::ClipConfig::sd2(),
        clip::ClipConfig::sdxl_2(),
    );
    assert_eq!(
        sd15.activation,
        clip::Activation::QuickGelu,
        "OpenAI's CLIP"
    );
    assert_eq!(sd2.activation, clip::Activation::Gelu, "OpenCLIP");
    assert_eq!(
        sd2.layers, 23,
        "SD 2.x ships 23, not 24 — see ClipConfig::sd2"
    );
    assert_eq!(sdxl2.hidden, 1280);
    assert!(sdxl2.projection, "bigG carries a text_projection");
    assert!(!sd15.projection, "CLIP-L as SD 1.5 ships it does not");

    let vision = clip_vision::VisionConfig::vit_h_14();
    assert_eq!(vision.grid(), 16, "224 / 14");
    assert_eq!(
        vision.sequence_length(),
        257,
        "256 patches plus the class token"
    );
    assert!(vision.projection, "ViT-H projects 1280 to 1024");
}

/// Pooling takes the **first** highest id.
#[test]
fn pooling_takes_the_first_highest_token() {
    let s = Stream::gpu();
    let ids = Array::from_slice_i32(&[1, 9, 9, 9], &[1, 4]).unwrap();
    let hidden = tiny(&[0.0, 1.0, 2.0, 3.0], &[1, 4, 1]);
    let pooled = clip::pool(&hidden, &ids, &s).unwrap();
    assert_eq!(
        pooled.to_vec_f32(&s).unwrap(),
        vec![1.0],
        "the first 9 is at index 1; the last would give 3"
    );
}

/// The prior's configuration and its scheduler's shared scalars.
#[test]
fn the_prior_config_and_its_shared_scalars() {
    let cfg = prior::PriorConfig::karlo();
    assert_eq!(cfg.inner_dim(), 2048, "32 heads of 64");
    assert_eq!(cfg.sequence_length(), 81, "77 text positions plus four");

    let sched = sd_models::prior::PriorScheduler::new(25);
    let ts = sched.timesteps().to_vec();
    assert_eq!(ts.len(), 25);
    assert_eq!(*ts.last().unwrap(), 0, "the ladder ends at zero");
    // The last step adds no variance; an interior one does.
    assert!(sched.coefficients(0).unwrap().std.is_none());
    assert!(sched.coefficients(ts[0]).unwrap().std.is_some());
}

/// The adapters' own constants and predicates.
#[test]
fn the_adapter_helpers_exist_and_answer() {
    let empty: Weights = HashMap::new();
    assert!(
        !gligen::present(&empty, "anything"),
        "an empty map has no fuser"
    );
    assert!(
        !motion::present(&empty, "anything"),
        "nor any motion module"
    );
    assert_eq!(ip::NUM_TOKENS, 4, "the IP-Adapter's four tokens");
    assert_eq!(ip::sd15_order().len(), 16, "one entry per cross-attention");
    assert_eq!(motion::HEADS, 8);
}

/// `Motion` carries its own weights and the clip length.
#[test]
fn motion_carries_its_frame_count() {
    let w: Weights = HashMap::new();
    let m = Motion {
        weights: &w,
        frames: 16,
    };
    let ad = Adapters {
        motion: Some(&m),
        ..Default::default()
    };
    assert_eq!(ad.motion.map(|m| m.frames), Some(16));
}

/// The GGUF loaders exist and refuse a file that is not there, rather than
/// panicking or returning an empty map.
#[test]
fn the_gguf_loaders_refuse_a_missing_file() {
    let s = Stream::gpu();
    let missing = std::path::Path::new("/nonexistent/nothing.gguf");
    assert!(gguf::load(missing, &s).is_err());
    assert!(gguf::vae(missing, &s).is_err());
    assert!(gguf::unet(missing, &s).is_err());
    assert!(gguf::clip(missing, &s).is_err());
}
