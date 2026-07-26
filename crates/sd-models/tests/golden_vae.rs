//! Golden-tensor verification for the VAE decoder.
//!
//! Two kinds of test live here:
//!
//! * **Structural** — run always, with random weights. They catch architecture
//!   mistakes (wrong channel counts, wrong upsample factor) without needing a
//!   400 MB download, so CI can enforce them.
//! * **Numerical** — skip unless `tests/golden/vae_decoder/reference.safetensors`
//!   exists. Generate it with `xtask/golden/dump_reference.py`.
//!
//! The structural tests passing while the numerical ones fail is the normal
//! state of a port in progress, and the useful one: it means the shape of the
//! graph is right and the remaining bug is in an op or a constant.

use std::path::PathBuf;

use sd_models::vae::{AutoencoderKlDecoder, Decoder, DecoderConfig, VaeConfig};
use sd_tensor::nn::{VarBuilder, VarMap};
use sd_tensor::{testing, DType, Device, Tensor};

/// A deliberately tiny VAE so structural tests stay fast.
fn tiny_config() -> DecoderConfig {
    DecoderConfig {
        latent_channels: 4,
        out_channels: 3,
        // 3 levels -> 2 upsamples -> 4x spatial scale.
        block_out_channels: vec![32, 64, 64],
        layers_per_block: 1,
        norm_num_groups: 8,
        norm_eps: 1e-6,
    }
}

fn golden_path() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/sd-models; golden data lives at repo root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/vae_decoder/reference.safetensors")
}

#[test]
fn decoder_upsamples_by_two_per_block_and_emits_rgb() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let cfg = tiny_config();

    let decoder = Decoder::new(&cfg, vb).expect("decoder should build from a fresh VarMap");

    let (h, w) = (8usize, 8usize);
    let z = Tensor::zeros((1, cfg.latent_channels, h, w), DType::F32, &dev).unwrap();
    let out = decoder.forward(&z).expect("forward should succeed");

    // 3 blocks, upsample on all but the last => 2^2.
    let scale = 1 << (cfg.block_out_channels.len() - 1);
    assert_eq!(
        out.dims(),
        &[1, cfg.out_channels, h * scale, w * scale],
        "decoder must upsample by 2 per non-final block and emit out_channels"
    );
}

#[test]
fn sd15_config_gives_the_canonical_eight_times_upsample() {
    let cfg = VaeConfig::sd15();
    let scale = 1 << (cfg.block_out_channels.len() - 1);
    assert_eq!(scale, 8, "SD 1.5 VAE must be an 8x decoder");
    assert_eq!(cfg.latent_channels, 4);
    assert_eq!(cfg.norm_num_groups, 32);
    // The VAE uses a tighter epsilon than the torch default; getting this
    // wrong produces a small uniform offset that is easy to misread as noise.
    assert!((cfg.norm_eps - 1e-6).abs() < f64::EPSILON);
}

#[test]
fn sdxl_differs_from_sd15_only_in_latent_scaling() {
    let a = VaeConfig::sd15();
    let b = VaeConfig::sdxl();
    assert_eq!(a.block_out_channels, b.block_out_channels);
    assert_eq!(a.layers_per_block, b.layers_per_block);
    assert_ne!(a.scaling_factor, b.scaling_factor);
}

/// Numerical verification against `diffusers`.
///
/// Skips when reference data is absent — see `xtask/golden/README.md`.
#[test]
fn decoder_matches_diffusers_reference() {
    let path = golden_path();
    if !path.exists() {
        eprintln!(
            "SKIP decoder_matches_diffusers_reference: no reference data.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py vae --output tests/golden\n\
             \nSee xtask/golden/README.md."
        );
        return;
    }

    let dev = Device::Cpu;
    let refs = sd_tensor::safetensors::load(&path, &dev).expect("loading reference tensors");

    let vae_weights = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/vae_decoder/vae.safetensors");
    if !vae_weights.exists() {
        eprintln!(
            "SKIP: reference activations found but VAE weights are missing at {}.\n\
             Copy the SD 1.5 `vae/diffusion_pytorch_model.safetensors` there.",
            vae_weights.display()
        );
        return;
    }

    let vb = sd_loader::safetensors_var_builder(&[&vae_weights], DType::F32, &dev)
        .expect("loading VAE weights");
    let decoder = AutoencoderKlDecoder::new(&VaeConfig::sd15(), vb).expect("building decoder");

    let latent = refs.get("latent").expect("reference has 'latent'");
    let expected = refs.get("image").expect("reference has 'image'");

    let got = decoder.decode_raw(latent).expect("decode_raw");

    // Report the divergence before asserting, so a failure is actionable
    // rather than just red.
    let c = testing::closeness(&got, expected).expect("comparing tensors");
    eprintln!("vae decoder vs diffusers: {c}");

    testing::assert_close(
        &got,
        expected,
        testing::DEFAULT_ATOL,
        "vae decoder final image",
    )
    .unwrap();
}

/// The estimate that refuses an oversized decode now that chunked attention
/// no longer does. Wrong low lets the refusal through; wrong high blocks
/// legitimate work. It was wrong low by 18x until it counted the conv im2col.
#[test]
fn peak_alloc_counts_the_conv_im2col_not_just_activations() {
    let cfg = DecoderConfig::from(&VaeConfig::sd15());

    // Ground truth for the activation term, from
    // xtask/golden/dump_reference.py at a 32x32 latent: the largest captured
    // activation is up_block_2 at (1, 256, 256, 256) — 64 MiB of f32.
    let activation_at_32 = 256u64 * 256 * 256 * 4;
    let peak_at_32 = cfg.peak_alloc_bytes(1, 32, 32, DType::F32).unwrap();
    assert!(
        peak_at_32 > activation_at_32,
        "the peak must exceed the largest activation, since a 3x3 conv over it          allocates nine values per input channel per position: got {peak_at_32}          against an activation of {activation_at_32}"
    );

    // The 1024px decode that exhausts GPU memory: 256 input channels, 3x3,
    // at 1024x1024 is 9.66 GB. The old activation-only estimate said 1.07 GB.
    let at_1024 = cfg.peak_alloc_bytes(1, 128, 128, DType::F32).unwrap();
    assert_eq!(at_1024, 256 * 9 * 1024 * 1024 * 4, "im2col at 1024px");
    assert!(
        at_1024 > sd_tensor::ops::DEFAULT_ATTENTION_BUDGET_BYTES,
        "an untiled 1024 decode must be refused; tiling is what brings it under"
    );

    // One tile — 64 latent, 512px — is ordinary work and must be admitted.
    let one_tile = cfg.peak_alloc_bytes(1, 64, 64, DType::F32).unwrap();
    assert!(
        one_tile <= sd_tensor::ops::DEFAULT_ATTENTION_BUDGET_BYTES,
        "a single tile must fit the budget, or tiled decoding cannot run: {one_tile}"
    );
}

#[test]
fn an_oversized_decode_is_refused_before_it_allocates() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let cfg = DecoderConfig::from(&VaeConfig::sd15());
    let decoder = Decoder::new(&cfg, vb).expect("decoder builds");

    // A 384 latent. The tensor below is 2.4 MB; the decode it implies is not.
    let z = Tensor::zeros((1, 4, 384, 384), DType::F32, &dev).unwrap();
    let err = decoder
        .forward(&z)
        .expect_err("a 9.0 GiB activation must be refused");
    assert!(
        err.to_string().contains("refusing to allocate"),
        "unexpected error: {err}"
    );
}

/// The encoder, which img2img needs. Its downsampler pads asymmetrically —
/// bottom and right only — and a symmetric `padding: 1` produces the right
/// shape with a half-pixel shift per level. Only a numerical comparison sees
/// that, which is why this test exists rather than a shape check.
#[test]
fn encoder_matches_diffusers_reference() {
    let path = golden_path();
    let vae_weights = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/vae_decoder/vae.safetensors");
    if !path.exists() || !vae_weights.exists() {
        eprintln!("SKIP encoder_matches_diffusers_reference: no reference data.");
        return;
    }

    let dev = Device::Cpu;
    let refs = sd_tensor::safetensors::load(&path, &dev).expect("loading reference tensors");
    let Some(image) = refs.get("encoder_input") else {
        eprintln!("SKIP: reference predates the encoder; regenerate it.");
        return;
    };
    let expected = refs.get("encoder_moments").expect("encoder_moments");

    let vb = sd_loader::safetensors_var_builder(&[&vae_weights], DType::F32, &dev)
        .expect("loading VAE weights");
    let encoder = sd_models::vae::AutoencoderKlEncoder::new(&VaeConfig::sd15(), vb)
        .expect("building encoder");

    let (mean, logvar) = encoder.encode_dist(image).expect("encode_dist");
    // The reference is the concatenated moments; compare the halves we split.
    let got = sd_tensor::Tensor::cat(&[&mean, &logvar], 1).expect("cat");
    assert_eq!(got.dims(), expected.dims());

    let c = testing::closeness(&got, expected).expect("comparing tensors");
    eprintln!("vae encoder vs diffusers: {c}");
    testing::assert_close(&got, expected, testing::DEFAULT_ATOL, "vae encoder moments").unwrap();
}
