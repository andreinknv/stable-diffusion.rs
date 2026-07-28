//! Flux's VAE against `diffusers`.
//!
//! The same convolutional geometry as SD's, so this is not a new port — it is
//! a check that the existing one is genuinely parameterised rather than
//! accidentally specialised to SD. Three things differ and each has a silent
//! failure mode:
//!
//! - **16 latent channels** instead of 4. A hardcoded 4 gives a shape error,
//!   which is the harmless case.
//! - **No `quant_conv` / `post_quant_conv`.** Flux sets `use_quant_conv:
//!   false`. Building them anyway looks for weights that do not exist; not
//!   building them when they do exist silently drops a 1x1 convolution.
//! - **A latent shift** as well as a scale: `(x - shift) * scale`. Applying
//!   these in the wrong order leaves a recognisable image with wrong
//!   contrast — the failure that survives eyeballing.
//!
//! Regenerate with:
//! `python3 xtask/golden/dump_reference.py flux_vae --output tests/golden`

use std::path::PathBuf;

use sd_models::vae::{AutoencoderKlDecoder, AutoencoderKlEncoder, VaeConfig};
use sd_tensor::{testing, DType, Device};

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(name)
}

/// The reference is ~200 MB and stays out of git, so CI skips this.
macro_rules! refs_or_skip {
    ($dev:expr, $dir:expr) => {{
        let (r, w) = (
            golden(&format!("{}/reference.safetensors", $dir)),
            golden(&format!("{}/vae.safetensors", $dir)),
        );
        if !r.exists() || !w.exists() {
            sd_tensor::skip_missing_fixture!(
                "SKIP: no Flux VAE reference. Generate with \
                 `python3 xtask/golden/dump_reference.py flux_vae --output tests/golden`"
            );
            return;
        }
        (
            sd_tensor::safetensors::load(&r, $dev).unwrap(),
            sd_loader::safetensors_var_builder(&[&w], DType::F32, $dev).unwrap(),
        )
    }};
}

/// One suite per 16-channel model. The decoder is shared, so running the same
/// checks against both is what shows the config genuinely parameterises it
/// rather than the Flux path having been special-cased.
macro_rules! vae_suite {
    ($name:ident, $dir:literal, $cfg:expr) => {
        mod $name {
            use super::*;
            const DIR: &str = $dir;
            #[allow(non_snake_case)]
            fn CONFIG() -> VaeConfig {
                $cfg
            }
            #[test]
            fn decoder_matches_diffusers() {
                let dev = Device::Cpu;
                let (refs, vb) = refs_or_skip!(&dev, DIR);

                let dec = AutoencoderKlDecoder::new(&CONFIG(), vb).unwrap();
                let latent = refs.get("latent").unwrap();
                assert_eq!(
                    latent.dim(1).unwrap(),
                    16,
                    "the reference latent should be 16-channel; regenerate it"
                );

                let got = dec.decode_raw(latent).unwrap();
                let c = testing::closeness(&got, refs.get("image").unwrap()).unwrap();
                eprintln!(
                    "{DIR} decode: max_abs {:.3e}, mean_abs {:.3e}",
                    c.max_abs, c.mean_abs
                );
                assert!(
                    c.max_abs < 1e-4,
                    "flux decoder diverged: max_abs {:.3e}",
                    c.max_abs
                );
            }

            #[test]
            fn encoder_matches_diffusers() {
                let dev = Device::Cpu;
                let (refs, vb) = refs_or_skip!(&dev, DIR);

                let enc = AutoencoderKlEncoder::new(&CONFIG(), vb).unwrap();
                let (mean, logvar) = enc.encode_dist(refs.get("encoder_input").unwrap()).unwrap();

                let moments = refs.get("encoder_moments").unwrap();
                let want_mean = moments.narrow(1, 0, 16).unwrap();
                let want_logvar = moments.narrow(1, 16, 16).unwrap();

                let cm = testing::closeness(&mean, &want_mean).unwrap();
                let cl = testing::closeness(&logvar, &want_logvar).unwrap();
                eprintln!(
                    "{DIR} encode: mean max_abs {:.3e}, logvar max_abs {:.3e}",
                    cm.max_abs, cl.max_abs
                );

                // 2e-3, not the 1e-4 the decoder is held to, and the looser bound is
                // measured rather than guessed.
                //
                // Running diffusers' own Flux encoder in f32 and f64 and comparing gives
                // max_abs 9.605e-4 — its f32 noise floor. Our deviation from diffusers
                // f32 is 9.606e-4. We are *at* the floor, so a tighter bound would be
                // asserting that f32 is more precise than it is. The same measurement on
                // SD's encoder gives 1.226e-4, which is why that one holds to 1e-4: the
                // Flux VAE is genuinely ~8x worse conditioned, and its config says as
                // much by setting `force_upcast`.
                //
                // This is not a licence to be sloppy. A structural fault here is orders
                // of magnitude larger, not marginally: the symmetric-padding bug in this
                // same encoder measured 17.32.
                const ENCODER_NOISE_FLOOR: f64 = 2e-3;
                assert!(
                    cm.max_abs < ENCODER_NOISE_FLOOR,
                    "mean diverged: {:.3e}",
                    cm.max_abs
                );
                assert!(
                    cl.max_abs < ENCODER_NOISE_FLOOR,
                    "logvar diverged: {:.3e}",
                    cl.max_abs
                );

                // max_abs is set by a handful of unlucky positions, so on its own it
                // would tolerate a broad drift that moved every value by 1e-3. mean_abs
                // is ~4e-6 and pins that down.
                assert!(
                    cm.mean_abs < 5e-5 && cl.mean_abs < 5e-5,
                    "encoder drifted broadly rather than at isolated points: \
                     mean {:.3e}, logvar {:.3e}",
                    cm.mean_abs,
                    cl.mean_abs
                );
            }

            /// The scale and shift, in both directions.
            ///
            /// `encode`/`decode` differ from `encode_dist`/`decode_raw` only by this
            /// parameterisation, so comparing them against references that had it applied
            /// in Python isolates it from the convolutions entirely. Both orderings are
            /// checked because each is independently reversible-looking.
            #[test]
            fn latent_scale_and_shift_round_trip() {
                let dev = Device::Cpu;
                let (refs, vb) = refs_or_skip!(&dev, DIR);
                let cfg = CONFIG();

                // Forward: (mean - shift) * scale.
                let enc = AutoencoderKlEncoder::new(&cfg, vb.clone()).unwrap();
                let got = enc.encode(refs.get("encoder_input").unwrap()).unwrap();
                let c = testing::closeness(&got, refs.get("encoder_scaled_mean").unwrap()).unwrap();
                // Scaled by 0.3611, so the encoder's noise floor shrinks with it.
                assert!(
                    c.max_abs < 1e-3,
                    "scaled latent diverged: max_abs {:.3e} — check shift is applied \
                     before the scale, not after",
                    c.max_abs
                );

                // Inverse: x / scale + shift, all the way through to pixels.
                let dec = AutoencoderKlDecoder::new(&cfg, vb).unwrap();
                let got = dec.decode(refs.get("latent").unwrap()).unwrap();
                let c = testing::closeness(&got, refs.get("decoded_from_scaled").unwrap()).unwrap();
                assert!(
                    c.max_abs < 1e-4,
                    "unscaled decode diverged: max_abs {:.3e}",
                    c.max_abs
                );
            }
        }
    };
}

vae_suite!(flux, "flux_vae", VaeConfig::flux());
vae_suite!(sd35, "sd35_vae", VaeConfig::sd35());

/// Guards the config itself. These constants come from the checkpoint's
/// `config.json`; a typo in one is invisible until an image comes out flat.
#[test]
fn flux_config_differs_from_sd_where_it_should() {
    let (flux, sd) = (VaeConfig::flux(), VaeConfig::sd15());
    assert_eq!(flux.latent_channels, 16);
    assert_eq!(sd.latent_channels, 4);
    assert!(!flux.use_quant_conv, "Flux has no quant_conv");
    assert!(sd.use_quant_conv, "SD does");
    assert_ne!(flux.shift_factor, 0.0, "Flux latents are shifted");
    assert_eq!(sd.shift_factor, 0.0, "SD latents are not");
    // The shared half: if these ever diverge the decoder is no longer reusable
    // and this file is testing the wrong thing.
    assert_eq!(flux.block_out_channels, sd.block_out_channels);
    assert_eq!(flux.layers_per_block, sd.layers_per_block);
    assert_eq!(flux.norm_num_groups, sd.norm_num_groups);

    // SD 3.5 is structurally Flux's VAE with different latent constants. If
    // that ever stops being true the shared suite above is testing a lie.
    let s35 = VaeConfig::sd35();
    assert_eq!(s35.latent_channels, flux.latent_channels);
    assert_eq!(s35.use_quant_conv, flux.use_quant_conv);
    assert_eq!(s35.block_out_channels, flux.block_out_channels);
    assert_ne!(s35.scaling_factor, flux.scaling_factor);
    assert_ne!(s35.shift_factor, flux.shift_factor);
}
