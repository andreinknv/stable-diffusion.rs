//! The VAE decoder against an *unmodified* SD 1.5 checkpoint.
//!
//! `golden_vae.rs` loads `vae.safetensors`, which the dump script produces via
//! `vae.state_dict()`. That path silently renames the legacy attention keys on
//! load, so the test can never see a checkpoint that uses them — and the stock
//! SD 1.5 VAE does. The result was a decoder that passed every numerical test
//! and could not load the file most people actually download.
//!
//! This file loads the raw checkpoint. It is the test that would have caught
//! it, and the reason it lives beside the other one rather than replacing it:
//! both paths are worth pinning.

use std::path::PathBuf;

use sd_models::vae::{AutoencoderKlDecoder, VaeConfig};
use sd_tensor::{testing, DType, Device};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/vae_decoder")
}

#[test]
fn legacy_attention_keys_are_rewritten() {
    // A pure name check, so it runs in CI with no weights at all.
    assert_eq!(
        sd_loader::legacy_attention_key("decoder.mid_block.attentions.0.to_q.weight"),
        Some("decoder.mid_block.attentions.0.query.weight".to_string())
    );
    assert_eq!(
        sd_loader::legacy_attention_key("decoder.mid_block.attentions.0.to_out.0.bias"),
        Some("decoder.mid_block.attentions.0.proj_attn.bias".to_string())
    );
    for (modern, legacy) in [(".to_k.", ".key."), (".to_v.", ".value.")] {
        let name = format!("encoder.mid_block.attentions.0{modern}weight");
        assert_eq!(
            sd_loader::legacy_attention_key(&name),
            Some(format!("encoder.mid_block.attentions.0{legacy}weight"))
        );
    }

    // Anything else is left alone — most keys in a legacy checkpoint, and
    // every key in a modern one.
    assert_eq!(
        sd_loader::legacy_attention_key("decoder.conv_in.weight"),
        None
    );
    assert_eq!(
        sd_loader::legacy_attention_key("decoder.up_blocks.0.resnets.0.norm1.bias"),
        None
    );
}

#[test]
fn decoder_matches_diffusers_from_an_unmodified_checkpoint() {
    let refs_path = golden_dir().join("reference.safetensors");
    let legacy_weights = golden_dir().join("vae_legacy.safetensors");
    if !refs_path.exists() || !legacy_weights.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no reference data.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py vae --output tests/golden\n"
        );
        return;
    }

    let dev = Device::Cpu;
    let refs = sd_tensor::safetensors::load(&refs_path, &dev).expect("loading reference tensors");

    // The whole point: the same decoder, the same expected output, but the
    // weights come from the file as published rather than as re-exported.
    let vb = sd_loader::safetensors_var_builder(&[&legacy_weights], DType::F32, &dev)
        .expect("loading the stock VAE checkpoint");
    let decoder = AutoencoderKlDecoder::new(&VaeConfig::sd15(), vb)
        .expect("building the decoder from legacy-named weights");

    let latent = refs.get("latent").expect("reference has 'latent'");
    let expected = refs.get("image").expect("reference has 'image'");
    let got = decoder.decode_raw(latent).expect("decode_raw");

    let c = testing::closeness(&got, expected).expect("comparing tensors");
    eprintln!("stock checkpoint vs diffusers: {c}");
    testing::assert_close(
        &got,
        expected,
        testing::DEFAULT_ATOL,
        "vae decoder from an unmodified checkpoint",
    )
    .unwrap();
}
