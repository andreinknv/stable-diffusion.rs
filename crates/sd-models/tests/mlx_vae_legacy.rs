//! The MLX VAE decoder against an *unmodified* SD 1.5 checkpoint.
//!
//! `mlx_golden_vae` loads `vae.safetensors`, which the dump script produces via
//! `vae.state_dict()`. That path silently renames the legacy attention keys, so
//! the test can never see a checkpoint that uses them — and **the stock SD 1.5
//! VAE, which is the file most people download, does**. On the candle side the
//! result was a decoder that passed every numerical test and could not load the
//! real checkpoint.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_vae_legacy -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::{normalise_legacy_attention, vae, Weights};
use sd_tensor::mlx::{load_safetensors, Stream};

/// `mlx_golden_vae`'s decoder bound.
const ATOL: f32 = 1e-4;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/vae_decoder")
}

/// The rename is exact in both directions, and leaves everything else alone.
///
/// A pure name check, so it runs with no weights at all.
#[test]
fn legacy_attention_keys_are_rewritten_both_ways() {
    for (modern, legacy) in [
        (".to_q.", ".query."),
        (".to_k.", ".key."),
        (".to_v.", ".value."),
        (".to_out.0.", ".proj_attn."),
    ] {
        let m = format!("decoder.mid_block.attentions.0{modern}weight");
        let l = format!("decoder.mid_block.attentions.0{legacy}weight");
        assert_eq!(sd_loader::legacy_attention_key(&m), Some(l.clone()));
        assert_eq!(sd_loader::modern_attention_key(&l), Some(m.clone()));
        // And each is a no-op on the other layout's own names.
        assert_eq!(sd_loader::modern_attention_key(&m), None);
        assert_eq!(sd_loader::legacy_attention_key(&l), None);
    }

    // Everything that is not an attention projection is left alone — most keys
    // in a legacy checkpoint, and every key in a modern one.
    for name in [
        "decoder.conv_in.weight",
        "decoder.up_blocks.0.resnets.0.norm1.bias",
        "decoder.mid_block.attentions.0.group_norm.weight",
    ] {
        assert_eq!(sd_loader::modern_attention_key(name), None, "{name}");
        assert_eq!(sd_loader::legacy_attention_key(name), None, "{name}");
    }
}

/// Normalising a map with no legacy keys must change nothing at all.
#[test]
fn normalising_a_modern_checkpoint_is_a_no_op() {
    let p = golden().join("vae.safetensors");
    if !p.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no vae fixture.");
        return;
    }
    let original = load_safetensors(&p).expect("vae");
    let mut w: Weights = load_safetensors(&p).expect("vae");
    normalise_legacy_attention(&mut w);

    assert_eq!(w.len(), original.len(), "the tensor count must not change");
    let mut names: Vec<&String> = w.keys().collect();
    let mut before: Vec<&String> = original.keys().collect();
    names.sort();
    before.sort();
    assert_eq!(
        names, before,
        "no key in a modern checkpoint may be rewritten"
    );
}

/// **The same decoder, the same expected output, but the weights come from the
/// file as published rather than as re-exported.**
#[test]
fn the_decoder_runs_from_an_unmodified_checkpoint() {
    let refs_p = golden().join("reference.safetensors");
    let legacy_p = golden().join("vae_legacy.safetensors");
    if !refs_p.exists() || !legacy_p.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no legacy VAE fixture. Generate it with:\n\n    \
             python3 xtask/golden/dump_reference.py vae --output tests/golden\n"
        );
        return;
    }
    let s = Stream::gpu();
    let refs = load_safetensors(&refs_p).expect("reference");
    let mut w = load_safetensors(&legacy_p).expect("the stock VAE checkpoint");

    // Without this the decoder asks for `to_q` and the file has `query`.
    assert!(
        w.keys().any(|k| k.contains(".query.")),
        "this fixture is supposed to be the legacy layout; it has no `query` keys"
    );
    normalise_legacy_attention(&mut w);
    assert!(
        w.keys().any(|k| k.contains(".to_q.")),
        "normalisation did not produce the modern names"
    );

    let latent = refs
        .get("latent")
        .expect("latent")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let got = vae::decode(&latent, &w, &s)
        .expect("decoding from a legacy-named checkpoint")
        .transpose(&[0, 3, 1, 2], &s)
        .unwrap()
        .to_vec_f32(&s)
        .unwrap();
    let want = refs.get("image").expect("image").to_vec_f32(&s).unwrap();

    let (mut worst, mut peak) = (0.0f32, 0.0f32);
    for (a, b) in got.iter().zip(&want) {
        worst = worst.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    eprintln!("legacy checkpoint  peak {peak:.3}  max_abs {worst:.3e}  atol {ATOL:.0e}");
    assert!(
        worst <= ATOL,
        "the decoder from an unmodified checkpoint is {worst:.3e} out"
    );
}

/// Without normalising, the decode must **fail** rather than quietly returning
/// something.
///
/// The whole risk here is a silent one, so the loud version is asserted too.
#[test]
fn a_legacy_checkpoint_is_refused_before_it_is_normalised() {
    let legacy_p = golden().join("vae_legacy.safetensors");
    let refs_p = golden().join("reference.safetensors");
    if !legacy_p.exists() || !refs_p.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no legacy VAE fixture.");
        return;
    }
    let s = Stream::gpu();
    let refs = load_safetensors(&refs_p).expect("reference");
    let w = load_safetensors(&legacy_p).expect("the stock VAE checkpoint");
    let latent = refs
        .get("latent")
        .unwrap()
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();

    assert!(
        vae::decode(&latent, &w, &s).is_err(),
        "a legacy-named checkpoint must fail naming the missing tensor, not decode \
         something else"
    );
}
