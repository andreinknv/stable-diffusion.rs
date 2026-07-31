//! FLUX.2's transformer against `diffusers`.
//!
//! ```bash
//! .venv/bin/python xtask/golden/dump_reference.py flux2 --output tests/golden
//! cargo test -p sd-models --features mlx --test mlx_golden_flux2 -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::flux2::{self, Flux2Config};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

const ATOL: f32 = 2e-4;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/flux2")
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let (r, w) = (
        golden().join("reference.safetensors"),
        golden().join("flux2.safetensors"),
    );
    if !r.exists() || !w.exists() {
        return None;
    }
    Some((
        load_safetensors(&r).expect("reference"),
        load_safetensors(&w).expect("weights"),
    ))
}

/// The fixture's geometry: 4 heads of 32, two double and two single blocks.
fn config() -> Flux2Config {
    Flux2Config {
        hidden_size: 128,
        num_heads: 4,
        depth: 2,
        depth_single_blocks: 2,
        mlp_ratio: 3.0,
        axes_dim: vec![8, 8, 8, 8],
        theta: 2000.0,
        eps: 1e-6,
        time_channels: 256,
        guidance_embed: true,
    }
}

fn flat(a: &Array, s: &Stream) -> Vec<f32> {
    a.to_vec_f32(s).expect("read")
}

#[test]
fn the_transformer_matches_diffusers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no flux2 fixture. See the module docs.");
        return;
    };
    let s = Stream::gpu();
    let cfg = config();

    let img = refs.get("hidden_states").expect("hidden_states");
    let txt = refs
        .get("encoder_hidden_states")
        .expect("encoder_hidden_states");
    // The reference stores the raw timestep and guidance; the model scales
    // both by 1000 before embedding, which is the caller's job here.
    let scale = Array::scalar_f32(1000.0).unwrap();
    let timestep = refs
        .get("timestep")
        .expect("timestep")
        .mul(&scale, &s)
        .unwrap();
    let guidance = refs
        .get("guidance")
        .expect("guidance")
        .mul(&scale, &s)
        .unwrap();

    let img_ids = flat(refs.get("img_ids").expect("img_ids"), &s);
    let txt_ids = flat(refs.get("txt_ids").expect("txt_ids"), &s);

    let got = flux2::forward(
        img,
        &img_ids,
        txt,
        &txt_ids,
        &timestep,
        Some(&guidance),
        &cfg,
        &w,
        &s,
    )
    .expect("forward");

    let want = refs.get("output").expect("output");
    assert_eq!(got.shape(), want.shape(), "output shape");

    let (g, e) = (flat(&got, &s), flat(want, &s));
    let (mut worst, mut peak) = (0.0f32, 0.0f32);
    for (a, b) in g.iter().zip(&e) {
        worst = worst.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    eprintln!("flux2 output  peak {peak:.3}  max_abs {worst:.3e}");
    assert!(worst <= ATOL, "the FLUX.2 transformer is {worst:.3e} out");
}

/// **The modulation is shared across blocks**, which is the structural claim
/// most likely to be got wrong — a per-block scheme would need one of these
/// tensors per block and the checkpoint has exactly three.
#[test]
fn the_modulation_is_three_tensors_for_the_whole_model() {
    let Some((_, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no flux2 fixture.");
        return;
    };
    for name in [
        "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.linear.weight",
        "single_stream_modulation.linear.weight",
    ] {
        assert!(w.contains_key(name), "missing {name}");
    }
    // And there is no per-block modulation anywhere.
    let per_block: Vec<&String> = w
        .keys()
        .filter(|k| k.contains("_mod.") || k.contains(".modulation."))
        .collect();
    assert!(
        per_block.is_empty(),
        "found per-block modulation weights: {per_block:?}"
    );

    let cfg = config();
    // Six vectors for the double stream, three for the single.
    let img = w["double_stream_modulation_img.linear.weight"].shape();
    assert_eq!(img[0], 6 * cfg.hidden_size, "6 * hidden");
    let single = w["single_stream_modulation.linear.weight"].shape();
    assert_eq!(single[0], 3 * cfg.hidden_size, "3 * hidden");
}

/// **Four rotary axes, not three.** Flux.1's `image_ids` emits three
/// coordinates per token; feeding those here silently drops an axis and
/// misplaces every position.
#[test]
fn the_rotary_embedding_takes_four_axes() {
    let cfg = config();
    assert_eq!(cfg.axes_dim.len(), 4);
    assert_eq!(
        cfg.axes_dim.iter().sum::<usize>(),
        cfg.head_dim(),
        "the axes must cover the head dimension exactly"
    );
    let ids = flux2::image_ids(3, 4);
    assert_eq!(ids.len(), 3 * 4 * 4, "four coordinates per token");
    // Row-major over the grid, with the first and last axes zero.
    assert_eq!(&ids[..4], &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&ids[4..8], &[0.0, 0.0, 1.0, 0.0], "second token is (0, 1)");
}

/// **The latent plumbing: two different 2x2 operations and a flatten.**
///
/// `patchify` folds the VAE's 32 channels into the 128 the transformer takes;
/// `pack` flattens the grid into tokens and folds nothing, because FLUX.2's
/// `patch_size` is 1. Doing the first twice, or the second as if it were the
/// first, gives a latent of exactly the right shape and the wrong content.
///
/// Pinned against `diffusers` rather than against its own inverse: a
/// round-trip test passes for a *pair* of wrong permutations, which is how
/// SD 3.5's unpatchify got through here once already.
#[test]
fn the_latent_patching_matches_diffusers() {
    let Some((refs, _)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no flux2 fixture.");
        return;
    };
    let s = Stream::gpu();
    let latent = refs.get("latent_nchw").expect("latent_nchw");
    assert_eq!(latent.shape(), vec![1, 32, 8, 6]);

    let patched = flux2::patchify_latents(latent, &s).expect("patchify");
    let want = refs.get("latent_patchified").expect("latent_patchified");
    assert_eq!(patched.shape(), want.shape(), "patchified shape");
    assert_eq!(
        flat(&patched, &s),
        flat(want, &s),
        "patchify disagrees with diffusers"
    );

    let packed = flux2::pack_latents(&patched, &s).expect("pack");
    let want = refs.get("latent_packed").expect("latent_packed");
    assert_eq!(packed.shape(), want.shape(), "packed shape");
    assert_eq!(flat(&packed, &s), flat(want, &s), "pack disagrees");

    let unpatched = flux2::unpatchify_latents(&patched, &s).expect("unpatchify");
    let want = refs
        .get("latent_unpatchified")
        .expect("latent_unpatchified");
    assert_eq!(unpatched.shape(), want.shape(), "unpatchified shape");
    assert_eq!(flat(&unpatched, &s), flat(want, &s), "unpatchify disagrees");

    // And the round trips do close, which the above does not by itself imply.
    let back = flux2::unpack_latents(&packed, 4, 3, &s).expect("unpack");
    assert_eq!(
        flat(&back, &s),
        flat(&patched, &s),
        "pack/unpack round trip"
    );
    assert_eq!(
        flat(&flux2::patchify_latents(&unpatched, &s).unwrap(), &s),
        flat(&patched, &s),
        "patchify/unpatchify round trip"
    );
}
