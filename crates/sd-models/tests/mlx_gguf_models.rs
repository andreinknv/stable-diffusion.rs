//! Running MLX models from GGUF checkpoints.
//!
//! `mlx_gguf_agrees_with_candle` already establishes that the reader is
//! bit-exact. What is left is the other half of the problem, and it is not
//! about numbers: `stable-diffusion.cpp` writes the original CompVis/LDM
//! names, this project uses `diffusers` ones, and the decoder's block order is
//! **reversed** between them. A wrong translation loads cleanly and decodes
//! noise.
//!
//! **The f16 row is the control.** Every quantisation shares one name mapping,
//! so if f16 agrees with the golden reference then the mapping is exact and
//! every other row is quantisation alone. Without it a poor Q4_0 result cannot
//! be told apart from a subtly wrong translation.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_gguf_models -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::{clip, gguf, unet_forward, vae, UNetConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(name)
}

/// Pearson correlation. Preferred to an absolute bound because a wrong mapping
/// lands near zero whatever its magnitude, while quantisation shifts magnitude
/// without destroying structure.
fn correlation(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let cov: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f32 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f32 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va.sqrt() * vb.sqrt())
}

/// The VAE decoder from GGUF, against the safetensors golden reference.
///
/// The decoder is where a bad translation shows loudest: LDM's `up.0` is
/// diffusers' `up_blocks.3`, and running the blocks in file order produces an
/// image of exactly the right shape made of noise.
#[test]
fn the_vae_decodes_from_gguf() {
    let refs_p = golden("vae_decoder/reference.safetensors");
    let f16 = golden("gguf/sd15-f16.gguf");
    if !refs_p.exists() || !f16.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no vae reference or sd15 f16 gguf.");
        return;
    }
    let s = Stream::gpu();
    let refs = load_safetensors(&refs_p).expect("reference");
    let w = gguf::vae(&f16, &s).expect("vae from gguf");

    let latent = refs
        .get("latent")
        .expect("latent")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let got = vae::decode(&latent, &w, &s).expect("decode");
    let got = got
        .transpose(&[0, 3, 1, 2], &s)
        .unwrap()
        .to_vec_f32(&s)
        .unwrap();
    let want = refs.get("image").expect("image").to_vec_f32(&s).unwrap();

    let r = correlation(&got, &want);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("vae from f16 gguf: correlation {r:.6}, max_abs {max_abs:.4}");
    // f16 is the control row: this is the *same* weights at half precision, so
    // the only difference from the safetensors path is the rounding. A reversed
    // block order lands near zero here, not near 0.99.
    assert!(
        r > 0.999,
        "the VAE decoded from GGUF correlates {r:.4} with the reference; check the \
         decoder's block order, which is reversed across the LDM translation"
    );
}

/// CLIP's text tower from GGUF.
#[test]
fn clip_encodes_from_gguf() {
    let refs_p = golden("clip_encoder/reference.safetensors");
    let f16 = golden("gguf/sd15-f16.gguf");
    if !refs_p.exists() || !f16.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no clip reference or sd15 f16 gguf.");
        return;
    }
    let s = Stream::gpu();
    let refs = load_safetensors(&refs_p).expect("reference");
    let w = gguf::clip(&f16, &s).expect("clip from gguf");

    let ids = refs.get("token_ids").expect("token_ids");
    let f = ids.to_f32(&s).unwrap().to_vec_f32(&s).unwrap();
    let v: Vec<i32> = f.iter().map(|&x| x as i32).collect();
    let ids = Array::from_slice_i32(&v, &ids.shape()).unwrap();

    let got = clip::text_encoder(&ids, &w, &s)
        .expect("text encoder")
        .to_vec_f32(&s)
        .unwrap();
    let want = refs
        .get("last_hidden_state")
        .expect("last_hidden_state")
        .to_vec_f32(&s)
        .unwrap();

    let r = correlation(&got, &want);
    eprintln!("clip from f16 gguf: correlation {r:.6}");
    assert!(r > 0.999, "CLIP from GGUF correlates only {r:.4}");
}

/// The UNet from GGUF.
#[test]
fn the_unet_runs_from_gguf() {
    let refs_p = golden("unet_full/reference.safetensors");
    let f16 = golden("gguf/sd15-f16.gguf");
    if !refs_p.exists() || !f16.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no unet reference or sd15 f16 gguf.");
        return;
    }
    let s = Stream::gpu();
    let refs = load_safetensors(&refs_p).expect("reference");
    let w = gguf::unet(&f16, &s).expect("unet from gguf");

    let sample = refs
        .get("sample")
        .expect("sample")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let got = unet_forward(
        &sample,
        refs.get("timestep").expect("timestep"),
        refs.get("context").expect("context"),
        &UNetConfig::sd15(),
        &w,
        &s,
    )
    .expect("unet")
    .transpose(&[0, 3, 1, 2], &s)
    .unwrap()
    .to_vec_f32(&s)
    .unwrap();
    let want = refs.get("output").expect("output").to_vec_f32(&s).unwrap();

    let r = correlation(&got, &want);
    eprintln!("unet from f16 gguf: correlation {r:.6}");
    assert!(r > 0.999, "the UNet from GGUF correlates only {r:.4}");
}

/// **What each quantisation costs**, on one tower, against the same reference.
///
/// Ordered by descending fidelity. The point is the ordering as much as the
/// values: a Q4_0 that beats Q8_0 means the dequantiser is wrong for one of
/// them, which no single-row test can see.
#[test]
fn quantisation_costs_what_it_should() {
    let refs_p = golden("vae_decoder/reference.safetensors");
    if !refs_p.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no vae reference.");
        return;
    }
    let s = Stream::gpu();
    let refs = load_safetensors(&refs_p).expect("reference");
    let latent = refs
        .get("latent")
        .expect("latent")
        .transpose(&[0, 2, 3, 1], &s)
        .unwrap();
    let want = refs.get("image").expect("image").to_vec_f32(&s).unwrap();

    let rows = [
        ("f16 ", golden("gguf/sd15-f16.gguf")),
        ("Q8_0", golden("gguf/sd15-q8_0.gguf")),
        ("Q4_0", golden("gguf/sd15-q4_0.gguf")),
    ];
    let mut measured = Vec::new();
    for (label, path) in &rows {
        if !path.exists() {
            eprintln!("{label}  (absent)");
            continue;
        }
        let w = gguf::vae(path, &s).expect("vae from gguf");
        let got = vae::decode(&latent, &w, &s)
            .expect("decode")
            .transpose(&[0, 3, 1, 2], &s)
            .unwrap()
            .to_vec_f32(&s)
            .unwrap();
        let r = correlation(&got, &want);
        eprintln!("{label}  correlation {r:.6}");
        measured.push((*label, r));
    }
    if measured.len() < 2 {
        sd_tensor::skip_missing_fixture!("SKIP: fewer than two quantisations on disk.");
        return;
    }
    // Every row must be recognisably the same image — a wrong mapping is not
    // 0.9, it is near zero.
    for (label, r) in &measured {
        assert!(
            *r > 0.9,
            "{label} correlates {r:.4}; that is a translation fault, not quantisation"
        );
    }
    // And fidelity must fall with bit width, not rise.
    for pair in measured.windows(2) {
        let ((la, ra), (lb, rb)) = (pair[0], pair[1]);
        assert!(
            ra >= rb - 1e-6,
            "{lb} ({rb:.6}) beat {la} ({ra:.6}); one of the two dequantisers is wrong"
        );
    }
}

/// **A GGUF that carries no tensors for a tower must say so**, rather than
/// returning an empty map that then fails deep inside a forward pass naming one
/// arbitrary missing tensor.
#[test]
fn a_checkpoint_without_a_tower_is_refused() {
    let unrelated = golden("gguf/moe_shakespeare15M.gguf");
    if !unrelated.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no unrelated gguf on disk.");
        return;
    }
    let s = Stream::gpu();
    assert!(
        gguf::vae(&unrelated, &s).is_err(),
        "a language model carries no VAE and the loader must say so"
    );
}
