//! CPU/Metal parity for the VAE decoder, and the GPU out-of-memory boundary.
//!
//! A 1024px decode does not fit in GPU memory on a 36 GiB Mac, and until the
//! synchronize in `Decoder::forward` it did not say so. candle queues Metal
//! work and only inspects the command buffer's status when something
//! synchronizes, so the decode *returned a tensor* — of whatever the buffer
//! happened to hold — and the failure was never discovered. The symptom was
//! an image of horizontal noise bands.
//!
//! Root cause of the memory itself: **conv im2col, not the activations.**
//! `DecoderConfig::peak_alloc_bytes` now counts this: candle's conv2d
//! materialises an im2col intermediate of `cin * 9` values per output
//! position, which for the decoder's 256->128 convolution at 1024x1024 is
//! **9.66 GB, eighteen times the activation it accompanies.** An earlier
//! version counted activations alone and reported 1.07 GB, which is worse
//! than no estimate because it reads as reassurance.
//!
//! Ruled out along the way, each measured rather than assumed: individual ops
//! (`conv2d` including a 9.66 GB im2col, `silu`, `group_norm`,
//! `softmax_last_dim`, `matmul`, `upsample_nearest2d`) all agree CPU against
//! Metal; chunked attention is irrelevant (identical with it disabled and
//! with chunks 64x smaller); and the corruption was deterministic across
//! runs, which is what ruled memory pressure back *in* rather than out.

use std::path::PathBuf;

use sd_models::vae::{AutoencoderKlDecoder, VaeConfig};
use sd_tensor::{DType, Device, Tensor};

/// Largest latent edge that fits in GPU memory here.
const FITS: usize = 64;
/// Smallest edge known not to.
const DOES_NOT_FIT: usize = 128;

fn weights() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/vae_decoder/vae.safetensors");
    p.exists().then_some(p)
}

#[test]
fn metal_decode_matches_cpu_and_fails_loudly_when_it_cannot() {
    let Ok(metal) = Device::new_metal(0) else {
        eprintln!("SKIP: no Metal device");
        return;
    };
    let Some(w) = weights() else {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no VAE weights; run xtask/golden/dump_reference.py vae"
        );
        return;
    };
    let cpu = Device::Cpu;
    let cfg = VaeConfig::sd15();

    let vb = sd_loader::safetensors_var_builder(&[&w], DType::F32, &cpu).expect("cpu weights");
    let dec_cpu = AutoencoderKlDecoder::new(&cfg, vb).expect("cpu decoder");
    let vb = sd_loader::safetensors_var_builder(&[&w], DType::F32, &metal).expect("metal weights");
    let dec_metal = AutoencoderKlDecoder::new(&cfg, vb).expect("metal decoder");

    let latent = |edge: usize| -> Tensor {
        sd_tensor::rng::SeededRng::new(0)
            .randn((1, 4, edge, edge), &cpu)
            .unwrap()
    };

    // Within budget: Metal must agree with CPU.
    let z = latent(FITS);
    let a = dec_cpu.decode(&z).expect("cpu decode");
    let b = dec_metal
        .decode(&z.to_device(&metal).unwrap())
        .expect("metal decode should fit at this size")
        .to_device(&cpu)
        .unwrap();
    let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let d = av
        .iter()
        .zip(&bv)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    eprintln!("latent {FITS}: max|cpu - metal| = {d:.5}");
    assert!(d < 1e-2, "Metal diverged from CPU at a {FITS} latent ({d})");

    // Beyond it: an error, never a wrong tensor. That distinction is the
    // whole point — the failure mode this replaced returned noise.
    let z = latent(DOES_NOT_FIT);
    match dec_metal.decode(&z.to_device(&metal).unwrap()) {
        Err(e) => {
            let msg = e.to_string();
            eprintln!("latent {DOES_NOT_FIT}: refused as expected — {msg}");
            assert!(
                msg.contains("Insufficient Memory") || msg.to_lowercase().contains("memory"),
                "expected an out-of-memory error, got: {msg}"
            );
        }
        Ok(out) => {
            // Not a failure in itself — a bigger GPU, or a candle that stops
            // materialising im2col, would land here. But it must be *correct*,
            // not merely returned.
            let got = out.to_device(&cpu).unwrap();
            let want = dec_cpu.decode(&z).expect("cpu decode");
            let gv = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let wv = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let d = gv
                .iter()
                .zip(&wv)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                d < 1e-2,
                "a {DOES_NOT_FIT} latent decoded on Metal without erroring, but the result is \
                 wrong (max|diff| {d}). That is the silent-corruption bug returning."
            );
            eprintln!("latent {DOES_NOT_FIT}: now fits and matches CPU — update docs/backends.md");
        }
    }
}
