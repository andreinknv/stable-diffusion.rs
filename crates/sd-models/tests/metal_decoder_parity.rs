//! CPU/Metal parity for the VAE decoder.
//!
//! candle 0.11's Metal backend miscomputes a VAE decode at a 128x128 latent
//! (1024px output). Smaller latents agree with CPU to 1e-4; at 128 the two
//! differ by ~1.0 on a [-1, 1] image, which is total corruption — the decode
//! comes out as horizontal noise bands.
//!
//! Isolated as far as it goes: `conv2d`, `silu`, `softmax_last_dim` and
//! `matmul` were each compared CPU against Metal at these shapes and at
//! tensor sizes up to 2.15 GB, and all agree. Chunked attention is not
//! involved either — the divergence is identical with chunking disabled, and
//! with chunks 64x smaller. So it is the composition, not any one op, and it
//! is size-triggered rather than model-specific: SD 1.5 and SDXL share this
//! decoder architecture.
//!
//! Practical effect: **SDXL at its native 1024 is correct on CPU and wrong on
//! Metal.** SD 1.5 at 512 is unaffected, which is why this went unnoticed
//! until SDXL.
//!
//! This test documents the boundary rather than asserting the bug is absent,
//! so it will start failing — informatively — if a candle upgrade fixes it.

use std::path::PathBuf;

use sd_models::vae::{AutoencoderKlDecoder, VaeConfig};
use sd_tensor::{DType, Device, Tensor};

/// Largest latent edge the Metal decoder is known to get right.
const KNOWN_GOOD_LATENT_EDGE: usize = 64;
/// Smallest known-bad edge.
const KNOWN_BAD_LATENT_EDGE: usize = 128;

fn weights() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/vae_decoder/vae.safetensors");
    p.exists().then_some(p)
}

#[test]
fn metal_decoder_agrees_with_cpu_up_to_the_known_boundary() {
    let Ok(metal) = Device::new_metal(0) else {
        eprintln!("SKIP: no Metal device");
        return;
    };
    let Some(w) = weights() else {
        eprintln!("SKIP: no VAE weights; run xtask/golden/dump_reference.py vae");
        return;
    };
    let cpu = Device::Cpu;
    let cfg = VaeConfig::sd15();

    let vb = sd_loader::safetensors_var_builder(&[&w], DType::F32, &cpu).expect("cpu weights");
    let dec_cpu = AutoencoderKlDecoder::new(&cfg, vb).expect("cpu decoder");
    let vb = sd_loader::safetensors_var_builder(&[&w], DType::F32, &metal).expect("metal weights");
    let dec_metal = AutoencoderKlDecoder::new(&cfg, vb).expect("metal decoder");

    let mut rng = sd_tensor::rng::SeededRng::new(0);
    let decode_both = |edge: usize, rng: &mut sd_tensor::rng::SeededRng| -> f32 {
        let latent: Tensor = rng.randn((1, 4, edge, edge), &cpu).unwrap();
        let a = dec_cpu.decode(&latent).unwrap();
        let b = dec_metal
            .decode(&latent.to_device(&metal).unwrap())
            .unwrap()
            .to_device(&cpu)
            .unwrap();
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        av.iter()
            .zip(&bv)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };

    let good = decode_both(KNOWN_GOOD_LATENT_EDGE, &mut rng);
    eprintln!("latent {KNOWN_GOOD_LATENT_EDGE}: max|cpu - metal| = {good:.5}");
    assert!(
        good < 1e-2,
        "Metal used to agree with CPU at a {KNOWN_GOOD_LATENT_EDGE} latent and no longer does \
         ({good}). That is a new regression, not the known 1024 one."
    );

    let bad = decode_both(KNOWN_BAD_LATENT_EDGE, &mut rng);
    eprintln!("latent {KNOWN_BAD_LATENT_EDGE}: max|cpu - metal| = {bad:.5}");
    if bad < 1e-2 {
        panic!(
            "Metal now agrees with CPU at a {KNOWN_BAD_LATENT_EDGE} latent ({bad}). The known \
             candle Metal bug appears to be fixed — delete this assertion, drop the CPU-decode \
             caveat from docs/backends.md, and re-enable SDXL at 1024 on Metal."
        );
    }
    eprintln!(
        "known-bad boundary reproduced: Metal decode diverges by {bad:.3} at a \
         {KNOWN_BAD_LATENT_EDGE} latent. See this file's header."
    );
}
