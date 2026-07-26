//! Backend benchmark: runs the SD 1.5 VAE decoder and times it.
//!
//! ```bash
//! cargo run --release -p sd-cli --example backend_bench -- 64
//! cargo run --release -p sd-cli --features metal --example backend_bench -- 64
//! ```
//!
//! The argument is the latent edge length; the image is 8x that. Attention
//! sequence length is `n*n`, and our attention is O(seq^2) in memory, which is
//! the whole point of this benchmark — see docs/backends.md.
//!
//! Always discard the first call: candle compiles Metal pipelines lazily, so
//! it includes shader compilation and is not representative.

use sd_tensor::nn::{VarBuilder, VarMap};
use sd_tensor::{device, DType, Tensor};
use stable_diffusion_rs::models::vae::{Decoder, DecoderConfig};

fn main() -> anyhow::Result<()> {
    let dev = device::best()?;
    println!("device: {dev:?}");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    // SD 1.5 geometry, 64x64 latent -> 512x512 image.
    let cfg = DecoderConfig {
        latent_channels: 4,
        out_channels: 3,
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        norm_num_groups: 32,
        norm_eps: 1e-6,
    };
    let d = Decoder::new(&cfg, vb)?;
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    println!(
        "latent {n}x{n} -> image {}x{}  (attention seq = {})",
        n * 8,
        n * 8,
        n * n
    );
    let z = Tensor::zeros((1, 4, n, n), DType::F32, &dev)?;

    // Warm-up: triggers Metal pipeline compilation and any lazy init.
    let warm = std::time::Instant::now();
    let out = d.forward(&z)?;
    let _ = out.flatten_all()?.to_vec1::<f32>()?;
    println!("warm-up (incl. shader compilation): {:?}", warm.elapsed());

    let mut best = std::time::Duration::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let out = d.forward(&z)?;
        let _ = out.flatten_all()?.to_vec1::<f32>()?; // force sync to host
        best = best.min(t.elapsed());
    }
    println!("best of 3 (steady state):         {best:?}");
    println!("output: {:?}", out.dims());
    Ok(())
}
