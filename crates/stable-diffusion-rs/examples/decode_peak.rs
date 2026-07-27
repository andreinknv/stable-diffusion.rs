//! Peak memory of one decode, isolated from loading and denoising.
//!
//! `sdrs`'s own peak RSS is dominated by the UNet's 3.4 GB and the weight
//! load, so a 3 GB difference in the decode barely moves it — measured
//! end to end the two decoders looked 0.14 GB apart, which is only the
//! weights. This decodes and nothing else.
//!
//! ```text
//!   cargo run --release -p stable-diffusion-rs --example decode_peak -- taesd 64
//!
//!   latent edge   output      VAE            TAESD
//!   64            512 px      3.43 GB        0.49 GB     7.0x
//!   128           1024 px     3.22 GB *      1.71 GB
//!
//!   * tiled. `decode_tiled` splits anything above a 64-latent edge, so the
//!     VAE stays under ~3.4 GB at any size by seaming the image instead.
//!     TAESD does 1024 in one pass.
//! ```
//!
//! Measured with `/usr/bin/time -l` on an M4 Max, Metal, f32.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = sd_tensor::Device::new_metal(0).unwrap_or(sd_tensor::Device::Cpu);
    let which = std::env::args().nth(1).unwrap_or_else(|| "vae".into());
    let edge: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let latent = sd_tensor::Tensor::randn(0f32, 1.0, (1, 4, edge, edge), &dev)?;

    let out = if which == "taesd" {
        let vb = sd_loader::safetensors_var_builder(
            &[std::path::Path::new("tests/golden/taesd/taesd.safetensors")],
            sd_tensor::DType::F32,
            &dev,
        )?;
        sd_models::vae::TinyDecoder::new(4, 3, vb)?.decode(&latent)?
    } else {
        let vb = sd_loader::safetensors_var_builder(
            &[std::path::Path::new(
                "models/sd15/vae/diffusion_pytorch_model.safetensors",
            )],
            sd_tensor::DType::F32,
            &dev,
        )?;
        sd_models::vae::AutoencoderKlDecoder::new(&sd_models::vae::VaeConfig::sd15(), vb)?
            .decode_tiled(&latent)?
    };
    dev.synchronize()?;
    println!("{which}: decoded {edge}x{edge} -> {:?}", out.dims());
    Ok(())
}
