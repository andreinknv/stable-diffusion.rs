//! Export the sampling loop's inputs and output, for cross-checking.
use stable_diffusion_rs::models::flux::FluxConfig;
use stable_diffusion_rs::pipeline::{paths_in, FluxConfigRun, FluxPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = sd_tensor::device::best()?;
    let pipe = FluxPipeline::load(
        &paths_in(std::path::Path::new("tests/golden/flux")),
        &FluxConfig::mini(),
        &dev,
    )?;
    let cfg = FluxConfigRun {
        prompt: "a rusty crab on a beach, detailed photograph, golden hour".into(),
        width: 512,
        height: 512,
        steps: 20,
        guidance: 3.5,
        seed: 42,
    };
    let (txt, pooled, init) = pipe.sampling_inputs(&cfg)?;
    let (latent, _) = pipe.run_capturing_latent(&cfg)?;
    let m = std::collections::HashMap::from([
        ("txt".to_string(), txt),
        ("pooled".to_string(), pooled),
        ("init_packed".to_string(), init),
        ("final_latent".to_string(), latent),
    ]);
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "inputs.safetensors".into());
    sd_tensor::safetensors::save(&m, &out)?;
    eprintln!("wrote {out}");
    Ok(())
}
