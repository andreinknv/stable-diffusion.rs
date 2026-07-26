//! Generate one image with Flux. Fixtures live under `tests/golden/flux`.
use stable_diffusion_rs::models::flux::FluxConfig;
use stable_diffusion_rs::pipeline::{paths_in, FluxConfigRun, FluxPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "a rusty crab on a beach, detailed photograph, golden hour".into());
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let size: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let out = args.get(4).cloned().unwrap_or_else(|| "flux.png".into());

    let dev = sd_tensor::device::best()?;
    eprintln!("device: {dev:?}; loading four models…");
    let t0 = std::time::Instant::now();
    let paths = paths_in(std::path::Path::new("tests/golden/flux"));
    let pipe = FluxPipeline::load(&paths, &FluxConfig::mini(), &dev)?;
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let cfg = FluxConfigRun {
        prompt,
        width: size,
        height: size,
        steps,
        guidance: 3.5,
        seed: 42,
    };
    let t1 = std::time::Instant::now();
    let image = pipe.run_with_progress(&cfg, |i, n| eprintln!("  step {i}/{n}"))?;
    eprintln!("generated in {:.1}s", t1.elapsed().as_secs_f64());

    stable_diffusion_rs::image_io::save_png(&image, &out)?;
    eprintln!("wrote {out}");
    Ok(())
}
