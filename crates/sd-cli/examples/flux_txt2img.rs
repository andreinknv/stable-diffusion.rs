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
    // Read the geometry from the file rather than assuming: schnell and dev
    // are 19/38 blocks, flux-mini 5/10.
    let cfg = if paths.transformer.extension().is_some_and(|e| e == "gguf") {
        let (d, s) = stable_diffusion_rs::loader::flux_block_counts(&paths.transformer)?;
        let guidance = stable_diffusion_rs::loader::flux_has_guidance(&paths.transformer)?;
        eprintln!("checkpoint: {d} double + {s} single blocks, guidance={guidance}");
        FluxConfig {
            depth: d,
            depth_single_blocks: s,
            guidance_embed: guidance,
            ..FluxConfig::mini()
        }
    } else {
        FluxConfig::mini()
    };
    let pipe = FluxPipeline::load(&paths, &cfg, &dev)?;
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
    // Releasing rather than borrowing: this renders one image and exits, so
    // the transformer is dead weight by the time the VAE runs.
    let image = pipe.run_releasing(&cfg, |i, n| eprintln!("  step {i}/{n}"))?;
    eprintln!("generated in {:.1}s", t1.elapsed().as_secs_f64());

    stable_diffusion_rs::image_io::save_png(&image, &out)?;
    eprintln!("wrote {out}");
    Ok(())
}
