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
    // SD_STREAM_DIFFUSION=1 keeps the transformer's blocks in host memory and
    // copies each to the accelerator as it is reached.
    let mut placement = stable_diffusion_rs::pipeline::Placement::on(&dev);
    if std::env::var("SD_STREAM_DIFFUSION").is_ok() {
        placement = placement.with_streamed_diffusion();
    }
    eprintln!("diffusion residency: {:?}", placement.diffusion());
    let pipe = FluxPipeline::load_with_placement(&paths, &cfg, &placement)?;
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
    if let Some(b) = sd_tensor::sysmem::available_bytes() {
        eprintln!("available after load: {:.2} GB", b as f64 / 1e9);
    }

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
    // SD_PREVIEW_EVERY=5 writes the model's running x0 estimate every 5 steps.
    // It needs a tiny decoder — a VAE decode per step costs more than the
    // step — so point SD_TAESD at `madebyollin/taef1`.
    let preview_every: usize = std::env::var("SD_PREVIEW_EVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let pipe = match std::env::var("SD_TAESD") {
        Ok(p) => {
            eprintln!("decoding with TAESD from {p}");
            pipe.with_taesd(std::path::Path::new(&p))?
        }
        Err(_) => pipe,
    };
    let image = pipe.run_with_progress(&cfg, |p| {
        eprintln!("  step {}/{}", p.step, p.total);
        if preview_every > 0 && (p.step % preview_every == 0 || p.step == p.total) {
            let path = format!("{}-preview-{:03}.png", out.trim_end_matches(".png"), p.step);
            match pipe
                .preview(p.denoised)
                .map_err(|e| e.to_string())
                .and_then(|img| {
                    stable_diffusion_rs::image_io::save_png(&img, &path).map_err(|e| e.to_string())
                }) {
                Ok(()) => eprintln!("  wrote {path}"),
                Err(e) => eprintln!("  preview failed: {e}"),
            }
        }
    })?;
    eprintln!("generated in {:.1}s", t1.elapsed().as_secs_f64());

    stable_diffusion_rs::image_io::save_png(&image, &out)?;
    eprintln!("wrote {out}");
    Ok(())
}
