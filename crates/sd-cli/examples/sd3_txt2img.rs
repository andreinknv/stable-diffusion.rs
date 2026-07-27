//! Generate one image with SD 3.5. Fixtures under `tests/golden/sd35`.
use stable_diffusion_rs::models::sd3::Sd3Config;
use stable_diffusion_rs::pipeline::{sd3_paths_in, Placement, Sd3Pipeline, Sd3RunConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let prompt = a
        .get(1)
        .cloned()
        .unwrap_or_else(|| "a rusty crab on a beach, detailed photograph, golden hour".into());
    let steps: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(28);
    let size: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
    let out = a.get(4).cloned().unwrap_or_else(|| "sd3.png".into());

    let dev = sd_tensor::device::best()?;
    eprintln!("device: {dev:?}; loading five models…");
    let t0 = std::time::Instant::now();
    let paths = sd3_paths_in(std::path::Path::new("tests/golden/sd35"));
    // SD_TEXT_ENCODERS_ON=cpu keeps the three encoders off the accelerator.
    // They run once and then hold more memory than the transformer does.
    let mut placement = match std::env::var("SD_TEXT_ENCODERS_ON").as_deref() {
        Ok("cpu") => Placement::on(&dev).with_text_encoders_on(&sd_tensor::Device::Cpu),
        Ok("auto") => Placement::auto(&dev, Sd3Pipeline::stage_bytes(&paths)?)?,
        _ => Placement::on(&dev),
    };
    if std::env::var("SD_STREAM_DIFFUSION").is_ok() {
        placement = placement.with_streamed_diffusion();
    }
    eprintln!("diffusion residency: {:?}", placement.diffusion());
    eprintln!(
        "placement: compute {:?}, text encoders {:?}, vae {:?}",
        placement.compute(),
        placement.text_encoders(),
        placement.vae()
    );
    let pipe = Sd3Pipeline::load_with_placement(&paths, &Sd3Config::medium_35(), &placement)?;
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
    if let Some(b) = sd_tensor::sysmem::available_bytes() {
        eprintln!("available after load: {:.2} GB", b as f64 / 1e9);
    }

    let cfg = Sd3RunConfig {
        prompt,
        negative_prompt: String::new(),
        width: size,
        height: size,
        steps,
        cfg_scale: 4.5,
        seed: 42,
    };
    let t1 = std::time::Instant::now();
    // One image, then exit: drop the transformer before the decode.
    // SD_PREVIEW_EVERY=5 writes the model's running x0 estimate every 5 steps.
    // It needs a tiny decoder — a VAE decode per step costs more than the
    // step — so point SD_TAESD at `madebyollin/taesd3`.
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
