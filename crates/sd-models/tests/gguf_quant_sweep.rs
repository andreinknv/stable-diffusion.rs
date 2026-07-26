//! What each GGUF quantisation costs, measured against the same golden
//! references the safetensors path is held to.
//!
//! The f16 row is the control. Everything here shares one name mapping, so if
//! f16 agrees with the reference then the mapping is exact and every other
//! row is quantisation alone. Without that row, a poor Q4_0 result cannot be
//! told apart from a subtly wrong translation.

use std::path::PathBuf;

use sd_models::clip::{ClipTextConfig, ClipTextEncoder};
use sd_models::unet::{UNet2DConditionModel, UNetConfig};
use sd_models::vae::{AutoencoderKlDecoder, VaeConfig};
use sd_tensor::{testing, DType, Device, Tensor};

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(name)
}

/// Pearson correlation. Preferred to an absolute bound because a wrong
/// mapping lands near zero whatever its magnitude, while quantisation shifts
/// magnitude without destroying structure.
fn correlation(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let cov: f32 = a.iter().zip(&b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f32 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f32 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va.sqrt() * vb.sqrt())
}

#[test]
fn quantisation_cost_by_tower() {
    let dev = Device::Cpu;
    // Ordered by descending fidelity, not by file size — the k-quants are
    // produced locally by `sd-cli --example requantise` because no published
    // SD 1.5 carries them.
    let quants = [
        ("f16 ", golden("gguf/sd15-f16.gguf")),
        ("Q8_0", golden("gguf/sd15-q8_0.gguf")),
        ("Q6_K", golden("gguf/sd15-q6_k.gguf")),
        ("Q5_K", golden("gguf/sd15-q5_k.gguf")),
        ("Q4_K", golden("gguf/sd15-q4_k.gguf")),
        ("Q4_0", golden("gguf/sd15-q4_0.gguf")),
    ];
    let present: Vec<_> = quants.iter().filter(|(_, p)| p.exists()).collect();
    if present.is_empty() {
        eprintln!("SKIP: no gguf fixtures; see xtask/golden/README.md");
        return;
    }

    let vae_refs = sd_tensor::safetensors::load(golden("vae_decoder/reference.safetensors"), &dev);
    let clip_refs =
        sd_tensor::safetensors::load(golden("clip_encoder/reference.safetensors"), &dev);
    let unet_refs = sd_tensor::safetensors::load(golden("unet_full/reference.safetensors"), &dev);
    let (Ok(vae_refs), Ok(clip_refs), Ok(unet_refs)) = (vae_refs, clip_refs, unet_refs) else {
        eprintln!("SKIP: golden references missing");
        return;
    };

    eprintln!(
        "{:<6} {:>12} {:>12} {:>12}",
        "quant", "vae mean_abs", "unet corr", "clip corr"
    );
    let mut f16_worst: Option<f32> = None;
    let mut rows: Vec<(&str, f64, f32, f32)> = Vec::new();

    for (label, path) in present {
        let vb = sd_loader::vae_var_builder_from_gguf(path, DType::F32, &dev).unwrap();
        let dec = AutoencoderKlDecoder::new(&VaeConfig::sd15(), vb).unwrap();
        let img = dec.decode_raw(vae_refs.get("latent").unwrap()).unwrap();
        let vae_err = testing::closeness(&img, vae_refs.get("image").unwrap())
            .unwrap()
            .mean_abs;

        let vb = sd_loader::unet_var_builder_from_gguf(path, DType::F32, &dev).unwrap();
        let unet = UNet2DConditionModel::new(&UNetConfig::sd15(), vb).unwrap();
        let out = unet
            .forward(
                unet_refs.get("sample").unwrap(),
                unet_refs.get("timestep").unwrap(),
                unet_refs.get("context").unwrap(),
            )
            .unwrap();
        let unet_corr = correlation(&out, unet_refs.get("output").unwrap());

        let vb = sd_loader::clip_var_builder_from_gguf(path, DType::F32, &dev).unwrap();
        let clip = ClipTextEncoder::new(&ClipTextConfig::sd15(), vb).unwrap();
        let hidden = clip.forward(clip_refs.get("token_ids").unwrap()).unwrap();
        let clip_corr = correlation(&hidden, clip_refs.get("last_hidden_state").unwrap());

        eprintln!("{label:<6} {vae_err:>12.2e} {unet_corr:>12.4} {clip_corr:>12.4}");
        rows.push((label.trim(), vae_err, unet_corr, clip_corr));

        if label.trim() == "f16" {
            f16_worst = Some(unet_corr.min(clip_corr));
            // The control. f16 through this mapping must agree with the f32
            // reference almost exactly — anything less means the translation
            // is wrong, not the quantisation.
            assert!(
                unet_corr > 0.9999 && clip_corr > 0.9999,
                "f16 should be near-exact through the name map: unet {unet_corr}, clip {clip_corr}"
            );
            assert!(vae_err < 1e-3, "f16 vae decode drifted: {vae_err}");
        }
    }

    assert!(
        f16_worst.is_some(),
        "the f16 control is what makes the other rows interpretable; without it \
         a poor result cannot be told from a wrong mapping"
    );

    // The README and roadmap tell people to prefer Q4_K over Q4_0 at the same
    // bit width. Assert it so the recommendation cannot quietly go stale — a
    // broken k-quant path would otherwise still print a plausible table.
    let find = |q: &str| rows.iter().find(|r| r.0 == q).copied();
    if let (Some(k), Some(zero)) = (find("Q4_K"), find("Q4_0")) {
        assert!(
            k.2 > zero.2 && k.3 > zero.3 && k.1 < zero.1,
            "Q4_K should beat Q4_0 on every tower: \
             Q4_K (vae {:.2e}, unet {:.4}, clip {:.4}) vs Q4_0 (vae {:.2e}, unet {:.4}, clip {:.4})",
            k.1,
            k.2,
            k.3,
            zero.1,
            zero.2,
            zero.3
        );
    }

    // Fidelity must not increase as bits are removed. Checked pairwise down
    // the list, which catches a dtype silently falling back to F16 — that
    // would look "better" than the quantisation it claims to be.
    for pair in rows.windows(2) {
        let (hi, lo) = (pair[0], pair[1]);
        assert!(
            lo.2 <= hi.2 + 1e-4 && lo.3 <= hi.3 + 1e-4,
            "{} scored above {}, which has more bits per weight — \
             suspect a fallback to F16 rather than a genuine win",
            lo.0,
            hi.0
        );
    }
}
