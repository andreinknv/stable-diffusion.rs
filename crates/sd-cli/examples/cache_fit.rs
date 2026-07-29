//! What actually predicts how much the model's output will change?
//!
//! Step caching reuses a noise prediction while the model's output is not
//! moving much. It cannot know that without running the model, so it needs a
//! *predictor*: something cheap, computed before the forward pass, that
//! tracks the output change well enough to decide.
//!
//! The shipped predictor measures how far the **input latent** moved, and buys
//! about 9 % where the caching literature reports 1.5-2x. TeaCache's answer is
//! to predict from the **timestep embedding** instead, rescaled through a
//! polynomial fitted per model.
//!
//! This example does not assume any of that. It runs a generation with caching
//! off and records, per step, three candidate predictors alongside the thing
//! they are supposed to predict, then fits and scores each one. The output is
//! the polynomial to paste into the pipeline — measured on this model, on this
//! machine.
//!
//! # Why fit rather than borrow
//!
//! Because this project has already been burned twice by constants taken from
//! a paper without checking what they were constants *of* — once by
//! AnimateDiff's beta schedule, and once by this very feature, whose
//! "useful band" of 0.05-0.15 was TeaCache's number for TeaCache's rescaled
//! metric and had nothing to do with the metric actually implemented. A
//! polynomial fitted here cannot be wrong about which quantity it maps.
//!
//! ```bash
//! cargo run --release -p sd-cli --example cache_fit -- <model-dir> [steps]
//! ```

use anyhow::{Context, Result};
use stable_diffusion_rs as sd;

use sd::pipeline::{SamplerKind, Txt2ImgConfig, Txt2ImgPipeline};

/// Least-squares polynomial fit of `y` on `x`, by normal equations.
///
/// Degree 4, matching TeaCache. Solved with Gaussian elimination on a 5x5 —
/// small enough that conditioning is not a concern at these magnitudes, and
/// this is a one-off calibration rather than anything on a hot path.
fn polyfit(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let n = degree + 1;
    let mut ata = vec![vec![0f64; n + 1]; n];
    for (&xi, &yi) in x.iter().zip(y) {
        let powers: Vec<f64> = (0..n).map(|p| xi.powi(p as i32)).collect();
        for r in 0..n {
            for c in 0..n {
                ata[r][c] += powers[r] * powers[c];
            }
            ata[r][n] += powers[r] * yi;
        }
    }
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&a, &b| ata[a][col].abs().total_cmp(&ata[b][col].abs()))
            .unwrap_or(col);
        ata.swap(col, pivot);
        let d = ata[col][col];
        if d.abs() < 1e-18 {
            continue;
        }
        ata[col].iter_mut().skip(col).for_each(|v| *v /= d);

        for r in 0..n {
            if r == col {
                continue;
            }
            let f = ata[r][col];
            let pivot_row: Vec<f64> = ata[col][col..=n].to_vec();
            ata[r][col..=n]
                .iter_mut()
                .zip(&pivot_row)
                .for_each(|(v, p)| *v -= f * p);
        }
    }
    (0..n).map(|r| ata[r][n]).collect()
}

fn apply(coeffs: &[f64], x: f64) -> f64 {
    coeffs
        .iter()
        .enumerate()
        .map(|(p, c)| c * x.powi(p as i32))
        .sum()
}

/// Mean absolute error of a fit, in the units of the thing predicted.
fn score(coeffs: &[f64], x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().max(1) as f64;
    x.iter()
        .zip(y)
        .map(|(&xi, &yi)| (apply(coeffs, xi) - yi).abs())
        .sum::<f64>()
        / n
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| "models/sd15".to_string());
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    // The sampler matters more than it looks: an ancestral one injects fresh
    // noise every step, which decorrelates consecutive inputs and so
    // consecutive outputs. Whether caching has anything to exploit may be a
    // property of the sampler rather than of the model.
    let sampler = match args.next().as_deref() {
        Some("dpmpp2m") => SamplerKind::DpmPlusPlus2M,
        Some("lcm") => SamplerKind::Lcm,
        _ => SamplerKind::EulerAncestral,
    };

    let dev = sd_tensor::device::best()?;
    tracing::info!(%model, steps, ?sampler, ?dev, "fitting a step-cache predictor");
    let pipeline = Txt2ImgPipeline::load(std::path::Path::new(&model), &dev)
        .with_context(|| format!("loading {model}"))?;

    // Several prompts, because a predictor fitted to one image is fitted to
    // that image. The polynomial has to serve whatever is generated next.
    let prompts = [
        "a photograph of a crab on a beach",
        "an oil painting of a city at night",
        "a diagram of a bicycle, technical drawing",
    ];

    let mut latent_moves = Vec::new();
    let mut temb_moves = Vec::new();
    let mut output_moves = Vec::new();

    for (p, prompt) in prompts.iter().enumerate() {
        let cfg = Txt2ImgConfig {
            prompt: (*prompt).into(),
            steps,
            seed: 100 + p as u64,
            sampler,
            ..Default::default()
        };
        let series = pipeline
            .cache_calibration(&cfg)
            .with_context(|| format!("calibrating on {prompt:?}"))?;
        tracing::info!(prompt, points = series.len(), "recorded");
        for point in series {
            latent_moves.push(point.latent);
            temb_moves.push(point.temb);
            output_moves.push(point.output);
        }
    }

    println!(
        "\n{} points over {} prompts\n",
        output_moves.len(),
        prompts.len()
    );
    println!(
        "  {:>10}  {:>12}  {:>12}  {:>12}",
        "step", "latent", "temb", "output"
    );
    for (i, ((l, t), o)) in latent_moves
        .iter()
        .zip(&temb_moves)
        .zip(&output_moves)
        .enumerate()
        .take(steps.saturating_sub(1))
    {
        println!("  {i:>10}  {l:>12.6}  {t:>12.6}  {o:>12.6}");
    }

    let lo = output_moves.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = output_moves.iter().cloned().fold(0f64, f64::max);
    println!("\noutput change spans {lo:.4} to {hi:.4} — caching can only pay where this is small");

    for (name, xs) in [("latent", &latent_moves), ("temb", &temb_moves)] {
        let coeffs = polyfit(xs, &output_moves, 4);
        let err = score(&coeffs, xs, &output_moves);
        // Against predicting the mean, which is what a useless predictor
        // achieves. A fit that cannot beat this is not a predictor.
        let mean = output_moves.iter().sum::<f64>() / output_moves.len().max(1) as f64;
        let baseline = output_moves.iter().map(|o| (o - mean).abs()).sum::<f64>()
            / output_moves.len().max(1) as f64;
        println!("\n{name}: mean |error| {err:.6}, against {baseline:.6} for predicting the mean");
        println!(
            "  [{}]",
            coeffs
                .iter()
                .map(|c| format!("{c:.6e}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(())
}
