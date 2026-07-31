//! unCLIP's prior on MLX, against `tests/golden/unclip/prior_reference`.
//!
//! Four things, each failing for a reason the others cannot see.
//!
//! - **The transformer**, twice: once with the tokenizer's real attention mask
//!   and once with every position unmasked. The reference's prompt occupies 10
//!   of 77 positions and the two predictions differ by 0.60, so a port that
//!   ignores the mask agrees with exactly one of them. That matters here and
//!   nowhere else in this project — Stable Diffusion conditions on all 77
//!   positions, padding included, so ignoring the mask is the *habit*
//!   everywhere else in this codebase.
//! - **The step**, at an interior timestep with the noise draw pinned, and at
//!   the final one where no variance is added at all. The second is fully
//!   deterministic and pins the mean; the first pins the variance on top of it.
//! - **The standard deviations** the step multiplies its noise by.
//!   `fixed_small_log` returns a *deviation* where every other variance type
//!   returns a variance, so squaring or rooting it once more is wrong by
//!   exactly that much and still returns a plausible embedding.
//! - **`post_process`**, the un-whitening the image half depends on.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_golden_prior -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::prior::{self, PriorConfig};
use sd_models::prior::PriorScheduler;
use sd_tensor::mlx::{load_safetensors, Array, Stream};

/// `golden_prior.rs`'s bounds and its reasoning: the prior is 20 blocks over 81
/// tokens at 2048 wide, so it accumulates like any deep stack.
const PRIOR_RTOL: f32 = 1e-3;
const PRIOR_TOL: f32 = 1e-3;
/// One closed-form step on order-1 numbers.
const STEP_TOL: f32 = 1e-5;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unclip")
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let (refs, w) = (
        golden().join("prior_reference.safetensors"),
        golden().join("t2i_prior.safetensors"),
    );
    if !refs.exists() || !w.exists() {
        return None;
    }
    Some((
        load_safetensors(&refs).expect("reference"),
        load_safetensors(&w).expect("weights"),
    ))
}

/// The first element of a fixture tensor, whatever integer type it was dumped
/// as. Timesteps arrive as int64 and `to_vec_f32` refuses those outright rather
/// than reinterpreting the bits — which is the right refusal.
fn scalar(refs: &HashMap<String, Array>, key: &str, s: &Stream) -> f32 {
    refs.get(key)
        .unwrap_or_else(|| panic!("{key}"))
        .to_f32(s)
        .expect("cast")
        .to_vec_f32(s)
        .expect("read")[0]
}

/// Worst violation of `|a - b| <= atol + rtol * |b|`.
fn excess(got: &Array, want: &Array, s: &Stream, what: &str) -> f32 {
    let g = got.to_vec_f32(s).expect("got");
    let w = want.to_vec_f32(s).expect("want");
    assert_eq!(g.len(), w.len(), "{what}: element count");
    let (mut peak, mut worst, mut exc) = (0.0f32, 0.0f32, 0.0f32);
    for (a, b) in g.iter().zip(&w) {
        let d = (a - b).abs();
        worst = worst.max(d);
        peak = peak.max(b.abs());
        exc = exc.max(d - PRIOR_RTOL * b.abs());
    }
    let exc = exc.max(0.0);
    eprintln!("{what:<18} peak {peak:>7.3}  max_abs {worst:.3e}  excess {exc:.3e}");
    exc
}

/// The timestep the reference forward was dumped at. Not `step_timestep`,
/// which belongs to the *scheduler* fixtures below — the two are different
/// numbers and feeding one where the other belongs gives a well-shaped
/// prediction that matches nothing.
const FORWARD_TIMESTEP: f32 = 500.0;

fn run(
    refs: &HashMap<String, Array>,
    w: &HashMap<String, Array>,
    masked: bool,
    s: &Stream,
) -> Array {
    let cfg = PriorConfig::karlo();
    let mask = masked.then(|| refs.get("prior_mask").expect("prior_mask"));
    let timestep = Array::from_slice_f32(&[FORWARD_TIMESTEP], &[1]).unwrap();
    prior::forward(
        refs.get("prior_latents").expect("prior_latents"),
        &timestep,
        refs.get("text_embeds").expect("text_embeds"),
        refs.get("text_hidden").expect("text_hidden"),
        mask,
        &cfg,
        w,
        s,
    )
    .expect("prior forward")
}

/// **With the tokenizer's mask**, which is what the pipeline uses.
#[test]
fn the_prior_matches_diffusers_with_the_text_mask() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no unCLIP prior fixture.");
        return;
    };
    let s = Stream::gpu();
    let got = run(&refs, &w, true, &s);
    assert_eq!(got.shape(), vec![1, 768]);
    let e = excess(
        &got,
        refs.get("prior_out").expect("prior_out"),
        &s,
        "prior_out",
    );
    assert!(e <= PRIOR_TOL, "the masked prior is {e:.3e} out");
}

/// **And without it**, which is the same model reading padding as content.
///
/// Both are checked because agreeing with one is not evidence of agreeing with
/// the other: they differ by 0.60 on this prompt, so a port that drops the mask
/// passes the unmasked comparison and fails nothing else.
#[test]
fn the_prior_matches_diffusers_unmasked() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no unCLIP prior fixture.");
        return;
    };
    let s = Stream::gpu();
    let got = run(&refs, &w, false, &s);
    let want = refs.get("prior_out_unmasked").expect("prior_out_unmasked");
    let e = excess(&got, want, &s, "prior_out_unmasked");
    assert!(e <= PRIOR_TOL, "the unmasked prior is {e:.3e} out");

    // The two must genuinely differ, or neither comparison says anything.
    let masked = run(&refs, &w, true, &s);
    let (a, b) = (masked.to_vec_f32(&s).unwrap(), got.to_vec_f32(&s).unwrap());
    let spread = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("masked vs unmasked differ by {spread:.3}");
    assert!(
        spread > 0.1,
        "the mask changed the prediction by only {spread:.3e}; it is not reaching the \
         attention"
    );
}

/// One interior step, noise pinned, and the final step where no variance is
/// added at all.
#[test]
fn the_ddpm_step_matches_diffusers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no unCLIP prior fixture.");
        return;
    };
    let _ = w;
    let s = Stream::gpu();
    let sched = PriorScheduler::new(25);

    let t = scalar(&refs, "step_timestep", &s) as usize;
    let c = sched.coefficients(t).expect("coefficients");
    assert!(c.std.is_some(), "an interior step adds variance");
    let got = prior::step(
        refs.get("prior_out").expect("prior_out"),
        refs.get("prior_latents").expect("prior_latents"),
        refs.get("step_noise").expect("step_noise"),
        c,
        &s,
    )
    .expect("step");
    let want = refs.get("stepped").expect("stepped");
    let (a, b) = (got.to_vec_f32(&s).unwrap(), want.to_vec_f32(&s).unwrap());
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("step               max_abs {worst:.3e}  tol {STEP_TOL:.0e}");
    assert!(worst <= STEP_TOL, "the interior step is {worst:.3e} out");

    // The final step, which must add no variance whatever the noise is.
    let tf = scalar(&refs, "step_timestep_final", &s) as usize;
    let cf = sched.coefficients(tf).expect("coefficients");
    assert!(cf.std.is_none(), "the final step adds no variance");
    let got = prior::step(
        refs.get("prior_out").unwrap(),
        refs.get("prior_latents").unwrap(),
        refs.get("step_noise").unwrap(),
        cf,
        &s,
    )
    .expect("final step");
    let want = refs.get("stepped_final").expect("stepped_final");
    let (a, b) = (got.to_vec_f32(&s).unwrap(), want.to_vec_f32(&s).unwrap());
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("step (final)       max_abs {worst:.3e}  tol {STEP_TOL:.0e}");
    assert!(worst <= STEP_TOL, "the final step is {worst:.3e} out");
}

/// The standard deviations themselves, straight from `_get_variance`.
#[test]
fn the_step_deviations_match_diffusers() {
    let Some((refs, _)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no unCLIP prior fixture.");
        return;
    };
    let s = Stream::gpu();
    let sched = PriorScheduler::new(25);
    let ts = refs
        .get("probe_timesteps")
        .expect("probe_timesteps")
        .to_f32(&s)
        .unwrap()
        .to_vec_f32(&s)
        .unwrap();
    let want = refs
        .get("probe_stds")
        .expect("probe_stds")
        .to_vec_f32(&s)
        .unwrap();
    assert_eq!(ts.len(), want.len());

    for (t, expected) in ts.iter().zip(&want) {
        let c = sched.coefficients(*t as usize).expect("coefficients");
        let got = c.std.unwrap_or(0.0) as f32;
        assert!(
            (got - expected).abs() <= STEP_TOL,
            "at t={t}: std {got} against {expected}; `fixed_small_log` returns a \
             deviation, not a variance"
        );
    }
}

/// `post_process` un-whitens into the units the image half expects.
#[test]
fn post_process_un_whitens_with_the_checkpoints_statistics() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no unCLIP prior fixture.");
        return;
    };
    let s = Stream::gpu();
    let latents = refs.get("prior_latents").expect("prior_latents");
    let got = prior::post_process(latents, &w, &s).expect("post_process");

    // Computed directly from the fixture's own statistics rather than from a
    // dumped result: this is `x * std + mean` and nothing else, so restating it
    // is the check.
    let (x, mean, std) = (
        latents.to_vec_f32(&s).unwrap(),
        refs.get("clip_mean")
            .expect("clip_mean")
            .to_vec_f32(&s)
            .unwrap(),
        refs.get("clip_std")
            .expect("clip_std")
            .to_vec_f32(&s)
            .unwrap(),
    );
    let g = got.to_vec_f32(&s).unwrap();
    let mut worst = 0.0f32;
    for i in 0..x.len() {
        worst = worst.max((g[i] - (x[i] * std[i] + mean[i])).abs());
    }
    eprintln!("post_process       max_abs {worst:.3e}");
    assert!(worst <= STEP_TOL, "post_process is {worst:.3e} out");

    // And it is not the identity — skipping it returns the right shape at the
    // wrong scale, which is the failure it exists to prevent.
    let spread = x
        .iter()
        .zip(&g)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        spread > 1e-3,
        "post_process moved nothing ({spread:.3e}); the statistics are not being applied"
    );
}
