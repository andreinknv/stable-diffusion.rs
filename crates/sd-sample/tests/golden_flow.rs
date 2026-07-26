//! Rectified flow against `diffusers`' `FlowMatchEulerDiscreteScheduler`.
//!
//! The step rule is two lines and hard to get wrong. The *schedule* is where
//! these implementations fail, because the resolution-dependent warp is easy
//! to write plausibly and slightly off, and a slightly-off schedule produces a
//! slightly-worse image rather than an error. Both resolutions are checked so
//! that hardcoding one shift cannot pass.
//!
//! Regenerate with:
//! `python3 xtask/golden/dump_reference.py flow --output tests/golden`

use std::path::PathBuf;

use sd_sample::flow::{flow_euler_step, flow_sigmas, flow_timesteps, scale_noise, FlowMatchConfig};
use sd_tensor::{testing, Device};

fn refs(dev: &Device) -> Option<std::collections::HashMap<String, sd_tensor::Tensor>> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/flow/reference.safetensors");
    if !p.exists() {
        eprintln!(
            "SKIP: no flow reference. Generate with \
             `python3 xtask/golden/dump_reference.py flow --output tests/golden`"
        );
        return None;
    }
    Some(sd_tensor::safetensors::load(p, dev).unwrap())
}

#[test]
fn sigma_schedule_matches_diffusers_at_both_resolutions() {
    let dev = Device::Cpu;
    let Some(r) = refs(&dev) else { return };
    let cfg = FlowMatchConfig::flux();

    for (label, seq_len) in [("1024tok", 1024usize), ("4096tok", 4096)] {
        let want_mu = r
            .get(&format!("mu_{label}"))
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0] as f64;
        let got_mu = cfg.mu(seq_len);
        assert!(
            (got_mu - want_mu).abs() < 1e-6,
            "{label}: mu {got_mu} vs diffusers {want_mu}"
        );

        let want: Vec<f64> = r
            .get(&format!("sigmas_{label}"))
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect();
        let got = flow_sigmas(&cfg, 20, seq_len);
        assert_eq!(got.len(), want.len(), "{label}: sigma count");

        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        eprintln!("{label}: mu {got_mu:.6}, worst sigma error {worst:.3e}");
        // f32 references, so this is the storage precision rather than ours.
        assert!(
            worst < 1e-6,
            "{label}: sigma schedule diverged by {worst:.3e}"
        );

        let want_t: Vec<f64> = r
            .get(&format!("timesteps_{label}"))
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect();
        let got_t = flow_timesteps(&cfg, &got);
        assert_eq!(got_t.len(), want_t.len(), "{label}: timestep count");
        let worst_t = got_t
            .iter()
            .zip(&want_t)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst_t < 1e-3,
            "{label}: timesteps diverged by {worst_t:.3e}"
        );
    }
}

#[test]
fn euler_step_matches_diffusers() {
    let dev = Device::Cpu;
    let Some(r) = refs(&dev) else { return };
    let cfg = FlowMatchConfig::flux();
    let sigmas = flow_sigmas(&cfg, 20, 4096);

    let i = r.get("step_index").unwrap().to_vec1::<f32>().unwrap()[0] as usize;
    let got = flow_euler_step(
        r.get("step_x").unwrap(),
        r.get("step_velocity").unwrap(),
        sigmas[i],
        sigmas[i + 1],
    )
    .unwrap();

    let c = testing::closeness(&got, r.get("step_out").unwrap()).unwrap();
    eprintln!("flow euler step: max_abs {:.3e}", c.max_abs);
    assert!(c.max_abs < 1e-5, "step diverged: {:.3e}", c.max_abs);
}

#[test]
fn scale_noise_matches_diffusers() {
    let dev = Device::Cpu;
    let Some(r) = refs(&dev) else { return };
    let cfg = FlowMatchConfig::flux();
    let sigmas = flow_sigmas(&cfg, 20, 4096);

    let i = r
        .get("scale_noise_index")
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()[0] as usize;
    let got = scale_noise(
        r.get("scale_noise_sample").unwrap(),
        r.get("scale_noise_noise").unwrap(),
        sigmas[i],
    )
    .unwrap();

    let c = testing::closeness(&got, r.get("scale_noise_out").unwrap()).unwrap();
    eprintln!("flow scale_noise: max_abs {:.3e}", c.max_abs);
    assert!(c.max_abs < 1e-5, "scale_noise diverged: {:.3e}", c.max_abs);
}
