//! The MLX samplers against the candle ones, step for step.
//!
//! `sd-sample`'s samplers are already gated on `tests/golden/samplers`, so
//! agreeing with them is agreeing with the reference — and that fixture is f64,
//! which MLX cannot load, so this is also the only route to it.
//!
//! Both sides are driven with the same tensors and the same sigma ladder, so a
//! difference is the arithmetic and nothing else.
//!
//! ```bash
//! cargo test -p stable-diffusion-rs --features mlx --test mlx_samplers_agree
//! ```
#![cfg(feature = "mlx")]

use sd_models::mlx::sample as mlx_sample;
use sd_sample::{sigmas_for_steps, DpmSolverPlusPlus2M, Schedule};
use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{Device, Tensor};

/// f32 arithmetic in a different order on a different device; this is the
/// project's documented float32 tolerance rather than a bound chosen to pass.
const TOL: f32 = 1e-5;

fn ramp(n: usize, scale: f32, shift: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.31).sin() * scale + shift)
        .collect()
}

fn candle_vec(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: element count");
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{what:<28} max_abs {worst:.3e}");
    assert!(worst <= TOL, "{what}: {worst:.3e} exceeds {TOL:.0e}");
}

#[test]
fn euler_ancestral_agrees_with_candle() {
    let shape = [1usize, 4, 8, 8];
    let n: usize = shape.iter().product();
    let (x, den, noise) = (ramp(n, 3.0, 0.2), ramp(n, 2.0, -0.4), ramp(n, 1.0, 0.0));
    let s = Stream::gpu();
    let dev = Device::Cpu;

    let schedule = Schedule::sd15();
    let sigmas = sigmas_for_steps(&schedule, 6);

    for i in 0..6 {
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);

        let want = sd_sample::euler_ancestral_step(
            &Tensor::from_vec(x.clone(), shape.as_slice(), &dev).unwrap(),
            &Tensor::from_vec(den.clone(), shape.as_slice(), &dev).unwrap(),
            sigma,
            sigma_next,
            &Tensor::from_vec(noise.clone(), shape.as_slice(), &dev).unwrap(),
        )
        .unwrap();

        let got = mlx_sample::euler_ancestral_step(
            &Array::from_slice_f32(&x, &shape).unwrap(),
            &Array::from_slice_f32(&den, &shape).unwrap(),
            sigma,
            sigma_next,
            &Array::from_slice_f32(&noise, &shape).unwrap(),
            &s,
        )
        .unwrap();

        assert_close(
            &got.to_vec_f32(&s).unwrap(),
            &candle_vec(&want),
            &format!("euler step {i} (sigma {sigma:.3})"),
        );
    }
}

/// DPM++ carries state, so this drives a whole ladder rather than one step —
/// a first-order fallback that never hands over to the second-order branch
/// would pass a single-step test and fail here.
#[test]
fn dpmpp_2m_agrees_with_candle_across_a_ladder() {
    let shape = [1usize, 4, 8, 8];
    let n: usize = shape.iter().product();
    let s = Stream::gpu();
    let dev = Device::Cpu;

    let schedule = Schedule::sd15();
    let sigmas = sigmas_for_steps(&schedule, 8);

    let mut candle_solver = DpmSolverPlusPlus2M::new();
    let mut mlx_solver = mlx_sample::DpmSolverPlusPlus2M::new();

    let mut x_c = Tensor::from_vec(ramp(n, 3.0, 0.1), shape.as_slice(), &dev).unwrap();
    let mut x_m = Array::from_slice_f32(&ramp(n, 3.0, 0.1), &shape).unwrap();

    for i in 0..8 {
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        // A different prediction each step, so the second-order extrapolation
        // actually has something to extrapolate from.
        let den = ramp(n, 2.0 + i as f32 * 0.3, -0.2);

        x_c = candle_solver
            .step(
                &x_c,
                &Tensor::from_vec(den.clone(), shape.as_slice(), &dev).unwrap(),
                sigma,
                sigma_next,
            )
            .unwrap();
        x_m = mlx_solver
            .step(
                &x_m,
                &Array::from_slice_f32(&den, &shape).unwrap(),
                sigma,
                sigma_next,
                &s,
            )
            .unwrap();

        assert_close(
            &x_m.to_vec_f32(&s).unwrap(),
            &candle_vec(&x_c),
            &format!("dpm++ step {i} (sigma {sigma:.3})"),
        );
    }
}

/// `reset` really discards the carry: without it the second image inherits the
/// first's `prev_denoised` and drifts in a way that reads as a bad seed.
#[test]
fn reset_discards_the_carried_state() {
    let shape = [1usize, 4, 4, 4];
    let n: usize = shape.iter().product();
    let s = Stream::gpu();
    let sigmas = sigmas_for_steps(&Schedule::sd15(), 4);

    // **The prediction must change between steps.** With a constant `denoised`
    // the second-order extrapolation is `den*(1 + inv) - prev*inv`, and with
    // `prev == den` that is exactly `den` — identical to the first-order
    // fallback. A constant here makes the carry unobservable and the test
    // vacuous, which is what the assertion below caught the first time.
    let run = |solver: &mut mlx_sample::DpmSolverPlusPlus2M| -> Vec<f32> {
        let mut x = Array::from_slice_f32(&ramp(n, 2.0, 0.0), &shape).unwrap();
        for i in 0..3 {
            let den = Array::from_slice_f32(&ramp(n, 1.5 + i as f32 * 0.4, 0.1), &shape).unwrap();
            x = solver.step(&x, &den, sigmas[i], sigmas[i + 1], &s).unwrap();
        }
        x.to_vec_f32(&s).unwrap()
    };

    let mut solver = mlx_sample::DpmSolverPlusPlus2M::new();
    let first = run(&mut solver);
    let carried = run(&mut solver);
    solver.reset();
    let after_reset = run(&mut solver);

    assert_eq!(first, after_reset, "reset must reproduce a fresh solver");
    assert_ne!(
        first, carried,
        "without reset the carry must change the result, or this test proves nothing"
    );
}
