//! Is our GELU more accurate than the one it replaces, or only different?
//!
//! Both use Abramowitz and Stegun 7.1.26. The difference is that candle forms
//! `1 + erf(u)` by subtraction, and ours reads `erfc` off the same polynomial
//! before any subtraction happens. The claim is that this matters in the left
//! tail. Truth is f64, computed with a series accurate well past f32.

use anyhow::Result;
use sd_tensor::{fused, ops, Device, Tensor};

/// erf in f64, by series below 2 and continued fraction above — far more
/// accurate than f32 needs, which is the point of a reference.
fn erf64(x: f64) -> f64 {
    let a = x.abs();
    if a < 2.0 {
        // Maclaurin: erf(x) = 2/sqrt(pi) * sum (-1)^n x^(2n+1) / (n!(2n+1))
        let mut term = a;
        let mut sum = a;
        for n in 1..60 {
            term *= -a * a / n as f64;
            sum += term / (2.0 * n as f64 + 1.0);
        }
        let v = sum * 2.0 / std::f64::consts::PI.sqrt();
        if x < 0.0 {
            -v
        } else {
            v
        }
    } else {
        // erfc via Lentz continued fraction, then erf = 1 - erfc.
        let mut f = 0.0f64;
        for k in (1..200).rev() {
            f = (k as f64 / 2.0) / (if k % 2 == 1 { a } else { 1.0 } + f);
        }
        let erfc = (-a * a).exp() / ((a + f) * std::f64::consts::PI.sqrt());
        let v = 1.0 - erfc;
        if x < 0.0 {
            -v
        } else {
            v
        }
    }
}

fn gelu64(x: f64) -> f64 {
    0.5 * x * (1.0 + erf64(x / std::f64::consts::SQRT_2))
}

fn main() -> Result<()> {
    let dev = sd_tensor::device::best()?;

    // The tail, where the subtraction bites.
    let xs: Vec<f32> = vec![
        -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.5, 2.0, 6.0,
    ];
    let n = xs.len();
    // The kernel takes [value | gate]; a unit value makes the output the
    // activation itself.
    let mut packed = vec![1.0f32; n * 2];
    packed[n..].copy_from_slice(&xs);
    let h = Tensor::from_vec(packed, (1, n * 2), &dev)?;

    let ours = fused::geglu(&h, n)?.flatten_all()?.to_vec1::<f32>()?;
    let theirs = ops::gelu(&Tensor::from_vec(xs.clone(), (1, n), &dev)?)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    println!("      x        f64 truth        candle          ours       who is closer");
    let (mut ours_wins, mut their_wins) = (0, 0);
    for (i, &x) in xs.iter().enumerate() {
        let truth = gelu64(x as f64);
        let (do_, dt) = (
            (ours[i] as f64 - truth).abs(),
            (theirs[i] as f64 - truth).abs(),
        );
        let verdict = if do_ < dt {
            ours_wins += 1;
            "ours"
        } else if dt < do_ {
            their_wins += 1;
            "candle"
        } else {
            "tie"
        };
        println!(
            "{x:7.2}  {truth:15.8e}  {:14.7e}  {:14.7e}   {verdict}",
            theirs[i], ours[i]
        );
    }
    println!("\nours closer at {ours_wins} points, candle closer at {their_wins}");

    // The specific failure: candle returns exactly zero once the subtraction
    // rounds erfc away, for every input past that point.
    let dead: Vec<f32> = xs
        .iter()
        .zip(&theirs)
        .filter(|(x, &t)| **x < 0.0 && t == 0.0)
        .map(|(x, _)| *x)
        .collect();
    if !dead.is_empty() {
        println!("candle returns exactly 0.0 at x = {dead:?}");
        for x in &dead {
            let i = xs.iter().position(|v| v == x).unwrap();
            println!(
                "  at x={x}: truth {:.3e}, ours {:.3e}",
                gelu64(*x as f64),
                ours[i]
            );
        }
    }
    let _ = Device::Cpu;
    Ok(())
}
