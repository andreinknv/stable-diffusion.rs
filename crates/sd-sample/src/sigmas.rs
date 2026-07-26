//! Selecting a sampling schedule from the 1000 training sigmas.

use super::Schedule;

/// Select `n` sigmas from the training sigmas, plus a trailing `0.0`.
///
/// Returns `n + 1` values, **descending**, ending at exactly `0.0`. Matches
/// k-diffusion's `get_sigmas_karras(..., use_karras=False)` — plain linear
/// interpolation over the training sigmas.
///
/// Two things about the shape of the output are load-bearing. It descends,
/// because sampling starts at maximum noise and an ascending list runs happily
/// and returns noise. And it is `n + 1` long, because `n` steps need `n + 1`
/// boundaries — the trailing zero is a real entry, not a sentinel.
pub fn sigmas_for_steps(schedule: &Schedule, n: usize) -> Vec<f64> {
    let train = schedule.sigmas();
    let mut out = Vec::with_capacity(n + 1);
    if train.is_empty() || n == 0 {
        out.push(0.0);
        return out;
    }
    let last = train.len() - 1;
    // With n == 1 there is no interval to divide; take the highest sigma.
    let step = if n > 1 {
        last as f64 / (n - 1) as f64
    } else {
        0.0
    };

    for i in 0..n {
        // Count down, so the first emitted sigma is the noisiest.
        let idx = (n - 1 - i) as f64 * step;
        let lo = idx.floor() as usize;
        let hi = (lo + 1).min(last);
        let frac = idx - lo as f64;
        out.push(train[lo] * (1.0 - frac) + train[hi] * frac);
    }
    out.push(0.0);
    out
}
