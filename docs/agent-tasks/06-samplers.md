# Task 06 — Samplers

**Difficulty:** low · **Depends on:** nothing · **Read `AGENTS.md` first.**

Euler ancestral and DPM++ 2M. Pure numerics, no weights, no model. Can run in
parallel with Tasks 01–04.

---

## Files you may modify

```
crates/sd-sample/src/sigmas.rs      (new)
crates/sd-sample/src/euler.rs       (new)
crates/sd-sample/src/dpmpp.rs       (new)
crates/sd-sample/src/lib.rs         (add the new modules and exports)
xtask/golden/dump_reference.py      (add `samplers` subcommand)
crates/sd-sample/tests/golden_samplers.rs  (new)
```

## Files you must NOT modify

```
crates/sd-models/**
crates/sd-tensor/**
crates/sd-loader/**
any Cargo.toml
```

The existing `Schedule` type in `crates/sd-sample/src/lib.rs` already works and
has passing tests. **Use it; do not rewrite it.** You may add to that file but
not change `Schedule`, `BetaSchedule`, or their tests.

---

## Part A — sigma schedule for N steps

`Schedule::sigmas()` gives 1000 training sigmas. Sampling uses ~20. Add:

```rust
/// Select `n` sigmas from the 1000 training sigmas, plus a trailing 0.0.
///
/// Returns `n + 1` values, descending, ending at exactly 0.0.
/// This matches k-diffusion's `get_sigmas_karras(..., use_karras=False)`,
/// i.e. plain linear interpolation over the training sigmas.
pub fn sigmas_for_steps(schedule: &Schedule, n: usize) -> Vec<f64>;
```

Algorithm:

```
train = schedule.sigmas()                  // length 1000, ascending
step  = (train.len() - 1) as f64 / (n - 1) as f64
for i in 0..n:
    idx  = (n - 1 - i) as f64 * step       // descending: start at the END
    lo   = idx.floor() as usize
    hi   = min(lo + 1, train.len() - 1)
    frac = idx - lo as f64
    out.push(train[lo] * (1.0 - frac) + train[hi] * frac)
out.push(0.0)
```

Result is descending — high noise first — and the final entry is exactly `0.0`.

## Part B — Euler ancestral

```rust
/// One Euler-ancestral step.
///
/// `x`         current latent
/// `denoised`  the model's predicted x0 for this step
/// `sigma`     sigma at this step
/// `sigma_next` sigma at the next step (0.0 on the last step)
/// `noise`     standard normal, same shape as `x`. Ignored when sigma_next==0.
pub fn euler_ancestral_step(
    x: &Tensor,
    denoised: &Tensor,
    sigma: f64,
    sigma_next: f64,
    noise: &Tensor,
) -> Result<Tensor>;
```

```
sigma_up   = min(sigma_next,
                 sqrt(sigma_next^2 * (sigma^2 - sigma_next^2) / sigma^2))
sigma_down = sqrt(max(0, sigma_next^2 - sigma_up^2))

d    = (x - denoised) / sigma
x    = x + d * (sigma_down - sigma)
if sigma_next > 0:
    x = x + noise * sigma_up
return x
```

Guard `sigma == 0.0` before dividing.

## Part C — DPM++ 2M

Stateful: it needs the previous step's `denoised`.

```rust
#[derive(Debug, Default)]
pub struct DpmSolverPlusPlus2M {
    prev_denoised: Option<Tensor>,
}

impl DpmSolverPlusPlus2M {
    pub fn new() -> Self;

    /// Call once per step, in order. Reset between images.
    pub fn step(
        &mut self,
        x: &Tensor,
        denoised: &Tensor,
        sigma: f64,
        sigma_next: f64,
    ) -> Result<Tensor>;

    pub fn reset(&mut self);
}
```

```
t      = -ln(sigma)
t_next = -ln(sigma_next)          // sigma_next == 0 -> final step, see below
h      = t_next - t

if prev_denoised is None or sigma_next == 0:
    // first-order fallback
    x_next = (sigma_next / sigma) * x - (exp(-h) - 1) * denoised
else:
    h_last = t - t_prev
    r      = h_last / h
    d      = (1 + 1/(2r)) * denoised - (1/(2r)) * prev_denoised
    x_next = (sigma_next / sigma) * x - (exp(-h) - 1) * d

store denoised as prev_denoised, and t as t_prev
return x_next
```

When `sigma_next == 0.0`, `t_next` is `+inf`. Handle that case with the
first-order branch and return `denoised` directly rather than computing `h`.

---

## Reference data

`samplers` subcommand. Do **not** import k-diffusion — implement the same
formulas in numpy in the dump script so it stays dependency-free, and use fixed
inputs:

```python
sigmas = [14.6146, 10.0, 6.0, 3.0, 1.5, 0.5, 0.0]
x        = rng(seed 0).standard_normal((1, 4, 8, 8)).astype("float32")
denoised = rng(seed 1).standard_normal((1, 4, 8, 8)).astype("float32")
noise    = rng(seed 2).standard_normal((1, 4, 8, 8)).astype("float32")
```

Save `tests/golden/samplers/reference.safetensors`:

```
sigmas_20                [21]     sigmas_for_steps(sd15_schedule, 20)
x  denoised  noise
euler_step_0 .. euler_step_5      one per sigma pair
dpmpp_step_0 .. dpmpp_step_5      sequential, state carried between steps
```

Also emit the numpy formulas as comments in the script so the Rust and Python
sides are visibly the same equations.

---

## The test

```rust
// No data needed:
#[test] fn sigmas_for_steps_returns_n_plus_one_descending()
#[test] fn sigmas_for_steps_ends_at_zero()
#[test] fn euler_with_zero_sigma_next_adds_no_noise()
#[test] fn dpmpp_first_step_falls_back_to_first_order()
#[test] fn dpmpp_reset_clears_state()

// Skips if data absent:
#[test] fn sigmas_match_reference()
#[test] fn euler_steps_match_reference()
#[test] fn dpmpp_steps_match_reference()
```

`dpmpp_reset_clears_state` matters: reusing a solver across images without
reset silently corrupts the second image. Nothing else catches that.

---

## Verification

```bash
python3 xtask/golden/dump_reference.py samplers --output tests/golden

cargo test -p sd-sample -- --nocapture
./scripts/check-seam.sh
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --workspace
```

## Definition of done

- [ ] All eight tests pass at `atol = 1e-4`
- [ ] The three existing `Schedule` tests still pass, unmodified
- [ ] `crates/sd-models/` and `crates/sd-tensor/` untouched
- [ ] No `Cargo.toml` changed

---

## Known traps

**Sigmas descend.** Sampling starts at maximum noise. An ascending list runs
and produces pure noise output.

**The list is `n + 1` long.** N steps need N+1 boundaries. The trailing `0.0`
is a real entry, not a sentinel.

**`sigma_up` uses `min`.** Dropping the `min` gives values that are fine for
most steps and wrong near the end.

**DPM++ is stateful and order-dependent.** Steps must be called in sequence.
Calling out of order, or reusing state across images, produces subtly wrong
output that looks like a bad seed.

**`sigma_next == 0` on the final step.** `-ln(0)` is infinity. Branch before
computing, do not let it propagate as `inf` and hope.

**Everything is `f64` on the scalar side, `f32` in tensors.** Keep sigma
arithmetic in `f64`; only the tensor ops are `f32`.

---

## If you get blocked

Report `BLOCKED` naming the part (A sigmas, B euler, C dpmpp) and the step
index that first diverged. Part A failing means interpolation or ordering;
B means the `sigma_up`/`sigma_down` formulas; C means the state handling.

Do not modify tests. Do not touch the existing `Schedule` tests.
