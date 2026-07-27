# Golden-tensor harness

Reference outputs from `diffusers`, used to verify the Rust port numerically.

## Why

A port of a diffusion model fails *quietly*. A transposed axis or a wrong
epsilon does not panic — it yields an image that looks almost right. Debugging
that from the final image alone means bisecting several hundred tensor ops by
hand.

So we capture reference tensors **per module**. When a test fails, it names the
block that diverged first.

## Generating references

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch diffusers safetensors accelerate

python3 xtask/golden/dump_reference.py vae --output tests/golden
```

This downloads the SD 1.5 VAE (~330 MB) once and writes
`tests/golden/vae_decoder/reference.safetensors`.

Every component has its own subcommand — `unet_full`, `sd3`, `flux_transformer`,
`controlnet`, and the rest; `--help` lists them. Each writes a
`reference.safetensors` and symlinks the checkpoint it used next to it, so the
Rust test has one fixed path to open and no knowledge of the HuggingFace cache.

## Running the tests

```bash
cargo test -p sd-models --test golden_vae -- --nocapture
```

Golden tests **skip** when reference data is absent, so CI stays green without
committing hundreds of megabytes. Numerical verification is a local step.

## Tolerances

`atol = 1e-4`, `rtol = 1e-3` for f32 — but **`atol` alone is wrong wherever the
tensors are not order-1**, and several here are not.

Do not guess a bound. Measure what the reference does against *itself*:

```bash
python3 xtask/golden/reference_precision.py unet
python3 xtask/golden/reference_precision.py vae
```

That runs the same diffusers module in float32 and in float64 on identical
inputs. Neither run has a bug, so the gap between them is float32's own noise
floor for that computation, and a tolerance below it tests summation order
rather than this port. Measured:

| tensor | peak | max_abs | max_rel |
|---|---|---|---|
| `mid_output` | 16.169 | 1.108e-4 | 6.850e-6 |
| `down_11` | 19.219 | 1.083e-4 | 5.636e-6 |
| `encoder_moments` | 18.063 | 7.751e-5 | 4.291e-6 |
| `output` (UNet) | 3.889 | 9.700e-6 | 2.494e-6 |

**`mid_output` cannot be held to `atol = 1e-4`: diffusers misses that against
its own f64 by 1.108e-4.** That bound passed only because candle happened to
sum near PyTorch's order, and it broke the moment Apple's Accelerate reordered
it. `golden_unet.rs` and `golden_vae.rs` now use `atol + rtol*|want|` via
`testing::allclose_excess`, with the numbers above quoted where the constants
are set.

Both halves are needed: a relative term allows nothing where `want` is near
zero, and an absolute term alone is the bound that failed. Sensitivity of the
result, checked by perturbing the reference: passes at 0.1% (that is `rtol`),
fails at 0.2% and above, against a measured noise floor of 0.0007%. Real
porting bugs are far past that — the VAE's asymmetric-padding bug showed
17.32.

ControlNet is compared correction by correction — twelve for the skips plus
one for the mid block — rather than as a single tensor. It has no image of its
own to look at, so those thirteen tensors *are* its whole observable behaviour,
and the index of the first bad one localises the fault the way the UNet's skip
index does. A correct port lands at 1.45e-5 worst-case against a 1e-3 bound.

## Reading a failure

Check the intermediate tensors in order: `post_quant_conv`, `conv_in`,
`mid_block`, `up_block_0..3`, `conv_out`. **The first one that diverges is the
bug.** Everything downstream is just carrying the error forward.

Common causes, roughly in order of how often they occur:

| Symptom | Likely cause |
|---|---|
| Shapes mismatch | up-block channel order — they are the *reverse* of `block_out_channels` |
| Diverges at `mid_block` | attention axis order, or GroupNorm applied after flattening instead of before |
| Diverges at first `up_block` | resnet count — decoder uses `layers_per_block + 1`, not `layers_per_block` |
| Small uniform offset | GroupNorm epsilon (VAE uses `1e-6`, not the `1e-5` default) |
| Correct but blurry | upsample mode — must be nearest, not bilinear |
