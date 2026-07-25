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

## Running the tests

```bash
cargo test -p sd-models --test golden_vae -- --nocapture
```

Golden tests **skip** when reference data is absent, so CI stays green without
committing hundreds of megabytes. Numerical verification is a local step.

## Tolerances

`atol = 1e-4`, `rtol = 1e-3` for f32.

Tighter and you chase phantom failures caused by accumulation order — candle
and PyTorch will not sum in the same sequence. Looser and real bugs slip
through. If a test fails at `1e-3`, that is a bug, not noise.

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
