# Roadmap

Ordered by **ops validated per unit of effort**, with visible output early.

## Milestone 1 — first correct image

| | Component | State |
|---|---|---|
| ✅ | Workspace, seam, CI, seam lint | done |
| ✅ | safetensors loading | done |
| ✅ | VAE decoder — structural tests | done |
| ✅ | VAE decoder — numerical vs `diffusers` | verified, max_abs 3.7e-5 |
| ⬜ | CLIP tokenizer (BPE) | |
| ⬜ | CLIP text encoder | |
| ⬜ | UNet | |
| ⬜ | Euler ancestral, DPM++ 2M | |
| ⬜ | `sdrs txt2img` | |

The VAE decoder is **numerically verified** against `diffusers` as of
2026-07-25: `max_abs = 3.678e-5`, `mean_abs = 4.408e-7` on the full 256x256
decode, against a tolerance of 1e-4. So the port is correct, and it is now a
usable oracle for changes to the ops underneath it.

Reproduce with `python3 xtask/golden/dump_reference.py vae --output tests/golden`
followed by `cargo test --release -p sd-models --test golden_vae`. The
references are ~460 MB and stay out of git, so this remains a local step.

Next concrete task: the CLIP tokenizer and text encoder, which is the last
piece before a UNet has anything to condition on.

## Milestone 2 — usable

- SDXL (same geometry, second text encoder, different latent scaling)
- img2img
- GGUF loading, then k-quant dequantization (`Q4_K`, `Q5_K`, `Q8_0` first —
  they cover most community models)
- Metal and CUDA paths through the seam

## Milestone 3 — breadth

SD 2.x · SD 3 · Flux (schnell, dev) · T5 text encoder · ControlNet · LoRA ·
TAESD · ESRGAN upscaling · inpainting

This is the phase that took upstream years, and it parallelizes well: each
architecture is independent and verifiable against its own golden data.

## Deliberately not doing

- **Training.** Inference only.
- **Our own GPU kernels — for now.** The seam makes this a per-kernel decision
  we can take later, when profiling says which one matters. Writing ~60 kernels
  per backend up front is 8–15 months before anything renders.
- **A GUI.** A good library first; someone else can build the UI.

## Highest-value optimisation available now

**Chunked / flash attention in `sd-tensor`.** Measured, not speculative: our
naive attention materialises a 4096x4096 score matrix at 512x512, which makes
Metal *slower than CPU* despite being 4-5x faster at smaller resolutions. Fully
behind the seam, so it needs no model code changes, and the existing golden
tests verify it. See [backends.md](backends.md) for the numbers.

## Upstream contributions worth making

- **candle: drop the `onig` C dependency.** One-line feature swap, verified to
  build and pass every test. See [native-deps.md](native-deps.md). Better
  still: make `tokenizers` optional in `candle-core` and feature-gate
  `quantized::tokenizer`, which most candle users do not need.

## Good first issues

- BPE tokenizer with a golden test against `transformers`
- Additional samplers (DDIM, Heun, LMS) against reference trajectories
- GGUF header parsing (metadata only, before dequantization)
- Chunked `scaled_dot_product_attention` that doesn't materialize the full
  score matrix — a self-contained win entirely behind the seam, and currently
  the biggest one. Benchmark it with
  `cargo run --release -p sd-cli --features metal --example backend_bench -- 64`
