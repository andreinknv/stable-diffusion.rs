# Roadmap

Ordered by **ops validated per unit of effort**, with visible output early.

## Milestone 1 — first correct image

| | Component | State |
|---|---|---|
| ✅ | Workspace, seam, CI, seam lint | done |
| ✅ | safetensors loading | done |
| ✅ | VAE decoder — structural tests | done |
| ✅ | VAE decoder — numerical vs `diffusers` | verified, max_abs 3.7e-5 |
| ✅ | CLIP tokenizer (BPE) | verified id-for-id vs HuggingFace |
| ✅ | CLIP text encoder | verified layer by layer vs `transformers` |
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

The CLIP tokenizer matches HuggingFace id for id, and the text encoder matches
`transformers` layer by layer. Both halves of the conditioning path are done.

Next concrete task: the **UNet** (docs/agent-tasks/03 through 05), which is the
large one — build it block by block and verify each against golden data rather
than waiting for a whole-model comparison.

A note for whoever verifies it. CLIP's activations peak at 851 and f32 cannot
hold 1e-4 absolute at that magnitude, so `golden_clip_encoder.rs` compares with
`|a-b| <= atol + rtol*|b|`. Expect to need the same for any tensor whose values
leave the order-1 range; `testing::assert_close` is absolute-only.

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

**A fused attention kernel for Metal.** Our attention materialises a 4096x4096
score matrix at 512x512, which makes Metal *slower than CPU* despite being
4-5x faster at smaller resolutions.

Chunked attention (`ops::chunked_attention`) is **done** and bounds the memory,
which is what makes larger latents reachable at all — but it does not close the
speed gap, and measurement could not distinguish it from unchunked on a noisy
machine. Closing the gap needs softmax fused into the matmul so the score
matrix never reaches memory, which candle does not expose for our shapes. That
is a hand-written kernel, and it is the remaining work. See
[backends.md](backends.md) for what was measured and what was not.

## Upstream contributions worth making

- **candle: drop the `onig` C dependency.** One-line feature swap, verified to
  build and pass every test. See [native-deps.md](native-deps.md). Better
  still: make `tokenizers` optional in `candle-core` and feature-gate
  `quantized::tokenizer`, which most candle users do not need.

## Good first issues

- BPE tokenizer with a golden test against `transformers`
- Additional samplers (DDIM, Heun, LMS) against reference trajectories
- GGUF header parsing (metadata only, before dequantization)
- A repeatable benchmark harness for `backend_bench`: run each configuration N
  times and report a median and spread. Single runs on a busy machine vary by
  ±40%, which is how the chunk-size question ended up unanswerable. Small,
  self-contained, and it unblocks every performance claim after it.
