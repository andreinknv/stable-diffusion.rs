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
| ✅ | UNet | verified: 12 skips, mid block, output vs `diffusers` |
| ✅ | Euler ancestral, DPM++ 2M | verified vs numpy reference |
| 🔴 | `sdrs txt2img` | works; stock VAE needs a key rename in sd-loader |

The VAE decoder is **numerically verified** against `diffusers` as of
2026-07-25: `max_abs = 3.678e-5`, `mean_abs = 4.408e-7` on the full 256x256
decode, against a tolerance of 1e-4. So the port is correct, and it is now a
usable oracle for changes to the ops underneath it.

Reproduce with `python3 xtask/golden/dump_reference.py vae --output tests/golden`
followed by `cargo test --release -p sd-models --test golden_vae`. The
references are ~460 MB and stay out of git, so this remains a local step.

The CLIP tokenizer matches HuggingFace id for id, and the text encoder matches
`transformers` layer by layer. Both halves of the conditioning path are done.

The UNet is done and verified end to end (tasks 03-05): timestep embedding,
time-conditioned resnets, the spatial transformer, and the assembly, each
checked against `diffusers` at `atol = 1e-4` — including all twelve skip
tensors individually, which is what makes a failure localizable across 25
blocks.

Milestone 1 is functionally complete: `sdrs txt2img` produces a real image —
see `assets/crab-512-dpmpp2m-seed42.png`, 512x512, 20 steps, DPM++ 2M, seed 42,
113 s on CPU. The same seed twice gives byte-identical PNGs.

One thing stands between that and "done", and it is small:

**Stock SD 1.5 VAE weights will not load.** They use the legacy diffusers
attention names (`query`/`key`/`value`/`proj_attn`) where the decoder expects
`to_q`/`to_k`/`to_v`/`to_out.0`. The golden VAE test does not catch it because
its reference is exported through `vae.state_dict()`, which diffusers renames
on load. The fix is a key-conversion map in **sd-loader** — conversion belongs
there, not in the model code — plus a golden test that loads a *raw* checkpoint
rather than a re-exported one. Until then `txt2img` needs a re-exported VAE.

A stock download also has no `tokenizer/tokenizer.json`; the repository ships
vocab.json + merges.txt. Copy it from `openai/clip-vit-large-patch14`. The
pipeline's error message says so.

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

- **Legacy VAE key conversion in sd-loader** (see milestone 1). Small, and it
  is the last thing between `sdrs txt2img` and working on an unmodified
  download. Pair it with a golden test that loads a raw checkpoint rather than
  a `state_dict()` re-export, since that is precisely what hid the problem.
- Additional samplers (DDIM, Heun, LMS) against reference trajectories
- GGUF header parsing (metadata only, before dequantization)
- A repeatable benchmark harness for `backend_bench`: run each configuration N
  times and report a median and spread. Single runs on a busy machine vary by
  ±40%, which is how the chunk-size question ended up unanswerable. Small,
  self-contained, and it unblocks every performance claim after it.
