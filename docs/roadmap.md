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
| ✅ | `sdrs txt2img` | works on a stock SD 1.5 download |

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

It runs on an unmodified download: sd-loader converts the legacy VAE attention
key names (`query`/`key`/`value`/`proj_attn`) that stock SD 1.5 uses. That bug
survived a fully green golden suite because the VAE reference weights come from
`vae.state_dict()`, and diffusers renames legacy keys on load — so the test had
only ever seen a re-export. `golden_vae_legacy.rs` now loads the raw file.
**Generalise the lesson: verify against a published artifact, not one your own
tooling round-tripped.**

One rough edge remains. A stock download has no `tokenizer/tokenizer.json` —
the repository ships vocab.json + merges.txt — so copy it from
`openai/clip-vit-large-patch14`. The pipeline's error says so. Building a CLIP
tokenizer from vocab+merges directly means reconstructing the normalizer,
pre-tokenizer and post-processor by hand, which is the sort of subtle-mismatch
surface this project verifies rather than guesses at; it is worth doing
properly, with a golden test, rather than quickly.

A note for whoever verifies it. CLIP's activations peak at 851 and f32 cannot
hold 1e-4 absolute at that magnitude, so `golden_clip_encoder.rs` compares with
`|a-b| <= atol + rtol*|b|`. Expect to need the same for any tensor whose values
leave the order-1 range; `testing::assert_close` is absolute-only.

## Milestone 2 — usable

| | Component | State |
|---|---|---|
| ✅ | VAE encoder | verified vs `diffusers`, max_abs 8.2e-5 |
| ✅ | img2img | works; strength verified visually at 0.35 and 0.75 |
| ✅ | SDXL — text encoder 2 | verified vs `transformers` |
| ✅ | SDXL — UNet | verified vs `diffusers`, max_abs 1.4e-5 |
| 🔴 | SDXL — end to end | correct on CPU; Metal corrupts the 1024 decode |
| ⬜ | GGUF loading, then k-quants | |
| ⬜ | Metal and CUDA paths through the seam | Metal verified end to end |

The VAE encoder's downsampler pads **asymmetrically** (bottom and right only).
A symmetric `padding: 1` gives the right shape and a half-pixel shift per
level — 17.32 max_abs against a correct 8.2e-5. Worth knowing before writing
any other downsampling path.

Remaining milestone 2 work:

- SDXL (same geometry, second text encoder, different latent scaling)
- GGUF loading, then k-quant dequantization (`Q4_K`, `Q5_K`, `Q8_0` first —
  they cover most community models)
- CUDA through the seam. Metal is verified end to end at 512; CUDA is untested.
- **The Metal 1024 decode defect** (see backends.md). Either find it in candle
  and fix upstream, or run the decode on CPU above the known-good latent size.
  It is the only thing between SDXL and working on GPU.

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

- **Load the CLIP tokenizer from vocab.json + merges.txt**, so a stock SD 1.5
  download works with no manual file copying. Needs a golden test against the
  existing tokenizer.json path — the two must agree id for id on
  `xtask/golden`'s prompt set.
- Additional samplers (DDIM, Heun, LMS) against reference trajectories
- GGUF header parsing (metadata only, before dequantization)
- A repeatable benchmark harness for `backend_bench`: run each configuration N
  times and report a median and spread. Single runs on a busy machine vary by
  ±40%, which is how the chunk-size question ended up unanswerable. Small,
  self-contained, and it unblocks every performance claim after it.
