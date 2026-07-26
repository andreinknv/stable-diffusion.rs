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
| ✅ | SDXL — end to end | 1024 on Metal in 89 s, f16 weights + tiled decode |
| ✅ | GGUF | `--gguf` generates images; use Q4_K over Q4_0, see below |
| ⬜ | Metal and CUDA paths through the seam | Metal verified end to end |

The VAE encoder's downsampler pads **asymmetrically** (bottom and right only).
A symmetric `padding: 1` gives the right shape and a half-pixel shift per
level — 17.32 max_abs against a correct 8.2e-5. Worth knowing before writing
any other downsampling path.

Remaining milestone 2 work:

- ~~**Better quantisations than Q4_0.**~~ **Done — use Q4_K, not Q4_0.**
  Measured by `sd-models --test gguf_quant_sweep`, every row against the same
  golden references the safetensors path is held to:

  | quant | size | VAE `mean_abs` | UNet corr | CLIP corr |
  |---|---|---|---|---|
  | f16 | 2133 MB | 6.3e-5 | 1.0000 | 1.0000 |
  | Q8_0 | 1764 MB | 6.3e-4 | 0.9999 | 0.9999 |
  | Q6_K | 1711 MB | 2.3e-3 | 0.9998 | 0.9986 |
  | Q5_K | 1664 MB | 2.4e-3 | 0.9992 | 0.9949 |
  | Q4_K | 1619 MB | 5.5e-3 | 0.9976 | 0.9852 |
  | Q4_0 | 1567 MB | 8.9e-3 | 0.9848 | 0.9746 |

  **The f16 row is the point of the table.** It is the control: f16 through
  the GGUF name map reproduces the f32 reference exactly, so the mapping is
  correct and every other row is quantisation error alone. Without it, a poor
  Q4_0 image cannot be distinguished from a subtly wrong translation — which
  is what we spent time suspecting.

  Q4_K costs 3% more file than Q4_0 and recovers nearly all the lost detail;
  Q4_0's characteristic soft, low-contrast output is gone. Q8_0 is
  indistinguishable from f16. The text encoder is the weakest tower at every
  quantisation, consistent with CLIP's magnitude-851 activations.

  Two things worth knowing before doing this on another model:

  **No published SD 1.5 carries k-quants, for a structural reason.** k-quants
  use blocks of 256 along the fastest axis; SD 1.5's UNet is built from 320-
  and 640-channel blocks, and 320 % 256 = 64. 497 of 1131 tensors cannot take
  one and fall back to F16 — which is why Q4_K is *larger* than Q4_0 here
  rather than comparable. Generate them locally:

  ```bash
  cargo run --release -p sd-cli --example requantise -- \
    sd15-f16.gguf sd15-q4_k.gguf Q4_K
  ```

  **Only 37% of SD 1.5's parameters are quantisable at all.** Convolution
  weights have a 3-wide fastest axis, so no block quantisation applies and
  they stay F16 in every published file. Quantisation reaches the attention
  and linear layers — and the entire text encoder, which is why CLIP absorbs
  the damage.

  Still open: keeping the text encoder at F16 while the UNet stays quantised.
  Q4_K makes this much less urgent, and unlike the above it is plumbing (two
  weight sources for one pipeline), not measurement.
- **Keep quantised weights quantised.** Today every GGUF weight is
  dequantised to f32 at load, so quantisation buys disk and nothing else —
  SD 1.5 occupies 4.26 GB of RAM whether the file is Q4_0 or f32. candle's
  `QMatMul` holds weights in their quantised form and dequantises per
  operation. This is the blocker for the larger architectures below: Flux is
  12B parameters, which is 48 GB of f32 and will not load on a 36 GB machine
  at any quantisation. Doing this before attempting them is the right order.
- CUDA through the seam. Metal is verified end to end at 512; CUDA is untested.
- **SD 1.5 in f16 too**, if it is ever worth it. SDXL needed it to fit; SD 1.5
  does not, and switching would mean re-verifying every golden test against f16
  tolerances. Measure first.
- **img2img and SDXL together.** The SDXL pipeline is txt2img only; the encoder
  and `Strength` already exist, so this is composition rather than new maths.

## Milestone 3 — breadth

SD 2.x · SD 3 · Flux (schnell, dev) · ControlNet · LoRA · TAESD ·
ESRGAN upscaling · inpainting  —  ✅ Flux (flux-mini), ✅ T5 text encoder

This is the phase that took upstream years, and it parallelizes well: each
architecture is independent and verifiable against its own golden data.

**Flux runs.** `flux-mini` (3.2B) renders at 512x512 in 212 s on CPU —
`assets/flux-mini-512-crab.png`. Every component is verified separately:

| component | vs. reference | agreement |
|---|---|---|
| Flux VAE decode | `diffusers` | `max_abs` 1.4e-5 |
| Flux VAE encode | `diffusers` | at its f32 noise floor, 9.6e-4 |
| rectified flow sigmas | `diffusers` | 2.8e-8, at two resolutions |
| flow Euler step | `diffusers` | 1.2e-7 |
| T5 v1.1 encoder | `transformers` | output 1.9e-5 |
| Flux MMDiT | `diffusers` | relative drift 2.1e-6 |
| 20-step sampling loop | `diffusers` | `max_abs` 6.5e-5 |
| Flux schnell (12B) | — | loads quantised, 4.87 GB, renders |
| SD 3.5 VAE | `diffusers` | `max_abs` 1.2e-5 |
| SD 3.5 MMDiT | `diffusers` | `max_abs` 5.5e-6 |

Three things worth carrying to SD 3, which shares most of this:

**F16 does not work for either large model.** T5's activations pass 190,000
and the transformer NaNs; f16 stops at 65,504. T5's weights are therefore held
quantised and expanded per matmul, which keeps activations in f32 and costs
2.7 GB instead of 18.8. bf16 would suit both, but candle's CPU backend has no
bf16 matmul — that is the single change that would most simplify this.

~~**A striping artifact remains in the bottom ~15% of the image.**~~
**Resolved: it is the checkpoint's, not ours.** flux-mini's output carries an
elevated horizontal gradient in its last two latent rows — 1.31x a typical
row — which the VAE renders as a band of vertical striping.

Attributed by elimination, each step measured rather than argued:

| question | test | answer |
|---|---|---|
| Is it the VAE? | decode *our* latent with `diffusers`' VAE | no — 2.8e-5 agreement, and diffusers renders the identical band |
| Is it the sampling loop? | run `diffusers`' loop on *our* inputs and noise | no — final latents agree to 6.5e-5 |
| Is it the model? | compare last-row gradient | yes — diffusers' own latent shows the same 1.31x |

Handing the reference implementation our conditioning *and* our initial noise
is what made this quick: it removes the tokenizer, both text encoders and the
RNG from the comparison, so a mismatch could only have been the loop.

Both checks are now permanent, in `golden_flux_sampling.rs`. One verifies the
twenty-step loop, which no per-component test covers — a compounding error
shows up there and nowhere else. The other pins the artifact itself, so it is
not re-investigated, and so a genuine regression in the last rows is still
caught.

~~**Full Flux is still out of reach on 36 GB.**~~ **Done — schnell runs.**
12B parameters, **4.87 GB resident** against ~48 GB at F32, rendering 512x512
in 166 s on CPU: `assets/flux-schnell-512-crab.png`. The whole stack comes to
about 8 GB, so the constraint is gone rather than merely met.

Quantised residency did that, and it is worth being precise about why it was
worth building. It began as a memory optimisation, turned out to be the
*correctness* fix for T5 (F16 overflows), and is now the only reason a 12B
model runs here at all. One mechanism, three payoffs.

No name mapping was needed: published Flux GGUFs carry the original
black-forest-labs names, which is what `sd-models/flux` already asks for.
Geometry and guidance are read from the file (`flux_block_counts`,
`flux_has_guidance`) rather than assumed, so schnell's 19/38 blocks and dev's
guidance embedding are picked up automatically and passing a guidance scale
to schnell is an error instead of a silent no-op.

The schnell image also settles the artifact question from the other
direction: it shows no bottom-edge striping, which is what the elimination
already concluded — that band belongs to flux-mini.

Still open here: **dev**, which is gated on HuggingFace and needs an account,
and **Metal**, since all Flux work so far is CPU-only.

**SD 3.5's MMDiT is ported and verified** — `max_abs` 5.5e-6 against
diffusers, relative drift 8.3e-7. Its VAE and sampler were config only; the
transformer was the work.

Four ways it differs from Flux, each of which fails quietly if assumed:
learned positional embeddings cropped from the **centre** of a 384x384 table
rather than RoPE; every block joint, with no single-stream half; the last
block's context half `pre_only`, contributing keys and values and then
discarding its own output; and SD 3.5's second image self-attention in the
first 13 blocks, which modulate nine ways instead of six.

The bug worth recording: **patchify and unpatchify are not inverses here.**
The patch embedding is a convolution, so its flattened kernel runs
`(channel, ph, pw)` — the order Flux packs in. The final linear instead emits
`(ph, pw, channel)`. Reusing Flux's inverse gave an image of the right shape
with every 2x2 patch transposed internally: coherent colour, destroyed
detail, no error. It cost `max_abs` 4.42 on an output whose scale is 2.73,
and fixing it alone took that to 9.3e-3.

The remaining 9.3e-3 was not ours either. The *same weights* are published
twice — a single fp16 file and a converted fp32 copy — and they differ by up
to 2e-3, which 24 blocks with activations reaching 97,000 amplify. Generating
the reference from the file Rust actually reads took it to 5.5e-6. Worth
remembering generally: when two copies of a checkpoint exist, verify against
the one you load.

Still to do for a working SD 3.5 pipeline: the three text encoders need
wiring together (CLIP-L and CLIP-G pooled and concatenated to 2048, T5 for
the sequence) — all three encoders already exist and are verified, so this is
assembly rather than new models. Conveniently, k-quants suit these models far better than they suit SD 1.5 —
their hidden sizes are multiples of 256, so nothing falls back to F16, which
is why city96 can publish `Q4_K` for SD 3.5 and nobody can for SD 1.5.

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
- ~~A repeatable benchmark harness for `backend_bench`~~ — **done**: five
  repeats, reported as a median plus spread, with `SD_BENCH_REPEATS` to
  override. A minimum would flatter a machine that was briefly idle; the
  median plus the spread answers "what does this cost, and do I believe it".
