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
ESRGAN upscaling · inpainting  —  ✅ Flux (flux-mini), ✅ T5 text encoder,
✅ LoRA (SD 1.5 dense path), ✅ LCM sampling, ✅ inpainting, ✅ ControlNet, ✅ TAESD, ✅ SD 2.x

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
| SD 3.5 end to end | — | 512x512 in 311 s on CPU |

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

**SD 3.5 renders**: 512x512, 20 steps, 311 s on CPU —
`assets/sd35-medium-512-crab.png`.

Its conditioning is the most involved in the project and worth recording,
because two parts of it look like mistakes:

- CLIP-L and CLIP-G sequences concatenate on the *feature* axis to 2048 and
  are then **zero-padded to T5's 4096**, so half of every CLIP token is
  empty by design. T5's sequence is concatenated on the *token* axis
  instead.
- The CLIP sequences come from the **penultimate** layer while the pooled
  vectors come from the **projection head** — two different depths of the
  same forward pass.

Also: SD 3 uses real classifier-free guidance, two transformer passes per
step, where Flux distils guidance into a conditioning input and needs one.
And `pooled()` is correct here where Flux wanted `pooled_hidden()`, because
SD 3 ships `CLIPTextModelWithProjection` and Flux ships a plain
`CLIPTextModel`. Neither distinction produces an error if got wrong — only a
worse image. Conveniently, k-quants suit these models far better than they suit SD 1.5 —
their hidden sizes are multiples of 256, so nothing falls back to F16, which
is why city96 can publish `Q4_K` for SD 3.5 and nobody can for SD 1.5.

## Deliberately not doing

- **Training.** Inference only.
- **Our own GPU kernels — for now.** The seam makes this a per-kernel decision
  we can take later, when profiling says which one matters. Writing ~60 kernels
  per backend up front is 8–15 months before anything renders.
- **A GUI.** A good library first; someone else can build the UI.

## ~~Highest-value optimisation available now~~ — resolved without a kernel

This entry used to call for a hand-written fused attention kernel for Metal,
on the grounds that our attention materialises a 4096x4096 score matrix and
that "candle does not expose [a fused kernel] for our shapes". **That was
wrong, and checking it before writing the kernel is what caught it.**

candle 0.11 ships `candle_nn::ops::sdpa`, a fused Metal kernel accepting head
dimensions of 32, 64, 72, 80, 96, 128, 256 and 512 — which covers Flux (128),
SD 3.5 (64), T5 (64), CLIP (64) and SDXL (64). `attention_with_path` was
already routing to it; only the doc comment claiming it was unreachable was
stale, written when SD 1.5 and SDXL were the only models here.

Measured with `--example attention_path`:

| shape | CPU | Metal | |
|---|---|---|---|
| SDXL UNet 1024 | 453.7 ms | **14.8 ms** | fused |
| Flux 1024 | 957.0 ms | **43.8 ms** | fused |
| Flux 512 | 108.8 ms | **8.5 ms** | fused |
| SD 3.5 512 | 37.1 ms | **2.7 ms** | fused |
| T5-XXL 154 tok | 12.6 ms | **0.3 ms** | fused |
| SD 1.5 UNet 512 | 110.7 ms | 16.8 ms | chunked |

So Metal is no longer slower than CPU for attention — it is 12-30x faster
wherever the fused path applies. Chunked attention remains the fallback and
still earns its place: it bounds the allocation, and it is what SD 1.5's
40- and 160-wide heads use, along with any masked shape, since the kernel
wants a `[batch, heads, seq_q, seq_k]` mask and `causal_mask` is `[1, 1, s, s]`.

### ~~Metal end to end: fast, and currently wrong~~ — fixed

Flux schnell at 512x512, 4 steps: **20.8 s on Metal against 159.3 s on CPU**,
a 7.7x speedup, and the image is now the same crab the CPU renders. It used to
be a flat orange field with a corrupted strip along the top.

Verified across the workspace afterwards, since nothing quantised was
separable from this bug while it stood:

| model | CPU | Metal | agreement with CPU |
|---|---|---|---|
| SD 1.5 512, 20 steps | 113 s | **17.5 s** | max 1/255, 98.8% of pixels exact |
| SDXL 1024, 20 steps | — | **86.5 s** | renders correctly |
| Flux schnell 512, 4 steps | 159 s | **20.8 s** | mean 9.2/255, same image |
| SD 3.5 medium 512, 20 steps | 230 s (dense) | **24.5 s** | now Q4_K_M by default |
| SD 3.5 medium 256, 20 steps | 71.8 s | **9.0 s** | mean 7.0/255, same image |
| Flux mini 512, 20 steps | 212 s | does not fit | dense f32, 12.8 GB resident |

SD 1.5's near-exact agreement and Flux's mean 9.2/255 are both expected: SD 1.5
is 20 steps through a shallow UNet, Flux is 4 steps through 57 blocks whose CPU
path carries 0.3-1.9% quantisation noise per layer that Metal does not. Same
picture either way; not interchangeable as files.

**SD 3.5's 25.1 s is a real run but not a reliable one, and the reason is
worth recording.** Its transformer is dense f32 — 10.2 GB — and loading it
leaves about 1.1 GB free on a 36 GB machine; with anything else running, the
job dies in denoise **step 1**. It was recorded here and in the handoff as a
*VAE decode* failure, which was wrong: candle queues Metal work and inspects
the command buffer only when something synchronises, so the failure is
attributed to whatever waits first, and the decode was simply the first thing
to wait. A `synchronize()` after each step moves it to step 1.

The fix was not a smaller decode tile — it was the 1.79 GB Q4_K_M GGUF sitting
unused in the fixtures. **That turned out to need no name mapping at all**,
which is worth recording because the handoff predicted otherwise: city96's
SD 3.5 GGUF carries the original Stability names
(`joint_blocks.0.context_block.attn.qkv.weight`, `x_embedder`, `pos_embed`,
`final_layer.linear`) and `sd_models::sd3` already asks for exactly those. The
Flux GGUF loader does no renaming either — it was rejecting the file purely on
a `double_blocks.` sentinel check.

So `sd3_qtensors_from_gguf` is `flux_qtensors_from_gguf` with a different
sentinel, and the two share one body. Reading the tensor names out of the file
answered in two minutes what had been written up as a mapping project.

SD 3.5 now loads in ~4 s instead of 14.7 s, renders 512 on Metal in 24.5 s,
and keeps working under memory pressure that killed the dense build. The cost
is CPU speed — 93 s against 72 s at 256 — because candle's CPU quantised
matmul quantises the activation per call where a dense f32 matmul just runs
gemm.

**Root cause: candle 0.11's Metal quantised matmul ignores the activation's
`start_offset`.** A tensor that is a view into the middle of a larger buffer is
read *from the beginning of that buffer*. Nothing errors — the shapes are
right, the kernel runs, and the answer is the product of the wrong rows.

The trap that makes it survive review is that **`contiguous()` does not save
you**. `narrow` along anything but the last axis of a contiguous tensor yields
a layout candle already calls contiguous — the elements are consecutive, they
merely start late — so `contiguous()` is a no-op and the offset persists.
`force_contiguous()` is the one that always copies.

Flux hit it in every double-stream block. Attention runs on the text and image
tokens joined, then splits them apart again:

```text
  txt_attn = attn.narrow(1, 0, 512)        offset 0          -> correct
  img_attn = attn.narrow(1, 512, 1024)     offset 512*3072   -> read the text rows
```

So all 19 blocks projected the text half of the attention output in place of
the image half. The fix is `sd_tensor::quantized::without_storage_offset`,
applied inside `QLinear::forward` — one place, because the seam is where every
quantised matmul in the workspace passes.

**Why the per-op check missed it for three sessions.** `--example metal_check`
compared freshly built tensors, and a fresh tensor always owns its buffer at
offset 0. The op is correct exactly when tested and wrong exactly when used.
Every isolated check passed — attention at 1.9e-7 across every sequence
length, QLinear at every row count and quantisation type, norms, RoPE, trig,
`cat`/`narrow`, weight loading, dequantisation to 1e-8 — while the composed
model was 50% wrong. `metal_check` now includes an offset case, and it fails
loudly when the workaround is removed.

**What actually localised it**, in order, each step halving the space:

1. Cross-decode each device's latent on *both* devices — the VAE agreed to
   4e-5 both ways, so the decode was innocent and the latent was already bad.
2. Compare the loop's inputs — noise bit-identical, CLIP pooled 1e-5, T5 2%.
3. Feed the transformer *identical* inputs on both devices — 50% divergence,
   so the transformer alone was enough.
4. Dump every intermediate against a **dense f32 reference built from the same
   weights**, not against the other device. That is what made it readable:
   Metal tracked full precision better than CPU (≤0.12%) up to the attention
   output and then jumped to 36% at one projection.
5. Recompute that projection in numpy from Metal's *own* dumped input — 36%
   off, so the op was wrong rather than its input.
6. Compare against the product of the buffer's first rows — matched to 1.6e-2.

Step 4 is the transferable one. CPU-vs-Metal conflates two error sources and
reads as noise; against a full-precision reference the culprit is a single
line in a table.

### CPU flash attention: a short-sequence win, not a general one

candle 0.11's `candle_nn::attention::flash_attn` is now wired in as
`ops::flash_attention_cpu` and reported as `AttentionPath::FlashCpu`. The
survey below called it "potentially the largest single win available". It is
not, and the shape of *why* is the useful part.

The kernel streams one output row at a time under a running softmax maximum,
so it never materialises the score matrix — but it also gets no register
blocking across query rows and re-reads the key axis for each one. Against a
tuned gemm that is a losing trade at large sequences and a winning one at
small, and the crossover is sharp. Measured across `head_dim`
{40, 64, 80, 128, 160}, `heads` {8, 12, 20, 24, 64}, batch {1, 2}:

| seq | 64 | 128 | 256 | 512 | 768 | 1024 | 4096 |
|---|---|---|---|---|---|---|---|
| h=24, d=128 | 2.9x | 5.5x | 2.4x | 1.2x | 1.0x | 0.9x | 1.0x |
| h=8, d=40 | 1.5x | 4.6x | 2.4x | 1.1x | 0.7x | 0.6x | 0.5x |

So it is taken only at or below 512 tokens (`ops::DEFAULT_FLASH_CPU_MAX_SEQ`,
override with `SD_FLASH_CPU_MAX_SEQ`, `0` disables). That covers CLIP and
SD 1.5/SDXL's two deepest UNet levels, and deliberately excludes SD 3.5
(1178), Flux (1536+) and the UNet levels that dominate a denoise step.

**T5 is excluded too, and not by the length rule.** Its relative-position bias
is a full `[batch, heads, n, n]` tensor; the kernel indexes a mask flat as
`q_pos * seq_k + kv_pos` with no head axis, so `flash_cpu_supported` refuses
it. This is worth spelling out because the benchmark actively misleads here:
`--example attention_path` times T5's 154-token shape *unmasked* and reports
5-8x, and an earlier draft of this section turned that into a claim that
"every text encoder" benefits. It does not, and no measurement of a shape the
model never produces could have caught it — reading the caller did.
`the_real_text_encoder_shapes_take_the_paths_we_think_they_do` in sd-models'
`api_contract` now pins both halves.

**End to end the effect is below this machine's noise floor**, which is what
the mechanism predicts: the eligible calls add up to roughly 0.4 s of an
SD 1.5 generation, and about 13 ms of an SD 3.5 one. SD 1.5 at 512x512, 20
steps ran 113.3 s with it against 114.4 s without. SD 3.5 was run four times
alternating:

| | flash off | flash on |
|---|---|---|
| pair 1 (off first) | 245.2 s | 216.2 s |
| pair 2 (on first) | 230.4 s | 228.7 s |

The first pair looks like an 11.8% win and **is not one** — the spread between
two runs of the *same* configuration (245.2 against 230.4) is larger than the
difference between configurations, and reversing the order collapses the gap
to 0.7%. Quoting pair 1 alone would have put a fabricated 12% in this
document. Images differ by at most 1/255, on 1.0% of pixels for SD 1.5 and
0.6% for SD 3.5.

The shapes it wins are real but they are not where the time goes — attention
at 4096 tokens is, and that is the one place this kernel loses. Beating gemm
there needs a blocked CPU kernel that tiles over query rows as well as keys,
which is a real kernel rather than a wiring change.

Two things found on the way in, both recorded in the `sd-tensor` doc comments:
candle dispatches `batch > 1` to a "varlen" kernel whose repacking costs up to
4.1x more than simply looping the batch-1 kernel, so `flash_attention_cpu`
loops; and unlike the Metal kernel it accepts `causal_mask`'s `[1, 1, s, s]`
directly, because it indexes the mask flat.

### Candle capabilities we are not using

A survey after the fused-attention surprise, since the same mistake was
clearly available twice:

- **`candle_nn::ops::rms_norm`** — a fused RMSNorm. We hand-wrote one in
  `t5`, one in `flux` and one in `sd3`. Three copies of an op candle already
  has, each doing its own f32 upcast.
- ~~**`candle_nn::cpu_flash_attention::run_flash_attn_cpu`**~~ — done, and it
  was **not** "the largest single win available" as this list guessed. See
  [CPU flash attention](#cpu-flash-attention-a-short-sequence-win-not-a-general-one)
  below for what it actually bought. Note the entry point named here is a
  deprecated shim; the live API is `candle_nn::attention::flash_attn`.
- **`candle_nn::rotary_emb::{rope, rope_i, rope_thd}`** — fused RoPE. Flux's
  axis-wise 2x2 formulation may not map onto it directly, but that is worth
  establishing rather than assuming.
- **`candle_nn::ops::{layer_norm, pixel_shuffle, pixel_unshuffle}`** — a
  fused LayerNorm, and shuffles that are exactly patchify/unpatchify.
- `candle-transformers` ships its own flux, t5, clip and stable_diffusion
  models. We do not want those — implementing the models is the point of this
  project — but they are a reference to check ambiguous conventions against.

- Broadening the fused attention path to SD 1.5 by materialising the causal
  mask to the shape the kernel wants, which is a reshape rather than a kernel.

## Upstream contributions worth making

- **candle: the Metal quantised matmul ignores `start_offset`.** This is a
  silent wrong-answer bug, not a performance note: any quantised model that
  feeds a `narrow`ed activation to a linear layer gets the product of the
  wrong rows, with correct shapes and no error. It cost this project three
  sessions of a corrupted Flux. We work around it in `QLinear::forward`
  (`without_storage_offset`), but the fix belongs in candle's Metal backend —
  add the layout offset when binding the activation buffer, as the CPU backend
  already does. A reproducer is four lines: quantise any weight, `narrow` an
  activation off dim 0, and compare against `force_contiguous()` of the same
  view. `--example metal_check` contains it.
- **candle: implement f16 matmul in the Accelerate CPU backend.** Today it
  bails outright (`cpu_backend/mod.rs:1497`), which means `--features
  accelerate` — worth 1.7-1.9x on CPU — cannot be used with any f16 model.
  `Linear` and `Conv2d` reach the same path, so it is not a niche gap: an SDXL
  run dies on its first convolution. The fix is to convert per gemm tile to
  f32 and call `cblas_sgemm`; converting whole tensors instead defeats the
  point, which is why this cannot be worked around downstream. Reproducer:
  `matmul` two f16 CPU tensors with the feature enabled.
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
