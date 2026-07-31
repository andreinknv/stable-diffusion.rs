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
- ~~**Keep quantised weights quantised.**~~ **Done**, and it turned out to be
  the load-bearing piece it was predicted to be. `weights::Source::Quantized`
  holds a `QLinear` and dequantises per operation; `FluxTransformer` and
  `Sd3Transformer` both build `from_quantized`, with streaming variants on
  top. Flux schnell is 12B parameters in 6.32 GB resident, which is the only
  reason it runs here at all — see handoff.md, where the same mechanism turned
  out to be the *correctness* fix for T5 as well, since f16 overflows it.
- **CUDA through the seam** — the only unticked box in this file, and open
  purely for want of a device. Metal is verified end to end and has per-module
  CPU parity; nothing about CUDA is known either way.
- ~~**SD 1.5 in f16 too.**~~ **Measured, and the answer is no.**
  `--example unet_dtype` times one UNet forward at a guidance batch of 2, 512,
  synchronising inside the timed region, f32 and f16 alternated:

  ```text
    f32          698.6 ms
    f16          637.1 ms
    f32 again    700.5 ms
  ```

  **1.10x**, worth about 1.2 s on a 20-step image, plus 1.7 GB of residency on
  a machine where SD 1.5 already fits in f32 with room to spare. Against that:
  casting at the sampler boundary inside the most-verified loop in the project,
  and re-verifying every golden test at f16 tolerances. SDXL took f16 because
  it did not fit otherwise, which is a different reason and still a good one.

  Worth re-running if a machine with a much wider f16 advantage appears —
  the example exists now, so it is one command.
- ~~**img2img and SDXL together.**~~ **Done** — `SdxlPipeline::run_img2img`,
  verified end to end through encoder tiling, strength, sampler and decoder.
  It was composition rather than new maths, as predicted.

## Milestone 3 — breadth

SD 2.x · SD 3 · Flux (schnell, dev) · ControlNet · LoRA · TAESD ·
ESRGAN upscaling · inpainting  —  ✅ Flux (flux-mini), ✅ T5 text encoder,
✅ LoRA (SD 1.5 dense path), ✅ LCM sampling, ✅ inpainting, ✅ ControlNet, ✅ TAESD, ✅ SD 2.x, ✅ ESRGAN upscaling, ✅ IP-Adapter, ✅ seamless tiling, ✅ textual inversion, ✅ two-pass (hires), ✅ checkpoint merging, ✅ motion modules, ✅ area conditioning, ✅ InstructPix2Pix, ✅ GLIGEN, ✅ unCLIP, ✅ step caching (TeaCache predictor)

**unCLIP generates from an image embedding**, `sdrs unclip` — either from a
reference image, or from a prompt through the prior. The last capability
outstanding from the third integration issue, and the least code of any of
them: its UNet is SD 2.x with one extra module, and the prior reuses the
timestep embedding, the masked-attention primitive and the cosine ladder that
were already here.

It also found two defects in code that had been green for the whole project:
a pooled CLIP embedding read from the last padding position rather than the
first EOS (affecting Flux, SD 3 and GLIGEN), and a Metal-only matmul refusal on
a narrowed view. Both are written up in [handoff.md](handoff.md#traps-this-codebase-has-already-paid-for).

| component | vs. reference | agreement |
|---|---|---|
| noise augmentation, level 0 | `diffusers` | 4.9e-6, floor 6.4e-6 |
| noise augmentation, level 250 | `diffusers` | 1.5e-5, floor 1.0e-5 |
| image embeds (this ViT-H) | `transformers` | 1.3e-6 |
| whole UNet with `class_labels` | `diffusers` | 2.0e-4, floor 2.8e-4 |
| the unconditional (zero) row | `diffusers` | 6.2e-4 |
| prior transformer, masked | `diffusers` | 3.2e-6 |
| prior transformer, unmasked | `diffusers` | 4.7e-6 |
| prior text encoder, projected | `transformers` | 9.7e-7 |
| one prior DDPM step | `diffusers` | 4.8e-7 |
| the t2i UNet under the prior | `diffusers` | 2.0e-3, floor 1.5e-3 |

The floors are the reference's own f32-against-f64 spread, from
`reference_precision.py unclip`. Two of them are unusually high for what looks
like arithmetic on a 1024-vector, and both took measuring rather than
guessing: `1 - alpha` cancels catastrophically near the top of the ladder, and
the noise level's sinusoid is evaluated at arguments as large as the level, so
rounding its frequency to f32 costs `250 * 6e-8` in the argument and `cos`
passes that straight through.

Each architecture is independent and verifiable against its own golden
data, so this phase parallelizes: a port can be started, finished and checked
without touching the others.

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
is why `Q4_K` is publishable for SD 3.5 and not for SD 1.5.

## Deliberately not doing

- **Training.** Inference only.
- **Our own GPU kernels — for now.** The seam makes this a per-kernel decision
  we can take later, when profiling says which one matters. Writing ~60 kernels
  per backend up front is 8–15 months before anything renders.
- **A GUI.** A good library first; someone else can build the UI.

## Corrections to the tick marks above

Three capabilities in the Milestone 3 line were ticked on the candle backend
and **did not survive its removal**. Nothing in `crates/*/src` mentions any of
them, and nothing in `crates/*/tests` does either:

| capability | state |
|---|---|
| seamless tiling | **restored** — circular padding, verified against edge statistics |
| InstructPix2Pix | **restored** — verified against `timbrooks/instruct-pix2pix` |
| checkpoint merging | **restored** — `sdrs merge`, with a safetensors writer |

All three are back, and re-adding them against the candle implementation as a
known-good target was exactly as cheap as this section predicted. What each
one cost is worth recording, because in every case the port was the easy half:

**Seamless tiling.** `pad_circular` is twenty lines. The decision that took
thought was where the mode lives: a thread-local guard held inside `denoise`
and `decode` rather than at the eight public entry points, because a guard at
each is a guard one of them will miss, and a run that tiles in the sampler but
not the decoder has a seam with nothing to point at. `decode` takes it as a
parameter, so a new decode path is a compile error until it says whether it
tiles.

Verified by measuring the wrap against the image's *own* smoothness rather than
an absolute threshold — a photograph of grass and one of a brick wall have very
different neighbour statistics. Tiled, opposite edges differ by 6.83 against a
typical neighbouring-pixel step of 7.26; untiled, 20.75 against 10.26.

The first attempt used 256x256 and could not tell the two apart: SD 1.5 puts
low-frequency mush at the borders at that size, so the untiled image already
wrapped cleanly. The test said so itself rather than passing vacuously, which
is the property worth copying — a control assertion that fails when the setup
cannot discriminate.

**InstructPix2Pix.** Three things fail quietly here and all three cost a
plausible image rather than an error: the guidance batch has *three* rows
(instruction, image, and a true unconditional that sees neither); the image
latent is **not** scaled by 0.18215, unlike every other latent in this
codebase; and the source joins on the channel axis, which is what the extra
four input channels are for.

It also needed a guard in the other direction. A plain `txt2img` on an
InstructPix2Pix checkpoint used to fail inside the first convolution with
`Expect the input channels ... to match but got (2,32,32,4) and (320,3,3,8)` —
true, and useless. It now names the checkpoint kind and the method to use.

**And it is size-dependent**, which took measuring. InstructPix2Pix is SD 1.5
based and trained at 512; at 256 it restructures the picture rather than
editing it. On an identical source and instruction:

| size | correlation with source | mean change |
|---|---|---|
| 256 | 0.325 | 100.3 |
| 512 | 0.748 | 72.4 |

The first version of that test used a synthetic gradient as its source and
measured 0.357 at 512 — **the same trap this file already records for img2img**,
which once used a `torch.randn` source and measured nonsense because noise is
not something an autoencoder can represent. A block-and-gradient is the same
mistake more subtly. The source is now generated.

**Checkpoint merging** needed a safetensors *writer*, which the MLX port never
had. Its one real hazard is laziness: MLX computes nothing until asked, so
saving an unevaluated array writes a file of exactly the right shape holding
whatever the buffer contained. `save_safetensors` evaluates first, and a test
hands it a deliberately lazy expression to prove it.

The lesson generalises, and it is the same one this file records about
verifying against a published artifact: **a tick mark describes a backend, not
a project.** When the backend changed, the ticks needed re-earning and most of
them were, silently, because the port was thorough. Three were not.

## What the command line cannot reach

This is the largest gap between what the project *is* and what a user can
*run*, and it is bigger than any missing model.

`sdrs` had four commands — `txt2img`, `img2img`, `upscale`, `info` — while
Flux, SD 3.5, quantised loading, GGUF and unCLIP were implemented, verified
against diffusers, and reachable only by writing Rust. **The headline result of
the whole project, a 12B Flux running quantised on a 36 GB laptop, had no
command line.**

Now nine commands, and that run works:

```bash
sdrs flux --model models/flux --variant schnell \
  --transformer-gguf flux1-schnell-Q4_K_S.gguf --t5-gguf t5xxl-Q4_K_S.gguf \
  --prompt "a rusty crab on a beach at sunset"
# resident: 11.6 GiB      512x512 in 30 s
```

| capability | reachable from `sdrs` |
|---|---|
| Flux (schnell, dev, mini), from a directory or GGUF | `sdrs flux` |
| SD 3.5 | `sdrs sd3` |
| quantised-at-rest loading | `--bits` on both |
| unCLIP / image variation | `sdrs unclip` |
| instruction editing | `sdrs instruct` |
| checkpoint merging | `sdrs merge` |

That first Flux run paid for itself immediately by finding two real defects.
`FluxPaths` defaulted the T5 tokenizer to `tokenizer_2/spiece.model`, which
`T5Tokenizer` cannot read — so the *default* path failed for every checkpoint,
with `stream did not contain valid UTF-8`, an error naming neither the file nor
the problem. Both are fixed, and a sentencepiece path now says what it is.

**Still not on the command line:** IP-Adapter, GLIGEN, TAESD and Canny
preprocessing. All are implemented and verified; each needs an image argument
and a flag, and none is load-bearing for the claim above.

## Against `stable-diffusion.cpp`

The reference implementation this project set out to be a Rust answer to.
Checked against `leejet/stable-diffusion.cpp` at `master-805-e31a86c`
(2026-07-30) by reading `include/stable-diffusion.h` and the `docs/` tree
rather than the README's prose.

### Where the two agree

SD 1.x, SD 2.x, SDXL, SD 3.x, Flux (schnell/dev-class), ControlNet, LoRA,
IP-Adapter, TAESD, ESRGAN, inpainting, img2img, textual inversion, two-pass
hires, step caching, GGUF weights, quantised residency, VAE tiling, negative
prompts, and a cancellable generation with progress.

### Sampling: the widest gap, and the cheapest to close

sd.cpp ships **20 samplers and 16 schedulers**; this project ships 3 and 1.

| | sd.cpp | here |
|---|---|---|
| samplers | Euler, Euler A, Heun, DPM2, DPM++ 2S a, DPM++ 2M, DPM++ 2M v2, DPM++ 2M SDE, DPM++ 2M SDE BT, iPNDM, iPNDM_v, LCM, DDIM trailing, TCD, RES multistep, RES 2S, ER-SDE, Euler CFG++, Euler A CFG++, Euler GE | Euler A, DPM++ 2M, LCM |
| schedulers | discrete, Karras, exponential, AYS, GITS, SGM uniform, simple, smoothstep, KL-optimal, LCM, bong tangent, beta, and four model-specific | linear over the training sigmas |

**Karras is the one that matters most.** It is what most published step counts
and most community presets assume, so a 20-step comparison against any other
implementation is currently comparing two different schedules. It is roughly
fifteen lines and verifiable against `k_diffusion.sampling.get_sigmas_karras`
directly — the same shape of golden test `sigmas.rs` already has.

After that, in value order: **Heun** and **DPM++ 2S a** (better images at low
step counts), **DDIM** (the baseline every paper reports against), and
**exponential**/**SGM uniform** scheduling.

### Guidance and conditioning

| | sd.cpp | here |
|---|---|---|
| CFG rescale / skip-layer guidance | `sd_slg_params_t`, per-layer, with a start/end window | no |
| separate image and text CFG | `txt_cfg`, `img_cfg` | one scale |
| distilled guidance | explicit | Flux only, implicit |
| `--clip-skip` | yes | no |
| `eta`, `flow_shift`, custom sigmas | yes | no |

**`--clip-skip` is the notable omission.** The machinery is present —
`clip::penultimate` exists and SDXL uses it — but it is hardwired per
architecture rather than exposed, and a large fraction of community SD 1.5
checkpoints are trained expecting `clip_skip = 2`. Without it those models are
being run wrong, and the result is a worse picture with no error, which is this
project's most-repeated failure shape.

**Skip-layer guidance** matters specifically for SD 3.5, which is where it was
introduced and where its absence is visible in anatomy.

### Models it has and this does not

Chroma, Qwen Image, Z-Image, FLUX.2, HiDream, Ideogram4, and a dozen more from
2026; the video models (Wan 2.1/2.2, LTX-2, HunyuanVideo 1.5); and Qwen Image
Edit. FLUX.1-Kontext is now here — see below.

**These are the genuinely large remaining item**, and worth being precise about
why rather than listing them as a gap. Each is a new transformer: its own
block structure, its own conditioning, its own name mapping, and — the part
that dominates — its own golden references generated from `diffusers` and
checked tensor by tensor. That is this project's standard and the reason its
numbers mean anything; skipping it would be faster and would produce ports
nobody should trust.

In rough order of cost, cheapest first:

- **Chroma** is a Flux variant — same block structure, a different modulation
  scheme and no pooled CLIP input. The closest to free.
- **Qwen Image** and **Z-Image** are new DiTs of familiar shape.
- **FLUX.2** is a new generation, not a variant.
- **The video models** are the largest by a wide margin: a temporal axis
  through every block, a 3D VAE, and memory characteristics this machine has
  not been asked for before.

**Kontext is done, structurally.** It was the one worth taking first for the
reason given — it reuses the Flux transformer unmodified — and that turned out
to be exactly right: the transformer needed no change at all.

The mechanism is that the reference image is encoded through the same VAE,
packed into the same 2x2 patches, and its tokens **appended to the image
stream**; every block attends over both, and the extra tokens are dropped
before unpacking. What makes it work rather than produce a double exposure is
the rotary embedding's **third axis** — the one Flux otherwise never uses, with
every ordinary image token at `t = 0`. The reference sits at `t = 1`, so the
model can tell the picture it is making from the picture it was given while
both occupy the same `(h, w)` grid. `image_ids_at` is that index.

**Unverified for edit quality, and that is the honest state.** Running the
mechanism on schnell's weights exercises the plumbing — the reference reaches
the model, 99.4% of output bytes change, the shape survives — but schnell was
not trained for it, so the result is not an edit. Verifying the capability
needs FLUX.1-Kontext's own weights, which are gated. The structural test says
so in its own doc comment rather than letting green read as "Kontext works".

### Features it has that are not about models

- **Generation parameters embedded in the PNG**, as an A1111-compatible text
  chunk. Small, and it makes every output self-describing — which suits a
  project that cares about reproducibility more than most.
- **Preview callbacks during sampling** (`PREVIEW_TAE`), decoding the running
  latent every *n* steps. `Progress` already carries `denoised` for exactly
  this; nothing consumes it.
- **Batch generation from one load.** `sdrs` reloads the model per invocation,
  so generating four seeds costs four loads — about 3 s each for SD 1.5.
- **imatrix-guided quantisation.**
- **RPC / multi-device**, which MLX also exposes (`distributed.h`) and which
  neither this project nor most users need.

### Where this project is ahead

Worth stating plainly, because a gap list reads as a deficit report:

- **Every component is checked against `diffusers` or `transformers` with a
  recorded tolerance**, and the references are regenerated from the file that
  is actually loaded. sd.cpp has no comparable numeric gate.
- **The backend is confined behind one crate**, enforced by a CI lint.
- **Memory safety**, and no C or C++ compiles into this build at all — also CI
  enforced.
- The failure modes this file documents — the reversed decoder block order, the
  asymmetric encoder padding, the patchify/unpatchify asymmetry, the pooled
  embedding read from the wrong position — are recorded with the measurement
  that found each one.
- **A tokenizer cannot be missing.** CLIP's vocabulary is vendored, so a
  directory holding nothing but `unet/`, `vae/` and `text_encoder/` renders.
  sd.cpp reads the tokenizer out of the checkpoint and fails without one.

### A note on running the suite here

The pipeline tests each load a full SD 1.5 into unified memory, and cargo's
default parallelism is the core count. On a 36 GB machine that is an OOM kill —
`signal: 9, SIGKILL`, with no failing assertion, which reads exactly like a
crash in the code under test and is not one. `--test-threads=3` is the working
setting. Worth knowing before spending an afternoon on it, and worth
remembering that this failure mode has twice been misdiagnosed here in the
other direction: a real assertion failure blamed on the OOM killer.

## Using MLX properly

MLX is now the only backend. What is bound in `sd-tensor/src/mlx.rs` is 62 of
its entry points, and what that leaves unused is worth being specific about.

### Measured, and the answer was no

**A per-step `eval` in the sampling loop.** The loop never forces evaluation,
so the natural worry is that twenty steps accumulate as one lazy graph and
nothing is retired until the decode. Measured at 768x768, 20 steps, alternating
to control for machine state:

```text
  without per-step eval   3.95 GB peak RSS   20.8 s
  with    per-step eval   3.99 GB peak RSS   21.7 s
  without                 4.11 GB            21.7 s
  with                    3.99 GB            21.4 s
```

**No effect on either.** MLX's scheduler already retires the graph
incrementally; adding a synchronisation point buys nothing and costs the
overlap. Recorded so it is not proposed again.

### Measured, and worth a flag rather than a default

**f16 weights for the UNet.** The stock SD 1.5 checkpoint is F32 for all 686
tensors, so every matmul runs at f32 today. Casting the UNet at load, same
alternating protocol:

| | peak RSS | wall |
|---|---|---|
| f32 | 4.03 GB | 23.8 s |
| f16 | 2.85 GB | 21.9 s |

**1.10x faster and 1.15 GB smaller.** The speed matches what the candle path
measured for the same change, which is reassuring rather than surprising — a
diffusion step at this size is bandwidth-bound before it is FLOP-bound.

The image is not the same image: mean byte difference 1.08 of 255, PSNR
36.8 dB, 58.5% of bytes identical. That is a different sample of comparable
quality, not a degraded one — but it means f16 belongs behind a flag with that
number attached, not silently on. For SDXL, Flux and SD 3.5, where memory is
the binding constraint, the 1.4x residency saving is the interesting half.

**bf16 is not bound at all** (`MLX_FLOAT16` is, `MLX_BFLOAT16` is not), and it
is the better choice for the large models: T5's activations pass 190,000 and
f16 stops at 65,504, which is why T5's weights are held quantised today. bf16
has f32's exponent range, so it would make that a dtype choice rather than a
workaround.

### Not measured yet, in rough order of expected value

- ~~**`mlx_compile`**~~ — **built, measured, and not adopted.**

  The FFI is done and stays: `sd_tensor::mlx::Compiled` wraps
  `mlx_closure_new_func_payload` and `mlx_compile`, with a trampoline that
  catches panics rather than unwinding through C. It is the tool for any future
  candidate, and `sd-tensor/tests/mlx_compile.rs` gates it — results identical
  to the composition, a new input shape retraced rather than reused, and a
  failing closure surfaced as an error.

  **The compiler works. The application does not pay.** Flux's `norm_modulate`
  was the best candidate in the codebase: pure, entirely elementwise, five
  calls per double block across 57 blocks, and the same shape the candle-era
  hand-written adaLN kernel took 5.26x on. In isolation at 1024x3072 MLX fuses
  it at **1.44x**, bit-identical — 407 us to 283 us. Wired into a real Flux
  schnell run, 16 steps at 512, alternated:

  ```text
    compiled     56.9 s   56.2 s
    composed     55.4 s   56.6 s
  ```

  Nothing, and marginally negative. The prediction this entry used to make was
  right for the reason it gave: a step is mostly quantised matmul, which a
  fuser does not touch, so 1.44x of the elementwise share does not surface. The
  wiring was reverted rather than kept behind a flag — an unused code path and
  a thread-local for no measured gain is a worse trade than the composition.

  Recorded at this length because the isolated number is genuinely encouraging
  and would invite a second attempt. What would change the answer is a model
  whose steps are *not* dominated by matmul, or fusing something much larger
  than one chain — not this chain again.
- ~~**`mlx_fast_rope`**~~ — **checked, and it does not fit.** Reading the
  signature before writing the call is what caught it, exactly as it did for
  the candle-era claim that no fused attention kernel existed:

  ```c
  int mlx_fast_rope(mlx_array* res, const mlx_array x, int dims,
                    bool traditional, mlx_optional_float base, float scale,
                    int offset, const mlx_array freqs, const mlx_stream s);
  ```

  `offset` is a **scalar**, so the kernel applies position `offset + index`
  uniformly along the sequence. Flux's positions are not a sequence: `embed_nd`
  takes `[seq, 3]` integer coordinates — a `(t, h, w)` grid flattened — and
  splits the head dimension into three segments with *different* widths and so
  different frequency sets per segment. Neither the arbitrary per-token
  positions nor the three-way head split can be expressed through a scalar
  offset and one `freqs` array. The hand-written `rotate` stays.
- **`mlx_get_peak_memory` / `mlx_set_wired_limit` / `mlx_set_cache_limit`.**
  Cheap, and the right instrument for every measurement above: peak RSS was
  used here because MLX's own accounting is not bound, and `sdrs info` reports
  free system memory rather than what a run would actually need. On a 36 GB
  machine the wired limit is the difference between a large model running and
  the machine swapping.
- **`mlx_fast_metal_kernel`.** Custom Metal kernels from a source string at
  runtime, with no build step and no `.metal` file in the repository. The
  roadmap's old position — "our own GPU kernels, for now, no" — was priced
  against writing and shipping ~60 kernels per backend. This is a much lower
  price for the handful that profiling names.
- **`mlx_async_eval`**, for overlapping the decode with the next generation in
  a batch.
- **`mlx_export_function`**, which serialises a traced graph. Speculative here.
- **`distributed.h`**, multi-device. Real, and not what one laptop needs.

## The library surface

The pipeline API grew by accretion and now has an entry point that is hard to
call correctly:

```rust
pipe.txt2img_with(&cfg, hint, ip_tokens, &objs, &regions,
                  cache_threshold, cancel, &mut progress)?
```

Eight positional parameters, five of which are `None` or empty in almost every
call, and two adjacent `Option<&Array>` that the compiler cannot tell apart.
There is already an `Extras` struct doing this job internally; the fix is to
make it the public surface, or to give `MlxPipeline` a builder so that the
common call stays short and the rare one names its arguments.

Three smaller things in the same spirit:

- **`Txt2ImgConfig` has no `clip_skip`, no batch count, and no scheduler
  choice** — see the sd.cpp comparison above for why the first matters most.
- **`ModelPaths` describes exactly one layout**, the `diffusers` directory
  tree. The community norm is a single `.safetensors` file, and `sd_loader`
  already knows how to translate LDM names — the mapping exists, nothing calls
  it from the safetensors path.
- **`load` and `load_on` are duplicated per pipeline** — four times, plus
  `load_unclip` and `load_unclip_on`. A `Device` on the paths struct, or a
  `with_device` builder, removes the pairing.

## Deliberately not doing

- **Training.** Inference only.
- **A GUI.** A good library first; someone else can build the UI.
- **CUDA.** MLX 0.6 exposes `cuda.h` and `mlx-c` builds against it, so the
  seam no longer forbids it — but there is no device here to verify against,
  and an unverified backend is worth less than no backend.

## Good first issues

- **Karras sigmas**, verified against `k_diffusion.sampling.get_sigmas_karras`.
  The highest value-per-line item in this file.
- **`--clip-skip`**, plumbed through `Txt2ImgConfig` to `clip::penultimate`.
- **`sdrs flux` and `sdrs sd3`**, wrapping pipelines that already exist.
- **A1111-compatible PNG metadata** on every write.
- **`mlx_get_peak_memory`** bound and reported by `sdrs info`.
- Additional samplers (DDIM, Heun, DPM++ 2S a) against reference trajectories.
- GGUF header parsing (metadata only, before dequantisation).
- ~~Load the CLIP tokenizer from vocab.json + merges.txt~~ — **done, and then
  made unnecessary.** Neither `stable-diffusion-v1-5` nor
  `stabilityai/stable-diffusion-xl-base-1.0` publishes `tokenizer.json`, so
  this was not a papercut affecting some downloads but the path every download
  takes. `ClipTokenizer::open` reads either form and falls back to a
  vocabulary vendored in `sd-models`, so **a tokenizer can no longer be
  missing**: a directory holding nothing but `unet/`, `vae/` and
  `text_encoder/` renders, byte-identically to the full checkpoint.

  One vocabulary suffices because it is a constant of the architecture rather
  than a property of the weights, which was checked rather than assumed: SDXL's
  `tokenizer_2` — the OpenCLIP bigG tower, trained by a different organisation
  — ships `vocab.json` and `merges.txt` byte for byte identical to
  `openai/clip-vit-large-patch14`. What differs between towers is the padding
  token. Verified id-for-id over ten prompts covering case, whitespace,
  contractions, digits, punctuation, NFC accents, emoji, empty, and
  truncation.
