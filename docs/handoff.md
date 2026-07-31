# Handoff

## "go" starts here

Saying **go** means: read this file, take the first unchecked item under
**Start here** below, and begin. No further instruction is needed or expected.

### Start here

Steps 1–3 are done and the port is most of the way through step 4. What is left
is at the bottom of this list; the crossed-out items are kept because their
findings still bind.

1. ~~**Decide the fork.**~~ — **decided 2026-07-30: (b), MLX's own shape.**
   `sd-tensor` does *not* emulate candle's API. See [The fork to decide before
   writing any code](#the-fork-to-decide-before-writing-any-code) for the
   reasoning, which turned on `mlx-c`'s actual signatures rather than on taste.

2. ~~**Spike GGUF before binding anything.**~~ — **done 2026-07-30, and it
   does not block.** Reading and dequantising GGUF needs one Python library and
   no candle. SD 3.5 medium dequantises to 4.9 GB of f16 and needs nothing
   else; Flux schnell would be 23.8 GB and has to stay quantised, which costs a
   cosine of ~0.995 through MLX's own 4-bit. Full numbers and the
   double-quantisation trap under [Order of work](#order-of-work).

3. ~~**Bind `mlx-c` behind `sd-tensor`.**~~ — **done.** `crates/sd-tensor/src/mlx.rs`
   carries the hand-written FFI: `Array`, `Stream`, ~45 ops, safetensors
   loading, and the thread-local error trampoline that the first draft owed.
   `build.rs` links mlx-c/mlx only under `--features mlx`, via
   `MLX_C_PREFIX`/`MLX_PREFIX` or Homebrew. `scripts/check-seam.sh` still
   passes: nothing outside `sd-tensor` names either backend.

4. **Port the models.** Everything below is green against the same golden
   fixtures the candle path uses, at the same measured tolerances:

   | model | gate | worst |
   |---|---|---|
   | SD 1.5 UNet | `mlx_golden_unet` | 3.353e-4 excess |
   | SD 2.x / SDXL / unCLIP UNets | `mlx_golden_unet` | within `UNET_TOL` |
   | VAE encoder + decoder | `mlx_golden_vae` | 1.9e-5 |
   | CLIP text tower | `mlx_golden_clip` | within f32 ULP at peak 851 |
   | T5 v1.1 | `mlx_golden_t5` | 2.6e-5 |
   | SD 3.5 MMDiT | `mlx_golden_sd3` | 3.75e-6 |
   | Flux DiT | `mlx_golden_flux` | 3.49e-6 |
   | TAESD | `mlx_golden_taesd` | 7.42e-6 decode, 3.21e-6 encode |
   | Real-ESRGAN | `mlx_golden_esrgan` | 2.205e-6 |
   | ControlNet | `mlx_golden_controlnet` | 2.038e-5 |
   | LoRA | `mlx_lora` | bit-identical to `sd-loader` |
   | IP-Adapter | `mlx_golden_ip_adapter` | 1.200e-5 |
   | GLIGEN | `mlx_golden_gligen` | 1.049e-5 |
   | AnimateDiff | `mlx_golden_motion` | 7.63e-6 module, 4.59e-5 whole UNet |
   | SDXL text encoder 2 | `mlx_golden_sdxl_text_encoder` | 1.909e-4 excess |
   | CLIP vision (ViT-H) | `mlx_golden_clip_vision` | 1.509e-4 excess |
   | Flux VAE | `mlx_golden_flux_vae` | 2.83e-5 decode |
   | SDXL ControlNet | `mlx_golden_controlnet_sdxl` | 4.53e-5 mid |
   | unCLIP prior | `mlx_golden_prior` | 2.85e-6 masked |
   | Tiled VAE | `mlx_vae_tiled` | mean_abs 0.008 against whole |
   | GGUF reader | `mlx_gguf_agrees_with_candle` | bit-exact |
   | GGUF models | `mlx_gguf_models` | 1.000000 from f16 |

5. **img2img and inpainting** — **done**, `mlx_img2img`. `vae::encode` gives
   the distribution's mean, `sample::noise_to_sigma` noises it to where
   `strength` starts, `sample::latent_mask` reduces a pixel mask 8x8 by *max*,
   and `sample::restore_outside_mask` runs the composite inside the loop. See
   [img2img and inpainting on MLX](#img2img-and-inpainting-on-mlx).

6. **What is actually left, and it is two things.** Both are real, and the
   first is the larger.

   **(a) The pipeline layer.** `MlxPipeline` now exists —
   `crates/stable-diffusion-rs/src/mlx/`, gated by `mlx_pipeline` — and does
   **txt2img, img2img and inpaint on SD 1.5 through the public API**: load a
   model directory, call `txt2img`, get bytes. Same seed gives the same image
   byte for byte, the two samplers are distinguishable, and a size that does
   not divide into latent cells is refused.

   What it does *not* yet carry, all of which the candle `Txt2ImgPipeline` does
   and all of which is orchestration rather than new model work:

   - SD 3.5, Flux and unCLIP pipelines (the models are ported; the pipelines
     around them are not). **SDXL is done** — `SdxlPipeline`, txt2img and
     img2img, verified at its native 1024 in 24 s
   - AnimateDiff frame batching. **ControlNet, LoRA, IP-Adapter and GLIGEN
     are wired**: `attach_controlnet` (several stack, corrections sum, scale 0
     is exactly zero), `attach_lora` (errors rather than half-applying),
     `attach_ip_adapter` (scale 0 exactly zero), and `generate(..., boxes)`
     for GLIGEN — which refuses boxes on a UNet with no fuser layers rather
     than dropping them
   - step caching, region/area prompts, two-pass hires, model placement,
     progress reporting and cancellation
   - textual inversion, area prompts, per-step conditioning, upscaling
   - AnimateDiff frame batching

   So the shape has changed since the last entry: this is no longer "the
   pipeline layer is entirely candle", it is "one pipeline of five works, with
   the conditioning features unwired".

   **(b) Quantised-at-rest inference does not exist on MLX**, so full-size Flux
   and T5-XXL cannot leave candle. The MLX GGUF loader dequantises to f32;
   Flux schnell is 11.89B parameters, which is **47.6 GB** dense and does not
   fit on this machine at all. The candle path keeps those weights quantised
   and dequantises per operation — `FluxTransformer::from_quantized`,
   `resident_bytes` — and MLX has its own scheme with no equivalent built here.

   `mlx_gguf_large` pins (b) as a test rather than a promise: the geometry of
   both checkpoints is verified from the tensor directory, which costs nothing,
   and the dense footprint is asserted to *not* fit. If that assertion ever
   fails, the limitation is stale and the test says so.

   What (b) would cost is already measured, under [Order of
   work](#order-of-work): requantising into MLX's own 4-bit scheme at a cosine
   of ~0.995 against the GGUF values — and **quantise from the original f16
   checkpoint, not from the GGUF**, or the error sits on top of Q4_K's own.
   `flux_schnell_gguf`'s tolerance would then have to be re-derived from
   `xtask/golden/reference_precision.py`, not widened until it passed.

   Everything else in the 405 is ported, covered under a different name, or
   dies with candle — see [Closing the MLX test
   gap](#closing-the-mlx-test-gap-2026-07-31).

7. **Run both backends in parallel** until every golden test passes on MLX.
   Delete candle in one commit, not gradually.

Verification, unchanged and non-negotiable:

```bash
SD_REQUIRE_FIXTURES=1 SD_TEST_MODEL_DIR=$(pwd)/models/sd15 \
SD_TEST_SDXL_DIR=$(pwd)/models/sdxl \
SD_TEST_INIT_IMAGE=$(pwd)/assets/controlnet-crab-canny-512.png \
SD_TEST_CONTROLNET=<a ControlNet .safetensors, e.g. lllyasviel/sd-controlnet-canny> \
cargo test --release --workspace --features metal --no-fail-fast
```

Without those variables the suite reports 14 failures that are only unset
paths, and `cargo test` stops at the first failing binary — so a truncated run
looks like a clean one. `--no-fail-fast` is not optional.

**`SD_TEST_CONTROLNET` was missing from this block until 2026-07-30**, which
made the command fail one test by construction:
`control_maps_must_match_the_attached_controlnets` needs it, and with
`SD_REQUIRE_FIXTURES=1` a skip is a failure — by design, and the design is
right. With it set the run is 405 passed, 0 failed. It wants the `.safetensors`
**file**, not the directory containing it; a directory fails with
`expected a .safetensors file`.

## Decision: the backend moves to MLX

Taken deliberately, after measuring the alternative. The evidence both ways is
in [roadmap.md](roadmap.md); this section is what to do about it.

**Why, in one line:** every performance win this project found had to be
hand-written, and MLX's lazy graph fuses that class of thing automatically —
so the question was never "can we beat candle" (we did, repeatedly) but "how
many kernels are we prepared to own forever."

### What the measurements say to carry across

Nothing here is wasted, but be clear about what survives.

**Survives, and is the most valuable thing in the repo:** the golden tests and
their tolerances. Every one was measured against diffusers/transformers with
`xtask/golden/reference_precision.py`, never guessed. **They are how the MLX
port gets proven correct.** Do not loosen one to make a port pass — that rule
does not change with the backend.

**Survives as knowledge, not code:**
- candle's `gelu_erf` returns *exactly zero* for every input below about -6,
  because it forms `1 + erf(u)` by subtraction. Check whether MLX does the
  same; the fix is to read `erfc` off the polynomial before subtracting.
  See `fused.rs` and `--example gelu_tail`.
- Reductions: sequential row sums lose accuracy with row length. This bit
  twice — candle's CPU `rms_norm` is 6-9x less accurate than a blocked
  reduction, enough to fail `golden_t5`. Verify MLX reduces in a tree.
- A composition and a kernel **trade places depending on the backend**. Three
  separate times: fused won on Metal and lost on CPU, for the same op.

**Becomes dead code on the move:**
- `sd-tensor/src/fused.rs` — the three Metal kernels (group_norm 6.09x, adaLN
  5.26x, GEGLU 3.65x). MLX's fusion should subsume them. **Measure that it
  does** rather than assuming; if MLX does not fuse norm-then-modulate, that
  is 492 ms a Flux step and worth a custom op.
- The 1x1 convolution routing in `conv.rs`.
- `sd-tensor/src/mps.rs` and the four objc2 dependencies.

**The hardest piece, and the reason to schedule it early:** `quantized.rs` and
GGUF. MLX has its own quantisation and does not read GGUF natively. Flux
schnell GGUF is in the test suite and will not port by itself.

### What the seam actually hides

Measured 2026-07-30, from source rather than from the index: cartograph's
resolved edges undercount cross-crate Rust badly here (3,538 resolved against
22,783 unresolved, and a cross-seam edge query returned 10 symbols), so every
figure below was counted against the code and not taken from the graph.

**596 import sites, 49 distinct items, 102 files, five crates** reach through
`sd_tensor`:

| what | sites | share |
|---|---|---|
| candle types re-exported verbatim | 467 | 78% |
| `sd-tensor` shadows, deliberately candle-shaped | 40 | 7% |
| genuinely `sd-tensor`'s own | 89 | 15% |
| **candle-shaped overall** | **507** | **85%** |

`lib.rs:23` hands out `Tensor, DType, Device, Error, Result, Module, Shape,
IndexOp, D, Layout, CpuStorage, CustomOp1/2/3` straight from `candle_core`;
`nn` hands out `Linear, LayerNorm, Embedding, VarBuilder, VarMap` from
`candle_nn`. `sd_tensor::Tensor` **is** `candle_core::Tensor` — one type, two
names.

So `check-seam.sh` passes exactly as designed, and it is not lying: it greps
for `use candle_(core|nn|transformers)` outside `sd-tensor`, and there is
none. **The seam hides the crate name, not the API shape.** It does bound the
blast radius to one crate, which is the valuable half and worth keeping. The
other half — "in principle nothing above it changes" — holds only if MLX can
be made to present candle's surface.

**This does not change the decision.** Owning kernels forever, convolution at
48% of a step, and candle's two silent wrong-answer bugs are all unaffected.
It changes what step 1 is.

#### The fork to decide before writing any code

Step 1 is not "bind the ~20 ops `sd-models` calls". It is "reimplement
candle's API on MLX for 102 files this project has committed to not touching".
There are two ways, and they produce entirely different first commits:

**(a) Emulate candle's API over MLX.** The 102 files do not change. The cost is
a candle-compatibility layer owned forever — which recreates the thing this
move exists to escape.

**(b) Change the call sites.** Roughly 500 of them, in MLX's own shape. A much
larger diff up front and nothing permanent to maintain. MLX's lazy graph fits
candle's eager `Result`-per-op shape poorly, so a shim would be fighting the
backend as well as carrying it.

**Decided 2026-07-30: (b).** Recorded here because every later estimate depends
on it.

What settled it was reading `mlx-c`'s actual surface rather than reasoning
about it. Ops are `int mlx_add(mlx_array* res, const mlx_array a, const
mlx_array b, const mlx_stream s)`: out-param result, status return, an explicit
stream per call, manual `mlx_array_free`, and a **global** error handler rather
than per-call errors. Most of that wraps into candle's shape mechanically. Two
things do not:

- **Errors are global.** Reconstructing candle's `Result` per op needs a
  thread-local trampoline around `mlx_set_error_handler`.
- **Lazy against eager, which is the one that decides it.** candle's API means
  every `Result<Tensor>` is a materialised value. MLX's speed *comes from*
  fusing the graph before `mlx_eval`. Emulating candle faithfully forces a
  choice between evaluating per op — discarding the fusion that produced the
  2.53x — and staying lazy while `Result` reports success for work that has not
  run, so failures surface at `eval` attributed to the wrong operation.

(a) would therefore have spent a permanent shim to protect 102 files and risked
the performance the move exists for. (b) changes roughly 500 call sites once,
in mechanical batches, each gated by the golden tests.

What makes this concrete, in order of how load-bearing it is:

- **`VarBuilder`** — 46 imports across 47 files. Candle's prefixed lazy weight
  loader, and how every model here is constructed. MLX has no equivalent;
  something has to be written before a single model loads.
- **`Result` / `Error`** — 33 imports of candle's error type. Already conceded
  at `lib.rs:35`: "It cannot be an `Error` variant, because `Error` is
  candle's."
- **`CustomOp1/2/3`, `CpuStorage`, `Layout`** — candle's backend-extension
  interface, and how all three fused Metal kernels dispatch. MLX extends
  differently, so `fused.rs` does not port: it is rewritten or dropped. Budget
  for that rather than assuming MLX's fusion subsumes it.
- **`QTensor` / `GgmlDType` / gguf `Content`** — the GGUF path, which is the
  scheduling problem noted below.

### Order of work

1. **Decide (a) or (b) above, and write the answer down.** No binding work
   until this is settled; the two paths do not share a first commit.
2. **Spike GGUF**, ahead of the binding work rather than after it — see
   **Why GGUF moved** below.
3. **Bind `mlx-c` behind `sd-tensor`.** The seam bounds this to one crate —
   but see [What the seam actually hides](#what-the-seam-actually-hides)
   before estimating it: 85% of what crosses the seam is candle-shaped, so
   this step carries the API surface, not just ~20 ops. `mlx-c` is Apple's own
   C API — bind it directly rather than taking a third-party shim
   (`cargo search mlx` finds no first-party Rust crate).
4. **Port SD 1.5 first**, end to end, against `golden_unet` and
   `golden_vae`. The checkpoint is on disk and the tests are the strictest.
   Do not port a second model until the first is green.
5. **Keep candle building in parallel** behind a feature until parity is
   proven on every golden test. Delete it in one commit, not gradually.
6. **Finish GGUF**, with `flux_schnell_gguf` as the gate, carrying whatever
   the spike established.

**Why GGUF moved.** This document previously said both "the hardest piece, and
the reason to schedule it early" and "GGUF last". Both could not hold, and the
early reading is the right one: quantisation is load-bearing here, not a
finishing touch. SD 3.5 defaults to Q4_K_M (1.79 GB against 10.2 GB dense, and
the dense build died in denoise step 1 under load), Flux schnell GGUF is a test
gate, and Flux mini already does not fit on this machine. MLX has its own
quantisation and does not read GGUF. If that cannot be bridged the port strands
exactly the models that need it most — and leaving it last means finding that
out after five architectures have been ported.

#### Spike run 2026-07-30: GGUF is not a blocker, but Flux and SD 3.5 need different answers

Reading the container needs nothing from candle and costs nothing: the
`gguf` Python package parsed both checkpoints in 0.03 s, and dequantising the
largest tensor took 0.10 s (SD 3.5) and 0.22 s (Flux). Handing the result to
MLX is a plain `mx.array`. The feared piece — "MLX does not read GGUF" — is
true and irrelevant: dequantisation is a library call, not a subsystem.

What actually decides it is memory, and the two models diverge:

| checkpoint | on disk | elements | dense f16 | verdict |
|---|---|---|---|---|
| SD 3.5 medium Q4_K_M | 1.7 GB | 2.47B | **4.9 GB** | dequantise on load, no requantisation, no quality loss |
| Flux schnell Q4_K_S | 6.3 GB | 11.89B | **23.8 GB** | too tight beside activations on 36 GB — must stay quantised |

Both are mixed containers, not uniformly quantised: SD 3.5 is 402 F16 / 215
Q4_K / 48 Q5_K tensors, Flux is 468 F32 / 304 Q4_K / 4 F16. Quantised tensors
carry 4.50 bits/weight on disk against 16 in MLX, a 3.6x expansion.

So SD 3.5 medium can take the simple path and lose nothing. Flux cannot, and
has to be requantised into MLX's own scheme. Round-tripping the largest tensor
through `mx.quantize`/`mx.dequantize` at group size 64, against the values GGUF
dequantised to:

```text
  Flux    4-bit   max|diff| 3.35e-2   mean 3.13e-3   cosine 0.995822
  Flux    8-bit   max|diff| 2.00e-3   mean 1.86e-4   cosine 0.999973
  SD 3.5  4-bit   max|diff| 2.18e-2   mean 7.65e-4   cosine 0.994762
  SD 3.5  8-bit   max|diff| 1.28e-3   mean 4.56e-5   cosine 0.999978
```

8-bit is near-lossless and roughly halves the f16 footprint; 4-bit costs a
cosine of ~0.995 and is the only thing that keeps Flux comfortably in memory.

**The trap in those 4-bit numbers.** They compare MLX's quantisation against
*GGUF-dequantised* values, so the error sits on top of Q4_K's own loss — two
lossy quantisations in series. If MLX quantisation is used at all, quantise
from the original f16 checkpoint rather than from the GGUF.
`models--adamo1139--stable-diffusion-3.5-medium-ungated` is already in the
cache, so this is measurable rather than theoretical.

**Consequence for the gates.** `flux_schnell_gguf` would then be checking a
differently-quantised model, not the same one. That tolerance must be
re-derived from `xtask/golden/reference_precision.py`, not widened until the
test passes — rule 3, and it does not bend for a port.

### Numbers to beat

The current Metal path, measured this session, interleaved three times each:

```text
  SD 1.5, 512x512, 20 steps:  15.736 s  ->  12.830 s   (18.5% faster)
```

That 12.830 s is what an MLX port has to beat to be worth the move. It is not
a low bar: it already includes three fused kernels candle does not have.

Also measured, and the thing MLX most plausibly improves on: candle's
convolution is im2col plus a matmul, materialising up to 283 MB per call at
~25 GB/s, and **Apple's own conv is 1.99x faster** (`--example mps_conv`).
Convolution is ~48% of a step. If MLX does not beat candle's conv, the move
has not paid for itself.

#### It does. Measured 2026-07-30: MLX's conv is 4.1x candle's

Every 3x3 convolution in one SD 1.5 forward, the shapes in
`examples/sd15_inventory.txt`, batch 2, f32 both sides, min of 10 after warmup
on both sides:

```text
  candle Metal (--example conv_breakdown)   336.7 ms   = 211.0 gemm + 125.7 im2col
  MLX 0.29.3  (mx.conv2d, NHWC)              81.3 ms
                                            -------
                                              4.14x
```

MLX's convolution is also **faster than the bare gemm candle performs under
it** (81.3 against candle's 211.0 gemm alone), which is the signature of a
direct convolution rather than an im2col round trip. The 37% ceiling the
`conv_breakdown` header derives — "a perfect direct convolution could remove at
most 37% of it" — bounds what removing im2col could win *within candle's
structure*. MLX beats it by not having that structure.

Verified before being believed: `mx.conv2d` at this exact call shape agrees
with an independent numpy float64 im2col reference to 1.3e-6 relative at
f32, across four shapes including the 4->320 and 320->4 extremes. The 4.1x is
not a mis-shaped call that skips work.

**Two caveats that matter more than the headline.**

- **MLX convolutions are channels-last.** These numbers are NHWC in, `(out,
  kh, kw, in)` weights. That is MLX's native layout and candle's is NCHW, so
  each framework was measured in its own idiom — fair for "which is faster",
  but a port carries an NHWC conversion through every model, or eats a
  transpose per call. Budget it; do not discover it.
- **candle's gemm here is 2.4x slower than MLX's own** (211.0 against 89.2 at
  identical shapes and dtype). [roadmap.md](roadmap.md) argues the gemm quarter
  is unbeatable because candle's matmul kernels *are* Apple's MLX code. The
  kernel may well be the same; this path is not. The gap is dispatch and
  layout around it, most likely `broadcast_matmul` materialising its broadcast.
  That is worth confirming, because it means part of the "already optimal"
  quarter is in fact on the table.

#### End to end, measured 2026-07-30: MLX is 2.5x at equal precision

Same weights (`models/sd15`), 512x512, 20 steps, CFG 7.5, one image. Both sides
load outside the clock, take one warm run for shader compilation, and
synchronise before stopping — the methodology in `examples/sd15_step_time.rs`,
mirrored on the MLX side. Denoise **and** VAE decode are inside the clock on
both, because the Rust `pipeline.run` returns an image. Interleaved three
times each on an otherwise idle machine:

```text
  candle  f32   12.780 s   639.0 ms/step   (12.780, 12.834, 12.780)
  MLX     f32    5.047 s   252.3 ms/step   ( 5.066,  5.047,  5.044)   2.53x
  MLX     f16    3.916 s   195.8 ms/step   ( 3.912,  3.916,  3.923)   3.26x
```

**f32 is the honest comparison.** `Txt2ImgPipeline` loads every module as
`DType::F32` (txt2img.rs), so 2.53x is like for like. The f16 row is what MLX
would actually be run at and is listed separately rather than being passed off
as the same test; candle would need its own f16 path to claim it.

Peak memory: 8.49 GB at f32, 6.66 GB at f16.

**The trap that cost the first three attempts, recorded so it is not paid
again.** MLX keeps a buffer pool that grows across generations. On a machine
also hosting a 15 GB colima VM this tips into swap, and a swapping run measures
the pager rather than the backend: run 2 of one loop took 10.8 s and run 4 of
the *same loop* took 115.5 s, a 10x swing with nothing changed. `mx.clear_cache()`
between timed runs fixes it; stopping colima for the measurement removed the
last of the variance and took the spread under 0.5%. Any re-measurement needs
both, or the numbers are fiction. This is rule 2 and rule 6 arriving together.

**Held to `golden_unet`, not to eyesight — it failed, and the cause was one
constant.** Run against the same `tests/golden/unet_full` fixture, the same
weights and the same bound as `full_unet_matches_diffusers`:

```text
  before   max_abs 5.628e-4   atol 1e-4   FAIL
  after    max_abs 1.061e-5   atol 1e-4   PASS
  this project's UNet, same fixture:  1.1e-5
```

**MLX now agrees with diffusers as closely as the candle port does.** All
twelve skips are clean, and `mid_output` lands at 1.224e-4 — which is the
regime `UNET_RTOL` exists for, since `reference_precision.py` measured
diffusers missing its *own* f64 by 1.108e-4 on that tensor. Its excess beyond
the relative term is 5.085e-5, well inside `UNET_ATOL`.

**The bug: `Transformer2DModel`'s GroupNorm epsilon.** `mlx-examples` builds it
as `nn.GroupNorm(norm_num_groups, in_channels, pytorch_compatible=True)` with no
`eps`, taking MLX's default of 1e-5. diffusers hardcodes **1e-6** there. That is
precisely the trap `unet/attention.rs` already warns about at the top of the
file — *"Three `eps` values are in play... unifying them would be tidier and
wrong"* — committed by someone else, in sixteen places. Setting
`SPATIAL_NORM_EPS` on those norms is the entire fix, and it costs nothing: same
ops, same shapes, one different scalar inside a normalisation already being
computed.

**How it was found, because the method generalises.** Localised by first bad
skip, exactly as `down_pass_skips_match_diffusers` is written to do: `conv_in`
exact at 7.153e-7 ruled out weight loading and the NCHW/NHWC transposes;
`down_01` at 2.425e-4 put it in the first block; bisecting that block against a
float64 recomputation of diffusers' arithmetic showed the resnet faithful at
9.402e-6, which left the transformer. **Two hypotheses were tested and both were
wrong** — recorded so they are not re-tested:

- The sinusoidal timestep embedding. `mlx-examples` genuinely parameterises
  those frequencies differently (2.131e-4 on `temb`), but substituting
  diffusers' exact formula moved the output 5.628e-4 -> 5.540e-4, 1.5% of it.
- The attention formulation. MLX dispatches to the fused
  `mx.fast.scaled_dot_product_attention` where diffusers materialises the score
  matrix; swapping in the materialised form left `down_01` bit-identical at
  2.425e-4.

**What this means for the 2.53x.** It was measured before the fix, on code that
failed the gate — but the fix is a scalar, not an algorithm, so the figure
stands. Re-timed after it at 5.540 s against 5.047 s, the difference being that
colima was running; op count is unchanged by construction.

Two further patches were needed to load SD 1.5 at all, and both are the sort of
thing a real port will hit:

- SD 1.5's text encoder ships a `position_ids` buffer that SD 2.1 does not. It
  is a non-learned arange and is dropped.
- SD 1.5's VAE predates diffusers' `to_q`/`to_k`/`to_v` rename and uses
  `query`/`key`/`value`/`proj_attn`. Plain Linear at identical shapes, so a
  rename — but a silent mismatch here decodes to colour-corrupted output rather
  than failing, which is exactly the failure mode `golden_vae` exists to catch.

---

# Handoff

Updated 2026-07-28.

## Say "go" to resume

**"go" means: read this file, take the first item under [Next](#next) that is
not struck through, and start it.** No further instruction is needed and none
should be waited for.

Working rules this project holds to, in rough order of how often they have
mattered:

1. **Verify against a published reference, tensor by tensor.** Every feature
   here has a number against `diffusers`/`transformers`. A feature without one
   is not done, however well it appears to run.
2. **Measure before optimising, and measure the pair both ways.** Single A/B
   runs have lied three times in this file's history.
3. **A tolerance below the reference's own f32-vs-f64 noise floor is measuring
   float32, not the port.** `xtask/golden/reference_precision.py` establishes
   the floor; quote it where a bound is set.
4. **Do not borrow a constant from a paper without checking what it is a
   constant *of*.** This cost two runs in one session — AnimateDiff's beta
   schedule and the step-cache threshold band.
5. **Run the gates before committing**: `cargo fmt --all`,
   `cargo clippy --workspace --all-targets --features metal -- -D warnings`,
   `bash scripts/check-seam.sh`, `cargo test --release --workspace`. And at
   least once per change that touches numbers, the run that actually verifies:
   `SD_REQUIRE_FIXTURES=1 SD_TEST_MODEL_DIR=$(pwd)/models/sd15 cargo test
   --release --workspace` — a plain run reports the same green with no
   fixtures at all.
6. **Check free memory before a large run.** Metal allocations are wired; an
   oversized one takes the machine, not the process.

## What this project does today

Every capability below is verified against `diffusers`/`transformers` with a
recorded number. **378 tests, all gates green**, plus 7 GPU smoke tests behind the `metal`
feature (`cargo test --features metal --test metal_smoke`) — they are not in
the default count because a machine without a GPU cannot run them.

| | |
|---|---|
| **Architectures** | SD 1.5, SD 2.x, SDXL, SD 3.5, Flux (schnell, mini), unCLIP (image *and* text) |
| **Conditioning** | LoRA (dense *and* quantised), ControlNet (several at once), IP-Adapter, GLIGEN boxes, unCLIP image embeddings, textual inversion, area/regional prompts, per-step conditioning |
| **Editing** | img2img, inpainting, InstructPix2Pix |
| **Animation** | AnimateDiff motion modules, frame batching, explicit latent in/out |
| **Output** | TAESD (4 variants), tiled VAE decode, ESRGAN 4x, two-pass hires, seamless tiling |
| **Runtime** | Metal + CPU, GGUF quantisation, block streaming, step previews, cancellation, determinism, checkpoint merging |
| **Formats** | safetensors, GGUF, pickled `.bin` (converted by the dumper) |

Three integration issues drove most of this. **All three are now closed** —
unCLIP was the last capability outstanding from #3, and video/audio/3D are
explicitly out of scope by the author's own ranking.

## Where things stand

Five architectures render, and **Metal produces the right image for every
one that fits** — the Flux corruption is fixed (see the trap on storage
offsets below). Metal is 6-9x faster across the board, so it is now the
sensible default rather than a broken option.

| model | CPU | Metal | agreement |
|---|---|---|---|
| SD 1.5 512, 20 steps | 113 s | **17.5 s** | max 1/255, 98.8% of pixels exact |
| SDXL 1024, 20 steps | — | **86.5 s** | verified this session; `assets/sdxl-crab-1024-metal-f16.png` |
| Flux schnell (12B) 512, 4 steps | 159 s | **20.8 s** | mean 9.2/255, same image |
| SD 3.5 medium 512, 20 steps | 230 s (dense) | **24.5 s** † | now Q4_K_M by default |
| SD 3.5 medium 256, 20 steps | 71.8 s (dense) | **9.0 s** | mean 7.0/255, same image |
| Flux mini (3.2B) 512, 20 steps | 212 s | does not fit ‡ | — |
| unCLIP 768, 20 steps | — | **36 s** § | components verified; see below |
| unCLIP text-to-image 768, 20 steps | — | **67 s** wall | prior adds 25 steps over a 768-vector |

† **Now the quantised checkpoint, and reliable.** `sd3_paths_in` prefers
`sd35-medium-q4_k_m.gguf` (1.79 GB) over the dense `.safetensors` (10.2 GB at
f32) exactly as Flux's `paths_in` does. Loading drops from 14.7 s to ~4 s, and
512 renders in 24.5 s on a quiet machine — and, more to the point, it keeps
working under load, where the dense build died in denoise **step 1** leaving
1.1 GB free of 36 GB.

That step-1 failure was recorded here for several sessions as a *VAE decode*
failure, which was wrong: candle queues Metal work and reports a failed
command buffer only at the next synchronisation point, so the blame lands on
whatever waits first. See the trap below.

§ 27.7 s of denoising at 1.38 s/step plus 8.4 s of tiled VAE decode. Wall clock
is 61 s: loading is another 25 s, because unCLIP holds a **fourth tower** — a
2.5 GB ViT-H — on top of the usual three, and all of it dense f32.

‡ Flux mini holds dense f32 weights: 12.8 GB for a 3.2B model, against
schnell's 6.8 GB for 12B held as Q4_K. Quantisation is what makes the *larger*
model the one that runs — worth remembering before reaching for a dense
checkpoint. Unlike SD 3.5, no quantised flux-mini has been published, so this
one has no equivalent fix available.

**Quantised is not free on CPU**, and the default now takes that trade
knowingly: SD 3.5 at 256 on CPU is 93 s quantised against 72 s dense, because
candle's CPU quantised matmul quantises the activation per call where a dense
f32 matmul just runs gemm. It is the right default anyway — Metal is 6-9x
faster than either and the quantised form is what fits there — but if you are
benchmarking CPU specifically, point `Sd3Paths::transformer` at the
`.safetensors` and expect the dense numbers.

**The VAE decode now sizes its own tile.** `decode_tiled` picks the largest
edge from 64 down whose projected peak fits the memory that is actually free,
so a decode that would not fit degrades to a seamed-but-correct image instead
of dying. `SD_VAE_TILE_LATENT` still overrides it outright. Where 64 already
fits, nothing changes — SD 1.5 on Metal is bit-identical before and after.

**Each pipeline stage can run on its own device.** `pipeline::Placement` is
caller-supplied policy — `Placement::on(&gpu).with_text_encoders_on(&Device::Cpu)`
— passed to `load_with_placement` on Flux and SD 3. `Placement::auto` picks
one from projected residency against free memory. The default is unchanged
(everything on one device), so nothing moves unless asked.

Measured on SD 3.5 at 512, encoders moved to the CPU: **4.4 GB freed on the
accelerator** (9.84 GB available after load against 14.24), for about 8 s of
one-time CPU text encoding. The images agree at mean 4.1/255, which is f32
reduction order in the encoders, not a defect.

**That trade is much better on a discrete GPU than it looks here**, and that
is the reason it exists: on unified memory the bytes come from one pool either
way, while on an 8-12 GB card 4.4 GB is often the difference between running
and not — and the weights never cross PCIe at all. The mechanism is the same;
this machine measures its weakest case. `stable-diffusion.cpp` reached the
same design: `backend`, `params_backend`, `max_vram` and `split_mode` are
fields of its public `sd_ctx_params_t`, not CLI flags.

**The diffusion model's blocks can stream.** `Placement::with_streamed_diffusion()`
keeps a quantised transformer's blocks in host memory and copies each to the
compute device as it is reached, releasing it after — `stable-diffusion.cpp`'s
`--offload-to-cpu`, and the answer for a GPU too small to hold the model at
all, where no static placement helps. Flux only, so far.

Output is **bit-identical** to the resident path — the copy moves quantised
block bytes verbatim, so nothing is rounded twice.

**How much it costs depends on how tightly you want the memory held**, and
that is a real dial rather than a tuning detail. Dropping a block frees
nothing on Metal by itself: candle pools its buffers and returns them only
inside `drop_unused_buffers`, which runs on synchronise. So the interval
between synchronises *is* the peak residency. On Flux schnell, 512, 4 steps,
against 20.9 s resident:

```text
  sync every  1    29.5 s    ~1 block resident   (191 MB)   default
  sync every  4    26.6 s
  sync every  8    25.2 s
  sync every 19    25.3 s    whole stack pools   (3.6 GB)
```

`SD_STREAM_SYNC_EVERY` sets it. The default is 1 because the reason to stream
is the memory.

**An earlier version of this note claimed 25.1 s and "2.4 GB freed" — that was
measured before the synchronise existed**, so the pool was growing and the run
was not holding the memory it claimed. Two lessons: leaving the synchronise
out is not merely untidy — with no release at all a 25 s run degraded past
60 s *per step* as the machine started swapping; and a memory claim measured
without checking when memory is actually returned is not a measurement.

`quantized::to_device` is the primitive; `FluxTransformer::resident_bytes`
reports 0 for streamed blocks, because saying otherwise would report the
opposite of what streaming achieves.

**One-shot runs release before decoding.** `FluxPipeline::run_releasing` and
`Sd3Pipeline::run_releasing` consume the pipeline, drop everything the decode
does not need, and then synchronise — on Metal a drop alone frees *nothing*,
because candle pools its buffers and only returns them inside
`drop_unused_buffers`, which runs on synchronise. Measured worth: about
0.7 GB for SD 3.5. Real, and far less than hoped; see the table note above for
what actually dominates.

Every component is verified against `diffusers`/`transformers` — the full
table is in [roadmap.md](roadmap.md). 378 tests, all gates green
(`cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`,
`scripts/check-seam.sh`, `scripts/check-native-deps.sh`).

**unCLIP generates variations**, `sdrs unclip --model models/unclip
--init-image X`. The last capability outstanding from the third integration
issue, and the smallest: its UNet is SD 2.x with **one extra module**, so the
config is two lines and the blocks are untouched.

The A/B that makes it a demonstration rather than a hope: same seed, same
*empty* prompt, same everything, two different references. A crab photograph
gives a crab on sand (`assets/unclip-crab-noise0.png`); an armoured figure in
a field gives an armoured figure in a field, from behind, in the same pose
(`assets/unclip-knight-noise0.png`). The reference is the only thing choosing
the subject.

All three assets are `--cfg-scale 5.0 --steps 20 --width 768 --height 768
--seed 42`. **Not the CLI's default of 10.0**, which is diffusers' published
figure — and the difference is stylistic rather than a defect, checked across
seeds 42, 7 and 3 rather than asserted from one: at 10.0 the output runs to
crushed blacks and hard vignetting, at 5.0 it is warmer and flatter. The
default stays at the reference's because neither is wrong and because
diffusers pairs 10.0 with PNDM where this runs euler_a; pass 5.0 to reproduce
the assets.

Worth setting expectations either way: with an **empty prompt** this
checkpoint is a stylised photographer, not a neutral one. Every variation
above is more saturated and more contrasty than its reference. That is the
model, not the port — the same is visible in diffusers' own examples.

**No pixel of the reference reaches the model.** That is the difference from
everything else here that takes an image. img2img starts from the picture's
own latent and stops the schedule early, so the composition survives; a
ControlNet gets a map at spatial resolution; IP-Adapter gives the cross
attention four extra tokens to look at. unCLIP passes a **single 1024-vector
describing the whole image** and nothing else, so the output is composed from
scratch — which is why the crab variation is a front-on close-up where the
source is a three-quarter view.

**The noise level is the point, not a detail.** An unaugmented CLIP embedding
is a strong enough signal that the model reproduces the reference and
generates almost nothing, so unCLIP noises the embedding on purpose and the
level is the dial. At 0 it is a tight variation; at 500 the same reference
still gives a crab on sand but the model has invented an ocean, a beach and a
far more elaborate animal (`assets/unclip-crab-noise500.png`).

The level conditions the model **twice** — by being the amount of noise mixed
in, and by having its own sinusoid appended to the vector. That is what makes
`class_labels` 2048 wide where the embedding is 1024.

Verified component by component against diffusers, with every bound measured
first:

```text
  noise augmentation, level 0     4.9e-6     floor 6.4e-6
  noise augmentation, level 250   1.5e-5     floor 1.0e-5
  image embeds (this ViT-H)       1.3e-6
  whole UNet with class_labels    2.0e-4     floor 2.8e-4
  the unconditional (zero) row    6.2e-4
```

**Two of those floors are far higher than a 1024-vector of arithmetic
suggests, and the first attempt at measuring them was wrong by 40x.** It held
the schedule at diffusers' f32 constants and routed the sinusoid through
`get_timestep_embedding`, which hardcodes f32 internally — so the f64 "run"
returned f32 numbers and reported a floor of 2.652e-07. Against that, a
correct implementation failed. Recomputing *both* at f64 gives the real
figures and shows where they come from: `1 - alpha` cancels near the top of
the ladder, turning an absolute 2e-8 difference in the schedule into a
relative 5e-4 one; and the sinusoid is evaluated at arguments as large as the
level, where rounding the frequency to f32 costs `250 * 6e-8` in the argument
and `cos` passes it through undamped. **When measuring a noise floor, check
that the high-precision run is actually running in high precision.**

**Reference images are read the way CLIP reads them**: shortest edge to 224,
then a centre crop, with Catmull-Rom because `resample: 3` in the shipped
`preprocessor_config.json` is PIL's bicubic. Both this and the IP-Adapter path
used to squash to a square, which is identical on a square reference and
changes every proportion in a wide one. `load_clip_square` in `image_io`.

**The checkpoint's normalizer is the identity** — `mean` all zeros, `std` all
ones, the constructor's defaults, never trained. So the golden comparison runs
straight through it and cannot see `scale` and `unscale` swapped or either one
dropped. Unit tests on synthetic statistics pin the formula instead; the gap
is recorded rather than left for whoever meets a checkpoint that does ship
statistics.

The **unconditional row is zeros of the full 2048**, not an absent argument
and not an augmented zero embedding — the last would still carry the level's
sinusoid and mean something. Checked against diffusers rather than assumed,
because all three run.

**The whole checkpoint ships pickled**, so `dump_reference.py unclip` converts
all five components — 1834 tensors, 7.2 GB — and assembles `models/unclip` as
it goes. It downloads outside the shared cache and deletes each `.bin` as soon
as it is converted, because holding both forms at once is more disk than this
machine has spare.

`Txt2ImgPipeline::load` detects unCLIP from the presence of
`class_embedding.linear_1.weight` and reads its width rather than assuming
2048. Text-only generation on such a checkpoint is refused by name: with
nothing supplied it would run on the zero vector — the guidance batch's own
unconditional row — and return a perfectly ordinary image of nothing in
particular.

**LCM sampling works**, `--sampler lcm`, in both SD 1.5 and SDXL. It is not
another ODE solver: a consistency model maps any point on the trajectory
straight to its origin, so the loop jumps to `x0` and re-noises out to the next
sigma rather than integrating. Four steps instead of twenty.

With `lcm-lora-sdv1-5` at 4 steps and `--cfg-scale 1.0`: a clean photographic
crab in **31 s**, against 57 s for 8 steps of `dpmpp2m` that came out blown
out. Two controls make it a demonstration rather than a hope — the LCM sampler
*without* the adapter gives a brown blob, and the adapter without the LCM
sampler blows out. Each piece is necessary.

Two things make it look broken if missed, both documented on the module:
guidance must be ~1, because the distillation folded one in already; and the
timesteps are a fixed subset of the distillation ladder
(`[999, 759, 519, 279]` at four steps), not an even spread.

**AnimateDiff motion modules are ported, wired and verified** — the module at
**2.662e-7**, and the whole UNet with all 21 inserted at **5.440e-6**.
`UNet2DConditionModel::new_with_motion` installs them through the same
construction-scoped thread-local `unet::ip` uses, one after each resnet, and
refuses unless all 21 are consumed.

The end-to-end comparison is the one that matters: *where* each module lands is
invisible to a per-module check, since every insertion order keeps every shape
valid. The order turned out right — down, then mid, then up, matching this
UNet's own construction order, unlike the IP-Adapter's index list.

A motion module is a small transformer whose attention runs across *frames*
rather than pixels, and our own `Attention` and `FeedForward` are reused
verbatim — the checkpoint's names match theirs exactly. **21 of them, one per
resnet**: down 2x4, mid 1, up 3x4. Note `down_blocks.3` has two despite having
no attentions, which is the clue that they attach to resnets rather than to
transformers.

Three things were wrong on the first attempt and **every one of them kept all
shapes valid**, which is why this was verified numerically rather than by
inspection:

- **The positional encoding goes on the normed states inside each attention
  path, and twice** — once before `attn1`, once before `attn2`. Adding it once
  to the residual stream is the natural reading and is wrong by ~3.
- **The GroupNorm spans frames.** `[b*f, c, h, w]` is regrouped to
  `[b, c, f, h, w]` first, so each group's statistics are taken over the whole
  clip. Normalising per frame is wrong by ~3.
- **The temporal permute happens once, in the module**, not per block: the
  blocks receive `[b*h*w, f, c]` already.

Frame count is ambient (`motion::with_frames`), for the same reason the
IP-Adapter's weights are: it must reach 21 modules and is uniform. The UNet has
no frame axis — frames ride on the batch.

**On MLX this is `crates/sd-models/src/mlx/motion.rs`, and the ambient state is
gone.** All three traps above transferred unchanged and are stated in the
module rather than rediscovered. Two things came out simpler:

- **No ordered name list, and no counter.** `MotionSource::sd15_names` exists
  because candle builds modules in construction order and has to hand them out
  one at a time. The MLX blocks already know their own prefix, so the path is
  `{prefix}.motion_modules.{i}` and the mid block's is
  `mid_block.motion_modules.0`. The 21-entry list and the thread-local that
  serves it have no MLX counterpart.
- **No `with_frames` guard.** The frame count travels in `Motion { weights,
  frames }`, inside the `Adapters` bundle the pass already carries.

`Adapters` replaced the `ip`/`objs` argument pair across `down_block`,
`down_pass`, `mid_block` and `up_block` — the list had outgrown its slot and
motion would have been the third. `unet_forward_with` keeps its old signature
and delegates; `unet_forward_adapters` is the struct form, and the only one
that can carry motion.

**NHWC changes where the regroup lands.** candle folds `[b*f, c, h, w]` to
`[b, c, f, h, w]`; channels-last means the MLX side folds to
`[b, f*h, w, c]` instead. Group statistics span the clip either way, which is
the property that matters — but the shapes are not transliterations of each
other and reading one as the other gives the per-frame normalisation this
document warns about.

Four tests gate it: the module against the fixture (**7.629e-6** at peak
4.611), the whole 21-module UNet (**4.593e-5** excess, inside even the plain
`ATOL`), a perturbation probe, and a check that the adapter changes the output
at all.

**The perturbation probe took two attempts, and the first was a wrong test
rather than a wrong module.** The obvious invariant — feed four identical
frames, demand four identical outputs — is false by construction: the
positional encoding differs per frame, so identical inputs *must* give
different outputs. The probe that works pokes one pixel of one frame and
compares how far the response travels along each axis: **4.145e-1** to the same
pixel in other frames against **1.647e-2** to other pixels. The second number
is not zero and cannot be — GroupNorm's statistics span the clip, so everything
leaks into everything through the mean and variance. That leak is what the
ratio measures against.

Its first run measured 3.365e-1 against 2.245e-1, a ratio of 1.5. That was the
probe's grid, not the module: at 2x2 a single poked pixel is a sixteenth of
each group's population, so the norm leak swamped the signal. At the UNet's own
8x8 the separation is 25x.

## Closing the MLX test gap, 2026-07-31

After the model ports, twenty-two candle test binaries had no MLX counterpart.
Most were already covered under a different name — `golden_unet_attention` and
`golden_unet_blocks` by `mlx_golden_unet`, `golden_clip_encoder` by
`mlx_golden_clip`, `golden_flux_transformer` by `mlx_golden_flux`. Some never
need one: `golden_clip_tokenizer` touches no tensor, and `metal_parity` /
`metal_decoder_parity` compare candle CPU against candle Metal and die with
candle. Six were real gaps, and all six are now closed:

| gap | why it was separate |
|---|---|
| SDXL text encoder 2 | plain `gelu`, penultimate layer, projected pooling |
| CLIP vision (ViT-H) | the tower IP-Adapter and unCLIP condition on |
| Flux VAE | 16 latent channels, no quant convs, a latent *shift* |
| SDXL ControlNet | `addition_embed_type: "text_time"` |
| unCLIP prior | DDPM over a 768-vector, and a load-bearing text mask |
| Tiled VAE | the seam, which no whole-image comparison catches |

Two pieces of shared machinery came out of it, both on the same principle —
**scalar or string logic that both backends need should exist once**:

- `PriorScheduler::coefficients` returns the DDPM step's scalars. The tensor
  work is three lines on either side; the formulation is the part that is easy
  to get subtly wrong, and it is now written down in one place.
- `sd_loader::ldm` is called from the MLX GGUF loader unchanged. A second copy
  of a name mapping is precisely how two backends come to disagree about which
  tensor is which.

Plus `conditioned_temb`, extracted from the UNet's forward so an SDXL
ControlNet builds the identical conditioning vector rather than a parallel one.

### Four tests failed first on their own premises

Worth recording, because in each case the instinct was to suspect the port:

- **The AnimateDiff perturbation probe.** "Four identical frames must give four
  identical outputs" is false by construction — the positional encoding differs
  per frame.
- **The img2img round trip.** It encoded `encoder_input` from the VAE fixture,
  which is `torch.randn` — noise no autoencoder can represent. The fixture's
  `image` is the decoder's own output and round-trips at 0.0373.
- **The Flux VAE round trip.** The fixture unscales `latent`, not
  `encoder_scaled_mean`.
- **The unCLIP prior.** The reference forward was dumped at timestep 500;
  `step_timestep` belongs to the scheduler fixtures beside it.

**A fixture named for its role is not evidence of its contents.** Three of the
four were fixed by reading `xtask/golden/dump_reference.py` rather than by
changing any code.

### The ledger, so the next reader does not have to rebuild it

28 MLX test binaries against 43 candle ones. The gap is not 15 ports:

| candle binary | status |
|---|---|
| `golden_{clip_encoder,clip_vision,controlnet,controlnet_sdxl,esrgan}` | ported |
| `golden_{flux_transformer,flux_vae,gligen,ip_adapter,motion,prior}` | ported |
| `golden_{sd3,sdxl_text_encoder,sdxl_unet,t5,taesd,unclip}` | ported |
| `golden_{unet,unet_attention,unet_blocks,vae,vae_legacy,vae_tiled}` | ported |
| `gguf_quant_sweep`, `lora_coverage` | ported |
| `golden_clip_tokenizer`, `ldm_names`, `gguf_header`, `seam_invariants` | backend-free — nothing to port |
| `golden_samplers`, `golden_flow`, `golden_flux_sampling` | `sd-sample` is scalar; shared already |
| `fused_{adaln,geglu,group_norm}`, `metal_{parity,decoder_parity,smoke}`, `norm_reduction` | die with candle |
| `flux_schnell_gguf`, `t5_xxl_gguf` | geometry ported; running them needs (b) above |
| `pipeline_smoke`, `api_contract` | need (a) above — SD 1.5 txt2img/img2img/inpaint is done, the rest of the surface is not |

The last two rows are the whole remaining list.

### The workspace would not build under `--features mlx` at all

Found only by running the real verification command rather than a per-crate
one. Seven `sd-cli` examples reach into `sd_tensor::mps` or `sd_tensor::fused`,
which exist only under `metal`, and none was gated — so `cargo test --workspace
--features mlx` died in the build before compiling a single test. They now
carry `required-features = ["metal"]` in `sd-cli/Cargo.toml`, and `sd-cli` has
an `mlx` feature so `--workspace --features mlx` resolves for every member
instead of quietly building that one with defaults.

Worth stating plainly because it is the same shape as the `SD_TEST_CONTROLNET`
omission this document already records: **a command that is not run in full is
not a passing command.** Both were invisible to every narrower invocation.

### Bounds that were derived rather than chosen

Two tests needed a looser bound than `DEFAULT_ATOL`, and in both cases the
number came from a measurement that already existed:

- **SDXL's bigG tower, 5e-4.** The excess is 1.909e-4 on the penultimate state,
  and `diagnose_the_residual` says why that is accumulation: **one element in
  98,560** violates, at reference value 0.038, where 32 layers of f32 leave an
  absolute floor near 2.4e-4. Everything with `|ref| > 1` agrees to 3.077e-4
  relative.
- **The Flux encoder, 2e-3.** Transcribed from `golden_flux_vae.rs`, which
  measured diffusers' own encoder in f32 against f64 at **9.605e-4** — the
  reference's own noise floor. candle sits at 9.606e-4, exactly on it; this
  port at 1.515e-3, which is what a different reduction order costs there. SD's
  encoder measures 1.226e-4 by the same method, which is why it holds to 1e-4.

Neither is a licence to widen. A structural fault in the Flux encoder measures
17.32, not 2e-3.

## img2img and inpainting on MLX

Ported 2026-07-31, gated by `mlx_img2img`. The models underneath were already
gated — the VAE encoder at `mlx_golden_vae`, the UNet at `mlx_golden_unet` — so
what needed testing was the part with no reference tensor to compare against:
whether `strength` and the mask *mean* what they claim. Both fail silently. A
run that ignores its strength still produces an image; an inpaint that quietly
repaints the whole canvas still produces an image.

`Strength` is imported from the pipeline crate rather than reimplemented, on
the same grounds as `sd_sample::Schedule`: `start_index` is arithmetic on two
integers and touches no tensor, so the two backends call the same function and
cannot drift apart. What is new on the MLX side is four things, all in
`mlx/sample.rs` and `mlx/vae.rs`:

- `vae::encode_dist` / `vae::encode` — the moments split into `(mean, logvar)`,
  and the **mean**, unscaled. Not a draw: the sampler supplies all the
  randomness, so drawing here too would add variance the seed does not control.
- `sample::noise_to_sigma` — what makes strength mean something.
- `sample::latent_mask` — 8x8 **maximum**, not mean, for the reason the candle
  side records: one white pixel in a block must free that latent cell, and
  averaging would give 1/64 — an almost-frozen cell producing a hard seam
  exactly at the mask edge. This needed a `max` reduction added to
  `sd-tensor/src/mlx.rs`; only `sum` and `mean` existed.
- `sample::restore_outside_mask` — the composite, **inside the loop**. That is
  what keeps the model's context honest: it sees the true surroundings at every
  step, so what it paints joins up with them.

Measured on the VAE fixture at 8 steps:

```text
  distance from source   strength 0  0.0373   0.25  0.0607   0.95  0.6791
  masked edit            inside 0.4777   outside 0.0543
```

**The first run of this test failed for a reason that was the test's fault, and
it is worth writing down.** It encoded `encoder_input` from the VAE fixture,
assuming it was an image. It is `torch.randn(1, 3, 256, 256)` — white noise,
which no autoencoder can represent. The round trip measured 0.8656 and the mask
containment failed, both because the VAE could not reproduce the source, not
because anything in the pipeline was wrong. The fixture's `image` — the
decoder's own output, and therefore on the VAE's manifold — round-trips at
0.0373. **A fixture named for its role is not evidence of its contents.**

**Two-pass generation works**, `--hires 1024`, and the failure it fixes is
visible rather than theoretical. SD 1.5 asked to compose at 1024 directly
produced **three knights** where one was asked for
(`assets/hires-off-1024-duplicates.png`); composing at 512 and refining at
1024 gives one, sharp (`assets/hires-on-512to1024.png`) — and is *faster*,
93.8 s against 115.1 s, because the first pass runs at the smaller size.

Three upscale modes between the passes, and the choice is not cosmetic:
latent-nearest (default), latent-bilinear, and pixel-lanczos, which is the
only one that pays for a VAE round trip. **Nearest introduces no colours that
were not already there** — every interpolating mode invents intermediate
values, which is right for photographic work and destructive for anything with
a fixed palette.

The second pass draws from `seed + 1`, so the two passes do not draw the same
noise for differently-sized latents.

**AnimateDiff renders**, `--motion-adapter <file> --frames 16`. Verified
through the UNet at the pipeline's own shapes (1.125e-5), and the output now
tracks the reference — see [Next](#next) for the beta-schedule trap that made
this look broken for a long time.

**Step caching works now**, `--cache-threshold`, and the reason it did not
before was the predictor *and* the sampler. SD 1.5, 512, 20 steps, `dpmpp2m`;
`evaluated` is how many of the twenty steps ran the model, which is the exact
saving:

```text
  0.00    20/20    22.6 s   baseline
  0.05    20/20            nothing skipped
  0.10    12/20    20.4 s   clean, mean 15.0/255 from the baseline
  0.20     7/20    15.6 s   clean, mean 23.0/255 — 1.45x end to end
  0.40     4/20     6.7 s   degraded: speckle and smeared edges
```

**0.10 to 0.20 is the usable band**, and 0.20 skips 65 % of the steps for
1.45x end to end — about **2.9x on the denoising itself**, the rest being load
and decode that caching cannot touch. The predecessor bought about 9 %.

**Caching is now refused with `euler_a` and `lcm`, and that is the real
finding.** An ancestral sampler draws fresh noise every step, so consecutive
predictions never stop moving. Measured relative L1 change of the output
between steps, three prompts:

```text
  euler_a    0.34 .. 0.90    never small
  dpmpp2m    0.02 .. 0.78    small through the middle of the run
```

There is nothing to reuse, and reusing anyway does not degrade gracefully —
it returns colour speckle (`assets/cache-euler-a-speckle.png`, threshold 0.10,
against `assets/cache-dpmpp2m-threshold020.png` for what a *more* aggressive
setting looks like on the right sampler). The old 9 % ceiling was measured under `euler_a`,
which is the default sampler, so **the feature was being evaluated in the one
regime where it cannot work.**

The predictor is TeaCache's: relative change in the **timestep embedding**,
rescaled through a fitted polynomial into an estimate of the output change,
accumulated. `--example cache_fit` fits it — and fitting rather than borrowing
is the point, since this feature has already been burned once by a constant
taken from the paper that described a different metric.

Two things the fit said that the plan had backwards. On `dpmpp2m` the
timestep embedding is **1.6x the better predictor**, as expected — but under
`euler_a` the *latent* is better, and the latent's degree-4 fit reaches 1.7e4
coefficients over a narrow range, which is overfitting rather than
prediction. And `Progress::evaluated` now reports steps actually run, because
wall clock on this machine has varied 2x on the same configuration and a
timed A/B has lied about this feature before.

Still open: the polynomial is **per model**, fitted on SD 1.5. SDXL and SD 2.x
need their own, which is one command each.

**Runtime LoRA on quantised bases** closes the gap `lora.rs` documented:
`y = QMatMul(x) + ((x @ down^T) @ up^T) * scale`, so the quantised weight is
never touched. Verified against a *dense merge* — and the bound is measured,
not chosen: the quantiser's own noise on that layer is **1.103e-2**, and with
the runtime LoRA it is **1.103e-2**. The correction adds nothing.

This is what lets a style LoRA reach the models that actually fit here — Flux
schnell at Q4_K is 6.32 GB and runs, where dense flux-mini does not fit at
all.

**GLIGEN generates from boxes**, `sdrs ground --box "0.05,0.4,0.45,0.95=a
wooden bench"`. The only conditioning here that addresses *placement*: text
cannot do it reliably and a ControlNet needs a picture of the layout.
Coordinates are **relative**, not pixels — both are plausible readings of four
numbers, so the type says so and the CLI refuses anything outside `0..1`.

Swapping two boxes moves the objects (`assets/gligen-*.png`): the tree sits
centre-right with one layout and hard left with the other, same seed and
prompt.

**Grounding runs for the first 30 % of the schedule, then stops.** Not a flag
inside the loop — the denoise is called twice, with the guard held across the
first call only, because the guard *is* the mechanism. Holding it throughout
costs image quality for placement already achieved, which is what the paper
calls scheduled sampling.

Phrases are **pooled** CLIP embeddings, one per box, not 77-token sequences:
grounding is about what a phrase means as a whole.

**GLIGEN's model side is verified**: the grounding projection at
**7.629e-6**, and a **grounded UNet at 1.419e-6** with all 16 fusers in place.
Pipeline wiring is what remains; see [Next](#next).

The fusers sit **between `attn1` and `attn2`** — grounding conditions the image
tokens before they meet the text, not after — and each is gated by
`tanh(alpha)`, resolved once at load since the gates are learned scalars that
do not move during inference.

**No installation machinery, unlike the other two adapters.** The weights live
at `<block>.fuser` and a transformer block already holds a builder scoped to
itself, so each block asks for its own by name. There is no index list, and so
none of the ordering trap that cost the most time on the IP-Adapter. A
checkpoint without grounding simply has no such tensor.

**Grounding off is exactly ordinary SD 1.5**, tested against a separate
reference: with no tokens supplied every fuser is skipped. That is what makes
*scheduled sampling* — GLIGEN grounds for roughly the first 30 % of the
schedule then finishes free — expressible by dropping a guard part way through
a run rather than by a flag threaded anywhere.

Boxes are expanded into sinusoids at eight frequencies, the timestep-embedding
trick applied to space. **The axis order is the whole subtlety**: the axes are
`(coordinate, frequency, sin/cos)` and flatten as
`(frequency, sin/cos, coordinate)`. Every permutation produces 64 numbers and
loads against the same weights; only one lines up with what the MLP was
trained on, and the rest give grounding tokens that are wrong without being
malformed. That is what the golden comparison is for.

**A masked slot uses a *learned* null, not zeros** — both for the phrase and
the position. Padding with zeros reads as "a phrase whose embedding happens to
be zero", which is a different thing and one the model was never shown. The
reference masks one of three slots off so this is exercised rather than
assumed, and a second test pins that a masked slot does not depend on its
contents.

The checkpoint ships as a pickled `.bin`, so `dump_reference.py gligen`
converts the whole UNet to safetensors as a side effect — 966 tensors.

**Instruction editing works**, `sdrs instruct --prompt "make it winter with
snow" --init-image X`. Different from img2img: that takes a description of the
*result* plus a strength, this takes a description of the **change**, and the
source is held by the model's own conditioning rather than by stopping the
schedule early. `assets/instructpix2pix-crab-winter.png` — the same crab, in
the same pose, in a snowed-over scene.

**Three predictions per step, not two.** Ordinary guidance contrasts a prompt
against nothing; this contrasts three things so that instruction adherence and
image fidelity become independent axes:

```text
  pred = uncond + text_scale * (text - image) + image_scale * (image - uncond)
```

The rows are `[text+image, uncond+image, uncond+zeros]`, and the **zeroed
image latent in the third row** is what makes the middle term mean "what the
image contributes" rather than "what the prompt contributes".

`--image-guidance` is measurably a second axis — mean distance from the source
at 1.0 / 1.5 / 2.5 is **65.7 / 44.4 / 30.6**, monotone, and there is a test
that checks all three rather than one (one would pass with the parameter
ignored, two could pass on noise).

**Two traps, both silent.** The 8-channel `conv_in` is detected from its own
shape, since it is invisible in the cross attention that identifies SD 2.x.
And the source latent is **not scaled by 0.18215** — every other latent here
is, but InstructPix2Pix was trained on the raw encoder output, and scaling it
multiplies the conditioning by 5.5 and returns a plausible image that ignores
the source.

**Regional prompts work**, `sdrs area --region mask.png=prompt` (repeatable).
Each region contributes its own noise prediction, blended by its mask —
composed *before* sampling, so the regions see each other rather than being
generated separately and composited, which leaves visible joins.

**The control is real but soft, and that is inherent.** Two complementary
half-masks with irreconcilable prompts ("red brick wall" over "green grass")
put brick where the mask says and ground below
(`assets/area-conditioning-brick-over-ground.png`), but the model still
resolves the whole frame into one coherent scene. It has to: each step blends
predictions, then the *next* step's UNet sees a single latent and re-imposes
global structure. Anyone expecting a hard partition will be disappointed, and
that is a property of prediction-blending rather than of this implementation.

Two invariants are exact and tested, and they are what pin the normalisation:
an **empty mask is bit-identical to a plain run**, and a **full mask is
bit-identical to a plain run of that region's prompt**. Either would break if
the base leaked in at the wrong weight.

**The mask is mean-pooled to the latent grid, not max-pooled** — the opposite
of `latent_mask` for inpainting, and deliberately. An inpaint needs a cell
freed if *any* pixel under it is free; a region boundary should fade across
the cell it straddles. Reusing the inpaint pooling would give every region an
8-pixel-wide hard edge.

Cost is one UNet call per region per step on top of the base, which is
inherent to conditioning spatially.

**A clip is a batch of frames**, `--frames 8`. The pipeline no longer assumes
batch 1: the latent draw, the guidance concatenation, the timestep tensor and
the sampler's noise draws all follow the frame count, and `--output clip.png`
writes `clip-000.png`, `clip-001.png`, ...

**The frame count is read from the latent**, not the config, inside the loop.
That is the one thing every path already agrees on — and a caller supplying
its own latent through `run_with_latent` sets the count by doing so, which is
exactly what the coherence techniques in the animation issue need.

Three things this had to get right, each of which runs when wrong:

- **Conditioning is per frame.** The guidance batch is
  `[uncond x n, cond x n]`, not interleaved, and the reference UNet does not
  repeat it for you — one row where `n` are expected fails inside the *spatial*
  cross-attention. Interleaving instead runs and guides each frame by another
  frame's conditioning.
- **The guidance split is `narrow(0, 0, n)` / `narrow(0, n, n)`**, matching how
  the batch was concatenated.
- **The VAE decodes one frame at a time.** Frames are independent through it,
  so decoding `n` together only multiplies the largest single allocation — a
  three-frame 512 decode is 6.8 GiB in one call and trips the memory guard.
  Looping is byte-identical at one frame's peak, and there is a test that says
  so.

`frames: 1` is bit-identical to the previous behaviour, which is the property
that matters most here since every still-image path runs through this loop.

Motion modules pick the count up automatically — `denoise_inner` installs
`motion::with_frames` for the run.

**Checkpoints merge**, `sdrs merge --a X --b Y --alpha 0.3`. Loader-level
arithmetic — `(1-alpha)*a + alpha*b` per tensor, on the CPU regardless of the
active device, since moving gigabytes onto an accelerator to add them pays a
transfer for no gain.

What it refuses is the point. Merging is only meaningful within one
architecture, and the failure otherwise is quiet: an SD 1.5 and an SDXL share
enough tensor *names* to produce a file that loads and renders noise. So shape
mismatches and one-sided tensors are both refused — with a count and an
example — rather than skipped, since skipping silently takes one side's
weights and yields a third model nobody asked for. `--allow-unmatched` opts
back in.

Verified on real weights: merging the SD 1.5 VAE with itself at alpha 0.5 is
**bit-identical** across all 248 tensors, and merging it with the UNet is
refused naming `conv_in.bias`.

**Textual inversion works**, `--embedding <file>` (repeatable), triggered by
the file stem. Kilobytes against a checkpoint's gigabytes, which is the whole
point of it.

**A learned embedding has no token id**, so it is *spliced*, not looked up.
The trigger is tokenised like any other word — all it does is reserve
positions — and after the embedding lookup those rows are overwritten with the
learned vectors. Multi-vector embeddings expand the trigger to as many copies
as they have vectors, so a short trigger cannot silently drop the tail.

That needed `embed_tokens` and `forward_embeds` split apart on the text
encoder, with the splice landing *between* them: position embeddings are added
after, because a learned vector occupies a position like any other token.

Three tests, and each catches something the others do not: a prompt naming the
trigger must differ from the same prompt without the embedding loaded; a
prompt *not* naming it must be **bit-identical**, which is what fails if the
splice writes to arbitrary positions; and a wrong-width embedding is refused,
since SD 2.x's 1024 in an SD 1.5 prompt would otherwise be a shape error from
inside the transformer.

Three file layouts are accepted (`emb_params`, `string_to_param.*`, and a bare
single-tensor file) because a user downloading one has no reason to know which
tool wrote it.

**IP-Adapter works**, `sdrs txt2img --ip-adapter <ckpt> --image-encoder <dir>
--ip-image <img>`, verified against diffusers at **2.267e-6** through the whole
UNet and **1.016e-6** at scale 0.

The A/B that shows it: same prompt ("a photograph of a castle on a hill"),
same seed, a crab photograph as reference. `--ip-scale 0` gives a castle;
`--ip-scale 1` gives a crab on sand — the reference overrides the prompt
entirely at full strength, which is why published guidance uses 0.5-0.7. Both
in `assets/`.

**Decoupled cross-attention, not concatenation.** Each of the sixteen cross-
attention layers gains a second key/value pair and returns
`attn(text) + scale * attn(image)`, with `to_out` applied once to the sum.
Appending the image tokens to the text ones is a *different function* —
attention is not linear in K and V — and a plausible-looking one.

**The image tokens ride on the end of the context tensor** and the attention
layers split them off. That convention is why nothing between the pipeline and
the attention needed a new parameter: no block type, no transformer, no UNet
signature changed.

**The weights reach sixteen layers without sixteen parameters either.** The
source is installed thread-locally for the duration of
`UNet2DConditionModel::new_with_ip` and each cross-attention pulls the next
slot as it is built — construction-scoped, released by a guard, never read
outside that one call. It refuses if the count does not come out exactly at
sixteen, because consuming too few would leave the deepest layers
unconditioned and still render.

**The index order was the risk, and it is pinned.** The checkpoint numbers its
entries by diffusers' flat processor order — down blocks, up blocks, then
**mid** — while this UNet builds down, **mid**, up. Entries sit at *odd*
indices because that list alternates self- and cross-attention. So slot `i`
maps to key `2 * order[i] + 1` with
`order = [0..5, 15, 6..14]`. A wrong mapping mostly fails to load, but
*between the two 1280-wide regions it would not*, and the image would simply
be wrong — which is why the verification is end-to-end through the UNet rather
than per module.

The strength is thread-local rather than a process global, unlike
`conv::seamless`: two threads may want different strengths, and a shared one
made two tests race.

**CLIP's vision tower is ported and verified** — 1.270e-4 on the sequence,
3.485e-6 on the pooled vector, against `transformers`. This is IP-Adapter's
foundation, and it is the part that was missing; the adapter itself is next
(see [Next](#next)).

Structurally it is the text tower with a different front end, and
`ClipEncoderLayer` is reused literally. Three things differ and each is a
silent-wrongness bug if missed:

- **No causal mask.** An image has no order to respect. The layer takes a mask
  either way, so this passes zeros — the additive identity for attention
  logits. Reusing the text mask would let each patch see only those before it
  in raster order and still emit the right shape.
- **The class token is prepended**, because the pooled output reads position 0.
- **`pre_layrnorm`** is spelled that way in the checkpoint, so that is the
  name this loads. Correcting the spelling fails to find the tensor.

**IP-Adapter consumes the *projected* embedding, not the pooled one.** The
tower is 1280 wide and `visual_projection` narrows it to 1024, which is what
`image_proj` expects — `image_embeds` is a separate method from `pooled` for
that reason. The widths differ, so the mistake fails to load rather than
running wrong.

The adapter's own layout is confirmed: 4 image tokens from
`Linear(1024 -> 4*768)` plus a LayerNorm, and 16 `to_k_ip`/`to_v_ip` pairs at
**odd** indices 1..31 — odd because the flat processor list alternates self-
and cross-attention and only cross-attention gets them. Their order is
**down blocks, then up blocks, then mid**, read off the shapes
(320,320,640,640,1280,1280 | 1280x3, 640x3, 320x3 | 1280). That ordering is
worth having written down: it is not the order the UNet runs in, and guessing
it wrong would put every correction on the wrong layer while every shape stayed
valid.

**Cancellation, a per-step conditioning hook, and prompt-budget queries** —
the remaining asks from the integration issue.

`Cancel` is a token on the config rather than a callback return value, so the
ordinary `ProgressFn` stays a plain `FnMut` and callers who never cancel write
nothing. Checked at the top of each step, and the error names how far it got.

`run_conditioned` takes a slice of pre-encoded `Conditioning` and a
`(step, total) -> index` selector. That covers two asks at once: encode once
and reuse across a sequence, and *vary* the conditioning per step. The
motivating result is gating a negative prompt to a middle window of the
schedule — 65.1 % to 80.4 % on object removal (Ban et al., ECCV 2024) — which
is not expressible with one fixed conditioning. A single-entry slice with a
constant selector is bit-identical to the plain run, and there is a test that
says so; a second test gates a negative to a window and asserts the image
*changes*, which is what fails if the selector is ignored.

`ClipTokenizer::content_token_count`, `content_capacity` and `will_truncate`
answer the budgeting question. The counts are unintuitive and now pinned:
`"16-bit"` is **four** tokens and `"32x32"` is **five**, because CLIP's BPE
splits digits singly, and every comma costs one. Over the limit this
implementation **truncates, not chunks** — a term past the boundary is
discarded rather than encoded into a second window, which is a real difference
from `stable-diffusion.cpp` and is now tested rather than left to be
discovered.

**Explicit latent in and out**, `initial_latent` and `run_with_latent`, which
is what makes frame-to-frame coherence reachable by a caller: shared initial
latents across frames, correlated noise, interpolation between keyframes,
carrying a latent forward. None of that is expressible through a seed.

Two tests hold it together. `initial_latent` fed back to `run_with_latent`
must reproduce the seeded run **bit-identically** — so the initial draw still
happens even when a latent is supplied, keeping the sampler's own noise
sequence aligned — and a *different* latent must produce a different image,
which is the test that fails if the argument is quietly ignored.

**Determinism is now a tested guarantee**, not an assumption: same seed and
parameters give byte-identical output across runs *and across pipeline
instances*, the second of which is what would catch a load path that is not
bit-stable.

**Several ControlNets can be bound at once.** `ControlConfig::controls` is a
`Vec<Control>`, one map and strength per attached net, and the corrections are
summed before the UNet sees them — which is what diffusers does, and is
correct because each was trained against the same frozen base, so they are
independent additions rather than alternatives. Pose for a figure plus depth
for the scene is the motivating pairing. A count mismatch is refused rather
than zipped to the shorter list, which would otherwise hand the wrong net the
wrong hint at shapes that stay valid.

**4x upscaling works**, `sdrs upscale --model <esrgan_x4.safetensors>
--input <img>`, verified against the reference RRDBNet at **1.669e-6**. Real-
ESRGAN is a pure convolutional network — no attention, no normalisation, no
diffusion — so it runs after generation and knows nothing about it.
`assets/esrgan-crab-2048.png` is 512 -> 2048.

**It found a silent-corruption bug in candle's Metal convolution, and the
boundary is exactly `i32::MAX`.** A 3x3 convolution over `out_h * out_w`
positions at 64 channels builds an im2col matrix of `out_h * out_w * 64 * 9`
elements. Past `i32::MAX` the Metal kernel returns a dark, horizontally banded
image — **no error, no failed command buffer, nothing to catch**. CPU renders
the same input correctly, which is what identified it as candle's rather than
this port's.

The threshold was measured, not inferred, and it lands where the arithmetic
says it should:

```text
  output 1928 px   2,141,097,984 elements   under i32::MAX   correct
  output 1936 px   2,158,903,296 elements   over  i32::MAX   corrupt
  sqrt(i32::MAX / (64*9)) = 1930
```

`upscale_tiled` splits anything above it, 384 px tiles with 16 px of context,
and the seam is not visible: the column-to-column jump at the tile boundary is
0.24 where the largest jump anywhere in the image is 4.43. Against a one-pass
CPU render, 98.8 % of pixels agree within 16/255 — the rest is high-frequency
detail where a tile has less context to work with, which is the cost of tiling
and not a defect.

`upscale_in_tiles(image, tile, pad)` takes both explicitly so the tiling is
testable without a 2000 px image: give every tile a padding larger than the
image and the result must be **exactly** the one-pass result, which is what
pins the crop offsets and the stitching order.

**SD 2.x renders**, `sdrs txt2img --model <sd21-dir>`, verified against
diffusers at **3.696e-5** on the UNet output with all twelve skips and the mid
block inside the same bound SD 1.5 uses. `assets/sd21-crab-512.png`.

Almost all of it was already there. SD 2.x is SD 1.5's block geometry with a
1024-wide OpenCLIP ViT-H behind it: SDXL-style head counts (`[5, 10, 20, 20]`,
all 64 wide), `Linear` transformer projections rather than 1x1 convolutions,
and `ClipActivation::Gelu`. Two things are worth knowing:

**The text encoder has 23 layers, not 24.** SD 2.x conditions on the
penultimate hidden state, so the conversion to diffusers format drops the last
layer outright — the shipped checkpoint has 23 and the ordinary "last layer,
then `final_layer_norm`" path is then exactly right. Reaching for
`penultimate_hidden_state` here would silently condition on layer 22.

**It is v-prediction, and that is invisible in the weights.** The model outputs
`v`, not noise, so `x0 = x/(1+sigma^2) - v*sigma/sqrt(1+sigma^2)`. Sampled as
epsilon it loads, runs, and reports nothing wrong — it just returns saturated
colour noise. That is measured: forcing the detection to `Epsilon` for this
checkpoint and rendering the same seed gives no crab at all, where
v-prediction gives a sharp one.

Both the architecture and the prediction type are **detected, not flagged**.
The architecture comes from a tensor shape — the cross-attention key
projection is 768 wide for SD 1.5 and 1024 for SD 2.x — via
`sd_tensor::tensor_shape`, which reads a safetensors header without loading
data. The prediction type comes from a substring test on
`scheduler_config.json`, deliberately: the token is unambiguous and a JSON
parser for one boolean is not worth a dependency this workspace otherwise does
without.

**Stock SD 2.x is gated on HuggingFace** — every `stabilityai/stable-diffusion-2*`
repo 401s, and so do the community mirrors, while unrelated repos return 200.
The verification above uses `friedrichor/stable-diffusion-2-1-realistic`, an
open fine-tune with byte-identical architecture and the same v-prediction
scheduler. Anyone with an accepted licence can point the same dumper at the
stock checkpoint.

**Flux and SD 3.5 have CLI subcommands**: `sdrs flux --model <dir>` and
`sdrs sd3 --model <dir>`. `--model` is a *directory* and that was the whole
design question — Flux needs four checkpoints plus two tokenizers, SD 3 six
files, and naming each would be flags nobody remembers. `paths_in` and
`sd3_paths_in` already took a directory, so the answer was sitting in the
codebase. Both support `--taesd`, `--preview-every` and `--stream`; SD 3 also
takes `--encoders-on-cpu`.

**Step previews work for all four architectures** — SD 1.5 and SDXL via
`--preview-every N`, Flux and SD 3.5 via `SD_PREVIEW_EVERY` on their examples.
This is what the tiny decoder was for: a 20-step run no longer sits blank for
two minutes.

**Flux and SD 3.5 needed a different `x0`, and getting it wrong would have
looked plausible.** They are rectified flow: the model predicts a *velocity*,
not noise, so the estimate is `x - sigma*v` — the inverse of the forward
process `x = sigma*noise + (1-sigma)*x0` — and not the DDPM `x - sigma*eps`.
Flux also carries its latents 2x2-*packed* through the loop, so the estimate
has to be unpacked before anything decodes it; handing over the packed form
gives a tensor of an entirely reasonable shape that decodes to nonsense.

Both are confirmed by a property that falls out for free and is worth
keeping: **at the last step the x0 estimate equals the returned image
exactly.** `sigma_next` is 0 there, so the sampler lands on `x0`; a wrong
formula would not land. Measured against the final image, mean absolute
difference per 8-bit channel:

```text
  flux  step 1/4   18.1      sd3.5  step  5/20   20.6
        step 2/4   10.2             step 10/20   10.8
        step 3/4    0.2             step 15/20    5.5
        step 4/4    0.0             step 20/20    0.0
```

Flux schnell's estimate after a *single* step is already close to final
(`assets/flux-preview-step01-of-04.png`) — which is what four-step
distillation buys, seen directly.

**All four TAESD checkpoints are ported**: `taesd` (SD 1.5/2.x), `taesdxl`,
`taesd3` and `taef1`, the last two at 16 latent channels. Same architecture
throughout, so `TinyDecoder::new` takes the width; a 4-channel file in a
16-channel slot fails to load, which is the one mismatch in the family that
is loud.

**The preview is the `x0` estimate, not the sampler's latent**, and that is
the whole design. The latent at step 5 of 20 is `x0 + sigma*noise` with sigma
still near 4, so decoding it shows a field of coloured noise — the first
version did exactly that and looked broken. The model's `x0` prediction is
already computed every step; decoded, it is a blurry crab at step 5 that
sharpens as the run proceeds (`assets/preview-step05-of-20.png`). Every
diffusion UI shows this and now so does this one.

`ProgressFn` therefore carries a `Progress` struct rather than three
positional arguments. A fourth positional `&Tensor` would have been easy to
ignore, which is the opposite of the point.

**SDXL has it too**, with `taesdxl` — verified at 1.61e-5 decoding, 7.03e-5
encoding, and pinned as a *different* checkpoint from `taesd`: the two share an
architecture, so the wrong one loads without complaint and decodes in visibly
wrong colours. Nothing in the code can tell them apart, so the path is the
caller's to get right.

This is where it pays most. `decode_tiled` splits anything above a 64-latent
edge, so a 1024 VAE decode is four tiles with their seams; TAESD does it in one
pass. Measured at 1024, 4 steps, **both orderings**:

```text
  vae   161.7 s      taesd 117.4 s     (reversed)
  taesd 123.8 s      vae   164.9 s
```

38-48 s, direction consistent. The absolute numbers are inflated — the machine
was at load average 20-24 from unrelated work — which is also why they cannot
be compared with the 86.5 s in the table above. `assets/sdxl-1024-taesdxl-crab.png`.

**TAESD decodes**, `sdrs txt2img --taesd <ckpt>`. About 5 MB of 3x3
convolutions against the VAE's 330 — no attention, no GroupNorm, no sampling
head — verified against `diffusers.AutoencoderTiny` at **1.86e-5** decoding
and **1.24e-5** encoding.

Compared through the public `decode`/`encode` rather than the layer stack, on
purpose. TAESD's architecture is too simple to get wrong; what is easy to get
wrong is its *conventions* — its `scaling_factor` is 1.0, so it takes the
sampler's latent with **no `/ 0.18215`**, and it soft-clamps its input with
`tanh(x/3)*3` and returns `2x - 1`. Each of those is a plausible image and no
error if missed.

**The decode is 8-11 s cheaper at 512** on this machine, measured at one step,
where decode is a large fraction of the run rather than lost in it. At twenty
steps that is about 7 % of a 125 s run, and **below run-to-run variance** — the
first 20-step pair suggested a 25 s saving and the second suggested none. The
one-step figure reproduces in both directions and is the one to trust.

**The memory win is 7x, and it is the activations, not the weights.** An
earlier note here said the win was unrealised because `with_taesd` kept the
VAE resident. That was wrong twice over: the 189 MB of VAE decoder weights is
now dropped (`Decoder` is an enum, so attaching TAESD *replaces* rather than
adds, with a synchronise because a drop alone frees nothing on Metal) — and
that 189 MB was never the interesting number anyway. Decoding is:

```text
  latent edge   output      VAE            TAESD
  64            512 px      3.43 GB        0.49 GB     7.0x
  128           1024 px     3.22 GB *      1.71 GB
```

`* tiled.` `decode_tiled` splits anything above a 64-latent edge, so the VAE
holds ~3.4 GB at any size by **seaming the image** instead. TAESD does 1024 in
one pass. Reproduce with
`cargo run --release -p stable-diffusion-rs --example decode_peak -- taesd 64`.

Measuring this end to end is what hid it: `sdrs`'s peak RSS is dominated by
the UNet's 3.4 GB, so the two decoders looked 0.14 GB apart — which is only
the weights. The decode had to be isolated before the real number appeared.

Output agreement with the VAE is mean 18.9/255, max 214 — TAESD is genuinely
lossy, which is the trade, not a defect.

**ControlNet works**, `sdrs controlnet --controlnet <ckpt> --init-image X`, for
SD 1.5. A ControlNet is a copy of the UNet's down and mid stack that reads a
control map and emits one correction per skip connection; the UNet adds them
before the up pass consumes them and is otherwise untouched. So the module is
short — `DownBlock2D`, `MidBlock2DCrossAttn` and `TimestepEmbedding` are the
UNet's own, reused verbatim — and the only new parts are the hint encoder and
the zero convolutions.

Verified against `lllyasviel/sd-controlnet-canny` **correction by correction**,
all thirteen: worst excess **1.45e-5** against a 1e-3 bound. Comparing them
individually rather than as one tensor is the point — a ControlNet has no image
of its own, so those thirteen are its entire observable behaviour, and the
index of the first bad one localises the fault.

The A/B that makes it a demonstration rather than a hope: same prompt, same
seed, `--control-scale 0` gives a centred crab on a marble pedestal — exactly
what the prompt asks for. At scale 1 the crab takes the source photo's sprawled
pose and no pedestal appears at all, because the edge map says "sand". Both in
`assets/`, with the edge map beside them.

**Canny edge detection is built in** (`crate::canny`), so no second tool is
needed. Full four stages, and the last two are what matter: non-maximum
suppression, which makes edges one pixel wide instead of thick bands, and
hysteresis, which is a flood from the strong pixels rather than a raster sweep —
a sweep misses any chain running backwards.

**The thresholds matter more than they look.** The defaults (0.1 / 0.2) are
right for clean subjects and far too sensitive for a textured photograph: on a
crab on sand they turn the sand into a field of speckle, which the model then
faithfully renders as background hatching. 0.25 / 0.45 gives the clean result
in `assets/`. That the speckle *did* come through is itself evidence the
control is being followed.

**Inpainting works**, `sdrs inpaint --init-image X --mask M`, on any SD 1.5
checkpoint — no 9-channel inpaint UNet required. The mask follows the universal
convention: **white repaints**.

Two properties hold exactly, and both are tested:

- **The untouched region is bit-identical to the input.** Latent blending alone
  cannot give that, because it preserves the *encoded* original and a VAE round
  trip is lossy; the result is composited against the original in pixel space
  at the end.
- **A mask of all black returns the input unchanged**, max diff 0 over the
  whole image.

The latent mask is 8x8 **max**-pooled, not averaged. A latent cell is not a
pixel: if any pixel under it is free, the whole cell must be free, or the cells
straddling the mask edge end up nearly frozen and leave a hard seam exactly
where it shows most.

A visible seam remains on *large* holes — a half-image mask is the worst case,
and the one in `assets/` shows it. That is the honest limit of blending
without a dedicated inpaint checkpoint: the model never sees the mask, so it
has no way to compose across the boundary. Fixing it means the 9-channel UNet,
which is a checkpoint change, not a code change.

**A rounding bug in every image this project has saved was found by that
exactness check** and is fixed. `tensor_to_rgb8` used `v as u8`, which
truncates; `b/127.5 - 1` followed by `(x+1)*127.5` lands just below the
integer often enough that a load-and-save darkened most pixels by one level.
It now rounds, matching diffusers. This is why the inpaint invariant was worth
asserting as *exact* rather than *close*: at max-diff 1 it reads as noise, and
a tolerance would have hidden a real defect in the output path.

**LoRA adapters load and merge**, for SD 1.5's dense path:
`sdrs txt2img --lora <file> --lora-scale <f>`, or
`Txt2ImgPipeline::load_with_lora`. Verified against a published adapter
(`latent-consistency/lcm-lora-sdv1-5`, 278 layers): every entry finds a weight,
each is merged exactly once, `--lora-scale 0` is **bit-identical** to no LoRA
end to end, and scale 1 changes exactly those 278 tensors and nothing else.
A partial match is refused rather than rendered — see `PipelineError::LoraMismatch`.

Quantised bases are not supported: merging into them means dequantise and
requantise, which is lossy. That needs runtime application instead, which is
the same design question as the streamed-block cache.

**Attention now has four paths**, and `ops::attention_with_path` reports which
one ran: `Fused` (candle's Metal SDPA), `FlashCpu` (candle's CPU flash kernel,
new — taken only at or below 512 tokens, and never for T5, see
[roadmap.md](roadmap.md#cpu-flash-attention-a-short-sequence-win-not-a-general-one)),
`Chunked` and `Naive`. Compare paths with `--example attention_path`, which
prints both CPU timings side by side and flags a bad dispatch choice. Note its
rows are all unmasked, so they do not tell you what a *masked* caller gets —
that is pinned by tests instead.

## Next

In priority order. Struck-through items are done and kept for their reasoning.

### ~~1. A Metal smoke test~~ — done

`tests/metal_smoke.rs`, gated on the `metal` feature. One forward per
architecture on the GPU, asserting only that it loads, runs, and returns
finite non-flat values — correctness stays the golden suite's job.

It earned its place on the first run by catching a real bug: `Txt2ImgPipeline::run`
on an InstructPix2Pix checkpoint failed with `in_channel mismatch between input
(4, groups 1) and kernel (8)` from deep inside a convolution. That is now
`PipelineError::NeedsInstruct`, which names the fix.

Two things worth keeping about how it is written:

- **A memory refusal skips, it does not fail.** "This machine is busy" and
  "this model is broken on the GPU" are different answers, and a smoke test
  that goes red when something else is running teaches people to ignore it.
- **`the_smoke_list_covers_what_the_repo_links` fails if a model directory is
  linked but not exercised**, so adding an architecture without adding GPU
  coverage is caught rather than remembered.

### ~~1. unCLIP — image-embedding conditioning~~ — done

`sdrs unclip`, against `diffusers/stable-diffusion-2-1-unclip-i2i-h` — the
stock `stabilityai/stable-diffusion-2-1-unclip` is gated but the diffusers
mirrors are open. See the section above for the numbers and the two traps.

It turned out to be the *smallest* remaining capability rather than a full
architecture: the UNet is SD 2.x with a `class_embedding`, so `UNetConfig`
gained one `Option` and the blocks were untouched. What took the time was the
noise augmentation's tolerance, which is a story about measuring floors
properly rather than about diffusion.

**The text-to-image half is done too**, `sdrs unclip --model models/unclip-t2i
--prompt "..."` with no reference image: the prior samples an image embedding
from the prompt and the image half proceeds exactly as if a photograph had
produced one. `assets/unclip-t2i-crab-768.png`, 768px in 67 s.

It was smaller than it looked. The prior is a 20-block, 2048-wide transformer,
and most of what it needs already existed: `timestep_embedding` and
`TimestepEmbedding` verbatim, the masked-attention primitive CLIP's text tower
already uses, and — exactly — the `squaredcos_cap_v2` ladder written for the
noise augmentation, which is the prior's sampling schedule too. What was
genuinely new is one assembly and one sampler.

**`diffusers/stable-diffusion-2-1-unclip-t2i-h` will not load here.** Its
prior emits a 768-wide ViT-L embedding, while its image half is the ViT-H
one — `image_normalizer` 1024 wide and a UNet
whose class projection is 2048, being twice that. The two ends do not meet.
**`-t2i-l` is the consistent pairing**: 768 throughout, class projection 1536.
`with_prior` checks the two widths and names the mismatch rather than letting
it fail inside the augmentation.

Worth knowing when assembling model directories: `-i2i-h` and `-t2i-h` share
their UNet, VAE and text encoder **byte for byte** (identical sha256), but
`-t2i-l` shares none of them. One directory serves the first two; the third
needs its own, which is why there is a `models/unclip-t2i` beside
`models/unclip`. The *prior* is identical across both t2i mirrors, so it is
downloaded once.

Five things the prior gets to be wrong about, all of which run:

- **The sequence is `[77 text | pooled text | timestep | latent | prd]`**, 81
  positions, and the answer is read from the **last** one — a learned `prd`
  token that exists to have somewhere to put it. Reading the latent's position
  returns a well-shaped vector that is not the prediction.
- **The attention mask is load-bearing**, uniquely here. Every other CLIP
  consumer in this project ignores the tokenizer's mask because SD conditions
  on all 77 positions, padding included. The prior masks padding *and* applies
  a causal mask, and the reference's ten-token prompt moves the prediction by
  **0.604** between masked and unmasked — so both are dumped and both compared.
- **The model predicts the sample, not the noise.** `prediction_type:
  "sample"`, so there is no `x0` to derive; and the variance is
  `fixed_small_log`, where diffusers' `_get_variance` returns a *standard
  deviation* while every other variance type returns a variance. Squaring or
  rooting it once more gives a quieter run and a plausible image.
- **The feed-forward is plain GELU, not GEGLU** — every SD transformer here is
  GEGLU. The projection widths differ (8192 against 16384), so this one fails
  to load rather than running.
- **`clip_mean`/`clip_std` un-whiten the result.** Skipping it hands the image
  half a vector at the wrong scale.

Verified against diffusers:

```text
  prior transformer, masked        3.2e-6
  prior transformer, unmasked      4.7e-6     the mask is worth 0.604
  prior text encoder, hidden       6.3e-6
  prior text encoder, projected    9.7e-7
  one DDPM step                    4.8e-7
  the final step (no variance)     0.0        exactly
  the t2i UNet under it            2.0e-3     floor 1.5e-3
```

### ~~1. A real predictor for step caching~~ — done

TeaCache's predictor is in: the relative change in the timestep embedding,
rescaled through a polynomial fitted by `--example cache_fit`. 0.20 skips 65 %
of the steps for a clean image — 1.45x end to end, 2.9x on the denoising —
against about 9 % before. See the section above.

The finding worth carrying: **the old ceiling was the sampler, not the
predictor.** `euler_a` re-noises every step, so consecutive predictions never
stop moving; caching is now refused there rather than producing speckle. The
whole feature had been evaluated in the one regime where it cannot work.

Remaining: the polynomial is per model and fitted on SD 1.5. SDXL and SD 2.x
want their own, which is one command each.

### 1. Newer architectures worth the port

Scouted rather than surveyed — every line below is from the published
`model_index.json` and file listings, not from a blog post. **The previous
version of this entry was wrong about HiDream in a way that would have cost
whoever picked it up a day.**

| | transformer (Q4) | text encoders | VAE |
|---|---|---|---|
| **HiDream-I1** | 10.7 GB | CLIP-L ✅, CLIP-G ✅, T5 ✅, **Llama** ✗ | `AutoencoderKL` ✅ |
| **Qwen-Image** | 11.9 GB | **Qwen2.5-VL** ✗ | `AutoencoderKLQwenImage` ✗ |
| **FLUX.2 [dev]** | 19.3 GB | a 10-shard LLM ✗ | `ae.safetensors` |

**HiDream-I1 is the one to start with**, and for the opposite of the reason
this entry used to give. It claimed "8B pixel-native, no external VAE and no
separate text encoders... *less* work than its size suggests". Its
`model_index.json` says `vae: AutoencoderKL` and lists **four** text encoders.
It is more work than the old note claimed — but three of those four already
exist here, and the VAE is the standard one this project has verified since
milestone 1. So what is actually new is the transformer plus a Llama encoder,
and candle ships Llama already.

**Qwen-Image needs two new towers**: `Qwen2_5_VLForConditionalGeneration` — a
vision-language model, not a text encoder in the CLIP or T5 sense — and its
own `AutoencoderKLQwenImage`. Largest of the three by work, not by size.

**FLUX.2 is gated and `[klein]` is not obtainable.** `FLUX.2-dev` is
`gated: auto` and its raw files 401 without credentials; `FLUX.2-klein` 401s
outright, so the old note that it is "Apache-2.0" describes a licence on
something that cannot currently be downloaded. The quantised mirror
(`city96/FLUX.2-dev-gguf`) is ungated, which is the practical route — as it
was for Flux.1 schnell — but its text encoder is ten shards of a large LLM and
is the real cost.

All three publish Q4 GGUFs, so the quantised path this project already has is
the one to use; none of them fits dense on 36 GB.

### 2. Extend streaming past Flux and SD 3.5, and measure on a discrete GPU

`Residency::Streamed` works for Flux and SD 3.5, both quantised. Gaps:

- **Dense checkpoints cannot stream.** `quantized::to_device` moves quantised
  block bytes verbatim, which is what makes it cheap and bit-exact; a dense
  model would move 4x the bytes with no shortcut. flux-mini — the model that
  most needs this at 12.8 GB dense — is therefore the one that cannot have it.
- **Nothing prefetches**, and per the profile below that is not the first fix.

And the honest one: **the payoff has never been measured on the hardware it is
for.** On unified memory the host copy comes from the same pool. On a discrete
card it should take VRAM from 6.66 GB to ~192 MB by construction. **If a CUDA
machine appears, measure this before anything else here** — CUDA is also
entirely untested, so that visit should cover both.

### Also open

- ~~**ControlNet for SDXL.**~~ **Done**, and it was not config-only — the
  entry used to claim it would be "a config and a checkpoint, not new code".
  `ControlNet::new` took a `UNetConfig` but built only a `TimestepEmbedding`
  from it, ignoring `cfg.addition`, and `forward` had nowhere to put a pooled
  embedding or time ids. An SDXL ControlNet is `addition_embed_type:
  "text_time"` and is conditioned on both, exactly as the SDXL UNet is.

  `ControlNet::forward_sdxl` now takes them, through the same arithmetic and
  the same slot the UNet uses. Verified correction by correction against
  `diffusers/controlnet-canny-sdxl-1.0`: nine skips plus the mid block, all
  within 1e-3, and a plain `forward` on such a checkpoint is refused by name.

  **Not** the `-small` distilled variant, which is tempting at 640 MB against
  5 GB and is a different architecture: all `DownBlock2D`,
  `transformer_layers_per_block: [0, 0, 0]`, no mid block. `UNetConfig::sdxl()`
  does not describe it. Supporting it would be its own config, not this one.

  Still open: wiring it into `SdxlPipeline`, which is plumbing — the model
  side is verified.

- ~~`candle_nn::rotary_emb::{rope, rope_i, rope_thd}`~~ **Done — `rope_i`, and
  it is worth 1.35x on a whole Flux run.** Flux rotates *interleaved adjacent
  pairs*, which is `rope_i`'s convention; `rope` splits the head in half and is
  a different function on the same shapes. The old form built an explicit 2x2
  `[[cos, -sin], [sin, cos]]` per frequency and then narrowed it back apart —
  four times the memory for the same numbers, and several strided passes where
  the kernel does one.

  Measured by `--example rope_path`, minimum of 12, **synchronising inside the
  timed region**: 61.7 ms against 2.4 ms at Flux's 1024 shape (26x), 13.0
  against 0.64 at 512 (20x), agreeing to 5.3e-8. End to end, interleaved,
  minimum of three: **23.15 s against 31.31 s**, fused lower on every pass,
  same image to mean 0.022/255. `SD_FLUX_ROPE=matrix` restores the old path so
  the pair can be re-measured.

  Two methodological notes, both paid for here. A first timing reported 14
  million elements rotated in 9 microseconds — 1.5 TB/s, which is the tell
  that **Metal was queuing the work and the timer was measuring enqueue**. And
  a first end-to-end run compared against the *recorded* 20.8 s baseline and
  showed nothing; only alternating the two in one session separated a 35 %
  difference from a 40 % spread.
- ~~`candle_nn::ops::{pixel_shuffle, pixel_unshuffle}`~~ **Checked: not
  applicable, and no benchmark needed to say so.** They are patchify by
  another name only in the spatial sense — `pixel_unshuffle` is
  `reshape → permute(0,1,3,5,2,4) → reshape` to `[b, c*4, h/2, w/2]`, where
  Flux's `pack_latents` permutes to `[b, tokens, c*4]`. Same class of
  operation, different output layout: adopting candle's would mean its permute
  *plus* a flatten and a transpose to get back to a sequence. Strictly more
  work, so it cannot be faster.
- ~~Broaden fused attention by materialising `causal_mask`.~~ **Done, and the
  premise was exactly right.** `attention_with_path` already offers its mask
  to candle's fused kernel, which **declines the broadcast `[1,1,s,s]` form
  and takes `[b,h,s,s]`** — so every masked attention here was silently on the
  naive path. Measured by `--example masked_attention_path`, minimum of 20,
  synchronised:

  ```text
    CLIP-L text tower   naive 416.9 us   fused 165.2 us   2.5x
    unCLIP prior        naive 794.9 us   fused 211.2 us   3.8x
  ```

  Agreeing to 2.3e-7 and 3.3e-7. The prior gains most because it runs 20
  blocks x 25 steps of it. Both now expand once and share the result across
  layers.

  Worth recording that this **puts back an expansion removed earlier in the
  same session**, when it was shown not to be the fix for a Metal failure. It
  was not; it is a 2.5-3.8x speed-up, which is a different claim, and this
  time it is measured.
- A **blocked CPU attention kernel**, tiling over query rows as well as keys.
  This is the one that would matter: attention at 4096 tokens is where CPU
  time actually goes, and it is exactly where candle's CPU flash kernel loses
  to gemm — see the measurements in
  [roadmap.md](roadmap.md#cpu-flash-attention-a-short-sequence-win-not-a-general-one).
  Real kernel work, not wiring.
- `--example attention_path` **flags two rows as "flash is faster here", by
  design** — they are known misses, not regressions:
  - *Flux 1024*, 1.15-1.25x. CPU flash has a second crossing at
    `head_dim = 128`: it sags to 0.85x around 1024 tokens then climbs back by
    4608. `DEFAULT_FLASH_CPU_MAX_SEQ` ignores it because capturing it needs
    two disjoint intervals fitted to one `head_dim` on one machine.
  - *SDXL cross-attention*, 1.07-1.12x, repeatable. But SD 1.5's
    cross-attention at the same sequence lengths and `head_dim = 40` is
    0.83x, so this one is not separable by sequence length at all — it would
    need `head_dim` in the rule. Worth ~0.8 ms per call, and SDXL runs on
    Metal anyway.
- Flux **dev** is gated on HuggingFace and needs the user's account.
- CUDA is untested — no device available here.
- ~~SDXL img2img is unverified end to end after the encoder-tiling fix.~~
  **Verified.** Encoder (tiled) through strength, sampler and decoder: the
  strength maps to steps correctly (`--steps 8 --strength 0.35` runs 3), the
  output holds the input's composition at low strength and leaves it at high
  (mean 24.5/255 from the input at 0.35 against 90.0 at 0.9), and forcing the
  encoder to tile changes the result by mean 6.6/255 — the "close to but not
  identical" that tiling is documented to be, not a defect.

## SDXL below its native resolution is garbage, and that is not a bug

Worth writing down because it looks exactly like one. SDXL at 256x256 produces
saturated colour noise with no subject at all — and it does so in **txt2img**,
which is what rules the pipeline out. An img2img at strength 0.9 looks equally
broken for the same reason: at that strength the output is mostly generated
rather than preserved, so it inherits the same failure.

At 512 it is coherent but heavily stylised; 1024 is where it belongs. If an
SDXL result looks like a bug, render the same prompt as txt2img at the same
size before investigating anything else — that single control separates "the
resolution" from "the code" in one run.

This is also why 1024 is a poor size to *test* at. Verifying the img2img path
means exercising encoder tiling, which only engages above
`TILE_LATENT_EDGE * 8`; lowering `SD_VAE_TILE_LATENT` makes a 256px image tile
into four, covering the same code at a sixteenth of the pixels. Reaching for
native resolution instead put a 1024 VAE *encode* on the GPU — wired memory,
the allocation class that has taken this machine down before — and did exactly
that again.

## The suite reported the same 362 green with no fixtures at all

Measured, not suspected: renaming `tests/golden` aside and running the whole
suite gave **362 passed, 0 failed** — identical to the run with every fixture
in place. Golden data is gitignored and generated locally, every numerical test
returns early when it is missing, and an early return is a pass. Nothing in the
output separated "verified" from "did nothing".

**`SD_REQUIRE_FIXTURES=1 cargo test` is now the run that means something.**
Skipping still works by default, because the fixtures really are too large to
commit; it is just no longer the only option. Every fixture-missing skip goes
through `sd_tensor::skip_missing_fixture!`, which prints a uniform line and
panics under the flag. Environmental skips — no GPU, a memory refusal, an unset
`SD_TEST_*` — stay permissive, because generating fixtures would not fix them.

Turning it on immediately found **18 pipeline property tests that had never
run here**: determinism across runs, cancellation, textual inversion, hires,
regional prompts, frame batching, instruct guidance. They gate on
`SD_TEST_MODEL_DIR`, which nobody had set. They all pass — but nothing had
been checking, and those are the properties this project most relies on.

```bash
SD_REQUIRE_FIXTURES=1 \
  SD_TEST_MODEL_DIR=$(pwd)/models/sd15 \
  SD_TEST_CONTROLNET=$(pwd)/tests/golden/controlnet/controlnet.safetensors \
  cargo test --release --workspace
```

Note the **absolute** path: `cargo test -p <crate>` runs with the package
directory as its working directory, so a relative `./models/sd15` silently
resolves to nothing and every test skips again.

**The line the flag draws** is between golden data `dump_reference.py`
produces — which it demands — and third-party assets a user supplies, which it
does not. A textual-inversion embedding and an InstructPix2Pix checkpoint are
downloads this repository cannot generate, so those tests still skip
permissively and say which variable to set. Demanding them would make the flag
fail on every machine, and a gate that is always red gets ignored — the same
reasoning that makes a memory refusal skip rather than fail in the GPU smoke
test.

## CPU against Metal, per module

`metal_parity.rs` runs each architecture's forward on both devices and compares.
Until now only the VAE decoder had this, and everything else was verified on
CPU alone — which is how a Metal-only failure in the unCLIP prior passed nine
golden tests. First measurements:

```text
  clip-l pooled     6.484e-7
  sd15 unet         2.301e-6
  unclip prior      4.049e-6
  unclip unet       1.083e-4
```

## Traps this codebase has already paid for

Each of these cost real time. They generalise.

**Verify against a published artifact, not one your own tooling
round-tripped.** A legacy VAE attention-naming bug survived a fully green
suite because the reference weights came from `vae.state_dict()`, which
diffusers renames on load.

**When a checkpoint exists in two published forms, verify against the one you
load.** SD 3.5's single fp16 file and its converted fp32 copy differ by up to
2e-3, which 24 blocks amplified into a 9.3e-3 output mismatch that looked
like a bug in our code. Regenerating the reference from the file Rust reads
took it to 5.5e-6.

**Absolute tolerances are wrong for anything that is not order-1 — including
the UNet, which nobody suspected.** CLIP peaks at 851, T5 at ~200,000, SD
3.5's blocks at ~97,000; the UNet's mid block peaks at a mere 16, and a 1e-4
*absolute* bound on it still turned out to be **below the reference's own
noise floor**. `xtask/golden/reference_precision.py` runs a diffusers module
against itself in f64: `mid_output` shows diffusers missing its own f64 by
1.108e-4. That test could only ever pass by accident of summation order, and
it duly broke the first time Apple's Accelerate reordered it — at 1.087e-4,
*closer to the reference than the reference's own f32 was*.

So: use `testing::allclose_excess(a, b, rtol)` with an absolute floor beside
it, and **measure before choosing either number.** The script exists now; run
it rather than picking a value that turns the light green. That is how the
Flux VAE encoder's 2e-3 bound and T5's 3e-3 were set, and both sat at or below
the reference's own noise floor.

**A pooled CLIP embedding was being read from the wrong position, in three
places, for the whole history of this project.** `transformers` finds the EOS
with `argmax`, which returns the *first* maximum; Rust's `max_by_key` returns
the *last*. SD 1.5's tokenizer **pads with EOS**, so a ten-token prompt has 68
copies of 49407 and the two rules are 67 positions apart. Flux's pooled
conditioning, SD 3's CLIP-L half and GLIGEN's phrase embeddings were all
reading the final padding slot.

It survived because the only golden test covering pooling is SDXL's second
encoder, whose tokenizer pads with `!` — leaving one maximum, where both rules
agree. Every other test feeds *pre-computed* conditioning tensors and never
tokenizes, so none of them touched the function. It surfaced only when the
unCLIP prior's text encoder was compared against `transformers` and missed by
**1.72**; the fix takes it to 9.7e-7.

Two lessons. **A helper used by five callers and covered by one is covered by
none** — the covered case was the one where the bug is unreachable. And when
a suite verifies modules against saved tensors, the *tokenizer-to-model* seam
is exactly what no golden test sees. `pooling_takes_the_first_eos_not_the_last`
in `golden_clip_encoder.rs` is structural and needs no reference data, so it
cannot regress quietly again.

**When a fix works, find out which half of it was the fix.** The prior's Metal
failure was cleared by two changes made together: materialising the attention
mask from `[b, 1, s, s]` to `[b, heads, s, s]`, and a `contiguous()` on the
narrowed output. The second was the entire fix. Reverting the first and
re-running took two minutes and showed it had never been needed — otherwise a
32x-larger mask would have shipped, along with a confident paragraph in the
module docs explaining why it was necessary, which the next person would have
believed. Two simultaneous changes and one observation cannot tell you which
one worked.

**A helper used by five callers and covered by one test is covered by none**
if the covered case is the one where the bug cannot appear. That is not a
figure of speech — see the pooled-EOS entry below. The general defence is
tests that assert *invariants between* modules rather than a module against
saved tensors: `seam_invariants.rs` holds the ones that need no fixtures, so
they run on a fresh clone. The most valuable of them asserts that unCLIP's
class projection is exactly twice an image embedding, which is the property
the published `-t2i-h` checkpoint violates.

**`forward()` then `pooled()` encodes the prompt twice**, because
`pooled_hidden` runs the encoder itself. SD 3 was doing exactly that for both
CLIP towers — four full forwards per generation where two suffice — and so was
the first version of the prior's text path. `ClipTextEncoder::pool` and
`::project` take an already-computed sequence, so a caller that wants the
sequence *and* the pooled vector pays once. Worth ~0.2 % of a generation and
found by reading rather than profiling; the reason to fix it is that the
duplicate call is invisible at the call site, not that it was slow.

**A tensor that does not own its buffer is still a different tensor — and this
time CPU was the one that hid it.** The prior reads its answer with
`narrow(1, 80, 1)`, leaving a `[b, 2048]` view whose row stride is the
sequence's `81 * 2048`. CPU computes the right answer from it; candle's Metal
matmul refuses outright, with `Invalid matmul arguments [165888, 1] [1, 2048]
(2, 768, 2048)` from inside the kernel and nothing naming the line. Every
golden test passed — they all run on CPU — and the GPU smoke test is what
caught it, which is the second time that file has earned its place on a new
architecture's first run. Read the `mnk` triple in that message: `(2, 768,
2048)` is `m, n, k`, and `165888 / 2048 = 81` named the tensor.

**Check that the high-precision run is actually running in high precision.**
`reference_precision.py` earns its keep by measuring the reference against
itself in f64 — but two things in the unCLIP path silently stayed f32: the
DDPM schedule, whose constants diffusers builds once in f32 and hands to both
runs, and `get_timestep_embedding`, which hardcodes `dtype=torch.float32`
inside. The "f64 run" therefore returned f32 numbers, the reported floor came
out 40x too low, and a correct implementation failed against it. Before
trusting a floor, confirm the f64 run's *inputs and constants* are f64 too,
not just its weights.

**Cancellation moves a noise floor by four orders of magnitude.** The same
measurement showed why the level-0 augmentation is twenty times noisier than
level 250: near the top of the ladder `alpha` is within 4e-5 of one, so an
absolute 2e-8 difference in it is a *relative* 5e-4 difference in `1 - alpha`,
which is what the noise gets multiplied by. Wherever a bound is set on
something computed from `1 - x` with `x` near 1, measure it there rather than
extrapolating from a well-conditioned neighbour.

**Test the smallest input that reaches the code, not the most realistic one.**
Verifying SDXL img2img meant exercising encoder tiling, which engages only
above 512px — so native 1024 looked like the honest choice and wedged the
machine on a wired Metal allocation. `SD_VAE_TILE_LATENT=16` tiles a 256px
image into four and covers the same path for a sixteenth of the memory. The
knob had been added earlier the same day and forgotten.

**Do not `git checkout` a file to undo a temporary edit if it also holds
uncommitted work.** Reverting a one-line test perturbation this way discarded
an hour of tolerance work in the same file. Copy the file aside and restore
from the copy, or commit first.

**F16 is not a safe way to halve memory.** T5's activations exceed 65,504 and
go NaN around block 10; Flux's transformer NaNs too. Hold weights quantised
instead (`weights::Source::Quantized`) — activations stay f32 and residency
drops further than F16 would have. bf16 would fix it but candle's CPU backend
has no bf16 matmul.

**On Metal, the op that reports the error is rarely the op that caused it.**
candle queues GPU work and only inspects the command buffer when something
synchronises, so a failure is attributed to the first thing that *waits*.
SD 3.5 at 512 "failed in the VAE decode" through several sessions and a
written handoff note; synchronising after each denoise step put it at step 1,
where it had been all along. Before believing where a Metal error happened,
put a `device.synchronize()` after each stage and watch it move.

**A tensor that does not own its buffer is a different tensor.** candle 0.11's
Metal quantised matmul ignores `start_offset`, so a view into the middle of a
buffer is read from the beginning of it — silently, with correct shapes. Flux
rendered a flat orange field for three sessions because every block projects
`attn.narrow(1, 512, 1024)`. **`contiguous()` does not fix this**: a narrow off
any axis but the last is *already* contiguous by candle's definition — the
elements are consecutive, they just start late — so it is a no-op.
`force_contiguous()` always copies. The workaround lives in
`QLinear::forward`, at the seam, because that is where every quantised matmul
in the workspace passes.

**Test tensors always own their buffers, which is why per-op checks missed
it.** `metal_check` built fresh inputs, so it exercised the op in exactly the
condition where it is correct. Attention agreed to 1.9e-7 at every sequence
length; QLinear at every row count and quant type; norms, RoPE, trig, cat,
narrow, weight loading, dequantisation to 1e-8 — all green, while the composed
model was 50% wrong. When every part passes and the whole fails, suspect
*provenance*: build the op's input the way the model builds it, not the way a
test does.

**Bisect against a full-precision reference, not against the other device.**
CPU-vs-Metal conflates two error sources. CPU's quantised matmul carries
0.3-1.9% activation-quantisation noise, so a per-layer CPU/Metal table read as
plausible drift everywhere and pointed at nothing. Dequantising the *same*
weights into a dense f32 model — only feasible at depth 1, which was enough —
turned it into one obvious line: Metal tracked truth better than CPU
(≤0.12%) through the whole block, then jumped to 36% at a single projection.
Right after that, recomputing that one op in numpy from *Metal's own dumped
input* proved the op wrong rather than its input.

**Check whether candle already does it.** The roadmap called a hand-written
fused Metal attention kernel the highest-value work available; candle 0.11
already shipped one and `attention_with_path` was already routing to it. Only
a stale doc comment said otherwise.

**…and then check whether it is actually faster, on your shapes.** The
follow-up to that lesson overcorrected: the survey listed CPU flash attention
as "potentially the largest single win available" on the strength of it
existing. It is 2-7x faster below 512 tokens and up to 2x slower above, and
everything above 512 is where CPU time actually goes — net, nothing
measurable end to end. Wiring it in unconditionally would have made SD 1.5
self-attention 2x slower while the enum said `FlashCpu` and every test stayed
green. Benchmark the new path against the existing one across the shapes you
really run, before the dispatcher prefers it, and keep both.

**On this machine, benchmark by interleaved minimum — never by mean, and
never from one A/B pair.** Two traps, both paid for on 2026-07-26:

- A mean-of-5 microbenchmark reported figures 10x apart on back-to-back runs
  of the same binary. Noise here is one-sided — preemption, page faults and
  thermal excursions only add time — so take the *minimum* of N runs after a
  warm-up, and *alternate* between the two things compared so drift lands on
  both. `--example attention_path` does both and says why.
- At whole-generation scale one A/B pair said CPU flash made SD 3.5 11.8%
  faster. Running the same pair with the order reversed said 0.7%, and two
  runs of the *identical* configuration differed by more than the two
  configurations did. **Always run the pair both ways round.** Had the first
  pair been believed, a fabricated 12% would now be in roadmap.md, and the
  arithmetic said it was impossible — the shapes involved are worth 0.4 s.
  When a measurement disagrees with the mechanism, suspect the measurement.

**Measure before diagnosing.** The Flux bottom-edge striping was assumed to
be our packing or positional encoding. Handing diffusers *our* conditioning
and *our* initial noise removed everything but the loop, and the loop agreed
to 6.5e-5 — the artifact is the checkpoint's.

## Machine

36 GB M4 Max. A previous session **crashed the whole machine** with an 81 GiB
wired Metal allocation. Metal allocations are wired and can take down the OS
rather than failing the process.

`sd_tensor::sysmem::check_headroom` guards loads against *actual* free memory
(`SD_MEMORY_HEADROOM` overrides). Check `memory_pressure` before large runs.
Note free space can read misleadingly low — macOS purgeable space once showed
5.3 GB where 142 GB was actually available.

## Running things

```bash
# fixtures live under tests/golden/, gitignored, mostly symlinks into ~/.cache/huggingface
cargo run --release -p sd-cli --example flux_txt2img -- "<prompt>" <steps> <size> out.png
cargo run --release -p sd-cli --example sd3_txt2img  -- "<prompt>" <steps> <size> out.png
cargo run --release -p sd-cli --example attention_path      # which path, how fast
cargo run --release -p sd-cli --example metal_check         # CPU vs Metal per op
cargo run --release -p sd-cli --example requantise -- in.gguf out.gguf Q4_K

# golden references (local only; CI skips the numerical tests)
python3 xtask/golden/dump_reference.py <component> --output tests/golden
#   components: vae flux_vae sd3 flux_transformer flux_sampling t5 flow clip_* unet_* samplers sdxl_* unclip unclip_prior

# tolerances: what the reference's own f32 misses its f64 by
python3 xtask/golden/reference_precision.py <unet|vae|taesd|unclip>
python3 xtask/golden/reference_precision.py unclip --model-id models/unclip-t2i

# unCLIP, both halves
sdrs unclip --model models/unclip --init-image ref.png --cfg-scale 5.0
sdrs unclip --model models/unclip-t2i --prompt "a crab on a beach"
```

`flux_txt2img` prefers `flux-schnell-q4_k_s.gguf` when present and falls back
to flux-mini; it reads block counts and guidance from the file rather than
assuming.

## Fixtures

Regenerable, so deleting them is cheap:

- `tests/golden/flux/` — flux-mini (6.4 GB), flux schnell Q4_K_S (6.8 GB),
  T5-XXL Q4_K_S (2.7 GB), Flux VAE, tokenizers
- `tests/golden/sd35/` — SD 3.5 single-file (5.1 GB), Q4_K_M gguf, VAE,
  CLIP-L, CLIP-G
- `tests/golden/gguf/` — SD 1.5 quant sweep. The k-quants were deleted to
  save space; regenerate with `--example requantise` from `sd15-f16.gguf`.
- `models/unclip/` — the image-variation unCLIP checkpoint converted from
  pickled `.bin` to safetensors, 7.2 GB, written by `dump_reference.py unclip`;
  `tests/golden/unclip/` symlinks into it rather than holding a second copy.
- `models/unclip-t2i/` — the *text-to-image* one, 9.9 GB, from
  `dump_reference.py unclip_prior`. A separate directory because `-t2i-l`
  shares nothing with `-i2i-h` but the prior. Both are regenerable, so
  deleting either to reclaim space costs one download.

Ungated mirrors matter: `black-forest-labs/*` and `stabilityai/*` are gated.
Use `Freepik/flux.1-lite-8B` for the Flux VAE and T5 tokenizer, and
`adamo1139/stable-diffusion-3.5-medium-ungated` for SD 3.5.
