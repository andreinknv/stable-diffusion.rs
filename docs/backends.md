# Backend evaluation

Why candle, what the alternatives are, and what the seam does — and does not —
protect against.

This exists so the question is not re-litigated from scratch every time someone
reasonably asks "why not burn?". The answer has real numbers behind it.

## The layers

The word "backend" gets used for two different things. Keeping them apart makes
most of the confusion go away:

```
model code           UNet, VAE, CLIP                        <- ours
tensor library       candle | burn                          <- candle is HERE
kernel authoring     CUDA C++ | MSL | cubecl | rust-gpu     <- and cubecl is HERE
GPU API              cudarc | wgpu | metal
```

A tensor library gives you `Tensor`, dtypes, broadcasting, memory, model
loading and a few hundred ops. A kernel language gives you a way to write *one*
op. They are not substitutes.

## The candidates

| | What it is | Verdict |
|---|---|---|
| **candle** | tensor library | **in use** |
| **burn** | tensor library on cubecl | credible, not now — see below |
| **cubecl** | kernel language, compiles to cuda/hip/metal/spirv/wgpu/cpp | not a candle replacement |
| **rust-gpu** | rustc backend, Rust -> SPIR-V | not a candle replacement; not production-ready |
| **tch-rs** | LibTorch bindings | ~2 GB C++ dependency |
| **ort** | ONNX Runtime bindings | C++ dependency; wrong shape for this |
| **tract** | pure-Rust inference | CPU only |

### Why not cubecl or rust-gpu directly

Neither ships a tensor library. cubecl's crates are all compiler and runtime —
`cubecl-core`, `cubecl-ir`, `cubecl-opt`, plus codegen targets. rust-gpu is
`rustc_codegen_spirv` and friends.

Building on either means writing `Tensor`, strides, dtypes, broadcasting,
memory management and ~60 op families yourself. That is the from-scratch
option, costed at 8-15 months elsewhere. rust-gpu additionally targets only
SPIR-V — no CUDA, no native Metal — and its own README says it is "not yet
production-ready".

### Why not burn (yet)

burn is a real alternative and stronger than a first glance suggests.
`burn-store` has dedicated `safetensors/` and `pytorch/` loaders plus
`keyremapper.rs`, which is precisely the diffusers weight-name mapping we need.
Layers live in `burn-nn`. Governance is better than candle's: a company behind
it, contributor spread 720/442/313/172, and ~100 commits per 90 days against
candle's ~48. `cubecl` gives it one kernel source compiling to CUDA, ROCm,
Metal, wgpu and SPIR-V, which is architecturally nicer than per-backend kernels
and would make GPU kernels genuinely Rust.

Two things keep us on candle:

**GGUF.** 2 code hits in burn against 94 in candle. Every quantised model
people actually download is a `.gguf` with Q4_K/Q5_K/Q8_0 weights. On burn we
would write the dequantisation path and the quantised matmul kernels ourselves
before rendering anything — which collapses back into the from-scratch option.

**The type model, which the seam does not abstract.** See below.

Worth noting the asymmetry: candle's problems are a one-line upstream fix
(`onig`) and a bus factor. burn's is months of quantisation work. Those are not
comparable in cost.

## What the seam does not protect against

The seam is real and it earns its keep, but it has been oversold in earlier
notes and this is the correction.

```
candle:  Tensor                       dynamic rank, dynamic dtype, no type params
burn:    Tensor<const D: usize, K>    rank is a CONST GENERIC in the type
```

`sd-tensor` is type aliases and re-exports over candle's `Tensor`. Moving to
burn changes the *shape of every signature in the workspace*:
`fn forward(&self, xs: &Tensor)` becomes `fn forward(&self, xs: Tensor<4>)`,
and every intermediate value carries its rank in its type.

**The seam contains the dependency. It cannot contain a change in the type
system's shape.** Concretely:

| Change | Seam protects? |
|---|---|
| Replace one kernel with our own | **yes** — that is the design case |
| candle renames or changes an API | **yes** |
| Swap to a candle-shaped library | **yes** |
| Swap to burn | **no** — model code needs rewriting too |
| Drop a transitive dependency | **no** — that is upstream's call |

So the escape hatch is cheap and worth keeping, but "we can swap the backend in
one crate" is only true for backends shaped like candle. Do not plan on it for
burn.

## Does candle GPU actually work?

Yes. Verified on an Apple M4 Max, macOS, release build — full SD 1.5 VAE
decoder geometry, best of 3 after a warm-up:

| latent | image | attention seq | CPU | Metal | |
|---:|---:|---:|---:|---:|---|
| 16x16 | 128x128 | 256 | 567 ms | **114 ms** | 5.0x faster |
| 32x32 | 256x256 | 1024 | 1.97 s | **458 ms** | 4.3x faster |
| 64x64 | 512x512 | 4096 | 9.31 s | 11.43 s | **0.8x — slower** |

Reproduce with:

```bash
cargo run --release -p sd-cli --example backend_bench -- 32
cargo run --release -p sd-cli --features metal --example backend_bench -- 32
```

The benchmark times five decodes after a warm-up and reports a **median with
its spread**, not a best-of-N. A minimum answers "how fast could this go on a
quiet machine", which is the wrong question when comparing two implementations:
it reports whichever one caught the quietest moment. Raise the sample count
with `SD_BENCH_REPEATS`.

Above ~15% spread it prints a warning and you should not quote the median.
That is not hypothetical — the numbers in this table were taken on a quiet
machine, and a later attempt to compare chunk sizes on a loaded one produced a
71% spread and a median twice the figure below for identical code. **Check the
spread before believing any comparison, including the ones in this file.**

Mind the argument. Memory is `n^4`, so the next size up costs 16x, and on Metal
the score matrix is wired GPU memory the OS cannot reclaim — overshooting RAM
panics the machine rather than failing the process (this has happened, at
`-- 384`). `sd_tensor::ops::check_attention_budget` refuses past a 2 GiB
projection before allocating, and it sits in the seam, so the CLI and the models
are covered too — not just this benchmark. Override with
`SD_ATTENTION_BUDGET_BYTES` only deliberately; see rule 8 in
[AGENTS.md](../AGENTS.md).

Metal works — conv2d, group norm, attention and upsampling all execute on the
GPU, and it is 4-5x faster than CPU. Then at production resolution it falls off
a cliff and loses to the CPU.

**The cliff is our own code, not candle's.**
`ops::scaled_dot_product_attention` materialises the full `seq x seq` score
matrix. At a 64x64 latent, `seq = 4096`, so that is a 4096x4096 f32 matrix —
**67 MB, allocated per attention call**, before softmax. The scaling is
unmistakable: quadrupling `seq` (32->64) makes CPU 4.7x slower and Metal 25x
slower, because the GPU is the one starved for memory bandwidth.

Two caveats on the numbers: the first Metal call took 36 s including lazy
shader compilation, so always warm up; and an M4 Max has unusually fast CPU
matmul, so the CPU column flatters itself relative to a typical x86 machine.

### What not to do about it: candle's fused SDPA

`candle_nn::ops::sdpa` looks like the free fix. It is not, for us. In 0.11 it is
Metal-only (`cpu_fwd` bails outright), and on Metal it declines every shape this
workspace runs:

- f32 at `head_dim = 512` is explicitly excluded — that is exactly the VAE
  attention block, single-head over 512 channels, i.e. the call that produced
  the cliff below.
- the UNet's `head_dim = 40` is not in its supported set at all.
- a mask must be materialised to `[batch, heads, seq_q, seq_k]`, while
  `ops::causal_mask` is `[1, 1, s, s]`, so CLIP's causal path declines too.

`ops::attention_with_path` is wired up to use it where it applies and reports
which path actually ran, so this can be re-checked on a candle bump rather than
re-litigated from memory. Today it reports `Naive` everywhere. Do not record
"fused attention landed" as a memory win without checking that value.

### What was done about it: chunked attention

`ops::chunked_attention` computes the score matrix in query tiles rather than
materialising it whole. It landed entirely inside `sd-tensor` with no model
code changes — the case the seam was designed for — and it is verified against
the `diffusers` golden reference, including at one query row per chunk
(`max_abs` 3.43e-5 versus 3.678e-5 unchunked, tolerance 1e-4).

**What it buys: memory and reach, not speed.** Peak score memory is bounded
near the chunk target instead of growing as `n^4`, so sizes that were
previously refused now decode — a 160 latent needs a 2.4 GiB score matrix in
one piece, and runs in 64 MiB tiles.

**What it does not buy: the cliff.** Chunking performs the same arithmetic with
more kernel launches and a concatenation, so it cannot be faster than not
chunking; the honest ceiling was "how much does it cost". Attempting to measure
that at latent 64 on an M4 Max gave 9.1/11.8/9.6 s unchunked against
17.3/12.3/8.0 s at 8 MiB chunks — distributions that overlap entirely. The
run-to-run variance on this machine (±40%, and a competing build swung one
sweep by 20%) is larger than the effect being measured, so **no speed claim is
made in either direction.** If you want a tuned chunk size, measure on a quiet
box, repeat each configuration, and set `SD_ATTENTION_CHUNK_BYTES`.

The default is therefore 64 MiB: exactly the SD 1.5 512x512 score matrix, so
the common case stays single-chunk and pays nothing, and only larger geometries
split.

### What is still to do about it

The cliff above is unchanged, because closing it needs softmax *fused* into the
matmul so the score matrix never reaches memory at all. That is a kernel, not a
scheduling change — candle does not expose one for these shapes (see above), so
it means writing Metal. Tracked in [roadmap.md](roadmap.md).

Also note `--features accelerate` on Apple silicon: BLAS-accelerated CPU
matmul that adds **no native compilation** (it links a system framework). Worth
enabling for CPU builds regardless of the GPU story — see
[native-deps.md](native-deps.md).

## Does the whole pipeline work on Metal?

Yes. `sdrs txt2img --features metal` runs end to end and produces the same
picture as CPU. Verified 2026-07-26 at 256x256, 20 steps, DPM++ 2M, seed 42.

The two images are **not** byte-identical, and that is expected rather than a
defect: `SeededRng` makes the *noise* device-independent, but f32 reduction
order is not, and twenty sequential UNet evaluations compound the difference.

| | CPU vs Metal, same seed |
|---|---|
| mean abs difference | 0.9 / 255 |
| max abs difference | 35 / 255 |
| pixels exactly equal | 27% |

Indistinguishable by eye; not interchangeable as files. Reproducing a PNG
byte-for-byte requires the same device and build.

Metal took 14.3 s against CPU's 27.5 s for that run. **That is one run on a
machine with other work on it, not a benchmark** — see the spread warning
above before quoting it.

## The VAE decode at 1024 does not fit in GPU memory

A 1024px decode needs more GPU memory than a 36 GiB Mac has, and until
recently it did not say so — it returned an image of horizontal noise bands.

**The memory is conv im2col, not activations.**
`DecoderConfig::peak_activation_bytes` counts activation tensors, and at 1024
the largest is 1.07 GB, comfortably inside the 2 GiB budget. But candle's
conv2d materialises an im2col intermediate holding `cin * 9` values per output
position:

| convolution | activation counted | im2col actually allocated | ratio |
|---|---:|---:|---:|
| 256 -> 256 @ 512px | 0.27 GB | 2.42 GB | 9x |
| 256 -> 128 @ 1024px | 0.54 GB | **9.66 GB** | **18x** |

So the budget guard is measuring the wrong thing by an order of magnitude at
these sizes. It is still worth having — it catches the `n^4` attention blowup
it was written for — but **do not read `peak_activation_bytes` as a true peak.**

**Why it was silent.** candle queues Metal work and only inspects the command
buffer's status when something synchronizes. Nothing did, so a decode whose
command buffer failed with `kIOGPUCommandBufferCallbackErrorOutOfMemory`
returned a tensor of whatever the buffer held, and the error was discovered
never. `Decoder::forward` now synchronizes once at the end, which costs
nothing next to a decode and turns the corruption into an error you can read.

Ruled out along the way, each measured rather than assumed:

- **individual ops** — `conv2d` (including the 9.66 GB im2col case), `silu`,
  `group_norm`, `softmax_last_dim`, `matmul` and `upsample_nearest2d` all
  agree CPU against Metal at these shapes, to 3e-5 or better;
- **chunked attention** — the corruption was identical with chunking disabled
  and with chunks 64x smaller;
- **the UNet and sampler** — a single denoise step decodes to a clean blurry
  image, and the latent statistics stay sane through every step.

The corruption was also *deterministic* across runs, which is what pointed
back at memory rather than away from it: an allocation that fails the same way
every time still fails.

**Consequence: SDXL at its native 1024 works on CPU and runs out of memory on
Metal here.** Use `--cpu` for 1024. SD 1.5 at 512 is unaffected on either.

**Tiling fixes the decode.** `AutoencoderKlDecoder::decode_tiled` decodes in
overlapping 64-latent tiles and cross-fades the overlaps, so no single
convolution needs more than one tile's im2col. Measured against a whole-image
decode at 768px: `mean_abs` 9.96e-3, and the worst column-to-column step is
1.53x the mean, so seams are not visible. Latents at or below one tile take
the untiled path and are bit-identical to before. Verified standalone on
Metal at 1024.

**It is not sufficient for SDXL on this machine, and the reason is weight
residency.** The pipeline loads fp16 checkpoints and upcasts them to f32:

| component | fp16 file | resident as f32 |
|---|---:|---:|
| unet | 5135 MB | 10270 MB |
| text_encoder_2 | 1389 MB | 2779 MB |
| text_encoder | 246 MB | 492 MB |
| vae | 167 MB | 335 MB |
| **total** | **6938 MB** | **13876 MB** |

13.9 GB of weights sit on the GPU before a single activation is allocated,
which on a 36 GiB machine shared with other work leaves too little for the
decode — SDXL fails at 768 as well as 1024, while the same tiled decode
succeeds standalone.

**Fixed by holding the models in f16.** The SDXL pipeline now loads the UNet
and both text encoders as f16 and keeps only the VAE in f32, halving
residency to about 7 GB. SDXL renders at its native 1024 on Metal in 89 s —
see `assets/sdxl-crab-1024-metal-f16.png`.

Three things had to stay f32, and each for a stated reason:

- **the VAE.** SDXL's overflows in fp16 — a well-known defect, which is why
  `madebyollin/sdxl-vae-fp16-fix` exists. At 167 MB, keeping it f32 costs
  nothing worth having.
- **the sampler.** Sigmas reach 14.6, and `sigma^2 + 1` in the input scaling
  would lose precision in f16 for no benefit — the latent is tiny beside the
  weights.
- **the timestep sinusoids.** Their frequencies span several orders of
  magnitude, so they are computed in f32 and cast to the model dtype at the
  boundary. `UNet2DConditionModel::dtype()` and
  `ClipTextEncoder::dtype()` expose where that boundary is, so callers cast
  deliberately rather than by assumption.

SD 1.5 still runs entirely in f32: it fits comfortably, and changing it would
mean re-verifying every golden test against f16 tolerances for no gain.

## Two guards, and why they are separate

**A shape guard**, `ops::check_alloc_budget`, deterministic: a shape goes in,
a byte count comes out, and the same input always gives the same answer. It
catches the `n^4` attention blowup and the 9.66 GB conv im2col. Being
deterministic is what makes it testable, and it is why the golden tests can
assert on it.

It is also blind. It compares one allocation against a fixed ceiling and has
no idea whether the machine has 8 GB or 128 GB, or that something else is
holding 13 GB right now. A run can pass it and still spend fifteen minutes
paging — which is exactly what happened to an img2img run at 1024.

**A memory guard**, `sysmem::check_headroom`, runtime: it asks the OS what is
free and refuses a job that would take more than 80% of it. Applied once where
a pipeline loads, not per operation, so the deterministic tests stay
deterministic.

The projection includes the **weights**, via `sd_loader::resident_bytes`,
which reads safetensors headers and computes what they will occupy at the
target dtype. That is the term that matters: weights stay resident for the
whole run and dominate everything else. A fp16 checkpoint loaded as f32
occupies twice its file size, and that doubling is what made SDXL fail to fit.

The 80% is calibrated, not picked. SDXL in f16 needs about 8.9 GB and renders
fine when that much is spare; the same model in f32 needs 16.3 GB and used to
page for fifteen minutes before failing. Override with `SD_MEMORY_HEADROOM`.

Worth seeing the consequence plainly: **the same command can be admitted or
refused depending on what else is running.** SDXL rendered in 89 s with
15.2 GB free, and was refused later the same day with 9.7 GB free. That is the
guard working, not flapping — the second run would have thrashed.

Where the platform cannot be asked, the check allows. A guard that refuses on
missing information is worse than one that admits it does not know.

## When to revisit

Switch to burn if **all three** hold:

1. Milestone 1 renders a correct image on candle first. Never migrate away from
   something that does not work yet.
2. GPU performance is the measured bottleneck *after* attention is fixed — the
   table above shows the current bottleneck is ours, not candle's.
3. GGUF has landed in burn, or we have decided to own quantisation anyway.

Until then the honest position is: candle has the worse governance and the
better feature coverage, and feature coverage is what ships images.
