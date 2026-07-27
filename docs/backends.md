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

### What candle's fused SDPA actually does: most Metal shapes, no CPU ones

`candle_nn::ops::sdpa` is Metal-only in 0.11 — its `cpu_fwd` bails outright —
but on Metal it takes **every unmasked shape in this workspace except SD 1.5's
UNet**, and it is worth having:

```text
                         CPU        Metal
  SDXL UNet 1024    453.7 ms    14.8 ms  fused
  Flux 1024         957.0 ms    43.8 ms  fused
  SD 3.5 512         37.1 ms     2.7 ms  fused
  SD 1.5 UNet 512   110.7 ms    16.8 ms  chunked (head_dim 40 is unsupported)
```

It still declines two things: f32 at `head_dim = 512`, which exceeds Metal's
32 KB of threadgroup memory and is exactly the VAE attention block, and a mask
that is not `[batch, heads, seq_q, seq_k]` — `ops::causal_mask` is
`[1, 1, s, s]`, so CLIP's causal path declines.

**This paragraph used to say it declined everything, and that claim outlived
the fact by several sessions.** The roadmap consequently listed writing a fused
Metal kernel as the highest-value work available, when candle had already
shipped one and `attention_with_path` was already routing to it. Re-check the
path values with `--example attention_path` on a candle bump rather than
re-reading this section; that is what the function returns them for.

### What CPU got instead: candle's flash kernel, for short sequences only

candle 0.11 also ships a CPU flash kernel
(`candle_nn::attention::flash_attn`), reached via `ops::flash_attention_cpu`
and reported as `AttentionPath::FlashCpu`. It streams one output row at a time
under a running softmax maximum, so like the Metal kernel it never
materialises the score matrix.

**It is not a uniform win, which is the whole story here.** It gets no register
blocking across query rows and re-reads the key axis per row, so it beats the
gemm-based path only while the gemms are too small to amortise their blocking.
Measured on an M4 Max across `head_dim` {40, 64, 80, 128, 160}, `heads`
{8, 12, 20, 24, 64} and batch {1, 2}, the crossover is at **512 tokens**: 2-7x
faster below it, up to 2x slower above. `ops::DEFAULT_FLASH_CPU_MAX_SEQ` holds
that limit and the measurements behind it; `SD_FLASH_CPU_MAX_SEQ` overrides it,
and `0` disables the path.

So it serves CLIP at 77 tokens and the UNet blocks at 16x16 and 8x8, and stays
out of the way of everything that dominates a denoise step. **T5 is not
served**, despite its 154 tokens: its relative-position bias is a full
`[batch, heads, n, n]` tensor and the kernel indexes masks flat, with no head
axis, so `ops::flash_cpu_supported` refuses it.

**End to end that is not measurable on this machine**, and the mechanism says
it should not be: the eligible calls total about 0.4 s of an SD 1.5
generation and about 13 ms of an SD 3.5 one. SD 1.5 at 512x512, 20 steps ran
113.3 s with it and 114.4 s without; SD 3.5 run four times alternating gave
245.2 / 216.2 / 228.7 / 230.4 s, a spread within one configuration wider than
the gap between them. Images differ by at most 1/255. See
[roadmap.md](roadmap.md) for the full table and why the first pair's apparent
12% was an artefact of run order. **The cliff below is a large-sequence
problem and this kernel is a small-sequence win, so it does not touch it.**

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

On Metal the cliff is closed for every shape candle's fused kernel accepts,
which is all of them bar SD 1.5's `head_dim = 40` UNet and the VAE's
`head_dim = 512` block. Those two, and the CPU path above 512 tokens, still
materialise the score matrix a tile at a time.

On CPU there is no equivalent left to reach for: candle's flash kernel is the
fused option, and it is slower than the gemm path at exactly the large
sequences where the cliff lives. Closing that would mean a blocked CPU kernel
that tiles over query rows as well as keys — real work, not a scheduling
change. Tracked in [roadmap.md](roadmap.md).

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

### The quantised models needed a workaround first

SD 1.5 and SDXL hold dense weights, so the above held for them all along.
**Flux did not**, and the reason is worth knowing before writing any new
Metal-facing code: candle 0.11's Metal quantised matmul ignores the
activation's `start_offset`, reading a view into the middle of a buffer from
the *start* of that buffer. Flux rendered a flat orange field for three
sessions because every double-stream block projects
`attn.narrow(1, 512, 1024)`, and `contiguous()` does not move a narrow off
dim 0 — candle already calls that layout contiguous.

`sd_tensor::quantized::without_storage_offset` copies such an activation
before the matmul, inside `QLinear::forward`. Flux schnell at 512x512, 4 steps
now renders correctly on Metal in **20.8 s against 159.3 s on CPU**. See
[roadmap.md](roadmap.md) for how it was localised, why every per-op check
passed while it was broken, and the verification table for the other models.

### What still does not fit on Metal, and the tile knob

Correctness is no longer the constraint; memory is. Two runs die *after* the
denoise loop completes, in the VAE decode, because the transformer is still
resident and never used again:

- **SD 3.5 at 512** needs `SD_VAE_TILE_LATENT=32`. At the default 64 the
  decode is a single 2.42 GB tile and SD 3.5's 10 GB transformer leaves no
  room. With the smaller tile it renders in 25.1 s, no visible seam.
- **Flux mini at 512** does not fit at any tile size tried. Its 3.2B
  parameters are dense f32 — 12.8 GB — against schnell's 6.8 GB for 12B held
  as Q4_K. Quantisation is what makes the *larger* model the one that runs.

`sd_models::vae::tile_latent_edge` reads that variable and the load-time
headroom projections in `sdxl.rs` and `txt2img.rs` honour it too, so lowering
the tile also lowers the bar a load has to clear. The default stays 64 because
tiling changes the image: the decoder is not shift-invariant, so tiles are
blended rather than abutted.

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
