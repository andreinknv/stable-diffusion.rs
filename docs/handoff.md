# Handoff

Written 2026-07-26. Say **"go"** to resume: read this file, pick the top
unstarted item under [Next](#next), and start it.

## Where things stand

Four architectures render, and **Metal now produces the right image for every
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
table is in [roadmap.md](roadmap.md). 268 tests, all gates green
(`cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`,
`scripts/check-seam.sh`, `scripts/check-native-deps.sh`).

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

In the order I would do them, with why.

### ~~1. Overlap the block copy with compute~~ — tried, and it is slower

Built and reverted. The prefetch itself worked exactly as intended: a
background thread copying block `i+1` while block `i` computes took the copy
wait from ~220 ms per step to ~15 ms. The run got *slower* by about the same
amount, and the whole generation went from **28.5 s to 36.0 s**.

**The cause is in candle, and it is device-independent.**
`MetalDevice::allocate_buffer` takes a write lock on a single shared buffer
pool (`Arc<RwLock<BufferMap>>`). The prefetch thread allocates roughly two
dozen buffers per block; the compute thread allocates activations constantly.
They serialise on that lock, so the two never overlap — they take turns with
extra contention on top.

That matters for whether to revisit this. The obvious reading of the numbers
is "unified memory shares bandwidth, so a discrete card would be different" —
but the lock is not a memory-architecture property. It would serialise the
same way over PCIe.

So this is not "prefetch does not pay on this machine", it is "prefetch cannot
pay until allocation stops being globally serialised". Two routes, neither
small: pre-allocate each block's buffers once and reuse them across steps
(sidestepping the allocator entirely), or fix the pool upstream. Measure the
lock before either — this was diagnosed by reading candle's source, not by
profiling it.

### 1. TAESD — the tiny decoder

`madebyollin/taesd` is a ~5 MB distilled replacement for the VAE decoder. Two
things it buys, and the second is the one that matters here:

- **Step previews.** Decoding is currently far too slow to show intermediate
  latents, so a 20-step run is a blank wait. TAESD decodes cheaply enough to
  preview every step.
- **A decode that always fits.** `decode_tiled` already degrades gracefully,
  but a decoder small enough to never need tiling removes the failure mode
  rather than managing it.

Architecturally it is simple — a stack of 3x3 convolutions and ReLU residual
blocks, no attention, no GroupNorm — so this is a short port with an easy
reference (`diffusers.AutoencoderTiny`). The trap to watch: TAESD's latent
convention is its own (`latent_magnitude = 3`, `latent_shift = 0.5`), *not*
the SD VAE's `0.18215`, and mixing them gives a washed-out image rather than
an error. Verify against `AutoencoderTiny.decode` end to end, not just the
module stack, so the scaling is covered by the test.

### 2. Extend streaming past Flux and SD 3.5, and measure it on a discrete GPU

`Residency::Streamed` works for **Flux and SD 3.5**, both quantised. Gaps, in
the order they matter:

- **Dense checkpoints cannot stream at all.** `quantized::to_device` moves
  quantised block bytes verbatim, which is what makes it cheap and bit-exact;
  a dense model would move 4x the bytes with no equivalent shortcut. Flux mini
  — the model that most needs this, at 12.8 GB dense — is therefore the one
  that cannot have it. Whether a dense path is worth it is a measurement
  nobody has made.
- **Nothing prefetches**, and per the profile above that is not the first
  thing to fix. `stable-diffusion.cpp` overlaps copy and compute
  (`stream_layers`); worth doing once the build cost is gone.

And the honest one: **the payoff has not been measured on the hardware it is
for.** On unified memory the host copy sits in the same pool, so what the
device gives up it gives up to the same allocator — the mechanism is
demonstrable there (one block resident instead of the stack, at the default
sync interval) but the *benefit* is not. On a discrete card it should take
VRAM from 6.66 GB to ~192 MB, by construction rather than by measurement. If a
CUDA machine ever appears, measure this before anything else here.

### ~~3. Deduplicate RMSNorm onto `candle_nn::ops::rms_norm`~~ — done, and the
answer was no

The three copies are now one, in `ops::rms_norm`, and `PlainLayerNorm`'s two
copies are one in `ops::plain_layer_norm`. But **candle's fused kernels are
not what they were deduped onto**, and that is the part worth keeping: its
`rms_norm` sums each row with a sequential `.sum::<f32>()` where
`mean_keepdim` reduces in blocks, so it is 4-11x less accurate, worst at long
rows. Measured against f64 at `[1, 154, 4096]`: 9.695e-7 for ours against
9.627e-6 for candle's. Swapping T5 onto it moved `golden_t5` to 3.891e-3,
past a 3e-3 bound that was itself measured. The speed it buys is 2.1x on that
shape and *negative* — 2.7x slower — at `[1, 77, 768]`.

`candle_nn::ops::layer_norm` is the same shape of question and is listed below
untried. Measure it the same way before adopting it; the fused-is-better
assumption has now failed twice.

### Also open

- **ControlNet for SDXL.** The mechanism is architecture-independent —
  `forward_controlled` takes plain tensors and `ControlNet::new` takes a
  `UNetConfig` — so this should be config and a checkpoint, not new code. Worth
  confirming that claim rather than assuming it.
- **Multiple ControlNets at once.** diffusers sums the corrections from several
  before applying them. The summing is trivial; the question is whether the
  API should take a list, and that is worth deciding once rather than twice.
- `candle_nn::rotary_emb::{rope, rope_i, rope_thd}` — fused RoPE. Flux's
  axis-wise 2x2 form may not map onto it; establish rather than assume.
- `candle_nn::ops::{layer_norm, pixel_shuffle, pixel_unshuffle}` — the last
  two are patchify/unpatchify by another name.
- Broaden fused attention to SD 1.5 by materialising `causal_mask` from
  `[1,1,s,s]` to `[b,h,s,s]`. A reshape, not a kernel.
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
#   components: vae flux_vae sd3 flux_transformer flux_sampling t5 flow clip_* unet_* samplers sdxl_*
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

Ungated mirrors matter: `black-forest-labs/*` and `stabilityai/*` are gated.
Use `Freepik/flux.1-lite-8B` for the Flux VAE and T5 tokenizer, and
`adamo1139/stable-diffusion-3.5-medium-ungated` for SD 3.5.
