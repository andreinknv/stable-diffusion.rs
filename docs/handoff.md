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

Measured on Flux schnell, 512, 4 steps, Metal: **21.0 s resident against
25.1 s streamed**, +19.5%, and the images are **bit-identical** — the copy
moves quantised block bytes verbatim, so nothing is rounded twice.
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
table is in [roadmap.md](roadmap.md). 243 tests, all gates green
(`cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`,
`scripts/check-seam.sh`, `scripts/check-native-deps.sh`).

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

### 1. Extend streaming past Flux, and measure it on a discrete GPU

`Residency::Streamed` exists and works — see the note above — but only for
**Flux's quantised transformer**. Three gaps, in the order they matter:

- **SD 3.5's MMDiT does not stream.** Same shape of change:
  `Sd3Transformer` needs the `Blocks` split that `flux/mod.rs` now has, and
  `sd3_qtensors_from_gguf` already loads the weights it would keep in host
  memory. This is the mechanical one.
- **Dense checkpoints cannot stream at all.** `quantized::to_device` moves
  quantised block bytes verbatim, which is what makes it cheap and bit-exact;
  a dense model would move 4x the bytes with no equivalent shortcut. Flux mini
  — the model that most needs this, at 12.8 GB dense — is therefore the one
  that cannot have it. Whether a dense path is worth it is a measurement
  nobody has made.
- **Nothing prefetches.** Each block is copied when it is reached, so the
  copy and the compute serialise. `stable-diffusion.cpp` overlaps them
  (`stream_layers`), which should hide most of the 19.5% this currently costs.
  That is the obvious next win and needs no new concepts.

And the honest one: **the payoff has not been measured on the hardware it is
for.** On unified memory the host copy sits in the same pool, so freeing the
device buys only about 2.4 GB of the 6.78 GB in play. On a discrete card the
same mechanism should take VRAM from 6.78 GB to one block, ~192 MB — by
construction, not by measurement. If a CUDA machine ever appears, measure this
first.

### 2. Deduplicate RMSNorm onto `candle_nn::ops::rms_norm` — **done, and the
answer was no**

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
- SDXL img2img is unverified end to end after the encoder-tiling fix.

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
