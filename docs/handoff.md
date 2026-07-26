# Handoff

Written 2026-07-26. Say **"go"** to resume: read this file, pick the top
unstarted item under [Next](#next), and start it.

## Where things stand

Four architectures render, all **on CPU only**:

| model | result |
|---|---|
| SD 1.5 | 512x512, 20 steps, 113 s — `assets/crab-512-dpmpp2m-seed42.png` |
| SDXL | 1024x1024, 20 steps, 89 s **on Metal** — `assets/sdxl-crab-1024-metal-f16.png` |
| Flux schnell (12B) | 512x512, 4 steps, 166 s — `assets/flux-schnell-512-crab.png` |
| Flux mini (3.2B) | 512x512, 20 steps, 212 s — `assets/flux-mini-512-crab.png` |
| SD 3.5 medium | 512x512, 20 steps, 311 s — `assets/sd35-medium-512-crab.png` |

Every component is verified against `diffusers`/`transformers` — the full
table is in [roadmap.md](roadmap.md). 224 tests, all gates green
(`cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`,
`scripts/check-seam.sh`, `scripts/check-native-deps.sh`).

## Next

In the order I would do them, with why.

### 1. CPU flash attention  — largest available win, unexplored

`candle_nn::cpu_flash_attention::run_flash_attn_cpu(q, k, v, mask, scale,
max_bias, softcap)` exists in candle 0.11 and nothing here has tried it. CPU
is the path every model actually runs on today, and attention is the
dominant cost: 957 ms for one Flux-1024 attention call against 43.8 ms fused
on Metal.

Wire it as a new arm in `ops::attention_with_path`
(`crates/sd-tensor/src/lib.rs`), returning a new `AttentionPath` variant so
tests can assert which path ran — the existing enum exists for exactly that
reason. Measure with `--example attention_path`, which already prints path
and timing per model shape.

### 2. Localise the Metal corruption — Metal is 7.3x faster and wrong

Flux schnell on Metal renders in 22.7 s instead of 166 s and produces a flat
orange field with a corrupted top strip. `--example metal_check` compares CPU
against Metal per op; everything it covers agrees (fused attention 1.9e-7,
QLinear within quantisation noise, matmul exact). So it is composition or an
op that check does not reach.

Extend `metal_check` to the untested ops in this order: the VAE decode
(convolutions), `PlainLayerNorm`/`RmsNorm`, then RoPE. **Metal has failed in
this exact shape here before** — a 1024 decode returned noise because candle
never checks the Metal command buffer unless something synchronises, and the
real error was invisible until `Decoder::forward` was made to synchronize. Try
forcing a synchronize after each stage first; it is cheap and it worked last
time.

### 3. Deduplicate RMSNorm onto `candle_nn::ops::rms_norm`

Three hand-written copies exist — `sd-models/src/t5/mod.rs`,
`flux/mod.rs`, `sd3/mod.rs` — each doing its own f32 upcast. candle ships a
fused one. Mechanical, and the golden tests (`golden_t5`,
`golden_flux_transformer`, `golden_sd3`) will catch any error immediately.

### Also open

- `candle_nn::rotary_emb::{rope, rope_i, rope_thd}` — fused RoPE. Flux's
  axis-wise 2x2 form may not map onto it; establish rather than assume.
- `candle_nn::ops::{layer_norm, pixel_shuffle, pixel_unshuffle}` — the last
  two are patchify/unpatchify by another name.
- Broaden fused attention to SD 1.5 by materialising `causal_mask` from
  `[1,1,s,s]` to `[b,h,s,s]`. A reshape, not a kernel.
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

**Absolute tolerances are wrong for text encoders.** CLIP peaks at 851, T5 at
~200,000, SD 3.5's blocks at ~97,000. Use `testing::allclose_excess(a, b,
rtol)`. Where a loose bound is unavoidable, *measure the reference
implementation's own f32-vs-f64 spread* and cite it — that is how the Flux
VAE encoder's 2e-3 bound and T5's 3e-3 were set, and both turned out to sit
at or below the reference's own noise floor.

**F16 is not a safe way to halve memory.** T5's activations exceed 65,504 and
go NaN around block 10; Flux's transformer NaNs too. Hold weights quantised
instead (`weights::Source::Quantized`) — activations stay f32 and residency
drops further than F16 would have. bf16 would fix it but candle's CPU backend
has no bf16 matmul.

**Check whether candle already does it.** The roadmap called a hand-written
fused Metal attention kernel the highest-value work available; candle 0.11
already shipped one and `attention_with_path` was already routing to it. Only
a stale doc comment said otherwise.

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
