# stable-diffusion.rs

Diffusion model inference in pure Rust.

A ground-up Rust implementation: no cmake, no git submodules, no vendored
inference engine. One system library — MLX, installed with `brew install
mlx-c` — and then `cargo build`.

> **Status: it renders.** Six architectures — Stable Diffusion 1.5, 2.x,
> SDXL, SD 3.5, Flux schnell, and unCLIP — on Apple GPU via MLX, with
> img2img, inpainting, ControlNet, LoRA, IP-Adapter, GLIGEN, textual
> inversion, AnimateDiff clips, region prompts, two-pass hires, step caching
> and 4x upscaling. Every model component is verified tensor-by-tensor
> against `diffusers`/`transformers`, not eyeballed.

<p align="center">
  <img src="assets/readme-sdxl-lighthouse.png" width="520"
       alt="SDXL at 1024x1024: a lighthouse on a cliff at dusk">
</p>
<p align="center"><em>SDXL, 1024&times;1024, 30 steps, DPM++ 2M.<br>
<code>sdrs txt2img --model models/sdxl --sdxl --steps 30 --sampler dpmpp2m</code></em></p>

### What fits on a 36 GB laptop

Weights are held **quantised at rest** — packed bits, dequantised inside the
matmul kernel a tile at a time — so models that cannot be loaded densely run
anyway:

| model | resident | dense f32 |
|---|---|---|
| Flux schnell + T5-XXL | **13.3 GB** | 66 GB |
| SD 3.5 medium + T5-XXL | **10.1 GB** | ~40 GB |
| SDXL | 10.3 GB | 10.3 GB |

4-bit everywhere is *not* good enough — measured 0.933 cosine against dense
across a whole transformer, which is a visibly different picture. The layers
that multiply the residual stream are held at 8 bits and the rest at 4, which
reaches 0.992 at 19 % of dense. The numbers and the reasoning are in
[`quantized.rs`](crates/sd-models/src/mlx/quantized.rs).

## Why

Three things this implementation provides:

- **Almost no build ceremony.** `brew install mlx-c`, then one `cargo build`.
  No cmake, no submodule init, no toolchain file; every Rust dependency
  resolves through cargo like any other crate.
- **Memory-safe model loading.** Weight parsers ingest files people download
  from the internet. These are safe Rust, with `unsafe` confined to the FFI
  boundary and one documented `mmap`. safetensors and GGUF both load, and a
  GGUF checkpoint in the older CompVis naming is translated on the way in.
- **Embeddable.** A normal crate you add to a Rust application, not an FFI
  boundary you marshal across.

## Design

Models, samplers, and loaders are implemented here. The tensor math is
[MLX](https://github.com/ml-explore/mlx), which sits behind a seam:

```
sd-cli ──┐
         ├── sd-models ──┐
sd ──────┤   sd-sample   ├── sd-tensor ── MLX
         └── sd-loader ──┘      ▲
                                └── the only crate that names a backend
```

`sd-tensor` binds `mlx-c` directly — hand-written, because `bindgen` would put
libclang in every build of the crate for a few dozen declarations.
Everything else goes through it, enforced in CI by
[`scripts/check-seam.sh`](scripts/check-seam.sh).

**Caveat on "pure Rust":** our code is 100% Rust, and no dependency compiles C.
`tokenizers` needs a regex backend and we ask for `fancy-regex`, which is pure
Rust, rather than `onig`, which is not. Audited and enforced in CI by
[`scripts/check-native-deps.sh`](scripts/check-native-deps.sh), which fails if
anything starts compiling C. MLX itself is a system library — installed with
`brew install mlx-c`, linked, not built here — and its kernels are Metal, which
no dependency choice can make Rust. Full audit in
[docs/native-deps.md](docs/native-deps.md).

The seam is not decoration: **the backend has already been replaced once.** It
was candle until 2026, and the swap touched `sd-tensor` plus new model code
rather than the 102 files that use tensors. Why MLX, what the move cost and
what the seam does *not* protect against are in
[docs/backends.md](docs/backends.md).

## Build

```bash
brew install mlx-c      # pulls mlx; once, not per build
cargo build --release
```

`build.rs` finds MLX through `MLX_C_PREFIX`/`MLX_PREFIX` or Homebrew, and
links nothing at all without the `mlx` feature — so `--no-default-features`
still compiles on a machine that has never seen it, which is what lets CI
check the crate graph anywhere.

**Apple silicon only, today.** MLX is Apple's. The seam is what would make
another backend a bounded change rather than a rewrite — it has already
survived one swap, from candle — but nothing else is wired up.

```bash
./target/release/sdrs info
```

The binary is `sdrs`, not `sd` — that name belongs to a
[widely used find & replace tool](https://crates.io/crates/sd) and installing
over it would be rude.

## Use

A model directory in the standard `diffusers` layout:

```bash
sdrs txt2img --model ./models/sd15 \
  --prompt "a rusty crab on a beach" \
  --steps 20 --seed 42 -o out.png

sdrs txt2img --sdxl --model ./models/sdxl \
  --prompt "a rusty crab on a beach, golden hour" \
  --width 1024 --height 1024 -o out.png

sdrs img2img --model ./models/sd15 --init-image out.png \
  --prompt "a watercolour painting of a crab" --strength 0.75 -o painted.png

# Generate a variation of an image: its subject, not its pixels. No pixel of
# the reference reaches the model — only one CLIP embedding of the whole image.
sdrs unclip --model ./models/unclip --init-image ref.png --cfg-scale 5.0

# Or from a prompt alone, through the checkpoint's prior.
sdrs unclip --model ./models/unclip-t2i --prompt "a crab on a beach"

# A single quantised checkpoint, as stable-diffusion.cpp writes them. These
# carry no tokenizer, so one has to be supplied.
sdrs txt2img --gguf sd15-q4_k.gguf --tokenizer tokenizer.json \
  --prompt "a rusty crab on a beach" -o out.png

# No published SD 1.5 ships k-quants — its 320-channel blocks do not divide
# into their 256-value blocks. Make one from any other GGUF:
cargo run --release -p sd-cli --example requantise -- in.gguf out.gguf Q4_K
```

The same seed on the same device and build reproduces a file byte for byte.
Across devices it reproduces the *picture*, not the file — f32 reduction order
differs per backend. Do not build a cache key on cross-device byte equality.

### Two things that will bite you

**A stock SD 1.5 or SDXL download has no `tokenizer/tokenizer.json`** — the
repositories ship the slow tokenizer (`vocab.json` + `merges.txt`). Copy
`tokenizer.json` from `openai/clip-vit-large-patch14`. The error says so.

**Large jobs are refused rather than allowed to thrash.** `sdrs` checks what
the machine actually has free before loading, weights included, and declines
if the run would not fit. That is why the same command can succeed and later
be refused: something else took the memory. `SD_MEMORY_HEADROOM` overrides it.
See [docs/backends.md](docs/backends.md).

## Verification

Every component is checked against `diffusers`/`transformers` tensor by
tensor, per module:

| component | what is compared | agreement |
|---|---|---|
| VAE decoder | final image | `max_abs` 3.7e-5 |
| VAE encoder | latent moments | `max_abs` 8.2e-5 |
| CLIP tokenizer | 6 prompts | id-for-id |
| CLIP text encoder | all 12 layers + output | output `max_abs` 3.5e-5 |
| UNet (SD 1.5) | 12 skips, mid block, output | output `max_abs` 1.1e-5 |
| SDXL text encoder 2 | penultimate + pooled | `max_abs` 1.4e-4 |
| SDXL UNet | output, with micro-conditioning | `max_abs` 1.4e-5 |
| samplers | 6 steps each, both solvers | `max_abs` ~1e-7 |
| ControlNet (SD 1.5) | all 13 corrections | worst excess 1.5e-5 |
| ControlNet (SDXL) | 9 corrections + mid, with micro-conditioning | within 1e-3 |
| Flux MMDiT | whole transformer | relative drift 2.1e-6 |
| SD 3.5 MMDiT | whole transformer | `max_abs` 5.5e-6 |
| T5 v1.1 encoder | output | `max_abs` 1.9e-5 |
| unCLIP UNet | output with image conditioning | 2.0e-4, floor 2.8e-4 |
| unCLIP prior | 20-block transformer | 3.2e-6 |
| TAESD (4 variants) | decode and encode | `max_abs` 1.9e-5 |
| Real-ESRGAN | whole RRDBNet | `max_abs` 1.7e-6 |

The per-layer comparisons inside CLIP use `|a-b| <= atol + rtol*|b|` rather
than absolute error, because CLIP carries activations of magnitude 851 and f32
cannot hold 1e-4 absolute at that scale — a detail that would otherwise read
as a failure. `docs/roadmap.md` explains it.

```bash
python3 xtask/golden/dump_reference.py vae --output tests/golden
cargo test -p sd-models --test golden_vae -- --nocapture
```

Reference data is generated locally and stays out of git, so CI runs the
structural tests and skips the numerical ones. `dump_reference.py` has a
subcommand per component.

**Skipping is a pass, so a plain run reports the same green with no fixtures
at all.** `SD_REQUIRE_FIXTURES=1` turns every "no reference data" skip into a
failure, which is the run that actually verifies something:

```bash
SD_REQUIRE_FIXTURES=1 SD_TEST_MODEL_DIR=$(pwd)/models/sd15 \
  cargo test --release --workspace
```

This matters more than it sounds. A diffusion port fails *quietly* — a
transposed axis yields a plausible but wrong image with no stack trace.
Per-module reference tensors turn "the output looks off" into "`up_block_2`
diverged." See [xtask/golden/README.md](xtask/golden/README.md).

Work in progress and what to pick up next is in
[docs/handoff.md](docs/handoff.md).

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). There is no
CLA. This is MIT and stays MIT.

Good first targets are in [docs/roadmap.md](docs/roadmap.md); porting a model
architecture against the golden harness is well-scoped, independently
verifiable work.

## License

MIT. See [LICENSE](LICENSE).

Builds on the work of others — see [Standing on](#standing-on) below and
[NOTICE](NOTICE) for the details. Briefly:
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (MIT,
© 2023 leejet) for architecture and quantization groundwork,
[MLX](https://github.com/ml-explore/mlx) (MIT) for compute,
and [diffusers](https://github.com/huggingface/diffusers) (Apache-2.0) as the
numerical reference.

*Stable Diffusion is a trademark of Stability AI. This project is not
affiliated with or endorsed by Stability AI; the name describes the models it
runs.*

## Standing on

What this project uses, and for what:

- **[MLX](https://github.com/ml-explore/mlx)** and
  **[mlx-c](https://github.com/ml-explore/mlx-c)** — tensor and compute
  backend, reached only through `sd-tensor`. Its fused kernels are used where
  they apply: `scaled_dot_product_attention`, `layer_norm`, `rms_norm` and
  quantised matmul.
- **[candle](https://github.com/huggingface/candle)** — the backend until
  2026, and the reference the MLX port was checked against tensor by tensor
  while both existed.
- **[diffusers](https://github.com/huggingface/diffusers)** and
  **[transformers](https://github.com/huggingface/transformers)** — the
  reference implementations every component here is compared against, tensor
  by tensor. Module layouts and parameter names follow theirs so pretrained
  weights load unmodified.
- **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** —
  architecture layouts, GGUF conventions, weight-name maps and the
  block-streaming and placement designs.
- **[tokenizers](https://github.com/huggingface/tokenizers)** — CLIP's BPE,
  loaded from `tokenizer.json`.

The models themselves belong to their authors and carry their own licences —
Stability AI, Black Forest Labs, kakaobrain, OpenAI, Google, and the research
behind ControlNet, IP-Adapter, GLIGEN, AnimateDiff, LCM, LoRA and TeaCache.
Several are only testable on a laptop because other people publish quantised
conversions and ungated mirrors. [NOTICE](NOTICE) has the full list.
