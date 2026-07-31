# stable-diffusion.rs

Diffusion model inference in pure Rust.

A ground-up Rust implementation: no cmake, no git submodules, no vendored
inference engine. One system library — MLX, installed with `brew install
mlx-c` — and then `cargo build`.

> **Status: it renders.** Eight architectures — Stable Diffusion 1.5, 2.x,
> SDXL, SD 3.5, Flux (schnell/dev), FLUX.1-Kontext, FLUX.2, and unCLIP — on
> Apple GPU via MLX, with
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
| FLUX.2 klein-4B + Qwen3 | **6.7 GB** | 15.8 GB (bf16) |
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

## Install

```bash
brew install mlx-c          # once; pulls mlx
cargo build --release
./target/release/sdrs info
```

The binary is `sdrs`, not `sd` — that name belongs to a
[widely used find & replace tool](https://crates.io/crates/sd).

**GPU or CPU, your choice.** `--cpu` runs the whole pipeline on MLX's CPU
backend. Measured on an M4 Max, SD 1.5 at 256&times;256 and 4 steps: **8.2 s on
the GPU, 14.5 s on the CPU**, producing the same picture — 99.8 % of bytes
identical, largest difference 1/255. The gap widens sharply with size; the GPU
is the point, and the flag exists because a machine whose GPU is busy should
still be able to run.

**Apple silicon, today.** MLX ships a Metal backend and upstream has added
CUDA; this project has only ever been run on Apple silicon, so that is all it
claims. The seam is what would keep another backend a bounded change rather
than a rewrite — it has already survived one swap, from candle, which deleted
37,928 lines and touched `sd-tensor` plus new model code rather than the 102
files that use tensors.

## Use

Point `--model` at a directory in the standard `diffusers` layout. An
unmodified HuggingFace snapshot works as it downloads — **including its
tokenizer, whichever form it ships or if it ships none.** CLIP's vocabulary is
vendored in the library, and it is the same 49,408 entries for every model
here; a checkpoint's own copy is preferred when present.

```bash
# The basics
sdrs txt2img --model models/sd15 --prompt "a rusty crab on a beach" -o out.png

# SDXL, at the size it was trained for
sdrs txt2img --model models/sdxl --sdxl --width 1024 --height 1024 \
  --prompt "a lighthouse at dusk" --steps 30 --sampler dpmpp2m -o out.png

# Redraw an existing image. --strength 0 returns it, 1 ignores it.
sdrs img2img --model models/sd15 --init out.png --strength 0.6 \
  --prompt "a watercolour painting" -o painted.png

# Repaint only where the mask is white
sdrs img2img --model models/sd15 --init photo.png --mask mask.png \
  --prompt "a bunch of flowers" -o fixed.png

# Make it 4x bigger
sdrs upscale --model models/sd15 --weights esrgan_x4.safetensors \
  --input out.png -o big.png

# Edit by instruction, rather than by describing the result
sdrs instruct --model models/instruct-pix2pix --init photo.png \
  --prompt "make it winter, with snow" -o edited.png

# Flux schnell: 12B parameters, quantised at rest, on a 36 GB laptop
sdrs flux --model models/flux --variant schnell \
  --transformer-gguf flux1-schnell-Q4_K_S.gguf --t5-gguf t5xxl-Q4_K_S.gguf \
  --prompt "a rusty crab on a beach at sunset" -o out.png

# FLUX.2 — a Qwen3 text encoder, quantised at rest
sdrs flux2 --model models/flux2-klein --variant klein-4b \
  --prompt "a rusty crab on a beach at sunset" --steps 4 -o out.png

# SD 3.5, with skip-layer guidance for anatomy
sdrs sd3 --model models/sd35 --prompt "a lighthouse at dusk" \
  --slg-layers 7,8,9 -o out.png

# FLUX.1-Kontext: edit by instruction, keeping the picture
sdrs flux --model models/flux --variant dev --reference photo.png \
  --transformer-gguf flux1-kontext-dev-Q4_K_S.gguf --t5-gguf t5xxl-Q4_K_S.gguf \
  --prompt "make it winter, with snow" --guidance 2.5 -o edited.png

# Blend two checkpoints
sdrs merge --a base.safetensors --b style.safetensors --alpha 0.3 -o mix.safetensors
```

### Going further

Every flag below is optional and composes with the rest.

| want | flag |
|---|---|
| the schedule everyone else's step counts assume | `--scheduler karras` |
| a sampler other than the default | `--sampler dpmpp2m` (also `euler`, `heun`, `dpmpp2s-a`, `ddim`, `lcm`) |
| what most community finetunes expect | `--clip-skip 2` |
| several images from one load | `-n 4` (seeds run `seed`, `seed+1`, ...) |
| a texture that tiles | `--seamless` |
| less memory, a comparable picture | `--precision f16` |
| a style adapter | `--lora lcm.safetensors --lora-scale 1.0` |
| follow an image's edges/depth | `--controlnet cn.safetensors --control-map edges.png` |
| a trained trigger word | `--embedding mystyle=mystyle.safetensors` |
| a short animation | `--motion-adapter adapter.safetensors --frames 16` |
| detail without duplicated subjects | `--hires 1024x1024 --hires-strength 0.55` |
| different prompts in different places | `--region left.png="sunflowers"` |
| fewer model evaluations | `--cache-threshold 0.2` (needs `--sampler dpmpp2m`) |
| upscale in the same run | `--upscale esrgan_x4.safetensors` |
| style and content from an image | `--ip-adapter ip.safetensors --image-encoder enc.safetensors --reference ref.png` |
| put a thing in a place | `--grounded-box 0.1,0.1,0.5,0.5="a red car"` |
| edges without preparing a map | `--canny photo.png` (with `--controlnet`) |
| a fast, soft preview decode | `--taesd taesd.safetensors` |

**`--hires` is not the same as generating big.** A model composes at its
training resolution and duplicates subjects above it — two heads, two horizons.
Compose small, refine large.

**`--cache-threshold` needs a deterministic sampler.** `euler-a`, `dpmpp2s-a`
and `lcm` draw fresh noise every step, so there is nothing to reuse; asking
anyway is an error rather than a silent no-op.

**`heun` and `dpmpp2s-a` evaluate the model twice per step**, so twenty of
their steps cost about what forty Euler steps do. The progress line reports
evaluations alongside steps, so the difference is visible rather than inferred.

**Every image records how it was made.** The generation parameters go into the
PNG as a text chunk in the format A1111 and its readers use, so an image a year
later can still say its seed, sampler, schedule and step count.

### One thing that will bite you

**The same seed reproduces the same picture, and on the same machine the same
file.** Across machines the reduction order differs, so bytes will not match.
Do not build a cache key on cross-machine byte equality.

### As a library

```rust
use stable_diffusion_rs::mlx::MlxPipeline;
use stable_diffusion_rs::config::Txt2ImgConfig;

let pipe = MlxPipeline::load("models/sd15".as_ref())?;
let (w, h, rgb) = pipe.txt2img(&Txt2ImgConfig {
    prompt: "a rusty crab on a beach".into(),
    ..Default::default()
})?;
```

The library exposes more than the CLI does — SD 3.5 and Flux pipelines,
unCLIP image variations, IP-Adapter and GLIGEN grounding all have entry points
under `stable_diffusion_rs::mlx`.

## Verification

Run the suite with [`scripts/test.sh`](scripts/test.sh), which sets
`--test-threads=3`. That is not tuning: each pipeline test loads a full model
into unified memory, and at cargo's default the OOM killer takes the test
binary — surfacing as `signal: 9` with no failing assertion, which reads
exactly like a crash in the code under test.


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
