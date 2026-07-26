# stable-diffusion.rs

Diffusion model inference in pure Rust.

A ground-up Rust implementation in the spirit of
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — no
cmake, no git submodules, no vendored inference engine. `cargo build` and you
have a binary.

> **Status: it renders.** Text-to-image and image-to-image work for
> Stable Diffusion 1.5 and SDXL, on CPU and Apple GPU. Every model component is
> verified tensor-by-tensor against `diffusers`/`transformers` — not eyeballed.
> GGUF reads, dequantises and name-maps — VAE, UNet and text encoder all load
> from a real quantised checkpoint — but is not yet wired into the pipelines. CUDA compiles but
> is untested. See the [roadmap](docs/roadmap.md).

<p align="center">
  <img src="assets/sdxl-crab-1024-metal-f16.png" width="420"
       alt="SDXL, 1024x1024: a rusty crab on a beach">
</p>
<p align="center"><em>SDXL at 1024x1024, 20 steps, DPM++ 2M, 89 s on an
M4 Max. <code>"a rusty crab on a beach, detailed photograph, golden
hour"</code></em></p>

## Why

`stable-diffusion.cpp` is excellent, and this project exists because of it, not
in spite of it. Three things a Rust implementation gets you:

- **No build ceremony.** One `cargo build`, no cmake, no submodule init, no
  toolchain file. Dependencies resolve through cargo like any other crate.
- **Memory-safe model loading.** Weight parsers ingest files people download
  from the internet, and the C++ equivalents have a CVE history. Ours is safe
  Rust with `unsafe` confined to a single documented `mmap`. (safetensors
  loads today; GGUF parses and dequantises, but SD checkpoints in it still
  need a name map.)
- **Embeddable.** A normal crate you add to a Rust application, not an FFI
  boundary you marshal across.

## Design

Models, samplers, and loaders are ours. The tensor math is
[candle](https://github.com/huggingface/candle) — the same bargain
`stable-diffusion.cpp` makes with `ggml`, which it vendors unmodified.

The difference is that candle sits behind a seam:

```
sd-cli ──┐
         ├── sd-models ──┐
sd ──────┤   sd-sample   ├── sd-tensor ── candle
         └── sd-loader ──┘      ▲
                                └── the only crate that names candle
```

`sd-tensor` is a thin re-export surface plus the handful of ops candle lacks.
Everything else goes through it, enforced in CI by
[`scripts/check-seam.sh`](scripts/check-seam.sh).

**Caveat on "pure Rust":** our code is 100% Rust, but one transitive dependency
compiles C. `candle-core` depends on `tokenizers` with the `onig` feature,
which builds `onig_sys` — the oniguruma C regex library — so a C compiler is
required. You never invoke it yourself, and there is still no cmake or
submodule ceremony.

This is **fixable in one line upstream**, and we have verified the fix. It is
also the *only* native code in a CPU build — audited and enforced in CI by
[`scripts/check-native-deps.sh`](scripts/check-native-deps.sh), which fails if
anything else starts compiling C. GPU builds additionally contain CUDA C++ /
Metal shader kernels, which no dependency choice can make Rust. Full audit in
[docs/native-deps.md](docs/native-deps.md).

That keeps one option open: candle is pre-1.0, maintained largely by one
person, and — like ggml — tuned for language models rather than diffusion. If a
kernel becomes the bottleneck, we replace it in one crate instead of rewriting
every model. Cheap to keep, impossible to add later.

Why candle and not burn, cubecl or rust-gpu — with benchmarks, and an honest
account of what the seam does *not* protect against — is in
[docs/backends.md](docs/backends.md).

## Build

```bash
cargo build --release                        # CPU
cargo build --release --features accelerate  # CPU + Apple BLAS (no native code added)
cargo build --release --features metal       # Apple GPU
cargo build --release --features cuda        # NVIDIA
```

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

This matters more than it sounds. A diffusion port fails *quietly* — a
transposed axis yields a plausible but wrong image with no stack trace.
Per-module reference tensors turn "the output looks off" into "`up_block_2`
diverged." See [xtask/golden/README.md](xtask/golden/README.md).

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). There is no
CLA. This is MIT and stays MIT.

Good first targets are in [docs/roadmap.md](docs/roadmap.md); porting a model
architecture against the golden harness is well-scoped, independently
verifiable work.

## License

MIT. See [LICENSE](LICENSE).

Builds on the work of others — [NOTICE](NOTICE) has the details, but briefly:
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (MIT,
© 2023 leejet) for architecture and quantization groundwork,
[candle](https://github.com/huggingface/candle) (MIT/Apache-2.0) for compute,
and [diffusers](https://github.com/huggingface/diffusers) (Apache-2.0) as the
numerical reference.

*Stable Diffusion is a trademark of Stability AI. This project is not
affiliated with or endorsed by Stability AI; the name describes the models it
runs.*
