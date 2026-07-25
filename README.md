# stable-diffusion.rs

Diffusion model inference in pure Rust.

A ground-up Rust implementation in the spirit of
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — no C++
toolchain, no cmake, no submodules. `cargo build` and you have a binary.

> **Status: early.** The VAE decoder is implemented and structurally verified;
> numerical verification against `diffusers` is wired up and runs locally.
> Text-to-image is not working yet. See the [roadmap](docs/roadmap.md).

## Why

`stable-diffusion.cpp` is excellent, and this project exists because of it, not
in spite of it. Three things a Rust implementation gets you:

- **No build ceremony.** Cross-compiling to an ARM target is `cargo build
  --target`, not an afternoon with cmake and a toolchain file.
- **Memory-safe model loading.** GGUF and safetensors parsers ingest files
  people download from the internet, and the C++ equivalents have a CVE
  history. Ours is safe Rust with `unsafe` confined to a single documented
  `mmap`.
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

That keeps one option open: candle is pre-1.0, maintained largely by one
person, and — like ggml — tuned for language models rather than diffusion. If a
kernel becomes the bottleneck, or if candle stalls, we replace it in one crate
instead of rewriting every model. Cheap to keep, impossible to add later.

## Build

```bash
cargo build --release                      # CPU
cargo build --release --features metal     # Apple GPU
cargo build --release --features cuda      # NVIDIA
```

```bash
./target/release/sdrs info
```

The binary is `sdrs`, not `sd` — that name belongs to a
[widely used find & replace tool](https://crates.io/crates/sd) and installing
over it would be rude.

## Verification

The port is checked against `diffusers` tensor by tensor, per module:

```bash
python3 xtask/golden/dump_reference.py vae --output tests/golden
cargo test -p sd-models --test golden_vae -- --nocapture
```

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
