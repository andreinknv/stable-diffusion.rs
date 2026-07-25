# Contributing

Contributions are welcome. **There is no CLA** — this project is MIT and stays
MIT. Your commits stay yours.

## Getting set up

```bash
cargo build --workspace
cargo test --workspace
./scripts/check-seam.sh
```

## The one hard rule

**Only `sd-tensor` may depend on candle.** No other crate may `use
candle_core` / `candle_nn` or declare them in `Cargo.toml`.

If you need an op the seam doesn't expose, add it to
`crates/sd-tensor/src/lib.rs` and use it from there. CI enforces this and it is
not negotiable — the seam is what keeps swapping the compute backend a
one-crate change rather than a rewrite. See [docs/seam.md](docs/seam.md).

## Porting a model component

This is the highest-value contribution, and the workflow is deliberately
mechanical:

1. **Add a reference dump** to `xtask/golden/dump_reference.py`, capturing
   per-module activations via forward hooks — not just the final output.
2. **Write the failing test** in `crates/sd-models/tests/`. It should skip when
   reference data is absent so CI stays green.
3. **Implement** until it passes at `atol = 1e-4`.
4. **Add a structural test** that runs without reference data (shapes, channel
   counts, scale factors) so CI catches regressions.

Follow `diffusers` parameter naming exactly. Pretrained weights should load
without a conversion table; if a checkpoint uses the original CompVis layout,
conversion belongs in `sd-loader`, not in the model.

## Debugging a numerical mismatch

Compare intermediates **in order**. The first tensor that diverges is the bug;
everything after it is carrying the error forward.

Before suspecting a kernel, check — roughly in order of likelihood:

- axis order (`permute` / `transpose` / `reshape`)
- parameter naming (a silently-missing weight loads as zeros or errors late)
- normalization epsilon (the VAE uses `1e-6`, not torch's `1e-5` default)
- activation variant — `gelu`, `gelu_erf` and `quick_gelu` are three different
  functions and models are sensitive to the difference
- block counts (the VAE decoder uses `layers_per_block + 1`)

## Style

- `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings`
  must pass.
- Comment *why*, not *what*. Non-obvious constants and layout choices deserve a
  line; `// increment i` does not.
- New `unsafe` needs a `// SAFETY:` comment stating the invariant. This crate
  parses untrusted files — memory safety is a feature we advertise.

## Performance

Correctness first, on CPU, in f32. Get the golden test passing before adding a
GPU path. Debugging a wrong kernel and a wrong architecture simultaneously is
how ports stall.

## Reporting bugs

Include: OS, `rustc --version`, features enabled, the model file, and the
output of `sdrs info`. For wrong-image bugs, the golden test output is far more
useful than the image.
