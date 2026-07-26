# The native-code budget

Our code is entirely Rust. Exactly **one** transitive dependency compiles C,
and it is avoidable with a **one-line upstream change** we have tested end to
end.

This is enforced, not aspirational: [`scripts/check-native-deps.sh`](../scripts/check-native-deps.sh)
runs in CI and fails if any crate outside the allowlist compiles native code.
A C dependency arrives transitively, in someone else's `Cargo.toml`, and
nothing about your build looks different when it does — so it gets checked.

```
$ ./scripts/check-native-deps.sh
native deps ok [default]: 1 allowlisted, 0 unexpected
    onig_sys (known)
```

## Full audit

Measured across every backend:

| Build | Compiles native code | Removable? |
|---|---|---|
| default (CPU) | `onig_sys` only | **yes** — one line upstream |
| `--features accelerate` | `onig_sys` only | same; Accelerate links a system framework, compiles nothing |
| `--features metal` | `onig_sys` + `candle-metal-kernels` | no — Metal Shading Language |
| `--features cuda` | `onig_sys` + `candle-kernels`, `cudaforge` | no — CUDA C++ |

Nothing else in the tree pulls `cc`. There are no other `-sys` crates.
`ring`/`rustls` are **not** reachable from our binaries — they appear only
under `cargo deny --all-features`, which enables backends we do not ship by
default.

**GPU kernels cannot be Rust here.** CUDA kernels are CUDA C++ and Metal
shaders are MSL; candle has no Rust-authored kernel path. The only way to an
all-Rust GPU stack is a backend with a Rust kernel DSL — `cubecl` (burn) or
`rust-gpu`. That is a seam-level decision, not a dependency tweak. See
[seam.md](seam.md).

So: **after the `onig` fix, CPU builds are 100% Rust.** GPU builds contain
kernel code that is inherently not Rust, and no dependency choice changes that.

## What is actually native

Measured, not assumed — count the object files a build produces:

| Crate | Native artifacts | Verdict |
|---|---:|---|
| `onig_sys` | **49 objects, 3.5 MB** | real C compilation (oniguruma) |
| `esaxx-rs` | **0 objects, 0 bytes** | pure Rust here |

`esaxx-rs` appears in `cargo tree` and compiles nothing: its C++ path is behind
its own `cpp` feature, enabled only by `tokenizers/esaxx_fast`, which
`candle-core` does not turn on. Being in the dependency graph is not the same
as building native code — check the artifacts before claiming otherwise.

So `onig_sys` is the entire problem.

## Where it comes from

```
sd-tensor -> candle-core -> tokenizers (features = ["onig"]) -> onig -> onig_sys
```

`candle-core/Cargo.toml:38`:

```toml
tokenizers = { workspace = true, features = ["onig"] }
```

Used by exactly one file, `candle-core/src/quantized/tokenizer.rs` (11 KB),
which reconstructs a tokenizer from GGUF metadata — an LLM convenience feature
that no diffusion model touches.

It is the ggml pattern again: a tensor library carrying an LLM-shaped
dependency that every other user pays for. Note that **we cannot switch it off
from our side.** Cargo unifies features across the graph, so once `candle-core`
asks for `tokenizers/onig`, nothing downstream can un-ask.

## The fix

`tokenizers` already ships a pure-Rust regex backend, selected by feature, used
for WASM builds where C cannot compile — `tokenizers/src/utils/mod.rs`:

```rust
#[cfg(all(feature = "fancy-regex", not(feature = "onig")))]
pub use fancy::SysRegex;
#[cfg(feature = "onig")]
pub use crate::utils::onig::SysRegex;
```

Same type name, two implementations. So the change is a drop-in:

```diff
-tokenizers = { workspace = true, features = ["onig"] }
+tokenizers = { workspace = true, features = ["fancy-regex"] }
```

Note `onig` wins when both are enabled, which is why *adding* `fancy-regex`
downstream does not help. The `onig` feature has to be removed at the source.

## Verified

Applied to candle 0.11.0 and built this workspace against it:

```
onig in dependency tree ....... eliminated
fancy-regex v0.14.0 ........... substituted
cargo build --workspace ....... success
cargo test --workspace ........ 23 passed, 0 failed
```

No source changes anywhere — not in candle, not here. Only the feature flag.

## Using it today

Add to the **workspace root** `Cargo.toml`:

```toml
[patch.crates-io]
candle-core = { git = "https://github.com/<you>/candle", branch = "fancy-regex" }
candle-nn   = { git = "https://github.com/<you>/candle", branch = "fancy-regex" }
```

**Important limitation:** `[patch]` applies only to the workspace that declares
it. It does **not** propagate to anyone who depends on `stable-diffusion-rs` as
a library — they still get `onig`. That is why this is documented rather than
shipped as the default. Making our build depend on a candle fork would trade a
C dependency for a fork-maintenance burden, which is a worse deal and exactly
the trap we criticize elsewhere.

Worth doing if you are cross-compiling (a cross C toolchain for oniguruma is
real friction), targeting WASM, or auditing native code in your supply chain.

## Upstreaming

The right fix is a PR to candle making this change, or making `tokenizers`
optional in `candle-core` and feature-gating `quantized::tokenizer` — the
latter is better still, since most candle users need neither.

Either way it is small, mechanical, and benefits every candle user. Not yet
filed; see the roadmap.
