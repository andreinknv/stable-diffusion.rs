# The compute seam

## Rule

Only `sd-tensor` may name a compute backend. Enforced by
`scripts/check-seam.sh` in CI, at both the source level (`use candle_core`,
`use mlx_sys`) and the manifest level.

The rule names candle as well as MLX deliberately. It is about *any* backend,
not whichever one is current — a crate reaching straight for MLX is the same
mistake reaching for candle would have been.

## It was tested, and it held

The backend moved from candle to MLX in 2026. **102 files used tensors and one
of them named the library**, so the swap was bounded to `sd-tensor` plus new
model code — not a workspace-wide rewrite. The seam is the reason that was
possible, and this is the evidence that the rule earns its cost rather than
being a tidiness preference.

One thing the swap revealed: the rule kept *source* clean but a dependency can
still leak through a feature. `tokenizers` needed a regex backend that candle
had been enabling transitively; removing candle turned that into a build
error. Which is the good direction — the dependency was always real and is now
named.

## Why

Every inference engine depends on someone else's kernels.
`stable-diffusion.cpp` depends on `ggml` — and notably vendors it through a
personal fork, `leejet/ggml`, whose `master` carries **zero** custom commits
but whose branches (`optimize_conv_2d`, `new_operators`, `wan`, `sd3`) are a
staging area for diffusion ops that upstream ggml doesn't prioritize.

That fork is the tell. `ggml` is LLM-first: its roadmap follows `llama.cpp`, so
conv2d, group norm and diffusion attention variants are second-class. A
diffusion project ends up a minority stakeholder in someone else's roadmap.

candle carries the same risk, for the same reason — plus a bus factor of
roughly one. It is still the right choice today, because it is the only Rust
option where GGUF quantized inference already works, and that is what every
downloadable model needs. But "right today" is not "right forever."

The seam is the hedge. It costs almost nothing to maintain and cannot be
retrofitted once fifty files import `candle_core` directly.

## What belongs in it

- Re-exports: `Tensor`, `Device`, `DType`, `Result`, `Error`, `VarBuilder`,
  layer constructors
- Ops candle lacks, or where we want our own implementation
- Device selection
- Test helpers for the golden harness

## What does not

Model architecture, weight naming, schedules, sampling. Those are ours and
belong in their own crates. The seam is a **re-export surface, not an
abstraction layer** — it should stay boring and nearly free of logic. If it
grows opinions, it becomes a second thing to debug.

## Replacing candle

Should it come to that, the work is confined to `sd-tensor`:

1. Implement the same surface on the new backend. **But read
   [backends.md](backends.md) first** — for `burn` specifically the seam does
   *not* suffice, because burn puts tensor rank in the type
   (`Tensor<const D: usize>`), which changes every signature in the workspace,
   not just this crate.
2. Port the golden tests first. They are backend-agnostic and will tell you
   immediately what broke.

The realistic path isn't a wholesale swap, though — it's replacing *one kernel*
when profiling proves it matters. `ops::scaled_dot_product_attention` is not
just the current favourite, it is **measured** as the bottleneck: it
materializes a 4096×4096 score matrix at 512×512, enough to make Metal slower
than CPU. Numbers in [backends.md](backends.md).

## Native dependencies

`candle-core` pulls one C dependency — `onig_sys`, via `tokenizers`. It is
fixable with a one-line change upstream, which we have verified builds and
passes every test. Details, measurements and the workaround are in
[native-deps.md](native-deps.md).

Worth noting here because it is the same pattern as the ggml fork: a
diffusion-irrelevant LLM feature (reading a tokenizer out of GGUF metadata)
imposing a cost on every candle user. Not a reason to move off candle. A reason
to keep the seam.
