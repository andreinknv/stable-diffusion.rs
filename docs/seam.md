# The compute seam

## Rule

Only `sd-tensor` may name candle. Enforced by `scripts/check-seam.sh` in CI, at
both the source level (`use candle_core`) and the manifest level.

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

1. Implement the same surface on the new backend (`burn` is the credible
   alternative — a real team behind it, and `cubecl` compiles one kernel source
   to CUDA/ROCm/wgpu/Metal).
2. Port the golden tests first. They are backend-agnostic and will tell you
   immediately what broke.
3. No model crate changes.

The realistic path isn't a wholesale swap, though — it's replacing *one kernel*
when profiling proves it matters. `ops::scaled_dot_product_attention` is the
current favourite: it materializes the full `seq × seq` score matrix, which is
fine for VAE attention and won't be for UNet cross-attention.
