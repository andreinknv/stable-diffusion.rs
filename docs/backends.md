# Backend evaluation

Why candle, what the alternatives are, and what the seam does — and does not —
protect against.

This exists so the question is not re-litigated from scratch every time someone
reasonably asks "why not burn?". The answer has real numbers behind it.

## The layers

The word "backend" gets used for two different things. Keeping them apart makes
most of the confusion go away:

```
model code           UNet, VAE, CLIP                        <- ours
tensor library       candle | burn                          <- candle is HERE
kernel authoring     CUDA C++ | MSL | cubecl | rust-gpu     <- and cubecl is HERE
GPU API              cudarc | wgpu | metal
```

A tensor library gives you `Tensor`, dtypes, broadcasting, memory, model
loading and a few hundred ops. A kernel language gives you a way to write *one*
op. They are not substitutes.

## The candidates

| | What it is | Verdict |
|---|---|---|
| **candle** | tensor library | **in use** |
| **burn** | tensor library on cubecl | credible, not now — see below |
| **cubecl** | kernel language, compiles to cuda/hip/metal/spirv/wgpu/cpp | not a candle replacement |
| **rust-gpu** | rustc backend, Rust -> SPIR-V | not a candle replacement; not production-ready |
| **tch-rs** | LibTorch bindings | ~2 GB C++ dependency |
| **ort** | ONNX Runtime bindings | C++ dependency; wrong shape for this |
| **tract** | pure-Rust inference | CPU only |

### Why not cubecl or rust-gpu directly

Neither ships a tensor library. cubecl's crates are all compiler and runtime —
`cubecl-core`, `cubecl-ir`, `cubecl-opt`, plus codegen targets. rust-gpu is
`rustc_codegen_spirv` and friends.

Building on either means writing `Tensor`, strides, dtypes, broadcasting,
memory management and ~60 op families yourself. That is the from-scratch
option, costed at 8-15 months elsewhere. rust-gpu additionally targets only
SPIR-V — no CUDA, no native Metal — and its own README says it is "not yet
production-ready".

### Why not burn (yet)

burn is a real alternative and stronger than a first glance suggests.
`burn-store` has dedicated `safetensors/` and `pytorch/` loaders plus
`keyremapper.rs`, which is precisely the diffusers weight-name mapping we need.
Layers live in `burn-nn`. Governance is better than candle's: a company behind
it, contributor spread 720/442/313/172, and ~100 commits per 90 days against
candle's ~48. `cubecl` gives it one kernel source compiling to CUDA, ROCm,
Metal, wgpu and SPIR-V, which is architecturally nicer than per-backend kernels
and would make GPU kernels genuinely Rust.

Two things keep us on candle:

**GGUF.** 2 code hits in burn against 94 in candle. Every quantised model
people actually download is a `.gguf` with Q4_K/Q5_K/Q8_0 weights. On burn we
would write the dequantisation path and the quantised matmul kernels ourselves
before rendering anything — which collapses back into the from-scratch option.

**The type model, which the seam does not abstract.** See below.

Worth noting the asymmetry: candle's problems are a one-line upstream fix
(`onig`) and a bus factor. burn's is months of quantisation work. Those are not
comparable in cost.

## What the seam does not protect against

The seam is real and it earns its keep, but it has been oversold in earlier
notes and this is the correction.

```
candle:  Tensor                       dynamic rank, dynamic dtype, no type params
burn:    Tensor<const D: usize, K>    rank is a CONST GENERIC in the type
```

`sd-tensor` is type aliases and re-exports over candle's `Tensor`. Moving to
burn changes the *shape of every signature in the workspace*:
`fn forward(&self, xs: &Tensor)` becomes `fn forward(&self, xs: Tensor<4>)`,
and every intermediate value carries its rank in its type.

**The seam contains the dependency. It cannot contain a change in the type
system's shape.** Concretely:

| Change | Seam protects? |
|---|---|
| Replace one kernel with our own | **yes** — that is the design case |
| candle renames or changes an API | **yes** |
| Swap to a candle-shaped library | **yes** |
| Swap to burn | **no** — model code needs rewriting too |
| Drop a transitive dependency | **no** — that is upstream's call |

So the escape hatch is cheap and worth keeping, but "we can swap the backend in
one crate" is only true for backends shaped like candle. Do not plan on it for
burn.

## Does candle GPU actually work?

Yes. Verified on an Apple M4 Max, macOS, release build — full SD 1.5 VAE
decoder geometry, best of 3 after a warm-up:

| latent | image | attention seq | CPU | Metal | |
|---:|---:|---:|---:|---:|---|
| 16x16 | 128x128 | 256 | 567 ms | **114 ms** | 5.0x faster |
| 32x32 | 256x256 | 1024 | 1.97 s | **458 ms** | 4.3x faster |
| 64x64 | 512x512 | 4096 | 9.31 s | 11.43 s | **0.8x — slower** |

Reproduce with:

```bash
cargo run --release -p sd-cli --example backend_bench -- 32
cargo run --release -p sd-cli --features metal --example backend_bench -- 32
```

Metal works — conv2d, group norm, attention and upsampling all execute on the
GPU, and it is 4-5x faster than CPU. Then at production resolution it falls off
a cliff and loses to the CPU.

**The cliff is our own code, not candle's.**
`ops::scaled_dot_product_attention` materialises the full `seq x seq` score
matrix. At a 64x64 latent, `seq = 4096`, so that is a 4096x4096 f32 matrix —
**67 MB, allocated per attention call**, before softmax. The scaling is
unmistakable: quadrupling `seq` (32->64) makes CPU 4.7x slower and Metal 25x
slower, because the GPU is the one starved for memory bandwidth.

Two caveats on the numbers: the first Metal call took 36 s including lazy
shader compilation, so always warm up; and an M4 Max has unusually fast CPU
matmul, so the CPU column flatters itself relative to a typical x86 machine.

### What to do about it

Chunked or flash attention, computing the score matrix in tiles instead of
materialising it. This is:

- entirely inside `sd-tensor` — **no model code changes**
- independently verifiable against the existing golden tests
- the difference between GPU being useless and being 4-5x at real resolutions

It is the single highest-value optimisation currently available, and it is the
exact case the seam was designed for. Tracked in [roadmap.md](roadmap.md).

Also note `--features accelerate` on Apple silicon: BLAS-accelerated CPU
matmul that adds **no native compilation** (it links a system framework). Worth
enabling for CPU builds regardless of the GPU story — see
[native-deps.md](native-deps.md).

## When to revisit

Switch to burn if **all three** hold:

1. Milestone 1 renders a correct image on candle first. Never migrate away from
   something that does not work yet.
2. GPU performance is the measured bottleneck *after* attention is fixed — the
   table above shows the current bottleneck is ours, not candle's.
3. GGUF has landed in burn, or we have decided to own quantisation anyway.

Until then the honest position is: candle has the worse governance and the
better feature coverage, and feature coverage is what ships images.
