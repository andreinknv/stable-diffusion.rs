# Rules for AI coding agents

**Read this file completely before writing any code. It overrides your defaults.**

You are implementing one task from `docs/agent-tasks/`. You are not designing
anything. Every architectural decision has already been made. Your job is to
translate a precise specification into Rust that makes a specific test pass.

---

## The eight hard rules

### 1. Never modify a test file to make it pass

If a test fails, **the implementation is wrong**, not the test.

Editing a test, loosening a tolerance, adding `#[ignore]`, deleting an
assertion, or changing an expected value is the single worst thing you can do
here, because it destroys the only mechanism that tells us the port is correct.
A failing test is useful. A passing test that was edited to pass is worse than
no test.

If you believe a test is genuinely wrong: **stop and say so.** Do not change it.

### 2. Never import a compute backend outside `sd-tensor`

Forbidden everywhere except `crates/sd-tensor/`:

```rust
use candle_core::...;   // NO
use mlx_sys::...;       // NO — the rule is about any backend
```

Also forbidden: adding `candle-*` or `mlx-*` to any other crate's
`Cargo.toml`.

CI fails on this (`scripts/check-seam.sh`). If you need something the backend has
and `sd-tensor` doesn't expose, **add it to `crates/sd-tensor/src/lib.rs`**,
then use it from there. Your task file says whether that is expected.

### 3. Never add a dependency

Do not add anything to any `Cargo.toml` `[dependencies]` section. Everything
you need already exists. If you are sure something is missing, stop and say so.

### 4. Only touch the files your task lists

Each task file has a **"Files you may modify"** section. That list is complete.
Do not touch anything else — not to "clean up", not to fix an unrelated warning,
not to improve something you noticed.

### 5. Never invent an API

Only use functions listed in "Available API" below or already used in existing
code. If you cannot find a function that does what you need, **do not guess a
plausible name.** `sd-tensor`'s MLX surface is hand-written and covers what
the models needed, not everything MLX offers.

Check what exists:

```bash
grep -rn "pub fn" crates/sd-tensor/src/lib.rs
```

### 6. Match parameter names exactly

Weight names come from the pretrained checkpoint. `to_q` is not `q_proj`.
`conv_norm_out` is not `norm_out`. A single wrong name means the weight silently
fails to load and the test fails somewhere far away with no clue why.

Copy names character by character from your task's reference layout section.

### 7. Run the verification before saying you are done

Every task ends with exact commands. Run them. Paste the real output. Do not
claim a test passes without having run it.

### 8. Never run a size you have not costed

Attention memory here is **O(n^4)** in the latent edge, not O(n^2). `seq = n*n`,
and the score matrix is `seq x seq`. One step up a doubling sweep costs 16x:

| latent n | image | one score matrix |
|---:|---:|---:|
| 64 | 512x512 | 64 MiB |
| 128 | 1024x1024 | 1 GiB |
| 256 | 2048x2048 | 16 GiB |
| 384 | 3072x3072 | **81 GiB** |

Those figures are for *one* matrix. Peak use is at least double: scaling and
softmax each allocate another of the same size, and a mask adds a third. A
multi-head call multiplies again by `batch * heads`.

On a Metal build that matrix is **wired GPU memory**. The GPU cannot take page
faults, so the pages are pinned, unswappable, and invisible to jetsam.
Overshooting physical RAM does not fail your process — it takes the machine
down. On 2026-07-25 `backend_bench 384` on a 36 GiB Mac ended in a kernel
watchdog panic and a power-button reset. There was no error to catch and
nothing in the logs from the benchmark itself.

`ops::chunked_attention` now bounds the score matrix near 64 MiB regardless of
size, so attention itself no longer allocates the figures above. That moves the
largest allocation to a full-resolution up block: 9.0 GiB at a 384 latent. Both
are checked against the same 2 GiB budget before anything is allocated —
`ops::check_alloc_budget` in the seam, and
`DecoderConfig::peak_activation_bytes` for the decode as a whole.

**Do not raise `SD_ATTENTION_BUDGET_BYTES` to get past a refusal.** It exists
for a deliberate experiment on a machine you know can take it, not for clearing
an obstacle. If your task needs a larger latent, stop and say so.

One thing chunking does *not* do: it does not make large decodes fast — it is
the same arithmetic in tiles. What it buys is a decode that completes at all,
by keeping the largest single allocation to one tile's worth. `vae::decode_tiled`
than assuming.

The general form of this rule: before running anything parameterised by a size,
work out what that size allocates. Refusing is free; a wedged machine is not.

---

## Available API

This is the complete surface. **Nothing else exists.**

### Types — `use sd_tensor::{...}`

```
Tensor  Device  DType  Shape  Result  Error  Module  IndexOp  D  VarBuilder
```

### Layers — `use sd_tensor::nn::{...}`

```
conv2d(in_ch, out_ch, kernel_size, Conv2dConfig, vb) -> Result<Conv2d>
conv2d_no_bias(in_ch, out_ch, kernel_size, Conv2dConfig, vb) -> Result<Conv2d>
linear(in_dim, out_dim, vb) -> Result<Linear>
linear_no_bias(in_dim, out_dim, vb) -> Result<Linear>
group_norm(num_groups, num_channels, eps, vb) -> Result<GroupNorm>
layer_norm(size, LayerNormConfig, vb) -> Result<LayerNorm>
embedding(vocab_size, dim, vb) -> Result<Embedding>

Types: Conv2d Conv2dConfig Linear GroupNorm LayerNorm LayerNormConfig
       Embedding VarBuilder VarMap
```

All layers run via `.forward(&tensor)?` (requires `use sd_tensor::Module`).

### Ops — `use sd_tensor::ops::{...}`

```
silu(&Tensor)                              -> Result<Tensor>
swish(&Tensor)                             -> Result<Tensor>   // same as silu
gelu(&Tensor)                              -> Result<Tensor>   // erf-based, torch default
gelu_approx(&Tensor)                       -> Result<Tensor>   // tanh approximation
quick_gelu(&Tensor)                        -> Result<Tensor>   // CLIP: x*sigmoid(1.702x)
softmax(&Tensor, dim)                      -> Result<Tensor>
softmax_last_dim(&Tensor)                  -> Result<Tensor>
scaled_dot_product_attention(&q, &k, &v)   -> Result<Tensor>   // NO MASK SUPPORT
```

**`scaled_dot_product_attention` takes no mask.** If your task needs causal
masking, the task file tells you to add a masked variant to `sd-tensor`.

### Array methods (`sd_tensor::mlx::Array`)

Every op takes a `&Stream` as its last argument — MLX schedules onto a stream
and the handle is how a caller says which. `Stream::gpu()` is the one to use.

```
reshape(&[..], s)?   transpose(&[..], s)?   contiguous(s)?
narrow(axis, start, len, s)?   broadcast_to(&[..], s)?   take(&idx, axis, s)?
matmul(&other, s)?   add(&o, s)?   sub(&o, s)?   mul(&o, s)?   div(&o, s)?
sum(&axes, keepdims, s)?   mean(&axes, keepdims, s)?   max(&axes, keepdims, s)?
abs(s)?  exp(s)?  sqrt(s)?  rsqrt(s)?  cos(s)?  sin(s)?  log(s)?  tanh(s)?
silu(s)?  gelu(s)?  gelu_approx(s)?  quick_gelu(s)?  relu(s)?  sigmoid(s)?
maximum(&o, s)?   erf(s)?   astype(dtype, s)?   to_f32(s)?
layer_norm(w, b, eps, s)?   rms_norm(w, eps, s)?   group_norm(n, eps, w, b, s)?
sdpa(&k, &v, scale, s)?     sdpa_causal(..)?    sdpa_masked(.., &mask, s)?
conv2d(&kernel, stride, padding, dilation, groups, s)?
shape()   elem_count()   to_vec_f32(s)?
Array::from_slice_f32(&data, &shape)?   Array::from_slice_i32(..)?
Array::scalar_f32(v)?   concat(&[&a, &b], axis, s)?   eval(&[&a])?
```

**`shape()` returns `Vec<usize>`**, so destructure it:
`let [n, h, w, c] = x.shape()[..] else { return Err(...) };`

**Weights are NHWC-consuming but stored NCHW.** A `diffusers` convolution
kernel arrives as `(out, in, kh, kw)` and MLX wants `(out, kh, kw, in)`; the
`conv` helper in `sd_models::mlx` transposes at the point of use, so the maps
hold the original layout. Do not transpose them at load.

**MLX is lazy.** Nothing computes until `eval` or a read. That is usually
invisible and occasionally decisive: a graph holding references to tensors you
meant to drop keeps them alive. `quantized::from_gguf` documents the case where
it mattered.

### Random noise — `use sd_tensor::rng::SeededRng`

```
SeededRng::new(seed: u64) -> SeededRng
    .normals(n: usize) -> Vec<f32>
sd_tensor::rng::randn_nhwc(&mut rng, n, c, h, w) -> Result<Array>   // NHWC
```

**Draw order is the promise.** `randn_nhwc` fills NCHW-major then re-orders,
because that is the order the seed pins — drawing straight into NHWC would give
a different picture from the same seed.

The noise, not the image. f32 reduction order differs per backend, and twenty
UNet steps compound it: the same seed on CPU and Metal gives images that are
indistinguishable by eye but differ by a mean of 0.9/255, with only 27% of
pixels exactly equal. Reproducing a file byte-for-byte needs the same device
and build; reproducing the picture does not.

Do not add `rand` to any `Cargo.toml`. It is not reachable and you do not need
it.

### Test helpers — `use sd_tensor::testing::{...}`

```
assert_close(&got, &expected, atol, "label") -> Result<()>
closeness(&a, &b) -> Result<Closeness>          // Display shows shapes + max/mean diff
max_abs_diff(&a, &b) -> Result<f64>
DEFAULT_ATOL  // 1e-4
DEFAULT_RTOL  // 1e-3
```

### If you doubt whether something exists

`crates/sd-models/tests/api_contract.rs` exercises every API listed above. It
compiles and passes in CI, so everything in it is real, with the exact
signature shown. Read it before guessing.

```bash
cargo test -p sd-models --test api_contract
```

---

## Workflow — follow exactly, in order

1. **Read your task file completely.** All of it, before writing anything.
2. **Read the existing VAE decoder** at `crates/sd-models/src/vae/decoder.rs`.
   It is the reference for style, structure, and how `VarBuilder` paths work.
   Copy its patterns.
3. **Generate reference data** with the command in your task file.
4. **Run the test first, confirm it fails.** If it passes before you write
   anything, the task is already done or the test is broken — stop and say so.
5. **Implement.**
6. **Run the full verification block.**
7. **Report** using the template at the bottom of this file.

---

## `VarBuilder` — how weight paths work

`VarBuilder` maps Rust structure to checkpoint key names. `vb.pp("x")` appends
`x.` to the current prefix. (`pp` = "push prefix".)

```rust
let vb_blk = vb.pp("mid_block");            // "mid_block."
let vb_res = vb_blk.pp("resnets").pp("0");  // "mid_block.resnets.0."
let conv = conv3x3(512, 512, vb_res.pp("conv1"))?;
// loads: mid_block.resnets.0.conv1.weight
//        mid_block.resnets.0.conv1.bias
```

For numbered lists use `.pp(i.to_string())`.

**If a weight name is wrong, the error appears at load time and names the key
it could not find.** Read that message — it tells you the exact expected path.

---

## Debugging a numerical failure

The test prints a `Closeness` line: shapes, `max_abs`, `mean_abs`.

**Shapes differ** → structural bug. Wrong channel count, wrong transpose, wrong
reshape. Fix before looking at anything else.

**Shapes match, `max_abs` large (> 1.0)** → wrong operation or wrong axis. Most
often a `transpose`/`permute` in the wrong place.

**`max_abs` small but above tolerance (1e-4 .. 1e-1)** → wrong constant. Check,
in this order: normalization `eps`, activation variant (`gelu` vs `gelu_approx`
vs `quick_gelu`), attention scale factor.

**`max_abs` is `inf` or `NaN`** → division by zero or an uninitialized weight
that silently loaded as zeros.

Check intermediate tensors **in order**. The first one that diverges is the
bug; everything after it is carrying the error forward.

---

## Reporting template

End your work with exactly this:

```
## Task: <task file name>

### Files changed
- path/to/file.rs (new | modified)

### Verification output
$ cargo test -p <crate> --test <test> -- --nocapture
<PASTE REAL OUTPUT — not a summary, not what you expect it to say>

$ ./scripts/check-seam.sh
<PASTE REAL OUTPUT>

$ cargo clippy --workspace --all-targets -- -D warnings
<PASTE REAL OUTPUT>

### Status
DONE — all tests pass
or
BLOCKED — <exactly what failed, with the Closeness line if numerical>
```

**"BLOCKED" is an acceptable and useful answer.** Reporting a real failure is
far more valuable than claiming success. Do not fabricate output.
