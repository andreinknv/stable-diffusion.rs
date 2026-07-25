# Rules for AI coding agents

**Read this file completely before writing any code. It overrides your defaults.**

You are implementing one task from `docs/agent-tasks/`. You are not designing
anything. Every architectural decision has already been made. Your job is to
translate a precise specification into Rust that makes a specific test pass.

---

## The seven hard rules

### 1. Never modify a test file to make it pass

If a test fails, **the implementation is wrong**, not the test.

Editing a test, loosening a tolerance, adding `#[ignore]`, deleting an
assertion, or changing an expected value is the single worst thing you can do
here, because it destroys the only mechanism that tells us the port is correct.
A failing test is useful. A passing test that was edited to pass is worse than
no test.

If you believe a test is genuinely wrong: **stop and say so.** Do not change it.

### 2. Never import candle outside `sd-tensor`

Forbidden everywhere except `crates/sd-tensor/`:

```rust
use candle_core::...;   // NO
use candle_nn::...;     // NO
```

Also forbidden: adding `candle-core` or `candle-nn` to any other crate's
`Cargo.toml`.

CI fails on this (`scripts/check-seam.sh`). If you need something candle has
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
plausible name.** candle's API is not what you would expect it to be.

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

### Tensor methods (from candle, available on `Tensor`)

```
reshape(shape)?   transpose(d1, d2)?   permute(dims)?   contiguous()?
squeeze(dim)?     unsqueeze(dim)?      narrow(dim, start, len)?
matmul(&other)?   broadcast_add(&o)?   broadcast_mul(&o)?   broadcast_sub(&o)?
to_dtype(dtype)?  to_device(dev)?      flatten_all()?   flatten_from(dim)?
dims()            dims2()?  dims3()?  dims4()?   dim(i)?   rank()   elem_count()
affine(mul, add)? clamp(min, max)?     abs()?    sqr()?   sqrt()?   exp()?
neg()?  cos()?  sin()?  max(dim)?  min(dim)?  sum(dim)?  mean(dim)?
upsample_nearest2d(h, w)?    cat(&[..], dim)     get(idx)?
Tensor::zeros(shape, dtype, dev)?    Tensor::ones(...)?
Tensor::new(data, dev)?              Tensor::arange(start, end, dev)?
Tensor::from_vec(vec, shape, dev)?   Tensor::cat(&[&a, &b], dim)?
```

Arithmetic: `(&a + &b)?`, `(&a * &b)?`, `(&a - &b)?`, `(&a / &b)?`,
and with scalars: `(&a * 2.0f64)?`, `(&a + 1.0)?`.

### Test helpers — `use sd_tensor::testing::{...}`

```
assert_close(&got, &expected, atol, "label") -> Result<()>
closeness(&a, &b) -> Result<Closeness>          // Display shows shapes + max/mean diff
DEFAULT_ATOL  // 1e-4
DEFAULT_RTOL  // 1e-3
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
