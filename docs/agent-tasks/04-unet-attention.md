# Task 04 — UNet attention blocks

**Difficulty:** high · **Depends on:** Task 03 · **Read `AGENTS.md` first.**

The spatial transformer that injects text conditioning into the UNet. This is
the hardest task in the set. Work slowly and check each sub-block.

---

## Files you may modify

```
crates/sd-models/src/unet/attention.rs   (new)
crates/sd-models/src/unet/mod.rs         (add the new exports)
xtask/golden/dump_reference.py           (add `unet_attention` subcommand)
crates/sd-models/tests/golden_unet_attention.rs  (new)
```

## Files you must NOT modify

```
crates/sd-models/tests/golden_vae.rs
crates/sd-models/tests/golden_unet_blocks.rs
crates/sd-models/src/vae/**
crates/sd-models/src/unet/resnet.rs
crates/sd-models/src/unet/embeddings.rs
crates/sd-tensor/**
any Cargo.toml
```

---

## Three types, built bottom-up. Implement and test in this order.

### 1. `Attention` — multi-head, self *or* cross

```rust
pub struct Attention { /* your fields */ }

impl Attention {
    /// `cross_dim = None` means self-attention (k and v come from `xs`).
    /// `cross_dim = Some(768)` means cross-attention over the text context.
    pub fn new(
        query_dim: usize,
        cross_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        vb: VarBuilder,
    ) -> Result<Self>;

    /// `xs`: [b, seq_q, query_dim]
    /// `context`: None for self-attention, else [b, seq_kv, cross_dim]
    /// returns [b, seq_q, query_dim]
    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor>;
}
```

Parameter names and **bias settings** — these differ per projection:

```
to_q.weight        Linear(query_dim, inner_dim)   NO BIAS   <-- linear_no_bias
to_k.weight        Linear(kv_dim,    inner_dim)   NO BIAS   <-- linear_no_bias
to_v.weight        Linear(kv_dim,    inner_dim)   NO BIAS   <-- linear_no_bias
to_out.0.weight    Linear(inner_dim, query_dim)   HAS BIAS  <-- linear
to_out.0.bias
```

where `inner_dim = heads * dim_head` and
`kv_dim = cross_dim.unwrap_or(query_dim)`.

Using `linear` where `linear_no_bias` is required makes weight loading fail
with "cannot find tensor to_q.bias". Read that error literally — it means the
bias should not exist.

Forward:

```
kv     = context.unwrap_or(xs)
q      = to_q(xs)     [b, sq, inner]
k      = to_k(kv)     [b, sk, inner]
v      = to_v(kv)     [b, sk, inner]

reshape each to [b, s, heads, dim_head] then transpose(1, 2).contiguous()
                                                      -> [b, heads, s, dim_head]
out = ops::scaled_dot_product_attention(&q, &k, &v)   -> [b, heads, sq, dim_head]
out.transpose(1, 2).contiguous().reshape([b, sq, inner])
out = to_out(out)
```

No mask. Do not use the masked variant here.

### 2. `FeedForward` — GEGLU

```rust
pub struct FeedForward { /* your fields */ }

impl FeedForward {
    /// `mult` is 4 for SD 1.5, so inner = dim * 4.
    pub fn new(dim: usize, mult: usize, vb: VarBuilder) -> Result<Self>;
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}
```

Parameter names — note the gaps in the numbering, they are real:

```
net.0.proj.weight   Linear(dim, inner * 2)   HAS BIAS
net.0.proj.bias
net.2.weight        Linear(inner, dim)       HAS BIAS
net.2.bias
```

`net.1` is a dropout layer with no parameters, which is why the index jumps
from 0 to 2. Do not renumber them.

GEGLU forward:

```
h = net_0_proj(xs)                       [b, s, inner*2]
split h in half along the last dim   ->  hidden [b,s,inner], gate [b,s,inner]
h = hidden * gelu(gate)                  <-- ops::gelu, the erf one
return net_2(h)                          [b, s, dim]
```

Use `narrow(D::Minus1, 0, inner)` and `narrow(D::Minus1, inner, inner)` to
split. **`hidden` is the first half, `gate` is the second.** Swapping them
produces plausible garbage.

### 3. `BasicTransformerBlock`

```rust
pub struct BasicTransformerBlock { /* your fields */ }

impl BasicTransformerBlock {
    pub fn new(
        dim: usize,
        heads: usize,
        dim_head: usize,
        cross_dim: usize,   // 768
        vb: VarBuilder,
    ) -> Result<Self>;

    pub fn forward(&self, xs: &Tensor, context: &Tensor) -> Result<Tensor>;
}
```

Parameter names:

```
norm1.weight  norm1.bias      LayerNorm(dim, eps = 1e-5)
attn1.*                       Attention(dim, None,      heads, dim_head)  self
norm2.weight  norm2.bias      LayerNorm(dim, eps = 1e-5)
attn2.*                       Attention(dim, Some(768), heads, dim_head)  cross
norm3.weight  norm3.bias      LayerNorm(dim, eps = 1e-5)
ff.*                          FeedForward(dim, 4)
```

Forward — pre-layernorm, three residuals:

```
xs = attn1(norm1(xs), None)     + xs
xs = attn2(norm2(xs), Some(ctx)) + xs
xs = ff(norm3(xs))               + xs
```

### 4. `Transformer2DModel` — the spatial wrapper

```rust
pub struct Transformer2DModel { /* your fields */ }

impl Transformer2DModel {
    pub fn new(
        channels: usize,
        heads: usize,
        dim_head: usize,
        depth: usize,       // 1 for SD 1.5
        cross_dim: usize,   // 768
        vb: VarBuilder,
    ) -> Result<Self>;

    /// `xs`: [b, channels, h, w]
    /// `context`: [b, 77, 768]
    /// returns [b, channels, h, w]
    pub fn forward(&self, xs: &Tensor, context: &Tensor) -> Result<Tensor>;
}
```

Parameter names:

```
norm.weight  norm.bias           GroupNorm(32, channels, eps = 1e-6)   <-- 1e-6 here
proj_in.weight  proj_in.bias     Conv2d(channels, inner, kernel 1)
transformer_blocks.{i}.*         BasicTransformerBlock
proj_out.weight proj_out.bias    Conv2d(inner, channels, kernel 1)
```

Forward — note the reshape uses `permute`, not a bare `reshape`:

```
residual = xs
h = norm(xs)
h = proj_in(h)                                  [b, inner, hh, ww]
h = h.permute([0, 2, 3, 1]).contiguous()        [b, hh, ww, inner]
h = h.reshape([b, hh * ww, inner])
for blk in blocks: h = blk(h, context)
h = h.reshape([b, hh, ww, inner])
h = h.permute([0, 3, 1, 2]).contiguous()        [b, inner, hh, ww]
h = proj_out(h)
return h + residual
```

---

## Reference data

Add an `unet_attention` subcommand dumping, from
`unet.down_blocks[0].attentions[0]` of SD 1.5:

```
attn_input     [2, 320, 16, 16]      random, seed 0
context        [2, 77, 768]          random, seed 1
attn_output    [2, 320, 16, 16]
block_input    [2, 256, 320]         input to transformer_blocks[0]
block_output   [2, 256, 320]         output of transformer_blocks[0]
attn1_output   [2, 256, 320]         hook on transformer_blocks[0].attn1
attn2_output   [2, 256, 320]         hook on transformer_blocks[0].attn2
ff_output      [2, 256, 320]         hook on transformer_blocks[0].ff
```

Plus the weights: `tests/golden/unet_attention/attention.safetensors`.

The sub-block captures matter here more than anywhere else. With four
independently checkable stages, a failure localizes to one of them instead of
"the transformer is wrong".

---

## The test

```rust
// No data needed:
#[test] fn attention_preserves_shape()
#[test] fn cross_attention_accepts_different_context_length()
#[test] fn feedforward_halves_the_gated_projection()
#[test] fn transformer_preserves_spatial_dims()

// Skips if data absent — test in this order, they build on each other:
#[test] fn attn1_self_attention_matches_diffusers()
#[test] fn attn2_cross_attention_matches_diffusers()
#[test] fn feedforward_matches_diffusers()
#[test] fn transformer_2d_matches_diffusers()
```

---

## Verification

```bash
python3 xtask/golden/dump_reference.py unet_attention --output tests/golden

cargo test -p sd-models --test golden_unet_attention -- --nocapture
./scripts/check-seam.sh
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --workspace
```

## Definition of done

- [ ] All eight tests pass at `atol = 1e-4`
- [ ] No test file other than the new one is touched
- [ ] `crates/sd-tensor/` untouched
- [ ] No `Cargo.toml` changed

---

## Known traps

**`to_q`/`to_k`/`to_v` have no bias; `to_out.0` does.** Use `linear_no_bias`
for the first three and `linear` for the last.

**`ff.net.2`, not `ff.net.1`.** The missing index is dropout.

**GEGLU order: hidden first, gate second.** `h * gelu(g)`, not `g * gelu(h)`.

**Three different `eps` values are in play.** `Transformer2DModel.norm` is
GroupNorm at `1e-6`. The `LayerNorm`s inside the block are `1e-5`. The resnets
from Task 03 are `1e-5`. They are genuinely inconsistent in the reference
implementation — do not unify them.

**`permute` then `reshape`, with `.contiguous()` between.** A bare
`reshape([b, h*w, c])` from `[b, c, h, w]` interleaves channels and spatial
positions wrongly. It produces the right shape and wrong numbers.

**Cross-attention `k`/`v` come from `context`, `q` from `xs`.** And the two
have different sequence lengths — 256 vs 77 in the reference. If your code
assumes they match, it is wrong.

**`depth` is 1 for SD 1.5**, so there is exactly one entry in
`transformer_blocks`. Still index it as `transformer_blocks.0`.

---

## If you get blocked

Report `BLOCKED` and say which of the four numerical tests failed first:
`attn1` → self-attention or head reshape; `attn2` → cross wiring or kv dim;
`ff` → GEGLU split order; `transformer_2d` → the permute/reshape sandwich.

Do not modify tests. Do not loosen tolerances.
