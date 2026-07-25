# Task 02 — CLIP text encoder

**Difficulty:** medium · **Depends on:** Task 01 · **Read `AGENTS.md` first.**

Turn 77 token IDs into a `[batch, 77, 768]` conditioning tensor.

---

## Files you may modify

```
crates/sd-models/src/clip/text_encoder.rs   (new)
crates/sd-models/src/clip/mod.rs            (add the new exports)
xtask/golden/dump_reference.py              (add `clip_encoder` subcommand)
crates/sd-models/tests/golden_clip_encoder.rs  (new)
```

## Files you must NOT modify

```
crates/sd-models/tests/golden_vae.rs
crates/sd-models/tests/golden_clip_tokenizer.rs
crates/sd-models/tests/api_contract.rs
crates/sd-models/src/vae/**
crates/sd-tensor/**       <-- everything you need is already there
crates/sd-loader/**
any Cargo.toml
```

---

## Masked attention already exists — do not write it

CLIP is causal, and the seam already provides both pieces:

```rust
use sd_tensor::ops;

// [1, 1, 77, 77]; 0.0 where visible, -inf where masked.
let mask = ops::causal_mask(77, device)?;

// q, k, v are [batch, heads, seq, head_dim]
let out = ops::scaled_dot_product_attention_masked(&q, &k, &v, &mask)?;
```

Build the mask **once in `new()`** and store it on the struct. Do not rebuild
it per forward call, and do not construct the mask by hand.

---

## Configuration — SD 1.5 (`openai/clip-vit-large-patch14`)

```rust
#[derive(Debug, Clone)]
pub struct ClipTextConfig {
    pub vocab_size: usize,            // 49408
    pub hidden_size: usize,           // 768
    pub intermediate_size: usize,     // 3072
    pub num_hidden_layers: usize,     // 12
    pub num_attention_heads: usize,   // 12
    pub max_position_embeddings: usize, // 77
    pub layer_norm_eps: f64,          // 1e-5   <-- NOT 1e-6
}

impl ClipTextConfig {
    pub fn sd15() -> Self { /* the values above */ }
}
```

`head_dim = hidden_size / num_attention_heads = 64`.

---

## Reference parameter layout — copy these names exactly

```
text_model.embeddings.token_embedding.weight            [49408, 768]
text_model.embeddings.position_embedding.weight         [77, 768]

text_model.encoder.layers.{i}.layer_norm1.weight        [768]
text_model.encoder.layers.{i}.layer_norm1.bias          [768]
text_model.encoder.layers.{i}.self_attn.q_proj.weight   [768, 768]
text_model.encoder.layers.{i}.self_attn.q_proj.bias     [768]
text_model.encoder.layers.{i}.self_attn.k_proj.weight   [768, 768]
text_model.encoder.layers.{i}.self_attn.k_proj.bias     [768]
text_model.encoder.layers.{i}.self_attn.v_proj.weight   [768, 768]
text_model.encoder.layers.{i}.self_attn.v_proj.bias     [768]
text_model.encoder.layers.{i}.self_attn.out_proj.weight [768, 768]
text_model.encoder.layers.{i}.self_attn.out_proj.bias   [768]
text_model.encoder.layers.{i}.layer_norm2.weight        [768]
text_model.encoder.layers.{i}.layer_norm2.bias          [768]
text_model.encoder.layers.{i}.mlp.fc1.weight            [3072, 768]
text_model.encoder.layers.{i}.mlp.fc1.bias              [3072]
text_model.encoder.layers.{i}.mlp.fc2.weight            [768, 3072]
text_model.encoder.layers.{i}.mlp.fc2.bias              [768]

text_model.final_layer_norm.weight                      [768]
text_model.final_layer_norm.bias                        [768]
```

`i` runs `0..12`. Note it is `q_proj` here, **not** `to_q` — CLIP comes from
`transformers`, while the VAE comes from `diffusers`, and they use different
conventions. Do not "make them consistent".

---

## What to implement

```rust
use sd_tensor::{Result, Tensor, VarBuilder};

/// One transformer layer. Pre-layernorm.
#[derive(Debug)]
struct ClipEncoderLayer { /* your fields */ }

impl ClipEncoderLayer {
    fn new(cfg: &ClipTextConfig, vb: VarBuilder) -> Result<Self>;
    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor>;
}

/// The full text tower.
#[derive(Debug)]
pub struct ClipTextEncoder { /* your fields */ }

impl ClipTextEncoder {
    /// `vb` is rooted at the checkpoint root, so `text_model.*` resolves
    /// directly beneath it.
    pub fn new(cfg: &ClipTextConfig, vb: VarBuilder) -> Result<Self>;

    /// `token_ids` is `[batch, 77]` of dtype U32.
    /// Returns `[batch, 77, 768]`.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor>;
}
```

### Forward, precisely

```
1.  x = token_embedding(token_ids)                    -> [b, 77, 768]
2.  p = position_embedding(arange(0, 77))             -> [77, 768]
3.  x = x + p                                          (broadcast over batch)
4.  mask = causal mask, [1, 1, 77, 77]
5.  for layer in layers:  x = layer(x, mask)
6.  x = final_layer_norm(x)
7.  return x                                           -> [b, 77, 768]
```

### One layer, precisely — pre-layernorm, note the residuals

```
residual = x
x = layer_norm1(x)
x = self_attention(x, mask)
x = residual + x

residual = x
x = layer_norm2(x)
x = fc2(quick_gelu(fc1(x)))
x = residual + x
```

### Self-attention, precisely

```
q = q_proj(x)   k = k_proj(x)   v = v_proj(x)          each [b, 77, 768]
reshape each to [b, 77, 12, 64] then transpose(1,2)  -> [b, 12, 77, 64]
out = scaled_dot_product_attention_masked(q, k, v, mask)
transpose(1,2), reshape                              -> [b, 77, 768]
out = out_proj(out)
```

### The causal mask

Shape `[1, 1, 77, 77]`. Position `(i, j)` is `0.0` when `j <= i`, and
`f32::NEG_INFINITY` when `j > i`. Build it once in `new()` and store it; do not
rebuild per forward call.

---

## Reference data

Add a `clip_encoder` subcommand to `xtask/golden/dump_reference.py`, following
the `vae` subcommand's structure.

- Load `CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")`
- Tokenize `"a photo of an astronaut riding a horse on mars"` to 77 ids
- Register forward hooks capturing the **output of every encoder layer**, named
  `layer_00` .. `layer_11`
- Run and save to `tests/golden/clip_encoder/reference.safetensors`:

```
token_ids        [1, 77]   (int64 -> save as int64, cast in Rust)
embeddings       [1, 77, 768]   after step 3 above
layer_00         [1, 77, 768]
...
layer_11         [1, 77, 768]
last_hidden_state [1, 77, 768]   final output after final_layer_norm
```

Per-layer captures are the point. When the final output mismatches they tell
you *which layer* diverged first.

- Also copy the model weights to `tests/golden/clip_encoder/clip.safetensors`.

---

## The test

`crates/sd-models/tests/golden_clip_encoder.rs`:

```rust
#[test] fn config_sd15_has_expected_dimensions()        // no data needed
#[test] fn encoder_builds_with_random_weights()         // no data needed
#[test] fn encoder_output_shape_is_batch_77_768()       // random weights
#[test] fn matches_transformers_reference()             // skips if data absent
```

`matches_transformers_reference` must compare **each layer in order** and
report the first divergence before asserting on the final output:

```rust
for i in 0..12 {
    let name = format!("layer_{i:02}");
    if let Some(expected) = refs.get(&name) {
        let c = testing::closeness(&got_layers[i], expected)?;
        eprintln!("{name}: {c}");
    }
}
```

---

## Verification

```bash
python3 xtask/golden/dump_reference.py clip_encoder --output tests/golden

cargo test -p sd-models --test golden_clip_encoder -- --nocapture
./scripts/check-seam.sh
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --workspace
```

## Definition of done

- [ ] All four tests pass at `atol = 1e-4`
- [ ] Every `layer_NN` matches, not just the final output
- [ ] `git diff --stat -- crates/sd-models/tests/golden_vae.rs` empty
- [ ] `crates/sd-tensor/` is untouched
- [ ] No `Cargo.toml` changed

---

## Known traps

**`quick_gelu`, not `gelu`.** CLIP uses `x * sigmoid(1.702 * x)`. Using
`ops::gelu` gives output that looks reasonable and is wrong by ~1e-2. Use
`ops::quick_gelu`.

**`layer_norm_eps` is `1e-5`.** The VAE uses `1e-6`. Copying the VAE's value
produces a small uniform offset that is easy to misread as noise.

**Pre-layernorm, not post.** The norm is applied *before* attention and the
residual is added *after*. Getting this backwards still runs and still produces
77×768 output.

**`q_proj`, not `to_q`.** CLIP is a `transformers` model. The VAE is a
`diffusers` model. Different naming. Do not unify them.

**Head reshape order.** `[b, 77, 768] -> [b, 77, 12, 64] -> transpose(1,2)`.
Reshaping straight to `[b, 12, 77, 64]` interleaves the heads wrongly and
produces garbage that still has the right shape.

**`.contiguous()` after transpose.** candle requires it before `reshape` on a
transposed tensor, and the error message is not obvious.

**Positions are `0..77` always.** No offset, no padding-aware positions, even
though most tokens are EOS padding. CLIP attends over the full 77.

**SD 1.5 uses the output *after* `final_layer_norm`.** Do not skip it, and do
not apply the text projection — that is for CLIP retrieval, not for SD.

---

## If you get blocked

Report `BLOCKED` with the first `layer_NN` whose `Closeness` line exceeds
tolerance, and its `max_abs`. That single number localizes the bug better than
any description.

Do not modify tests. Do not loosen tolerances.
