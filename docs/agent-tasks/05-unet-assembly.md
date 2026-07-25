# Task 05 — UNet assembly

**Difficulty:** high · **Depends on:** Tasks 03 and 04 · **Read `AGENTS.md` first.**

Wire the resnets and transformers into the full `UNet2DConditionModel`. No new
math — this is plumbing. The skip connections are where it goes wrong.

---

## Files you may modify

```
crates/sd-models/src/unet/blocks.rs   (new)
crates/sd-models/src/unet/model.rs    (new)
crates/sd-models/src/unet/mod.rs      (add the new exports)
xtask/golden/dump_reference.py        (add `unet_full` subcommand)
crates/sd-models/tests/golden_unet.rs (new)
```

## Files you must NOT modify

```
crates/sd-models/tests/golden_vae.rs
crates/sd-models/tests/golden_unet_blocks.rs
crates/sd-models/tests/golden_unet_attention.rs
crates/sd-models/src/unet/resnet.rs
crates/sd-models/src/unet/attention.rs
crates/sd-models/src/unet/embeddings.rs
crates/sd-models/src/vae/**
crates/sd-tensor/**
any Cargo.toml
```

You are **composing** Tasks 03 and 04, not editing them. If you believe one has
a bug, stop and report it — do not fix it here.

---

## Config — SD 1.5

```rust
pub struct UNetConfig {
    pub in_channels: usize,          // 4
    pub out_channels: usize,         // 4
    pub block_out_channels: Vec<usize>,  // [320, 640, 1280, 1280]
    pub layers_per_block: usize,     // 2
    pub attention_head_dim: usize,   // 8
    pub cross_attention_dim: usize,  // 768
    pub norm_num_groups: usize,      // 32
    pub norm_eps: f64,               // 1e-5
}
```

**`attention_head_dim: 8` means 8 *heads*, not head dim 8.** This field is
misnamed in the SD 1.5 config and diffusers reads it as the head count. So at
320 channels: `heads = 8`, `dim_head = 320 / 8 = 40`.

## Block layout

```
down_blocks.0   CrossAttnDownBlock2D   320 -> 320    downsample
down_blocks.1   CrossAttnDownBlock2D   320 -> 640    downsample
down_blocks.2   CrossAttnDownBlock2D   640 -> 1280   downsample
down_blocks.3   DownBlock2D            1280 -> 1280  NO downsample, NO attention

mid_block       UNetMidBlock2DCrossAttn  1280

up_blocks.0     UpBlock2D              1280 -> 1280  NO attention   upsample
up_blocks.1     CrossAttnUpBlock2D     1280 -> 1280  upsample
up_blocks.2     CrossAttnUpBlock2D     1280 -> 640   upsample
up_blocks.3     CrossAttnUpBlock2D     640 -> 320    NO upsample
```

Down blocks have `layers_per_block` (2) resnets. **Up blocks have
`layers_per_block + 1` (3).** That asymmetry is real and required.

## Parameter names

```
conv_in.weight  conv_in.bias                Conv2d(4, 320, 3, padding=1)
time_embedding.linear_1.*  linear_2.*       from Task 03

down_blocks.{i}.resnets.{j}.*               ResnetBlock2D
down_blocks.{i}.attentions.{j}.*            Transformer2DModel  (blocks 0-2 only)
down_blocks.{i}.downsamplers.0.conv.*       Conv2d(c, c, 3, stride=2, padding=1)

mid_block.resnets.0.*                       ResnetBlock2D
mid_block.attentions.0.*                    Transformer2DModel
mid_block.resnets.1.*                       ResnetBlock2D

up_blocks.{i}.resnets.{j}.*                 ResnetBlock2D
up_blocks.{i}.attentions.{j}.*              Transformer2DModel  (blocks 1-3 only)
up_blocks.{i}.upsamplers.0.conv.*           Conv2d(c, c, 3, padding=1)

conv_norm_out.weight  conv_norm_out.bias    GroupNorm(32, 320, eps=1e-5)
conv_out.weight  conv_out.bias              Conv2d(320, 4, 3, padding=1)
```

## The skip connections — read this twice

This is the part that breaks. Get it exactly right.

**Down pass.** Push onto a stack in this order:

```
h = conv_in(sample)
skips = vec![h.clone()]                 <-- conv_in output goes on FIRST

for each down_block i:
    for j in 0..layers_per_block:
        h = resnets[j](h, temb)
        if has_attention: h = attentions[j](h, context)
        skips.push(h.clone())           <-- after EVERY resnet(+attn) pair
    if has_downsampler:
        h = downsamplers[0](h)
        skips.push(h.clone())           <-- downsampler output too
```

For SD 1.5 this yields **12 entries**. Assert that.

**Up pass.** Pop from the end, concatenate along the channel axis:

```
for each up_block i:
    for j in 0..(layers_per_block + 1):
        skip = skips.pop().unwrap()
        h = Tensor::cat(&[&h, &skip], 1)?    <-- h FIRST, skip SECOND, dim 1
        h = resnets[j](h, temb)
        if has_attention: h = attentions[j](h, context)
    if has_upsampler:
        h = upsample_nearest2d(2x) then upsamplers[0].conv(h)
```

The concatenation order matters: `[h, skip]`, never `[skip, h]`. The channel
count works out either way and the numbers are wrong.

Each up resnet's `in_channels` is therefore `prev_channels + skip_channels`,
which is why up-block resnets have larger inputs than you might expect. Compute
these from the actual skip shapes rather than hardcoding.

## Forward

```rust
impl UNet2DConditionModel {
    pub fn new(cfg: &UNetConfig, vb: VarBuilder) -> Result<Self>;

    /// `sample`:    [b, 4, h, w]
    /// `timestep`:  [b]  (f32)
    /// `context`:   [b, 77, 768]
    /// returns      [b, 4, h, w]
    pub fn forward(&self, sample: &Tensor, timestep: &Tensor, context: &Tensor)
        -> Result<Tensor>;
}
```

```
temb = timestep_embedding(timestep, block_out_channels[0])   // 320
temb = time_embedding(temb)                                  // [b, 1280]
h    = conv_in(sample)
... down, mid, up as above ...
h    = conv_norm_out(h)
h    = silu(h)
h    = conv_out(h)
```

---

## Reference data

`unet_full` subcommand, SD 1.5 UNet, seed 0:

```
sample        [1, 4, 32, 32]
timestep      [1]              value 500.0
context       [1, 77, 768]
down_0 .. down_11              every skip tensor, in push order
mid_output    [1, 1280, 4, 4]
output        [1, 4, 32, 32]
```

Dump the skip stack. When the final output is wrong, comparing skips tells you
immediately whether the down pass or the up pass is at fault — otherwise you
are bisecting 25 blocks blind.

Weights: point the test at the real
`unet/diffusion_pytorch_model.safetensors`; do not re-save it.

---

## The test

```rust
// No data needed:
#[test] fn config_sd15_has_four_blocks_and_768_cross_dim()
#[test] fn skip_stack_has_twelve_entries()
#[test] fn output_shape_matches_input_shape()

// Skips if data absent:
#[test] fn down_pass_skips_match_diffusers()   // compare all 12, in order
#[test] fn mid_block_matches_diffusers()
#[test] fn full_unet_matches_diffusers()
```

`down_pass_skips_match_diffusers` must print a `Closeness` line for each of the
12 and assert on the first failure, naming the index.

---

## Verification

```bash
python3 xtask/golden/dump_reference.py unet_full --output tests/golden

cargo test -p sd-models --test golden_unet -- --nocapture
./scripts/check-seam.sh
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --workspace
```

Tolerance is `atol = 1e-3` for the full UNet only — 25 blocks of accumulated
f32 reordering genuinely exceeds `1e-4`. Every individual sub-test stays at
`1e-4`. Do not loosen anything beyond this.

## Definition of done

- [ ] All six tests pass
- [ ] All 12 skip tensors match individually
- [ ] No test file other than the new one is touched
- [ ] `resnet.rs` and `attention.rs` unchanged
- [ ] No `Cargo.toml` changed

---

## Known traps

**Up blocks have `layers_per_block + 1` resnets.** Down blocks have
`layers_per_block`. Using 2 everywhere fails weight loading on
`up_blocks.0.resnets.2`.

**`conv_in`'s output is the first skip.** Forgetting it shifts the entire stack
by one, and every up block then concatenates the wrong tensor — at shapes that
happen to be valid for several of them.

**Downsampler outputs are pushed too**, after the resnet outputs for that block.

**`down_blocks.3` has no attention and no downsampler.**
**`up_blocks.0` has no attention. `up_blocks.3` has no upsampler.**

**Concatenate `[h, skip]` along dim 1.** Not `[skip, h]`.

**`attention_head_dim: 8` is the head *count*.** `dim_head = channels / 8`,
which varies per block: 40, 80, 160, 160.

**Upsample is nearest-2x then conv**, same as the VAE. Not a transposed conv.

**The timestep must be `[b]`, not a scalar.** Broadcasting a scalar through
`time_embedding` produces `[1, 1280]` where `[b, 1280]` is needed, and it only
shows up as a shape error deep inside a resnet.

---

## If you get blocked

Report `BLOCKED` with the first failing skip index. Index 0 means `conv_in`;
1–3 means down block 0; a failure only at index 11 with everything before it
green means the down pass is fine and the bug is in the up pass or the mid
block.

Do not modify tests. Do not edit Task 03 or 04 output.
