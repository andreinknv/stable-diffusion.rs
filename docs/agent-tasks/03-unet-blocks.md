# Task 03 — UNet resnet blocks and timestep embedding

**Difficulty:** medium · **Depends on:** nothing · **Read `AGENTS.md` first.**

The two building blocks the UNet is mostly made of. No attention here — that is
Task 04. This task can run in parallel with Tasks 01, 02 and 06.

---

## Files you may modify

```
crates/sd-models/src/unet/mod.rs        (new)
crates/sd-models/src/unet/embeddings.rs (new)
crates/sd-models/src/unet/resnet.rs     (new)
crates/sd-models/src/lib.rs             (add: pub mod unet;)
xtask/golden/dump_reference.py          (add `unet_blocks` subcommand)
crates/sd-models/tests/golden_unet_blocks.rs  (new)
```

## Files you must NOT modify

```
crates/sd-models/tests/api_contract.rs   <-- the API contract; never edit
crates/sd-models/tests/golden_vae.rs
crates/sd-models/src/vae/**
crates/sd-tensor/**            <-- everything you need already exists
crates/sd-loader/**
any Cargo.toml
```

---

## Part A — sinusoidal timestep embedding

A free function, no weights:

```rust
use sd_tensor::{Result, Tensor};

/// Sinusoidal timestep embedding, matching
/// `diffusers.models.embeddings.get_timestep_embedding` with
/// `flip_sin_to_cos=True`, `downscale_freq_shift=0`.
///
/// `timesteps` is `[batch]` (f32). Returns `[batch, dim]`.
pub fn timestep_embedding(timesteps: &Tensor, dim: usize) -> Result<Tensor>;
```

Exact algorithm — follow step by step:

```
half        = dim / 2
exponent    = -ln(10000) * arange(0, half) / half         [half]
emb         = exp(exponent)                                [half]
emb         = timesteps[:, None] * emb[None, :]            [batch, half]
emb         = concat([cos(emb), sin(emb)], dim=1)          [batch, dim]
```

**Order is `cos` then `sin`.** That is what `flip_sin_to_cos=True` means. The
naive `[sin, cos]` order runs fine and gives a wrong model.

## Part B — the timestep embedding MLP

```rust
/// `time_embedding.linear_1` -> silu -> `time_embedding.linear_2`
#[derive(Debug)]
pub struct TimestepEmbedding { /* your fields */ }

impl TimestepEmbedding {
    /// SD 1.5: in_dim = 320, out_dim = 1280.
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self>;

    /// `[batch, in_dim]` -> `[batch, out_dim]`
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}
```

Parameter names, exactly:

```
time_embedding.linear_1.weight   [1280, 320]
time_embedding.linear_1.bias     [1280]
time_embedding.linear_2.weight   [1280, 1280]
time_embedding.linear_2.bias     [1280]
```

`vb` is passed already rooted at `time_embedding`, so use `vb.pp("linear_1")`.

## Part C — `ResnetBlock2D` with time conditioning

This is **not** the VAE resnet. It takes a time embedding.

```rust
#[derive(Debug)]
pub struct ResnetBlock2D { /* your fields */ }

impl ResnetBlock2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        temb_channels: usize,   // 1280 for SD 1.5
        groups: usize,          // 32
        eps: f64,               // 1e-5   <-- NOT the VAE's 1e-6
        vb: VarBuilder,
    ) -> Result<Self>;

    /// `xs`: [b, in_channels, h, w]
    /// `temb`: [b, temb_channels]
    /// returns [b, out_channels, h, w]
    pub fn forward(&self, xs: &Tensor, temb: &Tensor) -> Result<Tensor>;
}
```

Parameter names:

```
norm1.weight  norm1.bias              GroupNorm(32, in_channels,  eps=1e-5)
conv1.weight  conv1.bias              Conv2d(in_channels, out_channels, 3, padding=1)
time_emb_proj.weight  time_emb_proj.bias   Linear(temb_channels, out_channels)
norm2.weight  norm2.bias              GroupNorm(32, out_channels, eps=1e-5)
conv2.weight  conv2.bias              Conv2d(out_channels, out_channels, 3, padding=1)
conv_shortcut.weight  conv_shortcut.bias   Conv2d(in, out, 1)  ONLY if in != out
```

Forward, precisely:

```
h = norm1(xs)
h = silu(h)
h = conv1(h)                                    [b, out, hh, ww]

t = silu(temb)
t = time_emb_proj(t)                            [b, out]
t = t.unsqueeze(2).unsqueeze(3)                 [b, out, 1, 1]
h = h + t                                       (broadcast over h, w)

h = norm2(h)
h = silu(h)
h = conv2(h)

shortcut = conv_shortcut(xs) if in != out else xs
return h + shortcut
```

**Note the ordering:** `silu` is applied to `temb` *before* `time_emb_proj`,
and the result is added *after* `conv1`, *before* `norm2`.

---

## Reference data

Add an `unet_blocks` subcommand to `xtask/golden/dump_reference.py`.

```python
import torch
from diffusers import UNet2DConditionModel
from diffusers.models.embeddings import get_timestep_embedding

unet = UNet2DConditionModel.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    subfolder="unet", torch_dtype=torch.float32)
unet.eval()

gen = torch.Generator().manual_seed(0)
timesteps = torch.tensor([0.0, 1.0, 500.0, 999.0])

# Part A
sin_emb = get_timestep_embedding(timesteps, 320, flip_sin_to_cos=True,
                                 downscale_freq_shift=0)

# Part B
temb = unet.time_embedding(sin_emb)

# Part C — first resnet of the first down block
blk = unet.down_blocks[0].resnets[0]
x = torch.randn(2, 320, 16, 16, generator=gen)
t2 = temb[:2]
with torch.no_grad():
    out = blk(x, t2)
```

Save to `tests/golden/unet_blocks/reference.safetensors`:

```
timesteps        [4]
sin_emb          [4, 320]
temb             [4, 1280]
resnet_input     [2, 320, 16, 16]
resnet_temb      [2, 1280]
resnet_output    [2, 320, 16, 16]
```

Also save the isolated block weights so Rust can load them, using
`safetensors.torch.save_file` over `blk.state_dict()` and
`unet.time_embedding.state_dict()`, to:

```
tests/golden/unet_blocks/resnet.safetensors
tests/golden/unet_blocks/time_embedding.safetensors
```

---

## The test

`crates/sd-models/tests/golden_unet_blocks.rs`:

```rust
// No reference data needed:
#[test] fn timestep_embedding_has_shape_batch_by_dim()
#[test] fn timestep_embedding_first_half_is_cosine()   // cos(0)=1 at t=0
#[test] fn resnet_preserves_spatial_dims()
#[test] fn resnet_changes_channel_count_when_asked()

// Skips if reference data absent:
#[test] fn timestep_embedding_matches_diffusers()
#[test] fn time_embedding_mlp_matches_diffusers()
#[test] fn resnet_block_matches_diffusers()
```

`timestep_embedding_first_half_is_cosine` is a cheap, powerful check: at
`t = 0`, `cos(0) = 1`, so the first `dim/2` entries must all be `1.0`, and the
second half `sin(0) = 0`. If you got the order backwards this test fails
immediately without any download.

---

## Verification

```bash
python3 xtask/golden/dump_reference.py unet_blocks --output tests/golden

cargo test -p sd-models --test golden_unet_blocks -- --nocapture
./scripts/check-seam.sh
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --workspace
```

## Definition of done

- [ ] All seven tests pass at `atol = 1e-4`
- [ ] `git diff --stat -- crates/sd-models/tests/golden_vae.rs` empty
- [ ] `crates/sd-tensor/` untouched
- [ ] No `Cargo.toml` changed

---

## Known traps

**`cos` before `sin`.** `flip_sin_to_cos=True`. Test 2 above catches it.

**`eps = 1e-5` in the UNet.** The VAE uses `1e-6`. Do not copy the VAE value.

**`silu(temb)` comes before `time_emb_proj`, not after.**

**The time embedding is added after `conv1`, before `norm2`.** Adding it at the
start or the end runs fine and is wrong.

**`unsqueeze` twice.** `[b, out]` must become `[b, out, 1, 1]` before adding to
`[b, out, h, w]`. Broadcasting `[b, out]` directly will either error or
broadcast along the wrong axis.

**`conv_shortcut` only exists when `in != out`.** Creating it unconditionally
makes weight loading fail for blocks that do not have it, with a confusing
"cannot find tensor" error.

**`ResnetBlock2D` here is not the VAE's `ResnetBlock`.** Do not import or
refactor the VAE one. They differ by the time embedding and by `eps`. Write a
separate type.

---

## If you get blocked

Report `BLOCKED` naming which of the three parts (A, B, C) failed, and paste
the `Closeness` line. Part A failing means the sinusoid order or the exponent;
Part B means the MLP wiring; Part C means the resnet.

Do not modify tests.
