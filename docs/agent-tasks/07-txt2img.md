# Task 07 — txt2img pipeline

**Difficulty:** medium · **Depends on:** Tasks 01–06, all merged and green ·
**Read `AGENTS.md` first.**

Tie everything together into a working `sdrs txt2img`. **Do not start this
until every prior task is merged and `cargo test --workspace` is green.**

---

## Files you may modify

```
crates/stable-diffusion-rs/src/pipeline/mod.rs      (new)
crates/stable-diffusion-rs/src/pipeline/txt2img.rs  (new)
crates/stable-diffusion-rs/src/lib.rs               (add: pub mod pipeline;)
crates/sd-cli/src/main.rs                           (add the txt2img subcommand)
crates/stable-diffusion-rs/tests/pipeline_smoke.rs  (new)
```

## Files you must NOT modify

```
crates/sd-models/tests/api_contract.rs   <-- the API contract; never edit
crates/sd-models/**        <-- everything you need is already public
crates/sd-sample/**
crates/sd-tensor/**
crates/sd-loader/**
any existing test file
any Cargo.toml
```

If something you need is not public, stop and report it. Do not widen a
visibility modifier in another crate.

---

## What to implement

```rust
pub struct Txt2ImgConfig {
    pub prompt: String,
    pub negative_prompt: String,   // "" by default
    pub width: usize,              // 512
    pub height: usize,             // 512
    pub steps: usize,              // 20
    pub cfg_scale: f64,            // 7.5
    pub seed: u64,
    pub sampler: SamplerKind,      // EulerAncestral | DpmPlusPlus2M
}

pub struct Txt2ImgPipeline { /* tokenizer, text encoder, unet, vae, schedule */ }

impl Txt2ImgPipeline {
    pub fn load(
        model_dir: &Path,
        device: &Device,
    ) -> Result<Self, PipelineError>;

    /// Returns [1, 3, height, width] in [-1, 1].
    pub fn run(&self, cfg: &Txt2ImgConfig) -> Result<Tensor, PipelineError>;
}
```

`load` expects the standard diffusers layout:

```
model_dir/
  tokenizer/tokenizer.json
  text_encoder/model.safetensors
  unet/diffusion_pytorch_model.safetensors
  vae/diffusion_pytorch_model.safetensors
```

Report a clear error naming the missing file if any is absent.

## The sampling loop

```
1.  cond   = text_encoder(tokenizer.encode(prompt))           [1, 77, 768]
2.  uncond = text_encoder(tokenizer.encode(negative_prompt))  [1, 77, 768]
3.  context = cat([uncond, cond], dim=0)                      [2, 77, 768]
        ^^ UNCOND FIRST. Order must match the chunk in step 7.

4.  sigmas = sigmas_for_steps(&schedule, steps)               [steps + 1]
5.  latent = randn(seed, [1, 4, height/8, width/8]) * sigmas[0]

6.  for i in 0..steps:
        sigma = sigmas[i]
        // classifier-free guidance: batch both conditionings
        latent_in = cat([latent, latent], dim=0)              [2, 4, h/8, w/8]
        latent_in = latent_in / sqrt(sigma^2 + 1)

        t = sigma_to_timestep(sigma)                          see below
        out = unet(latent_in, t, context)                     [2, 4, h/8, w/8]

7.      let (out_uncond, out_cond) = split out along dim 0
        noise_pred = out_uncond + cfg_scale * (out_cond - out_uncond)

        denoised = latent - sigma * noise_pred
        latent = sampler.step(latent, denoised, sigma, sigmas[i+1], noise)

8.  image = vae.decode(latent)     // applies scaling_factor internally
    return image
```

### `sigma_to_timestep`

The UNet takes a discrete timestep, not a sigma. Map back by finding the
nearest training sigma:

```rust
/// Nearest index in `schedule.sigmas()` to `sigma`, as an f32 timestep.
fn sigma_to_timestep(schedule: &Schedule, sigma: f64) -> f64;
```

Linear search over 1000 entries is fine — it runs 20 times per image.

### Seeding

Use `sd_tensor::rng::SeededRng`. It already exists and is already tested.

```rust
use sd_tensor::rng::SeededRng;

let mut rng = SeededRng::new(cfg.seed);
let latent = rng.randn((1, 4, cfg.height / 8, cfg.width / 8), device)?;
// ... and inside the loop, for the ancestral noise:
let noise = rng.randn(latent.dims(), device)?;
```

Create the `SeededRng` **once per image**, before the loop, and draw from it in
order — initial latent first, then one noise draw per step. Creating a new one
inside the loop makes every step use identical noise.

Three things you must not do:

- **Do not use `Device::set_seed`.** It returns an error on CPU.
- **Do not add `rand` to any `Cargo.toml`.** It is not reachable and not needed.
- **Do not try to match PyTorch's `randn`.** Explicitly out of scope. The
  requirement is that *our* output is reproducible, not that it matches torch.

## The CLI

Add to `crates/sd-cli/src/main.rs`, following the existing `Decode` subcommand:

```
sdrs txt2img \
  --model ./models/sd15 \
  --prompt "a rusty crab on a beach" \
  --negative-prompt "" \
  --steps 20 --cfg-scale 7.5 --seed 42 \
  --width 512 --height 512 \
  --sampler dpmpp2m \
  -o out.png
```

Log per-step progress with `tracing::info!` so a 20-step run on CPU does not
look hung. It will take minutes.

---

## The test

This task has **no golden reference** — end-to-end output depends on our own
RNG. Test the properties that must hold instead:

```rust
// No weights needed:
#[test] fn config_defaults_are_sane()
#[test] fn sigma_to_timestep_is_monotonic()
#[test] fn sigma_to_timestep_maps_max_sigma_near_999()
#[test] fn same_seed_gives_identical_latents()
#[test] fn different_seeds_give_different_latents()

// Skips unless SD_TEST_MODEL_DIR is set:
#[test] fn end_to_end_produces_finite_image_in_range()
```

The last one asserts the output is `[1, 3, 512, 512]`, contains no `NaN` or
`inf`, and has values within `[-1.5, 1.5]`. It does **not** assert the image
looks like anything — that is a human check.

After it passes, generate a real image and look at it. A test suite cannot tell
you the picture is a crab.

---

## Verification

```bash
cargo test --workspace
./scripts/check-seam.sh
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check

# with weights present:
SD_TEST_MODEL_DIR=./models/sd15 cargo test -p stable-diffusion-rs -- --nocapture

cargo run --release -p sd-cli -- txt2img \
  --model ./models/sd15 --prompt "a rusty crab on a beach" \
  --steps 20 --seed 42 -o out.png
```

## Definition of done

- [ ] All six tests pass
- [ ] `out.png` exists and **visibly depicts the prompt** — attach it
- [ ] Same seed twice produces byte-identical PNGs
- [ ] No existing test file modified
- [ ] No `Cargo.toml` changed

---

## Known traps

**Uncond first in the batch.** `cat([uncond, cond])` and the chunk in step 7
must agree. Reversing both is also correct; reversing one inverts guidance and
produces images that are the *opposite* of the prompt — a distinctive and
confusing symptom.

**Scale the initial latent by `sigmas[0]`.** Starting from unit-variance noise
gives washed-out output.

**`latent_in / sqrt(sigma^2 + 1)` before the UNet.** This is the k-diffusion
input scaling. Omitting it gives noisy, oversaturated results.

**`denoised = latent - sigma * noise_pred`.** The UNet predicts noise, not x0.
The sampler needs x0.

**The VAE's `decode` already divides by `scaling_factor`.** Do not divide
again. Use `decode`, not `decode_raw`.

**Reset the DPM++ solver between images.** State carried over corrupts the
second image.

**Height and width must be multiples of 8**, since latents are 1/8 scale.
Reject other values with a clear error rather than producing a shape panic.

---

## If you get blocked

Report `BLOCKED` with the stage: tokenizer, text encoder, UNet, sampler, or
VAE. Then dump the latent's min/max/mean at each step — a latent going to
`NaN` or exploding in magnitude localizes the bug faster than anything else.

Common signatures:
- pure noise output → sigma ordering or missing initial scaling
- inverted/opposite image → uncond/cond order
- washed out → missing `sqrt(sigma^2+1)` scaling
- `NaN` at step 1 → `sigma_to_timestep` returning an out-of-range index

Do not modify tests.
