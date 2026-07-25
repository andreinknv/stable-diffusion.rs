# Task 01 — CLIP BPE tokenizer

**Difficulty:** low · **Depends on:** nothing · **Read `AGENTS.md` first.**

Convert a text prompt into the 77 token IDs the CLIP text encoder expects.

---

## Files you may modify

```
crates/sd-models/src/clip/mod.rs          (new)
crates/sd-models/src/clip/tokenizer.rs    (new)
crates/sd-models/src/lib.rs               (add: pub mod clip;)
xtask/golden/dump_reference.py            (add the `clip_tokenizer` subcommand)
```

## Files you must NOT modify

```
crates/sd-models/tests/api_contract.rs   <-- the API contract; never edit
crates/sd-models/tests/*        <-- especially not these
crates/sd-tensor/**
crates/sd-loader/**
any Cargo.toml                  <-- the dependency you need is already added
scripts/check-seam.sh
```

---

## Do not implement BPE by hand

The `tokenizers` crate is **already a dependency of `sd-models`**. Use it. It
loads CLIP's `tokenizer.json` directly and is the same implementation
HuggingFace uses, so it matches the reference by construction.

Hand-writing byte-level BPE is roughly 150 lines of subtle code and you will get
the `</w>` word-boundary handling wrong. Do not attempt it.

---

## What to implement

Create `crates/sd-models/src/clip/tokenizer.rs` with exactly this public API:

```rust
use std::path::Path;

/// CLIP tokenizer. Wraps a HuggingFace `tokenizer.json`.
#[derive(Debug)]
pub struct ClipTokenizer {
    inner: tokenizers::Tokenizer,
    bos_token_id: u32,
    eos_token_id: u32,
    max_length: usize,
}

impl ClipTokenizer {
    /// Load from a `tokenizer.json` file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizeError>;

    /// Encode a prompt to exactly `max_length` (77) token IDs.
    ///
    /// Output is always exactly 77 ids:
    ///   [bos, ...prompt tokens..., eos, eos, eos, ...]
    ///
    /// Longer prompts are truncated so that the final id is still `eos`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizeError>;

    /// Encode several prompts. Every row is exactly `max_length` long.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, TokenizeError>;

    pub fn bos_token_id(&self) -> u32;  // 49406
    pub fn eos_token_id(&self) -> u32;  // 49407
    pub fn max_length(&self) -> usize;  // 77
}

#[derive(Debug, thiserror::Error)]
pub enum TokenizeError {
    #[error("tokenizer file not found: {0}")]
    NotFound(std::path::PathBuf),
    #[error("failed to load tokenizer: {0}")]
    Load(String),
    #[error("failed to encode text: {0}")]
    Encode(String),
}
```

And `crates/sd-models/src/clip/mod.rs`:

```rust
//! CLIP text encoder and tokenizer.

mod tokenizer;

pub use tokenizer::{ClipTokenizer, TokenizeError};
```

Then add `pub mod clip;` to `crates/sd-models/src/lib.rs`.

---

## Exact padding rules

These are not negotiable and the test checks each one:

| Rule | Value |
|---|---|
| Output length | **always exactly 77**, never more, never less |
| First token | `49406` (`<\|startoftext\|>`) |
| Token after the prompt | `49407` (`<\|endoftext\|>`) |
| Padding | `49407` repeated to fill (CLIP pads with EOS, **not** with 0) |
| Empty prompt `""` | `[49406, 49407, 49407, ..., 49407]` |
| Overlong prompt | truncate to 77 with `ids[76] == 49407` |

The truncation rule matters: after truncating, the last slot must still be EOS.

---

## Reference data

Add a `clip_tokenizer` subcommand to `xtask/golden/dump_reference.py`. Follow
the structure of the existing `vae` subcommand exactly.

It must:

1. Load `CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")`
2. Encode this exact list of prompts:

```python
PROMPTS = [
    "",
    "a photo of an astronaut riding a horse on mars",
    "a rusty crab on a beach",
    "A PHOTO OF A CAT",
    "hello, world! 123",
    "a " * 200,   # overlong, must truncate to 77
]
```

3. Encode each with `padding="max_length", max_length=77, truncation=True`
4. Write `tests/golden/clip_tokenizer/reference.json`:

```json
{
  "prompts": ["...", "..."],
  "ids": [[49406, 49407, ...], [...]],
  "bos_token_id": 49406,
  "eos_token_id": 49407,
  "max_length": 77
}
```

5. Also copy the tokenizer file to
   `tests/golden/clip_tokenizer/tokenizer.json` so the Rust test can load it.

---

## The test

Create `crates/sd-models/tests/golden_clip_tokenizer.rs`. **This is the one
test file you are allowed to create** — because it does not exist yet. Once
created you may not weaken it.

It must contain, at minimum:

```rust
#[test] fn encodes_to_exactly_77_ids()
#[test] fn empty_prompt_is_bos_then_all_eos()
#[test] fn overlong_prompt_truncates_with_eos_last()
#[test] fn matches_huggingface_reference()   // skips if reference.json absent
```

Skip-if-absent pattern — copy it from
`crates/sd-models/tests/golden_vae.rs::decoder_matches_diffusers_reference`.

Structural tests that need no reference data must still run in CI.

---

## Verification

Run every command. Paste real output.

```bash
python3 xtask/golden/dump_reference.py clip_tokenizer --output tests/golden

cargo test -p sd-models --test golden_clip_tokenizer -- --nocapture
./scripts/check-seam.sh
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

## Definition of done

- [ ] All four tests pass
- [ ] `git diff --stat -- crates/sd-models/tests/golden_vae.rs` is **empty**
- [ ] No `Cargo.toml` changed
- [ ] `clippy` clean, `fmt` clean, seam check passes
- [ ] Every prompt in `PROMPTS` matches the reference exactly, id for id

---

## Known traps

**Padding token is EOS, not zero.** CLIP pads with `49407`. Padding with `0`
produces a plausible-looking vector that yields subtly wrong embeddings later.
This is the single most common mistake here.

**`tokenizers` returns `Vec<u32>` not `Vec<i64>`.** Do not convert.

**The `tokenizers` crate errors are `Box<dyn Error>`.** Convert with
`.map_err(|e| TokenizeError::Load(e.to_string()))?`. Do not use `?` directly
and do not `unwrap()`.

**Do not lowercase the text yourself.** The tokenizer config already handles
casing. Doing it twice is harmless here but is a habit that breaks other
tokenizers.

**Truncation must keep EOS last.** `truncation=True` in HuggingFace cuts to 77
*including* placing EOS at the end. Naively taking `ids[..77]` after appending
gives a different final token.

---

## If you get blocked

Report `BLOCKED` with:

- the exact command you ran
- the exact error
- for a mismatch: the prompt, the first differing index, expected vs got

Do not modify the test. Do not loosen an assertion. Do not delete a prompt from
`PROMPTS`.
