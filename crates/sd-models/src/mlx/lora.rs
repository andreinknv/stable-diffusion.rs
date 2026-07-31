//! LoRA adapters merged into an MLX weight map.
//!
//! ```text
//!   W' = W + multiplier * (alpha / rank) * (up @ down)
//! ```
//!
//! Merged rather than applied at runtime, for the reason `sd-loader`'s LoRA
//! module gives: the correction goes into the tensor map before any model is
//! built, so every architecture here gets it without knowing LoRA exists. The
//! trade is that a merged adapter cannot change strength without reloading.
//!
//! # The naming problem, and why this does not guess
//!
//! Published LoRAs use kohya's flattened names — the diffusers path with every
//! `.` replaced by `_`. **That replacement is not invertible**: `to_out.0`
//! flattens to `to_out_0`, and `to_q` already contains an underscore, so
//! splitting on `_` cannot recover the original. Any rule that tries is right
//! for most layers and wrong for some.
//!
//! So the mapping is built from the *model* side: take each weight name the
//! checkpoint actually has, flatten it the same way, and look the result up.
//! That direction is exact by construction, and it makes a mismatch loud —
//! [`Applied::unmatched`] names every LoRA entry that found no home, which is
//! the check that matters, because a plausible-but-wrong mapping loads without
//! error and quietly produces a worse image.

use std::collections::HashMap;

use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{Error, Result};

use super::Weights;

/// Prefix kohya gives every UNet entry.
const UNET_PREFIX: &str = "lora_unet_";

/// kohya's flattening. The forward direction is exact; the reverse is not, and
/// `sd-loader`'s `flattening_matches_kohya_and_is_not_invertible` records why
/// nobody should try it.
fn flatten(path: &str) -> String {
    path.replace('.', "_")
}

/// Drop kohya's `lora_unet_` prefix if present.
///
/// Text-encoder entries carry `lora_te_` instead and keep it, so they cannot
/// match a UNet weight — which is correct: merging a text-encoder correction
/// into the UNet would be worse than ignoring it.
fn strip_prefix(stem: &str) -> &str {
    stem.strip_prefix(UNET_PREFIX).unwrap_or(stem)
}

/// One layer's low-rank correction.
struct Entry {
    down: Array,
    up: Array,
    /// `alpha / rank`, already divided.
    scale: f32,
}

/// What a merge did.
#[derive(Debug, Default)]
pub struct Applied {
    pub merged: usize,
    /// Every LoRA entry that found no weight to merge into. **Not a warning**:
    /// a partially-applied adapter is for a different architecture, or the
    /// mapping missed, and the result is a plausible image that is not the one
    /// the adapter describes.
    pub unmatched: Vec<String>,
}

/// A parsed adapter.
pub struct Lora {
    entries: HashMap<String, Entry>,
}

impl Lora {
    /// Parse `lora_down`/`lora_up`/`alpha` triples out of a loaded map.
    pub fn from_weights(raw: &Weights, s: &Stream) -> Result<Self> {
        let mut downs: HashMap<String, &Array> = HashMap::new();
        let mut ups: HashMap<String, &Array> = HashMap::new();
        let mut alphas: HashMap<String, f32> = HashMap::new();

        for (name, tensor) in raw {
            let Some((stem, kind)) = name.rsplit_once('.') else {
                continue;
            };
            // `lora_down.weight` and `lora_up.weight` split twice.
            let (stem, kind) = match kind {
                "weight" => match stem.rsplit_once('.') {
                    Some((s2, k2)) => (s2, k2),
                    None => continue,
                },
                _ => (stem, kind),
            };
            let key = strip_prefix(stem).to_string();
            match kind {
                "lora_down" => {
                    downs.insert(key, tensor);
                }
                "lora_up" => {
                    ups.insert(key, tensor);
                }
                "alpha" => {
                    // Published adapters are commonly f16, alpha included.
                    let v = tensor.to_f32(s)?.to_vec_f32(s)?;
                    if let Some(a) = v.first() {
                        alphas.insert(key, *a);
                    }
                }
                _ => {}
            }
        }

        let mut entries = HashMap::new();
        for (key, down) in downs {
            let Some(up) = ups.get(&key) else {
                return Err(Error::Msg(format!(
                    "mlx: {key} has a lora_down with no lora_up"
                )));
            };
            let rank = *down
                .shape()
                .first()
                .ok_or_else(|| Error::Msg(format!("mlx: {key}'s lora_down has no rank axis")))?
                as f32;
            // **A missing alpha means "no rescaling", which is alpha == rank,
            // not alpha == 0.** Getting that default wrong scales every
            // correction to nothing and the adapter silently does nothing.
            let alpha = alphas.get(&key).copied().unwrap_or(rank);
            entries.insert(
                key,
                Entry {
                    // f32 for the product, as `sd-loader` does: an f16 adapter
                    // merged into f32 weights would otherwise compute the
                    // correction at half precision.
                    down: down.to_f32(s)?,
                    up: up.to_f32(s)?,
                    scale: alpha / rank,
                },
            );
        }
        Ok(Self { entries })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Merge into `weights` in place.
    ///
    /// `multiplier` is the adapter's strength; **0 leaves every weight
    /// untouched**, bit for bit, rather than merely almost untouched.
    pub fn merge_into(
        &self,
        weights: &mut Weights,
        multiplier: f32,
        s: &Stream,
    ) -> Result<Applied> {
        let mut applied = Applied::default();
        let mut used: Vec<String> = Vec::new();

        let names: Vec<String> = weights.keys().cloned().collect();
        for name in names {
            let Some(stem) = name.strip_suffix(".weight") else {
                continue;
            };
            let flat = flatten(stem);
            let Some(entry) = self.entries.get(&flat) else {
                continue;
            };
            used.push(flat);
            if multiplier != 0.0 {
                let w = weights.get(&name).expect("just listed");
                let updated = merged(w, entry, multiplier, s)?;
                weights.insert(name, updated);
            }
            applied.merged += 1;
        }

        applied.unmatched = self
            .entries
            .keys()
            .filter(|k| !used.contains(k))
            .cloned()
            .collect();
        applied.unmatched.sort();
        Ok(applied)
    }
}

/// `W + multiplier * (alpha/rank) * (up @ down)`, shaped like `W`.
///
/// The reshape is what makes one expression cover linear layers, 1x1
/// convolutions and 3x3 convolutions together: `down` is `[rank, ...]` and `up`
/// is `[out, rank, ...]`, so flattening everything after the first axis turns
/// all three into the same matrix product.
fn merged(weight: &Array, entry: &Entry, multiplier: f32, s: &Stream) -> Result<Array> {
    let shape = weight.shape();
    let rank = entry.down.shape()[0];
    let out = entry.up.shape()[0];

    let down = entry
        .down
        .reshape(&[rank, entry.down.elem_count() / rank], s)?;
    // `up` may carry trailing 1-sized convolution axes; after the reshape it is
    // `[out, rank]` either way.
    let up = entry.up.reshape(&[out, entry.up.elem_count() / out], s)?;

    let delta = up
        .matmul(&down, s)?
        .mul(&Array::scalar_f32(entry.scale * multiplier)?, s)?
        .reshape(&shape, s)?;
    weight.add(&delta, s)
}
