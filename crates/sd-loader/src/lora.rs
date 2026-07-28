//! LoRA adapters, merged into a model's weights at load.
//!
//! A LoRA supplies a low-rank correction per layer — `down: [rank, in]` and
//! `up: [out, rank]` — and the adapted weight is
//!
//! ```text
//!   W' = W + multiplier * (alpha / rank) * (up @ down)
//! ```
//!
//! Merging rather than applying at runtime, because it costs nothing per step
//! and needs no model changes: the correction goes into the tensor map before
//! any model is built, so every architecture that loads from safetensors gets
//! it without knowing LoRA exists. The trade is that a merged adapter cannot
//! have its strength changed without reloading, and that a *quantised* base
//! cannot be merged into without dequantising and requantising — which is
//! lossy, and is why this is the dense path only. Runtime application is the
//! answer for quantised models and is not implemented here.
//!
//! # The naming problem, and why this does not guess
//!
//! Published LoRAs use kohya's flattened names:
//!
//! ```text
//!   lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight
//! ```
//!
//! which is the diffusers path `down_blocks.0.attentions.0.transformer_blocks
//! .0.attn1.to_q` with every `.` replaced by `_`. **That replacement is not
//! invertible.** `to_out.0` flattens to `to_out_0`, and `to_q` already
//! contains an underscore, so splitting on `_` cannot recover the original.
//! Any rule that tries will be right for most layers and wrong for some.
//!
//! So the mapping is built from the *model* side instead: take each weight
//! name the checkpoint actually has, apply the same flattening, and look the
//! result up. That direction is exact by construction. It also makes a
//! mismatch loud — [`Applied::unmatched`] names every LoRA entry that found no
//! home, which is the check that matters, because a plausible-but-wrong
//! mapping loads without error and quietly produces a worse image.

use std::collections::HashMap;

use sd_tensor::{DType, Device, Tensor};

use crate::LoadError;

/// Prefix kohya gives every UNet entry.
const UNET_PREFIX: &str = "lora_unet_";

/// One layer's low-rank correction.
#[derive(Debug)]
pub struct Entry {
    down: Tensor,
    up: Tensor,
    /// `alpha / rank`, already divided.
    ///
    /// Folded here because the two are only ever used together, and because a
    /// missing `alpha` means "no rescaling" — which is `alpha = rank`, not
    /// `alpha = 0`. Getting that default wrong scales every correction to
    /// nothing and looks exactly like a LoRA that does not work.
    scale: f64,
}

/// A parsed LoRA adapter, keyed by flattened layer name.
#[derive(Debug, Default)]
pub struct Lora {
    entries: HashMap<String, Entry>,
}

/// What [`Lora::merge_into`] did, for the caller to check rather than assume.
#[derive(Debug, Default)]
pub struct Applied {
    /// Layers whose weights were adapted.
    pub merged: usize,
    /// LoRA entries that matched no weight in the model.
    ///
    /// **Not empty means the adapter and the model disagree**, whether because
    /// it targets a different architecture or because the flattening missed.
    /// Either way the result is a partially-applied adapter, which produces a
    /// plausible image and is not the one the adapter describes.
    pub unmatched: Vec<String>,
}

impl Lora {
    /// Read a LoRA from a safetensors file.
    ///
    /// Entries whose `down`/`up` pair is incomplete are refused rather than
    /// skipped: half a correction is not a smaller correction, it is a
    /// different one.
    pub fn load(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let tensors = sd_tensor::safetensors::load(path, device)?;

        let mut downs: HashMap<String, Tensor> = HashMap::new();
        let mut ups: HashMap<String, Tensor> = HashMap::new();
        let mut alphas: HashMap<String, f64> = HashMap::new();

        for (name, tensor) in tensors {
            let Some((stem, kind)) = name.rsplit_once('.') else {
                continue;
            };
            match kind {
                "alpha" => {
                    let v = tensor
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    if let Some(a) = v.first() {
                        alphas.insert(strip_prefix(stem).to_string(), *a as f64);
                    }
                }
                "weight" => {
                    let Some((stem, which)) = stem.rsplit_once('.') else {
                        continue;
                    };
                    let stem = strip_prefix(stem).to_string();
                    match which {
                        "lora_down" => {
                            downs.insert(stem, tensor);
                        }
                        "lora_up" => {
                            ups.insert(stem, tensor);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        let mut entries = HashMap::with_capacity(downs.len());
        for (stem, down) in downs {
            let Some(up) = ups.remove(&stem) else {
                return Err(LoadError::Unsupported {
                    path: path.to_path_buf(),
                    reason: format!("{stem} has a lora_down with no lora_up"),
                });
            };
            // Rank is the down matrix's leading axis in every published
            // layout, linear and convolutional alike.
            let rank = down.dim(0)? as f64;
            // No alpha means no rescaling, i.e. alpha == rank.
            let alpha = alphas.get(&stem).copied().unwrap_or(rank);
            entries.insert(
                stem,
                Entry {
                    down,
                    up,
                    scale: if rank > 0.0 { alpha / rank } else { 0.0 },
                },
            );
        }
        if !ups.is_empty() {
            let mut orphans: Vec<_> = ups.into_keys().collect();
            orphans.sort();
            return Err(LoadError::Unsupported {
                path: path.to_path_buf(),
                reason: format!(
                    "{} lora_up entries have no lora_up partner, first: {}",
                    orphans.len(),
                    orphans[0]
                ),
            });
        }
        Ok(Self { entries })
    }

    /// The correction for a model path, if this adapter has one.
    ///
    /// Returns `(down, up, scale)` — the factors, *not* their product. That is
    /// the point of the runtime path: `up @ down` is a full-size dense matrix,
    /// and never forming it is what lets a LoRA apply to a **quantised** base
    /// without dequantising it.
    pub fn delta_for(&self, path: &str) -> Option<(&Tensor, &Tensor, f64)> {
        let entry = self.entries.get(&flatten(path))?;
        Some((&entry.down, &entry.up, entry.scale))
    }

    /// How many layers this adapter corrects.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Merge into a checkpoint's tensors, in place.
    ///
    /// `multiplier` is the user-facing strength; 0 leaves the weights exactly
    /// as they were, 1 applies the adapter as published.
    ///
    /// Walks the *model's* names and flattens each one to find its LoRA entry,
    /// for the reason given in the module documentation — the flattening is
    /// one-way, so it can only be applied, never undone.
    pub fn merge_into(
        &self,
        weights: &mut HashMap<String, Tensor>,
        multiplier: f64,
    ) -> Result<Applied, LoadError> {
        let mut applied = Applied::default();
        let mut used: HashMap<&str, ()> = HashMap::new();

        for (name, weight) in weights.iter_mut() {
            let Some(stem) = name.strip_suffix(".weight") else {
                continue;
            };
            let flat = flatten(stem);
            let Some(entry) = self.entries.get(&flat) else {
                continue;
            };
            used.insert(entry_key(&self.entries, &flat), ());
            if multiplier != 0.0 {
                *weight = merged(weight, entry, multiplier)?;
            }
            applied.merged += 1;
        }

        applied.unmatched = self
            .entries
            .keys()
            .filter(|k| !used.contains_key(k.as_str()))
            .cloned()
            .collect();
        applied.unmatched.sort();
        Ok(applied)
    }
}

/// Borrow the stored key so `used` can hold a `&str` into `entries`.
fn entry_key<'a>(entries: &'a HashMap<String, Entry>, flat: &str) -> &'a str {
    entries
        .get_key_value(flat)
        .map(|(k, _)| k.as_str())
        .unwrap_or("")
}

/// `down_blocks.0.attn1.to_q` -> `down_blocks_0_attn1_to_q`.
fn flatten(path: &str) -> String {
    path.replace('.', "_")
}

/// Drop kohya's `lora_unet_` prefix if present.
///
/// Text-encoder entries carry `lora_te_` instead and are simply left with
/// their prefix, so they will not match a UNet weight — which is correct:
/// merging a text-encoder correction into the UNet would be worse than
/// ignoring it, and [`Applied::unmatched`] reports them.
fn strip_prefix(stem: &str) -> &str {
    stem.strip_prefix(UNET_PREFIX).unwrap_or(stem)
}

/// `W + multiplier * (alpha/rank) * (up @ down)`, shaped like `W`.
///
/// The reshape is what makes one expression cover linear layers, 1x1
/// convolutions and 3x3 convolutions together: `down` is `[rank, ...]` and
/// `up` is `[out, rank, ...]`, so flattening everything after the first axis
/// turns all three into the same matrix product.
fn merged(weight: &Tensor, entry: &Entry, multiplier: f64) -> Result<Tensor, LoadError> {
    let dtype = weight.dtype();
    let rank = entry.down.dim(0)?;
    let out = entry.up.dim(0)?;

    let down = entry
        .down
        .to_dtype(DType::F32)?
        .reshape((rank, entry.down.elem_count() / rank))?;
    let up = entry
        .up
        .to_dtype(DType::F32)?
        .reshape((out, entry.up.elem_count() / out))?;
    // `up` may carry trailing 1-sized convolution axes; after the reshape it
    // is `[out, rank]` either way.
    let delta = up.matmul(&down)?;
    let delta = (delta * (entry.scale * multiplier))?.reshape(weight.shape())?;
    Ok((weight.to_dtype(DType::F32)? + delta)?.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattening_matches_kohya_and_is_not_invertible() {
        // The forward direction is exact, which is why the mapping is built
        // this way round. The reverse is genuinely ambiguous and this records
        // why nobody should try it: two different paths flatten to one name.
        assert_eq!(
            flatten("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"),
            "down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q"
        );
        assert_eq!(flatten("attn1.to_out.0"), "attn1_to_out_0");
        // `to_out.0` and a hypothetical `to_out_0` are indistinguishable once
        // flattened — the collision is real, not theoretical.
        assert_eq!(flatten("attn1.to_out_0"), flatten("attn1.to_out.0"));
    }

    #[test]
    fn the_unet_prefix_is_dropped_and_others_are_kept() {
        assert_eq!(strip_prefix("lora_unet_down_blocks_0"), "down_blocks_0");
        // Text-encoder entries keep theirs, so they cannot accidentally match
        // a UNet weight of the same shape.
        assert_eq!(strip_prefix("lora_te_text_model_x"), "lora_te_text_model_x");
    }
}
