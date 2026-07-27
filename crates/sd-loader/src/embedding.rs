//! Textual inversion: a learned token, loaded from a file.
//!
//! A few kilobytes against a checkpoint's gigabytes, which is the whole point
//! — it is the cheapest way for a user to bring a style, by three orders of
//! magnitude.
//!
//! # It has no token id, so it is spliced rather than looked up
//!
//! A learned embedding is not in CLIP's vocabulary. There is no id to encode
//! and nothing to look up. What happens instead: the trigger word is tokenised
//! like any other word, `n` of the resulting positions are reserved, and after
//! the embedding lookup those rows are *overwritten* with the learned vectors.
//!
//! That is why [`crate::embedding`] needs `ClipTextEncoder::embed_tokens` and
//! `forward_embeds` to be separate — and why the splice happens before
//! position embeddings are added, since a learned vector occupies a position
//! in the sequence like any other token.
//!
//! # Two file layouts, both in the wild
//!
//! The original textual-inversion release nests the tensor under
//! `string_to_param`; later tools write `emb_params` at the top level, and
//! some write a bare single-tensor file. All three are accepted, because a
//! user downloading an embedding has no reason to know which tool made it.

use std::collections::HashMap;

use sd_tensor::{Device, Tensor};

use crate::LoadError;

/// A learned embedding and the word that triggers it.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// The trigger, as written in a prompt.
    pub name: String,
    /// `[vectors, width]`. Most embeddings are 1-8 vectors.
    pub vectors: Tensor,
}

impl Embedding {
    /// How many token positions this embedding occupies in a prompt.
    pub fn len(&self) -> usize {
        self.vectors.dim(0).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Width of each vector — 768 for SD 1.5, 1024 for SD 2.x.
    ///
    /// Checked against the encoder before splicing: an SDXL embedding in an
    /// SD 1.5 prompt is the common mistake and would otherwise be a shape
    /// error from deep inside the transformer.
    pub fn width(&self) -> usize {
        self.vectors.dim(1).unwrap_or(0)
    }

    /// Load from a `.safetensors`, taking the trigger from the file stem.
    ///
    /// The stem, because that is what every tool that writes these uses as the
    /// prompt word and what a user will have been told to type.
    pub fn load(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("embedding")
            .to_string();
        let tensors = sd_tensor::safetensors::load(path, device)?;
        let vectors = pick(&tensors).ok_or_else(|| LoadError::Unsupported {
            path: path.to_path_buf(),
            reason: format!(
                "no embedding tensor found; expected `emb_params`, `string_to_param.*` \
                 or a single tensor, got {:?}",
                tensors.keys().collect::<Vec<_>>()
            ),
        })?;
        // A `[width]` file is one vector written without its leading axis.
        let vectors = if vectors.rank() == 1 {
            vectors.unsqueeze(0)?
        } else {
            vectors
        };
        Ok(Self { name, vectors })
    }
}

/// Find the embedding tensor across the layouts in circulation.
fn pick(tensors: &HashMap<String, Tensor>) -> Option<Tensor> {
    if let Some(t) = tensors.get("emb_params") {
        return Some(t.clone());
    }
    if let Some(t) = tensors.get("string_to_param.*") {
        return Some(t.clone());
    }
    // A bare single-tensor file, whatever it is called.
    if tensors.len() == 1 {
        return tensors.values().next().cloned();
    }
    // Otherwise the first 2D tensor, which is what a stray metadata entry
    // alongside the real one leaves.
    tensors.values().find(|t| t.rank() == 2).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sd_tensor::DType;

    fn map(pairs: &[(&str, Tensor)]) -> HashMap<String, Tensor> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn every_layout_in_circulation_is_recognised() {
        let dev = Device::Cpu;
        let t = Tensor::zeros((4, 768), DType::F32, &dev).unwrap();

        assert!(pick(&map(&[("emb_params", t.clone())])).is_some());
        assert!(pick(&map(&[("string_to_param.*", t.clone())])).is_some());
        assert!(pick(&map(&[("whatever", t.clone())])).is_some());
        // A metadata scalar alongside the real tensor: the 2D one wins.
        let step = Tensor::zeros(1, DType::F32, &dev).unwrap();
        let found = pick(&map(&[("step", step), ("emb", t)])).unwrap();
        assert_eq!(found.rank(), 2);
    }

    #[test]
    fn nothing_usable_is_reported_rather_than_guessed() {
        let dev = Device::Cpu;
        let a = Tensor::zeros(3, DType::F32, &dev).unwrap();
        let b = Tensor::zeros(5, DType::F32, &dev).unwrap();
        // Two 1D tensors and no 2D: there is no defensible choice.
        assert!(pick(&map(&[("a", a), ("b", b)])).is_none());
    }
}
