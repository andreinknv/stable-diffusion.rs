//! T5's SentencePiece tokenizer.
//!
//! Like [`ClipTokenizer`](crate::clip::ClipTokenizer) this wraps the
//! `tokenizers` crate, so the segmentation matches HuggingFace by
//! construction, and what this file owns is the padding contract.
//!
//! T5's differs from CLIP's in every particular, which is why it needs its own
//! type rather than a parameter:
//!
//! - Padding is with a real pad token (id 0), **not** with EOS. T5 was trained
//!   with padding masked out, so filling with EOS would feed the model a long
//!   run of meaningful tokens.
//! - There is no BOS. Sequences end with `</s>` and begin with content.
//! - The length is a pipeline choice rather than a model property. T5 uses
//!   relative position embeddings, so unlike CLIP's hard 77 it will accept any
//!   length; Flux picks 512.

use std::path::{Path, PathBuf};

use crate::clip::TokenizeError;

/// What Flux conditions on. Not a property of T5 — it is what the Flux
/// pipeline was distilled with, and shortening it changes the output.
pub const FLUX_MAX_LENGTH: usize = 512;

const PAD_TOKEN: &str = "<pad>";
const EOS_TOKEN: &str = "</s>";

/// T5 tokenizer.
#[derive(Debug)]
pub struct T5Tokenizer {
    inner: tokenizers::Tokenizer,
    pad_token_id: u32,
    eos_token_id: u32,
    max_length: usize,
}

impl T5Tokenizer {
    /// Load from a `tokenizer.json`, padding to `max_length`.
    pub fn from_file<P: AsRef<Path>>(path: P, max_length: usize) -> Result<Self, TokenizeError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(TokenizeError::NotFound(PathBuf::from(path)));
        }
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| TokenizeError::Load(e.to_string()))?;

        // Read the ids from the vocabulary rather than hardcoding 0 and 1. A
        // tokenizer that disagrees is worth failing on: padding with an id
        // that means something else is silent and produces worse images.
        let pad_token_id = inner
            .token_to_id(PAD_TOKEN)
            .ok_or_else(|| TokenizeError::Load(format!("vocabulary has no {PAD_TOKEN}")))?;
        let eos_token_id = inner
            .token_to_id(EOS_TOKEN)
            .ok_or_else(|| TokenizeError::Load(format!("vocabulary has no {EOS_TOKEN}")))?;

        Ok(Self {
            inner,
            pad_token_id,
            eos_token_id,
            max_length,
        })
    }

    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Encode to exactly [`Self::max_length`] ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizeError> {
        let encoded = self
            .inner
            .encode(text, true)
            .map_err(|e| TokenizeError::Encode(e.to_string()))?;
        Ok(self.fit(encoded.get_ids().to_vec()))
    }

    /// Truncate or pad to the context length.
    ///
    /// Truncation keeps `</s>` last, because a sequence that simply stops mid
    /// phrase is not something T5 saw in training. Padding does *not* move it:
    /// the pad tokens follow the terminator, which is what the model expects.
    fn fit(&self, mut ids: Vec<u32>) -> Vec<u32> {
        if ids.len() > self.max_length {
            ids.truncate(self.max_length);
            if let Some(last) = ids.last_mut() {
                *last = self.eos_token_id;
            }
        }
        ids.resize(self.max_length, self.pad_token_id);
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> Option<T5Tokenizer> {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/golden/flux/t5-tokenizer.json");
        if !p.exists() {
            eprintln!("SKIP: no T5 tokenizer fixture");
            return None;
        }
        Some(T5Tokenizer::from_file(p, 32).unwrap())
    }

    #[test]
    fn pads_to_the_context_length_with_pad_not_eos() {
        let Some(t) = tokenizer() else { return };
        let ids = t.encode("a rusty crab on a beach").unwrap();
        assert_eq!(ids.len(), 32);

        let eos_at = ids.iter().position(|&i| i == t.eos_token_id()).unwrap();
        assert!(
            ids[eos_at + 1..].iter().all(|&i| i == t.pad_token_id()),
            "everything after </s> must be pad; T5 masks padding in training, \
             so filling with EOS would feed it real tokens"
        );
        assert_ne!(
            t.pad_token_id(),
            t.eos_token_id(),
            "if these coincided the check above would be vacuous"
        );
    }

    #[test]
    fn truncation_keeps_the_terminator_last() {
        let Some(t) = tokenizer() else { return };
        let long = "a rusty crab on a beach ".repeat(40);
        let ids = t.encode(&long).unwrap();
        assert_eq!(ids.len(), 32);
        assert_eq!(
            *ids.last().unwrap(),
            t.eos_token_id(),
            "a truncated prompt must still terminate"
        );
    }

    #[test]
    fn different_prompts_tokenize_differently() {
        let Some(t) = tokenizer() else { return };
        assert_ne!(
            t.encode("a crab").unwrap(),
            t.encode("a horse").unwrap(),
            "padding should not have swallowed the content"
        );
    }
}
