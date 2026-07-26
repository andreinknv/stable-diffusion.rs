//! CLIP BPE tokenizer.
//!
//! This is a thin wrapper over the `tokenizers` crate, which loads CLIP's
//! `tokenizer.json` directly and is the same implementation HuggingFace uses —
//! so the BPE itself matches the reference by construction. What this file
//! actually owns is the *padding contract*, which is where the mistakes are:
//! CLIP wants exactly 77 ids, padded with EOS rather than zero, and an
//! overlong prompt truncated so the final id is still EOS.

use std::path::{Path, PathBuf};

/// CLIP's fixed context length. The text encoder's positional embedding is
/// this size, so it is a property of the model, not a tuneable.
const MAX_LENGTH: usize = 77;

const BOS_TOKEN: &str = "<|startoftext|>";
const EOS_TOKEN: &str = "<|endoftext|>";

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
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizeError> {
        let path = path.as_ref();
        // `Tokenizer::from_file` reports a missing file as a generic load
        // failure. Checking first turns the single most likely error — a
        // reference file that was never generated — into one that names itself.
        if !path.exists() {
            return Err(TokenizeError::NotFound(path.to_path_buf()));
        }
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| TokenizeError::Load(e.to_string()))?;

        // Read the special ids out of the vocabulary rather than hardcoding
        // 49406/49407: if a caller ever loads a tokenizer whose vocabulary
        // disagrees, that is worth failing on rather than silently padding
        // with an id that means something else.
        let bos_token_id = inner
            .token_to_id(BOS_TOKEN)
            .ok_or_else(|| TokenizeError::Load(format!("vocabulary has no {BOS_TOKEN}")))?;
        let eos_token_id = inner
            .token_to_id(EOS_TOKEN)
            .ok_or_else(|| TokenizeError::Load(format!("vocabulary has no {EOS_TOKEN}")))?;

        Ok(Self {
            inner,
            bos_token_id,
            eos_token_id,
            max_length: MAX_LENGTH,
        })
    }

    /// Encode a prompt to exactly `max_length` (77) token IDs.
    ///
    /// Output is always exactly 77 ids:
    /// `[bos, ...prompt tokens..., eos, eos, eos, ...]`
    ///
    /// Longer prompts are truncated so that the final id is still `eos`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizeError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| TokenizeError::Encode(e.to_string()))?;
        Ok(self.fit(encoding.get_ids()))
    }

    /// Encode several prompts. Every row is exactly `max_length` long.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, TokenizeError> {
        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| TokenizeError::Encode(e.to_string()))?;
        Ok(encodings.iter().map(|e| self.fit(e.get_ids())).collect())
    }

    fn fit(&self, ids: &[u32]) -> Vec<u32> {
        fit_to_context(ids, self.max_length, self.eos_token_id)
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }
}

/// Force `ids` to exactly `max_length`, padding or truncating with EOS.
///
/// Both directions end in EOS. Truncation overwrites the last slot rather than
/// just slicing, because a naive `ids[..77]` cuts mid-prompt and leaves an
/// ordinary word token where the encoder expects the sequence to terminate.
///
/// A free function rather than a method so the padding contract — the part of
/// this file that is ours rather than the `tokenizers` crate's, and where the
/// mistakes actually happen — is testable without a 2 MB vocabulary. The
/// integration tests need the real `tokenizer.json` and therefore skip when it
/// is absent; these do not, so they run in CI.
fn fit_to_context(ids: &[u32], max_length: usize, eos: u32) -> Vec<u32> {
    if max_length == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(max_length);
    out.extend_from_slice(&ids[..ids.len().min(max_length)]);
    out.resize(max_length, eos);
    out[max_length - 1] = eos;
    out
}

#[derive(Debug, thiserror::Error)]
pub enum TokenizeError {
    #[error("tokenizer file not found: {0}")]
    NotFound(PathBuf),
    #[error("failed to load tokenizer: {0}")]
    Load(String),
    #[error("failed to encode text: {0}")]
    Encode(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOS: u32 = 49406;
    const EOS: u32 = 49407;

    #[test]
    fn a_short_prompt_is_padded_with_eos_not_zero() {
        // Padding with 0 is the classic mistake: 0 is a real token id, so the
        // result looks plausible and produces subtly wrong embeddings.
        let got = fit_to_context(&[BOS, 320, 1125, EOS], 8, EOS);
        assert_eq!(got, vec![BOS, 320, 1125, EOS, EOS, EOS, EOS, EOS]);
    }

    #[test]
    fn an_overlong_prompt_keeps_eos_last() {
        // A naive `ids[..max]` would end on 5 here, leaving the encoder with
        // no terminator.
        let ids: Vec<u32> = (1..=10).collect();
        let got = fit_to_context(&ids, 5, EOS);
        assert_eq!(got, vec![1, 2, 3, 4, EOS]);
    }

    #[test]
    fn an_exact_fit_still_ends_in_eos() {
        let got = fit_to_context(&[BOS, 320, EOS], 3, EOS);
        assert_eq!(got, vec![BOS, 320, EOS]);
    }

    #[test]
    fn empty_input_is_all_padding() {
        assert_eq!(fit_to_context(&[], 4, EOS), vec![EOS, EOS, EOS, EOS]);
    }

    #[test]
    fn the_output_length_is_always_the_context_length() {
        for len in [0usize, 1, 2, 77] {
            for input in [0usize, 1, 5, 200] {
                let ids: Vec<u32> = (0..input as u32).collect();
                assert_eq!(fit_to_context(&ids, len, EOS).len(), len);
            }
        }
    }
}
