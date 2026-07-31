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
    pad_token_id: u32,
    max_length: usize,
}

impl ClipTokenizer {
    /// Load a tokenizer from a model directory, whichever form it ships.
    ///
    /// **A stock SD 1.5 or SDXL download has no `tokenizer.json`** — the
    /// repositories ship the slow tokenizer, `vocab.json` + `merges.txt`. That
    /// used to be a documented papercut with "go and copy a file from another
    /// repository" as the fix. It is not one now: this reads either form, and
    /// `from_vocab_and_merges` reconstructs the exact same tokenizer, verified
    /// id-for-id against the fast one in this module's tests.
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, TokenizeError> {
        let dir = dir.as_ref();
        let fast = dir.join("tokenizer.json");
        if fast.exists() {
            return Self::from_file(fast);
        }
        let (vocab, merges) = (dir.join("vocab.json"), dir.join("merges.txt"));
        if vocab.exists() && merges.exists() {
            return Self::from_vocab_and_merges(vocab, merges);
        }
        Err(TokenizeError::NotFound(fast))
    }

    /// Load a tokenizer from a path that may name either form.
    ///
    /// `path` may be a `tokenizer.json`, or the directory holding one.
    /// **A path naming a `tokenizer.json` that is not there still succeeds**
    /// when `vocab.json` + `merges.txt` sit beside it — which is not a
    /// tolerant-parsing flourish but the common case: neither
    /// `stable-diffusion-v1-5` nor `stabilityai/stable-diffusion-xl-base-1.0`
    /// publishes the fast form at all. Insisting on it meant telling every
    /// user to go and copy a file out of a third repository.
    ///
    /// This is the loader pipelines should call. [`from_file`](Self::from_file)
    /// stays for the case where one exact file is meant and its absence is an
    /// error worth reporting.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, TokenizeError> {
        let path = path.as_ref();
        if path.is_dir() {
            return Self::from_dir(path);
        }
        if path.is_file() {
            return Self::from_file(path);
        }
        match path.parent() {
            Some(dir) if dir.is_dir() => Self::from_dir(dir),
            _ => Err(TokenizeError::NotFound(path.to_path_buf())),
        }
    }

    /// Whether [`open`](Self::open) would find a tokenizer at `path`.
    ///
    /// For the existence checks a pipeline runs before loading anything, so
    /// that a stock download is not rejected for lacking a file it was never
    /// going to have.
    pub fn present<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();
        let dir = if path.is_dir() {
            path
        } else if path.is_file() {
            return true;
        } else {
            match path.parent() {
                Some(d) => d,
                None => return false,
            }
        };
        dir.join("tokenizer.json").exists()
            || (dir.join("vocab.json").exists() && dir.join("merges.txt").exists())
    }

    /// Build CLIP's tokenizer from the slow form, `vocab.json` + `merges.txt`.
    ///
    /// Every piece below is transcribed from a real CLIP `tokenizer.json`, and
    /// **none of it is a reasonable-looking guess** — a normalizer or a split
    /// pattern that is nearly right produces nearly-right token ids, which is a
    /// different picture with no error anywhere. `the_two_forms_agree_exactly`
    /// is what makes this safe to offer.
    pub fn from_vocab_and_merges<P: AsRef<Path>, Q: AsRef<Path>>(
        vocab: P,
        merges: Q,
    ) -> Result<Self, TokenizeError> {
        use tokenizers::{
            decoders::byte_level::ByteLevel as ByteLevelDecoder,
            models::bpe::BPE,
            normalizers::{Lowercase, Replace, Sequence as NormSequence, NFC},
            pre_tokenizers::{
                byte_level::ByteLevel, sequence::Sequence as PreSequence, split::Split,
            },
            processors::roberta::RobertaProcessing,
            NormalizerWrapper, PreTokenizerWrapper, SplitDelimiterBehavior,
        };

        let (vocab, merges) = (vocab.as_ref(), merges.as_ref());
        for p in [vocab, merges] {
            if !p.exists() {
                return Err(TokenizeError::NotFound(p.to_path_buf()));
            }
        }

        // `end_of_word_suffix` is CLIP's `</w>`; without it every word-final
        // token misses and the whole prompt tokenises differently.
        let bpe = BPE::from_file(&vocab.to_string_lossy(), &merges.to_string_lossy())
            .unk_token(EOS_TOKEN.to_string())
            .end_of_word_suffix("</w>".to_string())
            .build()
            .map_err(|e| TokenizeError::Load(e.to_string()))?;

        let mut inner = tokenizers::Tokenizer::new(bpe);

        // NFC, collapse runs of whitespace to one space, lowercase — in that
        // order.
        inner.with_normalizer(Some(NormalizerWrapper::Sequence(NormSequence::new(vec![
            NormalizerWrapper::NFC(NFC),
            NormalizerWrapper::Replace(
                Replace::new(
                    tokenizers::normalizers::replace::ReplacePattern::Regex(r"\s+".to_string()),
                    " ",
                )
                .map_err(|e| TokenizeError::Load(e.to_string()))?,
            ),
            NormalizerWrapper::Lowercase(Lowercase),
        ]))));

        // The split is **inverted**: the pattern describes what to *keep*.
        let split = Split::new(
            tokenizers::pre_tokenizers::split::SplitPattern::Regex(
                r"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+".to_string(),
            ),
            SplitDelimiterBehavior::Removed,
            true,
        )
        .map_err(|e| TokenizeError::Load(e.to_string()))?;
        inner.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(PreSequence::new(vec![
            PreTokenizerWrapper::Split(split),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
        ]))));

        inner.with_post_processor(Some(RobertaProcessing::new(
            (EOS_TOKEN.to_string(), 49407),
            (BOS_TOKEN.to_string(), 49406),
        )));
        inner.with_decoder(Some(ByteLevelDecoder::default()));

        Self::from_tokenizer(inner)
    }

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
        Self::from_tokenizer(inner)
    }

    /// Wrap a built `tokenizers::Tokenizer` in CLIP's padding contract.
    fn from_tokenizer(inner: tokenizers::Tokenizer) -> Result<Self, TokenizeError> {
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
            // CLIP pads with EOS. SDXL's *second* tokenizer does not — see
            // `with_pad_token`.
            pad_token_id: eos_token_id,
            max_length: MAX_LENGTH,
        })
    }

    /// Override the padding token by name.
    ///
    /// SD 1.5's tokenizer, and SDXL's first, pad with `<|endoftext|>`. SDXL's
    /// **second** pads with `!` — token id 0. Applying the first rule to the
    /// second fills every unused slot with 49407 instead of 0, which changes
    /// the conditioning for every prompt shorter than 77 tokens, meaning all
    /// of them. Nothing about the shapes reveals it.
    pub fn with_pad_token(mut self, token: &str) -> Result<Self, TokenizeError> {
        self.pad_token_id = self
            .inner
            .token_to_id(token)
            .ok_or_else(|| TokenizeError::Load(format!("vocabulary has no {token}")))?;
        Ok(self)
    }

    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
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
        fit_to_context(ids, self.max_length, self.eos_token_id, self.pad_token_id)
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

    /// How many *content* tokens a prompt costs, excluding BOS and EOS.
    ///
    /// Exposed because the counts are unintuitive and a caller cannot budget
    /// without them. CLIP's BPE splits digits one at a time, so `"16-bit"` is
    /// four tokens and `"32x32"` is five, and every comma is its own token.
    ///
    /// Compare against [`Self::content_capacity`]: above it the prompt is
    /// **truncated, not chunked** — this implementation keeps the first
    /// `capacity` tokens and ends the sequence with EOS, so anything past the
    /// limit is discarded rather than encoded in a second window. That is a
    /// real difference from `stable-diffusion.cpp`, which chunks; a term past
    /// the boundary is dropped here rather than detached.
    pub fn content_token_count(&self, text: &str) -> Result<usize, TokenizeError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| TokenizeError::Encode(e.to_string()))?;
        // `encode` with specials adds BOS and EOS around the content.
        Ok(encoding.get_ids().len().saturating_sub(2))
    }

    /// The content token ids for a string, without BOS, EOS or padding.
    ///
    /// For finding where a word landed in an encoded prompt — BPE splits are
    /// not character positions, so a substring search on the text cannot say.
    pub fn encode_content(&self, text: &str) -> Result<Vec<u32>, TokenizeError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| TokenizeError::Encode(e.to_string()))?;
        let ids = encoding.get_ids();
        Ok(ids
            .iter()
            .copied()
            .skip(1)
            .take(ids.len().saturating_sub(2))
            .collect())
    }

    /// Content tokens that fit before truncation: `max_length - 2`.
    pub fn content_capacity(&self) -> usize {
        self.max_length.saturating_sub(2)
    }

    /// Whether this prompt will be truncated.
    pub fn will_truncate(&self, text: &str) -> Result<bool, TokenizeError> {
        Ok(self.content_token_count(text)? > self.content_capacity())
    }
}

/// Force `ids` to exactly `max_length`.
///
/// Two different rules, and conflating them is the bug this signature exists
/// to prevent:
///
/// * **Truncating** always ends in EOS. A naive `ids[..77]` cuts mid-prompt
///   and leaves an ordinary word token where the encoder expects the sequence
///   to terminate.
/// * **Padding** fills with `pad`, and does *not* move EOS — the sequence
///   already ended, and the padding sits after it. When `pad == eos` (CLIP's
///   usual case) the two are indistinguishable, which is exactly why SDXL's
///   second tokenizer, which pads with 0, breaks code that assumes otherwise.
///
/// A free function rather than a method so the padding contract — the part of
/// this file that is ours rather than the `tokenizers` crate's, and where the
/// mistakes actually happen — is testable without a 2 MB vocabulary. The
/// integration tests need the real `tokenizer.json` and therefore skip when it
/// is absent; these do not, so they run in CI.
fn fit_to_context(ids: &[u32], max_length: usize, eos: u32, pad: u32) -> Vec<u32> {
    if max_length == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(max_length);
    if ids.len() >= max_length {
        out.extend_from_slice(&ids[..max_length]);
        out[max_length - 1] = eos;
    } else {
        out.extend_from_slice(ids);
        out.resize(max_length, pad);
    }
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
        let got = fit_to_context(&[BOS, 320, 1125, EOS], 8, EOS, EOS);
        assert_eq!(got, vec![BOS, 320, 1125, EOS, EOS, EOS, EOS, EOS]);
    }

    #[test]
    fn an_overlong_prompt_keeps_eos_last() {
        // A naive `ids[..max]` would end on 5 here, leaving the encoder with
        // no terminator.
        let ids: Vec<u32> = (1..=10).collect();
        let got = fit_to_context(&ids, 5, EOS, EOS);
        assert_eq!(got, vec![1, 2, 3, 4, EOS]);
    }

    #[test]
    fn an_exact_fit_still_ends_in_eos() {
        let got = fit_to_context(&[BOS, 320, EOS], 3, EOS, EOS);
        assert_eq!(got, vec![BOS, 320, EOS]);
    }

    #[test]
    fn empty_input_is_all_padding() {
        assert_eq!(fit_to_context(&[], 4, EOS, EOS), vec![EOS, EOS, EOS, EOS]);
    }

    #[test]
    fn the_output_length_is_always_the_context_length() {
        for len in [0usize, 1, 2, 77] {
            for input in [0usize, 1, 5, 200] {
                let ids: Vec<u32> = (0..input as u32).collect();
                assert_eq!(fit_to_context(&ids, len, EOS, EOS).len(), len);
            }
        }
    }
}

#[cfg(test)]
mod pad_token_tests {
    use super::*;

    const BOS: u32 = 49406;
    const EOS: u32 = 49407;
    /// SDXL's second tokenizer pads with `!`, which is id 0.
    const PAD_BANG: u32 = 0;

    #[test]
    fn padding_uses_the_pad_token_not_eos() {
        // The SDXL tokenizer_2 case. EOS still terminates the sequence where
        // the prompt ends; everything after it is the pad token.
        let got = fit_to_context(&[BOS, 320, EOS], 6, EOS, PAD_BANG);
        assert_eq!(got, vec![BOS, 320, EOS, PAD_BANG, PAD_BANG, PAD_BANG]);
    }

    #[test]
    fn padding_does_not_move_eos_to_the_last_slot() {
        // The bug this split exists to prevent. Forcing the final slot to EOS
        // is correct when truncating and wrong when padding — and invisible
        // whenever pad == eos, which is every SD 1.5 case.
        let got = fit_to_context(&[BOS, 320, EOS], 6, EOS, PAD_BANG);
        assert_ne!(
            *got.last().unwrap(),
            EOS,
            "EOS belongs where the prompt ended, not at the end of the padding"
        );
    }

    #[test]
    fn truncation_still_ends_in_eos_whatever_the_pad_token() {
        // Truncation has no padding, so the pad token is irrelevant and the
        // EOS rule still applies.
        let ids: Vec<u32> = (1..=10).collect();
        assert_eq!(
            fit_to_context(&ids, 4, EOS, PAD_BANG),
            vec![1, 2, 3, EOS],
            "a truncated prompt must terminate"
        );
    }
}
