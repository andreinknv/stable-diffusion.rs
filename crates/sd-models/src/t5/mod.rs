//! T5's tokenizer.
//!
//! The encoder itself is [`crate::mlx::t5`]. This is the sentencepiece side,
//! which touches no tensor and so is shared by every backend.

mod tokenizer;

pub use tokenizer::{T5Tokenizer, FLUX_MAX_LENGTH};
