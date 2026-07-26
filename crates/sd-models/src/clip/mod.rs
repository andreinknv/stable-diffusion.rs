//! CLIP text encoder and tokenizer.

mod text_encoder;
mod tokenizer;

pub use text_encoder::{ClipTextConfig, ClipTextEncoder};
pub use tokenizer::{ClipTokenizer, TokenizeError};
