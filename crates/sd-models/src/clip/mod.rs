//! CLIP text encoder and tokenizer.

mod text_encoder;
mod tokenizer;

pub use text_encoder::{ClipActivation, ClipTextConfig, ClipTextEncoder};
pub use tokenizer::{ClipTokenizer, TokenizeError};
