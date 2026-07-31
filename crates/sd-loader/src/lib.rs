//! Checkpoint naming, and nothing else.
//!
//! **Every function here is pure string work.** A checkpoint's tensors arrive
//! under whatever name its exporter chose — CompVis/LDM, llama.cpp,
//! `diffusers` old or new — and the models ask for one set of names. Deciding
//! which is which is the whole job.
//!
//! It lives apart from the backend deliberately: a second copy of a name
//! mapping is exactly how two implementations come to disagree about which
//! tensor is which, and that failure loads cleanly and produces noise.

/// CompVis/LDM names to `diffusers` ones.
pub mod ldm;
/// llama.cpp's T5 names to `transformers` ones.
pub mod t5_gguf;

pub use t5_gguf::t5_key;

/// Attention parameter names, modern diffusers -> the legacy layout.
///
/// diffusers renamed the VAE's attention block at some point; checkpoints
/// published before that — including the stock SD 1.5 VAE, which is what most
/// people download — still use the old names. The tensors are identical, only
/// the keys differ, so this is a pure rename with no reshape.
///
/// Model code stays on the modern names. Conversion belongs here: see the note
/// in `sd-models/src/lib.rs`.
const LEGACY_ATTENTION_KEYS: [(&str, &str); 4] = [
    (".to_q.", ".query."),
    (".to_k.", ".key."),
    (".to_v.", ".value."),
    (".to_out.0.", ".proj_attn."),
];

/// Only appears in the legacy layout, so its presence identifies one.
///
/// Public because a loader that wants to *ask* which layout a checkpoint uses
/// needs it, and there is exactly one right answer to that question.
pub const LEGACY_SENTINEL: &str = "proj_attn";

/// Rewrite a modern attention key to its legacy equivalent.
///
/// Returns `None` when the name needs no rewriting, which is every key in a
/// modern checkpoint and most keys in a legacy one.
pub fn legacy_attention_key(name: &str) -> Option<String> {
    LEGACY_ATTENTION_KEYS
        .iter()
        .find(|(modern, _)| name.contains(modern))
        .map(|(modern, legacy)| name.replace(modern, legacy))
}

/// Rewrite a legacy attention key to its modern equivalent.
///
/// The inverse of [`legacy_attention_key`], and needed by loaders that
/// normalise a checkpoint into modern names once rather than translating at
/// every lookup. Returns `None` when the name needs no rewriting.
pub fn modern_attention_key(name: &str) -> Option<String> {
    LEGACY_ATTENTION_KEYS
        .iter()
        .find(|(_, legacy)| name.contains(legacy))
        .map(|(modern, legacy)| name.replace(legacy, modern))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rename is exact in both directions and leaves everything else
    /// alone.
    #[test]
    fn the_attention_renames_round_trip() {
        for (modern, legacy) in LEGACY_ATTENTION_KEYS {
            let m = format!("decoder.mid_block.attentions.0{modern}weight");
            let l = format!("decoder.mid_block.attentions.0{legacy}weight");
            assert_eq!(legacy_attention_key(&m).as_deref(), Some(l.as_str()));
            assert_eq!(modern_attention_key(&l).as_deref(), Some(m.as_str()));
            // And each is a no-op on the other layout's own names.
            assert_eq!(modern_attention_key(&m), None);
            assert_eq!(legacy_attention_key(&l), None);
        }
        // A GroupNorm is not an attention projection.
        assert_eq!(
            modern_attention_key("decoder.mid_block.attentions.0.group_norm.weight"),
            None
        );
    }
}
