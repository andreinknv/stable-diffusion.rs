//! Translating a llama.cpp-layout T5 encoder GGUF to `transformers` names.
//!
//! The T5 encoders published for Flux (city96's, and anything ComfyUI writes)
//! follow llama.cpp's naming — `enc.blk.0.attn_q.weight` — rather than
//! HuggingFace's `encoder.block.0.layer.0.SelfAttention.q.weight`. The model in
//! `sd-models` is written against the HuggingFace names because that is what
//! the golden test verifies it with, so the translation happens here.
//!
//! Two mappings are worth flagging because they are guessable and wrong:
//!
//! - `ffn_gate` is `wi_0` and `ffn_up` is `wi_1`, not the reverse. In both
//!   conventions the *gate* is the projection that gets the activation, so
//!   swapping them applies GELU to the linear branch. That is a plausible
//!   network which produces plausible text embeddings, so nothing fails —
//!   the image is simply worse.
//! - `attn_rel_b` belongs to block 0 only, matching where transformers keeps
//!   the relative attention bias. Every other block shares it.

use std::collections::HashMap;

use sd_tensor::{DType, Device, Tensor, VarBuilder};

use crate::gguf::GgufInfo;
use crate::LoadError;

/// The HuggingFace name for a llama.cpp T5 tensor name, if we recognise it.
///
/// Returns `None` for anything unrecognised — decoder blocks in particular,
/// since Flux uses the encoder alone and a full T5 GGUF carries both.
pub fn t5_key(key: &str) -> Option<String> {
    if key == "token_embd.weight" {
        // transformers ties the encoder's embedding to `shared`.
        return Some("shared.weight".to_string());
    }
    if key == "enc.output_norm.weight" {
        return Some("encoder.final_layer_norm.weight".to_string());
    }

    let rest = key.strip_prefix("enc.blk.")?;
    let (index, leaf) = rest.split_once('.')?;
    index.parse::<usize>().ok()?;

    let mapped = match leaf {
        // Self-attention lives under layer.0.
        "attn_q.weight" => "layer.0.SelfAttention.q.weight",
        "attn_k.weight" => "layer.0.SelfAttention.k.weight",
        "attn_v.weight" => "layer.0.SelfAttention.v.weight",
        "attn_o.weight" => "layer.0.SelfAttention.o.weight",
        "attn_rel_b.weight" => "layer.0.SelfAttention.relative_attention_bias.weight",
        "attn_norm.weight" => "layer.0.layer_norm.weight",
        // Feed-forward under layer.1. `gate` takes the activation, so it is
        // wi_0; `up` is the linear branch, wi_1.
        "ffn_gate.weight" => "layer.1.DenseReluDense.wi_0.weight",
        "ffn_up.weight" => "layer.1.DenseReluDense.wi_1.weight",
        "ffn_down.weight" => "layer.1.DenseReluDense.wo.weight",
        "ffn_norm.weight" => "layer.1.layer_norm.weight",
        _ => return None,
    };
    Some(format!("encoder.block.{index}.{mapped}"))
}

/// Load a T5 encoder from a llama.cpp-layout GGUF.
pub fn t5_var_builder_from_gguf<'a>(
    path: impl AsRef<std::path::Path>,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'a>, LoadError> {
    let info = GgufInfo::open(&path)?;

    // Size the guard on what will actually be expanded. A T5-XXL encoder is
    // 4.7B parameters — 18.8 GB at f32 — so this is the check that decides
    // whether the machine survives, not a formality.
    let params: u64 = info
        .tensors
        .iter()
        .filter(|(k, _)| t5_key(k).is_some())
        .map(|(_, (shape, _))| shape.iter().map(|&d| d as u64).product::<u64>())
        .sum();
    if params == 0 {
        return Err(LoadError::Unsupported {
            path: info.path.clone(),
            reason: "no T5 encoder tensors found; expected llama.cpp names like \
                     `enc.blk.0.attn_q.weight`"
                .to_string(),
        });
    }
    sd_tensor::sysmem::check_headroom(
        params.saturating_mul(dtype.size_in_bytes() as u64),
        &format!("dequantising the T5 encoder from {}", info.path.display()),
    )?;

    let mut file = std::fs::File::open(&info.path).map_err(|e| LoadError::Unsupported {
        path: info.path.clone(),
        reason: format!("cannot open: {e}"),
    })?;
    crate::gguf::preflight(&mut file, &info.path)?;
    let content = candle_content(&mut file)?;

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for name in content.tensor_infos.keys() {
        let Some(mapped) = t5_key(name) else { continue };
        let t = content
            .tensor(&mut file, name, device)?
            .dequantize(device)?
            .to_dtype(dtype)?;
        tensors.insert(mapped, t);
    }

    tracing::debug!(tensors = tensors.len(), "loaded T5 encoder from gguf");
    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}

fn candle_content(file: &mut std::fs::File) -> Result<sd_tensor::gguf::Content, LoadError> {
    Ok(sd_tensor::gguf::Content::read(file)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_the_shared_and_final_tensors() {
        assert_eq!(t5_key("token_embd.weight").unwrap(), "shared.weight");
        assert_eq!(
            t5_key("enc.output_norm.weight").unwrap(),
            "encoder.final_layer_norm.weight"
        );
    }

    #[test]
    fn maps_attention_and_feed_forward() {
        assert_eq!(
            t5_key("enc.blk.7.attn_q.weight").unwrap(),
            "encoder.block.7.layer.0.SelfAttention.q.weight"
        );
        assert_eq!(
            t5_key("enc.blk.0.attn_rel_b.weight").unwrap(),
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        );
        assert_eq!(
            t5_key("enc.blk.23.ffn_norm.weight").unwrap(),
            "encoder.block.23.layer.1.layer_norm.weight"
        );
    }

    /// The mapping that has no shape consequence and so cannot fail loudly.
    #[test]
    fn gate_is_wi_0_and_up_is_wi_1() {
        assert_eq!(
            t5_key("enc.blk.3.ffn_gate.weight").unwrap(),
            "encoder.block.3.layer.1.DenseReluDense.wi_0.weight",
            "the gated branch takes the activation, so it is wi_0"
        );
        assert_eq!(
            t5_key("enc.blk.3.ffn_up.weight").unwrap(),
            "encoder.block.3.layer.1.DenseReluDense.wi_1.weight"
        );
        assert_eq!(
            t5_key("enc.blk.3.ffn_down.weight").unwrap(),
            "encoder.block.3.layer.1.DenseReluDense.wo.weight"
        );
    }

    #[test]
    fn ignores_the_decoder_and_anything_unrecognised() {
        // A full T5 GGUF carries both towers; Flux wants only the encoder.
        assert!(t5_key("dec.blk.0.attn_q.weight").is_none());
        assert!(t5_key("dec.output_norm.weight").is_none());
        assert!(t5_key("enc.blk.notanumber.attn_q.weight").is_none());
        assert!(t5_key("enc.blk.0.something_else.weight").is_none());
        assert!(t5_key("").is_none());
    }
}
