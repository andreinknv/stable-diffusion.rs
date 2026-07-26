//! Weight loading for stable-diffusion.rs.
//!
//! Milestone 1 supports `.safetensors`. GGUF (the format almost every
//! quantised community model ships in) is the next target — see
//! `docs/roadmap.md`.
//!
//! # Safety note
//!
//! This crate parses files that users download from the internet. Memory
//! safety here is a feature of the project, not an incidental benefit: the
//! equivalent C++ parsers have a CVE history. Keep `unsafe` confined to the
//! mmap call below and justify any addition.

pub mod gguf;
pub mod ldm;

pub use gguf::{
    clip_var_builder_from_gguf, gguf_var_builder, unet_var_builder_from_gguf,
    vae_var_builder_from_gguf, GgufInfo, Layout,
};

use std::path::{Path, PathBuf};

use sd_tensor::{DType, Device, VarBuilder};

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("model file not found: {0}")]
    NotFound(PathBuf),
    #[error("unsupported model format for {path}: {reason}")]
    Unsupported { path: PathBuf, reason: String },
    #[error(transparent)]
    Tensor(#[from] sd_tensor::Error),
}

pub type Result<T> = std::result::Result<T, LoadError>;

/// Weight file formats we can recognise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Safetensors,
    Gguf,
    Ckpt,
}

impl Format {
    /// Detect format from the file extension.
    pub fn detect(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "safetensors" => Some(Self::Safetensors),
            "gguf" => Some(Self::Gguf),
            "ckpt" | "pt" | "pth" | "bin" => Some(Self::Ckpt),
            _ => None,
        }
    }
}

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
const LEGACY_SENTINEL: &str = "proj_attn";

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

/// Whether any file names a tensor using the legacy attention layout.
///
/// Reads only the safetensors headers — no tensor data is touched, so this is
/// cheap even against a multi-gigabyte UNet.
fn uses_legacy_attention_names(paths: &[PathBuf]) -> Result<bool> {
    for path in paths {
        // SAFETY: same contract as the caller's — the file must not be mutated
        // while mapped. Dropped before returning.
        let mapped = unsafe { sd_tensor::safetensors::MmapedSafetensors::new(path)? };
        if mapped
            .tensors()
            .iter()
            .any(|(name, _)| name.contains(LEGACY_SENTINEL))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Bytes these checkpoints will occupy once loaded at `dtype`.
///
/// Reads only the safetensors headers, so this is cheap even for a
/// multi-gigabyte UNet and can be asked *before* committing to the load.
///
/// Not the file size: a fp16 checkpoint loaded as f32 occupies twice what it
/// takes on disk, and that doubling is precisely what made SDXL fail to fit.
pub fn resident_bytes<P: AsRef<Path>>(paths: &[P], dtype: DType) -> Result<u64> {
    let mut total: u64 = 0;
    for path in paths {
        let path = path.as_ref();
        // SAFETY: mapped read-only for the duration of this call and dropped
        // before returning; the caller's contract is the same as elsewhere.
        let mapped = unsafe { sd_tensor::safetensors::MmapedSafetensors::new(path)? };
        for (_, view) in mapped.tensors() {
            let elems: u64 = view.shape().iter().map(|&d| d as u64).product();
            total = total.saturating_add(elems.saturating_mul(dtype.size_in_bytes() as u64));
        }
    }
    Ok(total)
}

/// Open one or more `.safetensors` files as a [`VarBuilder`].
///
/// Checkpoints using the legacy diffusers attention names are adapted
/// transparently, so model code only ever asks for the modern names. Detection
/// is per-load and header-only; a modern checkpoint pays one header parse and
/// is otherwise untouched.
///
/// # Safety
///
/// Uses `mmap`. The caller must not modify the files while the returned
/// `VarBuilder` is alive.
pub fn safetensors_var_builder<'a, P: AsRef<Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'a>> {
    let owned: Vec<PathBuf> = paths.iter().map(|p| p.as_ref().to_path_buf()).collect();
    for p in &owned {
        if !p.exists() {
            return Err(LoadError::NotFound(p.clone()));
        }
        if Format::detect(p) != Some(Format::Safetensors) {
            return Err(LoadError::Unsupported {
                path: p.clone(),
                reason: "expected a .safetensors file".to_string(),
            });
        }
    }
    tracing::debug!(count = owned.len(), ?dtype, "loading safetensors");
    // SAFETY: documented in this function's contract — files must not be
    // mutated for the lifetime of the returned VarBuilder.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&owned, dtype, device)? };

    if uses_legacy_attention_names(&owned)? {
        tracing::debug!("legacy attention names detected; adapting");
        // `rename_f` maps the name the model *asks for* onto the name that is
        // *stored*, which is the direction needed here.
        return Ok(vb.rename_f(|name: &str| {
            legacy_attention_key(name).unwrap_or_else(|| name.to_string())
        }));
    }
    Ok(vb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_formats_by_extension() {
        assert_eq!(
            Format::detect(Path::new("m.safetensors")),
            Some(Format::Safetensors)
        );
        assert_eq!(Format::detect(Path::new("m.gguf")), Some(Format::Gguf));
        assert_eq!(Format::detect(Path::new("m.ckpt")), Some(Format::Ckpt));
        assert_eq!(Format::detect(Path::new("README.md")), None);
        assert_eq!(Format::detect(Path::new("noext")), None);
    }

    #[test]
    fn missing_file_is_reported_as_not_found() {
        // VarBuilder has no Debug impl, so match on the Result directly
        // rather than using unwrap_err().
        let res = safetensors_var_builder(
            &["definitely-not-here.safetensors"],
            DType::F32,
            &Device::Cpu,
        );
        assert!(matches!(res, Err(LoadError::NotFound(_))));
    }

    #[test]
    fn wrong_extension_is_rejected_before_mmap() {
        // Cargo runs tests with CWD set to the crate root, so this file
        // exists — which is what we need to get past the NotFound check and
        // exercise the format check.
        let res = safetensors_var_builder(&["Cargo.toml"], DType::F32, &Device::Cpu);
        assert!(matches!(res, Err(LoadError::Unsupported { .. })));
    }
}
