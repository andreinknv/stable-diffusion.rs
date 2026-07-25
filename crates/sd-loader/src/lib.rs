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

/// Open one or more `.safetensors` files as a [`VarBuilder`].
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
