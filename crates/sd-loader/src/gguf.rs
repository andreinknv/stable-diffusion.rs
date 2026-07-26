//! Reading GGUF checkpoints.
//!
//! Metadata and the tensor directory only — enough to identify a file, see
//! what architecture and quantisation it carries, and decide whether we can
//! load it. Dequantisation comes after; see docs/roadmap.md.
//!
//! Parsing is candle's, exposed through `sd_tensor::gguf`. What lives here is
//! the part that is ours: opening the file safely, and turning the format's
//! loose key/value metadata into questions a caller actually asks.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use sd_tensor::gguf::{Content, GgmlDType, Value};

use crate::{LoadError, Result};

/// A GGUF checkpoint's header: what it is, and what is in it.
#[derive(Debug)]
pub struct GgufInfo {
    pub path: PathBuf,
    /// Every metadata key/value in the file, verbatim.
    pub metadata: HashMap<String, Value>,
    /// Tensor name -> (shape, quantisation).
    pub tensors: HashMap<String, (Vec<usize>, GgmlDType)>,
}

impl GgufInfo {
    /// Read the header. Tensor *data* is not touched, so this is cheap even
    /// for a multi-gigabyte checkpoint.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(LoadError::NotFound(path));
        }
        if crate::Format::detect(&path) != Some(crate::Format::Gguf) {
            return Err(LoadError::Unsupported {
                path,
                reason: "expected a .gguf file".to_string(),
            });
        }

        let mut file = std::fs::File::open(&path).map_err(|e| LoadError::Unsupported {
            path: path.clone(),
            reason: format!("cannot open: {e}"),
        })?;
        let content = Content::read(&mut file)?;

        let tensors = content
            .tensor_infos
            .iter()
            .map(|(name, info)| (name.clone(), (info.shape.dims().to_vec(), info.ggml_dtype)))
            .collect();

        Ok(Self {
            path,
            metadata: content.metadata,
            tensors,
        })
    }

    /// A metadata value as a string, if it is one.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key)?.to_string().ok().map(|s| s.as_str())
    }

    /// The model architecture, e.g. `"llama"` or `"sd"`.
    ///
    /// `general.architecture` is the one key GGUF requires, and it decides
    /// how every other key is namespaced.
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }

    /// Quantisation types present, with how many tensors use each.
    ///
    /// A checkpoint is rarely one type: k-quant models usually keep norms and
    /// embeddings at higher precision, so "is this Q4_K" is not a yes/no
    /// question and a caller deciding what it can load needs the spread.
    pub fn quantisations(&self) -> Vec<(GgmlDType, usize)> {
        let mut counts: HashMap<GgmlDType, usize> = HashMap::new();
        for (_, dtype) in self.tensors.values() {
            *counts.entry(*dtype).or_default() += 1;
        }
        let mut out: Vec<_> = counts.into_iter().collect();
        // Commonest first, then by name so the order is stable across runs —
        // a HashMap's is not, and this ends up in user-facing output.
        out.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| format!("{:?}", a.0).cmp(&format!("{:?}", b.0)))
        });
        out
    }

    /// Total elements across every tensor.
    pub fn parameter_count(&self) -> u64 {
        self.tensors
            .values()
            .map(|(shape, _)| shape.iter().map(|&d| d as u64).product::<u64>())
            .sum()
    }
}
