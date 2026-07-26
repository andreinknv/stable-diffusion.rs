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
use sd_tensor::{DType, Device, Tensor, VarBuilder};

use crate::{LoadError, Result};

/// Reject files we can identify but cannot read, before candle tries.
///
/// candle reports a big-endian GGUF as `unsupported magic/version
/// Gguf/50331648`, which is accurate and tells the reader nothing: 50331648
/// is `0x03000000`, version 3 with its bytes reversed. Files like this are
/// real — HuggingFace hosts big-endian builds for s390x — so the difference
/// between "corrupt" and "wrong byte order" is worth saying out loud.
fn preflight(file: &mut std::fs::File, path: &Path) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};

    let mut head = [0u8; 8];
    if file.read_exact(&mut head).is_err() {
        return Err(LoadError::Unsupported {
            path: path.to_path_buf(),
            reason: "file is shorter than a GGUF header".to_string(),
        });
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|e| LoadError::Unsupported {
            path: path.to_path_buf(),
            reason: format!("cannot seek: {e}"),
        })?;

    let magic = &head[..4];
    if magic != b"GGUF" {
        return Err(LoadError::Unsupported {
            path: path.to_path_buf(),
            reason: format!("not a GGUF file: expected magic \"GGUF\", found {magic:?}"),
        });
    }

    let version = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);
    // A plausible version read one way and implausible the other is the
    // signature of reversed bytes. Versions in the wild are 1 to 3.
    if !(1..=3).contains(&version) && (1..=3).contains(&version.swap_bytes()) {
        return Err(LoadError::Unsupported {
            path: path.to_path_buf(),
            reason: format!(
                "this is a big-endian GGUF (version {} stored byte-reversed). Only \
                 little-endian files are supported — most builds are, but s390x \
                 releases are not. Use the little-endian build of this model.",
                version.swap_bytes()
            ),
        });
    }
    Ok(())
}

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
        preflight(&mut file, &path)?;
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

impl GgufInfo {
    /// Bytes these tensors will occupy once dequantised to `dtype`.
    ///
    /// Nothing like the file size. A Q4_K checkpoint dequantised to f32 is
    /// roughly **eight times** what it takes on disk, and a caller sizing a
    /// load from the file size will be wrong by that factor.
    pub fn dequantised_bytes(&self, dtype: DType) -> u64 {
        self.parameter_count()
            .saturating_mul(dtype.size_in_bytes() as u64)
    }
}

/// Load a GGUF checkpoint as a [`VarBuilder`], dequantising as it goes.
///
/// Every tensor is expanded to `dtype` and held in memory — there is no
/// quantised-inference path here, so a 4-bit checkpoint costs what its
/// dequantised weights cost, not what the file does. The memory guard is
/// applied against that expanded figure before any of it is read.
///
/// # What this does not do
///
/// It does not rename anything. GGUF checkpoints from `stable-diffusion.cpp`
/// carry the original CompVis/LDM parameter names, while the models here use
/// the `diffusers` names — so this produces a `VarBuilder` whose keys are
/// whatever the file called them. Mapping those onto our models is a separate
/// piece of work, and it belongs beside the legacy-attention conversion in
/// this crate rather than in the models. See docs/roadmap.md.
pub fn gguf_var_builder<'a>(
    path: impl AsRef<Path>,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'a>> {
    let info = GgufInfo::open(&path)?;
    let expanded = info.dequantised_bytes(dtype);
    sd_tensor::sysmem::check_headroom(
        expanded,
        &format!(
            "dequantising {} ({} tensors) to {dtype:?}",
            info.path.display(),
            info.tensors.len()
        ),
    )?;

    let mut file = std::fs::File::open(&info.path).map_err(|e| LoadError::Unsupported {
        path: info.path.clone(),
        reason: format!("cannot open: {e}"),
    })?;
    preflight(&mut file, &info.path)?;
    let content = Content::read(&mut file)?;

    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(content.tensor_infos.len());
    for name in content.tensor_infos.keys() {
        let q = content.tensor(&mut file, name, device)?;
        tensors.insert(name.clone(), q.dequantize(device)?.to_dtype(dtype)?);
    }

    tracing::debug!(
        tensors = tensors.len(),
        bytes = expanded,
        "dequantised gguf"
    );
    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}

/// Which parameter-naming convention a checkpoint uses.
///
/// GGUF files carry no reliable declaration of this — a real
/// `stable-diffusion.cpp` SD 1.5 checkpoint has **no metadata at all**, not
/// even `general.architecture` — so it has to be inferred from tensor names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// The original CompVis/LDM names, as `stable-diffusion.cpp` writes them:
    /// `model.diffusion_model.*`, `first_stage_model.*`,
    /// `cond_stage_model.transformer.*`.
    Ldm,
    /// The `diffusers` names these models expect.
    Diffusers,
    /// Neither — a language model, or something we do not recognise.
    Unknown,
}

impl GgufInfo {
    /// Infer the naming convention from tensor names.
    ///
    /// By prefix rather than by metadata, because the metadata is not there.
    pub fn layout(&self) -> Layout {
        let has = |p: &str| self.tensors.keys().any(|k| k.starts_with(p));
        if has("model.diffusion_model.") || has("first_stage_model.") {
            Layout::Ldm
        } else if has("down_blocks.") || has("conv_in.") && has("mid_block.") {
            Layout::Diffusers
        } else {
            Layout::Unknown
        }
    }

    /// Rename an LDM key to its `diffusers` equivalent, where that is a
    /// straight rewrite.
    ///
    /// Only the text encoder is a straight rewrite: LDM prefixes CLIP with
    /// `cond_stage_model.transformer.` and leaves the rest identical to what
    /// `transformers` writes, so stripping the prefix is the whole mapping.
    ///
    /// The VAE and UNet are **not** rewrites. LDM stores the VAE as
    /// `decoder.up.0.block.0.conv1` against `decoder.up_blocks.N.resnets.0
    /// .conv1`, with the block order reversed and `nin_shortcut` for
    /// `conv_shortcut`; the UNet flattens everything into `input_blocks.N.M`
    /// slots that map onto `down_blocks`/`mid_block`/`up_blocks` by
    /// arithmetic rather than by name. Both need index translation, not
    /// substitution, and are not implemented — see docs/roadmap.md.
    pub fn ldm_to_diffusers(key: &str) -> Option<String> {
        key.strip_prefix("cond_stage_model.transformer.")
            .map(|rest| rest.to_string())
    }
}

/// Load just the VAE from an LDM-layout GGUF, translated to `diffusers` names.
///
/// The counterpart to [`gguf_var_builder`] for checkpoints that need
/// translating rather than passing through. Tensors outside the VAE are
/// skipped, so this works on a full SD checkpoint where UNet and CLIP share
/// the file.
///
/// Weights arrive dequantised, so quantisation error is baked in — a Q4_0
/// checkpoint decodes to a recognisably correct image, not a bit-identical
/// one. That is the format's trade, not a defect here.
pub fn vae_var_builder_from_gguf<'a>(
    path: impl AsRef<Path>,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'a>> {
    let info = GgufInfo::open(&path)?;
    if info.layout() != Layout::Ldm {
        return Err(LoadError::Unsupported {
            path: info.path.clone(),
            reason: format!(
                "expected an LDM-layout checkpoint (stable-diffusion.cpp writes these);                  tensor names look like {:?}",
                info.layout()
            ),
        });
    }

    // Only the VAE is expanded, so size the guard on that rather than on the
    // whole file — a full SD checkpoint is mostly UNet, which is skipped.
    let vae_params: u64 = info
        .tensors
        .iter()
        .filter(|(k, _)| k.starts_with("first_stage_model."))
        .map(|(_, (shape, _))| shape.iter().map(|&d| d as u64).product::<u64>())
        .sum();
    sd_tensor::sysmem::check_headroom(
        vae_params.saturating_mul(dtype.size_in_bytes() as u64),
        &format!("dequantising the VAE from {}", info.path.display()),
    )?;

    let mut file = std::fs::File::open(&info.path).map_err(|e| LoadError::Unsupported {
        path: info.path.clone(),
        reason: format!("cannot open: {e}"),
    })?;
    preflight(&mut file, &info.path)?;
    let content = Content::read(&mut file)?;

    // Every SD VAE has four resolution levels; the decoder's block order is
    // reversed against it.
    const BLOCKS: usize = 4;

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for name in content.tensor_infos.keys() {
        let Some(mapped) = crate::ldm::vae_key(name, BLOCKS) else {
            continue;
        };
        let t = content
            .tensor(&mut file, name, device)?
            .dequantize(device)?
            .to_dtype(dtype)?;
        // LDM stores the attention projections as 1x1 convolutions; our
        // Linear wants them 2-D.
        let t = match (mapped.squeeze_to_2d, t.dims()) {
            // [C, C, 1, 1] -> [C, C].
            (true, [a, b, 1, 1]) => t.reshape((*a, *b))?,
            (true, other) => {
                return Err(LoadError::Unsupported {
                    path: info.path.clone(),
                    reason: format!(
                        "{name} was expected to be a 1x1 convolution standing in for a \
                         linear, but its shape is {other:?}"
                    ),
                })
            }
            (false, _) => t,
        };
        tensors.insert(mapped.name, t);
    }

    tracing::debug!(tensors = tensors.len(), "loaded vae from gguf");
    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}
