//! Loading a Flux transformer from a GGUF, weights left quantised.
//!
//! Unlike the T5 and SD 1.5 loaders there is no name translation here: the
//! published Flux GGUFs (city96's, and anything ComfyUI writes) carry the
//! original black-forest-labs names — `double_blocks.0.img_attn.qkv.weight` —
//! which is exactly what `sd_models::flux` asks for. Worth stating explicitly
//! rather than leaving the reader to infer it from an absent mapping.
//!
//! Quantised residency is what makes full-size Flux reachable: dev and schnell
//! are 12B parameters, or 48 GB at F32, against roughly 6.8 GB held as Q4_K.

use std::collections::HashMap;
use std::sync::Arc;

use sd_tensor::gguf::QTensor;
use sd_tensor::Device;

use crate::gguf::GgufInfo;
use crate::LoadError;

/// Read every tensor, keeping the quantised block data.
pub fn flux_qtensors_from_gguf(
    path: impl AsRef<std::path::Path>,
    device: &Device,
) -> Result<HashMap<String, Arc<QTensor>>, LoadError> {
    let info = GgufInfo::open(&path)?;

    // Sized on the quantised footprint, which is what is actually held. A
    // dequantised guard here would refuse a load that fits comfortably.
    let bytes: u64 = info
        .tensors
        .values()
        .map(|(shape, dtype)| {
            let n: u64 = shape.iter().map(|&d| d as u64).product();
            n * dtype.type_size() as u64 / dtype.block_size() as u64
        })
        .sum();
    sd_tensor::sysmem::check_headroom(
        bytes,
        &format!("Flux transformer weights from {}", info.path.display()),
    )?;

    let mut file = std::fs::File::open(&info.path).map_err(|e| LoadError::Unsupported {
        path: info.path.clone(),
        reason: format!("cannot open: {e}"),
    })?;
    crate::gguf::preflight(&mut file, &info.path)?;
    let content = sd_tensor::gguf::Content::read(&mut file)?;

    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    if !names.iter().any(|n| n.starts_with("double_blocks.")) {
        return Err(LoadError::Unsupported {
            path: info.path.clone(),
            reason: "no `double_blocks.*` tensors; this does not look like a Flux \
                     transformer in black-forest-labs layout"
                .to_string(),
        });
    }

    let mut out = HashMap::with_capacity(names.len());
    for name in names {
        let t = content.tensor(&mut file, &name, device)?;
        out.insert(name, Arc::new(t));
    }
    tracing::debug!(tensors = out.len(), "loaded quantised Flux transformer");
    Ok(out)
}

/// How many double and single stream blocks the file carries.
///
/// Lets a caller pick the right config rather than guess: schnell and dev are
/// 19/38, while flux-mini is 5/10, and asking for the wrong one fails on a
/// missing tensor deep into the load.
pub fn flux_block_counts(path: impl AsRef<std::path::Path>) -> Result<(usize, usize), LoadError> {
    let info = GgufInfo::open(&path)?;
    let count = |prefix: &str| {
        info.tensors
            .keys()
            .filter_map(|k| k.strip_prefix(prefix))
            .filter_map(|r| r.split('.').next())
            .filter_map(|i| i.parse::<usize>().ok())
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
    };
    Ok((count("double_blocks."), count("single_blocks.")))
}

/// Whether the checkpoint carries a distilled guidance embedding.
///
/// dev does, schnell does not, and passing a guidance scale to a model without
/// one is an error rather than a no-op.
pub fn flux_has_guidance(path: impl AsRef<std::path::Path>) -> Result<bool, LoadError> {
    let info = GgufInfo::open(&path)?;
    Ok(info.tensors.keys().any(|k| k.starts_with("guidance_in.")))
}
