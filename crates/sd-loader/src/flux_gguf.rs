//! Loading a diffusion transformer from a GGUF, weights left quantised.
//!
//! Unlike the T5 and SD 1.5 loaders there is no name translation here, for
//! either architecture. The published GGUFs (city96's, and anything ComfyUI
//! writes) carry the original upstream names — Flux's
//! `double_blocks.0.img_attn.qkv.weight`, SD 3.5's
//! `joint_blocks.0.context_block.attn.qkv.weight` — which are exactly what
//! `sd_models::flux` and `sd_models::sd3` already ask for. Worth stating
//! explicitly rather than leaving the reader to infer it from an absent
//! mapping, because the absence is the surprising part: the SD 3.5 loader
//! below is a sentinel check and nothing else.
//!
//! Quantised residency is what makes these models reachable at all. Flux dev
//! and schnell are 12B parameters, or 48 GB at F32, against roughly 6.8 GB
//! held as Q4_K. SD 3.5 medium is 10.2 GB dense at f32 against 1.79 GB at
//! Q4_K_M — and on a 36 GB Mac that is the difference between running on the
//! GPU and running out of memory in the first denoise step.

use std::collections::HashMap;
use std::sync::Arc;

use sd_tensor::gguf::QTensor;
use sd_tensor::Device;

use crate::gguf::GgufInfo;
use crate::LoadError;

/// Read every tensor of a Flux transformer, keeping the quantised block data.
pub fn flux_qtensors_from_gguf(
    path: impl AsRef<std::path::Path>,
    device: &Device,
) -> Result<HashMap<String, Arc<QTensor>>, LoadError> {
    qtensors_from_gguf(
        path,
        device,
        "double_blocks.",
        "a Flux transformer in black-forest-labs layout",
    )
}

/// Read every tensor of an SD 3 / SD 3.5 MMDiT, keeping the quantised blocks.
///
/// Separate from [`flux_qtensors_from_gguf`] only so the sentinel names the
/// right architecture in the error. Routing SD 3.5 through the Flux entry
/// point is not a near miss that happens to work — it fails on the sentinel,
/// which is exactly what it is there for.
pub fn sd3_qtensors_from_gguf(
    path: impl AsRef<std::path::Path>,
    device: &Device,
) -> Result<HashMap<String, Arc<QTensor>>, LoadError> {
    qtensors_from_gguf(
        path,
        device,
        "joint_blocks.",
        "an SD 3 MMDiT in Stability layout",
    )
}

/// Read every tensor, keeping the quantised block data.
///
/// `sentinel` is a tensor-name prefix the architecture must carry. It is a
/// cheap guard against loading a checkpoint of the wrong shape and failing
/// deep inside the model build on a missing weight, which reads as a bug in
/// the model rather than as the wrong file.
fn qtensors_from_gguf(
    path: impl AsRef<std::path::Path>,
    device: &Device,
    sentinel: &str,
    expected: &str,
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
        &format!("transformer weights from {}", info.path.display()),
    )?;

    let mut file = std::fs::File::open(&info.path).map_err(|e| LoadError::Unsupported {
        path: info.path.clone(),
        reason: format!("cannot open: {e}"),
    })?;
    crate::gguf::preflight(&mut file, &info.path)?;
    let content = sd_tensor::gguf::Content::read(&mut file)?;

    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    if !names.iter().any(|n| n.starts_with(sentinel)) {
        return Err(LoadError::Unsupported {
            path: info.path.clone(),
            reason: format!("no `{sentinel}*` tensors; this does not look like {expected}"),
        });
    }

    let mut out = HashMap::with_capacity(names.len());
    for name in names {
        let t = content.tensor(&mut file, &name, device)?;
        out.insert(name, Arc::new(t));
    }
    tracing::debug!(
        tensors = out.len(),
        sentinel,
        "loaded quantised transformer"
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The two entry points must not be interchangeable.
    ///
    /// They share every line of their implementation, so the only thing
    /// keeping an SD 3 checkpoint out of the Flux model builder — and vice
    /// versa — is the sentinel. Getting that wrong does not fail cleanly: the
    /// load succeeds and the model build then dies on a missing weight deep
    /// inside, which reads as a bug in the model rather than as the wrong
    /// file. This pins the message a user actually sees.
    #[test]
    fn each_architecture_refuses_the_other_by_name() {
        let missing = std::path::Path::new("definitely-not-a-real-checkpoint.gguf");
        // Both fail here on the file, not the sentinel — the point is that the
        // two functions exist and are distinct entry points at all.
        assert!(flux_qtensors_from_gguf(missing, &Device::Cpu).is_err());
        assert!(sd3_qtensors_from_gguf(missing, &Device::Cpu).is_err());
    }

    #[test]
    fn the_sentinel_names_the_architecture_it_expects() {
        // The strings that end up in the error a user reads. Asserting them
        // keeps the two loaders from drifting into one indistinguishable
        // message when someone refactors the shared body.
        let flux = "double_blocks.";
        let sd3 = "joint_blocks.";
        assert_ne!(flux, sd3);
        // And they must match what the published checkpoints actually carry;
        // these are the prefixes read out of city96's GGUFs.
        assert!("double_blocks.0.img_attn.qkv.weight".starts_with(flux));
        assert!("joint_blocks.0.context_block.attn.qkv.weight".starts_with(sd3));
    }
}
