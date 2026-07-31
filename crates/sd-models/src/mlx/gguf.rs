//! Loading an MLX weight map from a GGUF checkpoint.
//!
//! Two things stand between a GGUF file and a runnable model, and only one of
//! them is about numbers.
//!
//! **Dequantisation** is `sd_tensor::mlx_gguf`, which is bit-exact against
//! candle's reader and needs nothing from it.
//!
//! **Translation** is the other half, and it is not a substitution.
//! `stable-diffusion.cpp` writes the original CompVis/LDM names, this project
//! uses the `diffusers` ones, and between them the VAE decoder's block order is
//! *reversed* and its attention projections are stored as 1x1 convolutions.
//! That logic lives in `sd_loader::ldm` and is pure string work over a flag —
//! no tensor is involved — so it is called from here rather than reimplemented.
//! A second copy of a name mapping is exactly how the two backends would come
//! to disagree about which tensor is which.
//!
//! # Layout
//!
//! Convolution weights stay in the `(out, in, kh, kw)` order diffusers uses.
//! The MLX kernels want `(out, kh, kw, in)` and transpose at the point of use,
//! so this loader converts *names*, not layouts, exactly as `load_safetensors`
//! does. Doing it here as well would transpose twice.

use std::collections::HashMap;
use std::path::Path;

use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{mlx_gguf, Error, Result};

use super::Weights;

/// Every tensor in a GGUF file, dequantised into MLX arrays under its own name.
///
/// No translation: this is the file as it stands, which is what a checkpoint in
/// `diffusers` naming (Flux, SD 3.5) wants.
pub fn load(path: &Path, s: &Stream) -> Result<Weights> {
    let raw = mlx_gguf::load(path)?;
    let mut out = HashMap::with_capacity(raw.len());
    for (name, (shape, values)) in raw {
        out.insert(name, Array::from_slice_f32(&values, &shape)?.contiguous(s)?);
    }
    Ok(out)
}

/// One tower of an LDM-layout GGUF, translated to `diffusers` names.
///
/// `map` is `sd_loader::ldm::vae_key`, `unet_key`, or the CLIP prefix strip —
/// whichever tower is wanted. Keys the mapper returns `None` for are skipped,
/// which is how "not mine" is told apart from "translated".
fn tower(
    path: &Path,
    map: impl Fn(&str) -> Option<sd_loader::ldm::Mapped>,
    s: &Stream,
) -> Result<Weights> {
    let raw = mlx_gguf::load(path)?;
    let mut out = HashMap::new();
    for (name, (shape, values)) in raw {
        let Some(mapped) = map(&name) else { continue };
        // `[C, C, 1, 1]` -> `[C, C]`: the VAE's attention projections are 1x1
        // convolutions standing in for linears. A property of the *source*
        // layout, decided by the mapper rather than guessed from the shape.
        let shape = if mapped.squeeze_to_2d {
            if shape.len() != 4 || shape[2] != 1 || shape[3] != 1 {
                return Err(Error::Msg(format!(
                    "gguf: {name} is marked as a 1x1 projection but its shape is {shape:?}"
                )));
            }
            vec![shape[0], shape[1]]
        } else {
            shape
        };
        out.insert(
            mapped.name,
            Array::from_slice_f32(&values, &shape)?.contiguous(s)?,
        );
    }
    if out.is_empty() {
        return Err(Error::Msg(format!(
            "gguf: {} carries no tensors for this tower",
            path.display()
        )));
    }
    Ok(out)
}

/// The VAE from an LDM-layout GGUF.
///
/// **The decoder's block order is reversed** across the translation — LDM
/// builds its `up` list by inserting at the front, so `up.0` is the last block
/// processed. Getting that backwards loads cleanly and decodes noise.
pub fn vae(path: &Path, s: &Stream) -> Result<Weights> {
    tower(path, |k| sd_loader::ldm::vae_key(k, 4), s)
}

/// The UNet from an LDM-layout GGUF.
pub fn unet(path: &Path, s: &Stream) -> Result<Weights> {
    tower(path, |k| sd_loader::ldm::unet_key(k, 2), s)
}

/// CLIP's text tower from an LDM-layout GGUF.
///
/// The only change is dropping `cond_stage_model.transformer.`; below that the
/// names already agree, because CLIP came from `transformers` on both sides.
pub fn clip(path: &Path, s: &Stream) -> Result<Weights> {
    tower(
        path,
        |k| {
            k.strip_prefix("cond_stage_model.transformer.")
                .map(|rest| sd_loader::ldm::Mapped {
                    name: rest.to_string(),
                    squeeze_to_2d: false,
                })
        },
        s,
    )
}
