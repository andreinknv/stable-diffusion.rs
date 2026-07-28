//! End-to-end generation pipelines.

use sd_models::vae::{AutoencoderKlDecoder, TinyDecoder};
use sd_tensor::Tensor;

mod flux;
pub mod placement;
mod sd3;
mod sdxl;
mod txt2img;

pub use flux::{image_token_count, paths_in, FluxConfigRun, FluxPaths, FluxPipeline};
pub use placement::{Placement, Residency, StageBytes};
pub use sd3::{sd3_paths_in, Sd3Paths, Sd3Pipeline, Sd3RunConfig};
pub use sdxl::SdxlPipeline;
pub use txt2img::{
    sigma_to_timestep, AreaConfig, Cancel, Conditioning, Control, ControlConfig, GroundedBox,
    GroundingConfig, HiresConfig, Img2ImgConfig, InpaintConfig, InstructConfig, PipelineError,
    Prediction, Progress, ProgressFn, Region, SamplerKind, Strength, Txt2ImgConfig,
    Txt2ImgPipeline, UnclipConfig, Upscale,
};

/// Whichever decoder a pipeline is using.
///
/// An enum rather than two `Option` fields so that "exactly one decoder" is a
/// property of the type. With two options the compiler demands a branch for
/// "neither", which cannot happen and would have to be an error nobody can
/// trigger.
#[derive(Debug)]
pub(crate) enum Decoder {
    Vae(Box<AutoencoderKlDecoder>),
    Tiny(Box<TinyDecoder>),
}

impl Decoder {
    /// Decode a sampler latent to `[1, 3, h, w]` in `[-1, 1]`.
    ///
    /// **Each decoder owns its own latent convention**, which is the reason
    /// this is one branch here rather than a scaling the callers apply. The
    /// VAE divides by its `scaling_factor` (0.18215 for SD 1.5, 0.13025 for
    /// SDXL); TAESD's is 1.0, so it takes the latent untouched. Applying the
    /// VAE's factor to TAESD multiplies its input by five and produces a
    /// washed-out image with no error anywhere.
    pub(crate) fn decode(&self, latent: &Tensor) -> Result<Tensor, PipelineError> {
        // **One frame at a time.** A clip is a batch, and the VAE has no
        // cross-frame interaction — decoding `n` together simply multiplies
        // the largest single allocation by `n`, which is how a three-frame
        // 512 decode reaches 6.8 GiB and trips the attention budget. Looping
        // gives byte-identical output at one frame's peak.
        let count = if latent.rank() == 4 {
            latent.dim(0)?
        } else {
            1
        };
        if count <= 1 {
            return self.decode_one(latent);
        }
        let frames = (0..count)
            .map(|i| self.decode_one(&latent.narrow(0, i, 1)?))
            .collect::<Result<Vec<_>, PipelineError>>()?;
        Ok(Tensor::cat(&frames, 0)?)
    }

    fn decode_one(&self, latent: &Tensor) -> Result<Tensor, PipelineError> {
        match self {
            Decoder::Tiny(tiny) => Ok(tiny.decode(latent)?),
            Decoder::Vae(vae) => Ok(vae.decode_tiled(latent)?),
        }
    }
}
