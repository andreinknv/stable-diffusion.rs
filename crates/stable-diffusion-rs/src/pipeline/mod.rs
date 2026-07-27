//! End-to-end generation pipelines.

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
    sigma_to_timestep, ControlConfig, Img2ImgConfig, InpaintConfig, PipelineError, Progress,
    ProgressFn, SamplerKind, Strength, Txt2ImgConfig, Txt2ImgPipeline,
};
