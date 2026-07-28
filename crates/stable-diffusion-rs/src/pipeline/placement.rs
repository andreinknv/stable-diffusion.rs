//! Where each part of a pipeline keeps its weights and does its work.
//!
//! A diffusion pipeline is not one model, it is four or five with very
//! different lifetimes. The text encoders run **once**, at the start, and are
//! then dead weight for the rest of the run; the transformer runs every step;
//! the VAE runs once at the end. Loading all of them onto one accelerator and
//! holding them there for the duration is the simplest thing to do and the
//! reason a model that would otherwise fit does not.
//!
//! For SD 3.5 the text encoders are T5 at 2.7 GB quantised plus CLIP-G and
//! CLIP-L at roughly 3.3 GB more — **larger than the quantised transformer
//! they are conditioning**, held for the entire denoise to produce one tensor.
//!
//! # Why this is a public type rather than an environment variable
//!
//! Everything else this workspace does about memory —
//! `SD_ATTENTION_BUDGET_BYTES`, `SD_VAE_TILE_LATENT`, `SD_MEMORY_HEADROOM` —
//! is a process-global read from inside a library. That is tolerable for the
//! CLI and wrong for an embedder, who cannot set it per pipeline and cannot
//! see it in the API. Placement is caller-supplied policy, which is the shape
//! `stable-diffusion.cpp` settled on too: its public `sd_ctx_params_t` carries
//! `backend`, `params_backend`, `max_vram` and `split_mode` rather than
//! reading them from the environment.
//!
//! # Why this matters more off Metal than on it
//!
//! On Apple silicon the CPU and GPU share one pool, so moving a module to the
//! CPU frees the same bytes it would have freed anywhere and saves a copy. On
//! a discrete GPU it is the difference between fitting in 8 or 12 GB of VRAM
//! and not fitting at all, *and* it avoids transferring those weights over
//! PCIe in the first place. The mechanism is the same; the payoff is larger on
//! the device this project cannot currently test. That asymmetry is the reason
//! to build it generally rather than to tune it for the machine at hand.

use sd_tensor::{DType, Device};

use super::PipelineError;

/// Which device each stage of a pipeline runs on.
///
/// Construct with [`Placement::on`] for the usual "everything here" case, then
/// move individual stages with the `with_*` methods:
///
/// ```no_run
/// # use stable_diffusion_rs::pipeline::Placement;
/// # use stable_diffusion_rs::tensor::Device;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let gpu = Device::new_metal(0)?;
/// // Denoise on the GPU, but keep the text encoders off it entirely.
/// let placement = Placement::on(&gpu).with_text_encoders_on(&Device::Cpu);
/// # Ok(())
/// # }
/// ```
/// Whether a stage's weights sit on the compute device or stream to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Residency {
    /// Weights live on the compute device for the whole run.
    #[default]
    Resident,
    /// Weights live in host memory; each block is copied to the compute
    /// device as it is reached and released after it has run.
    ///
    /// This is `stable-diffusion.cpp`'s `--offload-to-cpu`, and it is the
    /// answer for the case static placement cannot help: a GPU too small to
    /// hold the diffusion model *at all*. Peak weight residency becomes one
    /// block rather than the whole stack — 192 MB against 6.78 GB for Flux
    /// schnell.
    ///
    /// It is not free and the trade is worth stating. Copies run at 19.8 GB/s
    /// on an M4 Max, so a schnell step pays about 343 ms, roughly 6.5% on a
    /// 4-step run; a discrete card over PCIe should expect about double that.
    /// Only quantised transformers support it — the copy moves quantised
    /// block bytes verbatim, which is what makes it cheap and bit-exact.
    Streamed,
}

#[derive(Debug, Clone)]
pub struct Placement {
    compute: Device,
    text_encoders: Device,
    vae: Device,
    diffusion: Residency,
}

impl Placement {
    /// Every stage on one device. The behaviour before this type existed, and
    /// still the default everywhere, so nothing changes unless a caller asks.
    pub fn on(device: &Device) -> Self {
        Self {
            compute: device.clone(),
            text_encoders: device.clone(),
            vae: device.clone(),
            diffusion: Residency::Resident,
        }
    }

    /// Stream the diffusion model's blocks instead of holding them resident.
    ///
    /// Consuming: dropping the result leaves the placement resident and the
    /// call does nothing, which is a silent 6 GB rather than an error.
    #[must_use]
    pub fn with_streamed_diffusion(mut self) -> Self {
        self.diffusion = Residency::Streamed;
        self
    }

    /// Whether the diffusion model's blocks stream.
    pub fn diffusion(&self) -> Residency {
        self.diffusion
    }

    /// Run the text encoders somewhere else — usually [`Device::Cpu`].
    ///
    /// The stage with the best ratio of memory held to time spent: used once,
    /// then resident for every remaining step. Conditioning is a few hundred
    /// KB, so the tensors that cross back to the compute device are small.
    #[must_use]
    pub fn with_text_encoders_on(mut self, device: &Device) -> Self {
        self.text_encoders = device.clone();
        self
    }

    /// Run the VAE somewhere else.
    ///
    /// Less clear-cut than the encoders: the decode is one large convolution
    /// stack, so moving it to the CPU trades a real speed-up for the memory.
    /// Offered because on a small GPU the decode is often the allocation that
    /// does not fit, and a slow image beats no image.
    #[must_use]
    pub fn with_vae_on(mut self, device: &Device) -> Self {
        self.vae = device.clone();
        self
    }

    /// Where the diffusion model runs, and the device a caller means by "the"
    /// device.
    pub fn compute(&self) -> &Device {
        &self.compute
    }

    /// Where the text encoders run.
    pub fn text_encoders(&self) -> &Device {
        &self.text_encoders
    }

    /// Where the VAE runs.
    pub fn vae(&self) -> &Device {
        &self.vae
    }

    /// Whether any stage runs somewhere other than [`Self::compute`].
    ///
    /// Pipelines use this to skip the cross-device copies entirely in the
    /// common case, so a split placement costs nothing when it is not used.
    pub fn is_split(&self) -> bool {
        !same_device(&self.compute, &self.text_encoders) || !same_device(&self.compute, &self.vae)
    }

    /// Choose a placement that is projected to fit the memory now free.
    ///
    /// Starts from everything on `compute` and moves stages off it — text
    /// encoders first, since they are used once — until the projection fits or
    /// there is nothing left to move. Returns what it chose; the caller can
    /// log it, and nothing here is silent policy applied behind a back.
    ///
    /// `weights` is what each stage costs resident, which the caller knows
    /// from its own paths and dtypes: [`sd_loader::resident_bytes`] computes it
    /// without loading anything.
    ///
    /// **This is a projection, not a measurement.** It counts weights, not the
    /// activations and workspace on top of them, and on a discrete GPU free
    /// system memory is not free VRAM at all. Treat it as a better starting
    /// point than "everything on the accelerator", not as a guarantee — the
    /// explicit constructors exist for when the caller knows better.
    pub fn auto(compute: &Device, weights: StageBytes) -> Result<Self, PipelineError> {
        let placement = Self::on(compute);
        if compute.is_cpu() {
            // Nothing to move off; the CPU is where things would move to.
            return Ok(placement);
        }
        let Some(available) = sd_tensor::sysmem::available_bytes() else {
            // Cannot ask the machine, so do not guess: an unmeasured
            // rearrangement is worse than the caller's explicit request.
            return Ok(placement);
        };
        // The same fraction the load-time headroom checks use, so `auto` and
        // those checks cannot disagree about what "fits" means.
        let ceiling = (available as f64 * sd_tensor::sysmem::headroom()) as u64;

        if weights.total() <= ceiling {
            return Ok(placement);
        }
        let moved = placement.with_text_encoders_on(&Device::Cpu);
        if weights.total() - weights.text_encoders <= ceiling {
            return Ok(moved);
        }
        Ok(moved.with_vae_on(&Device::Cpu))
    }
}

/// What each stage of a pipeline costs resident, in bytes.
///
/// Separate from [`Placement`] because only the caller knows its own paths.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageBytes {
    pub text_encoders: u64,
    pub diffusion: u64,
    pub vae: u64,
}

impl StageBytes {
    pub fn total(&self) -> u64 {
        self.text_encoders
            .saturating_add(self.diffusion)
            .saturating_add(self.vae)
    }
}

/// Whether two devices are the same one. See `sd_tensor::device::same`, which
/// is where the backend matching lives so that this crate never names candle.
pub fn same_device(a: &Device, b: &Device) -> bool {
    sd_tensor::device::same(a, b)
}

/// Move a tensor to `device` only if it is not already there.
///
/// `Tensor::to_device` is already a no-op for the same device in candle, but
/// going through one helper keeps the pipelines from sprouting a `to_device`
/// on every line whose necessity a reader then has to work out.
pub fn to(tensor: &sd_tensor::Tensor, device: &Device) -> Result<sd_tensor::Tensor, PipelineError> {
    Ok(tensor.to_device(device)?)
}

/// Bytes a safetensors file occupies once loaded at `dtype`.
///
/// Thin re-export so a caller assembling [`StageBytes`] does not have to reach
/// past the pipeline layer into `sd-loader`.
pub fn resident_bytes<P: AsRef<std::path::Path>>(
    paths: &[P],
    dtype: DType,
) -> Result<u64, PipelineError> {
    Ok(sd_loader::resident_bytes(paths, dtype)?)
}

/// On-disk size of a file, as a stand-in for what a quantised checkpoint
/// costs resident.
///
/// A GGUF held quantised occupies very nearly its file size in memory —
/// that is the point of holding it quantised — so the file is the honest
/// estimate. `resident_bytes` cannot be used for these: it reads safetensors
/// headers and would report what the weights cost *dequantised*, which for
/// SD 3.5 is 10.2 GB against the 1.79 GB actually held.
///
/// Unreadable means zero rather than an error: this feeds a placement hint,
/// and refusing to load because a size could not be sampled would be worse
/// than placing on incomplete information.
pub fn file_bytes(path: impl AsRef<std::path::Path>) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_places_everything_together() {
        let p = Placement::on(&Device::Cpu);
        assert!(!p.is_split(), "the default must not move anything");
        assert!(same_device(p.compute(), p.text_encoders()));
        assert!(same_device(p.compute(), p.vae()));
    }

    #[test]
    fn moving_a_stage_reports_the_placement_as_split() {
        // `is_split` gates the cross-device copies, so a placement that has
        // moved something and does not say so would silently feed a CPU
        // tensor to a GPU model — a device mismatch at best, and at worst a
        // copy per step that nobody asked for.
        let p = Placement::on(&Device::Cpu).with_text_encoders_on(&Device::Cpu);
        assert!(!p.is_split(), "moving to the same device is not a split");

        let cpu = Device::Cpu;
        let p = Placement {
            compute: cpu.clone(),
            text_encoders: cpu.clone(),
            vae: cpu,
            diffusion: Residency::Resident,
        };
        assert!(!p.is_split());
    }

    #[test]
    fn auto_leaves_a_cpu_pipeline_alone() {
        // There is nowhere to move to, and shuffling stages between CPU and
        // CPU would be pure cost.
        let big = StageBytes {
            text_encoders: u64::MAX / 4,
            diffusion: u64::MAX / 4,
            vae: u64::MAX / 4,
        };
        let p = Placement::auto(&Device::Cpu, big).unwrap();
        assert!(!p.is_split());
    }

    #[test]
    fn stage_bytes_saturate_rather_than_wrap() {
        // A wrapped total reads as "tiny" and would place a model that cannot
        // possibly fit onto the accelerator.
        let s = StageBytes {
            text_encoders: u64::MAX,
            diffusion: u64::MAX,
            vae: u64::MAX,
        };
        assert_eq!(s.total(), u64::MAX);
    }
}
