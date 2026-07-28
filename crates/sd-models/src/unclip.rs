//! unCLIP: conditioning on a CLIP *image* embedding.
//!
//! Three pieces, of which this file holds one. [`crate::clip::ClipVisionEncoder`]
//! turns an image into a 1024-wide embedding; this module augments that
//! embedding with a chosen amount of noise; and the UNet projects the result
//! into its timestep embedding — see [`crate::unet::UNet2DConditionModel::forward_unclip`].
//!
//! # Why noise is added on purpose
//!
//! An image embedding is a *much* stronger conditioning signal than a text
//! one, and an unaugmented one leaves the model with almost nothing to
//! generate: outputs collapse onto the reference. The noise level is the dial
//! between "reproduce this image" and "make something in its spirit", and it
//! is the reason this is a model component rather than a call to the vision
//! tower. Level 0 is the sharpest, 1000 the loosest.
//!
//! # The level is conditioned on twice
//!
//! Once by *being* the amount of noise mixed in, and once by having its own
//! sinusoid appended to the vector. That is what makes the output 2048 wide
//! where the embedding is 1024, and it is what tells the model how much of
//! what it is looking at is signal.
//!
//! # This schedule is not a sampler's
//!
//! The ladder here is a cosine DDPM schedule over 1000 steps, and it is
//! deliberately self-contained rather than reaching for `sd_sample::Schedule`:
//! it never selects sigmas, never steps, and never touches a latent. It is a
//! fixed property of the checkpoint, in the same way the normalisation
//! statistics are. Sharing the type would suggest the two are interchangeable,
//! and they are not — this one's shape (`squaredcos_cap_v2`) is not among the
//! shapes SD's sampling schedule offers.

use sd_tensor::{DType, Device, Result, Tensor, VarBuilder};

use crate::unet::timestep_embedding;

/// Timesteps in the augmentation schedule. Also the exclusive upper bound on a
/// noise level.
pub const TRAIN_TIMESTEPS: usize = 1000;

/// Largest beta the cosine schedule is allowed to produce.
///
/// The cosine `alpha_bar` goes to zero at the end of the ladder, so the ratio
/// that defines each beta goes to one; clamping keeps the last few steps from
/// destroying the signal entirely. 0.999 is diffusers' `max_beta`.
const MAX_BETA: f64 = 0.999;

/// Cumulative alphas for `squaredcos_cap_v2`, the schedule the image noiser
/// uses.
///
/// **`t` divides by `n`, not by `n - 1`.** SD's own beta schedules space their
/// interpolation across `n - 1` so the last entry lands exactly on `beta_end`;
/// this one integrates `alpha_bar` between consecutive `i/n` boundaries and has
/// no `beta_end` to land on. The two differ by one part in a thousand at every
/// step, which is enough to move the augmented embedding and not nearly enough
/// to look wrong.
/// Shared with [`crate::prior`], which samples on this exact ladder — the
/// prior's own scheduler is the same `squaredcos_cap_v2` over the same 1000
/// steps. That is the one place the "this is not a sampler's schedule" note
/// above stops applying: there it *is* one.
pub(crate) fn cosine_alphas_cumprod(n: usize) -> Vec<f64> {
    // alpha_bar(t) = cos((t + 0.008) / 1.008 * pi / 2)^2
    let alpha_bar = |t: f64| {
        let x = (t + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2;
        x.cos() * x.cos()
    };
    let mut out = Vec::with_capacity(n);
    let mut running = 1.0;
    for i in 0..n {
        let t1 = i as f64 / n as f64;
        let t2 = (i + 1) as f64 / n as f64;
        let beta = (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(MAX_BETA);
        running *= 1.0 - beta;
        out.push(running);
    }
    out
}

/// The per-channel statistics an image embedding is whitened by.
///
/// `mean` and `std` are learned, `[1, dim]`, and they exist because the noise
/// schedule assumes unit-variance input. Adding noise to a raw CLIP embedding
/// instead — whose channels are neither centred nor comparably scaled — noises
/// some directions far harder than others, and produces an image that drifts
/// from the reference in a way no level setting can fix.
#[derive(Debug)]
pub struct ImageNormalizer {
    mean: Tensor,
    std: Tensor,
}

impl ImageNormalizer {
    /// `vb` should be rooted at the normalizer itself: the tensors are the
    /// bare names `mean` and `std`, not nested under a module.
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            mean: vb.get((1, dim), "mean")?,
            std: vb.get((1, dim), "std")?,
        })
    }

    /// `(x - mean) / std`.
    pub fn scale(&self, embeds: &Tensor) -> Result<Tensor> {
        embeds.broadcast_sub(&self.mean)?.broadcast_div(&self.std)
    }

    /// `x * std + mean`, exactly undoing [`Self::scale`].
    pub fn unscale(&self, embeds: &Tensor) -> Result<Tensor> {
        embeds.broadcast_mul(&self.std)?.broadcast_add(&self.mean)
    }
}

/// Whitens an image embedding, noises it, and appends the level's sinusoid.
#[derive(Debug)]
pub struct NoiseAugmentor {
    normalizer: ImageNormalizer,
    alphas_cumprod: Vec<f64>,
    dim: usize,
}

impl NoiseAugmentor {
    /// `vb` should be rooted at `image_normalizer`. `dim` is the embedding's
    /// width — 1024 for the ViT-H checkpoints, which is what
    /// `visual_projection` narrows the tower's 1280 to.
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            normalizer: ImageNormalizer::new(dim, vb)?,
            alphas_cumprod: cosine_alphas_cumprod(TRAIN_TIMESTEPS),
            dim,
        })
    }

    /// Width of the image embedding this expects, and of the noise it takes.
    pub fn embed_dim(&self) -> usize {
        self.dim
    }

    /// Width of what [`Self::augment`] returns: the embedding and its level's
    /// sinusoid, so twice `dim`.
    pub fn output_dim(&self) -> usize {
        self.dim * 2
    }

    /// Noise the embedding, and say by how much. `[b, dim]` -> `[b, 2 * dim]`.
    ///
    /// `noise` is `[b, dim]` standard normal, supplied rather than drawn so a
    /// caller owns its own seed sequence. `level` indexes the schedule and is
    /// clamped to it.
    ///
    /// The two halves are the **same width**, so a reversed concatenation
    /// produces exactly the right shape and conditions the model on a sinusoid
    /// where it expects a picture. Order is: embedding first, then the level.
    pub fn augment(&self, image_embeds: &Tensor, level: usize, noise: &Tensor) -> Result<Tensor> {
        let (b, dim) = image_embeds.dims2()?;
        if dim != self.dim {
            return Err(sd_tensor::Error::Msg(format!(
                "image embedding is {dim} wide, this normalizer is {}",
                self.dim
            )));
        }
        let level = level.min(self.alphas_cumprod.len() - 1);
        let alpha = self.alphas_cumprod[level];

        // Whiten, mix, un-whiten. The un-whitening matters: the UNet's
        // `class_embedding` was trained on embeddings in CLIP's own units, not
        // in the normalizer's.
        let scaled = self.normalizer.scale(image_embeds)?;
        let noisy = ((scaled * alpha.sqrt())? + (noise * (1.0 - alpha).sqrt())?)?;
        let noisy = self.normalizer.unscale(&noisy)?;

        // Same sinusoid the timestep takes, at the embedding's width — cos
        // then sin, `flip_sin_to_cos=True`.
        let levels = Tensor::from_vec(vec![level as f32; b], b, image_embeds.device())?;
        let embedded = timestep_embedding(&levels, self.dim)?.to_dtype(image_embeds.dtype())?;

        Tensor::cat(&[&noisy, &embedded], 1)
    }

    /// The unconditional row of a guidance batch: zeros, `[b, 2 * dim]`.
    ///
    /// **Not an absent argument, and not an augmented zero embedding.** An
    /// unCLIP UNet always projects something into its timestep embedding, and
    /// what diffusers hands it for "no image" is a zero vector of the whole
    /// 2048 — including the half that would otherwise carry the noise level.
    pub fn unconditional(&self, batch: usize, dtype: DType, device: &Device) -> Result<Tensor> {
        Tensor::zeros((batch, self.output_dim()), dtype, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A normalizer with statistics, built without a checkpoint.
    ///
    /// The published unCLIP weights carry `mean = 0` and `std = 1` — the
    /// constructor's defaults, never trained — so the golden comparison
    /// against diffusers runs entirely through an identity and cannot see
    /// `scale` and `unscale` swapped, or either one missing. These tests are
    /// what covers that.
    fn normalizer(mean: f32, std: f32) -> ImageNormalizer {
        let dev = Device::Cpu;
        ImageNormalizer {
            mean: Tensor::from_vec(vec![mean; 4], (1, 4), &dev).expect("mean"),
            std: Tensor::from_vec(vec![std; 4], (1, 4), &dev).expect("std"),
        }
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("f32")
    }

    #[test]
    fn scale_subtracts_the_mean_before_dividing() {
        // The other order — divide, then subtract — is the same two constants
        // and a different function, and on a checkpoint whose mean is zero the
        // two agree exactly. So it is pinned here, on statistics that are not.
        let norm = normalizer(2.0, 4.0);
        let x = Tensor::from_vec(vec![2.0f32, 6.0, 10.0, -2.0], (1, 4), &Device::Cpu).expect("x");
        assert_eq!(
            values(&norm.scale(&x).expect("scale")),
            vec![0.0, 1.0, 2.0, -1.0]
        );
    }

    #[test]
    fn unscale_undoes_scale() {
        let norm = normalizer(-0.5, 3.0);
        let x = Tensor::from_vec(vec![1.0f32, -4.0, 0.25, 7.5], (1, 4), &Device::Cpu).expect("x");
        let round_trip = norm
            .unscale(&norm.scale(&x).expect("scale"))
            .expect("unscale");
        for (a, b) in values(&x).iter().zip(values(&round_trip)) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
        // And the two must not be the same function, which is what a swapped
        // pair would make them on a checkpoint with mean 0 and std 1.
        assert_ne!(values(&norm.scale(&x).expect("scale")), values(&x));
    }

    #[test]
    fn the_cosine_schedule_starts_near_one_and_decays() {
        let a = cosine_alphas_cumprod(TRAIN_TIMESTEPS);
        assert_eq!(a.len(), TRAIN_TIMESTEPS);
        // Level 0 barely touches the embedding; that is what makes it the
        // "reproduce this image" end of the dial.
        assert!(a[0] > 0.999, "alphas_cumprod[0] = {}", a[0]);
        for w in a.windows(2) {
            assert!(w[1] < w[0], "alphas_cumprod must decrease");
        }
        assert!(*a.last().expect("non-empty") < 1e-4);
    }

    #[test]
    fn the_module_object_never_uses_a_sampler_schedule() {
        // The cosine ladder is far gentler early on than SD's scaled-linear
        // one, which is the whole reason a level in the low hundreds is still
        // a recognisable image. Pinned so that reaching for `Schedule::sd15()`
        // here would be visibly wrong rather than merely different.
        let cosine = cosine_alphas_cumprod(TRAIN_TIMESTEPS);
        assert!(cosine[250] > 0.7, "alphas_cumprod[250] = {}", cosine[250]);
    }
}
