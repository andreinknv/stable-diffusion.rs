//! Text-to-image: tokenizer, text encoder, UNet, sampler, VAE.

use std::path::{Path, PathBuf};

use sd_models::clip::{
    preprocess, ClipTextConfig, ClipTextEncoder, ClipTokenizer, ClipVisionConfig, ClipVisionEncoder,
};
use sd_models::controlnet::ControlNet;
use sd_models::image::UnitImage;
use sd_models::ip_adapter::{ImageProjModel, NUM_TOKENS};
use sd_models::prior::{PriorConfig, PriorScheduler, PriorTransformer};
use sd_models::unclip::NoiseAugmentor;
use sd_models::unet::{UNet2DConditionModel, UNetConfig};
use sd_models::vae::{AutoencoderKlDecoder, AutoencoderKlEncoder, TinyDecoder, VaeConfig};

use super::Decoder;
use sd_sample::{
    euler_ancestral_step, lcm_sigmas, lcm_step, lcm_timesteps, sigmas_for_steps,
    DpmSolverPlusPlus2M, Schedule,
};
use sd_tensor::rng::SeededRng;
use sd_tensor::{DType, Device, Tensor};

/// How the first pass is enlarged before the second sees it.
///
/// The choice matters more than it looks. **Nearest** is the one to reach for
/// whenever the output must not gain colours absent from the input — every
/// interpolating mode invents intermediate values, which is right for
/// photographic work and destructive for anything with a fixed palette.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Upscale {
    /// Resize the *latent*, keeping the model's own texture. Cheap: no decode
    /// and re-encode, and no VAE round trip to lose detail to.
    #[default]
    LatentNearest,
    /// Latent-space bilinear. Smoother, and it invents values.
    LatentBilinear,
    /// Decode, resize in pixel space with Lanczos, re-encode.
    ///
    /// The most faithful to what an upscaler would see, and the most
    /// expensive — it pays for a full VAE round trip, which is itself lossy.
    PixelLanczos,
}

/// A two-pass generation: compose small, then add detail large.
///
/// Not the same as generating big. A model composes at its training resolution
/// and **duplicates subjects** when asked to compose above it — two heads, two
/// horizons. The first pass fixes composition at a size the model handles; the
/// second adds detail at a size it could not have composed coherently.
#[derive(Debug, Clone)]
pub struct HiresConfig {
    /// The first pass. Its width and height are the *native* size.
    pub base: Txt2ImgConfig,
    pub width: usize,
    pub height: usize,
    /// How much of the schedule the second pass re-runs. 0.5-0.7 is the usual
    /// band: high enough to add detail, low enough to keep the composition.
    pub strength: Strength,
    pub upscale: Upscale,
}

/// An instruction-guided edit: "change the sky to sunset".
///
/// Distinct from img2img, which takes a *description of the result* and a
/// strength. This takes a description of the **change**, and the source image
/// conditions the model through the UNet's extra four input channels rather
/// than by being partially noised — so the untouched parts are held by the
/// conditioning rather than by stopping the schedule early.
#[derive(Debug, Clone)]
pub struct InstructConfig {
    pub base: Txt2ImgConfig,
    pub init_image: PathBuf,
    /// How strongly to follow the *instruction*. `base.cfg_scale` is reused
    /// for this, as in every other config here.
    ///
    /// How strongly to keep the **source image**. Raising it holds more of the
    /// original; lowering it lets the edit roam. 1.5 is the published default,
    /// and it is a genuinely separate axis from text guidance — which is why
    /// this model needs three predictions per step rather than two.
    pub image_guidance: f64,
}

/// One grounded object: a phrase and where to put it.
#[derive(Debug, Clone)]
pub struct GroundedBox {
    /// `[x0, y0, x1, y1]` in **`[0, 1]`**, not pixels. Relative because the
    /// model was trained that way and because a box should survive a change of
    /// resolution — and because both are plausible readings of four numbers,
    /// which is why the type says so.
    pub bbox: [f32; 4],
    pub phrase: String,
}

/// Generation grounded on boxes: "put this thing *here*".
///
/// The only conditioning here that addresses *placement*. Text cannot do it
/// reliably and a ControlNet needs a picture of the layout; this takes the
/// boxes directly.
#[derive(Debug, Clone)]
pub struct GroundingConfig {
    pub base: Txt2ImgConfig,
    pub boxes: Vec<GroundedBox>,
    /// Fraction of the schedule to ground for. 0.3 is the published default.
    ///
    /// **Not 1.0.** GLIGEN grounds early, while composition is being decided,
    /// then finishes free — holding the model to the boxes all the way through
    /// costs image quality for placement it has already achieved. This is the
    /// "scheduled sampling" the paper describes.
    pub grounding_fraction: f64,
}

/// One region of the canvas, with its own prompt.
#[derive(Debug, Clone)]
pub struct Region {
    /// `[1, 1, height, width]` in `[0, 1]` at **pixel** resolution, like a
    /// ControlNet hint. Downsampled to the latent grid by **mean** pooling,
    /// not max: a region boundary should fade over a latent cell rather than
    /// claim it outright, which is the opposite of what an inpainting mask
    /// wants and is worth not copying by reflex.
    pub mask: Tensor,
    pub conditioning: Conditioning,
}

/// A generation with different prompts in different places.
///
/// Composed *before* sampling, by blending each region's noise prediction —
/// not by generating separately and compositing, which produces visible joins
/// because neither half ever saw the other.
#[derive(Debug, Clone)]
pub struct AreaConfig {
    /// The prompt outside every region, and the run's other settings.
    pub base: Txt2ImgConfig,
    pub regions: Vec<Region>,
}

/// A cancellation token, shared with whatever wants to stop a generation.
///
/// A token rather than a callback return value, so the ordinary
/// [`ProgressFn`] stays a plain `FnMut` and callers who never cancel write
/// nothing. Checked once per step: a step is not interruptible internally, so
/// cancelling costs at most one step of latency.
#[derive(Debug, Clone, Default)]
pub struct Cancel(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the generation to stop. Safe to call from any thread.
    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// A prompt pair, encoded once.
///
/// Encoding is not free and does not change between frames, so a caller
/// generating a sequence should do it once and reuse it — which also removes
/// a class of bug where re-encoding produces marginally different conditioning
/// per frame.
#[derive(Debug, Clone)]
pub struct Conditioning {
    /// `[2, 77, d]`: unconditional first, then conditional.
    context: Tensor,
}

impl Conditioning {
    /// The raw `[2, 77, d]` tensor, unconditional row first.
    pub fn context(&self) -> &Tensor {
        &self.context
    }
}

/// What the model's output means.
///
/// SD 1.5, SDXL and SD 2.x's 512 "base" checkpoints predict the noise. SD 2.1's
/// 768 checkpoints — and fine-tunes derived from them — predict `v` instead,
/// and the two are **not distinguishable from the weights**: a v-prediction
/// model run as epsilon loads and samples with no error anywhere, and returns
/// saturated colour noise — measured, not guessed; see
/// `assets/sd21-crab-512.png` for what the same seed produces when it is
/// right. The checkpoint's `scheduler_config.json` carries `prediction_type`;
/// this mirrors it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Prediction {
    /// The model outputs noise. `x0 = x - sigma*eps`.
    #[default]
    Epsilon,
    /// The model outputs `v`. See [`Prediction::denoise`] for the algebra.
    V,
}

impl Prediction {
    /// Recover the `x0` estimate from a model output.
    ///
    /// For `v`, diffusers computes
    /// `sqrt(alpha)*sample - sqrt(beta)*output` against the **scaled** input
    /// `x / sqrt(1+sigma^2)`, with `alpha = 1/(1+sigma^2)`. Substituting gives
    /// the form below in terms of the unscaled latent this crate carries:
    ///
    /// ```text
    ///   x0 = x/(1 + sigma^2)  -  v * sigma/sqrt(1 + sigma^2)
    /// ```
    ///
    /// At `sigma = 0` that is the identity, as it must be.
    pub fn denoise(
        self,
        latent: &Tensor,
        output: &Tensor,
        sigma: f64,
    ) -> Result<Tensor, PipelineError> {
        match self {
            Prediction::Epsilon => Ok((latent - (output * sigma)?)?),
            Prediction::V => {
                let s2 = sigma * sigma + 1.0;
                Ok(((latent / s2)? - (output * (sigma / s2.sqrt()))?)?)
            }
        }
    }
}

/// Which sampler to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SamplerKind {
    #[default]
    EulerAncestral,
    DpmPlusPlus2M,
    /// Latent consistency sampling. **Only meaningful with an LCM-distilled
    /// model or adapter**, and wants 4-8 steps at `cfg_scale` near 1 — the
    /// guidance is distilled in, so applying more on top double-counts it and
    /// blows the image out.
    Lcm,
}

/// Everything a single generation needs.
#[derive(Debug, Clone)]
pub struct Txt2ImgConfig {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub cfg_scale: f64,
    pub seed: u64,
    pub sampler: SamplerKind,
    /// Reuse the model's prediction between steps while it is estimated not
    /// to have moved much. 0 disables it, **bit-identically**.
    ///
    /// The units are *predicted relative change in the model's output*,
    /// accumulated since the last real evaluation. So 0.2 means "reuse until
    /// the prediction is estimated to have drifted 20 %", which is a statement
    /// about the model rather than about an arbitrary metric.
    ///
    /// # It needs a deterministic sampler
    ///
    /// **Refused with `euler_a` or `lcm`**, and that is the most important
    /// thing on this page. An ancestral sampler draws fresh noise every step,
    /// so consecutive predictions never stop moving and there is nothing to
    /// reuse. Measured relative L1 change of the output between steps, SD 1.5
    /// at 20 steps over three prompts:
    ///
    /// ```text
    ///   euler_a    0.34 .. 0.90    never small
    ///   dpmpp2m    0.02 .. 0.78    small through the middle of the run
    /// ```
    ///
    /// Forcing it anyway does not degrade gracefully — it returns colour
    /// speckle — so it errors instead.
    ///
    /// # Measured, SD 1.5, 512, 20 steps, `dpmpp2m`
    ///
    /// `evaluated` is how many of the twenty steps ran the model, which is the
    /// exact saving; wall clock on this machine is noisy enough that the same
    /// configuration has varied 2x, so the step count is the number to trust
    /// and the seconds are minimum-of-three, alternated.
    ///
    /// ```text
    ///   0.00    20/20    22.6 s   baseline
    ///   0.05    20/20            nothing skipped
    ///   0.10    12/20    20.4 s   clean, mean 15.0/255 from the baseline
    ///   0.20     7/20    15.6 s   clean, mean 23.0/255 — 1.45x end to end
    ///   0.40     4/20     6.7 s   degraded: speckle and smeared edges
    /// ```
    ///
    /// So the usable band is **0.10 to 0.20**, and 0.20 skips 65 % of the
    /// steps for a 1.45x end-to-end run — about 2.9x on the denoising itself,
    /// the rest being load and decode that caching cannot touch.
    ///
    /// # How the prediction is made
    ///
    /// TeaCache's method: the relative change in the **timestep embedding**,
    /// rescaled through a per-model fitted polynomial into an estimate of the
    /// output change, and accumulated. The polynomial is [`CACHE_RESCALE`],
    /// fitted here by `--example cache_fit` rather than borrowed.
    ///
    /// The predecessor measured how far the *input latent* moved and bought
    /// about 9 %. Both predictors were fitted against the true output change
    /// and scored: on `dpmpp2m` the timestep embedding is **1.6x** the better
    /// of the two, and its fit is far better conditioned — the latent's
    /// degree-4 coefficients reach 1.7e4 over a narrow range, which is
    /// overfitting rather than prediction.
    ///
    /// Worth knowing, because it is the opposite of what was assumed: under
    /// `euler_a` the *latent* is the better predictor of the two. It is
    /// predicting a quantity that never gets small, so it buys nothing either
    /// way.
    pub cache_threshold: f64,
    /// Frames to generate as one clip. 1 is a still image.
    ///
    /// **A clip is a batch**: `n` frames is a batch of `n`, denoised together
    /// so a motion module can attend across them. Without a motion adapter
    /// attached this just produces `n` independent images that share a
    /// schedule — which is a batch, not an animation.
    pub frames: usize,
    /// Set to stop the run early. `None` means it cannot be cancelled.
    pub cancel: Option<Cancel>,
}

impl Default for Txt2ImgConfig {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 512,
            height: 512,
            steps: 20,
            cfg_scale: 7.5,
            seed: 0,
            sampler: SamplerKind::default(),
            frames: 1,
            cache_threshold: 0.0,
            cancel: None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("missing model file: {0}\nExpected the standard diffusers layout under the model directory.")]
    MissingFile(PathBuf),
    #[error(
        "missing {0}\n\n\
         A stock SD 1.5 download does not contain this file: the repository ships the slow \
         tokenizer (vocab.json + merges.txt) instead. Copy `tokenizer.json` from \
         `openai/clip-vit-large-patch14`, which is the tokenizer SD 1.5 uses."
    )]
    MissingTokenizerJson(PathBuf),
    #[error(
        "loading the VAE from {path}: {source}\n\n\
         sd-loader adapts the legacy diffusers attention names \
         (`query`/`key`/`value`/`proj_attn`) that stock SD 1.5 weights use, so a missing \
         `to_q`-style tensor here means the checkpoint is neither layout — check it is an \
         SD 1.5 VAE and not, say, an SDXL one."
    )]
    VaeWeights {
        path: PathBuf,
        source: sd_tensor::Error,
    },
    #[error("{0} must be a multiple of 8 (latents are 1/8 scale), got {1}")]
    NotMultipleOfEight(&'static str, usize),
    #[error("steps must be at least 1")]
    NoSteps,
    #[error("tokenizer: {0}")]
    Tokenize(#[from] sd_models::clip::TokenizeError),
    #[error("loading weights: {0}")]
    Load(#[from] sd_loader::LoadError),
    #[error(
        "the LoRA does not match this model: {unmatched} of its layers have no weight here, \
         first `{first}`.\n\n\
         A LoRA names the layers it corrects, so entries with nowhere to go mean it was \
         trained for a different architecture — an SDXL adapter on SD 1.5, say. Applying \
         the rest would render a plausible image that is not the one the adapter describes, \
         which nothing downstream can detect, so it is refused here instead."
    )]
    LoraMismatch { unmatched: usize, first: String },
    #[error(
        "this pipeline has no ControlNet.\n\n\
         Attach one with `Txt2ImgPipeline::with_controlnet(path)` before calling \
         `run_control`. Running the control config without one would silently ignore \
         the control image and return an ordinary generation."
    )]
    NoControlNet,
    #[error(
        "this pipeline has no IP-Adapter.\n\n\
         Load one with `Txt2ImgPipeline::load_with_ip_adapter`. The adapter's weights live \
         inside the UNet's cross-attention layers, so unlike a ControlNet it cannot be \
         attached after the fact."
    )]
    NoIpAdapter,
    #[error(
        "the embedding `{name}` is {got} wide but this text encoder is {want}.\n\n\
         Embeddings are trained against one encoder: 768 for SD 1.5, 1024 for SD 2.x, \
         and SDXL's are a pair. Using the wrong one is the common mistake and would \
         otherwise surface as a shape error from inside the transformer."
    )]
    EmbeddingWidth {
        name: String,
        got: usize,
        want: usize,
    },
    #[error(
        "this checkpoint has no GLIGEN grounding.\n\n\
         Grounding lives inside the UNet's transformer blocks, so it needs a GLIGEN \
         checkpoint rather than an adapter — an ordinary SD 1.5 UNet has no fusers and \
         would ignore the boxes silently."
    )]
    NoGrounding,
    #[error(
        "this is an InstructPix2Pix checkpoint — use `run_instruct` (`sdrs instruct`).\n\n\
         Its UNet takes 8 input channels: the noisy latent *and* the source image's. \
         Plain text-to-image supplies only 4, which surfaces deep inside a convolution \
         as an `in_channel mismatch` rather than as the mistake it is."
    )]
    NeedsInstruct,
    #[error(
        "this is an unCLIP checkpoint — use `run_unclip` (`sdrs unclip`).\n\n\
         Its UNet projects a CLIP *image* embedding into every timestep embedding, so \
         there is no such thing as running it from text alone: with nothing supplied it \
         would condition on a zero vector, which is the guidance batch's unconditional \
         row and not a picture of anything."
    )]
    NeedsUnclip,
    #[error(
        "this checkpoint does not condition on an image.\n\n\
         `run_unclip` needs a UNet with a `class_embedding` — the open mirrors are \
         `diffusers/stable-diffusion-2-1-unclip-i2i-h` and friends. An ordinary SD 2.x \
         model has nowhere to put an image embedding, and adding one would silently do \
         nothing."
    )]
    NoUnclip,
    #[error(
        "this unCLIP checkpoint is missing {0}\n\n\
         unCLIP needs its own CLIP vision tower and the statistics its image embeddings \
         are whitened by. Both live beside the UNet in the published layout, under \
         `image_encoder/` and `image_normalizer/`."
    )]
    MissingUnclipPart(PathBuf),
    #[error(
        "this pipeline has no prior.\n\n\
         Running unCLIP without a reference image means sampling the image embedding from \
         the prompt, which needs the `prior/` half of a text-to-image unCLIP checkpoint. \
         Attach it with `Txt2ImgPipeline::with_prior(model_dir)`, or supply an image."
    )]
    NoPrior,
    #[error(
        "this unCLIP checkpoint has no image encoder, so it cannot read a reference \
         image.\n\n\
         The text-to-image variants ship no `image_encoder/` at all — a prompt is their \
         only input. Drop the reference image to generate from the prompt through the \
         prior, or point `--model` at an image-variation checkpoint such as \
         `diffusers/stable-diffusion-2-1-unclip-i2i-h`."
    )]
    NoImageEncoder,
    #[error(
        "the prior emits a {got}-wide embedding but this checkpoint's image half takes \
         {want}.\n\n\
         The two must come from the same unCLIP checkpoint. Karlo's prior is ViT-L (768) \
         and there are unCLIP models built on ViT-H (1024); \
         `diffusers/stable-diffusion-2-1-unclip-t2i-h` combines those two and so is not \
         loadable here. `-t2i-l` is 768 throughout, which is what this expects."
    )]
    PriorWidth { got: usize, want: usize },
    #[error(
        "step caching needs a deterministic sampler; {sampler} re-noises every step.\n\n\
         An ancestral sampler draws fresh noise each step, so consecutive predictions never \
         stop moving — measured on SD 1.5 at 20 steps, the model's output changes by 34-90 % \
         between steps under `euler_a` against 2-78 % under `dpmpp2m`. There is nothing to \
         reuse, and reusing anyway produces colour speckle rather than an image.\n\n\
         Use `--sampler dpmpp2m`, or set `--cache-threshold 0`."
    )]
    CacheNeedsDeterministicSampler { sampler: &'static str },
    #[error("cancelled after {completed} of {total} steps")]
    Cancelled { completed: usize, total: usize },
    #[error("tensor: {0}")]
    Tensor(#[from] sd_tensor::Error),
}

impl PipelineError {
    /// Whether this is the memory guard declining rather than a fault.
    ///
    /// Exposed so callers do not have to match on message text. The GPU smoke
    /// test skips on a refusal and fails on anything else, and getting that
    /// backwards makes the suite go red whenever the machine is busy — which
    /// teaches people to ignore it.
    pub fn is_memory_refusal(&self) -> bool {
        // The marker rather than a literal: one definition, in the module
        // that produces refusals.
        self.to_string().contains(sd_tensor::refusal::MARKER)
    }
}

/// What a progress callback is told after each denoising step.
///
/// A struct rather than positional arguments because the interesting field is
/// `latent`, and a fourth positional `&Tensor` would be easy to ignore — which
/// is the opposite of what it is for.
pub struct Progress<'a> {
    /// 1-based; equal to `total` on the last step.
    pub step: usize,
    pub total: usize,
    pub sigma: f64,
    /// The model's current estimate of the finished image, as a latent.
    ///
    /// **Not the sampler's latent**, and the difference is the whole value of
    /// this field. The latent at step 5 of 20 is `x0 + sigma*noise` with sigma
    /// still around 4, so decoding it shows noise; this is the `x0` the model
    /// predicts, which decodes to a blurry version of the final image and
    /// sharpens as the run proceeds. That is what a preview is for, and it is
    /// what every diffusion UI shows.
    ///
    /// Borrowed, and decoding it is the caller's choice: a full VAE decode per
    /// step costs more than the denoising does, which is exactly why
    /// [`Txt2ImgPipeline::with_taesd`] exists. `Txt2ImgPipeline::preview`
    /// decodes it with whichever decoder is attached.
    pub denoised: &'a Tensor,
    /// How many steps so far actually ran the model, as opposed to reusing a
    /// cached prediction.
    ///
    /// Equal to `step` when caching is off. Exposed because the saving from
    /// `cache_threshold` is *steps skipped*, and wall-clock on this machine is
    /// noisy enough that a timed A/B has lied about it before — this is the
    /// same quantity, measured exactly and for free.
    pub evaluated: usize,
}

/// Called after each denoising step.
///
/// A callback rather than a log line because this crate has no logging
/// dependency and adding one is out of scope. The CLI owns the reporting, and
/// a library caller can render progress however it likes.
pub type ProgressFn<'a> = &'a mut dyn FnMut(Progress<'_>);

/// How much of the schedule an img2img run replaces.
///
/// `1.0` ignores the input entirely; `0.0` returns it unchanged. The value
/// selects where in the sigma ladder to start: at strength `s` with `n` steps,
/// the run begins at index `n - round(n*s)` and executes the remaining steps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Strength(f64);

impl Strength {
    /// Clamped to `[0, 1]`; anything outside is a caller error, not a mode.
    pub fn new(v: f64) -> Self {
        Self(v.clamp(0.0, 1.0))
    }

    pub fn get(self) -> f64 {
        self.0
    }

    /// Index into a `steps + 1` sigma ladder at which to begin.
    ///
    /// Public because it is the whole meaning of the parameter: `steps -
    /// start_index(steps)` is how much work a run will actually do, and a
    /// caller sizing a progress bar or a time estimate needs it.
    pub fn start_index(self, steps: usize) -> usize {
        let run = (steps as f64 * self.0).round() as usize;
        steps.saturating_sub(run.min(steps))
    }
}

impl Default for Strength {
    fn default() -> Self {
        Self(0.75)
    }
}

/// Everything an inpaint needs: an img2img plus the mask that bounds it.
#[derive(Debug, Clone)]
pub struct InpaintConfig {
    pub base: Img2ImgConfig,
    /// Greyscale mask, resized to the run's dimensions. **White repaints.**
    pub mask: std::path::PathBuf,
}

/// What the sampler must leave alone, and what it is restoring.
///
/// Held together because they are only meaningful together: the mask says
/// where the original applies and `init` is the original to apply.
struct Keep<'a> {
    /// Latent-resolution mask, 1 where the model may write.
    mask: &'a Tensor,
    /// The encoded original, restored outside the mask at every step.
    init: &'a Tensor,
}

/// Everything an img2img generation needs.
#[derive(Debug, Clone)]
pub struct Img2ImgConfig {
    pub base: Txt2ImgConfig,
    /// Source image, resized to `base.width` x `base.height` on load.
    pub init_image: std::path::PathBuf,
    pub strength: Strength,
}

/// A generation conditioned on what an image *is*, rather than on how it looks.
///
/// Different from img2img and from IP-Adapter, and the difference is where the
/// image enters. img2img starts from the picture's own latent and stops the
/// schedule early, so composition survives; IP-Adapter gives the cross
/// attention extra tokens to look at. unCLIP hands the model a single CLIP
/// embedding of the *whole image* and nothing else — no pixels reach it — so
/// what comes back shares the subject and the feel of the reference and is
/// composed from scratch.
#[derive(Debug, Clone)]
pub struct UnclipConfig {
    pub base: Txt2ImgConfig,
    /// The image to take the embedding from, or `None` to have the **prior**
    /// invent one from the prompt.
    ///
    /// Those are the two halves of unCLIP and they need different checkpoints:
    /// a reference image needs `image_encoder/`, and `None` needs `prior/`.
    /// Attach the latter with [`Txt2ImgPipeline::with_prior`].
    ///
    /// Read at the tower's 224x224 by shortest edge and centre crop, which is
    /// what `CLIPImageProcessor` does.
    pub init_image: Option<PathBuf>,
    /// Steps and guidance for the prior, when one is running.
    ///
    /// Ignored when `init_image` is set. diffusers uses 25 and 4.0; the prior
    /// is cheap next to the image half — a 768-vector rather than a latent —
    /// so there is little reason to lower either.
    pub prior_steps: usize,
    pub prior_guidance: f64,
    /// How much noise is mixed into the embedding, `0..1000`.
    ///
    /// **The dial between "this image" and "something like it"**, and the
    /// reason the whole augmentation exists: an unaugmented CLIP embedding is
    /// a strong enough signal that the model reproduces the reference and
    /// generates almost nothing. 0 is the tightest, and diffusers' default.
    pub noise_level: usize,
}

/// A generation steered by a ControlNet.
#[derive(Debug, Clone)]
pub struct ControlConfig {
    pub base: Txt2ImgConfig,
    /// One control map per attached ControlNet, in the order they were
    /// attached, each with its own strength.
    ///
    /// A `Vec` because one is frequently not enough — pose for a figure plus
    /// depth or edges for the scene is a common pairing, and swapping models
    /// between generations is the alternative. The corrections are summed
    /// before the UNet sees them, which is what diffusers does.
    ///
    /// Each map is `[1, 3, height, width]` in `[0, 1]` at **pixel** resolution.
    /// A prepared tensor rather than a path, because what counts as a control
    /// map depends on which ControlNet is loaded and this crate cannot check
    /// the two agree — a caller with a pose skeleton it generated itself
    /// should not have to push it back through a detector.
    /// [`crate::canny`] makes one for the canny models.
    pub controls: Vec<Control>,
}

/// One control map and its strength.
#[derive(Debug, Clone)]
pub struct Control {
    pub hint: Tensor,
    /// 1.0 is the published strength; 0.0 contributes exactly nothing.
    pub scale: f64,
}

/// Repeat a `[2, s, d]` guidance context to `[2n, s, d]`.
///
/// Uncond rows first, then cond — the same layout the loop's `narrow` split
/// expects. Interleaving instead runs, and guides each frame by another
/// frame's conditioning: a subtle wrongness rather than an error.
fn repeat_per_frame(context: &Tensor, frames: usize) -> Result<Tensor, PipelineError> {
    if frames == 1 {
        return Ok(context.clone());
    }
    let uncond = context.narrow(0, 0, 1)?;
    let cond = context.narrow(0, 1, 1)?;
    let mut rows: Vec<Tensor> = Vec::with_capacity(frames * 2);
    rows.extend(std::iter::repeat_n(uncond, frames));
    rows.extend(std::iter::repeat_n(cond, frames));
    Ok(Tensor::cat(&rows, 0)?)
}

/// Repeat a single `[1, d]` row to `[n, d]`.
///
/// One row per frame, because the UNet carries frames on the batch and a
/// single row where `n` are expected fails inside the timestep embedding's
/// addition rather than broadcasting.
fn repeat_rows(row: &Tensor, n: usize) -> Result<Tensor, PipelineError> {
    if n == 1 {
        return Ok(row.clone());
    }
    let rows: Vec<Tensor> = std::iter::repeat_n(row.clone(), n).collect();
    Ok(Tensor::cat(&rows, 0)?)
}

/// Downsample a pixel-resolution region mask to the latent grid.
///
/// **Mean**, where an inpainting mask uses max. The reason is the opposite of
/// the one there: an inpaint needs a latent cell freed if *any* pixel under it
/// is free, while a region boundary should fade across the cell it straddles.
/// Reusing `latent_mask` here would give every region a hard edge at latent
/// resolution — 8 pixels wide in the output.
fn area_mask(mask_px: &Tensor, lh: usize, lw: usize) -> Result<Tensor, PipelineError> {
    let (_, _, h, w) = mask_px.dims4()?;
    if h != lh * 8 || w != lw * 8 {
        return Err(PipelineError::Tensor(sd_tensor::Error::Msg(format!(
            "region mask is {h}x{w}, expected {}x{}",
            lh * 8,
            lw * 8
        ))));
    }
    Ok(mask_px
        .reshape((1, 1, lh, 8, lw, 8))?
        .mean(5)?
        .mean(3)?
        .contiguous()?)
}

/// Regional prompts for one run.
struct Areas {
    /// `(mask, context)` per region. Masks are `[1, 1, lh, lw]` at *latent*
    /// resolution, contexts already doubled for the guidance batch.
    regions: Vec<(Tensor, Tensor)>,
}

/// One denoising run: what it starts from and everything optional about it.
///
/// **The point is that adding the next conditioning does not add a twelfth
/// parameter.** This began as five wrapper methods forwarding positionally
/// into an eleven-argument `denoise_inner`, each behind its own
/// `#[allow(clippy::too_many_arguments)]`, and every new capability — a
/// ControlNet, regions, unCLIP's class labels — widened all of them. Adding a
/// field here costs one line and cannot be passed in the wrong position.
struct Denoise<'a> {
    cfg: &'a Txt2ImgConfig,
    sigmas: &'a [f64],
    /// One or more conditionings; `select` picks between them per step.
    contexts: &'a [Tensor],
    select: Option<&'a mut dyn FnMut(usize, usize) -> usize>,
    /// Hold a region at the original — inpainting.
    keep: Option<Keep<'a>>,
    control: Option<Hints>,
    areas: Option<Areas>,
    /// unCLIP's augmented image embedding, already doubled for guidance.
    class_labels: Option<Tensor>,
}

impl<'a> Denoise<'a> {
    /// The plain run: text conditioning and nothing else.
    fn new(cfg: &'a Txt2ImgConfig, sigmas: &'a [f64], contexts: &'a [Tensor]) -> Self {
        Self {
            cfg,
            sigmas,
            contexts,
            select: None,
            keep: None,
            control: None,
            areas: None,
            class_labels: None,
        }
    }

    fn selecting(mut self, select: Option<&'a mut dyn FnMut(usize, usize) -> usize>) -> Self {
        self.select = select;
        self
    }

    fn keeping(mut self, keep: Option<Keep<'a>>) -> Self {
        self.keep = keep;
        self
    }

    fn controlled(mut self, control: Option<Hints>) -> Self {
        self.control = control;
        self
    }

    fn over_areas(mut self, areas: Areas) -> Self {
        self.areas = Some(areas);
        self
    }

    fn conditioned_on_image(mut self, class_labels: Tensor) -> Self {
        self.class_labels = Some(class_labels);
        self
    }
}

/// Where one sampler step is: the latent, the model's estimate, and the two
/// sigmas it moves between.
///
/// Grouped because the five travelled together through a twelve-argument
/// signature, and `sigma` and `sigma_next` are the same type in the same order
/// — the one pair a positional call can silently swap.
struct StepPoint<'a> {
    latent: &'a Tensor,
    denoised: &'a Tensor,
    sigma: f64,
    sigma_next: f64,
    t: f64,
}

/// The shape a sampler draws noise in.
#[derive(Clone, Copy)]
struct LatentShape {
    frames: usize,
    height: usize,
    width: usize,
}

impl LatentShape {
    fn dims(self) -> (usize, usize, usize, usize) {
        (self.frames, 4, self.height, self.width)
    }
}

/// Turns a relative change in the timestep embedding into a predicted
/// relative change in the model's output.
///
/// Degree-4 polynomial, least-squares fitted on **SD 1.5** over 57 steps from
/// three prompts by `--example cache_fit`. Fitted here rather than taken from
/// TeaCache's published tables, because this project has twice been caught
/// borrowing a constant without checking what it was a constant *of* — and
/// once was this very feature.
///
/// **Per model.** The coefficients describe SD 1.5's schedule and its
/// embedding widths; SDXL or SD 2.x need their own, which is one command.
/// Using these on another architecture is not catastrophic — the accumulator
/// is monotone either way — but the threshold stops meaning what it says.
const CACHE_RESCALE: [f64; 5] = [
    5.036842e-2,
    1.022504e-1,
    -4.397247e-1,
    5.716702e-1,
    -1.481600e-1,
];

/// The sampler's name if it re-noises each step, or `None` if it integrates.
///
/// Both ancestral kinds here draw fresh noise every step — Euler ancestral by
/// definition, and LCM because a consistency model jumps to `x0` and re-noises
/// out rather than integrating. Either way consecutive inputs are decorrelated
/// and there is no redundancy for a cache to find.
fn ancestral_name(sampler: SamplerKind) -> Option<&'static str> {
    match sampler {
        SamplerKind::EulerAncestral => Some("euler_a"),
        SamplerKind::Lcm => Some("lcm"),
        SamplerKind::DpmPlusPlus2M => None,
    }
}

/// Evaluate [`CACHE_RESCALE`], clamped at zero.
///
/// A least-squares polynomial is free to go negative where the data does not
/// constrain it, and a negative contribution would let the accumulator *fall*
/// — reusing a prediction for longer the further the model moved. Clamping is
/// what makes the accumulator monotone, which is what makes the threshold a
/// bound rather than a suggestion.
pub fn cache_rescale(moved: f64) -> f64 {
    CACHE_RESCALE
        .iter()
        .enumerate()
        .map(|(p, c)| c * moved.powi(p as i32))
        .sum::<f64>()
        .max(0.0)
}

/// One step of a step-cache calibration: the candidate predictors, and the
/// quantity they are trying to predict.
///
/// All three are **relative L1** distances from the previous step —
/// `|a - b|_1 / |b|_1` — which is what TeaCache accumulates and what makes the
/// numbers comparable across steps whose tensors differ in magnitude by orders
/// of magnitude.
#[derive(Debug, Clone, Copy, Default)]
pub struct CalibrationPoint {
    /// How far the scaled input latent moved. What the shipped predictor uses.
    pub latent: f64,
    /// How far the timestep embedding moved. What TeaCache uses.
    pub temb: f64,
    /// How far the guided noise prediction actually moved — the target.
    pub output: f64,
}

/// Relative L1 distance, `|a - b|_1 / |b|_1`.
///
/// Relative rather than absolute because the tensors involved span orders of
/// magnitude across a run, and an absolute threshold would mean something
/// different at step 1 than at step 19.
fn relative_l1(a: &Tensor, b: &Tensor) -> Result<f64, PipelineError> {
    let diff = (a - b)?
        .abs()?
        .sum_all()?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()? as f64;
    let base = b
        .abs()?
        .sum_all()?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()? as f64;
    Ok(diff / base.max(f64::EPSILON))
}

/// The control maps for one run, already doubled for the guidance batch.
struct Hints {
    /// `[2, 3, h, w]` each, paired with its strength, in ControlNet order.
    maps: Vec<(Tensor, f64)>,
}

/// A loaded SD 1.5 pipeline.
pub struct Txt2ImgPipeline {
    tokenizer: ClipTokenizer,
    text_encoder: ClipTextEncoder,
    unet: UNet2DConditionModel,
    decoder: Decoder,
    vae_encoder: AutoencoderKlEncoder,
    schedule: Schedule,
    device: Device,
    /// Spatial conditioning, in attachment order. Empty is the common case.
    controlnets: Vec<ControlNet>,
    /// What the UNet's output means. See [`Prediction`].
    prediction: Prediction,
    /// GLIGEN's grounding projection, when the checkpoint carries one.
    position_net: Option<sd_models::gligen::PositionNet>,
    /// Textual-inversion embeddings, spliced into prompts by trigger word.
    embeddings: Vec<sd_loader::embedding::Embedding>,
    /// The image tower and projection, when an IP-Adapter is attached. The
    /// adapter's own weights live inside the UNet.
    ip: Option<(ClipVisionEncoder, ImageProjModel)>,
    /// The noise augmentation, and the image tower when the checkpoint ships
    /// one, for an unCLIP UNet.
    unclip: Option<UnclipStack>,
    /// The prior and its own tokenizer and text encoder, when attached.
    ///
    /// Opt-in through [`Txt2ImgPipeline::with_prior`] rather than loaded with
    /// the rest, because it is 4.6 GB that an image-variation run never
    /// touches — and the two paths are alternatives, never both at once.
    ///
    /// Its text encoder is a *third* one: SD 1.5's CLIP-L with a projection
    /// head, beside the checkpoint's own 1024-wide OpenCLIP-H. Different
    /// vocabulary positions, different width, different weights.
    prior: Option<Box<PriorStack>>,
}

/// What an unCLIP UNet needs in front of it.
///
/// The augmentation is always required — every unCLIP checkpoint carries an
/// `image_normalizer` and the UNet cannot run without one. **The image tower
/// is not**: the text-to-image checkpoints ship no `image_encoder` at all,
/// because a prompt is their only input. So a pipeline can legitimately be
/// unCLIP with no way to read a reference image, and asking it for an image
/// variation is a clear error rather than a missing file.
struct UnclipStack {
    vision: Option<ClipVisionEncoder>,
    augmentor: NoiseAugmentor,
}

/// Everything the text half of unCLIP needs.
struct PriorStack {
    tokenizer: ClipTokenizer,
    text_encoder: ClipTextEncoder,
    prior: PriorTransformer,
}

impl std::fmt::Debug for Txt2ImgPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Txt2ImgPipeline")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

fn require(path: PathBuf) -> Result<PathBuf, PipelineError> {
    if path.exists() {
        Ok(path)
    } else {
        Err(PipelineError::MissingFile(path))
    }
}

/// [`require`], for the two files only an unCLIP checkpoint has.
///
/// Its own error because "missing model file" would send someone looking at
/// their download when what they actually have is an unCLIP UNet in a
/// directory assembled for an ordinary one.
fn require_unclip(path: PathBuf) -> Result<PathBuf, PipelineError> {
    if path.exists() {
        Ok(path)
    } else {
        Err(PipelineError::MissingUnclipPart(path))
    }
}

/// Reduce a pixel-resolution mask to latent resolution by 8x8 **maximum**.
///
/// Maximum, not average. A latent cell covers 64 pixels, and if any of them is
/// to be repainted the cell has to be free to change — averaging would leave
/// edge cells partly frozen and produce a visible seam of half-old, half-new
/// content exactly where the join needs to be cleanest. Erring toward
/// repainting dilates the mask by up to one latent cell, which the composite
/// in pixel space then trims back to the user's actual boundary.
fn latent_mask(mask_px: &Tensor, lh: usize, lw: usize) -> Result<Tensor, PipelineError> {
    let (_, _, h, w) = mask_px.dims4()?;
    if h != lh * 8 || w != lw * 8 {
        return Err(PipelineError::Tensor(sd_tensor::Error::Msg(format!(
            "mask is {h}x{w}, expected {}x{}",
            lh * 8,
            lw * 8
        ))));
    }
    Ok(mask_px
        .reshape((1, 1, lh, 8, lw, 8))?
        .max(5)?
        .max(3)?
        .contiguous()?)
}

/// Nearest index in the training sigmas to `sigma`, as a timestep.
///
/// The UNet takes a discrete timestep, not a sigma, so the continuous sampling
/// sigma has to be mapped back onto the 1000-entry training schedule. A linear
/// search is ample: this runs once per step, twenty times per image.
pub fn sigma_to_timestep(schedule: &Schedule, sigma: f64) -> f64 {
    let sigmas = schedule.sigmas();
    let mut best = 0usize;
    let mut best_dist = f64::INFINITY;
    for (i, &s) in sigmas.iter().enumerate() {
        let d = (s - sigma).abs();
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    best as f64
}

impl Txt2ImgPipeline {
    /// Load with a LoRA merged into the UNet.
    ///
    /// `multiplier` is the adapter's strength; 0 reproduces [`Self::load`]
    /// exactly, bit for bit.
    ///
    /// **A partially-applied adapter is refused rather than rendered.** If any
    /// of the LoRA's entries finds no weight in this UNet, the adapter is for
    /// a different architecture — or the name mapping missed — and the result
    /// would be a plausible image that is not the one the adapter describes.
    /// That is the failure worth erroring on, because nothing downstream can
    /// detect it.
    pub fn load_with_lora(
        model_dir: &Path,
        device: &Device,
        lora_path: &Path,
        multiplier: f64,
    ) -> Result<Self, PipelineError> {
        let lora = sd_loader::Lora::load(lora_path, device)?;
        Self::load_inner(model_dir, device, Some((&lora, multiplier)))
    }

    /// Load from the standard diffusers directory layout.
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self, PipelineError> {
        Self::load_inner(model_dir, device, None)
    }

    /// Which SD variant a checkpoint is, read from the weights themselves.
    ///
    /// The cross-attention key projection's input width is the text encoder's
    /// output width, and it is the one number that separates these two: 768 for
    /// SD 1.5, 1024 for SD 2.x. Reading it from a tensor shape rather than from
    /// `config.json` means no JSON dependency and no trusting a file that need
    /// not be present — and a checkpoint always carries its own shapes.
    ///
    /// Falls back to SD 1.5 when the tensor is absent, which keeps every
    /// existing checkpoint loading exactly as before.
    fn detect_variant(unet_path: &Path) -> (UNetConfig, ClipTextConfig) {
        // `conv_in` first: InstructPix2Pix is SD 1.5 with **eight** input
        // channels rather than four — the noisy latent concatenated with the
        // source image's — and that is invisible in the cross attention.
        if let Ok(Some(shape)) = sd_tensor::tensor_shape(unet_path, "conv_in.weight") {
            if shape.get(1) == Some(&8) {
                return (UNetConfig::instruct_pix2pix(), ClipTextConfig::sd15());
            }
        }
        const CROSS_KEY: &str = "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight";
        match sd_tensor::tensor_shape(unet_path, CROSS_KEY) {
            Ok(Some(shape)) if shape.last() == Some(&1024) => {
                // unCLIP is SD 2.x plus a `class_embedding`, and the tensor's
                // presence is the whole tell — the geometry is identical, so
                // there is nothing else to read. Its *width* is read too
                // rather than assumed at 2048: it is twice the image
                // embedding's, and a checkpoint built on a different tower
                // would differ there and nowhere else.
                const CLASS_KEY: &str = "class_embedding.linear_1.weight";
                if let Ok(Some(shape)) = sd_tensor::tensor_shape(unet_path, CLASS_KEY) {
                    if let Some(&dim) = shape.last() {
                        let mut cfg = UNetConfig::unclip();
                        cfg.class_projection = Some(dim);
                        return (cfg, ClipTextConfig::sd2());
                    }
                }
                (UNetConfig::sd2(), ClipTextConfig::sd2())
            }
            _ => (UNetConfig::sd15(), ClipTextConfig::sd15()),
        }
    }

    /// Whether the checkpoint's scheduler asks for v-prediction.
    ///
    /// A substring test on `scheduler_config.json`, deliberately: the file is
    /// small, the token is unambiguous, and a JSON parser for one boolean is
    /// not worth a dependency the rest of this workspace does without. Absent
    /// file means epsilon, which is right for every checkpoint that predates
    /// the option.
    ///
    /// It matters because the two are **indistinguishable from the weights** —
    /// a v-prediction model sampled as epsilon renders saturated colour noise
    /// and reports nothing wrong. That was checked by forcing this function to
    /// return `Epsilon` for an SD 2.1 checkpoint, not assumed.
    fn detect_prediction(model_dir: &Path) -> Prediction {
        let path = model_dir.join("scheduler/scheduler_config.json");
        match std::fs::read_to_string(path) {
            Ok(text) if text.contains("v_prediction") => Prediction::V,
            _ => Prediction::Epsilon,
        }
    }

    fn load_inner_with_ip(
        model_dir: &Path,
        device: &Device,
        ip_vb: Option<&sd_tensor::VarBuilder<'_>>,
    ) -> Result<Self, PipelineError> {
        Self::load_all(model_dir, device, None, ip_vb, None)
    }

    fn load_inner(
        model_dir: &Path,
        device: &Device,
        lora: Option<(&sd_loader::Lora, f64)>,
    ) -> Result<Self, PipelineError> {
        Self::load_all(model_dir, device, lora, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_all(
        model_dir: &Path,
        device: &Device,
        lora: Option<(&sd_loader::Lora, f64)>,
        ip_vb: Option<&sd_tensor::VarBuilder<'_>>,
        motion_vb: Option<&sd_tensor::VarBuilder<'_>>,
    ) -> Result<Self, PipelineError> {
        let tokenizer_path = model_dir.join("tokenizer/tokenizer.json");
        if !tokenizer_path.exists() {
            // Its own variant: this one is missing from a *correct* SD 1.5
            // download, so "file not found" would send people looking for a
            // broken download rather than for the file they actually need.
            return Err(PipelineError::MissingTokenizerJson(tokenizer_path));
        }
        let text_encoder_path = require(model_dir.join("text_encoder/model.safetensors"))?;
        let unet_path = require(model_dir.join("unet/diffusion_pytorch_model.safetensors"))?;
        let vae_path = require(model_dir.join("vae/diffusion_pytorch_model.safetensors"))?;

        // unCLIP carries a fourth tower — a 2.5 GB ViT-H — inside the
        // checkpoint. It is loaded whenever the UNet asks for one, because
        // there is no way to run such a checkpoint without it, and it is
        // counted here for the same reason the other three are.
        let image_encoder_path = model_dir.join("image_encoder/model.safetensors");
        let normalizer_path =
            model_dir.join("image_normalizer/diffusion_pytorch_model.safetensors");

        // See the same check in sdxl.rs: weights stay resident for the whole
        // run and dominate, so the projection has to include them.
        let mut resident = vec![&text_encoder_path, &unet_path, &vae_path];
        if image_encoder_path.exists() {
            resident.push(&image_encoder_path);
        }
        let weights = sd_loader::resident_bytes(&resident, DType::F32)?;
        // The *active* tile, not the default: see the note in sdxl.rs.
        let tile = sd_models::vae::tile_latent_edge()?;
        let decode_peak = sd_models::vae::DecoderConfig::from(&VaeConfig::sd15())
            .peak_alloc_bytes(1, tile, tile, DType::F32)
            .unwrap_or(0);
        sd_tensor::sysmem::check_headroom(
            weights.saturating_add(decode_peak),
            &format!("loading the pipeline from {}", model_dir.display()),
        )?;

        let (unet_cfg, clip_cfg) = Self::detect_variant(&unet_path);
        let prediction = Self::detect_prediction(model_dir);

        let tokenizer = ClipTokenizer::from_file(&tokenizer_path)?;

        let vb = sd_loader::safetensors_var_builder(&[&text_encoder_path], DType::F32, device)?;
        let text_encoder = ClipTextEncoder::new(&clip_cfg, vb)?;

        let vb = match lora {
            None => sd_loader::safetensors_var_builder(&[&unet_path], DType::F32, device)?,
            Some((lora, multiplier)) => {
                let (vb, applied) = sd_loader::safetensors_var_builder_with_lora(
                    &[&unet_path],
                    DType::F32,
                    device,
                    lora,
                    multiplier,
                )?;
                if !applied.unmatched.is_empty() {
                    return Err(PipelineError::LoraMismatch {
                        unmatched: applied.unmatched.len(),
                        first: applied.unmatched[0].clone(),
                    });
                }
                // No log line: this crate deliberately has no logging
                // dependency (see `ProgressFn`). The count is on `Applied` for
                // a caller that wants to report it.
                vb
            }
        };
        // Both adapters live *inside* the UNet's blocks, so they must be
        // present when it is built — unlike a ControlNet, which sits beside it
        // and can be attached afterwards.
        let unet = match (ip_vb, motion_vb) {
            (Some(ip), None) => {
                UNet2DConditionModel::new_with_ip(&unet_cfg, vb, ip.pp("ip_adapter"), NUM_TOKENS)?
            }
            (None, Some(motion)) => {
                UNet2DConditionModel::new_with_motion(&unet_cfg, vb, motion.clone())?
            }
            (Some(_), Some(_)) => {
                // They occupy different layers and would probably compose, but
                // each installs its own construction-scoped source and the two
                // have never been run together. Refused rather than guessed.
                return Err(PipelineError::Tensor(sd_tensor::Error::Msg(
                    "an IP-Adapter and a motion adapter together are untested".to_string(),
                )));
            }
            (None, None) => UNet2DConditionModel::new(&unet_cfg, vb)?,
        };

        // GLIGEN's grounding projection lives in the UNet file alongside the
        // fusers, so it is found the same way they are: by name, present or
        // absent. A checkpoint without grounding simply has no `position_net`.
        let unet_vb = sd_loader::safetensors_var_builder(&[&unet_path], DType::F32, device)?;
        let position_net = if unet_vb.contains_tensor("position_net.null_positive_feature") {
            Some(sd_models::gligen::PositionNet::new(
                768,
                unet_cfg.cross_attention_dim,
                unet_vb.pp("position_net"),
            )?)
        } else {
            None
        };

        let vb = sd_loader::safetensors_var_builder(&[&vae_path], DType::F32, device)?;
        let vae = AutoencoderKlDecoder::new(&VaeConfig::sd15(), vb).map_err(|source| {
            PipelineError::VaeWeights {
                path: vae_path.clone(),
                source,
            }
        })?;

        // Same file, both halves. The encoder is only used by img2img, but it
        // is cheap to build and mmap means the weights are not read twice.
        let vb = sd_loader::safetensors_var_builder(&[&vae_path], DType::F32, device)?;
        let vae_encoder = AutoencoderKlEncoder::new(&VaeConfig::sd15(), vb).map_err(|source| {
            PipelineError::VaeWeights {
                path: vae_path.clone(),
                source,
            }
        })?;

        // Loaded eagerly rather than through a `with_` builder, unlike every
        // other adapter here, because this one is not optional: a UNet with a
        // `class_embedding` cannot produce an image without it. Refusing at
        // load time beats refusing after the weights are resident.
        let unclip = if unet.takes_class_labels() {
            let normalizer_path = require_unclip(normalizer_path)?;
            // The width comes from the statistics themselves rather than from
            // the tower's config: the image-variation checkpoint is 1024 wide
            // and the text-to-image one 768, and there is no tower at all to
            // ask in the second case.
            let width = sd_tensor::tensor_shape(&normalizer_path, "mean")
                .ok()
                .flatten()
                .and_then(|s| s.last().copied())
                .ok_or_else(|| PipelineError::MissingUnclipPart(normalizer_path.clone()))?;
            let vb = sd_loader::safetensors_var_builder(&[&normalizer_path], DType::F32, device)?;
            let augmentor = NoiseAugmentor::new(width, vb)?;

            // Present only on the image-variation checkpoints. Absent is not
            // an error — it just means this model reads prompts, not pictures.
            let vision = if image_encoder_path.exists() {
                let vb =
                    sd_loader::safetensors_var_builder(&[&image_encoder_path], DType::F32, device)?;
                Some(ClipVisionEncoder::new(&ClipVisionConfig::vit_h_14(), vb)?)
            } else {
                None
            };
            Some(UnclipStack { vision, augmentor })
        } else {
            None
        };

        Ok(Self {
            tokenizer,
            text_encoder,
            unet,
            decoder: Decoder::Vae(Box::new(vae)),
            vae_encoder,
            // **AnimateDiff needs a linear beta schedule**, not SD 1.5's
            // scaled-linear. diffusers' own documentation says the checkpoints
            // "can be sensitive to the beta schedule" and recommends linear,
            // and it is not a small effect: the *reference* pipeline, run with
            // SD 1.5's default PNDM/scaled-linear, produces black-and-white
            // banded mush at 16 frames, and a recognisable car with
            // DDIM/linear at otherwise identical settings.
            //
            // Worth knowing because nothing warns you: a motion adapter loads
            // cleanly onto the wrong schedule and renders noise.
            schedule: match motion_vb {
                Some(_) => Schedule::new(1000, 0.00085, 0.012, sd_sample::BetaSchedule::Linear),
                None => Schedule::sd15(),
            },
            device: device.clone(),
            controlnets: Vec::new(),
            prediction,
            embeddings: Vec::new(),
            position_net,
            ip: None,
            unclip,
            prior: None,
        })
    }

    /// Load every tower from a single LDM-layout GGUF checkpoint.
    ///
    /// `tokenizer` is a separate path because **the checkpoint does not carry
    /// one**. `stable-diffusion.cpp` writes no GGUF metadata at all — not the
    /// vocabulary, not even `general.architecture` — so unlike a language
    /// model in this format, an SD checkpoint cannot supply its own
    /// tokenizer. Copy `tokenizer.json` from `openai/clip-vit-large-patch14`.
    ///
    /// Weights are dequantised on load: a 4-bit checkpoint costs what its
    /// expanded weights cost, not what the file does. The memory guard sizes
    /// each tower against that.
    pub fn load_gguf(
        gguf: &Path,
        tokenizer: &Path,
        device: &Device,
    ) -> Result<Self, PipelineError> {
        if !gguf.exists() {
            return Err(PipelineError::MissingFile(gguf.to_path_buf()));
        }
        if !tokenizer.exists() {
            return Err(PipelineError::MissingTokenizerJson(tokenizer.to_path_buf()));
        }

        let tokenizer = ClipTokenizer::from_file(tokenizer)?;

        let vb = sd_loader::clip_var_builder_from_gguf(gguf, DType::F32, device)?;
        let text_encoder = ClipTextEncoder::new(&ClipTextConfig::sd15(), vb)?;

        let vb = sd_loader::unet_var_builder_from_gguf(gguf, DType::F32, device)?;
        // GGUF checkpoints in this layout are SD 1.5; stable-diffusion.cpp
        // writes no metadata at all, so there is nothing to detect from.
        let unet = UNet2DConditionModel::new(&UNetConfig::sd15(), vb)?;

        let vb = sd_loader::vae_var_builder_from_gguf(gguf, DType::F32, device)?;
        let vae = AutoencoderKlDecoder::new(&VaeConfig::sd15(), vb).map_err(|source| {
            PipelineError::VaeWeights {
                path: gguf.to_path_buf(),
                source,
            }
        })?;
        let vb = sd_loader::vae_var_builder_from_gguf(gguf, DType::F32, device)?;
        let vae_encoder = AutoencoderKlEncoder::new(&VaeConfig::sd15(), vb).map_err(|source| {
            PipelineError::VaeWeights {
                path: gguf.to_path_buf(),
                source,
            }
        })?;

        Ok(Self {
            tokenizer,
            text_encoder,
            unet,
            decoder: Decoder::Vae(Box::new(vae)),
            vae_encoder,
            schedule: Schedule::sd15(),
            device: device.clone(),
            controlnets: Vec::new(),
            prediction: Prediction::Epsilon,
            embeddings: Vec::new(),
            position_net: None,
            ip: None,
            // A single-file GGUF is an LDM-layout SD 1.x/2.x checkpoint. No
            // published one carries unCLIP's image tower, and there is nowhere
            // in the format to put it.
            unclip: None,
            prior: None,
        })
    }

    /// Attach a ControlNet.
    ///
    /// Takes the pipeline by value and returns it, so a pipeline either has a
    /// ControlNet from the moment it is built or never does — there is no
    /// window in which a caller holds one it believes is controlled and is not.
    ///
    /// The ControlNet is built from `UNetConfig::sd15()`, the same config the
    /// UNet is, which is what guarantees a correction per skip at the right
    /// width. An SDXL ControlNet here will fail to load rather than run wrong.
    pub fn with_controlnet(mut self, path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(PipelineError::MissingFile(path.to_path_buf()));
        }
        let vb = sd_loader::safetensors_var_builder(&[path], DType::F32, &self.device)?;
        self.controlnets
            .push(ControlNet::new(&UNetConfig::sd15(), vb)?);
        Ok(self)
    }

    /// Use TAESD instead of the VAE for decoding.
    ///
    /// About 5 MB against the VAE's 330, and correspondingly faster. Lossier —
    /// fine detail is softened — so this is a speed and memory trade, not a
    /// free win, and it is opt-in for that reason.
    ///
    /// Only the *decoder* is replaced. Encoding (img2img, inpainting) still
    /// goes through the VAE, because a latent produced by TAESD's encoder and
    /// then denoised is not the same starting point as the VAE's.
    pub fn with_taesd(mut self, path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(PipelineError::MissingFile(path.to_path_buf()));
        }
        let vb = sd_loader::safetensors_var_builder(&[path], DType::F32, &self.device)?;
        let tiny = TinyDecoder::new(4, 3, vb)?;

        // Replacing rather than adding: the VAE decoder's 189 MB is dropped
        // here, and on Metal a drop alone frees nothing — candle pools its
        // buffers and returns them only inside `drop_unused_buffers`, which
        // runs on synchronise. The same reason `run_releasing` exists.
        self.decoder = Decoder::Tiny(Box::new(tiny));
        self.device.synchronize()?;
        Ok(self)
    }

    /// Decode a latent with whichever decoder is attached.
    ///
    /// TAESD takes the sampler's latent unscaled, where the VAE divides by
    /// `0.18215` first — each decoder owns its own convention, so the choice
    /// is a single branch here rather than a scaling the caller has to get
    /// right.
    fn decode(&self, latent: &Tensor) -> Result<Tensor, PipelineError> {
        self.decoder.decode(latent)
    }

    /// Decode a latent for previewing, with whichever decoder is attached.
    ///
    /// The same decode the final image gets — there is no reduced-quality
    /// preview path, because a preview that does not look like the result is
    /// worse than none. Attach TAESD first if this is called every step.
    pub fn preview(&self, latent: &Tensor) -> Result<Tensor, PipelineError> {
        self.decode(latent)
    }

    /// How many ControlNets are attached.
    ///
    /// [`ControlConfig::controls`] must have exactly this many entries.
    pub fn controlnet_count(&self) -> usize {
        self.controlnets.len()
    }

    pub fn has_controlnet(&self) -> bool {
        !self.controlnets.is_empty()
    }

    /// Generate under spatial control. Returns `[1, 3, height, width]`.
    pub fn run_control(&self, cfg: &ControlConfig) -> Result<Tensor, PipelineError> {
        self.run_control_with_progress(cfg, &mut |_| {})
    }

    /// [`Self::run_control`], reporting progress after each step.
    pub fn run_control_with_progress(
        &self,
        cfg: &ControlConfig,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        if self.controlnets.is_empty() {
            return Err(PipelineError::NoControlNet);
        }
        if cfg.controls.len() != self.controlnets.len() {
            return Err(PipelineError::Tensor(sd_tensor::Error::Msg(format!(
                "{} control maps for {} attached ControlNets — each needs its own",
                cfg.controls.len(),
                self.controlnets.len()
            ))));
        }
        let base = &cfg.base;
        if base.width % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("width", base.width));
        }
        if base.height % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("height", base.height));
        }
        if base.steps == 0 {
            return Err(PipelineError::NoSteps);
        }
        for (i, control) in cfg.controls.iter().enumerate() {
            let d = control.hint.dims4()?;
            if d.2 != base.height || d.3 != base.width {
                return Err(PipelineError::Tensor(sd_tensor::Error::Msg(format!(
                    "control map {i} is {}x{}, expected {}x{} — control maps are at \
                     pixel resolution, not latent",
                    d.2, d.3, base.height, base.width
                ))));
            }
        }

        let cond = self.encode(&base.prompt)?;
        let uncond = self.encode(&base.negative_prompt)?;
        let context = Tensor::cat(&[&uncond, &cond], 0)?;

        // Each hint is doubled to match the guidance batch. Both halves get
        // the same control: guidance contrasts the *prompts*, and giving the
        // unconditional half no control would make the contrast partly about
        // the control map instead.
        let dtype = self.unet.dtype();
        let maps = cfg
            .controls
            .iter()
            .map(|c| {
                Ok((
                    Tensor::cat(&[&c.hint, &c.hint], 0)?.to_dtype(dtype)?,
                    c.scale,
                ))
            })
            .collect::<Result<Vec<_>, PipelineError>>()?;

        let mut rng = SeededRng::new(base.seed);
        let (lh, lw) = (base.height / 8, base.width / 8);
        let sigmas = self.sigmas_for(base.sampler, base.steps);
        let latent = (rng.randn((cfg.base.frames.max(1), 4, lh, lw), &self.device)? * sigmas[0])?;

        let latent = self.denoise(
            latent,
            Denoise::new(base, &sigmas, std::slice::from_ref(&context))
                .controlled(Some(Hints { maps })),
            &mut rng,
            progress,
        )?;
        self.decode(&latent)
    }

    /// Encode a prompt to `[1, 77, 768]`.
    fn encode(&self, text: &str) -> Result<Tensor, PipelineError> {
        if self.embeddings.is_empty() {
            let ids = self.tokenizer.encode(text)?;
            let ids = Tensor::from_vec(ids, (1, self.tokenizer.max_length()), &self.device)?;
            return Ok(self.text_encoder.forward(&ids)?);
        }
        self.encode_with_embeddings(text)
    }

    /// [`Self::encode`] with textual-inversion triggers spliced in.
    ///
    /// The trigger is tokenised like any other word, and the *first* `n`
    /// positions it occupies are overwritten with the learned vectors. That
    /// works because the trigger only ever needs to reserve space — its own
    /// token embeddings are discarded — and because reserving is all a word
    /// with no vocabulary entry can do.
    ///
    /// A trigger that tokenises to fewer positions than the embedding has
    /// vectors is padded by repeating it in the prompt text, so the count is
    /// checked rather than assumed: a short trigger would otherwise silently
    /// drop the tail of a multi-vector embedding.
    fn encode_with_embeddings(&self, text: &str) -> Result<Tensor, PipelineError> {
        // Expand each trigger to as many copies as it has vectors, so it
        // reserves the right number of positions.
        let mut expanded = text.to_string();
        for emb in &self.embeddings {
            if emb.len() > 1 {
                let repeated = std::iter::repeat_n(emb.name.as_str(), emb.len())
                    .collect::<Vec<_>>()
                    .join(" ");
                expanded = expanded.replace(&emb.name, &repeated);
            }
        }

        let ids = self.tokenizer.encode(&expanded)?;
        let ids = Tensor::from_vec(ids.clone(), (1, self.tokenizer.max_length()), &self.device)?;
        let mut embeds = self.text_encoder.embed_tokens(&ids)?;

        // Where each trigger's first token landed. Recomputed from the ids
        // rather than from character offsets: BPE splits are not positions in
        // the string.
        for emb in &self.embeddings {
            let trigger_ids = self.tokenizer.encode_content(&emb.name)?;
            if trigger_ids.is_empty() {
                continue;
            }
            let width = embeds.dim(2)?;
            if emb.width() != width {
                return Err(PipelineError::EmbeddingWidth {
                    name: emb.name.clone(),
                    got: emb.width(),
                    want: width,
                });
            }
            let flat: Vec<u32> = ids.flatten_all()?.to_vec1::<u32>()?;
            let mut placed = 0usize;
            let mut i = 0usize;
            while i + trigger_ids.len() <= flat.len() && placed < emb.len() {
                if flat[i..i + trigger_ids.len()] == trigger_ids[..] {
                    let vector = emb.vectors.narrow(0, placed, 1)?.unsqueeze(0)?;
                    embeds = embeds.slice_assign(&[0..1, i..i + 1, 0..width], &vector)?;
                    placed += 1;
                    i += trigger_ids.len();
                } else {
                    i += 1;
                }
            }
        }

        Ok(self.text_encoder.forward_embeds(&embeds)?.0)
    }

    /// Generate. Returns `[1, 3, height, width]` in `[-1, 1]`.
    pub fn run(&self, cfg: &Txt2ImgConfig) -> Result<Tensor, PipelineError> {
        self.run_with_progress(cfg, &mut |_| {})
    }

    /// [`Self::run`], reporting progress after each step.
    pub fn run_with_progress(
        &self,
        cfg: &Txt2ImgConfig,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        self.run_with_latent(cfg, None, progress)
            .map(|(image, _)| image)
    }

    /// Load with an IP-Adapter attached, conditioning on a reference image.
    ///
    /// A separate constructor rather than a builder, because the adapter's
    /// weights go *inside* the UNet's cross-attention layers and so must be
    /// present when it is built — unlike a ControlNet, which sits beside it.
    ///
    /// `image_encoder_dir` holds CLIP's vision tower (`h94/IP-Adapter`'s
    /// `models/image_encoder`); `adapter` is `ip-adapter_sd15.safetensors`.
    pub fn load_with_ip_adapter(
        model_dir: &Path,
        device: &Device,
        adapter: &Path,
        image_encoder_dir: &Path,
    ) -> Result<Self, PipelineError> {
        let adapter = require(adapter.to_path_buf())?;
        let encoder_weights = require(image_encoder_dir.join("model.safetensors"))?;

        let ip_vb = sd_loader::safetensors_var_builder(&[&adapter], DType::F32, device)?;
        let mut pipeline = Self::load_inner_with_ip(model_dir, device, Some(&ip_vb))?;

        let vb = sd_loader::safetensors_var_builder(&[&encoder_weights], DType::F32, device)?;
        let vision = ClipVisionEncoder::new(&ClipVisionConfig::vit_h_14(), vb)?;
        let proj = ImageProjModel::new(1024, 768, NUM_TOKENS, ip_vb.pp("image_proj"))?;
        pipeline.ip = Some((vision, proj));
        Ok(pipeline)
    }

    /// Add a textual-inversion embedding, triggered by its file stem.
    ///
    /// Kilobytes against a checkpoint's gigabytes, which is the point. The
    /// trigger is the file name without its extension, because that is what
    /// every tool that writes these uses and what a user will have been told
    /// to type.
    pub fn with_embedding(mut self, path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(PipelineError::MissingFile(path.to_path_buf()));
        }
        let emb = sd_loader::embedding::Embedding::load(path, &self.device)?;
        self.embeddings.push(emb);
        Ok(self)
    }

    /// Trigger words currently registered.
    pub fn embedding_names(&self) -> Vec<&str> {
        self.embeddings.iter().map(|e| e.name.as_str()).collect()
    }

    /// Load with an AnimateDiff motion adapter attached.
    ///
    /// A separate constructor for the same reason as
    /// [`Self::load_with_ip_adapter`]: the 21 modules go *inside* the UNet's
    /// blocks, so they must be present when it is built.
    ///
    /// Attaching one does nothing on its own — set `Txt2ImgConfig::frames`
    /// above 1, or the temporal attention runs over a sequence of length one
    /// and the modules are close to inert.
    pub fn load_with_motion_adapter(
        model_dir: &Path,
        device: &Device,
        adapter: &Path,
    ) -> Result<Self, PipelineError> {
        let adapter = require(adapter.to_path_buf())?;
        let motion_vb = sd_loader::safetensors_var_builder(&[&adapter], DType::F32, device)?;
        Self::load_all(model_dir, device, None, None, Some(&motion_vb))
    }

    /// Whether an IP-Adapter is attached.
    pub fn has_ip_adapter(&self) -> bool {
        self.ip.is_some()
    }

    /// Turn a reference image into the tokens the UNet attends over.
    ///
    /// `image` is the reference at the tower's 224, from
    /// [`crate::image_io::load_clip_square`]. The [`UnitImage`] is the point:
    /// CLIP wants `[0, 1]` where everything touching a VAE is `[-1, 1]`, and
    /// the signed one is *accepted* here — it just describes a different
    /// picture.
    pub fn image_tokens(&self, image: &UnitImage) -> Result<Tensor, PipelineError> {
        let (vision, proj) = self.ip.as_ref().ok_or(PipelineError::NoIpAdapter)?;
        let embeds = vision.image_embeds(&preprocess(image)?)?;
        Ok(proj.forward(&embeds)?)
    }

    /// Conditioning that also attends over a reference image.
    ///
    /// The image tokens ride on the end of the context; the cross-attention
    /// layers split them off. That is why no other signature here changed.
    pub fn encode_conditioning_with_image(
        &self,
        prompt: &str,
        negative_prompt: &str,
        image: &UnitImage,
    ) -> Result<Conditioning, PipelineError> {
        let base = self.encode_conditioning(prompt, negative_prompt)?;
        let tokens = self.image_tokens(image)?;
        // Both guidance rows get the image, for the same reason both get the
        // control map: guidance contrasts the prompts.
        let doubled = Tensor::cat(&[&tokens, &tokens], 0)?;
        Ok(Conditioning {
            context: Tensor::cat(&[&base.context, &doubled], 1)?,
        })
    }

    /// Generate in two passes: compose at the native size, then add detail.
    ///
    /// Returns `[1, 3, height, width]`. The second pass starts from the
    /// enlarged first-pass latent noised to the sigma `strength` selects, so
    /// it refines rather than replaces — which is what keeps the composition.
    pub fn run_hires(&self, cfg: &HiresConfig) -> Result<Tensor, PipelineError> {
        self.run_hires_with_progress(cfg, &mut |_| {})
    }

    /// [`Self::run_hires`], reporting progress across *both* passes.
    pub fn run_hires_with_progress(
        &self,
        cfg: &HiresConfig,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        if cfg.width % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("width", cfg.width));
        }
        if cfg.height % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("height", cfg.height));
        }
        if cfg.width < cfg.base.width || cfg.height < cfg.base.height {
            return Err(PipelineError::Tensor(sd_tensor::Error::Msg(format!(
                "the second pass is {}x{}, smaller than the first at {}x{} — \
                 hires enlarges",
                cfg.width, cfg.height, cfg.base.width, cfg.base.height
            ))));
        }

        // Pass one, at the size the model composes well at.
        let (_, latent) = self.run_with_latent(&cfg.base, None, progress)?;

        let (lh, lw) = (cfg.height / 8, cfg.width / 8);
        let enlarged = match cfg.upscale {
            Upscale::LatentNearest => latent.upsample_nearest2d(lh, lw)?,
            Upscale::LatentBilinear => latent.interpolate2d(lh, lw)?,
            Upscale::PixelLanczos => {
                // The only mode that pays for a VAE round trip, and it is
                // lossy in both directions — which is the trade for resizing
                // where an upscaler would.
                let image = self.decode(&latent)?;
                let resized =
                    crate::image_io::resize_signed(&image, cfg.width as u32, cfg.height as u32)?;
                self.vae_encoder.encode(&resized)?
            }
        };

        // Pass two: noise the enlarged latent to where `strength` starts and
        // run the tail of the schedule from there.
        let second = Txt2ImgConfig {
            width: cfg.width,
            height: cfg.height,
            ..cfg.base.clone()
        };
        let sigmas = self.sigmas_for(second.sampler, second.steps);
        let start = cfg.strength.start_index(second.steps);
        if start >= second.steps {
            // Strength 0: the first pass, merely enlarged.
            return self.decode(&enlarged);
        }

        // A distinct seed for the second pass, derived from the first, so the
        // two passes do not draw the same noise for different-sized latents.
        let mut rng = SeededRng::new(second.seed.wrapping_add(1));
        let noise = rng.randn((1, 4, lh, lw), &self.device)?;
        let latent = (enlarged + (noise * sigmas[start])?)?;

        let contexts = vec![
            self.encode_conditioning(&second.prompt, &second.negative_prompt)?
                .context,
        ];
        let latent = self.denoise(
            latent,
            Denoise::new(&second, &sigmas[start..], &contexts),
            &mut rng,
            progress,
        )?;
        self.decode(&latent)
    }

    /// Edit an image by instruction.
    ///
    /// **Three predictions per step, not two.** Ordinary guidance contrasts a
    /// prompt against nothing; this contrasts three things — the instruction
    /// with the image, the image with nothing — so that text adherence and
    /// image fidelity can be traded independently:
    ///
    /// ```text
    ///   pred = uncond
    ///        + text_scale  * (text  - image)
    ///        + image_scale * (image - uncond)
    /// ```
    ///
    /// The batch rows are `[text+image, uncond+image, uncond+zeros]`, and the
    /// third row's *zeroed* image latent is what makes the middle term mean
    /// "what the image contributes" rather than "what the prompt contributes".
    pub fn run_instruct(&self, cfg: &InstructConfig) -> Result<Tensor, PipelineError> {
        self.run_instruct_with_progress(cfg, &mut |_| {})
    }

    /// [`Self::run_instruct`], reporting progress after each step.
    pub fn run_instruct_with_progress(
        &self,
        cfg: &InstructConfig,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        let base = &cfg.base;
        if base.width % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("width", base.width));
        }
        if base.height % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("height", base.height));
        }
        if base.steps == 0 {
            return Err(PipelineError::NoSteps);
        }

        let cond = self.encode(&base.prompt)?;
        let uncond = self.encode(&base.negative_prompt)?;
        // Instruction, then uncond twice — matching the three rows below.
        let context = Tensor::cat(&[&cond, &uncond, &uncond], 0)?;

        let image = crate::image_io::load_image(
            &cfg.init_image,
            base.width as u32,
            base.height as u32,
            &self.device,
        )?;
        // **Not scaled by 0.18215.** Every other latent in this crate is; this
        // one is not, because InstructPix2Pix was trained on the unscaled
        // encoder output. Scaling it multiplies the conditioning by 5.5 and
        // returns a plausible image that ignores the source.
        let (image_latents, _) = self.vae_encoder.encode_dist(&image)?;
        let zeros = image_latents.zeros_like()?;
        // The third row sees no image at all, which is what makes it the true
        // unconditional.
        let image_rows = Tensor::cat(&[&image_latents, &image_latents, &zeros], 0)?;

        let sigmas = self.sigmas_for(base.sampler, base.steps);
        let (lh, lw) = (base.height / 8, base.width / 8);
        let mut rng = SeededRng::new(base.seed);
        let mut latent = (rng.randn((1, 4, lh, lw), &self.device)? * sigmas[0])?;

        let steps = sigmas.len().saturating_sub(1);
        let mut dpm = DpmSolverPlusPlus2M::new();
        for i in 0..steps {
            if base.cancel.as_ref().is_some_and(Cancel::is_cancelled) {
                return Err(PipelineError::Cancelled {
                    completed: i,
                    total: steps,
                });
            }
            let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);

            let latent_in = Tensor::cat(&[&latent, &latent, &latent], 0)?;
            let latent_in = (latent_in / (sigma * sigma + 1.0).sqrt())?;
            // The source image joins on the **channel** axis, which is what
            // the extra four input channels are for.
            let latent_in = Tensor::cat(&[&latent_in, &image_rows], 1)?;

            let t = sigma_to_timestep(&self.schedule, sigma);
            let timestep = Tensor::from_vec(vec![t as f32; 3], 3, &self.device)?;
            let out = self.unet.forward(&latent_in, &timestep, &context)?;

            let (text, img, uncond_out) = (
                out.narrow(0, 0, 1)?,
                out.narrow(0, 1, 1)?,
                out.narrow(0, 2, 1)?,
            );
            let noise_pred = ((&uncond_out + ((&text - &img)? * base.cfg_scale)?)?
                + ((&img - &uncond_out)? * cfg.image_guidance)?)?;

            let denoised = self.prediction.denoise(&latent, &noise_pred, sigma)?;
            latent = match base.sampler {
                SamplerKind::EulerAncestral => {
                    let noise = rng.randn((1, 4, lh, lw), &self.device)?;
                    euler_ancestral_step(&latent, &denoised, sigma, sigma_next, &noise)?
                }
                SamplerKind::DpmPlusPlus2M => dpm.step(&latent, &denoised, sigma, sigma_next)?,
                SamplerKind::Lcm => {
                    let noise = rng.randn((1, 4, lh, lw), &self.device)?;
                    lcm_step(&latent, &denoised, sigma, sigma_next, t, &noise)?
                }
            };
            progress(Progress {
                step: i + 1,
                total: steps,
                sigma,
                denoised: &denoised,
                // This loop has no cache, so every step ran the model.
                evaluated: i + 1,
            });
        }
        self.decode(&latent)
    }

    /// Whether this checkpoint conditions on a CLIP image embedding.
    pub fn is_unclip(&self) -> bool {
        self.unclip.is_some()
    }

    /// Whether the prior is attached, so text alone can drive an unCLIP run.
    pub fn has_prior(&self) -> bool {
        self.prior.is_some()
    }

    /// Attach the prior, letting [`Self::run_unclip`] work without a reference
    /// image.
    ///
    /// Opt-in and not part of `load`, because it is 4.6 GB — a whole second
    /// diffusion model plus its own text encoder — and an image-variation run
    /// never touches it.
    ///
    /// **The prior and the image half must come from the same checkpoint.**
    /// Karlo's prior emits a 768-wide ViT-L embedding and there are unCLIP
    /// checkpoints built on a 1024-wide ViT-H one; the published
    /// `stable-diffusion-2-1-unclip-t2i-h` in fact pairs the two and cannot
    /// run. The widths are checked here rather than left to fail inside the
    /// augmentation.
    pub fn with_prior(mut self, model_dir: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let dir = model_dir.as_ref();
        let stack = self.unclip.as_ref().ok_or(PipelineError::NoUnclip)?;

        let tokenizer_path = dir.join("prior_tokenizer/tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(PipelineError::MissingUnclipPart(tokenizer_path));
        }
        let encoder_path = require_unclip(dir.join("prior_text_encoder/model.safetensors"))?;
        let prior_path = require_unclip(dir.join("prior/diffusion_pytorch_model.safetensors"))?;

        let weights = sd_loader::resident_bytes(&[&encoder_path, &prior_path], DType::F32)?;
        sd_tensor::sysmem::check_headroom(weights, "attaching the unCLIP prior")?;

        let tokenizer = ClipTokenizer::from_file(&tokenizer_path)?;
        // The prior's own encoder is CLIP-L **with** a projection head, which
        // SD 1.5's plain `CLIPTextModel` does not have — so this is
        // `sd15()` plus `projection_dim`, not `sd15()`.
        let cfg = ClipTextConfig {
            projection_dim: Some(PriorConfig::karlo().embedding_dim),
            ..ClipTextConfig::sd15()
        };
        let vb = sd_loader::safetensors_var_builder(&[&encoder_path], DType::F32, &self.device)?;
        let text_encoder = ClipTextEncoder::new(&cfg, vb)?;

        let vb = sd_loader::safetensors_var_builder(&[&prior_path], DType::F32, &self.device)?;
        let prior = PriorTransformer::new(&PriorConfig::karlo(), vb)?;

        let want = stack.augmentor.embed_dim();
        let got = prior.config().embedding_dim;
        if got != want {
            return Err(PipelineError::PriorWidth { got, want });
        }
        self.prior = Some(Box::new(PriorStack {
            tokenizer,
            text_encoder,
            prior,
        }));
        Ok(self)
    }

    /// Sample an image embedding from a prompt, using the prior.
    ///
    /// Twenty-five DDPM steps over a 768-vector — a whole diffusion run, and a
    /// cheap one: the thing being denoised is one embedding, not a latent
    /// image, so the whole loop costs less than a single UNet step.
    fn prior_embedding(
        &self,
        prompt: &str,
        negative_prompt: &str,
        cfg: &UnclipConfig,
        rng: &mut SeededRng,
    ) -> Result<Tensor, PipelineError> {
        let stack = self.prior.as_ref().ok_or(PipelineError::NoPrior)?;
        let dim = stack.prior.config().embedding_dim;

        // Both prompts, and their masks. The mask is the part that is easy to
        // skip — every other CLIP consumer in this crate ignores it — and the
        // prior attends over 77 positions of which a short prompt occupies ten.
        let (cond_ids, cond_mask) = self.prior_tokens(stack, prompt)?;
        let (uncond_ids, uncond_mask) = self.prior_tokens(stack, negative_prompt)?;

        // One forward per prompt, not two: the prior wants the sequence *and*
        // the projected pooled vector, and `forward` followed by `pooled`
        // would encode each prompt twice.
        let encode = |ids: &Tensor| -> Result<(Tensor, Tensor), PipelineError> {
            let hidden = stack.text_encoder.forward(ids)?;
            let pooled = stack
                .text_encoder
                .project(&stack.text_encoder.pool(&hidden, ids)?)?
                .ok_or(PipelineError::NoPrior)?;
            Ok((pooled, hidden))
        };
        let (cond_pooled, cond_hidden) = encode(&cond_ids)?;
        let (uncond_pooled, uncond_hidden) = encode(&uncond_ids)?;

        // Uncond first, then cond — the same layout the image half uses.
        let proj = Tensor::cat(&[&uncond_pooled, &cond_pooled], 0)?;
        let hidden = Tensor::cat(&[&uncond_hidden, &cond_hidden], 0)?;
        let mask = Tensor::cat(&[&uncond_mask, &cond_mask], 0)?;

        let scheduler = PriorScheduler::new(cfg.prior_steps.max(1));
        // Unit variance, unlike a latent: the prior's schedule starts at
        // alpha_cumprod near zero, so the noise *is* the starting point rather
        // than something scaled onto one.
        let mut latents = rng.randn((1, dim), &self.device)?;

        let total = scheduler.timesteps().len();
        for (done, &t) in scheduler.timesteps().iter().enumerate() {
            if cfg.base.cancel.as_ref().is_some_and(Cancel::is_cancelled) {
                // The prior's own step count, which is not the image half's —
                // a caller that cancels during the prior should be told where
                // it actually stopped rather than "0 of 20".
                return Err(PipelineError::Cancelled {
                    completed: done,
                    total,
                });
            }
            let doubled = Tensor::cat(&[&latents, &latents], 0)?;
            let timestep = Tensor::from_vec(vec![t as f32; 2], 2, &self.device)?;
            let predicted =
                stack
                    .prior
                    .forward(&doubled, &timestep, &proj, &hidden, Some(&mask))?;
            let uncond = predicted.narrow(0, 0, 1)?;
            let cond = predicted.narrow(0, 1, 1)?;
            let guided = (&uncond + ((cond - &uncond)? * cfg.prior_guidance)?)?;
            let noise = rng.randn((1, dim), &self.device)?;
            latents = scheduler.step(&guided, t, &latents, &noise)?;
        }
        // Back into CLIP's own units. Without this the image half conditions
        // on a whitened vector and produces a washed, generic picture.
        Ok(stack.prior.post_process(&latents)?)
    }

    /// Tokenize for the prior, returning ids and the attention mask.
    ///
    /// The mask is `1` up to and including the EOS and `0` for the padding
    /// after it — which is exactly what `CLIPTokenizer` reports and what this
    /// crate has never needed until now.
    fn prior_tokens(
        &self,
        stack: &PriorStack,
        prompt: &str,
    ) -> Result<(Tensor, Tensor), PipelineError> {
        let ids = stack.tokenizer.encode(prompt)?;
        // Content plus BOS and EOS. The EOS is attended over — it is where the
        // pooled embedding is read from — so the mask covers it and stops at
        // the padding that follows.
        let used = stack.tokenizer.content_token_count(prompt)? + 2;
        let len = ids.len();
        let mask: Vec<f32> = (0..len)
            .map(|i| if i < used.min(len) { 1.0 } else { 0.0 })
            .collect();
        Ok((
            Tensor::from_vec(ids, (1, len), &self.device)?,
            Tensor::from_vec(mask, (1, len), &self.device)?,
        ))
    }

    /// The augmented image embedding a reference image produces.
    ///
    /// `image` is the reference at the tower's 224, the same convention
    /// [`Self::image_tokens`] takes. `noise` is `[1, embed_dim]`, supplied so
    /// the caller owns its seed sequence.
    ///
    /// Exposed because it is the only interesting intermediate here: an
    /// unCLIP run is an ordinary SD 2.x run with this vector added into every
    /// timestep embedding, so a caller that wants to interpolate between two
    /// references, or reuse one across a sequence, needs the vector rather
    /// than the pipeline's opinion of it.
    pub fn image_conditioning(
        &self,
        image: &UnitImage,
        noise_level: usize,
        noise: &Tensor,
    ) -> Result<Tensor, PipelineError> {
        let stack = self.unclip.as_ref().ok_or(PipelineError::NoUnclip)?;
        let vision = stack.vision.as_ref().ok_or(PipelineError::NoImageEncoder)?;
        let embeds = vision.image_embeds(&preprocess(image)?)?;
        Ok(stack.augmentor.augment(&embeds, noise_level, noise)?)
    }

    /// Generate an image that shares a reference's subject, not its pixels.
    ///
    /// Needs an unCLIP checkpoint. An ordinary SD 2.x UNet has no
    /// `class_embedding`, so there would be nowhere for the embedding to go
    /// and the reference would be silently ignored — which is refused rather
    /// than rendered.
    pub fn run_unclip(&self, cfg: &UnclipConfig) -> Result<Tensor, PipelineError> {
        self.run_unclip_with_progress(cfg, &mut |_| {})
    }

    /// [`Self::run_unclip`], reporting progress after each step.
    pub fn run_unclip_with_progress(
        &self,
        cfg: &UnclipConfig,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        let base = &cfg.base;
        if base.width % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("width", base.width));
        }
        if base.height % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("height", base.height));
        }
        if base.steps == 0 {
            return Err(PipelineError::NoSteps);
        }
        let augmentor = &self
            .unclip
            .as_ref()
            .ok_or(PipelineError::NoUnclip)?
            .augmentor;

        // Drawn from the seed before the latent is, so a run is reproducible
        // and so that changing `--noise-level` alone changes only how much of
        // this noise is mixed in rather than which noise it is.
        let mut rng = SeededRng::new(base.seed);

        // The two halves of unCLIP. A reference image goes through the vision
        // tower; without one the prior samples an embedding from the prompt.
        // Everything after this point is identical — which is the whole point
        // of the architecture, and why the text-to-image variant costs a front
        // end rather than a pipeline.
        let embeds = match &cfg.init_image {
            Some(path) => {
                // The tower's own input size, not the output size. unCLIP
                // reads the reference at 224 whatever it is asked to draw at,
                // so a 4096px reference costs nothing extra here.
                let image = crate::image_io::load_clip_square(path, 224, &self.device)?;
                let stack = self.unclip.as_ref().ok_or(PipelineError::NoUnclip)?;
                let vision = stack.vision.as_ref().ok_or(PipelineError::NoImageEncoder)?;
                vision.image_embeds(&preprocess(&image)?)?
            }
            None => self.prior_embedding(&base.prompt, &base.negative_prompt, cfg, &mut rng)?,
        };
        let noise = rng.randn((1, augmentor.embed_dim()), &self.device)?;
        let conditioning = augmentor.augment(&embeds, cfg.noise_level, &noise)?;

        let frames = base.frames.max(1);
        // Uncond rows first, then cond — matching how `latent_in` and the
        // context are concatenated in the loop. The unconditional row is
        // **zeros of the full width**, which is what diffusers hands the model
        // for "no image": an augmented zero embedding would still carry the
        // noise level's sinusoid and mean something.
        let uncond = augmentor.unconditional(frames, conditioning.dtype(), &self.device)?;
        let cond = repeat_rows(&conditioning, frames)?;
        let class_labels = Tensor::cat(&[&uncond, &cond], 0)?;

        let context = self
            .encode_conditioning(&base.prompt, &base.negative_prompt)?
            .context;
        let context = repeat_per_frame(&context, frames)?;

        let sigmas = self.sigmas_for(base.sampler, base.steps);
        let (lh, lw) = (base.height / 8, base.width / 8);
        let latent = (rng.randn((frames, 4, lh, lw), &self.device)? * sigmas[0])?;

        let latent = self.denoise(
            latent,
            Denoise::new(base, &sigmas, std::slice::from_ref(&context))
                .conditioned_on_image(class_labels),
            &mut rng,
            progress,
        )?;
        self.decode(&latent)
    }

    /// Generate with objects placed by bounding box.
    ///
    /// Needs a GLIGEN checkpoint — an ordinary SD 1.5 UNet has no fusers, and
    /// grounding it would silently do nothing, so that is refused rather than
    /// ignored.
    pub fn run_grounded(&self, cfg: &GroundingConfig) -> Result<Tensor, PipelineError> {
        self.run_grounded_with_progress(cfg, &mut |_| {})
    }

    /// [`Self::run_grounded`], reporting progress after each step.
    pub fn run_grounded_with_progress(
        &self,
        cfg: &GroundingConfig,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        let base = &cfg.base;
        if base.width % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("width", base.width));
        }
        if base.height % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("height", base.height));
        }
        if base.steps == 0 {
            return Err(PipelineError::NoSteps);
        }
        let position_net = self
            .position_net
            .as_ref()
            .ok_or(PipelineError::NoGrounding)?;
        if cfg.boxes.is_empty() {
            return Err(PipelineError::Tensor(sd_tensor::Error::Msg(
                "run_grounded needs at least one box".to_string(),
            )));
        }

        // One row per box: the *pooled* phrase embedding, not the 77-token
        // sequence. GLIGEN grounds on what a phrase means as a whole.
        let n = cfg.boxes.len();
        let mut phrase_rows = Vec::with_capacity(n);
        let mut coords = Vec::with_capacity(n * 4);
        for grounded in &cfg.boxes {
            let ids = self.tokenizer.encode(&grounded.phrase)?;
            let ids = Tensor::from_vec(ids, (1, self.tokenizer.max_length()), &self.device)?;
            phrase_rows.push(self.text_encoder.pooled_hidden(&ids)?);
            coords.extend_from_slice(&grounded.bbox);
        }
        let phrases = Tensor::cat(&phrase_rows, 0)?.reshape((1, n, 768))?;
        let boxes = Tensor::from_vec(coords, (1, n, 4), &self.device)?;
        // Every slot is real here, so every mask is 1. The learned nulls exist
        // for callers batching a fixed number of slots.
        let masks = Tensor::ones((1, n), DType::F32, &self.device)?;

        let objs = position_net.forward(&boxes, &masks, &phrases)?;
        // Doubled for the guidance batch, like every other conditioning here.
        let objs = Tensor::cat(&[&objs, &objs], 0)?;

        let cond = self.encode(&base.prompt)?;
        let uncond = self.encode(&base.negative_prompt)?;
        let context = Tensor::cat(&[&uncond, &cond], 0)?;

        let sigmas = self.sigmas_for(base.sampler, base.steps);
        let (lh, lw) = (base.height / 8, base.width / 8);
        let mut rng = SeededRng::new(base.seed);
        let latent = (rng.randn((1, 4, lh, lw), &self.device)? * sigmas[0])?;

        // Grounded for the first fraction, free for the rest. Two calls rather
        // than a flag inside the loop: the guard *is* the mechanism, and
        // dropping it is how grounding turns off.
        let grounded_steps = ((base.steps as f64 * cfg.grounding_fraction.clamp(0.0, 1.0)).round()
            as usize)
            .min(base.steps);
        let latent = {
            let _guard = sd_models::unet::gligen::with_objs(objs);
            self.denoise(
                latent,
                Denoise::new(
                    base,
                    &sigmas[..=grounded_steps],
                    std::slice::from_ref(&context),
                ),
                &mut rng,
                progress,
            )?
        };
        let latent = if grounded_steps < base.steps {
            self.denoise(
                latent,
                Denoise::new(
                    base,
                    &sigmas[grounded_steps..],
                    std::slice::from_ref(&context),
                ),
                &mut rng,
                progress,
            )?
        } else {
            latent
        };
        self.decode(&latent)
    }

    /// Generate with different prompts in different regions.
    ///
    /// Each region contributes a noise prediction of its own, blended by its
    /// mask. Where masks overlap they average; where none covers, the base
    /// prompt applies alone.
    ///
    /// **Costs one UNet call per region per step**, on top of the base — three
    /// regions is four times the work. That is inherent to conditioning
    /// spatially rather than a shortcoming of this implementation, and it is
    /// why the base prediction is computed once and reused.
    pub fn run_area(&self, cfg: &AreaConfig) -> Result<Tensor, PipelineError> {
        self.run_area_with_progress(cfg, &mut |_| {})
    }

    /// [`Self::run_area`], reporting progress after each step.
    pub fn run_area_with_progress(
        &self,
        cfg: &AreaConfig,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        let base = &cfg.base;
        if base.width % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("width", base.width));
        }
        if base.height % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("height", base.height));
        }
        if base.steps == 0 {
            return Err(PipelineError::NoSteps);
        }

        let frames = base.frames.max(1);
        let (lh, lw) = (base.height / 8, base.width / 8);

        let regions = cfg
            .regions
            .iter()
            .enumerate()
            .map(|(i, region)| {
                let d = region.mask.dims4()?;
                if d.2 != base.height || d.3 != base.width {
                    return Err(PipelineError::Tensor(sd_tensor::Error::Msg(format!(
                        "region {i} mask is {}x{}, expected {}x{} — masks are at pixel \
                         resolution, not latent",
                        d.2, d.3, base.height, base.width
                    ))));
                }
                Ok((
                    area_mask(&region.mask, lh, lw)?,
                    repeat_per_frame(&region.conditioning.context, frames)?,
                ))
            })
            .collect::<Result<Vec<_>, PipelineError>>()?;

        let context = repeat_per_frame(
            &self
                .encode_conditioning(&base.prompt, &base.negative_prompt)?
                .context,
            frames,
        )?;

        let sigmas = self.sigmas_for(base.sampler, base.steps);
        let mut rng = SeededRng::new(base.seed);
        let latent = (rng.randn((frames, 4, lh, lw), &self.device)? * sigmas[0])?;

        let contexts = [context];
        let latent = self.denoise(
            latent,
            Denoise::new(base, &sigmas, &contexts).over_areas(Areas { regions }),
            &mut rng,
            progress,
        )?;
        self.decode(&latent)
    }

    /// Encode a prompt pair once, for reuse across a sequence.
    ///
    /// Uncond first, then cond — the order the guidance split in the sampling
    /// loop expects. Reversing exactly one of the two inverts guidance and
    /// produces the opposite of the prompt, which is a confusing symptom to
    /// debug from the image, so the pair is built here rather than by callers.
    pub fn encode_conditioning(
        &self,
        prompt: &str,
        negative_prompt: &str,
    ) -> Result<Conditioning, PipelineError> {
        let cond = self.encode(prompt)?;
        let uncond = self.encode(negative_prompt)?;
        Ok(Conditioning {
            context: Tensor::cat(&[&uncond, &cond], 0)?,
        })
    }

    /// Generate with the conditioning chosen per step.
    ///
    /// `select` is handed `(step, total)` — 1-based step — and returns an index
    /// into `conditioning`. Out-of-range indices are clamped to the last entry
    /// rather than erroring mid-run, since a run is expensive and the
    /// alternative is losing it to an off-by-one in a caller's schedule.
    ///
    /// This is what makes a published class of technique reachable. Gating a
    /// negative prompt to a middle window of the schedule improves
    /// object-removal success from 65.1 % to 80.4 % (Ban et al., ECCV 2024,
    /// arXiv:2406.02965); the same work finds a negative applied only in the
    /// first few steps can *generate* the thing it names. Neither is
    /// expressible with one fixed conditioning.
    ///
    /// A single-entry slice with `|_, _| 0` is exactly
    /// [`Self::run_with_latent`], and there is a test that says so.
    pub fn run_conditioned(
        &self,
        cfg: &Txt2ImgConfig,
        conditioning: &[Conditioning],
        select: &mut dyn FnMut(usize, usize) -> usize,
        latent: Option<&Tensor>,
        progress: ProgressFn<'_>,
    ) -> Result<(Tensor, Tensor), PipelineError> {
        if conditioning.is_empty() {
            return Err(PipelineError::Tensor(sd_tensor::Error::Msg(
                "run_conditioned needs at least one conditioning".to_string(),
            )));
        }
        self.generate(cfg, Some((conditioning, select)), latent, progress)
    }

    /// The latent a config would start from, without generating anything.
    ///
    /// Exposed so a caller can perturb it. Frame-to-frame coherence methods
    /// are almost all about controlling the *noise* rather than the seed —
    /// a shared initial latent across frames, correlated rather than
    /// independent noise, interpolation between keyframes — and none of them
    /// are reachable through a seed alone.
    ///
    /// Already scaled by the first sigma, so it is ready to hand back to
    /// [`Self::run_with_latent`] unchanged.
    pub fn initial_latent(&self, cfg: &Txt2ImgConfig) -> Result<Tensor, PipelineError> {
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);
        let sigmas = self.sigmas_for(cfg.sampler, cfg.steps);
        let mut rng = SeededRng::new(cfg.seed);
        Ok((rng.randn((1, 4, lh, lw), &self.device)? * sigmas[0])?)
    }

    /// Record, per step, how far each candidate predictor moved against how
    /// far the model's output actually moved.
    ///
    /// Caching is **off** throughout — the point is to observe the true
    /// output-change sequence, which a cached run by definition does not have.
    /// Used by `--example cache_fit` to fit the rescaling polynomial on this
    /// model rather than borrowing one whose provenance cannot be checked.
    pub fn cache_calibration(
        &self,
        cfg: &Txt2ImgConfig,
    ) -> Result<Vec<CalibrationPoint>, PipelineError> {
        if self.unet.in_channels() != 4 || self.unet.takes_class_labels() {
            return Err(PipelineError::NeedsInstruct);
        }
        let context = self
            .encode_conditioning(&cfg.prompt, &cfg.negative_prompt)?
            .context;
        let sigmas = self.sigmas_for(cfg.sampler, cfg.steps);
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);
        let mut rng = SeededRng::new(cfg.seed);
        let mut latent = (rng.randn((1, 4, lh, lw), &self.device)? * sigmas[0])?;

        let steps = sigmas.len().saturating_sub(1);
        let mut dpm = DpmSolverPlusPlus2M::new();
        let shape = LatentShape {
            frames: 1,
            height: lh,
            width: lw,
        };

        let mut series = Vec::with_capacity(steps);
        let (mut last_scaled, mut last_temb, mut last_out): (
            Option<Tensor>,
            Option<Tensor>,
            Option<Tensor>,
        ) = (None, None, None);

        for i in 0..steps {
            let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
            let t = sigma_to_timestep(&self.schedule, sigma);
            let timestep = Tensor::from_vec(vec![t as f32; 2], 2, &self.device)?;

            // Exactly what the loop feeds the UNet, so the predictor is
            // measured on the tensor it would really see.
            let latent_in = Tensor::cat(&[&latent, &latent], 0)?;
            let scaled = (latent_in / (sigma * sigma + 1.0).sqrt())?;
            let temb = self.unet.timestep_features(&timestep)?;

            let out = self.unet.forward(&scaled, &timestep, &context)?;
            let uncond = out.narrow(0, 0, 1)?;
            let cond = out.narrow(0, 1, 1)?;
            let noise_pred = (&uncond + ((cond - &uncond)? * cfg.cfg_scale)?)?;

            if let (Some(ps), Some(pt), Some(po)) = (&last_scaled, &last_temb, &last_out) {
                series.push(CalibrationPoint {
                    latent: relative_l1(&scaled, ps)?,
                    temb: relative_l1(&temb, pt)?,
                    output: relative_l1(&noise_pred, po)?,
                });
            }
            last_scaled = Some(scaled);
            last_temb = Some(temb);
            last_out = Some(noise_pred.clone());

            let denoised = self.prediction.denoise(&latent, &noise_pred, sigma)?;
            latent = self.step(
                cfg,
                &mut dpm,
                StepPoint {
                    latent: &latent,
                    denoised: &denoised,
                    sigma,
                    sigma_next,
                    t,
                },
                &mut rng,
                shape,
            )?;
        }
        Ok(series)
    }

    /// Generate from an explicit starting latent, returning the final one too.
    ///
    /// `latent` of `None` draws from the seed, which is exactly what
    /// [`Self::run_with_progress`] does — passing
    /// [`Self::initial_latent`]'s result is equivalent to passing `None`, and
    /// there is a test that says so.
    ///
    /// The returned latent is the *denoised* one, before decoding, so it can
    /// be carried into the next frame or re-decoded at another size.
    ///
    /// Note the seed still drives the sampler's per-step noise even when the
    /// initial latent is supplied: an ancestral sampler draws every step, and
    /// two frames sharing an initial latent but not a seed will still diverge.
    pub fn run_with_latent(
        &self,
        cfg: &Txt2ImgConfig,
        latent: Option<&Tensor>,
        progress: ProgressFn<'_>,
    ) -> Result<(Tensor, Tensor), PipelineError> {
        self.generate(cfg, None, latent, progress)
    }

    #[allow(clippy::type_complexity)]
    fn generate(
        &self,
        cfg: &Txt2ImgConfig,
        conditioning: Option<(&[Conditioning], &mut dyn FnMut(usize, usize) -> usize)>,
        latent: Option<&Tensor>,
        progress: ProgressFn<'_>,
    ) -> Result<(Tensor, Tensor), PipelineError> {
        if cfg.width % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("width", cfg.width));
        }
        if cfg.height % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("height", cfg.height));
        }
        if cfg.steps == 0 {
            return Err(PipelineError::NoSteps);
        }

        // Either the config's own prompts, encoded here, or a caller-supplied
        // set chosen per step.
        let (contexts, select): (Vec<Tensor>, Option<&mut dyn FnMut(usize, usize) -> usize>) =
            match conditioning {
                Some((set, select)) => (
                    set.iter().map(|c| c.context.clone()).collect(),
                    Some(select),
                ),
                None => (
                    vec![
                        self.encode_conditioning(&cfg.prompt, &cfg.negative_prompt)?
                            .context,
                    ],
                    None,
                ),
            };
        // One row per row of the guidance batch. The reference UNet does *not*
        // repeat conditioning across frames — hidden states carry frames on
        // the batch and the text must too. Passing one row where `n` are
        // expected fails inside the spatial cross-attention rather than
        // broadcasting, which is how this was found.
        let contexts = contexts
            .into_iter()
            .map(|c| repeat_per_frame(&c, cfg.frames.max(1)))
            .collect::<Result<Vec<_>, PipelineError>>()?;

        let sigmas = self.sigmas_for(cfg.sampler, cfg.steps);
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);

        // One generator per image, drawn in order: initial latent first, then
        // one noise draw per step. A fresh generator inside the loop would
        // give every step identical noise.
        let mut rng = SeededRng::new(cfg.seed);
        // Scaled by the first sigma — unit-variance noise gives washed-out
        // output. Drawn even when `latent` is supplied, so that the sampler's
        // subsequent draws land in the same sequence either way and a
        // caller-supplied `initial_latent` reproduces the seeded run exactly.
        // An 8-channel UNet cannot be driven from text alone; say so here
        // rather than letting it fail inside a convolution.
        if self.unet.in_channels() != 4 {
            return Err(PipelineError::NeedsInstruct);
        }
        // Nor can one that projects an image embedding into every timestep.
        // Without this it would run — on zeros, which is the guidance batch's
        // unconditional row — and return an image conditioned on nothing.
        if self.unet.takes_class_labels() {
            return Err(PipelineError::NeedsUnclip);
        }
        let frames = cfg.frames.max(1);
        let drawn = (rng.randn((frames, 4, lh, lw), &self.device)? * sigmas[0])?;
        let latent = match latent {
            Some(given) => {
                if given.dims() != drawn.dims() {
                    return Err(PipelineError::Tensor(sd_tensor::Error::Msg(format!(
                        "latent is {:?}, expected {:?} for {}x{}",
                        given.dims(),
                        drawn.dims(),
                        cfg.width,
                        cfg.height
                    ))));
                }
                given.to_device(&self.device)?
            }
            None => drawn,
        };

        let latent = self.denoise(
            latent,
            Denoise::new(cfg, &sigmas, &contexts).selecting(select),
            &mut rng,
            progress,
        )?;

        // `decode_tiled` applies the scaling factor, like `decode`, and falls
        // through to a whole-image decode for latents that already fit — so
        // 512px output is bit-identical to before. Above that it tiles, which
        // is what keeps a 1024px decode inside GPU memory.
        let image = self.decode(&latent)?;
        Ok((image, latent))
    }

    /// Generate inside a mask, leaving everything else alone.
    ///
    /// Latent blending rather than a dedicated inpainting checkpoint: at every
    /// step the region outside the mask is restored to the original, so any
    /// ordinary model can inpaint and no 9-channel UNet is required. The trade
    /// is that the model never *sees* a mask, so it infers the boundary from
    /// context alone — a dedicated checkpoint does better on large holes.
    ///
    /// The untouched region is exact: latent blending alone would return it
    /// through a VAE round trip, which is not lossless, so the result is
    /// composited against the original in pixel space at the end.
    pub fn run_inpaint(&self, cfg: &InpaintConfig) -> Result<Tensor, PipelineError> {
        self.run_inpaint_with_progress(cfg, &mut |_| {})
    }

    /// [`Self::run_inpaint`], reporting progress after each step.
    pub fn run_inpaint_with_progress(
        &self,
        cfg: &InpaintConfig,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        let base = &cfg.base.base;
        if base.width % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("width", base.width));
        }
        if base.height % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("height", base.height));
        }
        if base.steps == 0 {
            return Err(PipelineError::NoSteps);
        }

        let cond = self.encode(&base.prompt)?;
        let uncond = self.encode(&base.negative_prompt)?;
        let context = Tensor::cat(&[&uncond, &cond], 0)?;

        let (w, h) = (base.width as u32, base.height as u32);
        let image = crate::image_io::load_image(&cfg.base.init_image, w, h, &self.device)?;
        let mask_px = crate::image_io::load_mask(&cfg.mask, w, h, &self.device)?;
        let init = self.vae_encoder.encode(&image)?;
        let mask = latent_mask(&mask_px, base.height / 8, base.width / 8)?;

        let sigmas = self.sigmas_for(base.sampler, base.steps);
        let start = cfg.base.strength.start_index(base.steps);
        if start >= base.steps {
            // Strength 0 repaints nothing, so the original is the answer —
            // and returning it directly avoids a pointless VAE round trip.
            return Ok(image);
        }

        let mut rng = SeededRng::new(base.seed);
        let (lh, lw) = (base.height / 8, base.width / 8);
        let noise = rng.randn((1, 4, lh, lw), &self.device)?;
        let latent = (&init + (noise * sigmas[start])?)?;

        let latent = self.denoise(
            latent,
            Denoise::new(base, &sigmas[start..], std::slice::from_ref(&context)).keeping(Some(
                Keep {
                    mask: &mask,
                    init: &init,
                },
            )),
            &mut rng,
            progress,
        )?;
        let decoded = self.decode(&latent)?;
        Ok(crate::image_io::composite(&decoded, &image, &mask_px)?)
    }

    /// The sigma ladder a sampler wants.
    ///
    /// LCM does not take an even spread: it visits the subset of timesteps its
    /// distillation used, so it builds its own ladder. Everything else shares
    /// the usual one.
    fn sigmas_for(&self, sampler: SamplerKind, steps: usize) -> Vec<f64> {
        match sampler {
            SamplerKind::Lcm => lcm_sigmas(
                &self.schedule,
                &lcm_timesteps(
                    self.schedule.alphas_cumprod.len(),
                    sd_sample::ORIGINAL_INFERENCE_STEPS,
                    steps,
                ),
            ),
            _ => sigmas_for_steps(&self.schedule, steps),
        }
    }

    /// The sampling loop, shared by txt2img and img2img.
    ///
    /// `sigmas` is a full ladder of `n + 1` boundaries; img2img passes a
    /// suffix of one. `rng` is threaded in rather than created here so the
    /// caller controls draw order, which is what makes a seed reproducible.
    fn denoise(
        &self,
        latent: Tensor,
        run: Denoise<'_>,
        rng: &mut SeededRng,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        self.denoise_inner(latent, run, rng, progress)
    }

    /// One sampler step, shared by the ordinary path and the cached one.
    ///
    /// Extracted rather than duplicated: a cached step differs only in *where
    /// the prediction came from*, and two copies of the sampler match would
    /// drift the moment a sampler is added.
    fn step(
        &self,
        cfg: &Txt2ImgConfig,
        dpm: &mut DpmSolverPlusPlus2M,
        at: StepPoint<'_>,
        rng: &mut SeededRng,
        shape: LatentShape,
    ) -> Result<Tensor, PipelineError> {
        let StepPoint {
            latent,
            denoised,
            sigma,
            sigma_next,
            t,
        } = at;
        let draw = shape.dims();
        Ok(match cfg.sampler {
            SamplerKind::EulerAncestral => {
                let noise = rng.randn(draw, &self.device)?;
                euler_ancestral_step(latent, denoised, sigma, sigma_next, &noise)?
            }
            SamplerKind::DpmPlusPlus2M => dpm.step(latent, denoised, sigma, sigma_next)?,
            SamplerKind::Lcm => {
                // Fresh noise each step: LCM re-noises rather than
                // integrating, so a reused draw correlates the steps.
                let noise = rng.randn(draw, &self.device)?;
                lcm_step(latent, denoised, sigma, sigma_next, t, &noise)?
            }
        })
    }

    fn denoise_inner(
        &self,
        mut latent: Tensor,
        run: Denoise<'_>,
        rng: &mut SeededRng,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        let Denoise {
            cfg,
            sigmas,
            contexts,
            mut select,
            keep,
            control,
            areas,
            class_labels,
        } = run;
        // Caching is only meaningful where consecutive predictions are
        // similar, and an ancestral sampler guarantees they are not. Refused
        // rather than silently ignored: a caller who asked for caching and got
        // none would wonder why, and one who got it anyway would get speckle.
        if cfg.cache_threshold > 0.0 {
            if let Some(sampler) = ancestral_name(cfg.sampler) {
                return Err(PipelineError::CacheNeedsDeterministicSampler { sampler });
            }
        }
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);
        let steps = sigmas.len().saturating_sub(1);
        let mut dpm = DpmSolverPlusPlus2M::new();
        // Read from the latent rather than the config: the latent is the one
        // thing every path here already agrees on, and a caller supplying its
        // own through `run_with_latent` sets the count by doing so.
        let frames = latent.dim(0)?;
        let shape = LatentShape {
            frames,
            height: lh,
            width: lw,
        };
        // Reused prediction, and the drift accumulated since it was computed.
        let mut cached: Option<Tensor> = None;
        // The predictor's own state: the last timestep embedding, and the
        // accumulated *predicted* relative change in the model's output.
        let mut previous_temb: Option<Tensor> = None;
        let mut drift = 0f64;
        let mut evaluated = 0usize;
        // Motion modules need it too, and they are four levels down.
        let _frames_guard = sd_models::unet::motion::with_frames(frames);

        for i in 0..steps {
            // Checked before the work, so a cancel between steps costs nothing
            // and the error names how far it got — a caller showing progress
            // wants to know whether it was step 1 or step 19.
            if cfg.cancel.as_ref().is_some_and(Cancel::is_cancelled) {
                return Err(PipelineError::Cancelled {
                    completed: i,
                    total: steps,
                });
            }
            let sigma = sigmas[i];
            let sigma_next = sigmas[i + 1];

            // Classifier-free guidance: run both conditionings in one batch.
            let latent_in = Tensor::cat(&[&latent, &latent], 0)?;
            // k-diffusion input scaling. Omitting it gives noisy, oversaturated
            // results.
            let latent_in = (latent_in / (sigma * sigma + 1.0).sqrt())?;

            // Clamped rather than checked: a run is expensive, and losing one
            // to an off-by-one in a caller's schedule is a worse outcome than
            // silently reusing the last entry.
            let context = match select.as_mut() {
                Some(pick) => &contexts[pick(i + 1, steps).min(contexts.len() - 1)],
                None => &contexts[0],
            };

            let t = sigma_to_timestep(&self.schedule, sigma);
            // One entry per row of the guidance batch: 2 * frames, not 2.
            let timestep = Tensor::from_vec(vec![t as f32; 2 * frames], 2 * frames, &self.device)?;

            // **The predictor.** How far the timestep embedding moved, rescaled
            // through a fitted polynomial into an estimate of how far the
            // model's *output* will move, and accumulated. Two small matmuls
            // against a forward pass, so a skipped step costs essentially
            // nothing to decide.
            //
            // The rescaling is the whole idea: the raw embedding distance is
            // not comparable to an output distance, and without it the
            // threshold has no units. See `CACHE_RESCALE`.
            if cfg.cache_threshold > 0.0 {
                let temb = self.unet.timestep_features(&timestep)?;
                if let Some(prev) = &previous_temb {
                    if cached.is_some() {
                        let moved = relative_l1(&temb, prev)?;
                        drift += cache_rescale(moved);
                    }
                }
                previous_temb = Some(temb);
            }

            // Reuse, or evaluate and cache. The *last* step always evaluates:
            // it lands the image, and a reused prediction there shows up
            // directly in the output rather than being corrected later.
            let reuse = cfg.cache_threshold > 0.0
                && cached.is_some()
                && drift < cfg.cache_threshold
                && i + 1 < steps;
            if reuse {
                let noise_pred = cached.clone().expect("checked");
                let denoised = self.prediction.denoise(&latent, &noise_pred, sigma)?;
                latent = self.step(
                    cfg,
                    &mut dpm,
                    StepPoint {
                        latent: &latent,
                        denoised: &denoised,
                        sigma,
                        sigma_next,
                        t,
                    },
                    rng,
                    shape,
                )?;
                progress(Progress {
                    step: i + 1,
                    total: steps,
                    sigma,
                    denoised: &denoised,
                    evaluated,
                });
                continue;
            }
            evaluated += 1;
            drift = 0.0;

            let out = match &control {
                Some(hints) if !self.controlnets.is_empty() => {
                    // Each ControlNet sees the same scaled latent and timestep
                    // the UNet does. Feeding it the unscaled latent is a
                    // natural mistake that produces corrections of plausible
                    // magnitude for the wrong noise level.
                    //
                    // Several ControlNets **sum**, which is what diffusers
                    // does: they were each trained against the same frozen
                    // base, so their corrections are independent additions to
                    // it rather than alternatives to choose between.
                    let mut total: Option<sd_models::controlnet::Control> = None;
                    for (net, (hint, scale)) in self.controlnets.iter().zip(&hints.maps) {
                        let c = net.forward(&latent_in, &timestep, context, hint, *scale)?;
                        total = Some(match total {
                            None => c,
                            Some(acc) => sd_models::controlnet::Control {
                                down: acc
                                    .down
                                    .iter()
                                    .zip(&c.down)
                                    .map(|(a, b)| a + b)
                                    .collect::<sd_tensor::Result<Vec<_>>>()?,
                                mid: (acc.mid + c.mid)?,
                            },
                        });
                    }
                    match total {
                        Some(c) => self
                            .unet
                            .forward_controlled(&latent_in, &timestep, context, &c.down, &c.mid)?,
                        None => self.unet.forward(&latent_in, &timestep, context)?,
                    }
                }
                // The image embedding is fixed for the whole run — it does not
                // depend on the step or the latent — so it is built once by
                // the caller and handed in already doubled for the guidance
                // batch, uncond rows first.
                _ => match &class_labels {
                    Some(labels) => self
                        .unet
                        .forward_unclip(&latent_in, &timestep, context, labels)?,
                    None => self.unet.forward(&latent_in, &timestep, context)?,
                },
            };
            // The guidance batch is [uncond frames..., cond frames...], not
            // interleaved — matching how `latent_in` was concatenated and how
            // the context is laid out. Splitting it the other way runs and
            // guides each frame by another frame's conditioning.
            let out_uncond = out.narrow(0, 0, frames)?;
            let out_cond = out.narrow(0, frames, frames)?;
            let noise_pred = (&out_uncond + ((out_cond - &out_uncond)? * cfg.cfg_scale)?)?;
            if cfg.cache_threshold > 0.0 {
                cached = Some(noise_pred.clone());
            }

            // Regional prompts. Each region is a second prediction from its own
            // conditioning, blended in where its mask says so.
            //
            // A prediction per region per step, so `n` regions cost `n + 1`
            // UNet calls — the honest price of the feature, and the reason the
            // base prediction is reused rather than recomputed.
            let noise_pred = match &areas {
                Some(areas) if !areas.regions.is_empty() => {
                    let mut weighted: Option<Tensor> = None;
                    let mut total: Option<Tensor> = None;
                    for (mask, context) in &areas.regions {
                        let region_out = self.unet.forward(&latent_in, &timestep, context)?;
                        let u = region_out.narrow(0, 0, frames)?;
                        let c = region_out.narrow(0, frames, frames)?;
                        let pred = (&u + ((c - &u)? * cfg.cfg_scale)?)?;

                        let contribution = pred.broadcast_mul(mask)?;
                        weighted = Some(match weighted {
                            None => contribution,
                            Some(acc) => (acc + contribution)?,
                        });
                        total = Some(match total {
                            None => mask.clone(),
                            Some(acc) => (acc + mask)?,
                        });
                    }
                    let (weighted, total) = (
                        weighted.expect("at least one region"),
                        total.expect("at least one region"),
                    );
                    // Where masks overlap, average them; where none covers,
                    // fall back to the base prompt entirely. `coverage` is the
                    // total clamped to 1 so a single mask replaces the base
                    // rather than merely outvoting it.
                    let coverage = total.clamp(0.0, 1.0)?;
                    let divisor = total.clamp(1.0, f64::INFINITY)?;
                    let regional = weighted.broadcast_div(&divisor)?;
                    ((noise_pred.broadcast_mul(&(1.0 - &coverage)?)?)
                        + regional.broadcast_mul(&coverage)?)?
                }
                _ => noise_pred,
            };

            // The UNet predicts noise; the samplers want x0.
            let denoised = self.prediction.denoise(&latent, &noise_pred, sigma)?;

            latent = self.step(
                cfg,
                &mut dpm,
                StepPoint {
                    latent: &latent,
                    denoised: &denoised,
                    sigma,
                    sigma_next,
                    t,
                },
                rng,
                shape,
            )?;

            // Restore everything outside the mask to the original, noised to
            // the level the next step expects. Doing this *inside* the loop
            // rather than once at the end is what keeps the model's context
            // honest: it sees the true surroundings at every step, so what it
            // paints actually joins up with them.
            if let Some(k) = &keep {
                let restored = if sigma_next > 0.0 {
                    let n = rng.randn((frames, 4, lh, lw), &self.device)?;
                    (k.init + (n * sigma_next)?)?
                } else {
                    k.init.clone()
                };
                latent =
                    (latent.broadcast_mul(k.mask)? + restored.broadcast_mul(&(1.0 - k.mask)?)?)?;
            }

            progress(Progress {
                step: i + 1,
                total: steps,
                sigma,
                denoised: &denoised,
                evaluated,
            });
        }
        Ok(latent)
    }

    /// Generate from an existing image. Returns `[1, 3, height, width]`.
    pub fn run_img2img(&self, cfg: &Img2ImgConfig) -> Result<Tensor, PipelineError> {
        self.run_img2img_with_progress(cfg, &mut |_| {})
    }

    /// [`Self::run_img2img`], reporting progress after each step.
    pub fn run_img2img_with_progress(
        &self,
        cfg: &Img2ImgConfig,
        progress: ProgressFn<'_>,
    ) -> Result<Tensor, PipelineError> {
        let base = &cfg.base;
        if base.width % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("width", base.width));
        }
        if base.height % 8 != 0 {
            return Err(PipelineError::NotMultipleOfEight("height", base.height));
        }
        if base.steps == 0 {
            return Err(PipelineError::NoSteps);
        }

        let cond = self.encode(&base.prompt)?;
        let uncond = self.encode(&base.negative_prompt)?;
        let context = Tensor::cat(&[&uncond, &cond], 0)?;

        let image = crate::image_io::load_image(
            &cfg.init_image,
            base.width as u32,
            base.height as u32,
            &self.device,
        )?;
        // The distribution mean, not a draw from it: the sampler supplies all
        // the randomness, so this stays a function of the seed alone.
        let latent = self.vae_encoder.encode(&image)?;

        let sigmas = self.sigmas_for(base.sampler, base.steps);
        let start = cfg.strength.start_index(base.steps);
        // Strength 0 means "return the input", and there is nothing to run.
        if start >= base.steps {
            return self.decode(&latent);
        }

        let mut rng = SeededRng::new(base.seed);
        let (lh, lw) = (base.height / 8, base.width / 8);
        // Noise the encoded latent to the sigma the run starts at. This is
        // what makes strength mean something: a later start is less noise and
        // so a smaller departure from the input.
        let noise = rng.randn((1, 4, lh, lw), &self.device)?;
        let latent = (latent + (noise * sigmas[start])?)?;

        let latent = self.denoise(
            latent,
            Denoise::new(base, &sigmas[start..], std::slice::from_ref(&context)),
            &mut rng,
            progress,
        )?;
        self.decode(&latent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_masked_pixel_frees_its_whole_latent_cell() {
        // Max, not mean, and this is the case that distinguishes them. One
        // white pixel in an 8x8 block means that block's latent cell must be
        // free to change: a latent cell is not a pixel, and averaging would
        // give 1/64 — an almost-frozen cell, producing a hard seam exactly at
        // the mask edge, where it is most visible.
        let dev = sd_tensor::Device::Cpu;
        let mut px = vec![0f32; 16 * 16];
        px[0] = 1.0; // top-left pixel only
        let m = sd_tensor::Tensor::from_vec(px, (1, 1, 16, 16), &dev).unwrap();

        let lm = latent_mask(&m, 2, 2).unwrap();
        assert_eq!(lm.dims(), &[1, 1, 2, 2]);
        let v = lm.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![1.0, 0.0, 0.0, 0.0], "only the covering cell frees");
    }

    #[test]
    fn a_mask_that_is_not_a_multiple_of_eight_is_refused() {
        // Silently truncating would shift the mask relative to the image,
        // repainting the wrong region and still returning a plausible picture.
        let dev = sd_tensor::Device::Cpu;
        let m = sd_tensor::Tensor::zeros((1, 1, 12, 16), sd_tensor::DType::F32, &dev).unwrap();
        assert!(latent_mask(&m, 2, 2).is_err());
    }
}
