//! The generation pipeline on MLX.
//!
//! Every model this needs is already ported and gated against diffusers. What
//! is here is the orchestration between them — and it is the part a caller
//! actually invokes, so until it exists the MLX port is a set of verified
//! pieces rather than a working backend.
//!
//! # What is shared with the candle pipeline, and why
//!
//! The configuration types — [`Txt2ImgConfig`], [`SamplerKind`], [`Strength`],
//! [`Progress`] — are plain data and touch no tensor, so they are *imported*
//! rather than mirrored. So is `sd_sample::Schedule`, which returns `Vec<f64>`.
//! A second copy of any of them is how the two backends would come to disagree
//! about what a seed or a strength means.
//!
//! What is genuinely rewritten is the tensor work: the guidance batch, the
//! sampler step, and the NCHW/NHWC boundary.
//!
//! # The one thing that must not drift: the noise
//!
//! Noise is drawn through `SeededRng` on the CPU and handed to MLX as plain
//! data, exactly as `mlx_end_to_end` does. That is deliberate rather than
//! lazy: it makes the two backends see **identical draws**, so any difference
//! between their images is the models and not the dice. Drawing on the GPU
//! would be faster and would make every cross-backend comparison meaningless.

use std::path::{Path, PathBuf};

use sd_models::clip::ClipTokenizer;
use sd_models::mlx::{
    clip, clip_vision, controlnet, gligen, ip, lora::Lora, normalise_legacy_attention, sample,
    unet_forward_adapters, vae, Adapters, UNetConfig, Weights,
};
use sd_sample::{sigmas_for_steps, Schedule};
use sd_tensor::mlx::{concat, load_safetensors, Array, Stream};
use sd_tensor::rng::SeededRng;
use sd_tensor::{Device, Tensor};

use crate::pipeline::{PipelineError, SamplerKind, Strength, Txt2ImgConfig};

pub mod sdxl;

pub use sdxl::SdxlPipeline;

/// `PipelineError` carries no free-form variant of its own, so a message goes
/// through the tensor error the way the candle pipeline's do.
pub(crate) fn msg(text: String) -> PipelineError {
    PipelineError::Tensor(sd_tensor::Error::Msg(text))
}

/// The discrete training timestep nearest a continuous sigma.
///
/// The UNet takes a training timestep, not a sigma. Handing it the sigma runs —
/// both are one number — and conditions on the wrong point of the schedule
/// entirely.
pub(crate) fn timestep_for(schedule: &Schedule, sigma: f64) -> f32 {
    schedule
        .sigmas()
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - sigma)
                .abs()
                .partial_cmp(&(*b - sigma).abs())
                .expect("finite sigmas")
        })
        .map(|(i, _)| i as f32)
        .unwrap_or(0.0)
}

/// A standard-normal draw of `[1, c, h, w]`, delivered in NHWC.
///
/// **Through `SeededRng` on the CPU**, which is the module's one deliberate
/// inefficiency: it makes both backends see identical draws, so a difference
/// between their images is the models and not the dice.
pub(crate) fn draw_noise(
    rng: &mut SeededRng,
    c: usize,
    h: usize,
    w: usize,
) -> Result<Array, PipelineError> {
    let t: Tensor = rng.randn((1, c, h, w), &Device::Cpu)?;
    let v = t.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0.0f32; v.len()];
    for ci in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[(y * w + x) * c + ci] = v[ci * h * w + y * w + x];
            }
        }
    }
    Ok(Array::from_slice_f32(&out, &[1, h, w, c])?)
}

/// `<|startoftext|>` and `<|endoftext|>` in CLIP's vocabulary.
const BOS: i32 = 49406;
const EOS: i32 = 49407;

/// Where a checkpoint's four pieces live under a model directory.
///
/// Laid out as `diffusers` does, which is what `SD_TEST_MODEL_DIR` points at.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub unet: PathBuf,
    pub vae: PathBuf,
    pub text_encoder: PathBuf,
    pub tokenizer: PathBuf,
}

impl ModelPaths {
    /// The `diffusers` layout under `root`.
    pub fn in_dir(root: &Path) -> Self {
        Self {
            unet: root.join("unet/diffusion_pytorch_model.safetensors"),
            vae: root.join("vae/diffusion_pytorch_model.safetensors"),
            text_encoder: root.join("text_encoder/model.safetensors"),
            tokenizer: root.join("tokenizer/tokenizer.json"),
        }
    }

    /// Every path, for an existence check that names what is missing rather
    /// than failing at whichever one is loaded first.
    pub fn missing(&self) -> Vec<&Path> {
        [
            self.unet.as_path(),
            self.vae.as_path(),
            self.text_encoder.as_path(),
            self.tokenizer.as_path(),
        ]
        .into_iter()
        .filter(|p| !p.exists())
        .collect()
    }
}

/// One ControlNet and how hard it steers.
///
/// `scale` multiplies every correction; **0 contributes exactly nothing**
/// rather than merely almost nothing, which is what makes it a usable off
/// switch.
pub struct Control {
    pub weights: Weights,
    pub scale: f64,
}

/// An attached IP-Adapter: its own weights, the image tower's, and the scale.
///
/// The adapter's attention weights live in the same map as `image_proj` — it
/// ships as one file — but the **vision tower is a separate checkpoint**, and a
/// large one. Held apart so a run that never conditions on an image does not
/// pay for it.
pub struct IpAdapter {
    weights: Weights,
    vision: Weights,
    scale: f32,
}

/// One grounded box for GLIGEN: where, and what.
///
/// Coordinates are `[x0, y0, x1, y1]` in `[0, 1]`, **not pixels** — the model
/// was trained on normalised boxes and pixel values put every box off the
/// canvas without an error.
pub struct GroundedBox {
    pub bbox: [f32; 4],
    pub phrase: String,
}

/// The per-run conditioning that is not the prompt.
///
/// Bundled because the sampling loop needs all of it and none of it belongs to
/// [`Txt2ImgConfig`], which is shared with the candle pipeline and holds no
/// tensors.
#[derive(Default)]
struct Extras<'a> {
    /// A ControlNet's control map, `[1, h, w, 3]` in `[-1, 1]`.
    hint: Option<&'a Array>,
    /// The IP-Adapter's four tokens, already doubled for the guidance batch.
    ip_tokens: Option<Array>,
    /// GLIGEN's grounding tokens, likewise doubled.
    objs: Option<&'a Array>,
}

/// A loaded SD 1.5-family pipeline on MLX.
pub struct MlxPipeline {
    tokenizer: ClipTokenizer,
    text_encoder: Weights,
    unet: Weights,
    vae: Weights,
    cfg: UNetConfig,
    vae_cfg: vae::VaeConfig,
    schedule: Schedule,
    stream: Stream,
    /// Spatial conditioning, in attachment order. Empty is the common case.
    controlnets: Vec<Control>,
    ip: Option<IpAdapter>,
}

impl MlxPipeline {
    /// Load SD 1.5 from a `diffusers` model directory.
    pub fn load(root: &Path) -> Result<Self, PipelineError> {
        let paths = ModelPaths::in_dir(root);
        let missing = paths.missing();
        if !missing.is_empty() {
            return Err(msg(format!(
                "mlx: {} is missing {}",
                root.display(),
                missing
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let stream = Stream::gpu();
        let tokenizer = ClipTokenizer::from_file(&paths.tokenizer)?;

        // **The stock checkpoint may use the legacy attention names.** The
        // decoder asks for `to_q` and a published VAE has `query`; converting
        // once here is what keeps every model module free of the concern.
        let mut vae = load_safetensors(&paths.vae)?;
        normalise_legacy_attention(&mut vae);
        let mut unet = load_safetensors(&paths.unet)?;
        normalise_legacy_attention(&mut unet);

        Ok(Self {
            tokenizer,
            text_encoder: load_safetensors(&paths.text_encoder)?,
            unet,
            vae,
            cfg: UNetConfig::sd15(),
            vae_cfg: vae::VaeConfig::sd15(),
            schedule: Schedule::sd15(),
            stream,
            controlnets: Vec::new(),
            ip: None,
        })
    }

    /// Attach an IP-Adapter.
    ///
    /// `adapter` carries `image_proj` and the per-layer `to_k_ip`/`to_v_ip`
    /// weights; `image_encoder` is the CLIP vision tower, which is a separate
    /// and much larger checkpoint.
    ///
    /// **Scale 0 contributes exactly nothing**, so it is a usable off switch
    /// rather than an approximation of one.
    pub fn attach_ip_adapter(
        &mut self,
        adapter: &Path,
        image_encoder: &Path,
        scale: f64,
    ) -> Result<(), PipelineError> {
        self.ip = Some(IpAdapter {
            weights: load_safetensors(adapter)?,
            vision: load_safetensors(image_encoder)?,
            scale: scale as f32,
        });
        Ok(())
    }

    /// The adapter's four tokens for a reference image.
    ///
    /// `image` is `[1, 224, 224, 3]` in **`[0, 1]`** — CLIP's own range, not
    /// the `[-1, 1]` a VAE uses. The wrong range is accepted and describes the
    /// wrong picture.
    fn ip_tokens(&self, image: &Array) -> Result<Option<Array>, PipelineError> {
        let Some(adapter) = &self.ip else {
            return Ok(None);
        };
        let s = &self.stream;
        let pixels = clip_vision::preprocess(image, s)?;
        // The **projected** embedding, 1024 wide for ViT-H — not the pooled
        // 1280, which is a different vector of a different width.
        let embeds = clip_vision::image_embeds(
            &pixels,
            &clip_vision::VisionConfig::vit_h_14(),
            &adapter.vision,
            s,
        )?;
        let tokens = ip::image_proj(&embeds, clip::HIDDEN, &adapter.weights, s)?;
        // Doubled to match the guidance batch, unconditional row first. The
        // unconditional row gets the *same* tokens: dropping the image there
        // would make guidance push away from it.
        Ok(Some(concat(&[&tokens, &tokens], 0, s)?))
    }

    /// Merge a LoRA into the UNet, in place.
    ///
    /// **Coverage is the thing that matters**, not the arithmetic. The merge is
    /// three lines and hard to get subtly wrong; the name mapping is where an
    /// adapter silently half-applies, and a half-applied adapter still renders
    /// a plausible image. So this errors on any layer that found no home rather
    /// than applying the rest.
    pub fn attach_lora(&mut self, path: &Path, multiplier: f64) -> Result<usize, PipelineError> {
        let raw = load_safetensors(path)?;
        let lora = Lora::from_weights(&raw, &self.stream)?;
        let applied = lora.merge_into(&mut self.unet, multiplier as f32, &self.stream)?;
        if !applied.unmatched.is_empty() {
            return Err(msg(format!(
                "mlx: {} of the LoRA's layers have no weight in this UNet, first `{}`. \
                 A LoRA names the layers it corrects, so entries with nowhere to go mean it \
                 was trained for a different architecture.",
                applied.unmatched.len(),
                applied.unmatched.first().map(String::as_str).unwrap_or("?")
            )));
        }
        Ok(applied.merged)
    }

    /// Attach a ControlNet. Several may be attached, and their corrections sum.
    ///
    /// **Built from the same config as this UNet**, which is checked where the
    /// corrections are added rather than here — a ControlNet for a different
    /// architecture emits a plausible number of plausible tensors and only the
    /// count catches it.
    pub fn attach_controlnet(&mut self, path: &Path, scale: f64) -> Result<(), PipelineError> {
        let mut weights = load_safetensors(path)?;
        normalise_legacy_attention(&mut weights);
        self.controlnets.push(Control { weights, scale });
        Ok(())
    }

    /// How many ControlNets are attached.
    pub fn controlnet_count(&self) -> usize {
        self.controlnets.len()
    }

    /// GLIGEN's grounding tokens for a set of boxes.
    ///
    /// Requires a checkpoint whose UNet carries `fuser` layers; an ordinary SD
    /// 1.5 UNet has nowhere to put them, so this errors rather than dropping
    /// the boxes silently.
    fn grounding(&self, boxes: &[GroundedBox]) -> Result<Option<Array>, PipelineError> {
        if boxes.is_empty() {
            return Ok(None);
        }
        if !gligen::present(
            &self.unet,
            "down_blocks.0.attentions.0.transformer_blocks.0",
        ) {
            return Err(msg(
                "mlx: grounded boxes were supplied but this UNet has no GLIGEN fuser layers".into(),
            ));
        }
        let s = &self.stream;
        let n = boxes.len();
        let mut rows = Vec::with_capacity(n);
        let mut coords = Vec::with_capacity(n * 4);
        for b in boxes {
            // **The phrase's pooled hidden state**, which is the EOS position
            // and not position 0. `clip::pool` takes the *first* highest token
            // id, which matters here because CLIP-L pads with EOS itself — the
            // last one is 60-odd positions past the end of the phrase.
            let ids = self.token_ids(&b.phrase)?;
            let hidden = clip::text_encoder(&ids, &self.text_encoder, s)?;
            rows.push(clip::pool(&hidden, &ids, s)?);
            coords.extend_from_slice(&b.bbox);
        }
        let refs: Vec<&Array> = rows.iter().collect();
        let phrases = concat(&refs, 0, s)?.reshape(&[1, n, clip::HIDDEN], s)?;
        let boxes_arr = Array::from_slice_f32(&coords, &[1, n, 4])?;
        // Every slot is real here, so every mask is 1. The learned nulls exist
        // for callers batching a fixed number of slots.
        let masks = Array::from_slice_f32(&vec![1.0; n], &[1, n])?;

        let objs = gligen::position_net(&boxes_arr, &masks, &phrases, &self.unet, s)?;
        // Doubled for the guidance batch, like every other conditioning here.
        Ok(Some(concat(&[&objs, &objs], 0, s)?))
    }

    /// Encode one prompt to `[1, 77, 768]`.
    ///
    /// An empty prompt is *not* an empty sequence: it is BOS followed by 76
    /// EOS, which is what the tokenizer produces and what the model was trained
    /// against. Feeding a zero tensor instead is a different unconditional.
    fn encode(&self, prompt: &str) -> Result<Array, PipelineError> {
        let ids = self.token_ids(prompt)?;
        Ok(clip::text_encoder(&ids, &self.text_encoder, &self.stream)?)
    }

    /// A prompt's 77 token ids.
    ///
    /// An empty prompt is *not* an empty sequence: it is BOS followed by 76
    /// EOS, which is what the tokenizer produces and what the model was trained
    /// against. A zero tensor instead is a different unconditional.
    fn token_ids(&self, prompt: &str) -> Result<Array, PipelineError> {
        let ids: Vec<i32> = if prompt.is_empty() {
            let mut v = vec![EOS; clip::MAX_POSITION];
            v[0] = BOS;
            v
        } else {
            self.tokenizer
                .encode(prompt)?
                .iter()
                .map(|&x| x as i32)
                .collect()
        };
        if ids.len() != clip::MAX_POSITION {
            return Err(msg(format!(
                "mlx: the tokenizer produced {} ids, CLIP takes {}",
                ids.len(),
                clip::MAX_POSITION
            )));
        }
        Ok(Array::from_slice_i32(&ids, &[1, clip::MAX_POSITION])?)
    }

    /// The guidance batch: unconditional row **first**.
    ///
    /// The order is a contract with [`sample::guidance`], which reads row 0 as
    /// the unconditional. Reversing it runs and drives the image away from the
    /// prompt instead of toward it.
    fn conditioning(&self, cfg: &Txt2ImgConfig) -> Result<Array, PipelineError> {
        let cond = self.encode(&cfg.prompt)?;
        let uncond = self.encode(&cfg.negative_prompt)?;
        Ok(concat(&[&uncond, &cond], 0, &self.stream)?)
    }

    /// The sigma ladder for a sampler and step count.
    fn sigmas(&self, sampler: SamplerKind, steps: usize) -> Vec<f64> {
        match sampler {
            SamplerKind::Lcm => sd_sample::lcm_sigmas(
                &self.schedule,
                &sd_sample::lcm_timesteps(
                    self.schedule.alphas_cumprod.len(),
                    sd_sample::ORIGINAL_INFERENCE_STEPS,
                    steps,
                ),
            ),
            _ => sigmas_for_steps(&self.schedule, steps),
        }
    }

    fn timestep_for(&self, sigma: f64) -> f32 {
        timestep_for(&self.schedule, sigma)
    }

    fn draw(
        &self,
        rng: &mut SeededRng,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<Array, PipelineError> {
        draw_noise(rng, c, h, w)
    }

    /// The sampling loop, shared by txt2img, img2img and inpaint.
    ///
    /// `sigmas` is a ladder of `n + 1` boundaries; img2img passes a suffix of
    /// one. `rng` is threaded in rather than created here so the caller
    /// controls draw order, which is what makes a seed reproducible.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        mut latent: Array,
        context: &Array,
        sigmas: &[f64],
        cfg_scale: f64,
        sampler: SamplerKind,
        keep: Option<(&Array, &Array)>,
        extras: &Extras<'_>,
        rng: &mut SeededRng,
    ) -> Result<Array, PipelineError> {
        let s = &self.stream;
        let [_, lh, lw, lc] = latent.shape()[..] else {
            return Err(msg(format!(
                "mlx: a latent should be [n, h, w, c], got {:?}",
                latent.shape()
            )));
        };
        let mut dpm = sample::DpmSolverPlusPlus2M::new();

        for i in 0..sigmas.len().saturating_sub(1) {
            let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);

            let latent_in = sample::scale_model_input(&latent, sigma, s)?;
            let t = self.timestep_for(sigma);
            let timestep = Array::from_slice_f32(&[t, t], &[2])?;

            // Several ControlNets sum. Running them here rather than once
            // outside the loop is not an optimisation missed: each is
            // conditioned on the current latent and timestep, so its
            // corrections differ every step.
            let control = self.control_for(&latent_in, &timestep, context, extras.hint)?;
            // The adapter walks its layers in visit order, so its counter has
            // to start from zero on every step rather than continue across a
            // run. `unet_forward_adapters` rewinds it.
            let ip = extras.ip_tokens.as_ref().map(|tokens| {
                ip::IpAdapter::new(
                    &self.ip.as_ref().expect("tokens imply an adapter").weights,
                    tokens.contiguous(s).expect("tokens"),
                    self.ip.as_ref().expect("adapter").scale,
                )
            });
            let ad = Adapters {
                control: control.as_ref(),
                ip: ip.as_ref(),
                objs: extras.objs,
                ..Default::default()
            };
            let out = unet_forward_adapters(
                &latent_in, &timestep, context, None, None, &ad, &self.cfg, &self.unet, s,
            )?;
            let noise_pred = sample::guidance(&out, cfg_scale, s)?;
            let denoised = sample::denoise_epsilon(&latent, &noise_pred, sigma, s)?;

            latent = match sampler {
                // Ancestral: a fresh draw every step, which is why step
                // caching is refused with it on the candle side too.
                SamplerKind::EulerAncestral | SamplerKind::Lcm => {
                    let noise = self.draw(rng, lc, lh, lw)?;
                    sample::euler_ancestral_step(&latent, &denoised, sigma, sigma_next, &noise, s)?
                }
                SamplerKind::DpmPlusPlus2M => dpm.step(&latent, &denoised, sigma, sigma_next, s)?,
            };

            // Inpainting: restore outside the mask at every step, so the model
            // sees the true surroundings and what it paints joins up with them.
            if let Some((mask, init)) = keep {
                let noise = self.draw(rng, lc, lh, lw)?;
                latent = sample::restore_outside_mask(&latent, init, mask, &noise, sigma_next, s)?;
            }
        }
        Ok(latent)
    }

    /// Every attached ControlNet's corrections, summed.
    ///
    /// `None` when nothing is attached, so the ordinary path allocates
    /// nothing. A hint is required once one is: a ControlNet with no map to
    /// read would emit corrections from a blank image, which steers the run
    /// toward an empty picture rather than doing nothing.
    fn control_for(
        &self,
        latent_in: &Array,
        timestep: &Array,
        context: &Array,
        hint: Option<&Array>,
    ) -> Result<Option<controlnet::Control>, PipelineError> {
        if self.controlnets.is_empty() {
            return Ok(None);
        }
        let s = &self.stream;
        let Some(hint) = hint else {
            return Err(msg(
                "mlx: a ControlNet is attached but no control map was supplied".into(),
            ));
        };
        // The hint is doubled to match the guidance batch, exactly as the
        // latent is — the ControlNet sees both rows.
        let hint = concat(&[hint, hint], 0, s)?;

        let mut total: Option<controlnet::Control> = None;
        for net in &self.controlnets {
            let c = controlnet::forward(
                latent_in,
                timestep,
                context,
                &hint,
                net.scale,
                &self.cfg,
                &net.weights,
                s,
            )?;
            total = Some(match total {
                None => c,
                Some(acc) => {
                    if acc.down.len() != c.down.len() {
                        return Err(msg(format!(
                            "mlx: two ControlNets emitted {} and {} corrections",
                            acc.down.len(),
                            c.down.len()
                        )));
                    }
                    let down = acc
                        .down
                        .iter()
                        .zip(&c.down)
                        .map(|(a, b)| a.add(b, s))
                        .collect::<Result<Vec<_>, _>>()?;
                    controlnet::Control {
                        down,
                        mid: acc.mid.add(&c.mid, s)?,
                    }
                }
            });
        }
        Ok(total)
    }

    /// A latent to `[h, w, 3]` bytes.
    fn decode(&self, latent: &Array) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        let s = &self.stream;
        let unscaled = self.vae_cfg.unscale(latent, s)?;
        let image = vae::decode_with(&unscaled, &self.vae_cfg, &self.vae, s)?;
        let [_, h, w, _] = image.shape()[..] else {
            return Err(msg(format!(
                "mlx: the decoder returned {:?}",
                image.shape()
            )));
        };
        // The VAE emits roughly [-1, 1]; the caller wants bytes.
        let bytes = image
            .to_vec_f32(s)?
            .iter()
            .map(|&v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        Ok((w, h, bytes))
    }

    /// Prompt to pixels. Returns `(width, height, RGB bytes)`.
    pub fn txt2img(&self, cfg: &Txt2ImgConfig) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        self.txt2img_controlled(cfg, None)
    }

    /// [`Self::txt2img`] with a control map for the attached ControlNets.
    ///
    /// `hint` is `[1, h, w, 3]` in `[-1, 1]` at the run's own resolution — a
    /// Canny edge map, a depth map, whatever the ControlNet was trained on.
    pub fn txt2img_controlled(
        &self,
        cfg: &Txt2ImgConfig,
        hint: Option<&Array>,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        self.generate(cfg, hint, None, &[])
    }

    /// Everything at once: a control map, an IP-Adapter reference image, and
    /// GLIGEN boxes.
    ///
    /// `reference` is `[1, 224, 224, 3]` in **`[0, 1]`** — CLIP's range, not
    /// the `[-1, 1]` a VAE uses.
    pub fn generate(
        &self,
        cfg: &Txt2ImgConfig,
        hint: Option<&Array>,
        reference: Option<&Array>,
        boxes: &[GroundedBox],
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        if cfg.width % 8 != 0 || cfg.height % 8 != 0 {
            return Err(msg(format!(
                "mlx: {}x{} does not divide into 8-pixel latent cells",
                cfg.width, cfg.height
            )));
        }
        let (lh, lw) = (cfg.height / 8, cfg.width / 8);
        let context = self.conditioning(cfg)?;
        let sigmas = self.sigmas(cfg.sampler, cfg.steps);

        let ip_tokens = match reference {
            Some(image) => self.ip_tokens(image)?,
            None => None,
        };
        let objs = self.grounding(boxes)?;
        let extras = Extras {
            hint,
            ip_tokens,
            objs: objs.as_ref(),
        };

        let mut rng = SeededRng::new(cfg.seed);
        // Sampling starts at maximum noise, so the latent is scaled by the
        // first sigma rather than used raw.
        let latent = self
            .draw(&mut rng, 4, lh, lw)?
            .mul(&Array::scalar_f32(sigmas[0] as f32)?, &self.stream)?;

        let latent = self.denoise(
            latent,
            &context,
            &sigmas,
            cfg.cfg_scale,
            cfg.sampler,
            None,
            &extras,
            &mut rng,
        )?;
        self.decode(&latent)
    }

    /// An image and a prompt to pixels.
    ///
    /// `image` is `[1, h, w, 3]` in `[-1, 1]`, already at `cfg.width` x
    /// `cfg.height`. `strength` selects where in the ladder the run begins: at
    /// strength `s` with `n` steps it starts at `n - round(n*s)`.
    ///
    /// **Does not yet carry a control map, a reference image or boxes.** With
    /// a ControlNet attached this errors rather than running unsteered — which
    /// is the right failure, but it does mean img2img and ControlNet cannot be
    /// combined here yet. `txt2img_controlled` and `generate` are the
    /// conditioned entry points.
    pub fn img2img(
        &self,
        cfg: &Txt2ImgConfig,
        image: &Array,
        strength: Strength,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        self.run_masked(cfg, image, strength, None)
    }

    /// img2img bounded by a mask. `mask_px` is `[1, h, w, 1]`, 1 where the
    /// model may write.
    ///
    /// Same conditioning limits as [`Self::img2img`].
    pub fn inpaint(
        &self,
        cfg: &Txt2ImgConfig,
        image: &Array,
        strength: Strength,
        mask_px: &Array,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        self.run_masked(cfg, image, strength, Some(mask_px))
    }

    fn run_masked(
        &self,
        cfg: &Txt2ImgConfig,
        image: &Array,
        strength: Strength,
        mask_px: Option<&Array>,
    ) -> Result<(usize, usize, Vec<u8>), PipelineError> {
        let s = &self.stream;
        // The distribution's mean, scaled — the sampler supplies all the
        // randomness, so drawing here too would add variance the seed does not
        // control.
        let init = vae::encode_scaled(image, &self.vae_cfg, &self.vae, s)?;
        let [_, lh, lw, _] = init.shape()[..] else {
            return Err(msg(format!("mlx: the encoder returned {:?}", init.shape())));
        };

        let sigmas = self.sigmas(cfg.sampler, cfg.steps);
        let start = strength.start_index(cfg.steps);
        // Strength 0 means "return the input", and there is nothing to run.
        if start >= cfg.steps {
            return self.decode(&init);
        }

        let mut rng = SeededRng::new(cfg.seed);
        let noise = self.draw(&mut rng, 4, lh, lw)?;
        let latent = sample::noise_to_sigma(&init, &noise, sigmas[start], s)?;

        let mask = mask_px.map(|m| sample::latent_mask(m, s)).transpose()?;
        let context = self.conditioning(cfg)?;
        let latent = self.denoise(
            latent,
            &context,
            &sigmas[start..],
            cfg.cfg_scale,
            cfg.sampler,
            mask.as_ref().map(|m| (m, &init)),
            &Extras::default(),
            &mut rng,
        )?;
        self.decode(&latent)
    }

    /// The stream this pipeline runs on, for callers that build their own
    /// tensors to hand in.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}
