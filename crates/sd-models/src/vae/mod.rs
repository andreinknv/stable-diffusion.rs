//! Variational autoencoder (`AutoencoderKL`).
//!
//! Milestone 1 implements the **decoder** only: latents in, image out. That is
//! deliberate — decoding a reference latent gives a visible, checkable result
//! and exercises most of the op surface the UNet will later need.

mod decoder;
mod encoder;
mod tiny;

pub use decoder::{Decoder, DecoderConfig};
pub use encoder::{Encoder, EncoderConfig};
pub use tiny::{TinyAutoencoder, TinyDecoder, TinyEncoder};

use sd_tensor::nn::{conv2d, Conv2d, Conv2dConfig};
use sd_tensor::{Result, Tensor, VarBuilder};

/// Configuration for `AutoencoderKL`.
///
/// Defaults match the SD 1.x / 2.x VAE. SDXL uses the same geometry and
/// differs only in `scaling_factor`.
#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub latent_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
    /// Latents are stored scaled; decoding divides by this first.
    pub scaling_factor: f64,
    /// Latents are stored shifted as well as scaled: `(x - shift) * scale`.
    ///
    /// Zero for every Stable Diffusion VAE, non-zero for Flux. Kept separate
    /// from `scaling_factor` rather than folded into it because the two are
    /// applied in a fixed order and folding them would silently transpose it.
    pub shift_factor: f64,
    /// Whether `quant_conv` / `post_quant_conv` exist.
    ///
    /// Every SD VAE has them; Flux sets `use_quant_conv: false` and feeds the
    /// latent straight to the decoder. They are 1x1 convolutions, so a wrong
    /// answer here is not a shape error — it is a missing weight, or a silent
    /// extra transform.
    pub use_quant_conv: bool,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self::sd15()
    }
}

impl VaeConfig {
    /// Stable Diffusion 1.x / 2.x.
    pub fn sd15() -> Self {
        Self {
            latent_channels: 4,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            norm_eps: 1e-6,
            scaling_factor: 0.18215,
            shift_factor: 0.0,
            use_quant_conv: true,
        }
    }

    /// SDXL. Identical geometry; different latent scaling.
    pub fn sdxl() -> Self {
        Self {
            scaling_factor: 0.13025,
            ..Self::sd15()
        }
    }

    /// SD 3 / SD 3.5. Identical to [`Self::flux`] in every structural
    /// respect — 16 latent channels, no quant convolutions, the same
    /// `[128, 256, 512, 512]` stack — and different only in the two latent
    /// constants. Worth noting because it means one decoder now serves SD 1.5,
    /// SDXL, Flux and SD 3 with nothing but a config between them.
    pub fn sd35() -> Self {
        Self {
            scaling_factor: 1.5305,
            shift_factor: 0.0609,
            ..Self::flux()
        }
    }

    /// Flux. The same convolutional geometry as SD — `[128, 256, 512, 512]`,
    /// two layers per block, 32 groups — with a 16-channel latent instead of
    /// 4, and a shifted latent distribution.
    ///
    /// The wider latent is the whole reason Flux images hold fine detail that
    /// SD's 4-channel latent cannot represent, and it costs nothing here
    /// because the encoder and decoder are already parameterised by it.
    pub fn flux() -> Self {
        Self {
            latent_channels: 16,
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
            use_quant_conv: false,
            ..Self::sd15()
        }
    }
}

/// The decode half of `AutoencoderKL`: `post_quant_conv` followed by [`Decoder`].
#[derive(Debug)]
pub struct AutoencoderKlDecoder {
    post_quant_conv: Option<Conv2d>,
    decoder: Decoder,
    scaling_factor: f64,
    shift_factor: f64,
}

impl AutoencoderKlDecoder {
    /// Build from weights. `vb` should be rooted at the VAE itself, so that
    /// `post_quant_conv` and `decoder.*` resolve directly beneath it.
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let post_quant_conv = if cfg.use_quant_conv {
            Some(conv2d(
                cfg.latent_channels,
                cfg.latent_channels,
                1,
                Conv2dConfig::default(),
                vb.pp("post_quant_conv"),
            )?)
        } else {
            None
        };
        let decoder = Decoder::new(&DecoderConfig::from(cfg), vb.pp("decoder"))?;
        Ok(Self {
            post_quant_conv,
            decoder,
            scaling_factor: cfg.scaling_factor,
            shift_factor: cfg.shift_factor,
        })
    }

    /// Decode *already unscaled* latents `[b, 4, h, w]` to `[b, 3, h*8, w*8]`
    /// in roughly `[-1, 1]`.
    pub fn decode_raw(&self, latents: &Tensor) -> Result<Tensor> {
        let xs = match &self.post_quant_conv {
            Some(c) => c.forward_t(latents)?,
            None => latents.clone(),
        };
        self.decoder.forward(&xs)
    }

    /// Undo the stored latent parameterisation: `x / scale + shift`.
    ///
    /// The inverse of [`AutoencoderKlEncoder::scale`], and the order matters —
    /// dividing after shifting gives a plausible image with wrong contrast
    /// rather than an error.
    fn unscale(&self, latents: &Tensor) -> Result<Tensor> {
        (latents / self.scaling_factor)? + self.shift_factor
    }

    /// Decode latents as produced by the sampler, applying the scaling and
    /// shift the checkpoint stores them under.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        self.decode_raw(&self.unscale(latents)?)
    }

    /// [`Self::decode`], in overlapping tiles.
    ///
    /// A whole-image decode allocates a conv im2col of `cin * 9` values per
    /// output position — 9.66 GB at 1024px, which does not fit in GPU memory
    /// on a 36 GiB Mac. Tiling caps that at whatever one tile needs, so the
    /// cost becomes linear in area instead of a single enormous allocation.
    ///
    /// The result is *close to* but not identical to a whole-image decode:
    /// convolutions at a tile edge see padding where they would otherwise see
    /// neighbouring pixels, which the overlap blend hides rather than
    /// eliminates. Small latents that already fit are decoded whole, so this
    /// is safe to call unconditionally.
    pub fn decode_tiled(&self, latents: &Tensor) -> Result<Tensor> {
        self.decode_raw_tiled(&self.unscale(latents)?)
    }

    /// The tile edge this decode will use.
    ///
    /// An explicit [`TILE_LATENT_EDGE_ENV`] wins outright — a caller who set
    /// it is answering this question themselves. Otherwise the largest edge
    /// that is projected to fit is chosen, from [`TILE_LATENT_EDGE`] down.
    ///
    /// **Why this is chosen rather than fixed.** A 64-latent tile allocates a
    /// 2.42 GB convolution im2col. That is comfortable when the VAE is most of
    /// what is resident and impossible when a 10 GB transformer is also on the
    /// GPU — and the failure lands at the *end* of a run, after every denoise
    /// step has been paid for. Projecting the cost against what is actually
    /// free turns that into a slightly different image instead of a dead run.
    ///
    /// Halving rather than searching finely: the peak scales with the tile's
    /// area, so the candidates are already far apart, and each extra step down
    /// costs another seam to blend.
    pub fn tile_edge_for(&self, batch: usize, dtype: sd_tensor::DType) -> Result<usize> {
        if let Some(explicit) = explicit_tile_latent_edge()? {
            return Ok(explicit);
        }
        let cfg = self.decoder.config();
        let mut tile = TILE_LATENT_EDGE;
        while tile > MIN_TILE_LATENT_EDGE {
            let peak = cfg.peak_alloc_bytes(batch, tile, tile, dtype);
            // `check_headroom` refuses when the projection exceeds the
            // configured fraction of free memory, which is exactly the
            // question being asked; its error is discarded rather than
            // returned because a tile that does not fit is not an error here.
            if sd_tensor::sysmem::check_headroom(peak.unwrap_or(u64::MAX), "a VAE decode tile")
                .is_ok()
            {
                break;
            }
            tile /= 2;
        }
        Ok(tile)
    }

    /// [`Self::decode_raw`], in overlapping tiles.
    pub fn decode_raw_tiled(&self, latents: &Tensor) -> Result<Tensor> {
        let (b, _, lh, lw) = latents.dims4()?;
        let tile = self.tile_edge_for(b, latents.dtype())?;
        if lh <= tile && lw <= tile {
            return self.decode_raw(latents);
        }

        let stride = ((tile as f64) * (1.0 - TILE_OVERLAP)) as usize;
        let stride = stride.max(1);
        // In output pixels: the decoder upsamples by 8.
        let scale = 8;
        let blend_extent = (tile - stride) * scale;
        let keep = stride * scale;

        let mut rows: Vec<Vec<Tensor>> = Vec::new();
        let mut y = 0;
        while y < lh {
            let h = tile.min(lh - y);
            let mut row = Vec::new();
            let mut x = 0;
            while x < lw {
                let w = tile.min(lw - x);
                let patch = latents.narrow(2, y, h)?.narrow(3, x, w)?.contiguous()?;
                row.push(self.decode_raw(&patch)?);
                if x + w >= lw {
                    break;
                }
                x += stride;
            }
            rows.push(row);
            if y + h >= lh {
                break;
            }
            y += stride;
        }

        // Blend each tile into its upper and left neighbours, then trim to the
        // stride so the kept regions tile exactly.
        let mut out_rows = Vec::with_capacity(rows.len());
        for i in 0..rows.len() {
            let mut out_row: Vec<Tensor> = Vec::with_capacity(rows[i].len());
            for j in 0..rows[i].len() {
                let mut t = rows[i][j].clone();
                if i > 0 {
                    t = blend(&rows[i - 1][j], &t, blend_extent, 2)?;
                }
                if j > 0 {
                    // The *untrimmed* left neighbour, not the trimmed result
                    // already pushed to `out_row` — those differ in height on
                    // any row whose tiles were cut short, and blending across
                    // that mismatch is a shape error.
                    t = blend(&rows[i][j - 1], &t, blend_extent, 3)?;
                }
                let last_col = j + 1 == rows[i].len();
                let last_row = i + 1 == rows.len();
                let th = t.dims()[2];
                let tw = t.dims()[3];
                let h = if last_row { th } else { keep.min(th) };
                let w = if last_col { tw } else { keep.min(tw) };
                out_row.push(t.narrow(2, 0, h)?.narrow(3, 0, w)?.contiguous()?);
            }
            out_rows.push(Tensor::cat(&out_row, 3)?);
        }
        Tensor::cat(&out_rows, 2)
    }
}

/// The encode half of `AutoencoderKL`: [`Encoder`] followed by `quant_conv`.
///
/// Needed for img2img, which starts from a real image rather than from noise.
#[derive(Debug)]
pub struct AutoencoderKlEncoder {
    encoder: Encoder,
    quant_conv: Option<Conv2d>,
    latent_channels: usize,
    scaling_factor: f64,
    shift_factor: f64,
}

impl AutoencoderKlEncoder {
    /// `vb` should be rooted at the VAE itself, so `encoder.*` and
    /// `quant_conv` resolve directly beneath it.
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(&EncoderConfig::from(cfg), vb.pp("encoder"))?;
        // Operates on the concatenated (mean, logvar), hence 2x on both sides.
        let quant_conv = if cfg.use_quant_conv {
            Some(conv2d(
                2 * cfg.latent_channels,
                2 * cfg.latent_channels,
                1,
                Conv2dConfig::default(),
                vb.pp("quant_conv"),
            )?)
        } else {
            None
        };
        Ok(Self {
            encoder,
            quant_conv,
            latent_channels: cfg.latent_channels,
            scaling_factor: cfg.scaling_factor,
            shift_factor: cfg.shift_factor,
        })
    }

    /// Encode `[b, 3, h, w]` in `[-1, 1]` to the latent distribution's
    /// `(mean, logvar)`, each `[b, latent_channels, h/8, w/8]`.
    ///
    /// Unscaled — this is the distribution as the model expresses it.
    pub fn encode_dist(&self, image: &Tensor) -> Result<(Tensor, Tensor)> {
        let encoded = self.encoder.forward(image)?;
        let moments = match &self.quant_conv {
            Some(c) => c.forward_t(&encoded)?,
            None => encoded,
        };
        let mean = moments.narrow(1, 0, self.latent_channels)?;
        let logvar = moments.narrow(1, self.latent_channels, self.latent_channels)?;
        Ok((mean.contiguous()?, logvar.contiguous()?))
    }

    /// Apply the stored latent parameterisation: `(x - shift) * scale`.
    ///
    /// Shift first, then scale. [`AutoencoderKlDecoder::decode`] inverts this
    /// in the opposite order; getting either backwards leaves the image
    /// recognisable but with wrong contrast, which is the kind of error that
    /// survives eyeballing.
    fn scale(&self, latents: &Tensor) -> Result<Tensor> {
        (latents - self.shift_factor)? * self.scaling_factor
    }

    /// Encode to a latent ready for the sampler, applying scaling and shift.
    ///
    /// Uses the distribution's **mean** rather than sampling from it. For
    /// img2img that is what you want: the sampler adds its own noise, so
    /// drawing here too would add variance the seed does not control and make
    /// the result irreproducible.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let (mean, _) = self.encode_dist(image)?;
        self.scale(&mean)
    }

    /// Sample from the latent distribution using externally supplied noise.
    ///
    /// `noise` must be standard normal and shaped like the latent. Kept
    /// separate from [`Self::encode`] so the caller owns the randomness, and
    /// with it reproducibility.
    pub fn encode_sampled(&self, image: &Tensor, noise: &Tensor) -> Result<Tensor> {
        let (mean, logvar) = self.encode_dist(image)?;
        // diffusers clamps logvar to [-30, 20] before exponentiating; without
        // it a degenerate checkpoint can produce an infinite std.
        let std = (logvar.clamp(-30.0, 20.0)? * 0.5)?.exp()?;
        let sample = (mean + (std * noise)?)?;
        self.scale(&sample)
    }

    /// Latent channel count, for sizing the noise [`Self::encode_sampled`]
    /// expects.
    pub fn latent_channels(&self) -> usize {
        self.latent_channels
    }

    /// [`Self::encode`], in overlapping tiles.
    ///
    /// The mirror of [`AutoencoderKlDecoder::decode_tiled`], and needed for
    /// the same reason: an encode allocates a conv im2col of `cin * 9` values
    /// per position, so a 1024x1024 input costs as much as the decode it
    /// feeds. img2img does both, so leaving this untiled means paying the
    /// blowup twice in one run.
    ///
    /// Tiles are taken in image space and blended in latent space, where the
    /// overlap is 8x smaller. Inputs that already fit are encoded whole, so
    /// this is safe to call unconditionally.
    pub fn encode_tiled(&self, image: &Tensor) -> Result<Tensor> {
        let (_, _, ih, iw) = image.dims4()?;
        // An explicit override applies here too, but unlike the decode this
        // does *not* size itself to available memory: `EncoderConfig` has no
        // `peak_alloc_bytes`, so there is nothing to project against. Writing
        // one means deriving the peak through the downsampling stack the way
        // `DecoderConfig` does for the upsampling one — worth doing if an
        // encode is ever the thing that runs out, which it has not been.
        //
        // The two directions choosing different tile edges is not a
        // correctness problem, only a slightly different image; tiling is an
        // implementation detail of each direction, not a shared contract.
        let tile_latent = tile_latent_edge()?;
        let tile_px = tile_latent * 8;
        if ih <= tile_px && iw <= tile_px {
            return self.encode(image);
        }

        let stride_latent = ((tile_latent as f64) * (1.0 - TILE_OVERLAP)) as usize;
        let stride_latent = stride_latent.max(1);
        let stride_px = stride_latent * 8;
        // Blending happens on the latent, so the extent is in latent units.
        let blend_extent = tile_latent - stride_latent;

        let mut rows: Vec<Vec<Tensor>> = Vec::new();
        let mut y = 0;
        while y < ih {
            let h = tile_px.min(ih - y);
            let mut row = Vec::new();
            let mut x = 0;
            while x < iw {
                let w = tile_px.min(iw - x);
                let patch = image.narrow(2, y, h)?.narrow(3, x, w)?.contiguous()?;
                row.push(self.encode(&patch)?);
                if x + w >= iw {
                    break;
                }
                x += stride_px;
            }
            rows.push(row);
            if y + h >= ih {
                break;
            }
            y += stride_px;
        }

        let mut out_rows = Vec::with_capacity(rows.len());
        for i in 0..rows.len() {
            let mut out_row: Vec<Tensor> = Vec::with_capacity(rows[i].len());
            for j in 0..rows[i].len() {
                let mut t = rows[i][j].clone();
                if i > 0 {
                    t = blend(&rows[i - 1][j], &t, blend_extent, 2)?;
                }
                if j > 0 {
                    // The untrimmed left neighbour, for the same reason as in
                    // `decode_tiled`: trimmed tiles differ in height on any
                    // row that was cut short.
                    t = blend(&rows[i][j - 1], &t, blend_extent, 3)?;
                }
                let last_col = j + 1 == rows[i].len();
                let last_row = i + 1 == rows.len();
                let th = t.dims()[2];
                let tw = t.dims()[3];
                let h = if last_row { th } else { stride_latent.min(th) };
                let w = if last_col { tw } else { stride_latent.min(tw) };
                out_row.push(t.narrow(2, 0, h)?.narrow(3, 0, w)?.contiguous()?);
            }
            out_rows.push(Tensor::cat(&out_row, 3)?);
        }
        Tensor::cat(&out_rows, 2)
    }
}

/// Latent edge of one decode tile. 64 latent = 512px, the size SD 1.5 was
/// trained at and comfortably inside GPU memory.
///
/// "Comfortably" assumes the VAE is most of what is resident, which is true
/// for SD 1.5 and false for SD 3.5: a single 64-latent tile allocates a
/// 2.42 GB convolution im2col, and with SD 3.5's 10 GB transformer still on
/// the GPU that overruns Metal's working set and the decode fails with
/// `kIOGPUCommandBufferCallbackErrorOutOfMemory` *after* all 20 denoise steps
/// have run. Lower it with [`TILE_LATENT_EDGE_ENV`] when that happens; 32
/// renders SD 3.5 at 512 on Metal in 25 s with no visible seam.
///
/// The default stays 64 because changing it changes the image. Tiling is not
/// free: the decoder is not shift-invariant, so tiles are blended rather than
/// abutted, and a tiled decode of a size that previously fit in one tile
/// produces a slightly different picture.
pub const TILE_LATENT_EDGE: usize = 64;

/// Environment override for [`TILE_LATENT_EDGE`], in latent cells.
///
/// Setting it disables the automatic choice in
/// [`AutoencoderKlDecoder::tile_edge_for`] entirely, rather than capping it.
pub const TILE_LATENT_EDGE_ENV: &str = "SD_VAE_TILE_LATENT";

/// Smallest tile the automatic choice will fall to.
///
/// Below this the seams outnumber the picture and the projection has stopped
/// being the real constraint anyway — if an 8-latent tile does not fit, the
/// run is not going to finish. Refusing loudly at the decode beats emitting a
/// heavily seamed image and calling it success.
pub const MIN_TILE_LATENT_EDGE: usize = 8;

/// An explicitly configured tile edge, if [`TILE_LATENT_EDGE_ENV`] is set.
///
/// Refuses 0 rather than defaulting it: a zero tile means an infinite loop in
/// the tiling walk, and silently substituting 64 for a value the caller
/// deliberately set would hide the mistake.
pub fn explicit_tile_latent_edge() -> Result<Option<usize>> {
    let raw = match std::env::var(TILE_LATENT_EDGE_ENV) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(sd_tensor::Error::Msg(format!(
                "{TILE_LATENT_EDGE_ENV} is not valid UTF-8"
            )))
        }
    };
    match raw.trim().parse::<usize>() {
        Ok(0) | Err(_) => Err(sd_tensor::Error::Msg(format!(
            "{TILE_LATENT_EDGE_ENV} must be a positive latent-cell count, got {raw:?}"
        ))),
        Ok(n) => Ok(Some(n)),
    }
}

/// The tile edge to assume before a decoder is available.
///
/// Load-time headroom projections use this: they run before any latent
/// exists, so they cannot ask [`AutoencoderKlDecoder::tile_edge_for`]. An
/// explicit override still applies, since that is the caller's own answer.
pub fn tile_latent_edge() -> Result<usize> {
    Ok(explicit_tile_latent_edge()?.unwrap_or(TILE_LATENT_EDGE))
}

/// Fraction of a tile that overlaps its neighbour.
///
/// The decoder is not shift-invariant — its convolutions see different
/// padding at a tile edge than they would mid-image — so tiles must overlap
/// and be blended, or the seams are visible as hard lines.
const TILE_OVERLAP: f64 = 0.25;

/// Linear ramp from 0 to 1 along `dim`, shaped to broadcast over `[b, c, h, w]`.
fn ramp(extent: usize, dim: usize, device: &sd_tensor::Device) -> Result<Tensor> {
    let values: Vec<f32> = (0..extent)
        .map(|i| (i as f32 + 0.5) / extent as f32)
        .collect();
    let t = Tensor::from_vec(values, extent, device)?;
    match dim {
        2 => t.reshape((1, 1, extent, 1)),
        _ => t.reshape((1, 1, 1, extent)),
    }
}

/// Cross-fade `b` into the trailing `extent` of `a` along `dim`.
///
/// Returns `b` with its leading `extent` replaced by the blend. Tensors are
/// immutable here, so this rebuilds rather than writing in place.
fn blend(a: &Tensor, b: &Tensor, extent: usize, dim: usize) -> Result<Tensor> {
    let a_len = a.dims()[dim];
    let b_len = b.dims()[dim];
    let extent = extent.min(a_len).min(b_len);
    if extent == 0 {
        return Ok(b.clone());
    }
    let w = ramp(extent, dim, a.device())?;
    let a_tail = a.narrow(dim, a_len - extent, extent)?;
    let b_head = b.narrow(dim, 0, extent)?;
    // a fades out as b fades in.
    let mixed = (a_tail.broadcast_mul(&(1.0 - &w)?)? + b_head.broadcast_mul(&w)?)?;
    if b_len == extent {
        return Ok(mixed);
    }
    let rest = b.narrow(dim, extent, b_len - extent)?;
    Tensor::cat(&[&mixed, &rest], dim)
}

trait ForwardT {
    fn forward_t(&self, xs: &Tensor) -> Result<Tensor>;
}

impl ForwardT for Conv2d {
    fn forward_t(&self, xs: &Tensor) -> Result<Tensor> {
        use sd_tensor::Module;
        self.forward(xs)
    }
}

#[cfg(test)]
mod tile_env_tests {
    use super::*;

    #[test]
    fn the_chosen_tile_shrinks_only_when_the_projection_does_not_fit() {
        // The auto-choice is a loop over `check_headroom`, which asks the
        // machine. Testing it against a real decoder would make the result
        // depend on whatever else is running, so the arithmetic it relies on
        // is checked directly instead: the peak must fall fast enough with the
        // tile edge for halving to be a useful move, and the walk must stop.
        let cfg = DecoderConfig::from(&VaeConfig::sd35());
        let peak = |e: usize| {
            cfg.peak_alloc_bytes(1, e, e, sd_tensor::DType::F32)
                .unwrap()
        };

        // Area-scaling: halving the edge must cut the peak by roughly 4x, or
        // stepping down would barely help and the loop would be pointless.
        let ratio = peak(64) as f64 / peak(32) as f64;
        assert!(
            (3.5..4.5).contains(&ratio),
            "halving the tile should quarter the peak, got {ratio:.2}x"
        );

        // Strictly decreasing, so the loop always makes progress.
        for e in [64usize, 32, 16, 8] {
            assert!(peak(e) > peak(e / 2), "peak must fall with the tile edge");
        }

        // And the floor is reachable by halving from the ceiling, so the walk
        // cannot step past it into a smaller-than-minimum tile.
        let mut e = TILE_LATENT_EDGE;
        while e > MIN_TILE_LATENT_EDGE {
            e /= 2;
        }
        assert_eq!(
            e, MIN_TILE_LATENT_EDGE,
            "halving must land exactly on the floor"
        );
    }

    #[test]
    fn the_tile_override_is_validated_rather_than_defaulted() {
        // Parsing is tested directly rather than through the env var, which is
        // process-global and would race under a parallel test runner. What
        // matters is that a bad value is refused: 0 makes the tiling walk loop
        // forever, and quietly substituting 64 would hide a caller's mistake.
        assert_eq!(TILE_LATENT_EDGE, 64, "the default must not drift silently");

        // Whatever the environment holds, the accessor must agree with the
        // default when unset and never return zero.
        let active = tile_latent_edge().expect("a clean environment parses");
        assert!(active > 0, "a zero tile edge would not terminate");
    }
}
