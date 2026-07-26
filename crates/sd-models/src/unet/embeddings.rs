//! Timestep conditioning: the sinusoid, and the MLP that widens it.

use sd_tensor::nn::{linear, Linear};
use sd_tensor::{ops, DType, Module, Result, Tensor};

/// Sinusoidal timestep embedding, matching
/// `diffusers.models.embeddings.get_timestep_embedding` with
/// `flip_sin_to_cos=True`, `downscale_freq_shift=0`.
///
/// `timesteps` is `[batch]` (f32). Returns `[batch, dim]`.
///
/// The halves are **cos then sin** — that is what `flip_sin_to_cos=True`
/// means. The natural `[sin, cos]` order runs, produces the right shape, and
/// gives a wrong model.
pub fn timestep_embedding(timesteps: &Tensor, dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let device = timesteps.device();

    // exponent = -ln(10000) * arange(0, half) / half
    let exponent = (Tensor::arange(0u32, half as u32, device)?.to_dtype(DType::F32)?
        * (-(10000f64.ln()) / half as f64))?;
    let freqs = exponent.exp()?;

    // [batch, 1] * [1, half] -> [batch, half]
    let args = timesteps
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;

    Tensor::cat(&[args.cos()?, args.sin()?], 1)
}

/// `time_embedding.linear_1` -> silu -> `time_embedding.linear_2`.
#[derive(Debug)]
pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    /// SD 1.5: `in_dim = 320`, `out_dim = 1280`.
    ///
    /// `vb` is already rooted at `time_embedding`.
    pub fn new(in_dim: usize, out_dim: usize, vb: sd_tensor::VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: linear(in_dim, out_dim, vb.pp("linear_1"))?,
            // Both sides are out_dim: the MLP widens once and then stays.
            linear_2: linear(out_dim, out_dim, vb.pp("linear_2"))?,
        })
    }

    /// `[batch, in_dim]` -> `[batch, out_dim]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = ops::silu(&self.linear_1.forward(xs)?)?;
        self.linear_2.forward(&xs)
    }
}
