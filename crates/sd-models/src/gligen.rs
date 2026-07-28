//! GLIGEN: grounded generation — "put this thing *here*".
//!
//! The only widely-supported way to say *where*. Text cannot address position
//! reliably, and a ControlNet needs a picture of the layout; GLIGEN takes
//! boxes and phrases directly.
//!
//! Two parts. [`PositionNet`] here turns `(box, phrase)` pairs into grounding
//! tokens, and a gated self-attention inside each transformer block lets the
//! image attend over them — see [`crate::unet::gligen`].
//!
//! # The Fourier embedding, and the axis order that is easy to get wrong
//!
//! A box is four numbers in `[0, 1]`. Feeding those to an MLP directly gives
//! it almost nothing to work with, so each coordinate is expanded into
//! sinusoids at eight frequencies — the same trick as a timestep embedding,
//! applied to space.
//!
//! The resulting axes are `(coordinate, frequency, sin/cos)` and are flattened
//! as **`(frequency, sin/cos, coordinate)`**. That permute is the whole
//! subtlety: any ordering produces 64 numbers and loads against the same
//! weights, and only the right one lines up with what the MLP was trained on.

use sd_tensor::nn::{linear, Linear, VarBuilder};
use sd_tensor::{ops, Module, Result, Tensor};

/// Frequencies per coordinate. 8 in every published GLIGEN.
pub const FOURIER_FREQS: usize = 8;
/// `freqs * 2 (sin, cos) * 4 (xyxy)`.
pub const POSITION_DIM: usize = FOURIER_FREQS * 2 * 4;

/// Expand boxes `[b, n, 4]` in `[0, 1]` to `[b, n, 64]` of sinusoids.
pub fn fourier_embed(boxes: &Tensor) -> Result<Tensor> {
    let (b, n, coords) = boxes.dims3()?;
    if coords != 4 {
        return Err(sd_tensor::Error::Msg(format!(
            "a box is 4 numbers (x0, y0, x1, y1), got {coords}"
        )));
    }
    // 100^(i/dim): a geometric ladder, like a timestep embedding's.
    let freqs: Vec<f32> = (0..FOURIER_FREQS)
        .map(|i| 100f32.powf(i as f32 / FOURIER_FREQS as f32))
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, 1, 1, FOURIER_FREQS), boxes.device())?
        .to_dtype(boxes.dtype())?;

    // [b, n, 4, 1] * [1, 1, 1, f] -> [b, n, 4, f]
    let scaled = boxes.unsqueeze(3)?.broadcast_mul(&freqs)?;
    // -> [b, n, 4, f, 2] with sin and cos last
    let stacked = Tensor::stack(&[scaled.sin()?, scaled.cos()?], 4)?;
    // **The permute.** (b, n, coord, freq, sincos) -> (b, n, freq, sincos, coord)
    stacked
        .permute((0, 1, 3, 4, 2))?
        .contiguous()?
        .reshape((b, n, POSITION_DIM))
}

/// Turns `(box, phrase)` pairs into grounding tokens for the UNet.
#[derive(Debug)]
pub struct PositionNet {
    linear_0: Linear,
    linear_1: Linear,
    linear_2: Linear,
    /// Stands in for a phrase where `mask` is 0 — a *learned* absence, not
    /// zeros. Padding with zeros instead reads as "a phrase whose embedding
    /// happens to be zero", which is a different thing and one the model was
    /// never shown.
    null_positive: Tensor,
    null_position: Tensor,
}

impl PositionNet {
    /// `positive_len` is the phrase embedding width (768 for SD 1.5's CLIP),
    /// `out_dim` the UNet's cross-attention width.
    pub fn new(positive_len: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let vb_l = vb.pp("linears");
        Ok(Self {
            // 0, 2, 4 — the odd indices are SiLU, which carries no weights.
            linear_0: linear(positive_len + POSITION_DIM, 512, vb_l.pp("0"))?,
            linear_1: linear(512, 512, vb_l.pp("2"))?,
            linear_2: linear(512, out_dim, vb_l.pp("4"))?,
            null_positive: vb.get(positive_len, "null_positive_feature")?,
            null_position: vb.get(POSITION_DIM, "null_position_feature")?,
        })
    }

    /// `boxes` `[b, n, 4]`, `masks` `[b, n]` in `{0, 1}`, `phrases`
    /// `[b, n, positive_len]`. Returns grounding tokens `[b, n, out_dim]`.
    ///
    /// `masks` exists so a fixed-size batch can carry fewer boxes than it has
    /// slots: a 0 replaces both the phrase and the position with their learned
    /// nulls, which is how the model was trained to see "no object here".
    pub fn forward(&self, boxes: &Tensor, masks: &Tensor, phrases: &Tensor) -> Result<Tensor> {
        let mask = masks.unsqueeze(2)?.to_dtype(boxes.dtype())?;
        let inverse = (1.0 - &mask)?;

        let position = fourier_embed(boxes)?;
        let position = (position.broadcast_mul(&mask)?
            + self
                .null_position
                .reshape((1, 1, POSITION_DIM))?
                .broadcast_mul(&inverse)?)?;

        let positive_len = phrases.dim(2)?;
        let phrases = (phrases.broadcast_mul(&mask)?
            + self
                .null_positive
                .reshape((1, 1, positive_len))?
                .broadcast_mul(&inverse)?)?;

        // Phrase first, then position — the order the weights expect, and one
        // that produces a working shape either way.
        let xs = Tensor::cat(&[&phrases, &position], 2)?;
        let xs = ops::silu(&self.linear_0.forward(&xs)?)?;
        let xs = ops::silu(&self.linear_1.forward(&xs)?)?;
        self.linear_2.forward(&xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sd_tensor::Device;

    #[test]
    fn a_box_expands_to_sixty_four_numbers() {
        let dev = Device::Cpu;
        let boxes = Tensor::new(&[[[0.1f32, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]], &dev).unwrap();
        let out = fourier_embed(&boxes).unwrap();
        assert_eq!(out.dims(), &[1, 2, POSITION_DIM]);
    }

    #[test]
    fn the_lowest_frequency_is_the_coordinate_itself() {
        // freq 0 is 100^0 = 1, so the first sin/cos pair is sin(x), cos(x) of
        // the raw coordinate. That pins both the ladder's base and the
        // flattening order: with the axes permuted differently, element 0
        // would be some other coordinate's sine.
        let dev = Device::Cpu;
        let boxes = Tensor::new(&[[[0.25f32, 0.5, 0.75, 1.0]]], &dev).unwrap();
        let v = fourier_embed(&boxes)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Layout is (freq, sin/cos, coord): the first four are sin at freq 0
        // for each of the four coordinates.
        for (i, x) in [0.25f32, 0.5, 0.75, 1.0].iter().enumerate() {
            assert!(
                (v[i] - x.sin()).abs() < 1e-6,
                "element {i} is not sin(x{i})"
            );
        }
        // Then the four cosines.
        for (i, x) in [0.25f32, 0.5, 0.75, 1.0].iter().enumerate() {
            assert!(
                (v[4 + i] - x.cos()).abs() < 1e-6,
                "element {} is not cos",
                4 + i
            );
        }
    }
}
