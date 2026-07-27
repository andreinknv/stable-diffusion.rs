//! Convolution with optional circular padding, for seamless tiling.
//!
//! Padding every convolution circularly makes an image tile **by
//! construction**: the model is never shown an edge, so there is no seam to
//! score, blend or cut afterwards. That is a different thing from generating
//! normally and repairing the join — a repair rewrites pixels the model drew
//! deliberately, and this does not.
//!
//! # Why the mode is ambient rather than a parameter
//!
//! It has to reach *every* convolution in the UNet and the VAE — several
//! hundred, across five model files — and threading a flag through every
//! constructor would touch all of them for a property that is uniform. So it
//! is process state, set for the duration of a generation with
//! [`seamless`], which returns a guard that restores the previous mode on
//! drop.
//!
//! This is the one piece of ambient state in the crate and it is a deliberate
//! trade. It is safe for the usual reason a rendering mode is: it is read, not
//! accumulated. It is **not** safe to generate a tiling and a non-tiling image
//! concurrently on two threads of one process — hold the guard for the whole
//! generation and serialise, or run them in separate processes.
//!
//! # The axes are independent
//!
//! A scrolling parallax layer wants horizontal wrapping only; forcing vertical
//! wrap on it makes the sky bleed into the floor. [`seamless`] takes one flag
//! per axis for that reason.

use std::sync::atomic::{AtomicU8, Ordering};

use candle_core::{Module, Result, Tensor};

const WRAP_X: u8 = 0b01;
const WRAP_Y: u8 = 0b10;

static MODE: AtomicU8 = AtomicU8::new(0);

/// Which axes currently wrap, as `(x, y)`.
pub fn wrapping() -> (bool, bool) {
    let bits = MODE.load(Ordering::Relaxed);
    (bits & WRAP_X != 0, bits & WRAP_Y != 0)
}

/// Restores the previous wrapping mode when dropped.
#[must_use = "the mode reverts when this guard is dropped"]
pub struct SeamlessGuard(u8);

impl Drop for SeamlessGuard {
    fn drop(&mut self) {
        MODE.store(self.0, Ordering::Relaxed);
    }
}

/// Wrap convolutions on the given axes until the returned guard is dropped.
///
/// ```ignore
/// let _tiling = sd_tensor::conv::seamless(true, true);
/// let image = pipeline.run(&cfg)?;   // tiles in both directions
/// ```
pub fn seamless(x: bool, y: bool) -> SeamlessGuard {
    let bits = (u8::from(x) * WRAP_X) | (u8::from(y) * WRAP_Y);
    SeamlessGuard(MODE.swap(bits, Ordering::Relaxed))
}

/// Pad `xs` by `pad` on every side, wrapping on the requested axes.
///
/// Wrapping is `narrow` + `cat`: the last `pad` columns are prepended and the
/// first `pad` appended, so the convolution reads across the join as if the
/// image repeated. Axes that do not wrap get zeros, which is what a normal
/// padded convolution does.
pub fn pad_for_conv(xs: &Tensor, pad: usize, wrap_x: bool, wrap_y: bool) -> Result<Tensor> {
    if pad == 0 {
        return Ok(xs.clone());
    }
    let (_, _, h, w) = xs.dims4()?;

    let padded_x = if wrap_x {
        // A feature map narrower than the padding cannot wrap meaningfully —
        // it would need to repeat itself more than once. Deep blocks reach
        // 4x4 with pad 1, which is fine; this guards the pathological case
        // rather than expecting it.
        if pad > w {
            return Err(candle_core::Error::Msg(format!(
                "cannot wrap {pad} columns of a {w}-wide feature map"
            )));
        }
        let left = xs.narrow(3, w - pad, pad)?;
        let right = xs.narrow(3, 0, pad)?;
        Tensor::cat(&[&left, xs, &right], 3)?
    } else {
        xs.pad_with_zeros(3, pad, pad)?
    };

    if wrap_y {
        if pad > h {
            return Err(candle_core::Error::Msg(format!(
                "cannot wrap {pad} rows of a {h}-tall feature map"
            )));
        }
        let top = padded_x.narrow(2, h - pad, pad)?;
        let bottom = padded_x.narrow(2, 0, pad)?;
        Tensor::cat(&[&top, &padded_x, &bottom], 2)
    } else {
        padded_x.pad_with_zeros(2, pad, pad)
    }
}

/// A 2D convolution that honours the ambient wrapping mode.
///
/// Shadows `candle_nn::Conv2d` under the same name, so model code that already
/// imports `Conv2d` from this crate picks it up without changing.
///
/// One `usize` larger than candle's, which matters: this type is a variant in
/// several enums and a second stored convolution pushed them past clippy's
/// size-difference threshold. The no-padding view needed for the wrapped path
/// is built on demand instead — its weights are `Arc`, so constructing it is a
/// refcount bump, and it only happens when tiling is on.
///
/// The ordinary path forwards straight to the configured convolution, so it is
/// exactly as it was: no separate pad kernel, no cost for callers who never
/// tile.
#[derive(Debug, Clone)]
pub struct Conv2d {
    inner: candle_nn::Conv2d,
    pad: usize,
}

impl Conv2d {
    pub fn new(inner: candle_nn::Conv2d) -> Self {
        let pad = inner.config().padding;
        Self { inner, pad }
    }
}

impl Module for Conv2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (wrap_x, wrap_y) = wrapping();
        if self.pad == 0 || !(wrap_x || wrap_y) {
            return self.inner.forward(xs);
        }
        let padded = pad_for_conv(xs, self.pad, wrap_x, wrap_y)?;
        let unpadded = candle_nn::Conv2d::new(
            self.inner.weight().clone(),
            self.inner.bias().cloned(),
            candle_nn::Conv2dConfig {
                padding: 0,
                ..*self.inner.config()
            },
        );
        unpadded.forward(&padded)
    }
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vb: candle_nn::VarBuilder,
) -> Result<Conv2d> {
    candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vb).map(Conv2d::new)
}

pub fn conv2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vb: candle_nn::VarBuilder,
) -> Result<Conv2d> {
    candle_nn::conv2d_no_bias(in_channels, out_channels, kernel_size, cfg, vb).map(Conv2d::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn wrapping_copies_the_opposite_edge() {
        let dev = Device::Cpu;
        // 1x1x2x4, values 0..8 so each position is identifiable.
        let xs = Tensor::from_vec(
            (0..8u32).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 1, 2, 4),
            &dev,
        )
        .unwrap();
        let out = pad_for_conv(&xs, 1, true, false).unwrap();
        assert_eq!(out.dims(), &[1, 1, 4, 6], "y is zero-padded, x is wrapped");

        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Row 1 of the result is the original first row, wrapped: the last
        // column comes first and the first column last.
        let row = &v[6..12];
        assert_eq!(row, &[3.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn a_non_wrapped_axis_gets_zeros() {
        let dev = Device::Cpu;
        let xs = Tensor::ones((1, 1, 2, 2), DType::F32, &dev).unwrap();
        let out = pad_for_conv(&xs, 1, false, false).unwrap();
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v[0], 0.0, "corner must be zero when nothing wraps");
        assert_eq!(v.iter().sum::<f32>(), 4.0, "only the original ones remain");
    }

    #[test]
    fn the_guard_restores_the_previous_mode() {
        assert_eq!(wrapping(), (false, false));
        {
            let _outer = seamless(true, true);
            assert_eq!(wrapping(), (true, true));
            {
                let _inner = seamless(true, false);
                assert_eq!(wrapping(), (true, false));
            }
            assert_eq!(wrapping(), (true, true), "inner guard must restore");
        }
        assert_eq!(wrapping(), (false, false), "outer guard must restore");
    }
}
