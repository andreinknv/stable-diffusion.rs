//! Image tensors that carry their range in the type.
//!
//! This project uses two pixel ranges and they are not interchangeable.
//! Everything that meets a VAE is `[-1, 1]`; everything that meets CLIP's
//! vision tower, or ESRGAN, is `[0, 1]`. Both are `[b, 3, h, w]` of `f32`, so
//! the compiler has never been able to tell them apart, and the failure is the
//! kind this codebase keeps a list of:
//!
//! > Feeding it a `[-1, 1]` image, which is what the rest of this crate uses,
//! > produces an embedding of the right shape describing the wrong picture.
//!
//! That comment sat on [`crate::clip::preprocess`] as the only defence.
//! [`UnitImage`] makes it a type error instead.
//!
//! # Why only one of the two ranges is a newtype
//!
//! Because only one direction is silent. A signed image handed to CLIP is
//! accepted and quietly wrong; the reverse — a `[0, 1]` image handed to the
//! VAE — is a washed-out picture that is obvious immediately. Wrapping the
//! range that fails loudly would be churn for its own sake, so the counterpart
//! is [`from_signed`] and [`into_signed`] rather than a second type.

use sd_tensor::{Result, Tensor};

/// A `[b, 3, h, w]` image in `[0, 1]`.
///
/// Construct it from something that is genuinely in that range —
/// [`crate::clip::preprocess`] and ESRGAN both assume it — or convert with
/// [`from_signed`].
#[derive(Debug, Clone)]
pub struct UnitImage(Tensor);

impl UnitImage {
    /// Wrap a tensor already in `[0, 1]`.
    ///
    /// Unchecked, and deliberately: clamping would hide the mistake this type
    /// exists to surface, and checking on every call would cost a device
    /// round trip. The point is that the *call site* has to say which range it
    /// believes it has.
    pub fn new(tensor: Tensor) -> Self {
        Self(tensor)
    }

    /// From a `[-1, 1]` image.
    pub fn from_signed(signed: &Tensor) -> Result<Self> {
        Ok(Self(((signed + 1.0)? * 0.5)?))
    }

    /// Back to `[-1, 1]`, for the VAE.
    pub fn into_signed(self) -> Result<Tensor> {
        (self.0 * 2.0)? - 1.0
    }

    pub fn tensor(&self) -> &Tensor {
        &self.0
    }

    pub fn dims(&self) -> &[usize] {
        self.0.dims()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sd_tensor::{DType, Device};

    #[test]
    fn the_two_ranges_round_trip() {
        let dev = Device::Cpu;
        let signed = Tensor::from_vec(vec![-1f32, 0.0, 1.0], (1, 3, 1, 1), &dev).unwrap();
        let unit = UnitImage::from_signed(&signed).unwrap();
        assert_eq!(
            unit.tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0.0, 0.5, 1.0]
        );
        let back = unit.into_signed().unwrap();
        let got = back.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in got.iter().zip([-1f32, 0.0, 1.0]) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn a_unit_image_keeps_its_shape() {
        let dev = Device::Cpu;
        let t = Tensor::zeros((2, 3, 8, 8), DType::F32, &dev).unwrap();
        assert_eq!(UnitImage::new(t).dims(), &[2, 3, 8, 8]);
    }
}
