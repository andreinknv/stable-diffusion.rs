//! Converting model output tensors to image files.

use sd_tensor::{DType, Result, Tensor};

/// Convert a decoder output `[b, 3, h, w]` in `[-1, 1]` to RGB8 bytes.
///
/// Returns `(width, height, rgb_bytes)` for the first image in the batch.
/// Values outside `[-1, 1]` are clamped, which is normal for VAE output.
pub fn tensor_to_rgb8(xs: &Tensor) -> Result<(u32, u32, Vec<u8>)> {
    let xs = if xs.rank() == 4 {
        xs.get(0)?
    } else {
        xs.clone()
    };
    let (c, h, w) = xs.dims3()?;
    if c != 3 {
        return Err(sd_tensor::Error::Msg(format!(
            "expected 3 channels, got {c}"
        )));
    }
    // [-1, 1] -> [0, 255]
    let xs = ((xs.to_dtype(DType::F32)? + 1.0)? * 127.5)?.clamp(0.0, 255.0)?;
    // CHW -> HWC for interleaved RGB output.
    let xs = xs.permute((1, 2, 0))?.contiguous()?;
    let flat = xs.flatten_all()?.to_vec1::<f32>()?;
    let bytes = flat.into_iter().map(|v| v as u8).collect();
    Ok((w as u32, h as u32, bytes))
}

/// Write a decoder output tensor to a PNG.
pub fn save_png<P: AsRef<std::path::Path>>(xs: &Tensor, path: P) -> Result<()> {
    let (w, h, bytes) = tensor_to_rgb8(xs)?;
    let buf = image::RgbImage::from_raw(w, h, bytes)
        .ok_or_else(|| sd_tensor::Error::Msg("RGB buffer size mismatch".to_string()))?;
    buf.save(path.as_ref())
        .map_err(|e| sd_tensor::Error::Msg(format!("failed to write image: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sd_tensor::Device;

    #[test]
    fn maps_signed_range_to_rgb8() {
        // -1 -> 0, 0 -> 127, 1 -> 255
        let t = Tensor::new(
            &[
                [[-1f32, 0.0], [1.0, -1.0]],
                [[0f32, 1.0], [-1.0, 0.0]],
                [[1f32, -1.0], [0.0, 1.0]],
            ],
            &Device::Cpu,
        )
        .unwrap();
        let (w, h, px) = tensor_to_rgb8(&t).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(px.len(), 12);
        assert_eq!(px[0], 0); // R at (0,0) from -1
        assert_eq!(px[1], 127); // G at (0,0) from 0
        assert_eq!(px[2], 255); // B at (0,0) from 1
    }

    #[test]
    fn rejects_wrong_channel_count() {
        let t = Tensor::zeros((4, 2, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(tensor_to_rgb8(&t).is_err());
    }

    #[test]
    fn clamps_out_of_range_values() {
        let t = Tensor::new(&[[[-5f32]], [[5f32]], [[0f32]]], &Device::Cpu).unwrap();
        let (_, _, px) = tensor_to_rgb8(&t).unwrap();
        assert_eq!(px[0], 0);
        assert_eq!(px[1], 255);
    }
}
