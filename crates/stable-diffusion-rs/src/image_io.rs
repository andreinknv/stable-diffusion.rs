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
    // `round`, not `as u8`, which truncates. The two differ by one level on
    // any value the float arithmetic lands just under — and `b/127.5 - 1`
    // followed by `(x+1)*127.5` does exactly that, so a load-and-save round
    // trip was darkening pixels by 1/255 rather than reproducing them. Found
    // by an inpaint whose untouched region was supposed to be bit-identical
    // and was off by one everywhere.
    let bytes = flat.into_iter().map(|v| v.round() as u8).collect();
    Ok((w as u32, h as u32, bytes))
}

/// Read an image file into a `[1, 3, h, w]` tensor in `[-1, 1]`.
///
/// The inverse of [`tensor_to_rgb8`], for img2img. Resizes to `(width,
/// height)` with a Lanczos filter — both must be multiples of 8, since the
/// encoder reduces by that factor and a non-multiple silently truncates.
pub fn load_image<P: AsRef<std::path::Path>>(
    path: P,
    width: u32,
    height: u32,
    device: &sd_tensor::Device,
) -> Result<Tensor> {
    let img = image::open(path.as_ref())
        .map_err(|e| sd_tensor::Error::Msg(format!("failed to read image: {e}")))?;
    let img = img
        .resize_exact(width, height, image::imageops::FilterType::Lanczos3)
        .to_rgb8();

    // [0, 255] -> [-1, 1], interleaved RGB -> CHW.
    let data: Vec<f32> = img
        .as_raw()
        .iter()
        .map(|&b| b as f32 / 127.5 - 1.0)
        .collect();
    Tensor::from_vec(data, (height as usize, width as usize, 3), device)?
        .permute((2, 0, 1))?
        .contiguous()?
        .unsqueeze(0)
}

/// Write a decoder output tensor to a PNG.
/// Load an inpainting mask as `[1, 1, height, width]` in `[0, 1]`.
///
/// **White means repaint.** That is the convention every published tool uses —
/// diffusers, A1111, ComfyUI — and inverting it is the single most likely way
/// to be confused by this feature, since a mask and its inverse are both
/// plausible pictures and the result of using the wrong one is simply the
/// wrong region changing.
///
/// Resized with nearest-neighbour rather than the Lanczos used for images: a
/// mask is a decision per pixel, not a signal, and a smooth filter invents
/// grey values at every edge that then read as "half repaint this".
pub fn load_mask<P: AsRef<std::path::Path>>(
    path: P,
    width: u32,
    height: u32,
    device: &sd_tensor::Device,
) -> Result<Tensor> {
    let img = image::open(path.as_ref())
        .map_err(|e| sd_tensor::Error::Msg(format!("failed to read mask: {e}")))?;
    let img = img
        .resize_exact(width, height, image::imageops::FilterType::Nearest)
        .to_luma8();
    let data: Vec<f32> = img.as_raw().iter().map(|&b| b as f32 / 255.0).collect();
    Tensor::from_vec(data, (1, 1, height as usize, width as usize), device)
}

/// Read an image at its native size into `[1, 3, h, w]` in `[0, 1]`.
///
/// Two differences from [`load_image`], both deliberate. **No resize**: an
/// upscaler's input size is the thing being scaled, so resizing first would
/// throw away exactly what it is for. And **`[0, 1]`, not `[-1, 1]`**:
/// Real-ESRGAN was trained on the unsigned range, and handing it signed values
/// returns a washed-out image with no error to notice.
pub fn load_rgb_unit<P: AsRef<std::path::Path>>(
    path: P,
    device: &sd_tensor::Device,
) -> Result<Tensor> {
    let img = image::open(path.as_ref())
        .map_err(|e| sd_tensor::Error::Msg(format!("failed to read image: {e}")))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    let data: Vec<f32> = img.as_raw().iter().map(|&b| b as f32 / 255.0).collect();
    Tensor::from_vec(data, (h as usize, w as usize, 3), device)?
        .permute((2, 0, 1))?
        .contiguous()?
        .unsqueeze(0)
}

/// Composite `generated` over `original` where `mask` is 1, in pixel space.
///
/// The last step of an inpaint, and not cosmetic. Blending in latent space
/// keeps the *encoded* original outside the mask, and a VAE round trip is not
/// lossless — so without this the untouched region comes back subtly altered,
/// which is exactly what an inpaint is supposed not to do. All three tensors
/// are `[1, c, h, w]`; the mask broadcasts over channels.
pub fn composite(generated: &Tensor, original: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let keep = (1.0 - mask)?;
    generated.broadcast_mul(mask)? + original.broadcast_mul(&keep)?
}

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
        // -1 -> 0, 0 -> 128, 1 -> 255. Zero sits exactly halfway between two
        // 8-bit levels, so it is the one input where rounding and truncation
        // disagree by choice rather than by float error; 128 is what
        // diffusers' `.round()` gives too.
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
        assert_eq!(px[1], 128); // G at (0,0) from 0
        assert_eq!(px[2], 255); // B at (0,0) from 1
    }

    #[test]
    fn rejects_wrong_channel_count() {
        let t = Tensor::zeros((4, 2, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(tensor_to_rgb8(&t).is_err());
    }

    #[test]
    fn every_u8_level_survives_a_round_trip() {
        // The bug this guards: `v as u8` truncates, and `b/127.5 - 1` followed
        // by `(x+1)*127.5` lands just under the integer often enough that a
        // load-and-save darkened most pixels by one level. Found by an inpaint
        // whose untouched region was supposed to come back bit-identical.
        let levels: Vec<f32> = (0..256).map(|b| b as f32 / 127.5 - 1.0).collect();
        let mut chw = Vec::with_capacity(256 * 3);
        for _ in 0..3 {
            chw.extend_from_slice(&levels);
        }
        let t = Tensor::from_vec(chw, (1, 3, 1, 256), &Device::Cpu).unwrap();
        let (_, _, px) = tensor_to_rgb8(&t).unwrap();
        for (i, rgb) in px.chunks(3).enumerate() {
            assert_eq!(rgb[0] as usize, i, "level {i} did not survive");
        }
    }

    #[test]
    fn compositing_takes_the_generated_pixel_only_where_the_mask_is_set() {
        let dev = Device::Cpu;
        let gen = Tensor::ones((1, 3, 1, 2), DType::F32, &dev).unwrap();
        let orig = (Tensor::ones((1, 3, 1, 2), DType::F32, &dev).unwrap() * -1.0).unwrap();
        // Keep the first pixel, repaint the second.
        let mask = Tensor::from_vec(vec![0f32, 1f32], (1, 1, 1, 2), &dev).unwrap();

        let out = composite(&gen, &orig, &mask).unwrap();
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for c in 0..3 {
            assert_eq!(v[c * 2], -1.0, "masked-out pixel must be the original");
            assert_eq!(v[c * 2 + 1], 1.0, "masked-in pixel must be generated");
        }
    }

    #[test]
    fn clamps_out_of_range_values() {
        let t = Tensor::new(&[[[-5f32]], [[5f32]], [[0f32]]], &Device::Cpu).unwrap();
        let (_, _, px) = tensor_to_rgb8(&t).unwrap();
        assert_eq!(px[0], 0);
        assert_eq!(px[1], 255);
    }
}
