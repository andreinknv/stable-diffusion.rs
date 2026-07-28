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

/// [`load_rgb_unit`], resized to an exact size.
///
/// Squashes to fit. For a square target and a non-square source that changes
/// the aspect of everything in the frame — see [`load_clip_square`], which is
/// what CLIP's own preprocessing does instead.
pub fn load_rgb_unit_resized<P: AsRef<std::path::Path>>(
    path: P,
    width: u32,
    height: u32,
    device: &sd_tensor::Device,
) -> Result<Tensor> {
    let img = image::open(path.as_ref())
        .map_err(|e| sd_tensor::Error::Msg(format!("failed to read image: {e}")))?
        .resize_exact(width, height, image::imageops::FilterType::Lanczos3)
        .to_rgb8();
    let data: Vec<f32> = img.as_raw().iter().map(|&b| b as f32 / 255.0).collect();
    Tensor::from_vec(data, (height as usize, width as usize, 3), device)?
        .permute((2, 0, 1))?
        .contiguous()?
        .unsqueeze(0)
}

/// Read a reference image the way CLIP's own preprocessing does: `[1, 3, e, e]`
/// in `[0, 1]`.
///
/// **Shortest edge to `edge`, then a centre crop** — not a squash to square,
/// which is what [`load_rgb_unit_resized`] does and what both the IP-Adapter
/// and unCLIP paths used to do. On a square reference the two agree exactly;
/// on a 16:9 one the squash compresses everything horizontally by 1.8x, and
/// the tower was never shown images like that. `CLIPImageProcessor` ships
/// `do_resize` + `do_center_crop` for precisely this reason.
///
/// **Catmull-Rom, not Lanczos**, because `resample: 3` in the shipped
/// `preprocessor_config.json` is PIL's `BICUBIC` and Catmull-Rom is the
/// cubic filter closest to it. Lanczos is sharper and therefore *further*
/// from the reference, which is the opposite of what is wanted at a boundary
/// this precise.
pub fn load_clip_square<P: AsRef<std::path::Path>>(
    path: P,
    edge: u32,
    device: &sd_tensor::Device,
) -> Result<Tensor> {
    let img = image::open(path.as_ref())
        .map_err(|e| sd_tensor::Error::Msg(format!("failed to read image: {e}")))?;
    let (w, h) = (img.width().max(1), img.height().max(1));

    // Scale so the *shorter* side lands on `edge`; the longer one overhangs
    // and is cropped. Scaling the longer side instead would leave the shorter
    // one short of the crop and is the natural mistake here.
    let scale = edge as f64 / w.min(h) as f64;
    let (rw, rh) = (
        ((w as f64 * scale).round() as u32).max(edge),
        ((h as f64 * scale).round() as u32).max(edge),
    );
    let img = img
        .resize_exact(rw, rh, image::imageops::FilterType::CatmullRom)
        .to_rgb8();

    // Integer-divided, matching torchvision's centre crop: an odd overhang
    // leaves the extra pixel on the bottom-right.
    let (x0, y0) = ((rw - edge) / 2, (rh - edge) / 2);
    let mut data = Vec::with_capacity((edge * edge * 3) as usize);
    for y in 0..edge {
        for x in 0..edge {
            let p = img.get_pixel(x0 + x, y0 + y);
            data.extend(p.0.iter().map(|&b| b as f32 / 255.0));
        }
    }
    Tensor::from_vec(data, (edge as usize, edge as usize, 3), device)?
        .permute((2, 0, 1))?
        .contiguous()?
        .unsqueeze(0)
}

/// Resize a `[1, 3, h, w]` image in `[-1, 1]`, via Lanczos in pixel space.
///
/// Through 8-bit, which is worth knowing: the round trip quantises. That is
/// acceptable here because the result is immediately re-encoded by a VAE whose
/// own round trip loses more, and it keeps one resize implementation rather
/// than two.
pub fn resize_signed(image: &Tensor, width: u32, height: u32) -> Result<Tensor> {
    let (w, h, bytes) = tensor_to_rgb8(image)?;
    let buf = image::RgbImage::from_raw(w, h, bytes)
        .ok_or_else(|| sd_tensor::Error::Msg("RGB buffer size mismatch".to_string()))?;
    let resized =
        image::imageops::resize(&buf, width, height, image::imageops::FilterType::Lanczos3);
    let data: Vec<f32> = resized
        .as_raw()
        .iter()
        .map(|&b| b as f32 / 127.5 - 1.0)
        .collect();
    Tensor::from_vec(data, (height as usize, width as usize, 3), image.device())?
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

/// Write every image in a batch, numbered.
///
/// `out.png` with four frames becomes `out-000.png` .. `out-003.png`. A single
/// image keeps the name it was given, so nothing changes for the common case —
/// numbering one file would be a surprise, and callers script against the name
/// they passed.
///
/// Returns the paths written, in order.
pub fn save_batch<P: AsRef<std::path::Path>>(xs: &Tensor, path: P) -> Result<Vec<String>> {
    let path = path.as_ref();
    let count = if xs.rank() == 4 { xs.dim(0)? } else { 1 };
    if count <= 1 {
        save_png(xs, path)?;
        return Ok(vec![path.to_string_lossy().into_owned()]);
    }

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("frame");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("png");
    let mut written = Vec::with_capacity(count);
    for i in 0..count {
        // Zero-padded to three, so `ls` and any glob sort in frame order
        // rather than 1, 10, 2.
        let name = format!("{stem}-{i:03}.{ext}");
        let out = match path.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => dir.join(name),
            _ => std::path::PathBuf::from(name),
        };
        save_png(&xs.narrow(0, i, 1)?, &out)?;
        written.push(out.to_string_lossy().into_owned());
    }
    Ok(written)
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

    /// Write a `w x h` image whose leftmost `red_cols` columns are red and the
    /// rest white, and return its path.
    fn striped(name: &str, w: u32, h: u32, red_cols: u32) -> std::path::PathBuf {
        let mut img = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let p = if x < red_cols {
                    image::Rgb([255u8, 0, 0])
                } else {
                    image::Rgb([255u8, 255, 255])
                };
                img.put_pixel(x, y, p);
            }
        }
        let path = std::env::temp_dir().join(name);
        img.save(&path).expect("writing the test image");
        path
    }

    fn max_red_excess(t: &Tensor) -> f32 {
        // How far any pixel's red channel exceeds its green: 0 for white,
        // ~1 for red. A cheap "is there any red here at all".
        let (r, g) = (t.narrow(1, 0, 1).unwrap(), t.narrow(1, 1, 1).unwrap());
        (r - g)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .fold(0f32, f32::max)
    }

    #[test]
    fn a_clip_reference_is_cropped_rather_than_squashed() {
        // The whole point of the shortest-edge-plus-crop rule: on a wide
        // image, content near the edge is *outside* the frame the tower sees.
        // Squashing keeps it and changes every proportion instead, which is
        // what this crate used to do for both IP-Adapter and unCLIP.
        let dev = Device::Cpu;
        let wide = striped("sdrs-clip-crop-wide.png", 448, 224, 56);

        let cropped = load_clip_square(&wide, 224, &dev).expect("crop");
        assert_eq!(cropped.dims(), &[1, 3, 224, 224]);
        // Source columns 112..336 are all white, so no red survives.
        assert!(
            max_red_excess(&cropped) < 0.05,
            "the cropped frame still contains edge content"
        );

        let squashed = load_rgb_unit_resized(&wide, 224, 224, &dev).expect("squash");
        assert!(
            max_red_excess(&squashed) > 0.5,
            "the squashed frame should still contain the red stripe"
        );
        let _ = std::fs::remove_file(&wide);
    }

    #[test]
    fn a_square_clip_reference_keeps_everything() {
        // The other half of the property: a square source loses nothing, so
        // every reference image used in this repo's assets is unaffected by
        // the change.
        let dev = Device::Cpu;
        let square = striped("sdrs-clip-crop-square.png", 224, 224, 56);
        let out = load_clip_square(&square, 224, &dev).expect("crop");
        assert_eq!(out.dims(), &[1, 3, 224, 224]);
        assert!(
            max_red_excess(&out) > 0.5,
            "a square reference must not be cropped"
        );
        let _ = std::fs::remove_file(&square);
    }
}
