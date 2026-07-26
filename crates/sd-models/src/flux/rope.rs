//! Flux's positional encoding: axis-wise rotary embeddings.
//!
//! Unlike SD's UNet, which gets position implicitly from convolution, Flux is
//! a pure transformer and every token carries an explicit `(t, h, w)`
//! coordinate. Each axis is rotated independently with its own slice of the
//! head dimension — `[16, 56, 56]`, summing to the 128-wide head — and the
//! results are concatenated.
//!
//! The rotation is stored as an explicit 2x2 matrix per frequency rather than
//! the complex-number form used elsewhere, because that is what the reference
//! implementation does and matching it exactly matters more here than being
//! clever: a transposed rotation is still a rotation, so it produces a
//! coherent image with the geometry subtly wrong.

use sd_tensor::{DType, Device, Result, Tensor};

/// Rotary frequencies for one axis.
///
/// `pos` holds the coordinate of every token along this axis. Returns
/// `[.., n, dim/2, 2, 2]` — a 2x2 rotation per token per frequency.
fn rope_axis(pos: &Tensor, dim: usize, theta: f64) -> Result<Tensor> {
    let device = pos.device();
    let half = dim / 2;

    // omega[i] = theta^(-2i/dim). Built on the host in f64: the exponent
    // spans several orders of magnitude and f32 loses the low frequencies,
    // which are the ones that encode long-range position.
    let omega: Vec<f32> = (0..half)
        .map(|i| (1.0 / theta.powf(2.0 * i as f64 / dim as f64)) as f32)
        .collect();
    let omega = Tensor::from_vec(omega, (1, half), device)?.to_dtype(DType::F32)?;

    // [.., n] x [half] -> [.., n, half]
    let pos = pos.to_dtype(DType::F32)?;
    let dims = pos.dims().to_vec();
    let flat = pos.flatten_all()?.reshape((pos.elem_count(), 1))?;
    let angles = flat.broadcast_mul(&omega)?;

    let (cos, sin) = (angles.cos()?, angles.sin()?);
    // Row-major 2x2: [[cos, -sin], [sin, cos]].
    let stacked = Tensor::stack(&[&cos, &sin.neg()?, &sin, &cos], 2)?;

    let mut shape = dims;
    shape.extend_from_slice(&[half, 2, 2]);
    stacked.reshape(shape)
}

/// Rotary embeddings for `[batch, seq, n_axes]` integer coordinates.
///
/// Returns `[batch, 1, seq, head_dim/2, 2, 2]`; the singleton broadcasts over
/// heads. `axes_dim` must sum to the head dimension.
pub fn embed_nd(ids: &Tensor, axes_dim: &[usize], theta: f64) -> Result<Tensor> {
    let n_axes = ids.dim(ids.rank() - 1)?;
    if n_axes != axes_dim.len() {
        return Err(sd_tensor::Error::Msg(format!(
            "ids carry {n_axes} axes but axes_dim has {} entries",
            axes_dim.len()
        )));
    }

    let mut per_axis = Vec::with_capacity(n_axes);
    for (i, &dim) in axes_dim.iter().enumerate() {
        let pos = ids.narrow(ids.rank() - 1, i, 1)?.squeeze(ids.rank() - 1)?;
        per_axis.push(rope_axis(&pos, dim, theta)?);
    }
    // Concatenate along the frequency axis, which is 3 back from the end
    // (freq, 2, 2). This is the axis the reference calls -3.
    let rank = per_axis[0].rank();
    let joined = Tensor::cat(&per_axis, rank - 3)?;
    joined.unsqueeze(1)
}

/// Apply rotary embeddings to `q` and `k`, both `[b, heads, seq, head_dim]`.
pub fn apply_rope(q: &Tensor, k: &Tensor, freqs: &Tensor) -> Result<(Tensor, Tensor)> {
    Ok((rotate(q, freqs)?, rotate(k, freqs)?))
}

fn rotate(xs: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let dtype = xs.dtype();
    let dims = xs.dims().to_vec();
    let (_, _, _, head_dim) = xs.dims4()?;

    // Pair up adjacent components: [.., head_dim] -> [.., head_dim/2, 2].
    let xs = xs
        .to_dtype(DType::F32)?
        .reshape((dims[0], dims[1], dims[2], head_dim / 2, 2))?;
    let x0 = xs.narrow(4, 0, 1)?;
    let x1 = xs.narrow(4, 1, 1)?;

    // freqs is [b, 1, seq, head_dim/2, 2, 2]; take the two matrix columns.
    let f = freqs.to_dtype(DType::F32)?;
    let c0 = f.narrow(5, 0, 1)?.squeeze(5)?;
    let c1 = f.narrow(5, 1, 1)?.squeeze(5)?;

    let out = (x0.broadcast_mul(&c0)? + x1.broadcast_mul(&c1)?)?;
    out.reshape(dims)?.to_dtype(dtype)
}

/// Token coordinates for a packed latent of `h x w` patches.
///
/// Axis 0 is time and stays zero for images; axes 1 and 2 are the patch row
/// and column. Text tokens get all-zero ids, which is what makes them
/// position-free relative to the image.
pub fn image_ids(batch: usize, h: usize, w: usize, device: &Device) -> Result<Tensor> {
    let mut v = Vec::with_capacity(h * w * 3);
    for row in 0..h {
        for col in 0..w {
            v.push(0f32);
            v.push(row as f32);
            v.push(col as f32);
        }
    }
    Tensor::from_vec(v, (1, h * w, 3), device)?
        .broadcast_as((batch, h * w, 3))?
        .contiguous()
}

/// All-zero ids for `n` text tokens.
pub fn text_ids(batch: usize, n: usize, device: &Device) -> Result<Tensor> {
    Tensor::zeros((batch, n, 3), DType::F32, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    const AXES: [usize; 3] = [16, 56, 56];

    #[test]
    fn embed_shape_matches_the_head_dimension() {
        let dev = Device::Cpu;
        let ids = image_ids(1, 4, 4, &dev).unwrap();
        let pe = embed_nd(&ids, &AXES, 10_000.0).unwrap();
        // [batch, 1 (heads), seq, head_dim/2, 2, 2]
        assert_eq!(pe.dims(), &[1, 1, 16, 64, 2, 2]);
        assert_eq!(AXES.iter().sum::<usize>(), 128, "axes must fill the head");
    }

    #[test]
    fn rotation_preserves_norm() {
        // A rotation must not change vector length. This is the property that
        // catches a malformed 2x2 — a matrix that is *nearly* a rotation
        // still yields a plausible image.
        let dev = Device::Cpu;
        let ids = image_ids(1, 4, 4, &dev).unwrap();
        let pe = embed_nd(&ids, &AXES, 10_000.0).unwrap();
        let q = Tensor::rand(-1f32, 1f32, (1, 2, 16, 128), &dev).unwrap();
        let (rq, _) = apply_rope(&q, &q, &pe).unwrap();

        let before = q.sqr().unwrap().sum_keepdim(3).unwrap();
        let after = rq.sqr().unwrap().sum_keepdim(3).unwrap();
        let err = (before - after)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(err < 1e-4, "rope changed vector norms by {err}");
    }

    #[test]
    fn position_zero_is_the_identity() {
        // At position 0 every angle is 0, so the rotation is the identity and
        // the vector must come back unchanged. Catches a sign error or a
        // transposed matrix, both of which leave norms intact.
        let dev = Device::Cpu;
        let ids = text_ids(1, 8, &dev).unwrap();
        let pe = embed_nd(&ids, &AXES, 10_000.0).unwrap();
        let q = Tensor::rand(-1f32, 1f32, (1, 2, 8, 128), &dev).unwrap();
        let (rq, _) = apply_rope(&q, &q, &pe).unwrap();
        let err = (&q - &rq)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(err < 1e-6, "position 0 should not rotate, moved by {err}");
    }

    #[test]
    fn different_positions_rotate_differently() {
        let dev = Device::Cpu;
        let ids = image_ids(1, 2, 2, &dev).unwrap();
        let pe = embed_nd(&ids, &AXES, 10_000.0).unwrap();
        let q = Tensor::ones((1, 1, 4, 128), DType::F32, &dev).unwrap();
        let (rq, _) = apply_rope(&q, &q, &pe).unwrap();
        let v = rq.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (first, second) = (&v[..128], &v[128..256]);
        let diff = first
            .iter()
            .zip(second)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            diff > 1e-3,
            "tokens at different positions must rotate differently, got {diff:.3e}"
        );
    }

    #[test]
    fn image_ids_enumerate_row_major() {
        let dev = Device::Cpu;
        let ids = image_ids(1, 2, 3, &dev).unwrap();
        let v = ids.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // (t, h, w) per token, row-major over a 2x3 grid.
        let want = [
            0., 0., 0., 0., 0., 1., 0., 0., 2., 0., 1., 0., 0., 1., 1., 0., 1., 2.,
        ];
        assert_eq!(v, want, "patch ordering must match the packing order");
    }
}
