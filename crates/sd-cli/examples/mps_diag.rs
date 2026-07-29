use anyhow::Result;
use sd_tensor::{mps, nn, DType, Module, Tensor, VarBuilder};
fn candle_conv(xs: &Tensor, k: &Tensor, ci: usize, co: usize) -> Result<Tensor> {
    let vb = VarBuilder::from_tensors([("weight".into(), k.clone())].into_iter().collect(), DType::F32, &xs.device().clone());
    let c = nn::conv2d_no_bias(ci, co, 3, nn::Conv2dConfig { padding: 1, ..Default::default() }, vb)?;
    Ok(c.forward(xs)?)
}
fn main() -> Result<()> {
    let dev = sd_tensor::device::best()?;
    let (ci, co) = (3usize, 3usize);   // square, so the transpose stays valid
    let mut rng = sd_tensor::rng::SeededRng::new(1);
    let xs = rng.randn((1, ci, 5, 5), &dev)?;
    let k = rng.randn((co, ci, 3, 3), &dev)?;

    let mps_out = mps::conv2d(&xs, &k, 1)?;
    let same = candle_conv(&xs, &k, ci, co)?;
    let swapped = candle_conv(&xs, &k.transpose(0, 1)?.contiguous()?, ci, co)?;
    // And a spatially flipped kernel: convolution vs cross-correlation.
    let flipped = candle_conv(&xs, &k.flip(&[2, 3])?.contiguous()?, ci, co)?;

    for (name, t) in [("candle OIHW", &same), ("candle transposed", &swapped), ("candle kernel-flipped", &flipped)] {
        println!("mps vs {name:<22} max_abs {:.3e}", sd_tensor::testing::max_abs_diff(&mps_out, t)?);
    }
    Ok(())
}
