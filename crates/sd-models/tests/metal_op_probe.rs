use sd_tensor::{Device, Tensor};

fn probe(name: &str, cin: usize, cout: usize, n: usize, metal: &Device) {
    let cpu = Device::Cpu;
    let x = Tensor::randn(0f32, 1f32, (1, cin, n, n), &cpu).unwrap();
    let w = Tensor::randn(0f32, 0.05f32, (cout, cin, 3, 3), &cpu).unwrap();
    let a = x.conv2d(&w, 1, 1, 1, 1).unwrap();
    let b = x
        .to_device(metal)
        .unwrap()
        .conv2d(&w.to_device(metal).unwrap(), 1, 1, 1, 1)
        .unwrap()
        .to_device(&cpu)
        .unwrap();
    let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let d = av
        .iter()
        .zip(&bv)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    // im2col holds cin*9 columns per output position.
    let im2col = (cin * 9) as u64 * (n * n) as u64;
    eprintln!(
        "{name:<34} im2col={im2col:>13} elems ({:>6.2} GB)  max|diff|={d:>10.3e} {}",
        im2col as f64 * 4.0 / 1e9,
        if d < 1e-3 { "ok" } else { "MISMATCH" }
    );
}

#[test]
fn metal_conv2d_by_input_channels() {
    let Ok(metal) = Device::new_metal(0) else {
        eprintln!("SKIP: no Metal device");
        return;
    };
    // Few output channels keeps the CPU reference affordable; the input shape
    // is the decoder's real one, which is what sets the im2col size.
    probe("conv 256ch @512  (ok range)", 256, 4, 512, &metal);
    probe("conv 128ch @1024 (decoder)", 128, 4, 1024, &metal);
    probe("conv 256ch @1024 (decoder)", 256, 4, 1024, &metal);
}

#[test]
fn metal_upsample_at_decoder_scale() {
    let Ok(metal) = Device::new_metal(0) else {
        return;
    };
    let cpu = Device::Cpu;
    for (c, n) in [(256usize, 256usize), (256, 512)] {
        let x = Tensor::randn(0f32, 1f32, (1, c, n, n), &cpu).unwrap();
        let a = x.upsample_nearest2d(n * 2, n * 2).unwrap();
        let b = x
            .to_device(&metal)
            .unwrap()
            .upsample_nearest2d(n * 2, n * 2)
            .unwrap()
            .to_device(&cpu)
            .unwrap();
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let d = av
            .iter()
            .zip(&bv)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "upsample [1,{c},{n},{n}] -> {}  max|diff|={d:.3e} {}",
            n * 2,
            if d < 1e-3 { "ok" } else { "MISMATCH" }
        );
    }
}
