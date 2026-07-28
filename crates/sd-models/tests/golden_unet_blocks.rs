//! Golden verification for the UNet's timestep embedding and resnet block.
//!
//! The three parts are checked separately and in order — sinusoid, MLP, resnet
//! — because each feeds the next. A failure in the sinusoid would otherwise
//! surface as a resnet mismatch and send you looking in the wrong file.

use std::path::PathBuf;

use sd_models::unet::{timestep_embedding, ResnetBlock2D, TimestepEmbedding};
use sd_tensor::nn::{VarBuilder, VarMap};
use sd_tensor::{testing, DType, Device, IndexOp, Tensor};

/// SD 1.5's UNet norms with 1e-5. The VAE uses 1e-6; copying that value gives
/// a small uniform offset that reads as noise.
const UNET_EPS: f64 = 1e-5;
const UNET_GROUPS: usize = 32;
const TEMB_CHANNELS: usize = 1280;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/unet_blocks")
}

/// Loads a reference file, or `None` when it was never generated.
fn refs(name: &str) -> Option<std::collections::HashMap<String, Tensor>> {
    let path = golden_dir().join(name);
    if !path.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no {name}.\n\
             Generate it with:\n\
             \n    python3 xtask/golden/dump_reference.py unet_blocks --output tests/golden\n"
        );
        return None;
    }
    Some(sd_tensor::safetensors::load(&path, &Device::Cpu).expect("loading reference"))
}

// -- structural: no reference data needed ---------------------------------

#[test]
fn timestep_embedding_has_shape_batch_by_dim() {
    let dev = Device::Cpu;
    let t = Tensor::new(&[0f32, 1.0, 500.0, 999.0], &dev).unwrap();
    for dim in [320usize, 128] {
        let emb = timestep_embedding(&t, dim).expect("embedding");
        assert_eq!(emb.dims(), &[4, dim]);
    }
}

#[test]
fn timestep_embedding_first_half_is_cosine() {
    // At t = 0 every frequency argument is 0, so cos gives 1 and sin gives 0.
    // With the halves swapped this fails instantly and needs no download —
    // which is the point, because `[sin, cos]` is the natural order to write
    // and produces a wrong model that otherwise looks fine.
    let dev = Device::Cpu;
    let t = Tensor::new(&[0f32], &dev).unwrap();
    let emb = timestep_embedding(&t, 320).expect("embedding");
    let row = emb.i(0).unwrap().to_vec1::<f32>().unwrap();

    assert!(
        row[..160].iter().all(|v| (v - 1.0).abs() < 1e-6),
        "first half must be cos(0) = 1, got {:?}",
        &row[..4]
    );
    assert!(
        row[160..].iter().all(|v| v.abs() < 1e-6),
        "second half must be sin(0) = 0, got {:?}",
        &row[160..164]
    );
}

fn tiny_resnet(in_c: usize, out_c: usize, dev: &Device) -> (VarMap, ResnetBlock2D) {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let block = ResnetBlock2D::new(in_c, out_c, TEMB_CHANNELS, UNET_GROUPS, UNET_EPS, vb)
        .expect("resnet builds");
    (varmap, block)
}

#[test]
fn resnet_preserves_spatial_dims() {
    let dev = Device::Cpu;
    let (_map, block) = tiny_resnet(320, 320, &dev);
    let xs = Tensor::zeros((2, 320, 16, 16), DType::F32, &dev).unwrap();
    let temb = Tensor::zeros((2, TEMB_CHANNELS), DType::F32, &dev).unwrap();

    let out = block.forward(&xs, &temb).expect("forward");
    assert_eq!(out.dims(), &[2, 320, 16, 16]);
}

#[test]
fn resnet_changes_channel_count_when_asked() {
    // in != out also means a conv_shortcut must exist, so this covers the
    // branch that is absent in the equal-channel case.
    let dev = Device::Cpu;
    let (_map, block) = tiny_resnet(320, 640, &dev);
    let xs = Tensor::zeros((2, 320, 16, 16), DType::F32, &dev).unwrap();
    let temb = Tensor::zeros((2, TEMB_CHANNELS), DType::F32, &dev).unwrap();

    let out = block.forward(&xs, &temb).expect("forward");
    assert_eq!(out.dims(), &[2, 640, 16, 16]);
}

// -- numerical: skip when the reference is absent --------------------------

#[test]
fn timestep_embedding_matches_diffusers() {
    let Some(refs) = refs("reference.safetensors") else {
        return;
    };
    let timesteps = refs.get("timesteps").expect("timesteps");
    let want = refs.get("sin_emb").expect("sin_emb");

    let got = timestep_embedding(timesteps, 320).expect("embedding");
    let c = testing::closeness(&got, want).expect("comparing");
    eprintln!("sin_emb: {c}");
    testing::assert_close(&got, want, testing::DEFAULT_ATOL, "timestep_embedding").unwrap();
}

#[test]
fn time_embedding_mlp_matches_diffusers() {
    let Some(refs) = refs("reference.safetensors") else {
        return;
    };
    let weights = golden_dir().join("time_embedding.safetensors");
    if !weights.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no time_embedding.safetensors");
        return;
    }

    let dev = Device::Cpu;
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev)
        .expect("loading time_embedding weights");
    let mlp = TimestepEmbedding::new(320, TEMB_CHANNELS, vb).expect("building MLP");

    let got = mlp
        .forward(refs.get("sin_emb").expect("sin_emb"))
        .expect("forward");
    let want = refs.get("temb").expect("temb");
    let c = testing::closeness(&got, want).expect("comparing");
    eprintln!("temb: {c}");
    testing::assert_close(&got, want, testing::DEFAULT_ATOL, "time_embedding").unwrap();
}

#[test]
fn resnet_block_matches_diffusers() {
    let Some(refs) = refs("reference.safetensors") else {
        return;
    };
    let weights = golden_dir().join("resnet.safetensors");
    if !weights.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no resnet.safetensors");
        return;
    }

    let dev = Device::Cpu;
    let vb = sd_loader::safetensors_var_builder(&[&weights], DType::F32, &dev)
        .expect("loading resnet weights");
    // down_blocks[0].resnets[0] is 320 -> 320, so it has no conv_shortcut.
    let block = ResnetBlock2D::new(320, 320, TEMB_CHANNELS, UNET_GROUPS, UNET_EPS, vb)
        .expect("building resnet");

    let got = block
        .forward(
            refs.get("resnet_input").expect("resnet_input"),
            refs.get("resnet_temb").expect("resnet_temb"),
        )
        .expect("forward");
    let want = refs.get("resnet_output").expect("resnet_output");

    let c = testing::closeness(&got, want).expect("comparing");
    eprintln!("resnet_output: {c}");
    testing::assert_close(&got, want, testing::DEFAULT_ATOL, "resnet block").unwrap();
}
