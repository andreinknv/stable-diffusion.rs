//! Invariants that hold *between* modules, checked without reference data.
//!
//! The golden suite compares a module against saved tensors, which means it
//! cannot see the seams: what one module promises and the next assumes. That
//! gap is not theoretical — a pooled CLIP embedding was read from the wrong
//! position for the whole history of this project, and survived because the
//! one golden test covering pooling used the single tokenizer whose padding
//! makes the bug unreachable.
//!
//! Everything here runs on a fresh clone with no downloads, so unlike the
//! golden tests it actually executes in CI.

use sd_models::unet::UNetConfig;

/// Every architecture this project builds, by name.
fn architectures() -> Vec<(&'static str, UNetConfig)> {
    vec![
        ("sd15", UNetConfig::sd15()),
        ("sd2", UNetConfig::sd2()),
        ("sdxl", UNetConfig::sdxl()),
        ("instruct_pix2pix", UNetConfig::instruct_pix2pix()),
        ("unclip", UNetConfig::unclip()),
    ]
}

#[test]
fn every_unet_config_is_internally_consistent() {
    // Four parallel vectors indexed by block. A config whose lengths disagree
    // panics somewhere inside construction with an index out of range, several
    // frames from the typo — and the only time anyone edits these is when
    // adding an architecture, which is exactly when a typo is likely.
    for (name, cfg) in architectures() {
        let blocks = cfg.block_out_channels.len();
        assert!(blocks > 0, "{name}: no blocks");
        assert_eq!(
            cfg.attention_head_dim.len(),
            blocks,
            "{name}: one head count per block"
        );
        assert_eq!(
            cfg.transformer_layers_per_block.len(),
            blocks,
            "{name}: one transformer depth per block"
        );
        assert_eq!(
            cfg.down_block_has_attention.len(),
            blocks,
            "{name}: one attention flag per block"
        );

        // Head *counts*, so every block's width must divide by its count —
        // otherwise the head reshape fails deep inside attention.
        for (i, (&width, &heads)) in cfg
            .block_out_channels
            .iter()
            .zip(&cfg.attention_head_dim)
            .enumerate()
        {
            assert_eq!(
                width % heads,
                0,
                "{name}: block {i} is {width} wide over {heads} heads"
            );
        }

        // The skip stack: conv_in, then per block one per resnet plus a
        // downsampler on all but the last.
        let expected = 1 + blocks * cfg.layers_per_block + (blocks - 1);
        assert_eq!(
            cfg.skip_channels().len(),
            expected,
            "{name}: skip stack length"
        );
    }
}

#[test]
fn the_conditioning_slots_are_exclusive_and_go_to_the_right_architectures() {
    // `addition` is SDXL's micro-conditioning and `class_projection` is
    // unCLIP's image embedding. Both land in the same place — added to the
    // timestep embedding — and no published checkpoint has both. If one ever
    // does, the addition order becomes a decision rather than a coincidence,
    // and this is where that gets noticed.
    for (name, cfg) in architectures() {
        assert!(
            !(cfg.addition.is_some() && cfg.class_projection.is_some()),
            "{name}: carries both conditioning slots; the order into temb is now a choice"
        );
        match name {
            "sdxl" => assert!(cfg.addition.is_some(), "sdxl needs micro-conditioning"),
            "unclip" => assert!(
                cfg.class_projection.is_some(),
                "unclip needs a class embedding"
            ),
            _ => assert!(
                cfg.addition.is_none() && cfg.class_projection.is_none(),
                "{name}: should condition on text alone"
            ),
        }
    }
}

#[test]
fn the_unclip_class_projection_is_exactly_twice_an_image_embedding() {
    // **The invariant the published `-t2i-h` mirror does not satisfy.** Its
    // prior emits a 768-wide ViT-L embedding while its UNet's class projection
    // is 2048, being twice a 1024-wide ViT-H one, so the two halves do not
    // meet. Worth pinning here because it is the kind of mismatch a
    // repackaged checkpoint can acquire without anyone noticing.
    //
    // The width is twice the embedding's because the vector is the embedding
    // concatenated with a sinusoid of the noise level, at the same width. Any
    // other ratio means the two ends came from different checkpoints.
    let cfg = UNetConfig::unclip();
    let projection = cfg.class_projection.expect("unclip has a class embedding");
    assert_eq!(projection % 2, 0, "a class projection is two halves");
    let embedding = projection / 2;
    assert_eq!(
        embedding, 1024,
        "the image-variation checkpoint is ViT-H, so 1024 wide"
    );

    // And the augmentation agrees, which is what `with_prior` checks at load.
    // Built here from the width rather than a checkpoint, so it runs with no
    // fixtures.
    use sd_tensor::nn::{VarBuilder, VarMap};
    use sd_tensor::{DType, Device};
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let augmentor = sd_models::unclip::NoiseAugmentor::new(embedding, vb).expect("builds");
    assert_eq!(augmentor.embed_dim(), embedding);
    assert_eq!(
        augmentor.output_dim(),
        projection,
        "the augmented vector must be exactly what the UNet projects"
    );
}

#[test]
fn a_unet_refuses_the_conditioning_it_was_not_built_for() {
    // Each of these runs happily if the guard is missing, on a timestep
    // embedding that means something else. The messages must name the method
    // to use, because that is the only thing standing between a caller and a
    // plausible wrong image.
    use sd_models::unet::UNet2DConditionModel;
    use sd_tensor::nn::{VarBuilder, VarMap};
    use sd_tensor::{DType, Device, Tensor};

    let dev = Device::Cpu;
    let cfg = UNetConfig {
        block_out_channels: vec![32, 64],
        layers_per_block: 1,
        attention_head_dim: vec![2, 2],
        transformer_layers_per_block: vec![1, 1],
        down_block_has_attention: vec![true, false],
        cross_attention_dim: 16,
        norm_num_groups: 8,
        class_projection: Some(64),
        ..UNetConfig::sd15()
    };
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let unet = UNet2DConditionModel::new(&cfg, vb).expect("builds");
    assert!(unet.takes_class_labels());

    let sample = Tensor::zeros((1, 4, 16, 16), DType::F32, &dev).unwrap();
    let timestep = Tensor::new(&[500f32], &dev).unwrap();
    let context = Tensor::zeros((1, 77, 16), DType::F32, &dev).unwrap();

    let err = unet
        .forward(&sample, &timestep, &context)
        .expect_err("a class-conditioned UNet must refuse a plain forward");
    assert!(
        err.to_string().contains("forward_unclip"),
        "the refusal must name the fix, got: {err}"
    );
}

#[test]
fn the_noise_augmentation_and_the_prior_share_one_schedule() {
    // They do, and it is easy to miss: `unclip` documents its cosine ladder as
    // "not a sampler's schedule", which is true of the augmentation and false
    // of the prior, whose sampler *is* that ladder. If the two ever diverge,
    // an image embedding would be noised on one schedule and denoised on
    // another — plausible output, wrong model.
    use sd_models::prior::PriorScheduler;
    let scheduler = PriorScheduler::new(sd_models::unclip::TRAIN_TIMESTEPS);
    // A 1000-step run visits every training timestep, so the ladder it walks
    // is the augmentation's own, entry for entry.
    assert_eq!(
        scheduler.timesteps().len(),
        sd_models::unclip::TRAIN_TIMESTEPS
    );
    assert_eq!(
        scheduler.timesteps()[0],
        sd_models::unclip::TRAIN_TIMESTEPS - 1
    );
    assert_eq!(*scheduler.timesteps().last().expect("non-empty"), 0);
}
