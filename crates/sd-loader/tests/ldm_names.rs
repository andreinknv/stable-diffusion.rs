//! LDM -> diffusers name translation for the VAE.
//!
//! The unit cases below are hand-checked against the real file. The test that
//! actually protects this is `every_vae_tensor_maps`, which requires the
//! translation to be *total* over a real checkpoint — a mapping that quietly
//! skips keys it does not recognise would pass every example and fail to load
//! a model.

use sd_loader::ldm::{vae_key, Mapped};

const BLOCKS: usize = 4;

fn map(k: &str) -> Option<Mapped> {
    vae_key(k, BLOCKS)
}

#[test]
fn the_decoder_block_order_is_reversed() {
    // LDM inserts at the front of its `up` list, so up.0 is the block
    // processed *last*. Its shape says so: decoder.up.0.block.0.conv1.weight
    // is [128, 256, 3, 3], the 256 -> 128 block diffusers calls up_blocks.3.
    assert_eq!(
        map("first_stage_model.decoder.up.0.block.0.conv1.weight")
            .unwrap()
            .name,
        "decoder.up_blocks.3.resnets.0.conv1.weight"
    );
    assert_eq!(
        map("first_stage_model.decoder.up.3.block.2.conv2.bias")
            .unwrap()
            .name,
        "decoder.up_blocks.0.resnets.2.conv2.bias"
    );
    // And the upsampler travels with its block.
    assert_eq!(
        map("first_stage_model.decoder.up.3.upsample.conv.weight")
            .unwrap()
            .name,
        "decoder.up_blocks.0.upsamplers.0.conv.weight"
    );
}

#[test]
fn the_encoder_block_order_is_not_reversed() {
    // Only one of the two towers flips: the encoder's `down` list is built in
    // forward order. Reversing both is the obvious symmetry and is wrong.
    assert_eq!(
        map("first_stage_model.encoder.down.0.block.0.conv1.weight")
            .unwrap()
            .name,
        "encoder.down_blocks.0.resnets.0.conv1.weight"
    );
    assert_eq!(
        map("first_stage_model.encoder.down.0.downsample.conv.bias")
            .unwrap()
            .name,
        "encoder.down_blocks.0.downsamplers.0.conv.bias"
    );
}

#[test]
fn attention_is_renamed_and_marked_for_reshape() {
    // Stored as 1x1 convolutions — [512, 512, 1, 1] — against a Linear that
    // wants [512, 512]. The rename alone would load a 4-D tensor into a 2-D
    // parameter and fail; the squeeze flag is what makes it work.
    for (ldm, ours) in [
        ("q", "to_q"),
        ("k", "to_k"),
        ("v", "to_v"),
        ("proj_out", "to_out.0"),
    ] {
        let m = map(&format!(
            "first_stage_model.decoder.mid.attn_1.{ldm}.weight"
        ))
        .unwrap();
        assert_eq!(
            m.name,
            format!("decoder.mid_block.attentions.0.{ours}.weight")
        );
        assert!(m.squeeze_to_2d, "{ldm} must be reshaped");
    }
    // The group norm is 1-D and must *not* be squeezed.
    let norm = map("first_stage_model.decoder.mid.attn_1.norm.weight").unwrap();
    assert_eq!(
        norm.name,
        "decoder.mid_block.attentions.0.group_norm.weight"
    );
    assert!(!norm.squeeze_to_2d);
}

#[test]
fn the_mid_block_resnets_are_numbered_from_one_to_zero() {
    assert_eq!(
        map("first_stage_model.encoder.mid.block_1.norm1.weight")
            .unwrap()
            .name,
        "encoder.mid_block.resnets.0.norm1.weight"
    );
    assert_eq!(
        map("first_stage_model.decoder.mid.block_2.conv2.bias")
            .unwrap()
            .name,
        "decoder.mid_block.resnets.1.conv2.bias"
    );
}

#[test]
fn the_odd_leaf_names_are_translated() {
    assert_eq!(
        map("first_stage_model.decoder.up.0.block.0.nin_shortcut.weight")
            .unwrap()
            .name,
        "decoder.up_blocks.3.resnets.0.conv_shortcut.weight"
    );
    assert_eq!(
        map("first_stage_model.decoder.norm_out.bias").unwrap().name,
        "decoder.conv_norm_out.bias"
    );
    // These two are already 1x1 Conv2d on both sides; nothing to do.
    assert_eq!(
        map("first_stage_model.quant_conv.weight").unwrap().name,
        "quant_conv.weight"
    );
    assert_eq!(
        map("first_stage_model.post_quant_conv.bias").unwrap().name,
        "post_quant_conv.bias"
    );
}

#[test]
fn keys_from_other_towers_are_declined_not_mangled() {
    // A mapper that returned something plausible for a UNet key would make
    // "not mine" indistinguishable from "translated".
    assert!(map("model.diffusion_model.input_blocks.1.0.in_layers.0.weight").is_none());
    assert!(map("cond_stage_model.transformer.text_model.final_layer_norm.weight").is_none());
    assert!(map("alphas_cumprod").is_none());
}

#[test]
fn every_vae_tensor_in_a_real_checkpoint_maps() {
    // The test that matters. Unit cases prove the shapes I looked at; this
    // proves there is nothing in the file I did not look at.
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/gguf/sd15-q4_0.gguf");
    if !p.exists() {
        eprintln!("SKIP: no sd15-q4_0.gguf; see xtask/golden/README.md");
        return;
    }
    let info = sd_loader::GgufInfo::open(&p).expect("reading the checkpoint");

    let vae: Vec<&String> = info
        .tensors
        .keys()
        .filter(|k| k.starts_with("first_stage_model."))
        .collect();
    assert!(vae.len() > 200, "expected a full VAE, found {}", vae.len());

    let unmapped: Vec<&&String> = vae.iter().filter(|k| map(k).is_none()).collect();
    assert!(
        unmapped.is_empty(),
        "{} of {} VAE tensors did not translate, e.g. {:?}",
        unmapped.len(),
        vae.len(),
        &unmapped[..unmapped.len().min(5)]
    );

    // And the translation must be injective: two LDM keys collapsing onto one
    // diffusers key would silently drop a weight.
    let mut seen = std::collections::HashSet::new();
    for k in &vae {
        let m = map(k).unwrap();
        assert!(seen.insert(m.name.clone()), "two keys map to {}", m.name);
    }
    eprintln!("{} VAE tensors, all mapped, no collisions", vae.len());
}

#[test]
fn only_the_attention_weight_is_reshaped_not_its_bias() {
    // The weight is a [C, C, 1, 1] kernel; the bias beside it is already 1-D.
    // Marking the whole projection for reshape indexes past the end of a
    // one-dimensional shape.
    let w = map("first_stage_model.decoder.mid.attn_1.q.weight").unwrap();
    let b = map("first_stage_model.decoder.mid.attn_1.q.bias").unwrap();
    assert!(w.squeeze_to_2d, "the kernel needs reshaping");
    assert!(!b.squeeze_to_2d, "the bias does not");
    assert_eq!(b.name, "decoder.mid_block.attentions.0.to_q.bias");
}

// -- the UNet --------------------------------------------------------------

use sd_loader::ldm::unet_key;

const LAYERS: usize = 2;

fn umap(k: &str) -> Option<Mapped> {
    unet_key(k, LAYERS)
}

#[test]
fn resnet_leaves_are_named_by_sequential_position_not_by_role() {
    // LDM names these by index into an nn.Sequential, so nothing in the name
    // says which is the norm and which is the conv. in_layers is
    // [GroupNorm, SiLU, Conv2d]; out_layers inserts a Dropout, which is why
    // its conv is .3 and not .2.
    let base = "model.diffusion_model.input_blocks.1.0";
    for (ldm, ours) in [
        ("in_layers.0.weight", "norm1.weight"),
        ("in_layers.2.weight", "conv1.weight"),
        ("emb_layers.1.bias", "time_emb_proj.bias"),
        ("out_layers.0.weight", "norm2.weight"),
        ("out_layers.3.bias", "conv2.bias"),
    ] {
        assert_eq!(
            umap(&format!("{base}.{ldm}")).unwrap().name,
            format!("down_blocks.0.resnets.0.{ours}")
        );
    }
}

#[test]
fn the_flat_slot_list_divides_into_blocks_by_arithmetic() {
    // Slot 0 is conv_in; then two resnets and a downsampler per block, with
    // the last block having no downsampler. None of that is in the name.
    let n = |i: usize, sub: &str, leaf: &str| {
        umap(&format!(
            "model.diffusion_model.input_blocks.{i}.{sub}.{leaf}"
        ))
        .map(|m| m.name)
    };
    assert_eq!(n(0, "0", "weight").as_deref(), Some("conv_in.weight"));
    assert_eq!(
        n(2, "0", "in_layers.0.weight").as_deref(),
        Some("down_blocks.0.resnets.1.norm1.weight")
    );
    // The downsampler's conv is called `op`, and only here.
    assert_eq!(
        n(3, "0", "op.weight").as_deref(),
        Some("down_blocks.0.downsamplers.0.conv.weight")
    );
    assert_eq!(
        n(4, "0", "in_layers.0.weight").as_deref(),
        Some("down_blocks.1.resnets.0.norm1.weight")
    );
    // The deepest block: resnets only, no attention, no downsampler.
    assert_eq!(
        n(11, "0", "in_layers.0.weight").as_deref(),
        Some("down_blocks.3.resnets.1.norm1.weight")
    );
}

#[test]
fn an_up_slot_index_one_is_attention_or_upsampler_depending_on_the_block() {
    // The genuinely ambiguous case. up_blocks.0 has no attention, so its
    // `.1` is the upsampler; every other block's `.1` is the attention and
    // the upsampler moves to `.2`. Position alone cannot tell them apart —
    // the `conv.` leaf is what does.
    assert_eq!(
        umap("model.diffusion_model.output_blocks.2.1.conv.weight")
            .unwrap()
            .name,
        "up_blocks.0.upsamplers.0.conv.weight"
    );
    assert_eq!(
        umap("model.diffusion_model.output_blocks.3.1.norm.weight")
            .unwrap()
            .name,
        "up_blocks.1.attentions.0.norm.weight"
    );
    assert_eq!(
        umap("model.diffusion_model.output_blocks.5.2.conv.bias")
            .unwrap()
            .name,
        "up_blocks.1.upsamplers.0.conv.bias"
    );
}

#[test]
fn the_head_and_the_timestep_mlp_are_sequential_indices_too() {
    assert_eq!(
        umap("model.diffusion_model.time_embed.0.weight")
            .unwrap()
            .name,
        "time_embedding.linear_1.weight"
    );
    assert_eq!(
        umap("model.diffusion_model.time_embed.2.bias")
            .unwrap()
            .name,
        "time_embedding.linear_2.bias"
    );
    assert_eq!(
        umap("model.diffusion_model.out.0.weight").unwrap().name,
        "conv_norm_out.weight"
    );
    assert_eq!(
        umap("model.diffusion_model.out.2.bias").unwrap().name,
        "conv_out.bias"
    );
    assert_eq!(
        umap("model.diffusion_model.middle_block.1.norm.weight")
            .unwrap()
            .name,
        "mid_block.attentions.0.norm.weight"
    );
}

#[test]
fn every_unet_tensor_in_a_real_checkpoint_maps() {
    // As with the VAE: unit cases prove the keys I looked at, this proves
    // there is nothing in the file I did not. 686 tensors.
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/gguf/sd15-q4_0.gguf");
    if !p.exists() {
        eprintln!("SKIP: no sd15-q4_0.gguf");
        return;
    }
    let info = sd_loader::GgufInfo::open(&p).expect("reading");

    let unet: Vec<&String> = info
        .tensors
        .keys()
        .filter(|k| k.starts_with("model.diffusion_model."))
        .collect();
    assert!(
        unet.len() > 600,
        "expected a full UNet, found {}",
        unet.len()
    );

    let unmapped: Vec<&&String> = unet.iter().filter(|k| umap(k).is_none()).collect();
    assert!(
        unmapped.is_empty(),
        "{} of {} UNet tensors did not translate, e.g. {:?}",
        unmapped.len(),
        unet.len(),
        &unmapped[..unmapped.len().min(6)]
    );

    let mut seen = std::collections::HashSet::new();
    for k in &unet {
        let m = umap(k).unwrap();
        assert!(seen.insert(m.name.clone()), "two keys map to {}", m.name);
    }
    eprintln!("{} UNet tensors, all mapped, no collisions", unet.len());
}
