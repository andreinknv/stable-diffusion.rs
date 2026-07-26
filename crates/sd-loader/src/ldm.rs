//! Translating CompVis/LDM parameter names to `diffusers` ones.
//!
//! `stable-diffusion.cpp` writes GGUF checkpoints with the original LDM
//! names. The models here use the `diffusers` names, so the two do not meet
//! without translation — and the translation is not a substitution:
//!
//! * **block order is reversed** in the decoder. LDM builds its `up` list by
//!   inserting at the front, so `up.0` is the *last* block processed. The
//!   shapes say so plainly: `decoder.up.0.block.0.conv1.weight` is
//!   `[128, 256, 3, 3]`, which is the 256 -> 128 block diffusers calls
//!   `up_blocks.3`. Getting this backwards loads cleanly and decodes noise.
//! * **attention is stored as 1x1 convolutions.** `mid.attn_1.q.weight` is
//!   `[512, 512, 1, 1]` where our `Linear` wants `[512, 512]`, so those four
//!   tensors need a reshape as well as a rename.
//! * the encoder's `down` list is *not* reversed — it is built in forward
//!   order — so only one of the two towers flips.

/// A translated key, and what has to happen to the tensor behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mapped {
    pub name: String,
    /// Drop trailing unit dimensions: `[C, C, 1, 1]` -> `[C, C]`.
    ///
    /// Only the VAE's attention projections need this. It is a property of
    /// the *source* layout, not of the tensor, so it is decided here rather
    /// than guessed from the shape at load time.
    pub squeeze_to_2d: bool,
}

impl Mapped {
    fn plain(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            squeeze_to_2d: false,
        }
    }
    fn squeezed(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            squeeze_to_2d: true,
        }
    }
}

/// Rewrite the leaf of a resnet key.
fn resnet_leaf(leaf: &str) -> String {
    // The only rename inside a resnet: LDM's "network-in-network" shortcut is
    // diffusers' `conv_shortcut`. norm1/conv1/norm2/conv2 already agree.
    leaf.replace("nin_shortcut", "conv_shortcut")
}

/// Rewrite a mid-block key, shared by encoder and decoder.
///
/// `prefix` is `encoder` or `decoder`; `rest` is what follows `mid.`.
fn mid_block(prefix: &str, rest: &str) -> Option<Mapped> {
    if let Some(leaf) = rest.strip_prefix("block_1.") {
        return Some(Mapped::plain(format!(
            "{prefix}.mid_block.resnets.0.{}",
            resnet_leaf(leaf)
        )));
    }
    if let Some(leaf) = rest.strip_prefix("block_2.") {
        return Some(Mapped::plain(format!(
            "{prefix}.mid_block.resnets.1.{}",
            resnet_leaf(leaf)
        )));
    }
    let attn = rest.strip_prefix("attn_1.")?;
    let (head, tail) = attn.split_once('.')?;
    let base = format!("{prefix}.mid_block.attentions.0");
    // Only the weight is a 1x1 convolution kernel. The bias beside it is
    // already 1-D and reshaping it would be an out-of-bounds index, so the
    // flag is a property of the individual tensor, not of the projection.
    let squeeze = |name: String| {
        if tail == "weight" {
            Mapped::squeezed(name)
        } else {
            Mapped::plain(name)
        }
    };
    // `norm` is a GroupNorm and keeps its 1-D shape; the four projections are
    // 1x1 convolutions standing in for linears.
    Some(match head {
        "norm" => Mapped::plain(format!("{base}.group_norm.{tail}")),
        "q" => squeeze(format!("{base}.to_q.{tail}")),
        "k" => squeeze(format!("{base}.to_k.{tail}")),
        "v" => squeeze(format!("{base}.to_v.{tail}")),
        "proj_out" => squeeze(format!("{base}.to_out.0.{tail}")),
        _ => return None,
    })
}

/// Translate one LDM VAE key.
///
/// `blocks` is the number of resolution levels — 4 for every SD VAE — and is
/// needed to reverse the decoder's block order.
///
/// Returns `None` for keys that are not part of the VAE, so a caller can tell
/// "not mine" from "translated".
pub fn vae_key(key: &str, blocks: usize) -> Option<Mapped> {
    let rest = key.strip_prefix("first_stage_model.")?;

    // The two quantisation convs are already 1x1 Conv2d on both sides.
    if rest.starts_with("quant_conv.") || rest.starts_with("post_quant_conv.") {
        return Some(Mapped::plain(rest));
    }

    let (tower, rest) = rest.split_once('.')?;
    if tower != "encoder" && tower != "decoder" {
        return None;
    }

    if rest.starts_with("conv_in.") || rest.starts_with("conv_out.") {
        return Some(Mapped::plain(format!("{tower}.{rest}")));
    }
    if let Some(leaf) = rest.strip_prefix("norm_out.") {
        return Some(Mapped::plain(format!("{tower}.conv_norm_out.{leaf}")));
    }
    if let Some(mid) = rest.strip_prefix("mid.") {
        return mid_block(tower, mid);
    }

    // down.{i}.* / up.{i}.*
    let (list, rest) = rest.split_once('.')?;
    let (idx, rest) = rest.split_once('.')?;
    let idx: usize = idx.parse().ok()?;

    match (tower, list) {
        ("encoder", "down") => {
            // Built in forward order: the index carries over unchanged.
            if let Some(leaf) = rest.strip_prefix("downsample.conv.") {
                return Some(Mapped::plain(format!(
                    "encoder.down_blocks.{idx}.downsamplers.0.conv.{leaf}"
                )));
            }
            let leaf = rest.strip_prefix("block.")?;
            let (j, leaf) = leaf.split_once('.')?;
            Some(Mapped::plain(format!(
                "encoder.down_blocks.{idx}.resnets.{j}.{}",
                resnet_leaf(leaf)
            )))
        }
        ("decoder", "up") => {
            // Built back to front, so reverse it.
            let out = blocks.checked_sub(1)?.checked_sub(idx)?;
            if let Some(leaf) = rest.strip_prefix("upsample.conv.") {
                return Some(Mapped::plain(format!(
                    "decoder.up_blocks.{out}.upsamplers.0.conv.{leaf}"
                )));
            }
            let leaf = rest.strip_prefix("block.")?;
            let (j, leaf) = leaf.split_once('.')?;
            Some(Mapped::plain(format!(
                "decoder.up_blocks.{out}.resnets.{j}.{}",
                resnet_leaf(leaf)
            )))
        }
        _ => None,
    }
}

/// Rewrite the leaf of a UNet resnet key.
///
/// LDM names these by their position in an `nn.Sequential`, which is why the
/// indices look arbitrary: `in_layers` is `[GroupNorm, SiLU, Conv2d]`, so the
/// norm is `.0` and the conv is `.2`; `out_layers` adds a Dropout, so its
/// conv is `.3`. Nothing in the name says which is which.
fn unet_resnet_leaf(leaf: &str) -> Option<String> {
    let (head, tail) = leaf.split_once('.')?;
    Some(match (head, tail) {
        ("in_layers", t) => match t.split_once('.')? {
            ("0", x) => format!("norm1.{x}"),
            ("2", x) => format!("conv1.{x}"),
            _ => return None,
        },
        ("emb_layers", t) => match t.split_once('.')? {
            ("1", x) => format!("time_emb_proj.{x}"),
            _ => return None,
        },
        ("out_layers", t) => match t.split_once('.')? {
            ("0", x) => format!("norm2.{x}"),
            ("3", x) => format!("conv2.{x}"),
            _ => return None,
        },
        ("skip_connection", t) => format!("conv_shortcut.{t}"),
        _ => return None,
    })
}

/// Where an `input_blocks.{i}` slot lands.
///
/// LDM flattens the down pass into one list: slot 0 is `conv_in`, then each
/// block contributes `layers_per_block` resnets followed by a downsampler,
/// except the last which has no downsampler. Nothing in the name says which
/// block a slot belongs to — it is arithmetic, and an off-by-one loads
/// cleanly and denoises toward nothing.
fn down_slot(i: usize, layers: usize) -> Option<(usize, DownRole)> {
    if i == 0 {
        return Some((0, DownRole::ConvIn));
    }
    let stride = layers + 1;
    let block = (i - 1) / stride;
    let within = (i - 1) % stride;
    if within < layers {
        Some((block, DownRole::Resnet(within)))
    } else {
        Some((block, DownRole::Downsampler))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DownRole {
    ConvIn,
    Resnet(usize),
    Downsampler,
}

/// Translate one LDM UNet key.
///
/// `layers` is `layers_per_block` — 2 for SD 1.5. Up blocks carry
/// `layers + 1` resnets, which is why the two lists divide differently.
pub fn unet_key(key: &str, layers: usize) -> Option<Mapped> {
    let rest = key.strip_prefix("model.diffusion_model.")?;

    // The timestep MLP, named by Sequential position again.
    if let Some(t) = rest.strip_prefix("time_embed.") {
        return match t.split_once('.')? {
            ("0", x) => Some(Mapped::plain(format!("time_embedding.linear_1.{x}"))),
            ("2", x) => Some(Mapped::plain(format!("time_embedding.linear_2.{x}"))),
            _ => None,
        };
    }
    // The output head: [GroupNorm, SiLU, Conv2d].
    if let Some(t) = rest.strip_prefix("out.") {
        return match t.split_once('.')? {
            ("0", x) => Some(Mapped::plain(format!("conv_norm_out.{x}"))),
            ("2", x) => Some(Mapped::plain(format!("conv_out.{x}"))),
            _ => None,
        };
    }
    if let Some(t) = rest.strip_prefix("middle_block.") {
        let (idx, leaf) = t.split_once('.')?;
        return match idx {
            "0" => Some(Mapped::plain(format!(
                "mid_block.resnets.0.{}",
                unet_resnet_leaf(leaf)?
            ))),
            // The transformer's own names already agree with diffusers.
            "1" => Some(Mapped::plain(format!("mid_block.attentions.0.{leaf}"))),
            "2" => Some(Mapped::plain(format!(
                "mid_block.resnets.1.{}",
                unet_resnet_leaf(leaf)?
            ))),
            _ => None,
        };
    }

    if let Some(t) = rest.strip_prefix("input_blocks.") {
        let (i, t) = t.split_once('.')?;
        let (sub, leaf) = t.split_once('.')?;
        let i: usize = i.parse().ok()?;
        let (block, role) = down_slot(i, layers)?;
        return match (role, sub) {
            (DownRole::ConvIn, "0") => Some(Mapped::plain(format!("conv_in.{leaf}"))),
            (DownRole::Resnet(j), "0") => Some(Mapped::plain(format!(
                "down_blocks.{block}.resnets.{j}.{}",
                unet_resnet_leaf(leaf)?
            ))),
            (DownRole::Resnet(j), "1") => Some(Mapped::plain(format!(
                "down_blocks.{block}.attentions.{j}.{leaf}"
            ))),
            // The downsampler's conv is called `op`, and only here.
            (DownRole::Downsampler, "0") => {
                let x = leaf.strip_prefix("op.")?;
                Some(Mapped::plain(format!(
                    "down_blocks.{block}.downsamplers.0.conv.{x}"
                )))
            }
            _ => None,
        };
    }

    if let Some(t) = rest.strip_prefix("output_blocks.") {
        let (i, t) = t.split_once('.')?;
        let (sub, leaf) = t.split_once('.')?;
        let i: usize = i.parse().ok()?;
        // Up blocks hold layers + 1 resnets, so they divide by that.
        let per = layers + 1;
        let block = i / per;
        let j = i % per;
        return match sub {
            "0" => Some(Mapped::plain(format!(
                "up_blocks.{block}.resnets.{j}.{}",
                unet_resnet_leaf(leaf)?
            ))),
            // Ambiguous by position: `.1` is the attention where the block
            // has one, and the upsampler where it does not. `conv.` on the
            // leaf is what distinguishes them.
            "1" => match leaf.strip_prefix("conv.") {
                Some(x) => Some(Mapped::plain(format!(
                    "up_blocks.{block}.upsamplers.0.conv.{x}"
                ))),
                None => Some(Mapped::plain(format!(
                    "up_blocks.{block}.attentions.{j}.{leaf}"
                ))),
            },
            "2" => {
                let x = leaf.strip_prefix("conv.")?;
                Some(Mapped::plain(format!(
                    "up_blocks.{block}.upsamplers.0.conv.{x}"
                )))
            }
            _ => None,
        };
    }
    None
}
