//! GLIGEN's gated self-attention on MLX — "put this thing *here*".
//!
//! ```text
//!   x = x + tanh(alpha_attn)  * attn(norm1([x ; objs]))[:, :n_visual]
//!   x = x + tanh(alpha_dense) * ff(norm2(x))
//! ```
//!
//! **`tanh(0) = 0`, so zeroed gates contribute exactly nothing.** That is how a
//! grounded checkpoint reproduces an ungrounded image when no boxes are given,
//! and it is asserted rather than assumed.
//!
//! The fuser sits **between** the two attentions: grounding conditions the
//! image tokens before they meet the text, not after. Putting it after runs
//! and produces a plausible image that follows the boxes less.
//!
//! The gates are learned scalars that do not change during inference, so the
//! `tanh` is taken once here rather than per call.

use sd_tensor::mlx::{concat, Array, Stream};
use sd_tensor::{Error, Result};

use super::{get, linear, Weights};

/// Frequencies per coordinate. 8 in every published GLIGEN.
pub const FOURIER_FREQS: usize = 8;
/// `freqs * 2 (sin, cos) * 4 (xyxy)`.
pub const POSITION_DIM: usize = FOURIER_FREQS * 2 * 4;

/// Is this checkpoint grounded? Asked by name — the weights live beneath the
/// block itself, so there is no ordering to get wrong.
pub fn present(w: &Weights, prefix: &str) -> bool {
    w.contains_key(&format!("{prefix}.fuser.alpha_attn"))
}

/// One transformer block's gated self-attention over the grounding tokens.
pub fn fuse(
    x: &Array,
    objs: &Array,
    heads: usize,
    w: &Weights,
    prefix: &str,
    s: &Stream,
) -> Result<Array> {
    let p = |n: &str| format!("{prefix}.fuser.{n}");
    let [b, n_visual, dim] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: gligen fuser got {:?}", x.shape())));
    };

    let gate_attn = get(w, &p("alpha_attn"))?.to_vec_f32(s)?[0].tanh();
    let gate_dense = get(w, &p("alpha_dense"))?.to_vec_f32(s)?[0].tanh();

    let projected = linear(
        objs,
        get(w, &p("linear.weight"))?,
        w.get(&p("linear.bias")),
        s,
    )?;
    let joined = concat(&[x, &projected], 1, s)?;

    // Self-attention over image tokens and grounding tokens together, then only
    // the image half is kept.
    let normed = joined.layer_norm(
        Some(get(w, &p("norm1.weight"))?),
        Some(get(w, &p("norm1.bias"))?),
        1e-5,
        s,
    )?;
    let seq = normed.shape()[1];
    let head_dim = dim / heads;
    let proj = |name: &str| -> Result<Array> {
        linear(
            &normed,
            get(w, &p(&format!("attn.{name}.weight")))?,
            w.get(&p(&format!("attn.{name}.bias"))),
            s,
        )?
        .reshape(&[b, seq, heads, head_dim], s)?
        .transpose(&[0, 2, 1, 3], s)
    };
    let attended = proj("to_q")?.sdpa(
        &proj("to_k")?,
        &proj("to_v")?,
        1.0 / (head_dim as f32).sqrt(),
        s,
    )?;
    let attended = attended
        .transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[b, seq, dim], s)?;
    let attended = linear(
        &attended,
        get(w, &p("attn.to_out.0.weight"))?,
        w.get(&p("attn.to_out.0.bias")),
        s,
    )?
    .narrow(1, 0, n_visual, s)?;

    let x = x.add(&attended.mul(&Array::scalar_f32(gate_attn)?, s)?, s)?;

    let normed = x.layer_norm(
        Some(get(w, &p("norm2.weight"))?),
        Some(get(w, &p("norm2.bias"))?),
        1e-5,
        s,
    )?;
    // GEGLU, the same shape the block's own feed-forward uses.
    let projected = linear(
        &normed,
        get(w, &p("ff.net.0.proj.weight"))?,
        w.get(&p("ff.net.0.proj.bias")),
        s,
    )?;
    let dims = projected.shape();
    let last = dims.len() - 1;
    let inner = dims[last] / 2;
    let value = projected.narrow(last, 0, inner, s)?;
    let gate = projected.narrow(last, inner, inner, s)?;
    let dense = linear(
        &value.mul(&gate.gelu(s)?, s)?,
        get(w, &p("ff.net.2.weight"))?,
        w.get(&p("ff.net.2.bias")),
        s,
    )?;

    x.add(&dense.mul(&Array::scalar_f32(gate_dense)?, s)?, s)
}

/// `(box, phrase)` pairs into grounding tokens.
///
/// **The axis order is the whole subtlety.** A box is four numbers in `[0, 1]`,
/// each expanded into sinusoids at eight frequencies. The resulting axes are
/// `(coordinate, frequency, sin/cos)` and are flattened as
/// `(frequency, sin/cos, coordinate)`. Any ordering produces 64 numbers and
/// loads against the same weights; only this one lines up with what the MLP was
/// trained on.
pub fn position_net(
    boxes: &Array,
    masks: &Array,
    phrases: &Array,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let [b, n, four] = boxes.shape()[..] else {
        return Err(Error::Msg(format!("mlx: gligen boxes {:?}", boxes.shape())));
    };
    if four != 4 {
        return Err(Error::Msg("mlx: a box is four numbers".into()));
    }

    let raw = boxes.to_vec_f32(s)?;
    let mut embedded = vec![0f32; b * n * POSITION_DIM];
    for bi in 0..b {
        for ni in 0..n {
            let mut at = 0usize;
            for f in 0..FOURIER_FREQS {
                // 100^(i/dim): a geometric ladder, like a timestep
                // embedding's. Not 2^i, which is the obvious guess and is a
                // different embedding entirely.
                let scale = 100f32.powf(f as f32 / FOURIER_FREQS as f32);
                for part in 0..2 {
                    for c in 0..4 {
                        let v = raw[(bi * n + ni) * 4 + c] * scale;
                        embedded[(bi * n + ni) * POSITION_DIM + at] =
                            if part == 0 { v.sin() } else { v.cos() };
                        at += 1;
                    }
                }
            }
        }
    }
    let position = Array::from_slice_f32(&embedded, &[b, n, POSITION_DIM])?;

    // A masked-out box is replaced by the learned null, not zeroed: the MLP was
    // trained with a real vector there.
    let mask = masks.reshape(&[b, n, 1], s)?;
    let inverse = Array::scalar_f32(1.0)?.sub(&mask, s)?;
    let null_pos =
        get(w, "position_net.null_position_feature")?.reshape(&[1, 1, POSITION_DIM], s)?;
    let position = position
        .mul(&mask, s)?
        .add(&null_pos.mul(&inverse, s)?, s)?;

    let dim = phrases.shape()[2];
    let null_txt = get(w, "position_net.null_positive_feature")?.reshape(&[1, 1, dim], s)?;
    let text = phrases.mul(&mask, s)?.add(&null_txt.mul(&inverse, s)?, s)?;

    // Text first, then position: `linears.0` is 832 wide, which is 768 + 64.
    let joined = concat(&[&text, &position], 2, s)?;
    let h = linear(
        &joined,
        get(w, "position_net.linears.0.weight")?,
        w.get("position_net.linears.0.bias"),
        s,
    )?
    .silu(s)?;
    let h = linear(
        &h,
        get(w, "position_net.linears.2.weight")?,
        w.get("position_net.linears.2.bias"),
        s,
    )?
    .silu(s)?;
    linear(
        &h,
        get(w, "position_net.linears.4.weight")?,
        w.get("position_net.linears.4.bias"),
        s,
    )
}
