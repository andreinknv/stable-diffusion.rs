//! A decoder-only LLM, used as a **text encoder**.
//!
//! The 2026 image models stopped using T5 and CLIP. Qwen-Image conditions on
//! Qwen2.5-VL, Z-Image on Qwen3, FLUX.2 on Mistral — and none of those is a
//! variation on T5, they are ordinary causal language models whose hidden
//! states are read instead of their logits. One implementation serves all
//! three, because at the level that matters they are the same transformer:
//!
//! | | Qwen3 | Qwen2.5-VL | Mistral Small 3.2 |
//! |---|---|---|---|
//! | norm | RMSNorm, pre | RMSNorm, pre | RMSNorm, pre |
//! | attention | GQA + QK-norm | GQA | GQA |
//! | `q`/`k`/`v` bias | no | **yes** | no |
//! | MLP | SwiGLU | SwiGLU | SwiGLU |
//! | position | RoPE | RoPE | RoPE |
//!
//! So [`LlmConfig`] is three booleans and some widths, not three ports.
//!
//! # What is deliberately absent
//!
//! **The sampling head.** `lm_head` is never run: a diffusion model conditions
//! on a hidden state, and computing 152,000 logits per token to discard them
//! would be most of the cost of the encoder.
//!
//! **The KV cache.** There is no generation loop here — one forward over the
//! whole prompt, once per image. A cache would be pure overhead.
//!
//! # Two things that fail quietly
//!
//! **The attention is causal.** These are decoders, and the checkpoints were
//! trained that way; running them bidirectionally produces a plausible
//! embedding of exactly the right shape from a model that never saw one.
//!
//! **QK-norm is per head, not per hidden.** Qwen3's `q_norm` and `k_norm` are
//! `[head_dim]` — 128 wide against a 2048-wide projection — so they apply
//! after the split into heads. Applying them before broadcasts a 128-wide
//! weight across a 2048-wide tensor, which MLX accepts when the sizes happen
//! to divide.

use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{Error, Result};

use super::{get, linear, Weights};

/// Geometry of a decoder used as a text encoder.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    /// Grouped-query attention: fewer key/value heads than query heads, each
    /// shared across `heads / kv_heads` queries. Equal to `heads` means plain
    /// multi-head attention.
    pub kv_heads: usize,
    /// **Not always `hidden / heads`.** Qwen3-0.6B is 1024 wide with 16 heads
    /// and a head dimension of 128, so the q projection is *wider* than the
    /// model. Deriving it by division gives 64 and a projection that does not
    /// match its weight.
    pub head_dim: usize,
    pub intermediate: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// Qwen2.5 puts a bias on q, k and v. Qwen3 and Mistral do not.
    pub qkv_bias: bool,
    /// Qwen3 normalises q and k per head before the rotation. Qwen2.5 and
    /// Mistral do not.
    pub qk_norm: bool,
}

impl LlmConfig {
    /// Qwen3-0.6B, the fixture this module is verified against.
    pub fn qwen3_0_6b() -> Self {
        Self {
            hidden: 1024,
            layers: 28,
            heads: 16,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 3072,
            rms_eps: 1e-6,
            rope_theta: 1.0e6,
            qkv_bias: false,
            qk_norm: true,
        }
    }

    /// Qwen3-4B — Z-Image's text encoder, whose `cap_feat_dim` is 2560.
    pub fn qwen3_4b() -> Self {
        Self {
            hidden: 2560,
            layers: 36,
            heads: 32,
            kv_heads: 8,
            intermediate: 9728,
            ..Self::qwen3_0_6b()
        }
    }

    /// Qwen2.5-VL's *text* half — Qwen-Image's encoder, `joint_attention_dim`
    /// 3584.
    ///
    /// The vision tower is not here. Qwen-Image conditions on text, and a
    /// checkpoint's vision weights are simply unused.
    pub fn qwen2_5_vl_7b() -> Self {
        Self {
            hidden: 3584,
            layers: 28,
            heads: 28,
            kv_heads: 4,
            head_dim: 128,
            intermediate: 18944,
            rms_eps: 1e-6,
            rope_theta: 1.0e6,
            qkv_bias: true,
            qk_norm: false,
        }
    }

    /// Mistral Small 3.2 — FLUX.2's text encoder.
    pub fn mistral_small_3_2() -> Self {
        Self {
            hidden: 5120,
            layers: 40,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 32768,
            rms_eps: 1e-5,
            rope_theta: 1.0e6,
            qkv_bias: false,
            qk_norm: false,
        }
    }

    /// How many query heads share each key/value head.
    pub fn group_size(&self) -> usize {
        self.heads / self.kv_heads.max(1)
    }
}

/// `cos` and `sin` for `seq` positions, each `[1, 1, seq, head_dim/2]`.
///
/// Built on the host in f64: the exponent spans several orders of magnitude
/// and f32 loses the low frequencies, which are the ones encoding long-range
/// position. The same reasoning as `flux::embed_nd`, and the same fix.
fn rope_tables(seq: usize, head_dim: usize, theta: f32, s: &Stream) -> Result<(Array, Array)> {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for (t, _) in (0..seq).enumerate() {
        for i in 0..half {
            let omega = 1.0 / (theta as f64).powf(2.0 * i as f64 / head_dim as f64);
            let angle = t as f64 * omega;
            cos[t * half + i] = angle.cos() as f32;
            sin[t * half + i] = angle.sin() as f32;
        }
    }
    Ok((
        Array::from_slice_f32(&cos, &[1, 1, seq, half])?.contiguous(s)?,
        Array::from_slice_f32(&sin, &[1, 1, seq, half])?.contiguous(s)?,
    ))
}

/// Apply the rotation to `[b, h, seq, head_dim]`.
///
/// **Split-half, not interleaved.** The first half of the head dimension is
/// paired with the second — `x1 = x[..half]`, `x2 = x[half..]` — which is what
/// `transformers` does for Llama, Qwen and Mistral. Flux interleaves adjacent
/// pairs instead, and this project implements both because they are genuinely
/// different conventions: either one applied to the other's model is still a
/// rotation, and still produces a coherent-looking result with the geometry
/// wrong.
fn apply_rope(x: &Array, cos: &Array, sin: &Array, s: &Stream) -> Result<Array> {
    let [b, h, n, d] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: llm rope got {:?}", x.shape())));
    };
    let half = d / 2;
    let x1 = x.narrow(3, 0, half, s)?;
    let x2 = x.narrow(3, half, half, s)?;

    // [-x2, x1] rotated: out1 = x1*cos - x2*sin, out2 = x2*cos + x1*sin.
    let out1 = x1.mul(cos, s)?.sub(&x2.mul(sin, s)?, s)?;
    let out2 = x2.mul(cos, s)?.add(&x1.mul(sin, s)?, s)?;
    sd_tensor::mlx::concat(&[&out1, &out2], 3, s)?.reshape(&[b, h, n, d], s)
}

/// Repeat each key/value head to cover its query group.
///
/// **A no-op when `kv_heads == heads`**, which is what makes one code path
/// serve grouped and ungrouped attention.
fn expand_kv(x: &Array, group: usize, s: &Stream) -> Result<Array> {
    if group <= 1 {
        return x.contiguous(s);
    }
    let [b, kv, n, d] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: llm kv got {:?}", x.shape())));
    };
    // Insert a group axis next to the head axis and broadcast along it, so
    // head `i` of the output comes from kv head `i / group`. Repeating the
    // whole tensor instead would interleave the groups and pair every query
    // with the wrong key.
    x.reshape(&[b, kv, 1, n, d], s)?
        .broadcast_to(&[b, kv, group, n, d], s)?
        .contiguous(s)?
        .reshape(&[b, kv * group, n, d], s)
}

/// One decoder layer: pre-norm attention, pre-norm SwiGLU, both residual.
fn layer(
    x: &Array,
    cfg: &LlmConfig,
    w: &Weights,
    prefix: &str,
    cos: &Array,
    sin: &Array,
    s: &Stream,
) -> Result<Array> {
    let p = |name: &str| format!("{prefix}.{name}");
    let [b, n, _] = x.shape()[..] else {
        return Err(Error::Msg(format!("mlx: llm layer got {:?}", x.shape())));
    };

    let h = x.rms_norm(Some(get(w, &p("input_layernorm.weight"))?), cfg.rms_eps, s)?;

    let project = |name: &str| -> Result<Array> {
        linear(
            &h,
            get(w, &p(&format!("self_attn.{name}.weight")))?,
            cfg.qkv_bias
                .then(|| w.get(&p(&format!("self_attn.{name}.bias"))))
                .flatten(),
            s,
        )
    };
    let split = |t: &Array, heads: usize| -> Result<Array> {
        t.reshape(&[b, n, heads, cfg.head_dim], s)?
            .transpose(&[0, 2, 1, 3], s)?
            .contiguous(s)
    };

    let mut q = split(&project("q_proj")?, cfg.heads)?;
    let mut k = split(&project("k_proj")?, cfg.kv_heads)?;
    let v = split(&project("v_proj")?, cfg.kv_heads)?;

    // **After the head split, before the rotation.** The norm weights are
    // `[head_dim]`, so applying them to the flat projection would broadcast a
    // 128-wide vector across 2048 — accepted, and wrong.
    if cfg.qk_norm {
        q = q.rms_norm(Some(get(w, &p("self_attn.q_norm.weight"))?), cfg.rms_eps, s)?;
        k = k.rms_norm(Some(get(w, &p("self_attn.k_norm.weight"))?), cfg.rms_eps, s)?;
    }

    let q = apply_rope(&q, cos, sin, s)?;
    let k = expand_kv(&apply_rope(&k, cos, sin, s)?, cfg.group_size(), s)?;
    let v = expand_kv(&v, cfg.group_size(), s)?;

    // **Causal.** These are decoders; a bidirectional pass produces the right
    // shape from a model that never saw one.
    let attended = q.sdpa_causal(&k, &v, 1.0 / (cfg.head_dim as f32).sqrt(), s)?;
    let merged = attended
        .transpose(&[0, 2, 1, 3], s)?
        .contiguous(s)?
        .reshape(&[b, n, cfg.heads * cfg.head_dim], s)?;
    let x = x.add(
        &linear(
            &merged,
            get(w, &p("self_attn.o_proj.weight"))?,
            w.get(&p("self_attn.o_proj.bias")),
            s,
        )?,
        s,
    )?;

    // SwiGLU: `down(silu(gate(x)) * up(x))`. The gate is `gate_proj`, not
    // `up_proj` — they are the same shape, and swapping them runs.
    let h = x.rms_norm(
        Some(get(w, &p("post_attention_layernorm.weight"))?),
        cfg.rms_eps,
        s,
    )?;
    let gate = linear(&h, get(w, &p("mlp.gate_proj.weight"))?, None, s)?.silu(s)?;
    let up = linear(&h, get(w, &p("mlp.up_proj.weight"))?, None, s)?;
    let ff = linear(
        &gate.mul(&up, s)?,
        get(w, &p("mlp.down_proj.weight"))?,
        None,
        s,
    )?;
    x.add(&ff, s)
}

/// Hidden states for `token_ids`, `[1, seq]` of i32.
///
/// Returns `layers + 1` tensors: the embedding, then each layer's output, with
/// the **final norm applied to the last** — matching `output_hidden_states` in
/// `transformers`, where `hidden_states[-1]` is normed and the rest are not.
///
/// Per-layer states rather than only the output, because a divergence in a
/// 28-layer stack localises to a layer instead of being reported at the end.
pub fn hidden_states(
    token_ids: &Array,
    cfg: &LlmConfig,
    w: &Weights,
    s: &Stream,
) -> Result<Vec<Array>> {
    let shape = token_ids.shape();
    let [_, seq] = shape[..] else {
        return Err(Error::Msg(format!(
            "mlx: llm token ids should be [n, seq], got {shape:?}"
        )));
    };

    let mut h = get(w, "model.embed_tokens.weight")?.take(token_ids, 0, s)?;
    let (cos, sin) = rope_tables(seq, cfg.head_dim, cfg.rope_theta, s)?;

    let mut out = Vec::with_capacity(cfg.layers + 1);
    out.push(h.contiguous(s)?);
    for i in 0..cfg.layers {
        h = layer(&h, cfg, w, &format!("model.layers.{i}"), &cos, &sin, s)?;
        out.push(h.contiguous(s)?);
    }
    // `transformers` norms only the last entry.
    let last = out.len() - 1;
    out[last] = h.rms_norm(Some(get(w, "model.norm.weight")?), cfg.rms_eps, s)?;
    Ok(out)
}

/// The last hidden state, post-norm — what a diffusion model conditions on.
pub fn encode(token_ids: &Array, cfg: &LlmConfig, w: &Weights, s: &Stream) -> Result<Array> {
    let states = hidden_states(token_ids, cfg, w, s)?;
    states
        .into_iter()
        .next_back()
        .ok_or_else(|| Error::Msg("mlx: llm produced no hidden states".into()))
}

/// The hidden state `depth` layers from the end.
///
/// Several of these models condition on an intermediate layer rather than the
/// last — the same choice SDXL makes with CLIP's penultimate — so which one is
/// a property of the *image* model, not of the encoder. `depth` of 1 is
/// [`encode`].
///
/// **Only the last state is normed**, so this returns a raw hidden state for
/// any depth above 1, matching `transformers`.
pub fn encode_at_depth(
    token_ids: &Array,
    cfg: &LlmConfig,
    depth: usize,
    w: &Weights,
    s: &Stream,
) -> Result<Array> {
    let states = hidden_states(token_ids, cfg, w, s)?;
    let index = states.len().checked_sub(depth.max(1)).ok_or_else(|| {
        Error::Msg(format!(
            "mlx: depth {depth} exceeds {} states",
            states.len()
        ))
    })?;
    states
        .into_iter()
        .nth(index)
        .ok_or_else(|| Error::Msg("mlx: llm hidden state index out of range".into()))
}
