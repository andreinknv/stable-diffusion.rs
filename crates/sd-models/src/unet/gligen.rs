//! GLIGEN's gated self-attention, one per transformer block.
//!
//! The grounding tokens from [`crate::gligen::PositionNet`] are consumed here:
//! each block lets its image tokens attend over them, gated by a learned
//! scalar.
//!
//! ```text
//!   x = x + tanh(alpha_attn)  * attn(norm1([x ; objs]))[:, :n_visual]
//!   x = x + tanh(alpha_dense) * ff(norm2(x))
//! ```
//!
//! # The gates make a fuser an exact identity when zeroed
//!
//! `tanh(0) = 0`, so zeroed gates contribute *exactly* nothing. That is how
//! these are trained against a frozen base, and it gives a free correctness
//! check rather than one that has to be invented.
//!
//! # Only the image tokens come back
//!
//! Attention runs over `[x ; objs]` and the result is truncated to the image
//! tokens. Keeping the grounding tokens' outputs would grow the sequence at
//! every block; dropping them is the definition, not an optimisation — they
//! are there to be attended *to*.
//!
//! # No installation machinery, unlike the other adapters
//!
//! The weights live at `<block>.fuser`, and a transformer block already holds
//! a `VarBuilder` scoped to itself. So each block asks for its own by name and
//! there is no index list to get wrong — which is exactly the trap the
//! IP-Adapter has and this does not. A checkpoint without fusers simply has no
//! such tensor, and [`Fuser::present`] says so.
//!
//! What *is* ambient is the tokens, for the same reason as `super::motion`'s
//! frame count: runtime data that must reach 16 blocks unchanged.

use std::cell::RefCell;

use sd_tensor::nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear, VarBuilder};
use sd_tensor::{DType, Module, Result, Tensor};

use super::attention::{Attention, FeedForward};

thread_local! {
    /// Grounding tokens for the current forward, `[b, n, cross_dim]`.
    static OBJS: RefCell<Option<Tensor>> = const { RefCell::new(None) };
}

/// The grounding tokens in scope, if any.
pub fn objs() -> Option<Tensor> {
    OBJS.with(|o| o.borrow().clone())
}

/// Restores the previous tokens when dropped.
#[must_use = "grounding is removed when this guard is dropped"]
pub struct ObjsGuard(Option<Tensor>);

impl Drop for ObjsGuard {
    fn drop(&mut self) {
        OBJS.with(|o| *o.borrow_mut() = self.0.take());
    }
}

/// Ground the next forward on these tokens.
///
/// Dropping the guard turns grounding off, which is how **scheduled sampling**
/// is expressed: GLIGEN grounds for roughly the first 30 % of the schedule and
/// then finishes unguided, because holding the model to the boxes throughout
/// costs image quality for placement it has already achieved.
pub fn with_objs(tokens: Tensor) -> ObjsGuard {
    ObjsGuard(OBJS.with(|o| o.borrow_mut().replace(tokens)))
}

/// One block's gated self-attention over the grounding tokens.
#[derive(Debug)]
pub struct Fuser {
    linear: Linear,
    attn: Attention,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    /// `tanh(alpha_attn)`, resolved at load — a learned scalar that does not
    /// change during inference, so the `tanh` is taken once rather than per
    /// block per step.
    gate_attn: f64,
    gate_dense: f64,
}

impl Fuser {
    /// Whether `vb`'s block carries grounding weights.
    pub fn present(vb: &VarBuilder) -> bool {
        vb.contains_tensor("fuser.alpha_attn")
    }

    /// `vb` is the *block's* builder; the fuser is at `fuser` beneath it.
    pub fn new(dim: usize, cross_dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        let gate = |name: &str| -> Result<f64> {
            vb.get((), name)?.to_dtype(DType::F64)?.to_scalar::<f64>()
        };
        Ok(Self {
            linear: linear(cross_dim, dim, vb.pp("linear"))?,
            // Self-attention over the concatenation, so no cross dimension.
            attn: Attention::new(dim, None, heads, dim / heads, vb.pp("attn"))?,
            ff: FeedForward::new(dim, 4, vb.pp("ff"))?,
            norm1: layer_norm(dim, LayerNormConfig::default(), vb.pp("norm1"))?,
            norm2: layer_norm(dim, LayerNormConfig::default(), vb.pp("norm2"))?,
            gate_attn: gate("alpha_attn")?.tanh(),
            gate_dense: gate("alpha_dense")?.tanh(),
        })
    }

    /// `xs` is `[b, n_visual, dim]`; `objs` is `[b, n, cross_dim]`.
    pub fn forward(&self, xs: &Tensor, objs: &Tensor) -> Result<Tensor> {
        let n_visual = xs.dim(1)?;
        let projected = self.linear.forward(&objs.to_dtype(xs.dtype())?)?;
        let joined = Tensor::cat(&[xs, &projected], 1)?;

        let attended = self.attn.forward(&self.norm1.forward(&joined)?, None)?;
        let attended = attended.narrow(1, 0, n_visual)?;
        let xs = (xs + (attended * self.gate_attn)?)?;

        let dense = self.ff.forward(&self.norm2.forward(&xs)?)?;
        xs + (dense * self.gate_dense)?
    }
}
