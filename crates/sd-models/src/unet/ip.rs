//! IP-Adapter's decoupled cross-attention: weights, wiring, and strength.
//!
//! An IP-Adapter conditions on an *image* alongside the text. Each of the
//! UNet's cross-attention layers gains a second key/value pair, and the layer
//! returns
//!
//! ```text
//!   attn(q, k_text, v_text)  +  scale * attn(q, k_image, v_image)
//! ```
//!
//! with `to_out` applied once, to the sum.
//!
//! **This is not the same as appending the image tokens to the text ones.**
//! Attention is not linear in K and V, so a concatenation is a different
//! function — and a plausible-looking one, which is why it is worth stating.
//!
//! # How the weights reach sixteen layers without sixteen parameters
//!
//! The modules are built four constructor levels below the UNet, and threading
//! an optional weight source through every one of them would touch every block
//! type for a property that only cross-attention cares about. Instead the
//! source is installed for the duration of `UNet2DConditionModel::new`, and
//! each cross-attention pulls the next slot as it is built.
//!
//! Construction-scoped, thread-local, and released by a guard — not runtime
//! state. It is only ever read while one `new` call is on the stack.
//!
//! # The index order is not the construction order
//!
//! The checkpoint numbers its entries by diffusers' flat processor list, which
//! visits **down blocks, then up blocks, then the mid block** — while this
//! UNet constructs down, mid, up. And the entries sit at *odd* indices,
//! because that list alternates self- and cross-attention and only
//! cross-attention has them.
//!
//! So slot `i` of the construction order maps to key `2 * order[i] + 1`. Get
//! this wrong and every correction lands on a differently-sized layer — which
//! usually fails to load, but between the two 1280-wide regions would not.
//! [`IpSource::sd15_order`] is the mapping, derived rather than typed out.

use std::cell::{Cell, RefCell};

use sd_tensor::{Result, VarBuilder};

thread_local! {
    /// Runtime strength. 1.0 is the published default.
    ///
    /// Thread-local rather than a process atomic, unlike `conv::seamless`:
    /// two generations on two threads may legitimately want different
    /// strengths, and a shared one also made two tests that set different
    /// values race with each other.
    static SCALE: Cell<f64> = const { Cell::new(1.0) };
}

/// The current IP-Adapter strength, on this thread.
pub fn scale() -> f64 {
    SCALE.with(Cell::get)
}

/// Restores the previous strength when dropped.
#[must_use = "the strength reverts when this guard is dropped"]
pub struct ScaleGuard(f64);

impl Drop for ScaleGuard {
    fn drop(&mut self) {
        SCALE.with(|s| s.set(self.0));
    }
}

/// Set the IP-Adapter strength until the guard drops.
///
/// Ambient for the same reason the weights are: it must reach sixteen layers
/// and is uniform across them. 0 makes the adapter contribute exactly nothing
/// — not approximately nothing — which is what makes an A/B meaningful.
pub fn with_scale(value: f64) -> ScaleGuard {
    ScaleGuard(SCALE.with(|s| s.replace(value)))
}

/// Weights for the decoupled path, consumed one cross-attention at a time.
pub struct IpSource<'a> {
    vb: VarBuilder<'a>,
    /// Checkpoint entry number per construction slot.
    order: Vec<usize>,
    next: usize,
    tokens: usize,
}

impl<'a> IpSource<'a> {
    /// `vb` should be rooted at `ip_adapter`.
    pub fn new(vb: VarBuilder<'a>, order: Vec<usize>, tokens: usize) -> Self {
        Self {
            vb,
            order,
            next: 0,
            tokens,
        }
    }

    /// Entry numbers for SD 1.5's sixteen cross-attention layers, in the order
    /// this UNet builds them: down blocks, then **mid**, then up blocks.
    ///
    /// The checkpoint's own order is down, up, mid — so mid, which is entry 15,
    /// is pulled forward to position 6 here.
    pub fn sd15_order() -> Vec<usize> {
        let down: Vec<usize> = (0..6).collect(); // three blocks, two each
        let up: Vec<usize> = (6..15).collect(); // three blocks, three each
        let mid = 15;
        let mut order = down;
        order.push(mid);
        order.extend(up);
        order
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

thread_local! {
    static INSTALLED: RefCell<Option<IpSource<'static>>> = const { RefCell::new(None) };
}

/// Installed weights, released on drop.
#[must_use = "the source is removed when this guard is dropped"]
pub struct SourceGuard;

impl Drop for SourceGuard {
    fn drop(&mut self) {
        INSTALLED.with(|s| *s.borrow_mut() = None);
    }
}

/// Install a source for the duration of a UNet construction.
///
/// # Safety
///
/// The lifetime is erased so the source can live in thread-local storage. The
/// guard removes it on drop, and callers must not let it outlive `vb` — which
/// is why this is `unsafe` and why the only caller is
/// `UNet2DConditionModel::new_with_ip`, where both live in one scope.
pub unsafe fn install(source: IpSource<'_>) -> SourceGuard {
    let erased: IpSource<'static> = unsafe { std::mem::transmute(source) };
    INSTALLED.with(|s| *s.borrow_mut() = Some(erased));
    SourceGuard
}

/// Take the next cross-attention's weights, if a source is installed.
///
/// Returns the `ip_adapter.N` builder and the image-token count.
pub(super) fn next_slot() -> Option<(VarBuilder<'static>, usize)> {
    INSTALLED.with(|s| {
        let mut slot = s.borrow_mut();
        let source = slot.as_mut()?;
        let entry = *source.order.get(source.next)?;
        source.next += 1;
        // Odd indices: the flat processor list alternates self- and
        // cross-attention, and only cross-attention carries these.
        Some((source.vb.pp((2 * entry + 1).to_string()), source.tokens))
    })
}

/// How many slots have been consumed, for the caller to check against the
/// sixteen it expected. A silent under-consumption would leave later layers
/// unconditioned and still render.
pub fn consumed() -> usize {
    INSTALLED.with(|s| s.borrow().as_ref().map_or(0, |src| src.next))
}

/// Whether every entry was used.
pub fn fully_consumed() -> Result<()> {
    INSTALLED.with(|s| match s.borrow().as_ref() {
        Some(src) if src.next != src.order.len() => Err(sd_tensor::Error::Msg(format!(
            "IP-Adapter: {} of {} entries were attached",
            src.next,
            src.order.len()
        ))),
        _ => Ok(()),
    })
}
