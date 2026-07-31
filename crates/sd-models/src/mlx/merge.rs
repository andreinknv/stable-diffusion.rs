//! Weighted averaging of checkpoints.
//!
//! `(1 - alpha) * a + alpha * b`, tensor by tensor. Nothing here knows what a
//! UNet is; the result is an ordinary checkpoint that loads through the usual
//! path.
//!
//! # What it refuses, and why
//!
//! A merge is only meaningful between checkpoints of the same architecture, and
//! the failure when they are not is quiet: two SD 1.5 variants merge cleanly,
//! while an SD 1.5 and an SDXL share enough tensor *names* to produce a file
//! that loads and renders noise. So:
//!
//! - **Shape mismatches are refused**, not skipped. Skipping would silently
//!   take one side's weights for that tensor, which is a third model neither
//!   caller asked for.
//! - **Tensors present in only one side are refused too**, unless
//!   [`MergeOptions::allow_unmatched`] says otherwise. A one-sided tensor means
//!   the two are not the same architecture, and the merged file would be
//!   incomplete in a way only discovered at load.
//!
//! Both are reported with a count and an example, because "they do not match"
//! without saying where is not actionable.

use sd_tensor::mlx::{Array, Stream};
use sd_tensor::{Error, Result};

use super::Weights;

/// How strictly to merge.
#[derive(Debug, Clone)]
pub struct MergeOptions {
    /// Weight of the *second* checkpoint. 0 is `a` exactly, 1 is `b` exactly.
    pub alpha: f64,
    /// Carry through tensors that only one side has, rather than refusing.
    ///
    /// Off by default: a one-sided tensor usually means the checkpoints are
    /// different architectures, and the merged file would be wrong in a way
    /// nothing notices until it is loaded.
    pub allow_unmatched: bool,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            allow_unmatched: false,
        }
    }
}

/// What the merge did, for the caller to report rather than assume.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Merged {
    /// Tensors averaged from both sides.
    pub blended: usize,
    /// Tensors carried through from one side, when allowed.
    pub carried: usize,
}

/// Average two weight maps.
///
/// `alpha` is **clamped** to `[0, 1]`: outside that range is extrapolation,
/// which is a different operation with different failure modes and is not what
/// a caller asking to merge means.
pub fn merge(
    a: &Weights,
    b: &Weights,
    options: &MergeOptions,
    s: &Stream,
) -> Result<(Weights, Merged)> {
    let alpha = options.alpha.clamp(0.0, 1.0);
    let mut out = Weights::with_capacity(a.len().max(b.len()));
    let mut report = Merged::default();

    let mut mismatched: Vec<&str> = Vec::new();
    let mut unmatched: Vec<&str> = Vec::new();

    let left_w = Array::scalar_f32((1.0 - alpha) as f32)?;
    let right_w = Array::scalar_f32(alpha as f32)?;

    for (name, left) in a {
        match b.get(name) {
            Some(right) if left.shape() == right.shape() => {
                let blended = left
                    .mul(&left_w, s)?
                    .add(&right.mul(&right_w, s)?, s)?
                    // Evaluated here rather than at the end: the whole point is
                    // to write a file, and holding every blend of a 3.4 GB
                    // checkpoint unevaluated means holding both inputs and the
                    // graph at once.
                    .contiguous(s)?;
                out.insert(name.clone(), blended);
                report.blended += 1;
            }
            Some(_) => mismatched.push(name),
            None => {
                unmatched.push(name);
                if options.allow_unmatched {
                    out.insert(name.clone(), left.contiguous(s)?);
                    report.carried += 1;
                }
            }
        }
    }
    for name in b.keys() {
        if !a.contains_key(name) {
            unmatched.push(name);
            if options.allow_unmatched {
                out.insert(name.clone(), b[name].contiguous(s)?);
                report.carried += 1;
            }
        }
    }

    if !mismatched.is_empty() {
        mismatched.sort_unstable();
        return Err(Error::Msg(format!(
            "merge: {} tensors have different shapes in the two checkpoints, first `{}` — \
             they are not the same architecture",
            mismatched.len(),
            mismatched[0]
        )));
    }
    if !unmatched.is_empty() && !options.allow_unmatched {
        unmatched.sort_unstable();
        return Err(Error::Msg(format!(
            "merge: {} tensors exist in only one checkpoint, first `{}` — pass \
             --allow-unmatched to carry them through if that is intended",
            unmatched.len(),
            unmatched[0]
        )));
    }

    Ok((out, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(name: &str, value: f32, shape: &[usize]) -> Weights {
        let n: usize = shape.iter().product();
        let t = Array::from_slice_f32(&vec![value; n], shape).unwrap();
        [(name.to_string(), t)].into_iter().collect()
    }

    fn value(w: &Weights, name: &str, s: &Stream) -> f32 {
        w[name].to_vec_f32(s).unwrap()[0]
    }

    /// The arithmetic, at both ends and the middle.
    #[test]
    fn alpha_selects_between_the_two() {
        let s = Stream::cpu();
        let (a, b) = (one("w", 0.0, &[2, 2]), one("w", 10.0, &[2, 2]));
        for (alpha, want) in [(0.0, 0.0), (0.5, 5.0), (1.0, 10.0), (0.25, 2.5)] {
            let (m, report) = merge(
                &a,
                &b,
                &MergeOptions {
                    alpha,
                    ..Default::default()
                },
                &s,
            )
            .unwrap();
            assert_eq!(report.blended, 1);
            assert!((value(&m, "w", &s) - want).abs() < 1e-6, "alpha {alpha}");
        }
    }

    /// **Out of range is clamped, not extrapolated.** Extrapolation is a
    /// different operation, and one a caller asking to *merge* is not asking
    /// for.
    #[test]
    fn alpha_is_clamped_rather_than_extrapolating() {
        let s = Stream::cpu();
        let (a, b) = (one("w", 0.0, &[2, 2]), one("w", 10.0, &[2, 2]));
        for (alpha, want) in [(2.0, 10.0), (-1.0, 0.0)] {
            let (m, _) = merge(
                &a,
                &b,
                &MergeOptions {
                    alpha,
                    ..Default::default()
                },
                &s,
            )
            .unwrap();
            assert!((value(&m, "w", &s) - want).abs() < 1e-6, "alpha {alpha}");
        }
    }

    /// **A shape mismatch is refused, not skipped.**
    ///
    /// Skipping would take one side's tensor, producing a third model neither
    /// caller asked for — and it would load and render.
    #[test]
    fn a_shape_mismatch_is_refused_and_named() {
        let s = Stream::cpu();
        let (a, b) = (one("w", 1.0, &[2, 2]), one("w", 1.0, &[4, 4]));
        let err = merge(&a, &b, &MergeOptions::default(), &s).expect_err("shapes differ");
        let text = format!("{err}");
        assert!(text.contains("`w`"), "the error must name a tensor: {text}");
        assert!(
            text.contains("architecture"),
            "and say what it means: {text}"
        );
    }

    /// **A one-sided tensor is refused by default**, and named.
    #[test]
    fn an_unmatched_tensor_is_refused_by_default() {
        let s = Stream::cpu();
        let a = one("only_in_a", 1.0, &[2, 2]);
        let b = one("only_in_b", 1.0, &[2, 2]);
        let err = merge(&a, &b, &MergeOptions::default(), &s).expect_err("neither side matches");
        assert!(format!("{err}").contains("only_in_"), "{err}");
    }

    /// ...and carried through when asked, from **both** sides.
    #[test]
    fn allow_unmatched_carries_from_both_sides() {
        let s = Stream::cpu();
        let a = one("only_in_a", 1.0, &[2, 2]);
        let b = one("only_in_b", 2.0, &[2, 2]);
        let (m, report) = merge(
            &a,
            &b,
            &MergeOptions {
                alpha: 0.5,
                allow_unmatched: true,
            },
            &s,
        )
        .unwrap();
        assert_eq!(report.blended, 0);
        assert_eq!(report.carried, 2, "one from each side");
        assert!((value(&m, "only_in_a", &s) - 1.0).abs() < 1e-6);
        assert!((value(&m, "only_in_b", &s) - 2.0).abs() < 1e-6);
    }

    /// A merge of a checkpoint with itself is that checkpoint, at any alpha.
    #[test]
    fn merging_a_checkpoint_with_itself_changes_nothing() {
        let s = Stream::cpu();
        let a = one("w", 3.25, &[2, 2]);
        for alpha in [0.0, 0.3, 0.5, 1.0] {
            let (m, _) = merge(
                &a,
                &a,
                &MergeOptions {
                    alpha,
                    ..Default::default()
                },
                &s,
            )
            .unwrap();
            assert!((value(&m, "w", &s) - 3.25).abs() < 1e-6, "alpha {alpha}");
        }
    }
}
