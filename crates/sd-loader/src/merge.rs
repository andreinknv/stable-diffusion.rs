//! Weighted averaging of checkpoints.
//!
//! `(1 - alpha) * a + alpha * b`, tensor by tensor. Loader-level work rather
//! than inference: nothing here knows what a UNet is, and the result is an
//! ordinary checkpoint that loads through the usual path.
//!
//! # What it refuses, and why
//!
//! A merge is only meaningful between checkpoints of the same architecture,
//! and the failure when they are not is quiet: two SD 1.5 variants merge
//! cleanly, an SD 1.5 and an SDXL share enough tensor *names* to produce a
//! file that loads and renders noise. So:
//!
//! * **Shape mismatches are refused**, not skipped. Skipping would silently
//!   take one side's weights for that tensor, which is a third model neither
//!   caller asked for.
//! * **Tensors present in only one side are refused too**, unless
//!   [`MergeOptions::allow_unmatched`] says otherwise. A missing tensor means
//!   the two are not the same architecture, and the merged file would be
//!   incomplete in a way only discovered at load.
//!
//! Both are reported with counts and an example, because "they do not match"
//! without saying where is not actionable.

use std::collections::HashMap;

use sd_tensor::{DType, Tensor};

use crate::LoadError;

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
#[derive(Debug, Default)]
pub struct Merged {
    /// Tensors averaged from both sides.
    pub blended: usize,
    /// Tensors carried through from one side, when allowed.
    pub carried: usize,
}

/// Average two tensor maps.
///
/// `alpha` is clamped to `[0, 1]`: outside that range is extrapolation, which
/// is a different operation with different failure modes and is not what a
/// caller asking to merge means.
pub fn merge(
    a: &HashMap<String, Tensor>,
    b: &HashMap<String, Tensor>,
    options: &MergeOptions,
) -> Result<(HashMap<String, Tensor>, Merged), LoadError> {
    let alpha = options.alpha.clamp(0.0, 1.0);
    let mut out = HashMap::with_capacity(a.len().max(b.len()));
    let mut report = Merged::default();

    let mut mismatched: Vec<String> = Vec::new();
    let mut unmatched: Vec<String> = Vec::new();

    for (name, left) in a {
        match b.get(name) {
            Some(right) if left.shape() == right.shape() => {
                let dtype = left.dtype();
                // In f32 regardless of storage: averaging f16 in f16 loses a
                // bit of every weight for nothing, since the result is written
                // back at the original width anyway.
                let l = left.to_dtype(DType::F32)?;
                let r = right.to_dtype(DType::F32)?;
                let blended = ((l * (1.0 - alpha))? + (r * alpha)?)?.to_dtype(dtype)?;
                out.insert(name.clone(), blended);
                report.blended += 1;
            }
            Some(_) => mismatched.push(name.clone()),
            None => {
                unmatched.push(name.clone());
                if options.allow_unmatched {
                    out.insert(name.clone(), left.clone());
                    report.carried += 1;
                }
            }
        }
    }

    for (name, right) in b {
        if !a.contains_key(name) {
            unmatched.push(name.clone());
            if options.allow_unmatched {
                out.insert(name.clone(), right.clone());
                report.carried += 1;
            }
        }
    }

    if !mismatched.is_empty() {
        mismatched.sort();
        return Err(LoadError::Unsupported {
            path: std::path::PathBuf::from("<merge>"),
            reason: format!(
                "{} tensors have different shapes in the two checkpoints, first `{}` — \
                 they are not the same architecture",
                mismatched.len(),
                mismatched[0]
            ),
        });
    }
    if !unmatched.is_empty() && !options.allow_unmatched {
        unmatched.sort();
        return Err(LoadError::Unsupported {
            path: std::path::PathBuf::from("<merge>"),
            reason: format!(
                "{} tensors exist in only one checkpoint, first `{}` — pass \
                 allow_unmatched to carry them through if that is intended",
                unmatched.len(),
                unmatched[0]
            ),
        });
    }

    Ok((out, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sd_tensor::Device;

    fn one(name: &str, value: f32, shape: (usize, usize)) -> HashMap<String, Tensor> {
        let n = shape.0 * shape.1;
        let t = Tensor::from_vec(vec![value; n], shape, &Device::Cpu).unwrap();
        [(name.to_string(), t)].into_iter().collect()
    }

    fn value(map: &HashMap<String, Tensor>, name: &str) -> f32 {
        map[name].flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
    }

    #[test]
    fn alpha_selects_between_the_two_checkpoints() {
        let a = one("w", 0.0, (2, 2));
        let b = one("w", 10.0, (2, 2));

        for (alpha, want) in [(0.0, 0.0), (0.25, 2.5), (0.5, 5.0), (1.0, 10.0)] {
            let (out, report) = merge(
                &a,
                &b,
                &MergeOptions {
                    alpha,
                    ..Default::default()
                },
            )
            .expect("merge");
            assert_eq!(report.blended, 1);
            assert!((value(&out, "w") - want).abs() < 1e-6, "alpha {alpha}");
        }
    }

    #[test]
    fn alpha_outside_zero_to_one_is_clamped_not_extrapolated() {
        // Extrapolation is a different operation with different failure modes.
        // Silently doing it because a caller typed 1.5 would be a surprise.
        let a = one("w", 0.0, (2, 2));
        let b = one("w", 10.0, (2, 2));
        let (out, _) = merge(
            &a,
            &b,
            &MergeOptions {
                alpha: 1.5,
                ..Default::default()
            },
        )
        .expect("merge");
        assert!((value(&out, "w") - 10.0).abs() < 1e-6);
    }

    #[test]
    fn a_shape_mismatch_is_refused_rather_than_taking_one_side() {
        // Skipping would silently keep `a`'s weights for that tensor, giving a
        // third model neither caller asked for.
        let a = one("w", 1.0, (2, 2));
        let b = one("w", 1.0, (4, 4));
        assert!(merge(&a, &b, &MergeOptions::default()).is_err());
    }

    #[test]
    fn a_one_sided_tensor_is_refused_unless_allowed() {
        let a = one("only_in_a", 1.0, (2, 2));
        let b = one("only_in_b", 2.0, (2, 2));

        assert!(merge(&a, &b, &MergeOptions::default()).is_err());

        let (out, report) = merge(
            &a,
            &b,
            &MergeOptions {
                allow_unmatched: true,
                ..Default::default()
            },
        )
        .expect("merge");
        assert_eq!(report.blended, 0);
        assert_eq!(report.carried, 2, "both sides carried through");
        assert!((value(&out, "only_in_a") - 1.0).abs() < 1e-6);
        assert!((value(&out, "only_in_b") - 2.0).abs() < 1e-6);
    }
}
