//! T5's relative position bucketing.
//!
//! T5 has no positional embeddings. Instead every attention score gets an
//! additive, per-head bias that depends only on the *distance* between the two
//! tokens. Distances are grouped into buckets: exact for near neighbours,
//! logarithmically coarsening further out, so a fixed table covers unbounded
//! distance.
//!
//! Small integer arithmetic, computed on the host. It runs once per forward
//! pass over a `seq x seq` grid and building it as a tensor graph would be
//! slower and much harder to read.

/// Bucket index for every `(query, key)` pair, row-major `[q_len * k_len]`.
///
/// With `bidirectional` set — which is what an encoder wants — the table is
/// split in half, one half for keys before the query and one for keys after,
/// so direction is preserved rather than collapsed onto distance.
pub fn relative_position_bucket(
    q_len: usize,
    k_len: usize,
    bidirectional: bool,
    num_buckets: usize,
    max_distance: usize,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(q_len * k_len);

    // Halved for the bidirectional split, and every later bound is in terms
    // of the halved value — a detail easy to lose, and one that silently
    // halves the model's usable position range if lost.
    let n_buckets = if bidirectional {
        num_buckets / 2
    } else {
        num_buckets
    };
    let max_exact = n_buckets / 2;

    for q in 0..q_len {
        for k in 0..k_len {
            // Key position minus query position: positive means the key is
            // to the *right* of the query.
            let relative = k as i64 - q as i64;

            let (mut bucket, distance) = if bidirectional {
                let sign_offset = if relative > 0 { n_buckets } else { 0 };
                (sign_offset, relative.unsigned_abs() as usize)
            } else {
                // Causal: everything ahead collapses to distance 0.
                (0, (-relative).max(0) as usize)
            };

            bucket += if distance < max_exact {
                // Near neighbours get their own bucket each.
                distance
            } else {
                // Beyond that, log-spaced. Clamped because the formula keeps
                // growing past the end of the table for distances over
                // `max_distance`, and indexing off the end of the embedding
                // is the failure this prevents.
                let ratio = (distance as f64 / max_exact as f64).ln()
                    / (max_distance as f64 / max_exact as f64).ln();
                let scaled = max_exact + (ratio * (n_buckets - max_exact) as f64) as usize;
                scaled.min(n_buckets - 1)
            };

            out.push(bucket as u32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUCKETS: usize = 32;
    const MAX_DIST: usize = 128;

    fn grid(n: usize) -> Vec<u32> {
        relative_position_bucket(n, n, true, BUCKETS, MAX_DIST)
    }

    #[test]
    fn every_bucket_is_in_range() {
        for &n in &[1usize, 8, 77, 512] {
            let g = relative_position_bucket(n, n, true, BUCKETS, MAX_DIST);
            assert_eq!(g.len(), n * n);
            assert!(
                g.iter().all(|&b| (b as usize) < BUCKETS),
                "a bucket index off the end of the table indexes out of bounds \
                 in the embedding, at n = {n}"
            );
        }
    }

    #[test]
    fn self_attention_lands_in_bucket_zero() {
        let n = 16;
        let g = grid(n);
        for i in 0..n {
            assert_eq!(g[i * n + i], 0, "distance 0 should be bucket 0 at {i}");
        }
    }

    #[test]
    fn direction_is_distinguished() {
        // The whole reason the table is split in half. Collapsing direction
        // gives a symmetric bias and a model that cannot tell word order.
        let n = 16;
        let g = grid(n);
        let (q, d) = (8usize, 3usize);
        let ahead = g[q * n + (q + d)];
        let behind = g[q * n + (q - d)];
        assert_ne!(
            ahead, behind,
            "keys {d} ahead and {d} behind must use different buckets"
        );
        assert!(
            ahead >= (BUCKETS / 2) as u32,
            "keys to the right belong in the upper half, got {ahead}"
        );
        assert!(behind < (BUCKETS / 2) as u32);
    }

    #[test]
    fn near_distances_are_exact_and_far_ones_saturate() {
        let n = 600;
        let g = relative_position_bucket(n, n, true, BUCKETS, MAX_DIST);
        let q = 0usize; // every key is to the right of query 0
        let max_exact = (BUCKETS / 2) / 2;

        // Exact region: one bucket per distance.
        for d in 1..max_exact {
            assert_eq!(
                g[q * n + d] as usize,
                BUCKETS / 2 + d,
                "distance {d} should have its own bucket"
            );
        }
        // Far region: saturates rather than running off the table.
        assert_eq!(
            g[q * n + (n - 1)] as usize,
            BUCKETS - 1,
            "the furthest distance should sit in the last bucket"
        );
        // And is monotonic in between — coarsening, never going backwards.
        for d in max_exact..(n - 1) {
            assert!(g[q * n + d] <= g[q * n + d + 1], "went backwards at {d}");
        }
    }

    #[test]
    fn causal_mode_collapses_everything_ahead() {
        let n = 8;
        let g = relative_position_bucket(n, n, false, BUCKETS, MAX_DIST);
        for q in 0..n {
            for k in (q + 1)..n {
                assert_eq!(
                    g[q * n + k],
                    0,
                    "unidirectional buckets treat everything ahead as distance 0"
                );
            }
        }
    }
}
