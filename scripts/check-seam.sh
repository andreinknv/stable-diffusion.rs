#!/usr/bin/env bash
# Enforce the compute seam.
#
# Only `sd-tensor` may name a backend. Every other crate goes through the seam,
# so replacing the compute backend stays a one-crate change rather than a
# workspace-wide rewrite.
#
# **That is not hypothetical.** This rule is what made replacing candle with
# MLX bounded: 102 files used tensors, and one of them named the library. The
# check is what kept it that way for the year before the swap.
#
# candle is listed alongside mlx because the rule is about *any* backend, not
# about whichever one is current. A crate reaching straight for MLX is the same
# mistake that reaching for candle would have been.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

violations=$(
  grep -rn --include='*.rs' -E '\b(use|extern crate)\s+(candle_(core|nn|transformers)|mlx_sys|mlx_rs)' crates \
    | grep -v '^crates/sd-tensor/' \
    || true
)

# A manifest naming a backend is the same violation one level up.
manifest_violations=$(
  grep -rn --include='Cargo.toml' -E '^\s*(candle-[a-z]+|mlx-[a-z]+)\s*[=.]' crates \
    | grep -v '^crates/sd-tensor/' \
    || true
)

if [ -n "$violations" ] || [ -n "$manifest_violations" ]; then
  echo "error: crates outside sd-tensor may not name a compute backend." >&2
  echo >&2
  [ -n "$violations" ] && echo "$violations" >&2
  [ -n "$manifest_violations" ] && echo "$manifest_violations" >&2
  echo >&2
  echo "Add what you need to sd-tensor and go through it." >&2
  exit 1
fi

echo "seam ok: no crate outside sd-tensor names a backend"
