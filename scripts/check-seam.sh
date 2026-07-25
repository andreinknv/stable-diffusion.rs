#!/usr/bin/env bash
# Enforce the compute seam.
#
# Only `sd-tensor` may name candle. Every other crate goes through the seam so
# that replacing the compute backend stays a one-crate change instead of a
# workspace-wide rewrite.
#
# This is the single check that keeps the seam real. Without it the abstraction
# erodes within a month and the escape hatch quietly disappears.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

violations=$(
  grep -rn --include='*.rs' -E '\b(use|extern crate)\s+candle_(core|nn|transformers)' crates \
    | grep -v '^crates/sd-tensor/' \
    || true
)

if [ -n "$violations" ]; then
  echo "error: crates outside sd-tensor may not depend on candle directly." >&2
  echo >&2
  echo "$violations" >&2
  echo >&2
  echo "Fix: add what you need to crates/sd-tensor/src/lib.rs and use it from there." >&2
  exit 1
fi

# Also catch it at the manifest level, which grep on .rs files would miss.
manifest_violations=$(
  grep -ln -E '^\s*candle-(core|nn|transformers)' crates/*/Cargo.toml \
    | grep -v 'crates/sd-tensor/Cargo.toml' \
    || true
)

if [ -n "$manifest_violations" ]; then
  echo "error: only sd-tensor may declare a candle dependency." >&2
  echo "$manifest_violations" >&2
  exit 1
fi

echo "seam ok: candle is confined to sd-tensor"
