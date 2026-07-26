#!/usr/bin/env bash
# Enforce the native-code budget.
#
# This project aims to be as close to all-Rust as its dependencies allow. That
# is only meaningful if it is checked: a C dependency arrives transitively, in
# someone else's Cargo.toml, and nothing about your build looks different when
# it does.
#
# So: every crate that compiles native code must be listed here, with a reason.
# Anything else fails the build.
#
# Usage:
#   ./scripts/check-native-deps.sh            # default (CPU) build
#   ./scripts/check-native-deps.sh metal      # with a backend feature

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

FEATURES="${1:-}"

# ---------------------------------------------------------------------------
# Crates permitted to compile C/C++, and why.
#
# The goal is for this list to be EMPTY on a default build. Removing an entry
# is a win; adding one needs a justification in docs/native-deps.md.
# ---------------------------------------------------------------------------
allowed_default=(
  # oniguruma, a C regex engine. Reaches us via candle-core -> tokenizers,
  # which hardcodes features = ["onig"]. Cargo unifies features across the
  # graph, so we cannot switch it off from here.
  #
  # Fixable upstream in one line (tokenizers ships a pure-Rust `fancy-regex`
  # backend exposing the same SysRegex type). Verified; see docs/native-deps.md.
  # Delete this entry when candle drops it.
  "onig_sys"
)

# GPU backends compile kernels written in CUDA C++ / Metal Shading Language.
# That is inherent to the GPU programming model, not a dependency choice —
# candle has no Rust-authored kernel path. Only a backend with a Rust kernel
# DSL (cubecl, rust-gpu) could remove these, which is a seam-level decision.
allowed_cuda=("candle-kernels" "cudaforge")
allowed_metal=("candle-metal-kernels")

allowed=("${allowed_default[@]}")
case "$FEATURES" in
  *cuda*)  allowed+=("${allowed_cuda[@]}") ;;
esac
case "$FEATURES" in
  *metal*) allowed+=("${allowed_metal[@]}") ;;
esac

# ---------------------------------------------------------------------------
# Find every package in the resolved graph with `cc` as a build-dependency.
# `cc` is the crate that shells out to a C/C++ compiler, so depending on it is
# the reliable signal that native code gets built.
# ---------------------------------------------------------------------------
# `-e normal,build` is required, not `-e build`: the path from our crates to
# `cc` runs through normal dependencies before reaching a build-dependency, and
# `-e build` alone finds nothing and silently reports a clean tree.
tree_args=(-p sd-cli -e normal,build -i cc)
[ -n "$FEATURES" ] && tree_args+=(--features "$FEATURES")

# Direct dependents of `cc` sit at depth 1, i.e. lines with no leading indent.
offenders=$(cargo tree "${tree_args[@]}" 2>/dev/null \
  | grep -oE '^[├└]── [a-z0-9_-]+' | sed 's/^[├└]── //' | sort -u || true)

label="${FEATURES:-default}"

if [ -z "$offenders" ]; then
  echo "native deps ok [$label]: nothing compiles C — this build is all Rust"
  exit 0
fi

unexpected=""
for c in $offenders; do
  ok=false
  for a in "${allowed[@]}"; do
    [ "$c" = "$a" ] && ok=true && break
  done
  $ok || unexpected="$unexpected $c"
done

if [ -n "$unexpected" ]; then
  echo "error: new native (C/C++) dependency introduced [$label]:" >&2
  for c in $unexpected; do
    echo "    $c" >&2
    cargo tree -p sd-cli ${FEATURES:+--features "$FEATURES"} -e normal,build -i "$c" 2>/dev/null \
      | head -8 | sed 's/^/      /' >&2
  done
  echo >&2
  echo "This project tracks its native-code surface deliberately." >&2
  echo "Either avoid the dependency, or add it to the allowlist in" >&2
  echo "scripts/check-native-deps.sh with a justification in docs/native-deps.md." >&2
  exit 1
fi

echo "native deps ok [$label]: $(echo $offenders | wc -w | tr -d ' ') allowlisted, 0 unexpected"
for c in $offenders; do echo "    $c (known)"; done
