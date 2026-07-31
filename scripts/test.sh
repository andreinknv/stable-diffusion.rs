#!/usr/bin/env bash
# The full suite, with the settings it actually needs.
#
# `--test-threads=3`: each pipeline test loads a full model into unified
# memory, and at cargo's default (the core count) the OOM killer takes the test
# binary. That surfaces as `signal: 9, SIGKILL` with no failing assertion,
# which reads exactly like a crash in the code under test and has been
# misdiagnosed in both directions here.
#
# `SD_TEST_MODEL_DIR` should point at a diffusers SD 1.5 directory. Without it
# most pipeline tests skip themselves; `SD_REQUIRE_FIXTURES=1` turns those
# skips into failures, which is what to use when you believe the data is there.
set -euo pipefail

threads="${SD_TEST_THREADS:-3}"
exec cargo test --release --workspace --features mlx "$@" -- --test-threads="$threads"
