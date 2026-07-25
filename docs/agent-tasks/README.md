# Agent task specifications

Self-contained tasks for AI coding agents. Each one is scoped so that an agent
with no context beyond `AGENTS.md` and its own task file can complete it, and so
that a wrong answer **fails a test rather than needing review**.

## Handing a task to an agent

Give it exactly two files and nothing else:

```
AGENTS.md
docs/agent-tasks/NN-<task>.md
```

Do not paste extra context, do not explain the architecture, do not add
suggestions of your own. Every deviation from the spec is a chance for the model
to start improvising, which is the failure mode this structure exists to
prevent.

One task per session. Never two.

## Dependency order

Tasks build on each other. Do not start one until its prerequisites are merged
and green.

```
01 clip-tokenizer ──┐
                    ├──► 07 txt2img
02 clip-encoder ────┤         ▲
                    │         │
03 unet-blocks ─────┤         │
        │           │         │
        ▼           │         │
04 unet-attention ──┤         │
        │           │         │
        ▼           │         │
05 unet-assembly ───┘         │
                              │
06 samplers ──────────────────┘
```

| # | Task | Depends on | Difficulty |
|---|---|---|---|
| 01 | [CLIP BPE tokenizer](01-clip-tokenizer.md) | — | low |
| 02 | [CLIP text encoder](02-clip-text-encoder.md) | 01 | medium |
| 03 | [UNet resnet + timestep embedding](03-unet-blocks.md) | — | medium |
| 04 | [UNet attention blocks](04-unet-attention.md) | 03 | high |
| 05 | [UNet assembly](05-unet-assembly.md) | 03, 04 | high |
| 06 | [Samplers](06-samplers.md) | — | low |
| 07 | [txt2img pipeline](07-txt2img.md) | 01–06 | medium |

01, 03 and 06 have no prerequisites and can run in parallel in separate
sessions.

## Why the tasks look repetitive

Every file repeats the same constraints, the same commands, and the same
warnings. That is deliberate. A constraint stated once gets ignored; a
constraint restated in the file the model is actually reading gets followed.

Do not "clean up" the duplication.

## Guardrails

These run in CI and catch a wrong answer without a human reading the diff:

| Guardrail | Catches |
|---|---|
| `scripts/check-seam.sh` | candle imported outside `sd-tensor` |
| golden tests | numerically wrong output, per module |
| structural tests | wrong shapes, channel counts, scale factors |
| `cargo clippy -D warnings` | sloppiness |
| `cargo deny check` | a dependency that changes our licensing |
| `git diff` on test files | **a test edited to make it pass** |

**Always check the last one by hand before merging:**

```bash
git diff --stat -- '*/tests/*' 'xtask/golden/*'
```

An agent that edited a test to go green has produced a repo that lies to you.
That is worse than an agent that failed honestly, and it is the one failure mode
none of the automated gates catch.

## Reviewing agent output

In order, stop at the first failure:

1. `git diff --stat -- '*/tests/*'` — **empty unless the task said otherwise**
2. `git diff -- '*/Cargo.toml'` — no new dependencies
3. `./scripts/check-seam.sh`
4. `cargo test --workspace`
5. `cargo clippy --workspace --all-targets -- -D warnings`
6. Read the diff for invented constants — a hardcoded `0.18215` in the wrong
   place, an `eps` that does not match the reference

Steps 1–5 are mechanical and can be scripted. Step 6 needs you.

## Adding a new task

Copy the structure of `02-clip-text-encoder.md` exactly. Every task must have:

- an explicit **Files you may modify** list (complete, no "and related files")
- an explicit **Files you must NOT modify** list
- **exact function signatures** to implement — not descriptions of them
- the **reference parameter layout**, verbatim, copyable
- **exact verification commands**
- a **known traps** section listing the specific ways this component is
  usually got wrong

If you cannot write the exact signatures yourself, the task is not ready to
delegate.
