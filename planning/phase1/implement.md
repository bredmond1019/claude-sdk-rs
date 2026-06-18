# Phase 1 — Remove Entire Subsystems — Implementation Report

**Branch:** `split-and-trim-for-portfolio`
**Date:** 2026-06-18
**Plan reference:** `planning/plans/split-and-trim-for-portfolio.md` §1
**Scope:** Step 1 only ("Remove entire subsystems"). Cargo.toml slim-down (§2) and
`lib.rs` rewrite (§3) are intentionally **out of scope** for this phase.

---

## What Phase 1 required (per plan §1)

- `rm -rf src/mcp/`
- `rm -rf src/cli/`
- `rm src/bin/claude-sdk-rs.rs` (the `[[bin]]` target removal itself is Phase 2)
- Delete cruft: `src/core/config_test.rs.bak` + the 4 `*.disabled.backup` files under `src/mcp/`

## What I did

All deletions staged with `git rm` (recorded in the index, not yet committed — no commit
was requested).

| Removal | Files | LOC |
|---|---|---|
| `src/mcp/` (customer_support, knowledge_base, helpscout/slack/notion clients & services) | 70 | 14,981 |
| `src/cli/` (analytics, cost, history, profiling, execution, output, session, commands) | 63 | 47,160 |
| `src/bin/claude-sdk-rs.rs` | 1 | 36 |
| `src/core/config_test.rs.bak` (cruft) | 1 | — |
| 4× `*.disabled.backup` (all under `src/mcp/`, removed with the directory) | (incl. above) | — |
| `tests/cli/` + `tests/mcp/` (see scope note below) | 13 | 5,944 |

**Totals staged:** 148 files deleted, **70,394 lines removed**.

Remaining tracked source/test tree:
```
src/core/**      src/runtime/**      tests/core/**      tests/runtime/**
```
35 `.rs` files, ~19,277 LOC of source+tests remaining (down from ~93k).

## Scope judgment calls (flag for review)

1. **Removed `tests/cli/` and `tests/mcp/` (13 files, 5,944 LOC)** — not enumerated
   verbatim in plan §1, but they are the integration-test directories *of* the removed
   subsystems. They `use claude_sdk_rs_interactive::…` and `use claude_sdk_rs_mcp::…` —
   crate names that **do not exist** in the current single-crate layout, so they were
   already dead (would not compile). Removing them is faithful to "remove entire
   subsystems" and required for a clean `cargo test`. **Please confirm this is acceptable.**

2. **Left `src/core/error.rs:257-304` in place** — there is a
   `#[cfg(feature = "cli")] impl From<crate::cli::error::InteractiveError> for Error`
   block in *kept* code that references the now-deleted `cli` module. This contradicts the
   plan's stated assumption that "core/runtime have **zero** `crate::cli`/`crate::mcp`
   references." It is `cfg(feature = "cli")`-gated, so it is **inert under the default
   build** and does not break Phase 1. I deliberately did **not** edit it here because
   touching `core/error.rs` belongs with the `cli`-feature removal in Phase 2/§3.
   **⚠️ Phase 2/3 MUST delete this `From` impl** when the `cli` feature is removed, or any
   `--features cli` / `--all-features` build will fail to compile (missing module).

## Verification performed

- ✅ **`cargo build` (default features) — passes.** Only one pre-existing, unrelated
  dead-code warning (`src/runtime/recovery.rs:36`, `avg_connection_duration` never read);
  not introduced by this phase.
- ✅ `grep -rn 'crate::cli|crate::mcp' src/core src/runtime` → only the single inert,
  cfg-gated `From` impl noted above. No other references in kept code.
- ✅ No `*.bak` / `*.disabled.backup` files remain anywhere (excluding `target/`).
- ✅ `src/` now contains only `core/` and `runtime/`.

## Known-dangling references (expected — deferred to later phases, NOT bugs)

These still point at the removed code and are correct to leave for now:

- **`Cargo.toml`** (Phase 2): `[features] mcp/cli/analytics/full` still defined; `[[bin]]`
  section (lines ~112-115) still points at the deleted `src/bin/claude-sdk-rs.rs`;
  `[[bench]]` sets `required-features = ["cli"]` (line ~124). Note: none of these break
  the *default* build — Cargo did not error on the missing `[[bin]]` path because its
  `required-features = ["cli"]` is inactive by default.
- **`src/lib.rs`** (Phase 3): `#[cfg(feature="mcp")] pub mod mcp;`,
  `#[cfg(feature="cli")] pub mod cli;`, the `pub use crate::mcp::{McpConfig, McpServer}`
  re-export (~line 339-340), and the crate-doc feature sections describing cli/analytics/mcp.
- **`src/core/error.rs`** (Phase 2/3): the cfg-gated `From<InteractiveError>` impl above.

## Build status after Phase 1

- Default `cargo build`: **green**.
- `cargo build --features cli|mcp|full` / `--all-features`: **will fail** (modules deleted,
  features still declared). This is expected and is resolved by Phase 2 (Cargo.toml) +
  Phase 3 (lib.rs + error.rs). The full verification gate runs after those phases.

## Not done (by design — later phases)

§2 Cargo.toml slim-down · §3 lib.rs rewrite · §4 examples audit · §5 docs trim ·
§6 naming consistency · §7 README · §8 CLAUDE.md · §9 CHANGELOG/version bump.

## Suggested review checklist

- [ ] Approve removal of `tests/cli/` + `tests/mcp/` as in-scope for "remove subsystems."
- [ ] Confirm the `src/core/error.rs` `From<InteractiveError>` impl is correctly deferred
      (and add it to the Phase 2/3 task list).
- [ ] Confirm nothing in `core`/`runtime` legitimately depended on removed code (build proves it).
- [ ] Deletions are staged but **not committed** — confirm whether a commit should close Phase 1.
