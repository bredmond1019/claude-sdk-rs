# Phase 1 Review Report

**Reviewer:** Claude (session 2026-06-18)
**Branch:** `split-and-trim-for-portfolio`
**Source:** `planning/phase1/implement.md`

---

## Summary verdict: APPROVED with notes

Phase 1 was executed correctly. All required deletions are staged, the default build is
green, and the two scope judgment calls are sound. Three issues warrant attention before
Phase 2 begins — none are blocking.

---

## What was done vs. what was required

| Required (plan §1) | Done | Notes |
|---|---|---|
| `rm -rf src/mcp/` (70 files, 14,981 LOC) | ✅ | |
| `rm -rf src/cli/` (63 files, 47,160 LOC) | ✅ | |
| `rm src/bin/claude-sdk-rs.rs` | ✅ | |
| Delete `src/core/config_test.rs.bak` | ✅ | |
| Delete 4× `*.disabled.backup` files | ✅ | Removed with `src/mcp/` |
| `tests/cli/` + `tests/mcp/` (13 files, 5,944 LOC) | ✅ | Scope judgment call — see below |

**Total removed:** 148 files, 70,394 LOC. Retained: 35 `.rs` files, ~19,277 LOC
across `src/core/` + `src/runtime/` + `tests/core/` + `tests/runtime/`.

---

## Scope judgment calls: both approved

**1. Removal of `tests/cli/` and `tests/mcp/`** — correct call. These test files
`use claude_sdk_rs_interactive::…` and `use claude_sdk_rs_mcp::…` — crate names that do
not exist in the single-crate layout. They were already dead code. Removing them is
required for a clean `cargo test` and is faithful to "remove entire subsystems."

**2. Deferring the `cfg(feature = "cli")` `From<InteractiveError>` impl in
`src/core/error.rs:257-304`** — correct call. Under the default build this block is
inert (the `cli` feature is inactive by default), so leaving it does not break Phase 1.
Touching it belongs with the `cli` feature removal in Phase 2/3.
**Action required:** Phase 2/3 must delete this `From` impl when the `cli` feature entry
is removed from `Cargo.toml`, or `--all-features` / `--features cli` builds will fail
at compile time (missing module).

---

## Dangling references: all correctly identified, all deferred

The implement.md accurately catalogued every known dangling reference and assigned each
to the correct future phase. No new ones were found:

| Location | Issue | Phase |
|---|---|---|
| `Cargo.toml` | `[features] cli/mcp/analytics/full`; `[[bin]]` target; `[[bench]] required-features = ["cli"]` | 2 |
| `src/lib.rs` | `#[cfg(feature="mcp")] pub mod mcp`, `#[cfg(feature="cli")] pub mod cli`, `pub use crate::mcp::…` re-exports | 3 |
| `src/core/error.rs` | `#[cfg(feature="cli")] impl From<crate::cli::error::InteractiveError>` | 2/3 |

All three are inert under `--no-default-features` / default builds, which is why
`cargo build` passes cleanly.

---

## Test suite results (run 2026-06-18)

### Unit + integration tests (`cargo test --lib --tests`)

```
test result: ok. 145 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out; finished in 2.01s
```

**All 145 unit/integration tests pass. 0 failures. 4 intentionally ignored.**

### Doctests (`cargo test --doc`)

```
test result: FAILED. 2 passed; 83 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.71s
```

**83 doctest failures — all pre-existing, none introduced by Phase 1.**

Confirmed via `git stash` / restore: the identical 83 failures occur on the pre-Phase-1
tree (before any staged deletions). Two root causes, both pre-existing naming drift:

1. **`use claude_sdk_rs_core::…`** throughout `src/core/config.rs`, `src/core/types.rs`,
   `src/core/error.rs`, `src/core/message.rs` doc examples. The crate is named
   `claude_sdk_rs`, not `claude_sdk_rs_core`. Rustc even suggests the fix.
2. **`use crate::core`** in `src/runtime/client.rs` doc examples. Internal `crate::`
   paths are not valid in doctests (doctests compile as separate crates).

These failures are part of the "AI slop" naming drift described in `planning/context.md`
and must be fixed in the Phase 3 (lib.rs + doc rewrite) or Phase 6 (naming consistency)
pass. They are **not a regression from Phase 1**.

### `--all-features` build

`cargo test --all-features` errors at target resolution:

```
error: can't find bin `claude-sdk-rs` at path `…/src/bin/claude-sdk-rs.rs`
```

Expected and documented: the `[[bin]]` entry in `Cargo.toml` still names the deleted
file, and `--all-features` activates the `cli` feature which makes the target required.
This is resolved by Phase 2 (remove `[[bin]]` from `Cargo.toml`).

---

## Issues found (none blocking Phase 2)

| # | Severity | Issue | Recommendation |
|---|---|---|---|
| 1 | Note | 83 pre-existing doctest failures due to `claude_sdk_rs_core` / `crate::core` wrong imports | Fix in Phase 3/6 along with naming consistency sweep |
| 2 | Note | `--all-features` build broken (expected — `[[bin]]` + deleted file) | Fix in Phase 2 |
| 3 | Required for Phase 2 | `From<InteractiveError>` impl in `error.rs` must be deleted alongside `cli` feature removal | Add to Phase 2 task list explicitly |

---

## Checklist (from implement.md)

- [x] Approve removal of `tests/cli/` + `tests/mcp/` as in-scope for "remove subsystems."
- [x] Confirm `src/core/error.rs` `From<InteractiveError>` impl is correctly deferred —
      **and add it explicitly to the Phase 2/3 task list.**
- [x] Confirm nothing in `core`/`runtime` legitimately depended on removed code — build proves it.
- [ ] **Deletions are staged but not committed — a commit should close Phase 1.**

---

## Recommendation

Phase 1 is complete and correct. Commit the staged deletions to close Phase 1, then
proceed to Phase 2 (Cargo.toml slim-down), keeping the three items above in mind.
