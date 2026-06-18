# Phase 6 — Naming Consistency: Review Report

> Plan ref: `planning/plans/split-and-trim-for-portfolio.md` §6
> Implement doc: `planning/phase6/implement.md`
> Reviewed: 2026-06-18

## Grade: PASS

---

## Summary

Phase 6 delivered everything §6 scoped and more. The original mandate was narrow
("grep and fix `claude-interactive` / `crate_interactive`"), but the implement doc
correctly identified that the real problem was broader naming drift in three distinct
forms, and fixed all of them. The headline result — doctests flipped from **2 passed /
83 failed** to **85 passed / 0 failed** — was independently verified by running
`cargo test --doc --all-features` against the current branch. Every factual claim in
the implement doc checked out.

---

## Verification Against Claims

| Claim | Verified |
|---|---|
| `cargo build` green | ✅ (only pre-existing `StreamHealthMetrics` dead-code warning) |
| `cargo build --features sqlite` green | ✅ |
| `cargo test --doc --all-features` → 85 passed / 0 failed | ✅ confirmed live |
| `cargo test --all-features` → 149 unit tests, 4 ignored, 0 failed | ✅ confirmed live |
| No stale `claude_sdk_rs_core` / `claude_sdk_rs_runtime` / `crate_interactive` / `claude-interactive` in `src tests examples docs benches Cargo.toml` | ✅ grep returns empty |
| No `crate::core` / `crate::runtime` inside `///` doc comments | ✅ grep returns empty |
| `tests/core/**` + `tests/runtime/**` are orphaned (no `[[test]]` wiring) | ✅ dirs exist; no `[[test]]` in `Cargo.toml` |

---

## What Was Done Well

**Scope diagnosis was accurate.** The implement doc correctly identified three
distinct defect classes where the plan only anticipated one. Fixing the residual
doctest API drift (builder `.build()?`, `Error::Timeout(_)`, `Message` enum,
`SessionId::new`, `Bash(ls)` CLI format, `TokenUsage.total`) was essential — without
those fixes the doctests would have been compilation-green but semantically broken.

**Surgical line guards.** The doctest-only `crate::core` → `claude_sdk_rs::core`
substitution was scoped to `///` comment lines so it didn't disturb the valid
`crate::core` / `crate::runtime` paths used in real (non-doc) code. That's the right
call and the implement doc called it out explicitly.

**Honest about what wasn't done.** The orphaned `tests/core/**` and
`tests/runtime/**` subdirectories were correctly flagged as unverified by compilation
(Cargo never compiles them) and left as a named follow-up rather than quietly
included in the "149 tests pass" number. This is honest accounting.

**No scope creep.** The phase stayed focused. Clippy findings, the orphaned test
wiring, and the remaining plan items (§7–§9) were all left for their appropriate
phases.

---

## Open Issues

### 1. Orphaned integration test directories (follow-up, not a §6 defect)

`tests/core/` (6 files) and `tests/runtime/` (5 files) are grep-clean on naming
but have never been compiled since the workspace was dissolved. They include files
named `penetration_tests.rs`, `chaos_engineering_tests.rs`, `security_tests.rs`,
and `property_tests.rs` — likely high-value tests if they compile. Wiring them in
may surface further API-drift failures. This is explicitly deferred; it should be
addressed before the final verification gate (§ post-9) rather than left indefinitely.

### 2. `missing_docs` warning on `sqlite` build (pre-existing, low priority)

`cargo build --features sqlite` emits a `missing_docs` warning in addition to the
`StreamHealthMetrics` dead-code warning from the default build. Neither is a §6
regression; both pre-date this phase. Noted for the final clippy gate.

---

## Remaining Plan Items

Phases 7–9 remain before the final verification gate:

- [ ] §7 — Write modern `README.md` (draft ready in plan)
- [ ] §8 — Write `CLAUDE.md` (draft ready in plan)
- [ ] §9 — CHANGELOG `2.0.0` entry
- [ ] Final gate — `cargo clippy --all-targets --all-features -- -D warnings`; `cargo package --allow-dirty`
- [ ] Follow-up — wire up `tests/core/**` + `tests/runtime/**` or remove them
