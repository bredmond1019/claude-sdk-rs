# Phase 5 — Documentation Trim: Review Report

> Reviewed: 2026-06-18. Branch: `split-and-trim-for-portfolio`.
> Implements: `planning/phase5/implement.md`

## Verdict: WARN

Phase 5 completed its primary goal — the docs surface is dramatically smaller and free of
wrong-project content. However, `docs/TESTING.md` retains three stale claims that contradict
the trimmed reality, and the plan's own verification step did not catch them.

---

## What passed

### Files removed as planned
All targeted removals confirmed absent:

| Expected removal | Status |
|---|---|
| `GETTING_STARTED.md`, `QUICK_START.md`, `MIGRATION.md`, `RELEASE_NOTES.md`, `ROADMAP.md` | ✅ gone |
| Root `TROUBLESHOOTING.md` (1,313 lines) | ✅ gone |
| `docs/FEATURE_FLAGS.md`, `docs/PERFORMANCE.md`, `docs/MIGRATION.md`, `docs/TROUBLESHOOTING.md` | ✅ gone |
| `docs/tutorials/08`, `09`, `10` | ✅ gone |
| `tests/TEST_INVENTORY.md` | ✅ gone (deviation from plan §5, justified) |

### Naming-drift audit — clean
`grep` across all kept non-README docs (`CONTRIBUTING.md`, `SECURITY.md`, `docs/**`,
`docs/tutorials/00`–`07`) for `claude_ai`, `claude-sdk-rs-{core,mcp,runtime,interactive}`,
`crate_interactive`, `claude-interactive`, `--workspace`, `src/cli`, `src/mcp` → **zero
matches**.

### Rewrites verified correct

**`CONTRIBUTING.md`** (181 lines, was 567): workspace/publish-chain language fully removed;
`cargo test --workspace` → `cargo test`; `Error::ClaudeNotAuthenticated` correctly changed to
`Error::NotAuthenticated` (verified against `src/core/error.rs:154`). No dangling links found.

**`docs/TROUBLESHOOTING.md`** (87 lines, new): uses `CLAUDE_BINARY` env var override, which
is verified present in `src/runtime/process.rs:23`. Organized by real `Error` variants. Links
`NVM_COMPATIBILITY.md`. No invented methods.

**`SECURITY.md`**: version table updated to `2.0.x` / `< 2.0` ✅.

**`docs/TESTING.md`** test-file table: all 15 referenced test files exist on disk ✅.
`cargo test` / `cargo test --all-features` commands are correct ✅. Dangling links to
`TEST_INVENTORY.md` and `PERFORMANCE.md` removed ✅.

### Final doc surface matches plan
```
README.md  CONTRIBUTING.md  SECURITY.md  CHANGELOG.md  LICENSE
docs/  NVM_COMPATIBILITY.md  SECURITY.md  TESTING.md  TROUBLESHOOTING.md
docs/tutorials/  00 01 02 03 04 05 06 07
```
Confirmed. ✅

---

## Issues found (WARN)

### 1. `docs/TESTING.md` — stale "workspace" framing (lines 3 and 16)
```
line 3:  "the claude-sdk-rs workspace"
line 16: "The claude-sdk-rs workspace follows a consistent testing structure across all crates:"
```
This is a single-crate library since Phase 1. These two sentences still read as a multi-crate
workspace, directly contradicting the cleanup goal. The adjacent code block also shows
`crate-name/` in the directory structure template (line 21), which is a generic placeholder
but reinforces the multi-crate framing.

### 2. `docs/TESTING.md` — stale test count of "1,172+" (lines 335 and 359)
```
line 335: "the comprehensive test suite (1,172+ tests)"
line 359: "the extensive test suite (1,172+ tests)"
```
The actual passing test count is **149** (`cargo test --all-features --lib --tests`). The
1,172 figure included tests from the deleted `cli`/`mcp`/`interactive` crates. A reviewer
running `cargo test` and seeing 149 tests against a doc claiming 1,172+ will immediately read
this as AI-generated inflation — exactly the problem this cleanup is trying to fix.

### 3. README dangling links (known, deferred — confirmed)
`README.md` still links `QUICK_START.md`, `DEV_SETUP.md`, and `CLAUDE.md` (lines 132–135,
454), none of which exist. This is correctly deferred to Phase 7 and noted in the implement
doc; confirmed as the only remaining dangling-link source outside `planning/`.

---

## Risks if left as-is

Issues 1 and 2 are in `docs/TESTING.md`, a document that a reviewer or contributor is
likely to read. The "1,172+ tests" figure is the higher-severity of the two: it is a
concrete, checkable number that is 8× wrong and reads as AI inflation. Leaving it increases
the exact "AI slop" risk this project is trying to eliminate. Both are 1-2 line fixes.

---

## Recommended follow-up (pre-Phase 6)

These are small and should be fixed before the Phase 6 naming-consistency pass, since Phase 6
touches source files and may trigger a doc re-check anyway.

1. **`docs/TESTING.md` line 3** — change "workspace" → the library crate name or just
   "claude-sdk-rs".
2. **`docs/TESTING.md` line 16** — change "workspace … all crates" → "single-crate library".
3. **`docs/TESTING.md` lines 335, 359** — change "1,172+" → "~149" (or drop the specific
   number and say "focused test suite").
