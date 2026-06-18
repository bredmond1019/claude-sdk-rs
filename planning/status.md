# Project Status

> Tracks work completed and remaining for the portfolio split-and-trim. See
> `planning/context.md` for why, and `planning/plans/split-and-trim-for-portfolio.md` for how.

**Current phase:** Phases 1–9 complete — all execution done. Next: run the verification gate below.
**Branch:** `split-and-trim-for-portfolio`

## Completed
- [x] Repo assessment / portfolio + AI-slop audit (core strong; `cli` + `mcp` are bloat/wrong-project)
- [x] Confirmed scope with owner: remove MCP entirely, remove CLI + analytics, pure library SDK
- [x] Wrote execution plan → `planning/plans/split-and-trim-for-portfolio.md`
- [x] Wrote `planning/context.md`
- [x] Wrote `planning/status.md`
- [x] **Phase 1 — Remove subsystems** (148 files, 70,394 LOC deleted; `cargo build` green)
  - Reviewed in `planning/phase1/implement.md`; report at `planning/phase1/report.md`
  - Unit/integration tests: **145 passed, 0 failed** (doctests: 83 pre-existing failures, not regressions)

## Decisions resolved
- [x] **Republish vs. fresh repo** → **fresh `*-portfolio` repo**; published `1.x` left
  untouched. Crate version still set to `2.0.0` in this repo to reflect the breaking API
  removal and stay consistent with the drafted README (`claude-sdk-rs = "2"`). *(2026-06-18)*

## Remaining (execution)
Ordered per the plan; check off as completed.

- [x] 1. Remove subsystems: `rm -rf src/mcp`, `rm -rf src/cli`, `rm src/bin/claude-sdk-rs.rs`; delete `*.bak`/`*.disabled.backup` cruft
- [x] 2. Slim `Cargo.toml`: dropped cli/mcp-only deps + unused `reqwest`/`url`/`bytes`, features → `default = []` + `sqlite`, removed `[[bin]]`, bumped to `2.0.0`; deleted `From<InteractiveError>` impl in `src/core/error.rs`. Report: `planning/phase2/implement.md`
  - **Correction to plan §2:** `num_cpus` + `libc` are USED in `src/runtime/error_handling.rs` — kept (plan wrongly listed them for removal).
  - Fixed a pre-existing `sqlite` bug (missing `mode=rwc`) surfaced once `--all-features` could compile; 4 sqlite tests now pass (149 total, was 145).
- [x] 3. Rewrite `src/lib.rs`: dropped `cli`/`mcp` modules + re-exports (incl. `McpConfig`/`McpServer`); rewrote crate doc to core + runtime + optional `sqlite` only; removed cli/analytics/mcp/full feature sections, the phantom `analytics` feature, the CLI-usage section, and stale example list. Build green (default + `sqlite`).
- [x] 4. Examples audit: removed `cli_interactive.rs` / `cli_analytics.rs` (gated on removed features), the non-functional `session_persistence.rs` stub, and the 5.2k-line `REAL_WORLD_EXAMPLES.md`; verified `05_complete_app.rs` + `advanced_permissions.rs` use only public API (kept); scrubbed `claude-interactive` / `claude_sdk_rs_core` drift + dangling `LIBRARY_COMPARISON.md` ref; fixed `run_examples.sh`. `cargo build --examples` clean. Report: `planning/phase4/implement.md`
- [x] 5. Docs trim: removed root `GETTING_STARTED`/`QUICK_START`/`MIGRATION`/`RELEASE_NOTES`/`ROADMAP`, folded `TROUBLESHOOTING` into a short `docs/` FAQ; pruned `docs/` (`FEATURE_FLAGS`, `PERFORMANCE`, dup `MIGRATION`/`TROUBLESHOOTING`) and cli/mcp tutorials `08`–`10`; rewrote `CONTRIBUTING.md` to single-crate, trimmed `docs/TESTING.md`, bumped `SECURITY.md` version table to `2.0.x`; also removed stale `tests/TEST_INVENTORY.md`. Surviving docs: `README`(§7-pending)/`CONTRIBUTING`/`SECURITY`/`CHANGELOG`/`LICENSE` + `docs/{NVM_COMPATIBILITY,SECURITY,TESTING,TROUBLESHOOTING}` + tutorials `00`–`07`. Report: `planning/phase5/implement.md`
- [x] 6. Naming consistency: swept old workspace crate names in doc examples + `tests/`
  (`claude_sdk_rs_core` → `claude_sdk_rs::core`, `claude_sdk_rs_runtime` → `claude_sdk_rs::runtime`,
  and doctest-only `crate::core`/`crate::runtime` → `claude_sdk_rs::*`). Also fixed the residual
  doctest API drift this uncovered (builder `.build()?`, `Error::Timeout(_)`, `Message` enum
  matching, `SessionId::new`, `Bash(ls)` cli format, `TokenUsage.total`). **Doctests now 85/0
  (was 2/83.)** Found: `tests/core/**` + `tests/runtime/**` are orphaned (no `[[test]]` wiring) —
  grep-clean but not compiled; left as a follow-up. Report: `planning/phase6/implement.md`;
  review: `planning/phase6/report.md` — **PASS**
- [x] 7. Write modern `README.md` (replaced the old 458-line cli/mcp/analytics README with a focused ~140-line SDK README; all examples verified against the real public API in `src/lib.rs`)
- [x] 8. Write `CLAUDE.md` (repo guidance for Claude Code; based on plan §8, layout + `ErrorContext`/`CLAUDE_BINARY`/stdin-dispatch references verified against source)
- [x] 9. CHANGELOG `2.0.0` entry (BREAKING: removed `cli`, `analytics`, `mcp`, `full`; features now `default = []` + `sqlite`; added a migration note)

## Verification gate (run after execution)
- [ ] `cargo build` + `cargo build --features sqlite` clean
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` clean
- [ ] `cargo test --all-features` green
- [ ] `cargo build --examples` compiles
- [ ] `cargo bench --no-run` compiles
- [ ] `cargo package --allow-dirty` dry-run OK
- [ ] `grep -rn "claude-interactive\|crate_interactive\|src/cli\|src/mcp"` → no stale refs
- [ ] `cargo doc --no-deps` builds

## Notes / log
- **2026-06-18** — Phase 1 complete. 148 files / 70,394 LOC removed (`src/cli`, `src/mcp`,
  `src/bin`, `tests/cli`, `tests/mcp`, cruft). Default build green. 145 unit/integration
  tests pass; 83 doctest failures are pre-existing naming-drift bugs (`claude_sdk_rs_core`
  wrong crate name in doc examples), not regressions. Phase 2 ready to start once republish
  decision is resolved.
- **2026-06-18** — Phase 2 complete. Slimmed `Cargo.toml` (version → `2.0.0`; features →
  `default = []` + `sqlite`; removed `[[bin]]`; dropped ~30 cli/mcp-only deps plus unused
  `reqwest`/`url`/`bytes` and dead dev-deps `insta`/`wiremock`/`dotenvy`/`tokio-test`).
  Deleted the `From<InteractiveError>` impl in `src/core/error.rs`. Removing the dead
  `[[bin]]` unblocked `--all-features`, which exposed a pre-existing `sqlite` connection bug
  (missing `mode=rwc`) — fixed in `src/core/session/sqlite_storage.rs`. Build green (default
  + `sqlite`); `cargo test --all-features` 149 passed / 0 failed (up from 145). 83 doctest
  failures remain (pre-existing naming drift → §6). `cargo package` produces a valid
  `v2.0.0`. Clippy still reports ~247 pre-existing findings across untouched core/runtime —
  out of Phase 2 scope, flagged for the final gate. Report: `planning/phase2/implement.md`.
- **2026-06-18** — Phase 3 complete. Rewrote `src/lib.rs`: removed the `#[cfg(feature =
  "cli")]` / `#[cfg(feature = "mcp")]` module declarations and the `McpConfig`/`McpServer`
  re-exports; rewrote the crate-level doc to describe only core + runtime + optional `sqlite`
  (dropped the cli/analytics/mcp/full feature sections, the phantom `analytics` feature, the
  CLI-usage section, and refreshed the example list to surviving examples). Kept the lint
  attributes and `prelude`. `cargo build` and `cargo build --features sqlite` green (only
  pre-existing warnings). The 83 doctest failures are unchanged and confirmed to be the
  pre-existing naming drift (`claude_sdk_rs_core` / `claude_sdk_rs_runtime` stale workspace
  crate names in doc examples) — to be fixed in §6, not introduced here.
- **2026-06-18** — Phase 4 complete. Examples audit: removed `cli_interactive.rs` and
  `cli_analytics.rs` (gated on the removed `cli`/`analytics` features), the non-functional
  `session_persistence.rs` stub (printed "cannot be run", referenced `claude_sdk_rs_core`), and
  the 5,211-line aspirational `REAL_WORLD_EXAMPLES.md` (untested code with non-existent deps +
  a clap CLI section contradicting the library-only story). Verified `05_complete_app.rs` and
  `advanced_permissions.rs` use only public SDK API → kept. Fixed examples-local naming drift
  (`claude-interactive` disclaimers in `01`/`02`; old-workspace sample output in `04_tools.rs`;
  `run_examples.sh` header) and the dangling `LIBRARY_COMPARISON.md` link in `examples/README.md`.
  13 examples survive; `cargo build --examples` compiles clean (pre-existing dead-code warnings
  only); grep for stale refs in `examples/` → none. Two deviations from plan §4 keep-list
  (removed `session_persistence.rs` and `REAL_WORLD_EXAMPLES.md` rather than trimming) with
  rationale in the report. Report: `planning/phase4/implement.md`.
- **2026-06-18** — Phase 5 post-review fixup. Patched three stale entries in `docs/TESTING.md`
  caught by the phase review (`planning/phase5/report.md`): "workspace … all crates" framing on
  lines 3 and 16 → single-crate language; "1,172+ tests" count on lines 335 and 359 → "~149
  tests" (actual passing count from `cargo test --all-features --lib --tests`). Phase 5 report
  grade: WARN → issues now resolved.
- **2026-06-18** — Phase 6 review complete. Grade: PASS. All claims in `planning/phase6/implement.md`
  independently verified: `cargo test --doc --all-features` confirmed 85/0 live; unit suite 149/0;
  stale-name grep empty across all shipping files; both default + `sqlite` builds clean. Open items
  noted: orphaned `tests/core/**` + `tests/runtime/**` (never compiled by Cargo, follow-up before
  final gate) and pre-existing `missing_docs` warning on `sqlite` build (final clippy gate).
  Report: `planning/phase6/report.md`.
- **2026-06-18** — Phase 9 complete. Added a `[2.0.0] - 2026-06-18` CHANGELOG entry above `[1.0.1]`:
  documents the focused library refocus and the breaking removals (`cli` + binary target,
  `analytics`, `mcp` module + `McpConfig`/`McpServer` re-exports, `full`), the new `default = []` +
  `sqlite` feature set, a migration note (core-only users → `"2"` with no code changes; cli/mcp/
  analytics users → pin `1.x`), and an internal-changes summary (dep slim-down, sqlite `mode=rwc`
  fix, docs/README/CLAUDE.md). Version confirmed `2.0.0` in `Cargo.toml`. All nine execution phases
  are now done — only the verification gate remains.
- **2026-06-18** — Phase 8 complete. Wrote `CLAUDE.md` (repo guidance for Claude Code) from plan §8.
  Verified every concrete reference against source before committing it to the file: `ErrorContext`
  (`src/runtime/error_handling.rs:16`), the `CLAUDE_BINARY` override (`src/runtime/process.rs:23`),
  and the stdin-vs-arg dispatch (`src/runtime/process.rs:131`). Adjusted the draft's git-hooks line
  to name both `.pre-commit-config.yaml` and `.githooks/` (both tracked), used the `--all-features`
  clippy invocation to match the verification gate, and added the explicit no-`Co-Authored-By` commit
  rule. Next: §9 CHANGELOG `2.0.0` entry.
- **2026-06-18** — Phase 7 complete. Replaced the stale 458-line root `README.md` (which still
  documented the removed cli/mcp/analytics features, `claude-interactive` naming, and a phantom
  `analytics` flag) with a focused ~140-line SDK README based on plan §7. Corrected three errors in
  the plan's draft against the verified public API in `src/lib.rs`: (1) the streaming example yields
  typed `Message` variants, not string chunks — used the `match Message::Assistant { content, .. }`
  pattern, not `print!("{}", chunk?)`; (2) `Config::builder()`/`Client::builder()` `.build()` returns
  a `Result`, so examples use `.build()?`; (3) used the CLI model alias `"sonnet"` instead of the
  stale `claude-3-sonnet-20240229` id. Grep for stale refs (`claude-interactive`/`analytics`/`mcp`/
  `claude_sdk_rs_core`) in README → clean. Next: §8 `CLAUDE.md`, §9 CHANGELOG.
- **2026-06-18** — Phase 5 complete. Docs trim. Removed root `GETTING_STARTED`, `QUICK_START`,
  `MIGRATION`, `RELEASE_NOTES`, `ROADMAP`, and the 1,313-line `TROUBLESHOOTING` (folded into a
  new ~90-line `docs/TROUBLESHOOTING.md` FAQ organized by real `Error` variants, using the
  verified `CLAUDE_BINARY` override). Pruned `docs/`: `FEATURE_FLAGS`, `PERFORMANCE`, duplicate
  `MIGRATION` + `TROUBLESHOOTING`. Removed cli/app tutorials `08`–`10`; kept + verified `00`–`07`
  are free of crate-name drift. Rewrote `CONTRIBUTING.md` (567→~190 lines, single-crate; dropped
  the multi-crate publish chain + dangling links; fixed `Error::ClaudeNotAuthenticated` →
  `Error::NotAuthenticated`). Trimmed `docs/TESTING.md` (real test-file table, `cargo test` not
  `--workspace`, fixed `claude_ai` import). Bumped `SECURITY.md` version table to `2.0.x`.
  Deviation: also removed `tests/TEST_INVENTORY.md` (~70% stale removed-crate content + "Agent 2"
  slop) and dropped its link. README rewrite + doctest naming drift remain (§6/§7). Report:
  `planning/phase5/implement.md`.
