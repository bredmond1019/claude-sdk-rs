# Project Status

> Tracks work completed and remaining for the portfolio split-and-trim. See
> `planning/context.md` for why, and `planning/plans/split-and-trim-for-portfolio.md` for how.

**Current phase:** Phase 3 complete — ready to begin Phase 4 (examples audit).
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
- [ ] 4. Examples audit: remove `cli_interactive.rs` / `cli_analytics.rs`; verify `05_complete_app.rs` + `advanced_permissions.rs`; trim example docs + `run_examples.sh`
- [ ] 5. Docs trim: remove/fold `GETTING_STARTED`, `QUICK_START`, `MIGRATION`, `RELEASE_NOTES`, `ROADMAP`, oversized `TROUBLESHOOTING`; prune `docs/` (`FEATURE_FLAGS`, `PERFORMANCE`, dup `MIGRATION`/`TROUBLESHOOTING`) and cli/mcp tutorials
- [ ] 6. Naming consistency: replace `claude-interactive` / `crate_interactive` → `claude-sdk-rs`
- [ ] 7. Write modern `README.md` (draft ready in plan §7)
- [ ] 8. Write `CLAUDE.md` (draft ready in plan §8)
- [ ] 9. CHANGELOG `2.0.0` entry (BREAKING: removed `cli`, `analytics`, `mcp`)

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
