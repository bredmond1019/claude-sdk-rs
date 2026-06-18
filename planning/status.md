# Project Status

> Tracks work completed and remaining for the portfolio split-and-trim. See
> `planning/context.md` for why, and `planning/plans/split-and-trim-for-portfolio.md` for how.

**Current phase:** Phase 1 complete — ready to begin Phase 2 (Cargo.toml slim-down).
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

## Blocked / needs decision
- [ ] **Republish vs. fresh repo** — bump to `2.0.0` and republish the trimmed crate, OR do
  the trim in a fresh `*-portfolio` repo leaving published `1.x` untouched. *Must resolve
  before Phase 2 begins (determines version bump target).*

## Remaining (execution)
Ordered per the plan; check off as completed.

- [x] 1. Remove subsystems: `rm -rf src/mcp`, `rm -rf src/cli`, `rm src/bin/claude-sdk-rs.rs`; delete `*.bak`/`*.disabled.backup` cruft
- [ ] 2. Slim `Cargo.toml`: drop cli/mcp-only deps, features → `default = []` + `sqlite`, remove `[[bin]]`, bump to `2.0.0`
  - **⚠️ Also delete `From<InteractiveError>` impl in `src/core/error.rs:257-304`** when removing `cli` feature
- [ ] 3. Rewrite `src/lib.rs`: drop `cli`/`mcp` gating + re-exports, fix crate doc (remove phantom `analytics` feature)
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
