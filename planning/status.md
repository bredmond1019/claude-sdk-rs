# Project Status

> Tracks work completed and remaining for the portfolio split-and-trim. See
> `planning/context.md` for why, and `planning/plans/split-and-trim-for-portfolio.md` for how.

**Current phase:** Planning complete — awaiting go-ahead to execute.
**Branch:** `split-and-trim-for-portfolio`

## Completed
- [x] Repo assessment / portfolio + AI-slop audit (core strong; `cli` + `mcp` are bloat/wrong-project)
- [x] Confirmed scope with owner: remove MCP entirely, remove CLI + analytics, pure library SDK
- [x] Wrote execution plan → `planning/plans/split-and-trim-for-portfolio.md`
- [x] Wrote `planning/context.md`
- [x] Wrote `planning/status.md`

## Blocked / needs decision
- [ ] **Republish vs. fresh repo** — bump to `2.0.0` and republish the trimmed crate, OR do
  the trim in a fresh `*-portfolio` repo leaving published `1.x` untouched. *Must resolve
  before deletions begin (determines where work happens).*

## Remaining (execution — not started)
Ordered per the plan; check off as completed.

- [ ] 1. Remove subsystems: `rm -rf src/mcp`, `rm -rf src/cli`, `rm src/bin/claude-sdk-rs.rs`; delete `*.bak`/`*.disabled.backup` cruft
- [ ] 2. Slim `Cargo.toml`: drop cli/mcp-only deps, features → `default = []` + `sqlite`, remove `[[bin]]`, bump to `2.0.0`
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
- _(append dated entries as work proceeds)_
