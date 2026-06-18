# Phase 5 — Documentation Trim (implement)

> Part of the portfolio split-and-trim. See `planning/context.md` (why),
> `planning/plans/split-and-trim-for-portfolio.md` §5 (plan), `planning/status.md` (tracking).
> Date: 2026-06-18. Branch: `split-and-trim-for-portfolio`.

## Goal

Cut the oversized / wrong-project / stale documentation surface down to a lean set that
matches a single library crate, and fold the one genuinely useful oversized doc
(`TROUBLESHOOTING`) into a short FAQ. README rewrite (§7), `CLAUDE.md` (§8), and the
CHANGELOG `2.0.0` entry (§9) are intentionally **out of scope** for this phase.

## Removed

### Root `.md`
- `GETTING_STARTED.md` (851 lines), `QUICK_START.md` (349), `MIGRATION.md` (447),
  `RELEASE_NOTES.md` (132), `ROADMAP.md` (113) — superseded / not needed for a focused SDK.
- `TROUBLESHOOTING.md` (1,313 lines) — **folded** into a concise `docs/TROUBLESHOOTING.md` FAQ
  (see below).

### `docs/`
- `FEATURE_FLAGS.md` (436) — features are now trivially `default = []` + `sqlite`.
- `PERFORMANCE.md` (568) — analytics-oriented, tied to the removed subsystems.
- `MIGRATION.md` (155) — duplicate of the root migration doc.
- `TROUBLESHOOTING.md` (745) — duplicate; replaced by the new short FAQ.

### `docs/tutorials/`
- `08-code-review-bot.md`, `09-documentation-generator.md`, `10-automated-test-writer.md` —
  cli/application-oriented tutorials, not core-SDK.

### Additional removal (deviation — not explicitly in §5)
- `tests/TEST_INVENTORY.md` — a "complete test listing" that was ~70% stale: 583 of its 837
  counted tests were in the removed `interactive` crate (+73 in `mcp`), every "Top Test File"
  pointed at a deleted crate, and it contained an "New Tests Added by Agent 2" section (an
  AI-slop tell). Keeping a kept-doc that is 14× wrong undermines the cleanup, so it was removed
  and its link dropped from `docs/TESTING.md`. The trimmed TESTING.md table now lists the real
  surviving test files by area.

## Rewritten / trimmed

- **`CONTRIBUTING.md`** (567 → ~190 lines) — rewrote from a multi-crate workspace guide to a
  single-crate guide. Removed the entire multi-crate publish-chain section (`publish.sh`
  ordering across `-core`/`-mcp`/`-runtime`/`-interactive`), `cargo test --workspace` /
  `-p claude-sdk-rs-core` commands, the stale "Project Structure" crate list, and dangling
  links (`API_EVOLUTION_GUIDELINES.md`, `CODE_OF_CONDUCT.md`, `docs/ARCHITECTURE.md`,
  `docs/PERFORMANCE.md`). Kept the useful, accurate parts (build/test/lint commands, the three
  test-style snippets, conventional commits, semver, security pointer). **Corrected** the test
  snippet's error variant `Error::ClaudeNotAuthenticated` → `Error::NotAuthenticated` (the real
  variant, verified in `src/core/error.rs`).
- **`docs/TESTING.md`** — replaced the test-count table (which listed `CLI History`,
  `CLI Commands`, `Analytics & Profiling`, "Other Modules … MCP, CLI, interactive") with a
  table of the real surviving `core`/`runtime`/`tests/` files; swapped `cargo test --workspace`
  / `-p claude-sdk-rs-core` for `cargo test` / `cargo test --all-features`; fixed a
  `claude_ai::Client` doctest import → `claude_sdk_rs::Client`; removed the `PERFORMANCE.md` and
  `TEST_INVENTORY.md` links.
- **`docs/TROUBLESHOOTING.md`** — **new** ~90-line FAQ replacing the two oversized versions.
  Organized by real `Error` variant (`BinaryNotFound`, `NotAuthenticated`, `Timeout(seconds)`,
  `ProcessError`, `SessionNotFound`) plus stream-format/metadata and `RUST_LOG` debugging.
  Uses the **verified** `CLAUDE_BINARY` env override (`src/runtime/process.rs:23`) instead of
  the old doc's non-existent `Config::builder().claude_binary_path(...)` method, and links to
  `NVM_COMPATIBILITY.md`.
- **`SECURITY.md`** — updated the Supported Versions table from `1.0.x` to `2.0.x` to match the
  breaking release. Left the historical security changelog intact as a record.

## Kept (verified SDK-only)

- Root: `LICENSE`, `CHANGELOG.md`, `README.md` (rewrite deferred to §7), `SECURITY.md`.
- `docs/`: `NVM_COMPATIBILITY.md` (SDK-relevant), `SECURITY.md` (already CLI-integration /
  auth / validation focused — no cli/mcp content to strip, left as-is), `TESTING.md` (trimmed).
- `docs/tutorials/`: `00`–`07`. Verified via grep that **none** of the kept tutorials contain
  workspace crate-name drift (`claude-sdk-rs-core`/`-mcp`/`-runtime`/`-interactive`),
  `--workspace`, `src/cli`, or `src/mcp`. `05-tool-integration` was kept (plan said "remove if
  cli-bound" — it is not; it covers the SDK's `ToolPermission` API).
- `scripts/`: all kept. `check_tutorial_syntax.py` / `test_tutorial_code.py` glob
  `docs/tutorials/*.md` (no hardcoded references to the removed `08`–`10`), so they continue to
  work against the surviving tutorials; release/version scripts kept.

## Verification

- Grep across kept docs (`CONTRIBUTING.md`, `SECURITY.md`, `docs/**`) for
  `claude_ai`, `claude-sdk-rs-{core,mcp,runtime,interactive}`, `crate_interactive`,
  `claude-interactive`, `--workspace`, `src/cli`, `src/mcp` → **no matches**.
- Grep for dangling links to removed docs (`GETTING_STARTED`, `QUICK_START`, `MIGRATION`,
  `RELEASE_NOTES`, `ROADMAP`, `FEATURE_FLAGS`, `PERFORMANCE`, `08/09/10` tutorials,
  `REAL_WORLD`, `TROUBLESHOOTING`) across kept files → only **`README.md`** (deferred to §7) and
  `planning/**` still reference them, as expected.
- No source files changed → build/test status unaffected by this phase.

## Final documentation surface

```
README.md  CONTRIBUTING.md  SECURITY.md  CHANGELOG.md  LICENSE
docs/  NVM_COMPATIBILITY.md  SECURITY.md  TESTING.md  TROUBLESHOOTING.md
docs/tutorials/  00 01 02 03 04 05 06 07
```

## Known deferred items (not Phase 5)
- **`README.md`** still links to the removed `GETTING_STARTED.md` / `QUICK_START.md` /
  `MIGRATION.md`, a never-existed `DEV_SETUP.md`, and the not-yet-written `CLAUDE.md`. The full
  README rewrite in **§7** resolves all of these; left intentionally untouched here.
- `docs/SECURITY.md` is 515 lines; it is SDK-focused and was left as-is. Could be shortened for
  length later, but it carries no wrong-project content.

## Status
Phase 5 complete. Ready for Phase 6 (naming consistency) — note the doctest naming-drift
fixes (`claude_sdk_rs_core` / `claude_sdk_rs_runtime` in `src/**` doc examples) and the README
rewrite still remain.
