# Phase 4 — Examples Audit (implement)

> Part of the portfolio split-and-trim. See `planning/context.md` (why),
> `planning/plans/split-and-trim-for-portfolio.md` §4 (plan), `planning/status.md` (tracking).
> Date: 2026-06-18. Branch: `split-and-trim-for-portfolio`.

## Goal

Reduce `examples/` to a coherent set that compiles against the library-only SDK
(`core` + `runtime` + optional `sqlite`), removing examples that depend on the now-removed
`cli`/`analytics`/`mcp` features and scrubbing naming-drift / dangling references.

## What I did

### Removed examples
- **`examples/cli_interactive.rs`** — gated entirely behind `#[cfg(feature = "cli")]`
  (feature removed in Phase 2). Only the `#[cfg(not(feature = "cli"))]` stub would survive.
- **`examples/cli_analytics.rs`** — gated entirely behind `#[cfg(feature = "analytics")]`
  (the phantom feature; removed in Phase 2/3).
- **`examples/session_persistence.rs`** *(deviation from plan keep-list — see below)* — was a
  17-line non-functional stub that only printed "cannot be run as a public example" and
  `std::process::exit(1)`, referencing the old workspace crate name `claude_sdk_rs_core`.
  It demonstrated nothing and was itself a naming-drift / AI-slop tell.

### Removed documentation
- **`examples/REAL_WORLD_EXAMPLES.md`** *(deviation — see below)* — a 5,211-line aspirational
  doc of 15 "production-ready" scenarios. It was untested code-in-markdown depending on crates
  not in `Cargo.toml` (`walkdir`, `syn`, `quote`, `clap`), and §6 ("CLI Tool with Natural
  Language Interface") builds a full clap CLI — directly contradicting the new library-only
  story. Exactly the bloat/AI-slop the trim targets.

### Verified and kept (use only public SDK API)
- **`05_complete_app.rs`** — uses `Client, Message, SessionId, StreamFormat, ToolPermission`
  + an in-example cost tracker. No cli/mcp imports. **Keep.**
- **`advanced_permissions.rs`** — uses `Client, StreamFormat` and the tool-permission API.
  No cli/mcp imports. **Keep.**
- Also kept per plan: `01_basic_sdk`, `02_sdk_sessions`, `03_streaming`, `04_tools`,
  `basic_usage`, `configuration`, `streaming`, `error_handling`, `session_management`,
  `system_prompts`, `async_await_guide`.

### Naming-drift / dangling-reference fixes (examples-local; overlaps §6)
- `01_basic_sdk.rs`, `02_sdk_sessions.rs` — removed the trailing
  "This uses ONLY the `claude-sdk-rs` crate, not `claude-interactive`." disclaimer lines
  (slop tell referencing a crate that no longer exists).
- `04_tools.rs` — rewrote the sample-output block that described the old multi-crate
  workspace (interactive CLI, `-core`/`-runtime` crates, "MCP support and macros") to describe
  the current single library crate; fixed a `claude-interactive` path and branch name in
  illustrative output.
- `run_examples.sh` — header comment "claude-ai and claude-interactive" → "the claude-sdk-rs SDK".
- `examples/README.md` — removed the bullet pointing at `LIBRARY_COMPARISON.md`, a file that
  does not exist in the repo.

## Deviations from the plan (§4) and rationale
1. **Removed `session_persistence.rs`** instead of keeping it. The plan keep-list named it as
   the sqlite example, but the file in the repo was a do-nothing stub referencing an internal
   crate name. Keeping a non-runnable "cannot be run" example in a portfolio piece is worse
   than not having it. *Note:* there is now no example demonstrating the `sqlite` feature — a
   real one could be written later if desired (out of Phase 4 scope; would be a genuine
   improvement, not a stub).
2. **Removed `examples/REAL_WORLD_EXAMPLES.md`** entirely rather than "trimming to match
   survivors." The doc could not be meaningfully mapped onto the surviving example files, and
   its size + untested code + non-existent deps + a clap CLI section make wholesale removal the
   correct trim. Reversible via git if a curated subset is wanted later.

## Verification
- `cargo build --examples` → **compiles clean** (only pre-existing dead-code warnings in
  `02_sdk_sessions.rs` and `async_await_guide.rs`; no errors).
- `grep` over `examples/` for `claude-interactive`, `crate_interactive`,
  `claude_sdk_rs_core`, `claude_sdk_rs_runtime`, `LIBRARY_COMPARISON`, `cli_interactive`,
  `cli_analytics`, `REAL_WORLD`, and `feature = "cli|mcp|analytics"` → **no matches**.
- Surviving examples (13): `01_basic_sdk`, `02_sdk_sessions`, `03_streaming`, `04_tools`,
  `05_complete_app`, `advanced_permissions`, `async_await_guide`, `basic_usage`,
  `configuration`, `error_handling`, `session_management`, `streaming`, `system_prompts`.

## Observations / recommendations (not done — out of Phase 4 scope)
- **Duplicate example coverage.** The numbered series (`01`–`05`) and the named series
  (`basic_usage`, `streaming`, `session_management`, `configuration`, `error_handling`) cover
  largely the same ground. The plan keeps both; for a tighter portfolio, consolidating to one
  series would reduce noise. Flagged for a possible follow-up.
- **Orphaned `examples/tests/`** (`mod.rs`, `example_tests.rs`, `integration_tests.rs`) are not
  wired as a cargo target (only `tests/*.rs` at crate root are). `example_tests.rs` uses only
  public API and is harmless; `integration_tests.rs` shells out to `cargo build --examples`.
  Consider wiring or removing in a later pass.
- `examples/README.md` documents only `01`–`05`, not the named examples. Left as-is; a fuller
  README pass fits Phase 5 (docs trim).

## Status
Phase 4 complete. Ready for Phase 5 (docs trim).
