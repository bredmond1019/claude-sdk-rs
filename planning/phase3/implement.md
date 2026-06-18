# Phase 3 — `src/lib.rs` Rewrite — Implementation Report

**Branch:** `split-and-trim-for-portfolio`
**Date:** 2026-06-18
**Plan reference:** `planning/plans/split-and-trim-for-portfolio.md` §3
**Scope:** Step 3 only ("`src/lib.rs` rewrite"). Examples (§4), docs (§5), naming
consistency (§6), README/CLAUDE.md/CHANGELOG (§7–9), and the final verification gate are
intentionally **out of scope** for this phase.

---

## What Phase 3 required (per plan §3)

- Remove the `#[cfg(feature = "cli")]` / `#[cfg(feature = "mcp")]` module declarations and
  re-exports (~lines 323–340 of the old file).
- Rewrite the crate-level doc comment: drop the cli/analytics/mcp/full feature sections;
  keep core + runtime + optional `sqlite`. Fix the doc that claims an `analytics` feature
  exists.
- Keep `prelude`.

## What I did

Rewrote `src/lib.rs` (346 → 261 lines; `git diff --stat`: 34 insertions, 119 deletions).

### Module declarations and re-exports
- **Removed** `#[cfg(feature = "mcp")] pub mod mcp;` and `#[cfg(feature = "cli")] pub mod cli;`.
  The crate now declares only `pub mod core;` and `pub mod runtime;`.
- **Removed** the MCP re-export block `#[cfg(feature = "mcp")] pub use crate::mcp::{McpConfig, McpServer};`.
- **Kept unchanged** the core re-exports (`ClaudeResponse, Config, Cost, Error, Message,
  MessageMeta, MessageType, ResponseMetadata, Result, Session, SessionId, SessionManager,
  StreamFormat, TokenUsage, ToolPermission`) and the runtime re-exports (`Client,
  MessageStream, QueryBuilder`).
- **Kept** the `prelude` module verbatim.
- **Kept** all six lint attributes (`deny(clippy::all)`, `warn(clippy::pedantic)`,
  `warn(missing_docs)`, and the four `allow`s).

### Crate-level doc comment
- **Removed** the four feature sections that documented removed/phantom features:
  - `cli` — Command Line Interface
  - `analytics` — Usage Analytics and Metrics (**this feature never existed in
    `[features]`** — the documented-but-nonexistent feature called out in `context.md`)
  - `mcp` — Model Context Protocol
  - `full` — All Features Enabled
- **Removed** the "CLI Module (`cli`)" and "MCP Module (`mcp`)" entries from the
  Architecture Overview, leaving only Core and Runtime.
- **Removed** the entire "## CLI Usage" section (the `cargo install --features cli`,
  `claude-sdk-rs query`, `claude-sdk-rs analytics dashboard` block).
- **Rewrote** the opening summary from "provides both programmatic access … and CLI
  interactive features … while also offering standalone CLI capabilities" to a
  pure-library description that also states the core mechanism (shells out to `claude -p`,
  maps a typed `Config` onto CLI flags, parses structured output; points at
  `src/runtime/process.rs`).
- **Kept** the optional-feature documentation, now reduced to a single `sqlite` section,
  with its `toml` snippet updated to `version = "2"`.
- **Refreshed** the bottom "## Examples" list — the old list referenced the now-deleted
  `cli_interactive.rs` / `cli_analytics.rs`. It now points at surviving examples
  (`01_basic_sdk`, `02_sdk_sessions`, `03_streaming`, `04_tools`, `configuration`,
  `error_handling`, `session_persistence`).
- **Minor wording:** renamed the "Memory Management" best-practices subheading to
  "Resource Management" (the original copy was generic and the GC-flavored framing was part
  of the AI-prose tells; the bullet content is unchanged and accurate for Rust).
- All remaining doc examples (`Basic SDK Usage`, `Advanced Configuration`, `Streaming
  Responses`, `Session Management`, `Error Handling`) are **unchanged** and use the public
  `claude_sdk_rs::` path, so they compile.

## Verification (this phase)

| Check | Result |
|-------|--------|
| `cargo build` | ✅ clean (1 pre-existing warning: dead field `avg_connection_duration` in `runtime/recovery.rs`) |
| `cargo build --features sqlite` | ✅ clean (2 pre-existing warnings: the above + missing-docs on `core/session/sqlite_storage`) |
| `cargo test --doc` | 2 passed / 83 failed — **unchanged from Phase 2**; see below |

### On the 83 doctest failures
These are **not introduced by this phase** and are the pre-existing naming drift already
logged in Phases 1–2. Confirmed by inspection: every failure is an unresolved import of a
stale **workspace** crate name — `claude_sdk_rs_core` and `claude_sdk_rs_runtime` — inside
doc examples in `src/core/**` and `src/runtime/**` (e.g. `src/core/config.rs:274` does
`use claude_sdk_rs_core::{Config, StreamFormat};`). These date from when the crate was a
multi-crate workspace. They are owned by **§6 (naming consistency)** and the final
verification gate, not by §3. `src/lib.rs`'s own doc examples are part of the 2 that pass.

The full verification gate (`clippy --all-targets --all-features -D warnings`, `cargo test
--all-features`, `--examples`, `bench --no-run`, `cargo package`, the stale-ref grep,
`cargo doc`) is run after execution per the plan and is **not** part of this phase.

## Out of scope / deferred (unchanged by this phase)

- §4 examples audit — `cli_interactive.rs` / `cli_analytics.rs` still present on disk.
- §5 docs trim, §6 naming consistency (incl. the 83 doctests), §7–9 README/CLAUDE.md/CHANGELOG.
- ~247 pre-existing clippy findings across untouched core/runtime — final-gate item.

## Files changed

- `src/lib.rs` — rewritten (34 insertions / 119 deletions).
- `planning/status.md` — Phase 3 checked off; phase header advanced to "ready to begin
  Phase 4"; log entry appended.
- `planning/phase3/implement.md` — this report.
