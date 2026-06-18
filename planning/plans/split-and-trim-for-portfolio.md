# Split & Trim `claude-sdk-rs` for Portfolio

## Context

`claude-sdk-rs` is a live, published crate (v1.0.1) built ~1+ year ago. Its **core is strong** — `src/runtime/process.rs` and `src/runtime/error_handling.rs` are careful, production-minded code wrapping the `claude` CLI binary, backed by a real ~888-test suite. But the crate suffers an **identity crisis**: ~93k LOC across 197 files, of which a 47k-line CLI/analytics platform (`src/cli`) and a 15k-line customer-support agent + HelpScout/Slack/Notion clients (`src/mcp`) are effectively two unrelated projects bolted onto the SDK. A reviewer reads this disproportion — plus tells like mock business logic (`src/mcp/server/customer_support/tools/process_invoice.rs:80`, hardcoded `149.99`), AI-prose docs ("garbage collection optimization" in a GC-less language), naming drift (`claude-interactive` vs `claude-sdk-rs`), a documented-but-nonexistent `analytics` feature, and stray `*.backup`/`*.bak` files — as AI slop.

**Goal:** reduce the repo to a focused, credible **pure-library Claude SDK** (core + runtime only), with a modern README and a `CLAUDE.md`, removing all excess code and documentation.

**Decisions (confirmed):**
- Remove the **MCP module entirely**.
- Remove the **CLI binary + analytics** entirely → pure library SDK.

**Why the split is low-risk:** `default = []`; `core`/`runtime` have **zero** `use crate::cli` / `use crate::mcp` references. `cli` and `mcp` are leaf, feature-gated modules. Removing them cannot break the SDK.

## Target End State

A library crate ≈ 12–15k LOC of source + tests:
- `src/core/**` — Config, types, error, message, session (incl. optional SQLite persistence)
- `src/runtime/**` — Client, QueryBuilder, process spawning, streaming, error handling, telemetry, recovery, backpressure
- `src/lib.rs` — rewritten, no `cli`/`mcp` gating
- Lean `Cargo.toml`, modern `README.md`, new `CLAUDE.md`, trimmed `docs/`

---

## Execution Plan

### 1. Remove entire subsystems
- `rm -rf src/mcp/` (~15k LOC: customer_support, knowledge_base, helpscout/slack/notion clients & services)
- `rm -rf src/cli/` (~47k LOC: analytics, cost, history, profiling, execution, output, session, commands)
- `rm src/bin/claude-sdk-rs.rs` and remove the `[[bin]]` target → crate becomes library-only
- Delete cruft: `src/core/config_test.rs.bak`, and all 4 `*.disabled.backup` files under `src/mcp/`

### 2. `Cargo.toml` slim-down
- **Drop optional deps** only used by cli/mcp: `clap`, `clap_complete`, `directories`, `dirs`, `colored`, `anyhow`, `tracing-subscriber`, `notify`, `prettytable-rs`, `tokio-util`, `futures-util`, `tokio-tungstenite`, `base64`, `prometheus`, `num_cpus`, `libc`. Move `tempfile` to `dev-dependencies` if a kept test/example needs it. Verify each is unused in core/runtime via `cargo build` before removing.
- **Keep:** tokio, serde, serde_json, thiserror, futures, async-trait, bytes, uuid, chrono, tracing, reqwest, url, which, parking_lot, rand, once_cell.
- **Features:** reduce to `default = []` and `sqlite = ["dep:sqlx"]` (drives `core/session/sqlite_storage.rs`). Remove `cli`, `analytics`, `mcp`, `full`.
- Remove the `[[bin]]` section; keep `[[bench]]` (bench is pure-SDK: `Client::builder`).
- Bump to `2.0.0` — this is a breaking removal for crates.io consumers.

### 3. `src/lib.rs` rewrite
- Remove the `#[cfg(feature = "cli")]` / `#[cfg(feature = "mcp")]` modules and re-exports (~lines 323–340).
- Rewrite the crate-level doc comment: drop the cli/analytics/mcp/full feature sections; keep core + runtime + optional `sqlite`. Fix the doc that claims an `analytics` feature exists.
- Keep `prelude`.

### 4. Examples audit (`examples/`)
- **Remove:** `cli_interactive.rs`, `cli_analytics.rs`.
- **Verify then keep/fix:** `05_complete_app.rs`, `advanced_permissions.rs` (drop if they pull cli/mcp).
- **Keep:** `01_basic_sdk`, `02_sdk_sessions`, `03_streaming`, `04_tools`, `basic_usage`, `configuration`, `streaming`, `error_handling`, `session_management`, `session_persistence`, `system_prompts`, `async_await_guide`.
- Trim `examples/README.md` and `examples/REAL_WORLD_EXAMPLES.md` to match survivors; update `run_examples.sh`.

### 5. Documentation trim (remove "excessive")
**Root `.md` — keep:** `README.md` (rewrite), `LICENSE`, `CHANGELOG.md`, `SECURITY.md`, `CONTRIBUTING.md` (trim to lib-only).
**Root `.md` — remove or fold into `docs/`:** `GETTING_STARTED.md` (21k), `QUICK_START.md` (8k), `MIGRATION.md` (12k), `RELEASE_NOTES.md`, `ROADMAP.md`, `TROUBLESHOOTING.md` (33k → trim to a short FAQ in `docs/`).
**`docs/` — keep lean:** `NVM_COMPATIBILITY.md` (genuinely useful, SDK-relevant), a trimmed `TESTING.md`, a trimmed `SECURITY.md`.
**`docs/` — remove:** `FEATURE_FLAGS.md` (features now trivial), `PERFORMANCE.md` (15k, analytics-oriented), the duplicate `docs/MIGRATION.md` / `docs/TROUBLESHOOTING.md`.
**`docs/tutorials/` — remove** cli/mcp-specific: `08-code-review-bot`, `09-documentation-generator`, `10-automated-test-writer`, and `05-tool-integration` if cli-bound. **Keep:** `00`–`04`, `06-session-management`, `07-advanced-usage` (verify SDK-only).
**`scripts/` — remove** `check_tutorial_syntax.py`, `test_tutorial_code.py` if their tutorials are gone; keep release/version scripts.

### 6. Naming consistency
- Grep for `claude-interactive` / `crate_interactive` across remaining files and fix to `claude-sdk-rs` (stale doc example paths in module headers).

### 7. New `README.md` (modern, SDK-focused) — draft

```markdown
# claude-sdk-rs

A type-safe, async Rust SDK for [Claude Code](https://docs.claude.com/claude-code).
It wraps the `claude` CLI binary in a fluent, strongly-typed API — giving you
structured responses, streaming, session continuity, and tool control from Rust.

[![crates.io](https://img.shields.io/crates/v/claude-sdk-rs.svg)](https://crates.io/crates/claude-sdk-rs)
[![docs.rs](https://docs.rs/claude-sdk-rs/badge.svg)](https://docs.rs/claude-sdk-rs)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Why

The Claude Code CLI is powerful but stringly-typed. `claude-sdk-rs` turns it into
a real library: typed config, three response formats (text / JSON / streaming JSON),
session resume, allowed/disallowed tool lists, and structured errors with telemetry —
all async on tokio.

## Install
\`\`\`toml
[dependencies]
claude-sdk-rs = "2"
# optional persistent sessions:
# claude-sdk-rs = { version = "2", features = ["sqlite"] }
\`\`\`
Requires the `claude` CLI on PATH (or set `CLAUDE_BINARY`).

## Quick start
\`\`\`rust
use claude_sdk_rs::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), claude_sdk_rs::Error> {
    let client = Client::new(Config::default());
    let reply = client.query("Write a haiku about Rust").send().await?;
    println!("{reply}");
    Ok(())
}
\`\`\`

## Streaming
\`\`\`rust
use claude_sdk_rs::{Client, Config, StreamFormat};
use futures::StreamExt;

let client = Client::new(Config::builder().stream_format(StreamFormat::StreamJson).build());
let mut stream = client.query("Explain ownership").stream().await?;
while let Some(chunk) = stream.next().await {
    print!("{}", chunk?);
}
\`\`\`

## Features
- Fluent `QueryBuilder` with model / system-prompt / tool / timeout control
- Streaming responses (`StreamFormat::StreamJson`) via `futures::Stream`
- Session continuity (`--continue` / `--resume`), optional SQLite persistence
- Structured `Error` types with rich context and telemetry hooks
- nvm-friendly binary discovery

## How it works
The SDK shells out to `claude -p` and maps a typed `Config` onto CLI flags
(`--output-format`, `--model`, `--system-prompt`, `--resume`, tool lists, …),
then parses the structured output back into Rust types. See `src/runtime/process.rs`.

## Architecture
- `core` — `Config`, `Error`, `Message`, `Session`, `StreamFormat`
- `runtime` — `Client`, `QueryBuilder`, process spawning, streaming, error handling

See [`docs/`](docs/) and [`examples/`](examples/). MIT licensed.
```

### 8. New `CLAUDE.md` (repo guidance for Claude Code) — draft

```markdown
# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is
`claude-sdk-rs` — a pure-library Rust SDK that wraps the `claude` CLI binary in a
typed async API. Library only; no CLI binary, no MCP, no analytics.

## Layout
- `src/core/` — Config, types, errors, messages, sessions (+ optional SQLite)
- `src/runtime/` — Client, QueryBuilder, process spawning (`process.rs`), streaming,
  error handling, telemetry, recovery, backpressure
- `examples/` — runnable SDK examples   • `benches/` — criterion benchmarks
- `tests/` — integration tests (core / runtime)

## How it works
`runtime::process` spawns the `claude` binary with `-p` and maps `Config` to CLI
flags (`--output-format`, `--model`, `--system-prompt`, `--resume`, tool lists…).
Respect the `CLAUDE_BINARY` override and the stdin-vs-arg dispatch in `process.rs`.

## Conventions
- Async on tokio; errors via `thiserror` `Error` with `ErrorContext`.
- New runtime errors must thread through `error_handling.rs` (context + telemetry).
- Keep `default = []`; gate persistence behind the `sqlite` feature.
- No new top-level subsystems — this crate is intentionally small and focused.

## Workflow
- Build: `cargo build` / `cargo build --features sqlite`
- Test: `cargo test --all-features`   • Lint: `cargo clippy --all-targets -- -D warnings`
- Format: `cargo fmt`   • Bench: `cargo bench`
- Pre-commit hooks live in `.pre-commit-config.yaml`; honor them.
```

### 9. CHANGELOG / version
- Add a `2.0.0` entry: "BREAKING: removed `cli`, `analytics`, and `mcp` modules; crate is now a focused library SDK." Optionally trim CHANGELOG history of cli/mcp-only lines.

## Verification (after execution)
1. `cargo build` and `cargo build --features sqlite` — clean.
2. `cargo clippy --all-targets --all-features -- -D warnings` — clean.
3. `cargo test --all-features` — surviving suite green (most of the 888 tests live in core/runtime).
4. `cargo build --examples` — all kept examples compile.
5. `cargo bench --no-run` — bench compiles.
6. `cargo package --allow-dirty` (dry run) — verify publishable, no dangling paths.
7. `grep -rn "claude-interactive\|crate_interactive\|src/cli\|src/mcp"` — no stale references in kept files/docs.
8. `cargo doc --no-deps` — doc comments build without the removed feature sections.

## Expected outcome
~93k LOC → ~12–15k LOC of focused, coherent SDK code; one clear story; modern README + `CLAUDE.md`; AI-slop signals (mock logic, GC-in-Rust prose, naming drift, doc/feature mismatch, backup files) eliminated. Verdict moves from "portfolio worthy with cleanup" to clean **portfolio worthy**.

---

## Open decision before execution
This is a **breaking, major-version** change for anyone depending on the published `cli`/`mcp` features. Two paths:
- **Bump to `2.0.0`** and publish the trimmed crate (recommended for a single clean portfolio artifact).
- **Do the trim in a fresh `*-portfolio` repo** and leave the published `1.x` crate untouched (preserves existing consumers).
