# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`claude-sdk-rs` — a pure-library Rust SDK that wraps the `claude` CLI binary in a
typed async API. Library only; no CLI binary, no MCP, no analytics.

## Layout

- `src/core/` — `Config`, types, errors, messages, sessions (+ optional SQLite storage)
- `src/runtime/` — `Client`, `QueryBuilder`, process spawning (`process.rs`), streaming,
  error handling, telemetry, recovery, backpressure
- `src/lib.rs` — public API surface and crate-level docs
- `examples/` — runnable SDK examples
- `benches/` — criterion benchmarks
- `tests/` — integration tests (core / runtime)
- `docs/` — tutorials and reference docs
- `planning/` — the portfolio split-and-trim plan, per-phase notes, and status

## How it works

`runtime::process` spawns the `claude` binary with `-p` and maps `Config` onto CLI
flags (`--output-format`, `--model`, `--system-prompt`, `--resume`, allowed/disallowed
tool lists, …), then parses the structured output back into Rust types. Respect the
`CLAUDE_BINARY` override and the stdin-vs-arg dispatch in `process.rs`.

## Conventions

- Async on tokio; errors via `thiserror` as the `Error` enum with `ErrorContext`.
- New runtime errors should thread through `error_handling.rs` (context + telemetry) —
  this pattern is the core of the crate; preserve it.
- Keep features minimal: `default = []`, with persistence gated behind `sqlite`.
- No new top-level subsystems — this crate is intentionally small and focused.

## Workflow

- Build: `cargo build` / `cargo build --features sqlite`
- Test: `cargo test --all-features`
- Lint: `cargo clippy --all-targets --all-features -- -D warnings`
- Format: `cargo fmt`
- Bench: `cargo bench`
- Git hooks live in `.pre-commit-config.yaml` / `.githooks/`; honor them.

## Commit rule

Never add `Co-Authored-By:` lines to commit messages.
