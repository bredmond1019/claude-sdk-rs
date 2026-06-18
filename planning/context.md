# Project Context

> Orientation for any new session working on this repo. Read this first.

## What this repo is

`claude-sdk-rs` is a **live, published Rust crate** (currently v1.0.1 on crates.io) that
wraps the **Claude Code CLI binary** (`claude`) in a type-safe, async API. It is *not* a
direct HTTP client for the Anthropic API — the runtime shells out to the `claude` binary
with `claude -p` and maps a typed `Config` onto CLI flags (`--output-format`, `--model`,
`--system-prompt`, `--resume`, allowed/disallowed tool lists, etc.), then parses the
structured output back into Rust types. The core mechanism lives in
`src/runtime/process.rs`.

## Why we are doing this work

The crate is being prepared as a **portfolio piece**. It was built ~1+ year ago and, while
the core is genuinely strong, it has grown into three projects fused into one crate:

1. **The SDK** (good) — `src/core` (~6.9k LOC) + `src/runtime` (~5.2k LOC). Careful,
   production-minded code: structured errors with context + telemetry, nvm-aware binary
   discovery, stdin-vs-arg dispatch, streaming, recovery, backpressure. Backed by a real
   ~888-test suite (incl. wiremock HTTP tests).
2. **A CLI + analytics platform** (bloat) — `src/cli` (~47k LOC): an interactive CLI plus a
   ~20k-line analytics subsystem (dashboards, memory/time-series/streaming "optimizers",
   profiling, cost, history). Wildly disproportionate to a CLI wrapper.
3. **A customer-support agent** (wrong project) — `src/mcp` (~15k LOC): a customer-support
   MCP workflow engine + HelpScout/Slack/Notion clients, including mock business logic
   (`src/mcp/server/customer_support/tools/process_invoice.rs` hardcodes `149.99`).

To a hiring manager/reviewer this disproportion plus several tells reads as **AI slop**:
- Mock business logic shipped in a published SDK
- AI-prose docs (e.g. "garbage collection optimization" in a GC-less language)
- Naming drift: the binary/docs say `claude-interactive` while the crate is `claude-sdk-rs`
- A documented `analytics` feature flag that does not exist in `[features]`
- Stray `*.backup` / `*.bak` / `*.disabled.backup` files
- Heavy doc surface (10 root `.md` files incl. a 33k `TROUBLESHOOTING.md`, plus `docs/` + tutorials)

## The decision

Reduce the repo to a **focused, credible, pure-library Claude SDK** = `core` + `runtime`
only. Confirmed scope:
- **Remove the MCP module entirely.**
- **Remove the CLI binary + analytics entirely** → library-only crate.
- Add a modern `README.md` and a `CLAUDE.md`; trim excessive documentation.

This is safe: `default = []`, and `core`/`runtime` have **zero** `use crate::cli` /
`use crate::mcp` references. `cli` and `mcp` are leaf, feature-gated modules — removing them
cannot break the SDK.

Target: ~93k LOC → ~12–15k LOC, one coherent story.

## Key facts a new session must respect

- **It's a published crate.** Removing the `cli`/`mcp` features is a **breaking change** →
  bump to `2.0.0`. There is an open decision (see below) on republish vs. fresh repo.
- **Keep the `sqlite` feature** — it drives `src/core/session/sqlite_storage.rs` (optional
  persistent sessions). Final features should be just `default = []` and `sqlite`.
- **Do not rewrite the core mechanism.** `src/runtime/process.rs` and
  `src/runtime/error_handling.rs` are the strongest evidence of skill — preserve them and the
  `ErrorContext`/telemetry pattern.
- Global rule: **never add `Co-Authored-By:` lines to commits.**

## Open decision (resolve before executing the trim)

1. **Bump to `2.0.0` and republish** the trimmed crate (single clean portfolio artifact), OR
2. **Do the trim in a fresh `*-portfolio` repo** and leave the published `1.x` untouched
   (preserves existing consumers).

## Where the plan lives

- `planning/plans/split-and-trim-for-portfolio.md` — the full execution plan (removal list,
  `Cargo.toml`/`lib.rs` changes, examples/docs audit, README + CLAUDE.md drafts, verification).
- `planning/status.md` — what's done and what remains.
