# Phase 6 — Naming Consistency (implementation report)

> Plan ref: `planning/plans/split-and-trim-for-portfolio.md` §6. See `planning/status.md`
> for the running log. Date: 2026-06-18.

## Goal

Plan §6 ("Naming consistency"): replace stale crate-name references
(`claude-interactive` / `crate_interactive` / the old workspace crate names
`claude_sdk_rs_core` / `claude_sdk_rs_runtime`) with the single library crate name
`claude_sdk_rs`. status.md additionally scoped this phase as the one that clears the
**83 pre-existing doctest failures** carried since Phase 1 (the failures were the
naming drift in doc examples).

## What was wrong

The naming drift came in **three forms**, only the first of which the original §6 grep
(`claude-interactive` / `crate_interactive`) would have caught:

1. **Old workspace crate names** `claude_sdk_rs_core::…` / `claude_sdk_rs_runtime::…`
   in `///` doc examples (`src/**`) and in `use` statements (`tests/**`). Dating from
   when this was a multi-crate workspace; in the single crate these names don't exist.
2. **`crate::core::…` / `crate::runtime::…` inside doc examples.** In a doctest,
   `crate` resolves to the synthetic per-example crate, **not** `claude_sdk_rs`, so
   these failed to resolve. (The *same* paths in real, non-doc code are correct and
   were left untouched.)
3. Residual **API drift in the doc examples themselves**, masked until now because the
   naming errors aborted doctest compilation first (see "Second defect class" below).

`claude-interactive` / `crate_interactive` appear only under `planning/` (historical
records) — no occurrences remained in shipping files.

## Changes

### 1. Crate-name sweep (the literal §6)

Both transforms preserve every submodule path because the crate genuinely exposes
`pub mod core` and `pub mod runtime` with all referenced submodules public
(`message`, `types`, `process`, `error_handling`, `backpressure`, `stream_config`,
`recovery`, `telemetry`):

- `claude_sdk_rs_core`  → `claude_sdk_rs::core`   (per-file `perl -pi`)
- `claude_sdk_rs_runtime` → `claude_sdk_rs::runtime`
- In **doc-comment lines only** (`^\s*///`, line-guarded so real code paths are
  untouched): `crate::core` → `claude_sdk_rs::core`, `crate::runtime` →
  `claude_sdk_rs::runtime`.

Files touched: all of `src/core/{config,error,message,types,session/session}.rs`,
`src/runtime/{client,process,stream,stream_config,backpressure}.rs`, `src/lib.rs`, and
every file under `tests/core/` and `tests/runtime/`.

### 2. Second defect class — doctest API drift (uncovered once naming compiled)

With naming fixed, doctests went 2→59 passing, exposing 26 then 12 genuine example
bugs (the doc examples had drifted from the current API). status.md framed §6 as the
phase that makes the doctests green, so these were fixed as part of it:

| Location | Drift | Fix |
|---|---|---|
| `lib.rs` (3 examples) | `Config/Client::builder().build()` returns `Result`, used as a value | `.build()?` |
| `lib.rs` error example | `Error::Timeout` is now a tuple variant `Timeout(u64)` | `Err(Error::Timeout(secs))` |
| `lib.rs` streaming example | `Message` is an enum; `msg.content`/`msg.stop_reason` no longer exist | match `Message::Assistant { content, .. }` / `Message::Result { .. }`; import `Message` |
| `runtime/client.rs` send_full | builder `.build()` result used directly | `.build()?` |
| `runtime/client.rs` (2 examples) | `.session("…".to_string())` — `session()` takes `SessionId` | `.session(SessionId::new("…"))` |
| `core/types.rs` ClaudeCliResponse / from_str | `?` used in a `()`-returning doctest main | `.unwrap()` |
| `core/types.rs` ToolPermission (2) | asserted old `"bash:ls"`; `to_cli_format()` returns `"Bash(ls)"` | assert `"Bash(ls)"` |
| `runtime/stream.rs` MessageStream | printed `stats.total_tokens` (`TokenUsage`, not `Display`) | `stats.total_tokens.total` |

## Verification

- `cargo build` / `cargo build --features sqlite` — green (only the pre-existing
  `StreamHealthMetrics` dead-code warning).
- `cargo test --doc --all-features` — **85 passed / 0 failed** (was 2 passed / 83
  failed at the start of the phase). This is the headline §6 result.
- `cargo test --all-features` — 149 unit tests pass, 4 ignored, 0 failed; 85 doctests
  pass.
- `cargo build --examples` — compiles.
- `cargo bench --no-run` — compiles.
- `cargo doc --no-deps --all-features` — builds clean (exit 0).
- `grep -rn "claude-interactive\|crate_interactive\|claude_sdk_rs_core\|claude_sdk_rs_runtime"`
  over `src tests examples docs benches` + root docs + `Cargo.toml` → **no matches**.
- No remaining `crate::core` / `crate::runtime` inside any `///` doc comment; the same
  paths in real code are intact (e.g. `src/lib.rs` re-exports, `src/runtime/process.rs`).

## Finding / follow-up (out of §6 scope)

**`tests/core/**` and `tests/runtime/**` are orphaned test targets.** They live in
subdirectories of `tests/` with no `[[test]]` entries in `Cargo.toml` and no parent
`tests/*.rs` that `mod`-declares them, so Cargo does **not** compile or run them — the
`149 passed` are all `#[cfg(test)]` unit tests under `src/`. Their naming sweep is
therefore grep-clean but **unverified by compilation**. This is a pre-existing
structural artifact from the old workspace layout (each crate carried its own `tests/`).
Wiring them up (via `[[test]]` entries or `tests/core.rs` / `tests/runtime.rs`
aggregators) is a separate decision from naming consistency and may surface further
API-drift failures, so it was intentionally left for a follow-up rather than folded
into Phase 6.

## Status

§6 complete. Remaining plan items: §7 (new `README.md`), §8 (new `CLAUDE.md`),
§9 (CHANGELOG `2.0.0` entry), then the full verification gate.
