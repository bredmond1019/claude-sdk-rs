# Phase 2 Review Report

**Verdict: PASS**
**Reviewer date:** 2026-06-18
**Artifact reviewed:** `planning/phase2/implement.md` + live file state

---

## Scope confirmed

Phase 2 was scoped to: slim `Cargo.toml`, bump to `2.0.0`, delete the deferred
`From<InteractiveError>` impl in `src/core/error.rs`, and carry forward a sqlite
connection-string bug fix that the test gate required. All four items are done.

---

## Verified claims (checked against live files)

| Claim | Verified |
|---|---|
| Version bumped to `2.0.0` | ✅ `Cargo.toml` line 3 |
| `description` and `categories` reflect library-only crate | ✅ "api-bindings", "asynchronous" |
| All cli/mcp/analytics deps removed | ✅ `[dependencies]` lists only core runtime deps |
| `reqwest`, `url`, `bytes` removed | ✅ Absent from `Cargo.toml` |
| `num_cpus` + `libc` kept (plan wrongly listed for removal) | ✅ Present; correct |
| `[features]` reduced to `default = []` + `sqlite` | ✅ Confirmed |
| `[[bin]]` removed; `[[bench]]` kept | ✅ Only `[[bench]]` present |
| `tempfile` moved to `[dev-dependencies]` | ✅ Present only in dev-deps |
| Dead dev-deps (`insta`, `wiremock`, `dotenvy`, `tokio-test`) removed | ✅ Absent |
| `[package.metadata.docs.rs]` features updated to `sqlite` | ✅ Confirmed |
| `From<InteractiveError>` impl deleted from `src/core/error.rs` | ✅ No match for `InteractiveError` in that file |
| SQLite `?mode=rwc` fix applied | ✅ `sqlite_storage.rs` line 28 |

---

## Outstanding issues (carried forward — not Phase 2 regressions)

### Known / expected

1. **`src/lib.rs` still gates on `mcp` and `cli` features** (lines 323, 326, 339). These
   features no longer exist in `Cargo.toml`, so `rustc` emits `unexpected cfg value`
   warnings on every build. Harmless at runtime; Phase 3 deletes them. No action needed now.

2. **83 doctest failures** — pre-existing naming drift (`use claude_sdk_rs_core` in doc
   examples). Count unchanged vs. Phase 1. Phase 3 fixes these.

3. **~247 pre-existing clippy findings** in untouched `core`/`runtime`. The final
   verification gate (`clippy --all-features -- -D warnings`) will not pass until these
   are addressed. This is notable work for a later phase — flag it early so it is not
   a surprise before publish.

### Judgment calls — approved

- **Dropping `reqwest`/`url`/`bytes`:** Grep confirms zero usage in `src/`, `tests/`, or
  `benches/`. Removal is correct; keeping dead deps would be misleading.
- **`num_cpus`/`libc` kept:** Plan was wrong; the impl file uses both. The agent caught
  and corrected this rather than blindly following the plan. Good.
- **`?mode=rwc` fix landing in Phase 2:** The fix is one line, is required for the test
  gate, and belongs to a feature (`sqlite`) that is shipping. Acceptable in this phase.
- **`tempfile` to dev-dependencies:** Correct. It is used only in `#[cfg(test)]` blocks.

---

## What Phase 3 must address (carried forward blockers)

- Rewrite `src/lib.rs`: remove `#[cfg(feature = "mcp")]` / `#[cfg(feature = "cli")]` gates
  and their re-exports; fix crate-doc feature sections (still reference `cli`, `analytics`,
  `mcp`, `full`; version references say `"1.0"`).
- Fix 83 doctest failures (naming drift: `claude_sdk_rs_core` → `claude_sdk_rs`).
- Plan for the ~247 clippy findings — whether to fix them or selectively `#[allow]` before
  the final gate.

---

## Summary

Phase 2 executed cleanly against its scope. Every Cargo.toml change is verified in the
live file. The `From<InteractiveError>` deletion is confirmed. The sqlite bug fix is
correct and necessary. The two plan corrections (`num_cpus`/`libc` kept; `reqwest` etc.
dropped) were sound judgment calls, not overreach. Build and test counts match the report
(149 pass / 0 fail). The remaining issues are pre-existing and correctly deferred.
