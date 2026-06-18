# Contributing to claude-sdk-rs

Thank you for your interest in contributing! `claude-sdk-rs` is a single, focused
library crate that wraps the `claude` CLI in a typed async API. This guide covers
how to build, test, and submit changes.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/claude-sdk-rs.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run the checks below
6. Commit, push, and open a pull request

## Development Setup

### Prerequisites

- Rust 1.70 or later
- The [Claude Code CLI](https://docs.claude.com/claude-code) installed and authenticated
  (required for integration tests; unit tests skip when it is absent)

### Build

```bash
cargo build                  # default (library only)
cargo build --features sqlite # with the optional SQLite session store
```

### Git hooks

```bash
./scripts/setup-git-hooks.sh
```

## Testing

```bash
# Run all tests
cargo test

# Run with all features (includes the sqlite session store)
cargo test --all-features

# Run a single test
cargo test test_name

# Run with output
cargo test -- --nocapture
```

When contributing features or fixes, please include appropriate tests.

**Unit test:**
```rust
#[test]
fn test_config_builder_with_timeout() {
    let config = Config::builder()
        .timeout_secs(120)
        .build();

    assert_eq!(config.timeout_secs, Some(120));
}
```

**Async test** (skips gracefully when the CLI is unavailable):
```rust
#[tokio::test]
async fn test_client_query() {
    let client = Client::new(Config::default());

    if which::which("claude").is_err() {
        return;
    }

    let result = client.query("test").send().await;
    assert!(result.is_ok() || matches!(result, Err(Error::NotAuthenticated)));
}
```

**Property test:**
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_session_id_roundtrip(s in "[a-zA-Z0-9-]{1,50}") {
        let id = SessionId::new(&s);
        assert_eq!(id.as_str(), s);
    }
}
```

See [docs/TESTING.md](docs/TESTING.md) for comprehensive testing guidelines.

## Code Style

All code must be formatted and lint-clean before committing:

```bash
cargo fmt --all          # format (required)
cargo fmt --check        # verify formatting
cargo clippy --all-features
cargo audit              # check for vulnerable dependencies
```

The pre-commit hook checks formatting and will reject unformatted commits.

Guidelines:
- Follow Rust standard style (enforced by `rustfmt`)
- Document public APIs
- Keep functions focused; use meaningful names
- Add tests for new functionality

## Project Structure

This is a single library crate:

- `src/core/` — `Config`, types, errors, messages, sessions (+ optional SQLite)
- `src/runtime/` — `Client`, query builder, process spawning, streaming, error handling,
  telemetry, recovery, backpressure
- `examples/` — runnable SDK examples
- `benches/` — criterion benchmarks
- `tests/` — integration tests (`core` / `runtime`)
- `docs/` — supplementary guides

## Pull Request Process

1. **Format**: `cargo fmt --all` (and confirm `cargo fmt --check` passes)
2. **Test**: `cargo test --all-features`
3. **Lint**: `cargo clippy --all-features`
4. **Audit**: `cargo audit`
5. Update documentation as needed
6. Add a CHANGELOG.md entry if user-facing
7. Write clear commit messages and link related issues

PRs require at least one approving review and passing CI. Breaking changes require a
major version bump.

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

Example:
```
feat(core): add timeout configuration option

Add support for custom timeout values in the Config builder.

Closes #123
```

## Versioning

We follow [semantic versioning](https://semver.org/): patch for fixes, minor for
backward-compatible features, major for breaking changes.

## Security

Do not report vulnerabilities through public issues. See [SECURITY.md](SECURITY.md)
for the disclosure process. When contributing, be mindful of input validation, avoid
hardcoded secrets, and consider the security implications of subprocess handling.

## Questions?

Open an issue or start a GitHub Discussion. Thank you for contributing! 🚀
