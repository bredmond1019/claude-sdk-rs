# Troubleshooting

A short FAQ for the most common issues. Each entry maps to a variant of
[`claude_sdk_rs::Error`](https://docs.rs/claude-sdk-rs).

## `Error::BinaryNotFound` — "Claude Code not found in PATH"

The SDK shells out to the `claude` CLI, which must be installed and on your `PATH`.

```bash
# Verify the CLI is installed and discoverable
which claude
claude --version
```

If `claude` lives somewhere non-standard (or you use nvm), point the SDK at it with the
`CLAUDE_BINARY` environment variable:

```bash
export CLAUDE_BINARY=/custom/path/to/claude
```

See [NVM_COMPATIBILITY.md](NVM_COMPATIBILITY.md) if the binary is installed under a Node
version manager.

## `Error::NotAuthenticated` — "Run 'claude auth' to authenticate"

The CLI is installed but not logged in. Authenticate once, then retry:

```bash
claude auth
```

## `Error::Timeout(seconds)` — requests time out

Long prompts or slow models can exceed the default timeout. Raise it on the config:

```rust
use claude_sdk_rs::{Client, Config};

let client = Client::new(Config::builder().timeout_secs(120).build());
```

## `Error::ProcessError` — the CLI exited unexpectedly

Usually the underlying `claude` invocation failed. Reproduce the call directly to see the
CLI's own error output:

```bash
claude -p "your prompt"
```

Check that your model name, allowed/disallowed tools, and other flags are valid.

## JSON parsing / empty responses

If you need structured responses (cost, tokens, session id) or streaming, set a non-text
`StreamFormat` — the default `Text` format carries no metadata:

```rust
use claude_sdk_rs::{Client, Config, StreamFormat};

let client = Client::new(Config::builder().stream_format(StreamFormat::Json).build());
let full = client.query("Hello").send_full().await?; // metadata available here
```

For streaming, use `StreamFormat::StreamJson` with `.stream()`.

## `Error::SessionNotFound` — session continuity not working

Session context is preserved through the CLI's `--resume`/`--continue` flags, which require
a JSON stream format. Make sure the client is configured with `StreamFormat::Json` (or
`StreamJson`) when relying on session continuity, and reuse the same `SessionId`.

## Enabling debug logging

The SDK uses [`tracing`](https://docs.rs/tracing). Install a subscriber in your app and set
`RUST_LOG` to see what the runtime is doing:

```bash
RUST_LOG=claude_sdk_rs=debug cargo run
```

## Still stuck?

Open an issue with a minimal reproduction, your OS, Rust version, `claude --version`, and the
full `Error` (its `Display` includes a `[Cxxx]` code). See [CONTRIBUTING.md](../CONTRIBUTING.md).
