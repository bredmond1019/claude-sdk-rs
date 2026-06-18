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

```toml
[dependencies]
claude-sdk-rs = "2"
tokio = { version = "1", features = ["full"] }

# optional persistent sessions:
# claude-sdk-rs = { version = "2", features = ["sqlite"] }
```

Requires the `claude` CLI on `PATH` (or set the `CLAUDE_BINARY` environment variable
to its location). Install it with `npm install -g @anthropic-ai/claude-code` and
authenticate with `claude auth`.

## Quick start

```rust
use claude_sdk_rs::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), claude_sdk_rs::Error> {
    let client = Client::new(Config::default());
    let reply = client.query("Write a haiku about Rust").send().await?;
    println!("{reply}");
    Ok(())
}
```

## Full response with metadata

Use `send_full()` to get the response content alongside cost, token usage, and the
session id (requires a structured stream format):

```rust
use claude_sdk_rs::{Client, StreamFormat};

#[tokio::main]
async fn main() -> Result<(), claude_sdk_rs::Error> {
    let client = Client::builder()
        .stream_format(StreamFormat::Json)
        .build()?;

    let response = client.query("Explain ownership").send_full().await?;
    println!("{}", response.content);
    if let Some(meta) = response.metadata {
        println!("cost: ${:.6}", meta.cost_usd.unwrap_or(0.0));
        println!("session: {}", meta.session_id);
    }
    Ok(())
}
```

## Streaming

A stream yields typed [`Message`] values as Claude produces them:

```rust
use claude_sdk_rs::{Client, Message, StreamFormat};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), claude_sdk_rs::Error> {
    let client = Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .build()?;

    let mut stream = client.query("Write a short story about a robot").stream().await?;
    while let Some(message) = stream.next().await {
        match message {
            Ok(Message::Assistant { content, .. }) => print!("{content}"),
            Ok(Message::Result { .. }) => break,
            Ok(_) => {}
            Err(e) => eprintln!("stream error: {e}"),
        }
    }
    Ok(())
}
```

## Configuration

```rust
use claude_sdk_rs::{Client, Config, StreamFormat};

let config = Config::builder()
    .model("sonnet")
    .system_prompt("You are a helpful Rust assistant.")
    .timeout_secs(60)
    .stream_format(StreamFormat::Json)
    .build()?;

let client = Client::new(config);
```

## Features

- Fluent `QueryBuilder` with model / system-prompt / tool / timeout control
- Three response formats: `Text`, `Json`, and streaming `StreamJson` via `futures::Stream`
- Session continuity (`--continue` / `--resume`), with optional SQLite persistence
- Structured [`Error`] types with rich context and telemetry hooks
- nvm-friendly binary discovery

## How it works

The SDK shells out to `claude -p` and maps a typed `Config` onto CLI flags
(`--output-format`, `--model`, `--system-prompt`, `--resume`, tool lists, …),
then parses the structured output back into Rust types. The core mechanism lives
in `src/runtime/process.rs`.

## Architecture

- `core` — `Config`, `Error`, `Message`, `Session`, `StreamFormat`
- `runtime` — `Client`, `QueryBuilder`, process spawning, streaming, error handling

## Requirements

- Rust 1.70 or later
- The `claude` CLI, installed and authenticated

## Learn more

See [`docs/`](docs/) for tutorials and the [`examples/`](examples/) directory for
runnable programs. Licensed under [MIT](LICENSE).

[`Message`]: https://docs.rs/claude-sdk-rs/latest/claude_sdk_rs/enum.Message.html
[`Error`]: https://docs.rs/claude-sdk-rs/latest/claude_sdk_rs/enum.Error.html
