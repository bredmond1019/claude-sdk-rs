# Part 1: Getting Started with claude-sdk-rs

Welcome to the claude-sdk-rs Rust SDK! This tutorial will guide you through setting up and making your first API calls to Claude AI using our type-safe, async-first Rust library.

## What is claude-sdk-rs?

The claude-sdk-rs SDK transforms the Claude Code CLI tool into a powerful programmatic API for Rust applications. It provides:

- **Type-safe API**: All responses and configurations are strongly typed
- **Async-first**: Built on Tokio for high-performance async operations  
- **Three response modes**: Simple text, full metadata, and streaming responses
- **Session management**: Persistent conversations with cost and token tracking
- **Tool integration**: Support for MCP servers and external tools
- **Feature flags**: Modular design to include only what you need

## Prerequisites

Before you begin, ensure you have:

1. **Rust 1.70+** installed
2. **Claude Code CLI** installed and authenticated
   ```bash
   # Install Claude Code CLI (if not already installed)
   curl -sSL https://claude.ai/install.sh | sh
   
   # Authenticate with your API key
   claude auth login
   ```

## Installation

Add claude-sdk-rs to your `Cargo.toml`:

```toml
[dependencies]
claude-sdk-rs = "0.1"
tokio = { version = "1.40", features = ["full"] }
```

Or with specific features:
```toml
# With SQLite persistence
claude-sdk-rs = { version = "0.1", features = ["sqlite"] }

# With MCP support
claude-sdk-rs = { version = "0.1", features = ["mcp"] }

# With CLI binary
claude-sdk-rs = { version = "0.1", features = ["cli"] }

# All features
claude-sdk-rs = { version = "0.1", features = ["full"] }
```

## Your First Query

Let's start with the simplest possible example:

```rust
use claude_sdk_rs::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the client with default configuration
    let client = Client::new(Config::default());

    // Send a simple query
    let response = client.query("What is 2 + 2?").send().await?;

    println!("Response: {}", response);

    Ok(())
}
```

Run this example:
```bash
cargo run
```

You should see Claude's response to your mathematical question!

## Understanding the Basic Structure

Let's break down what's happening:

### 1. Client Creation
```rust
let client = Client::new(Config::default());
```
This creates a new client with default settings. The client handles all communication with the Claude CLI.

### 2. Query Building
```rust
let response = client.query("What is 2 + 2?").send().await?;
```
- `client.query()` creates a query builder
- `.send()` executes the query and returns a simple string response
- `.await?` handles the async operation and error propagation

### 3. Error Handling
The SDK provides comprehensive error handling through the `Error` enum. Common errors include:
- `BinaryNotFound`: Claude CLI not installed
- `ProcessError`: CLI execution failures
- `Timeout`: Operation timeouts (default 30s)
- `SerializationError`: JSON parsing errors
- `ConfigError`: Configuration issues
- `Io`: I/O related errors

## Builder Pattern Alternative

You can also use the builder pattern for more control:

```rust
use claude_sdk_rs::{Client, Config, StreamFormat};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .stream_format(StreamFormat::Json)
        .timeout_secs(60) // 60 second timeout
        .build();
    
    let client = Client::new(config);

    let response = client.query("Hello, Claude!").send().await?;
    println!("Response: {}", response);

    Ok(())
}
```

## Next Steps

Now that you have a basic understanding, let's explore:

- **Part 2**: Learn about different response modes and query options
- **Part 3**: Dive into configuration options and customization
- **Part 4**: Explore streaming responses for real-time output

## Common Issues

**"Binary not found" error**: Ensure Claude CLI is installed and in your PATH
```bash
which claude  # Should show the path to Claude CLI
```

**Authentication errors**: Make sure you're logged in
```bash
claude auth status
```

**Timeout errors**: Increase timeout for complex queries
```rust
let config = Config::builder().timeout_secs(120).build(); // 2 minutes
let client = Client::new(config);
```

Ready to continue? Head to [Part 2: Basic Usage](02-basic-usage.md) to learn about the different ways to interact with Claude!