# claude-sdk-rs 🦀

[![Crates.io](https://img.shields.io/crates/v/claude-sdk-rs.svg)](https://crates.io/crates/claude-sdk-rs)
[![Documentation](https://docs.rs/claude-sdk-rs/badge.svg)](https://docs.rs/claude-sdk-rs)
[![Downloads](https://img.shields.io/crates/d/claude-sdk-rs.svg)](https://crates.io/crates/claude-sdk-rs)
[![CI](https://github.com/bredmond1019/claude-sdk-rust/workflows/CI/badge.svg)](https://github.com/bredmond1019/claude-sdk-rust/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust Version](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

A type-safe, async-first Rust SDK that wraps the [Claude Code CLI](https://claude.ai/code) to provide a powerful programmatic API for interacting with Claude AI. Build AI-powered applications with confidence using Rust's safety guarantees.

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Usage Examples](#-usage-examples)
- [Architecture](#-architecture)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [Performance](#-performance)
- [Security](#-security)
- [Requirements](#-requirements)
- [License](#-license)

## ✨ Key Features

- **🔒 Type-Safe API** - Strongly typed requests and responses with compile-time guarantees
- **⚡ Async/Await** - Built on Tokio for efficient concurrent operations
- **🔄 Multiple Response Modes** - Simple text, full metadata, or streaming responses
- **💾 Session Management** - Persistent conversations with automatic context preservation
- **🛠️ Tool Integration** - Support for MCP (Model Context Protocol) tools and external services
- **📊 Rich Metadata** - Access token usage, costs, session IDs, and timing information
- **🎯 Comprehensive Error Handling** - Detailed error types with actionable messages
- **⚙️ Flexible Configuration** - Builder patterns for intuitive setup
- **🔐 Granular Permissions** - Fine-grained tool access control with `Bash(command)` and MCP support
- **🎯 Conversation Control** - Limit turns and extend system prompts dynamically
- **🛡️ Advanced Security** - Configurable validation levels and permission controls

## 📦 Installation

### Prerequisites

1. **Install Claude Code CLI** (required for SDK operation):
   ```bash
   # Install via npm (recommended)
   npm install -g @anthropic-ai/claude-code
   
   # Or install via Homebrew on macOS
   brew install claude-code
   
   # Authenticate with your Claude account
   claude auth
   ```

2. **Verify Installation**:
   ```bash
   claude --version  # Should show the CLI version
   claude auth status  # Should show authentication status
   ```

### Add to Your Project

Add `claude-sdk-rs` to your `Cargo.toml`:

```toml
[dependencies]
claude-sdk-rs = "1.0"
tokio = { version = "1.40", features = ["full"] }
```

### Feature Flags

The SDK uses feature flags to provide only the functionality you need:

```toml
# Core SDK only (default) - minimal dependencies
claude-sdk-rs = "1.0"

# With CLI binary - adds command-line interface
claude-sdk-rs = { version = "1.0", features = ["cli"] }

# With analytics - usage metrics and performance tracking
claude-sdk-rs = { version = "1.0", features = ["analytics"] }

# With MCP support - Model Context Protocol for tools
claude-sdk-rs = { version = "1.0", features = ["mcp"] }

# With SQLite storage - persistent session management
claude-sdk-rs = { version = "1.0", features = ["sqlite"] }

# Everything enabled - all features
claude-sdk-rs = { version = "1.0", features = ["full"] }
```

### Install CLI Binary

To install the `claude-sdk-rs` CLI tool globally:

```bash
cargo install claude-sdk-rs --features cli
```

## 🚀 Quick Start

### Basic Usage

```rust
use claude_sdk_rs::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), claude_sdk_rs::Error> {
    // Create a client with default configuration
    let client = Client::new(Config::default());
    
    // Send a query and get the response
    let response = client
        .query("Explain Rust ownership in simple terms")
        .send()
        .await?;
    
    println!("Claude says: {}", response);
    Ok(())
}
```

## 📚 Documentation

- **[📦 Rust Crate](https://crates.io/crates/claude-sdk-rs)** - View on crates.io
- **[📖 Rust Docs](https://docs.rs/claude-sdk-rs)** - Complete API reference
- **[🚀 Quick Start Guide](QUICK_START.md)** - Get up and running in minutes
- **[🔧 Development Setup](DEV_SETUP.md)** - Build from source and contribute
- **[📂 Examples](examples/)** - Working code examples
- **[🏗️ Architecture](CLAUDE.md)** - Technical design and internals
- **[🔄 NVM Compatibility](docs/NVM_COMPATIBILITY.md)** - Using with Node Version Manager

## 💡 Usage Examples

### API Patterns: Two Ways to Send Queries

The SDK provides two patterns for sending queries to Claude:

**1. Builder Pattern (Recommended for new code)**
```rust
let response = client.query("Your message").send().await?;
let full_response = client.query("Your message").send_full().await?;
let stream = client.query("Your message").stream().await?;
```

**2. Direct Methods (Backward compatible)**
```rust
let response = client.send("Your message").await?;
let full_response = client.send_full("Your message").await?;
```

**When to use each pattern:**
- Use **Builder Pattern** for new applications - it's more flexible and supports additional configuration per query
- Use **Direct Methods** for simple use cases or when migrating existing code
- The builder pattern allows per-query customization (session IDs, output formats, etc.)

### Custom Configuration

```rust
use claude_sdk_rs::{Client, Config, StreamFormat};

let client = Client::builder()
    .model("claude-3-sonnet-20240229")
    .system_prompt("You are a helpful Rust programming assistant")
    .timeout_secs(60)
    .stream_format(StreamFormat::Json)
    .build();
```

### Enhanced Configuration Options

The SDK now supports advanced configuration features for better control:

```rust
use claude_sdk_rs::{Client, ToolPermission, SecurityLevel};

// Configure with system prompt extension and tool permissions
let client = Client::builder()
    .append_system_prompt("Additionally, be concise in your responses.")
    .max_turns(5)  // Limit conversation turns
    .disallowed_tools(vec![
        "Bash(rm)".to_string(),          // Block specific bash commands
        "mcp__dangerous__delete".to_string(), // Block MCP tools
    ])
    .skip_permissions(false)  // Require permission prompts
    .security_level(SecurityLevel::Balanced)  // Configure input validation
    .build();

// Tool-specific permissions
let tools_client = Client::builder()
    .allowed_tools(vec![
        ToolPermission::bash("git").to_cli_format(),
        ToolPermission::mcp("filesystem", "read").to_cli_format(),
        ToolPermission::All.to_cli_format(), // Allow all tools
    ])
    .build();
```

### Security Validation Levels

The SDK provides configurable security validation to balance usability with protection:

```rust
use claude_sdk_rs::{Client, SecurityLevel};

// Strict mode - blocks most special characters (high security)
let strict_client = Client::builder()
    .security_level(SecurityLevel::Strict)
    .build();

// Balanced mode - context-aware validation (default, recommended)
let balanced_client = Client::builder()
    .security_level(SecurityLevel::Balanced)  // Allows "create file.md"
    .build();

// Relaxed mode - only blocks obvious attacks (for trusted environments)
let relaxed_client = Client::builder()
    .security_level(SecurityLevel::Relaxed)
    .build();

// Disabled - no input validation (use with extreme caution)
let unsafe_client = Client::builder()
    .security_level(SecurityLevel::Disabled)
    .build();
```

**Security Level Guide:**
- **`Strict`**: Blocks most special characters, safest for untrusted input
- **`Balanced`**: Smart context-aware validation, allows legitimate queries like "create project-design-doc.md"
- **`Relaxed`**: Only blocks obvious attack patterns, good for controlled environments
- **`Disabled`**: No validation, only use in completely trusted scenarios

### Get Full Response Metadata

```rust
// Get response with metadata
let response = client
    .query("Write a haiku about Rust")
    .send_full()
    .await?;

println!("Response: {}", response.content);
if let Some(metadata) = response.metadata {
    println!("Cost: ${:.6}", metadata.cost_usd.unwrap_or(0.0));
    println!("Tokens: {:?}", metadata.tokens_used);
    println!("Session: {}", metadata.session_id);
}
```

### Streaming Responses

```rust
use futures::StreamExt;

let mut stream = client
    .query("Write a short story about a robot")
    .stream()
    .await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(message) => {
            if let Some(content) = message.content {
                print!("{}", content);
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### Session Management

```rust
// Sessions are automatically managed - context is preserved
let client = Client::builder()
    .stream_format(StreamFormat::Json)
    .build();

// First message
let response1 = client
    .query("My name is Alice and I love Rust programming")
    .send_full()
    .await?;

// Claude remembers the context
let response2 = client
    .query("What's my favorite programming language?")
    .send()
    .await?;
// Response: "Based on our conversation, your favorite programming language is Rust!"
```

### Error Handling

```rust
use claude_sdk_rs::Error;

match client.query("Hello").send().await {
    Ok(response) => println!("Success: {}", response),
    Err(Error::BinaryNotFound) => {
        eprintln!("Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code");
    }
    Err(Error::ProcessError(msg)) => {
        eprintln!("Process error: {}", msg);
    }
    Err(Error::Timeout) => {
        eprintln!("Request timed out");
    }
    Err(e) => eprintln!("Other error: {}", e),
}
```

## 🏗️ Architecture

The SDK is built as a single crate with modular organization and feature flags:

```
claude-sdk-rs/
├── src/
│   ├── lib.rs          # Main SDK public API
│   ├── core/           # Core types, config, errors
│   ├── runtime/        # Process execution and streaming
│   ├── mcp/            # Model Context Protocol (feature: mcp)
│   └── cli/            # CLI interface (feature: cli)
├── examples/           # Working examples
├── tests/              # Integration tests
└── benches/            # Performance benchmarks
```

### Feature Flags

- **Default**: Core SDK functionality with no extra dependencies
- **`cli`**: Adds command-line interface and interactive features
- **`mcp`**: Enables Model Context Protocol for tool integration
- **`sqlite`**: Adds SQLite-based session persistence
- **`analytics`**: Enables usage analytics (requires `cli`)
- **`full`**: Enables all features

## 🧪 Examples

Explore the [`examples/`](examples/) directory for complete working examples:

- [`basic_usage.rs`](examples/basic_usage.rs) - Simple queries and configuration
- [`streaming.rs`](examples/streaming.rs) - Real-time streaming responses
- [`session_management.rs`](examples/session_management.rs) - Multi-turn conversations
- [`error_handling.rs`](examples/error_handling.rs) - Comprehensive error handling
- [`configuration.rs`](examples/configuration.rs) - Advanced configuration options
- [`system_prompts.rs`](examples/system_prompts.rs) - System prompt extension and conversation control **NEW**
- [`advanced_permissions.rs`](examples/advanced_permissions.rs) - Granular tool permissions and security **NEW**
- [`session_persistence.rs`](examples/session_persistence.rs) - SQLite-based session storage (requires `sqlite` feature)
- [`cli_interactive.rs`](examples/cli_interactive.rs) - Interactive CLI usage (requires `cli` feature)

Run examples:
```bash
# Basic examples
cargo run --example basic_usage
cargo run --example streaming

# New features - advanced configuration
cargo run --example system_prompts
cargo run --example advanced_permissions

# Examples requiring features
cargo run --example session_persistence --features sqlite
cargo run --example cli_interactive --features cli
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development

```bash
# Clone the repository
git clone https://github.com/bredmond1019/claude-sdk-rust.git
cd claude-sdk-rust

# Build the crate
cargo build

# Build with all features
cargo build --all-features

# Run tests
cargo test

# Run tests with all features
cargo test --all-features

# Run linter
cargo clippy --all-features

# Format code
cargo fmt

# Run benchmarks
cargo bench
```

## 📈 Performance

The SDK is designed for minimal overhead:
- Zero-cost abstractions over the Claude CLI
- Efficient streaming with backpressure handling
- Connection pooling for concurrent requests
- Optimized JSON parsing with `serde`

## 🔒 Security

- **🛡️ Configurable Input Validation**: Four security levels from strict to disabled
- **🔐 Context-Aware Filtering**: Smart detection distinguishes legitimate queries from attacks
- **🚫 Command Injection Protection**: Advanced pattern detection for shell commands
- **🔒 Process Isolation**: Secure execution with proper sandboxing
- **🤐 Data Privacy**: Never logs sensitive data or API responses
- **⚙️ Granular Permissions**: Fine-grained tool access controls
- See [SECURITY.md](SECURITY.md) for complete security policy

### Security Features:
- **Balanced Validation (Default)**: Allows `"create project-design-doc.md"` while blocking `"$(rm -rf /)"`
- **Attack Pattern Detection**: Recognizes script injection, SQL injection, and command substitution
- **Legitimate Use Support**: Context-aware validation for markdown, Git commands, and file operations
- **Tool Restrictions**: Block specific bash commands or MCP tools individually

## 📋 Requirements

- **Rust**: 1.70 or later
- **Claude Code CLI**: Must be installed and authenticated
- **Operating Systems**: Linux, macOS, Windows
- **Architecture**: x86_64, ARM64

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **[GitHub Repository](https://github.com/bredmond1019/claude-sdk-rust)**
- **[Crates.io Package](https://crates.io/crates/claude-sdk-rs)**
- **[API Documentation](https://docs.rs/claude-sdk-rs)**
- **[Claude Code CLI](https://claude.ai/code)**
- **[Issue Tracker](https://github.com/bredmond1019/claude-sdk-rust/issues)**

---

<div align="center">

**[🚀 Quick Start](QUICK_START.md)** • **[🔧 Dev Setup](DEV_SETUP.md)** • **[📖 API Docs](https://docs.rs/claude-sdk-rs)** • **[💬 Discussions](https://github.com/bredmond1019/claude-sdk-rust/discussions)**

Made with ❤️ for the Rust community

</div>