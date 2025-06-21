# 🚀 Quick Start Guide for claude-sdk-rs

Get up and running with `claude-sdk-rs` in just a few minutes! This guide will walk you through installation, setup, and your first queries to Claude AI.

## 📋 Prerequisites

### 1. Install Rust
Ensure you have Rust 1.70 or later:
```bash
# Check your Rust version
rustc --version

# If you need to install/update Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Install Claude Code CLI
The SDK requires the Claude Code CLI to be installed and authenticated:

```bash
# Install via npm (recommended)
npm install -g @anthropic-ai/claude-code

# Or install via Homebrew on macOS
brew install claude-code

# After installation, authenticate:
claude auth
```

Verify installation:
```bash
claude --version
```

## 🔧 Installation

### Create a New Project
```bash
# Create a new Rust project
cargo new my-claude-app
cd my-claude-app
```

### Add Dependencies
Edit your `Cargo.toml`:

```toml
[dependencies]
claude-sdk-rs = "1.0"
tokio = { version = "1.40", features = ["full"] }
```

## 🎯 Your First Claude Query

Replace the contents of `src/main.rs`:

```rust
use claude_sdk_rs::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), claude_sdk_rs::Error> {
    println!("🤖 Welcome to claude-sdk-rs!");
    
    // Create a client with default settings
    let client = Client::new(Config::default());
    
    // Send a query to Claude
    let response = client
        .query("Hello Claude! Please introduce yourself in 2-3 sentences.")
        .send()
        .await?;
    
    println!("\nClaude's response:\n{}", response);
    
    Ok(())
}
```

Run your application:
```bash
cargo run
```

## 📖 Common Use Cases

### API Decision Guide

The SDK offers two ways to send queries to Claude:

**Builder Pattern (Recommended)**
```rust
let response = client.query("Hello Claude!").send().await?;
```

**Direct Method (Simple)**
```rust
let response = client.send("Hello Claude!").await?;
```

**Which should you use?**
- **Builder Pattern**: More flexible, allows per-query customization (sessions, formats, etc.)
- **Direct Method**: Simpler for basic use cases, backward compatible

### 1. Custom Configuration

```rust
use claude_sdk_rs::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), claude_sdk_rs::Error> {
    // Configure Claude with custom settings
    let config = Config::builder()
        .model("claude-3-sonnet-20240229")  // Use specific model
        .system_prompt("You are a helpful coding assistant specialized in Rust.")
        .timeout_secs(30)
        .build()?;
    
    let client = Client::new(config);
    
    let response = client
        .query("What are the benefits of Rust's ownership system?")
        .send()
        .await?;
    
    println!("{}", response);
    Ok(())
}
```

### 2. Getting Response Metadata

```rust
use claude_sdk_rs::{Client, StreamFormat};

#[tokio::main]
async fn main() -> Result<(), claude_sdk_rs::Error> {
    // Enable JSON format to get metadata
    let client = Client::builder()
        .stream_format(StreamFormat::Json)
        .build();
    
    // Use send_full() to get complete response with metadata
    let response = client
        .query("Write a haiku about programming")
        .send_full()
        .await?;
    
    println!("Response: {}", response.content);
    
    if let Some(metadata) = response.metadata {
        println!("\nMetadata:");
        println!("  Session ID: {}", metadata.session_id);
        println!("  Model: {}", metadata.model);
        
        if let Some(cost) = metadata.cost_usd {
            println!("  Cost: ${:.6}", cost);
        }
        
        if let Some(tokens) = metadata.tokens_used {
            println!("  Tokens: {} input, {} output", 
                tokens.input, tokens.output);
        }
    }
    
    Ok(())
}
```

### 3. Streaming Responses

```rust
use claude_sdk_rs::{Client, StreamFormat};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), claude_sdk_rs::Error> {
    let client = Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .build();
    
    println!("Streaming Claude's response:\n");
    
    let mut stream = client
        .query("Write a short story about a curious robot exploring a library")
        .stream()
        .await?;
    
    // Process the stream chunk by chunk
    while let Some(result) = stream.next().await {
        match result {
            Ok(message) => {
                // Print content as it arrives
                if let Some(content) = message.content {
                    print!("{}", content);
                    // Flush to ensure immediate output
                    use std::io::{self, Write};
                    io::stdout().flush().unwrap();
                }
                
                // Check if response is complete
                if message.stop_reason.is_some() {
                    println!("\n\nStream complete!");
                    break;
                }
            }
            Err(e) => eprintln!("Stream error: {}", e),
        }
    }
    
    Ok(())
}
```

### 4. Multi-turn Conversations

```rust
use claude_sdk_rs::{Client, Config, StreamFormat};

#[tokio::main]
async fn main() -> Result<(), claude_sdk_rs::Error> {
    // Enable JSON format for session tracking
    let client = Client::builder()
        .stream_format(StreamFormat::Json)
        .build();
    
    // First message - introduce ourselves
    println!("You: Hi Claude! My name is Alex and I'm learning Rust.");
    let response1 = client
        .query("Hi Claude! My name is Alex and I'm learning Rust.")
        .send()
        .await?;
    println!("Claude: {}\n", response1);
    
    // Second message - Claude remembers context
    println!("You: What's a good first project for someone with my interests?");
    let response2 = client
        .query("What's a good first project for someone with my interests?")
        .send()
        .await?;
    println!("Claude: {}\n", response2);
    
    // Third message - Continue the conversation
    println!("You: That sounds great! What's my name again?");
    let response3 = client
        .query("That sounds great! What's my name again?")
        .send()
        .await?;
    println!("Claude: {}", response3);
    
    Ok(())
}
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. "Claude binary not found"
```bash
# Make sure Claude CLI is installed
claude --version

# If not found, reinstall with: npm install -g @anthropic-ai/claude-code
```

#### 2. "Not authenticated" error
```bash
# Authenticate with Claude
claude auth

# Verify authentication
claude status
```

#### 3. Timeout errors
```rust
// Increase timeout for longer operations
let client = Client::builder()
    .timeout_secs(120)  // 2 minutes
    .build();
```

#### 4. JSON parsing errors
```rust
// Use Text format if you don't need metadata
let client = Client::builder()
    .stream_format(StreamFormat::Text)
    .build();
```

#### 5. Session context not preserved
```rust
// Ensure you're using the same client instance
let client = Client::new(Config::default()); // Create once
// Use this same client for all queries in the conversation
```

## 🎯 Next Steps

### Explore More Examples
Check out the [`examples/`](examples/) directory:
```bash
# Run different examples
cargo run --example basic_usage
cargo run --example streaming
cargo run --example session_management
cargo run --example error_handling
```

### Advanced Features

1. **Tool Integration** - Enable MCP tools for file access and more:
   ```toml
   claude-sdk-rs = { version = "1.0", features = ["mcp"] }
   ```

2. **CLI Binary** - Use the interactive CLI:
   ```toml
   claude-sdk-rs = { version = "1.0", features = ["cli"] }
   ```
   ```bash
   cargo install claude-sdk-rs --features cli
   claude-sdk-rs
   ```

3. **SQLite Storage** - Persist sessions and history:
   ```toml
   claude-sdk-rs = { version = "1.0", features = ["sqlite"] }
   ```

### Learn More
- 📖 [API Documentation](https://docs.rs/claude-sdk-rs)
- 🔧 [Development Setup](DEV_SETUP.md)
- 🏗️ [Architecture Guide](CLAUDE.md)
- 💡 [Examples Directory](examples/)

## 💬 Getting Help

- **Documentation**: [docs.rs/claude-sdk-rs](https://docs.rs/claude-sdk-rs)
- **GitHub Issues**: [Report bugs or request features](https://github.com/bredmond1019/claude-sdk-rust/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/bredmond1019/claude-sdk-rust/discussions)

---

<div align="center">

🎉 **Congratulations!** You're now ready to build with claude-sdk-rs!

</div>