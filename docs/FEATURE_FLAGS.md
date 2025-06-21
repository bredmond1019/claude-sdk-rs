# Feature Flags Guide for claude-sdk-rs

This document provides comprehensive documentation for all feature flags available in claude-sdk-rs, including usage examples and current status.

**Last updated: 2025-06-19**

## Overview

claude-sdk-rs uses Cargo feature flags to provide optional functionality while keeping the core SDK lightweight. This modular approach allows users to include only the features they need.

## Available Features

### Core Features (Always Available)

The default build includes:
- Basic Claude AI SDK functionality
- Configuration management
- Session handling
- Error handling
- Core types and traits

```toml
[dependencies]
claude-sdk-rs = "1.0"
```

### Optional Features

| Feature | Status | Description | Dependencies |
|---------|--------|-------------|--------------|
| `cli` | ✅ Working | Command-line interface and tools | clap, colored, directories |
| `analytics` | ✅ Working | Analytics dashboard (requires cli) | Included with cli |
| `sqlite` | ⚠️ Broken | SQLite session storage | sqlx, sqlite |
| `mcp` | ⚠️ Broken | Model Context Protocol support | tokio-tungstenite, base64 |
| `full` | ⚠️ Broken | All features enabled | All of the above |

## Feature Details

### Default (Core SDK)

**Status**: ✅ Fully working

The minimal SDK includes everything needed for basic Claude AI interactions:

```toml
[dependencies]
claude-sdk-rs = "1.0"
```

**What's included**:
- `Client` and `Config` for API interactions
- `StreamFormat` options (Text, Json, StreamJson)
- Session management (in-memory)
- Error handling with recovery
- All core types and traits

**Example usage**:
```rust
use claude_sdk_rs::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), claude_sdk_rs::Error> {
    let client = Client::new(Config::default());
    let response = client.query("Hello Claude!").send().await?;
    println!("{}", response);
    Ok(())
}
```

### CLI Feature

**Status**: ✅ Working

Enables the command-line interface and interactive tools.

```toml
[dependencies]
claude-sdk-rs = { version = "1.0", features = ["cli"] }
```

**What's included**:
- Interactive CLI binary (`claude-sdk-rs`)
- Command execution and history
- Session management via CLI
- Configuration management
- Output formatting and coloring
- File and directory utilities

**Enable CLI**:
```bash
# Build with CLI support
cargo build --features cli

# Install CLI binary
cargo install claude-sdk-rs --features cli

# Use CLI
claude-sdk-rs --help
```

**CLI Commands**:
```bash
# Interactive mode
claude-sdk-rs

# Direct query
claude-sdk-rs query "What is Rust?"

# Session management
claude-sdk-rs sessions list
claude-sdk-rs sessions new "My coding session"

# Configuration
claude-sdk-rs config show
claude-sdk-rs config set model claude-3-sonnet-20240229
```

### Analytics Feature

**Status**: ✅ Working (requires CLI)

Provides analytics dashboard and usage tracking.

```toml
[dependencies]
claude-sdk-rs = { version = "1.0", features = ["analytics"] }
```

**What's included**:
- Usage analytics and metrics
- Cost tracking
- Performance monitoring
- Interactive dashboard
- Report generation
- Alert system

**Example usage**:
```bash
# Enable analytics (automatically includes CLI)
cargo build --features analytics

# View analytics dashboard
claude-sdk-rs analytics dashboard

# Generate reports
claude-sdk-rs analytics report --period week

# View cost breakdown
claude-sdk-rs analytics costs
```

**Programmatic access**:
```rust
use claude_sdk_rs::cli::analytics::AnalyticsEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = AnalyticsEngine::new().await?;
    let summary = engine.generate_summary().await?;
    println!("Total cost today: ${:.4}", summary.today_cost);
    Ok(())
}
```

### SQLite Feature

**Status**: ⚠️ Currently broken

Enables SQLite-based session persistence and storage.

```toml
# Currently fails to compile
[dependencies]
claude-sdk-rs = { version = "1.0", features = ["sqlite"] }
```

**Known issues**:
- Compilation errors with serde_json::Error::custom
- Missing trait imports for error handling
- SQLite schema and migration issues

**Intended functionality**:
- Persistent session storage
- Query history in database
- Session metadata tracking
- Migration support

**Workaround**:
Use in-memory session management until fixed:
```rust
use claude_sdk_rs::{Client, Config};

// Sessions are maintained in memory within the client
let client = Client::new(Config::default());
// Use the same client instance for conversation continuity
```

### MCP Feature

**Status**: ⚠️ Currently broken

Enables Model Context Protocol support for external tool integration.

```toml
# Currently fails to compile  
[dependencies]
claude-sdk-rs = { version = "1.0", features = ["mcp"] }
```

**Known issues**:
- Import resolution errors in connection pooling
- Circuit breaker configuration issues
- Health monitoring module path problems
- Missing metrics feature flag

**Intended functionality**:
- External service integration
- Tool protocol support
- Connection pooling and load balancing
- Health monitoring
- Service discovery

**Workaround**:
Use basic tool integration via Claude CLI:
```rust
use claude_sdk_rs::{Client, Config};

let config = Config::builder()
    .allowed_tools(vec![
        "bash:ls".to_string(),
        "bash:cat".to_string(),
    ])
    .build()?;

let client = Client::new(config);
```

### Full Feature

**Status**: ⚠️ Currently broken (due to mcp and sqlite)

Enables all available features.

```toml
# Currently fails to compile
[dependencies]
claude-sdk-rs = { version = "1.0", features = ["full"] }
```

**Equivalent to**:
```toml
[dependencies]
claude-sdk-rs = { version = "1.0", features = ["cli", "analytics", "mcp", "sqlite"] }
```

## Recommended Feature Combinations

### For Basic Usage
```toml
[dependencies]
claude-sdk-rs = "1.0"
```

### For CLI Applications
```toml
[dependencies]
claude-sdk-rs = { version = "1.0", features = ["cli"] }
```

### For Analytics and Monitoring
```toml
[dependencies]
claude-sdk-rs = { version = "1.0", features = ["analytics"] }
```

### For Development (Currently Working Features)
```toml
[dependencies]
claude-sdk-rs = { version = "1.0", features = ["cli", "analytics"] }
```

## Build Examples

### Testing Feature Combinations

```bash
# Test core functionality
cargo build
cargo test --lib

# Test CLI features
cargo build --features cli
cargo test --features cli

# Test analytics features  
cargo build --features analytics
cargo test --features analytics

# Test working combination
cargo build --features "cli,analytics"

# Avoid these (will fail):
# cargo build --features mcp
# cargo build --features sqlite  
# cargo build --features full
```

### Running Examples with Features

```bash
# Core examples (no features needed)
cargo run --example basic_usage
cargo run --example streaming
cargo run --example session_management

# CLI examples
cargo run --example cli_interactive --features cli
cargo run --example cli_analytics --features analytics

# Check available examples
find examples -name "*.rs" | grep -E "(cli_|analytics_)"
```

## Development Notes

### Adding New Features

When adding a new feature:

1. Update `Cargo.toml` features section
2. Use `#[cfg(feature = "feature_name")]` for conditional compilation
3. Add feature-specific dependencies as optional
4. Update this documentation
5. Add examples demonstrating the feature
6. Test with various feature combinations

### Fixing Broken Features

Current priority for fixes:

1. **SQLite feature**: Fix serde_json error handling
2. **MCP feature**: Resolve import path issues  
3. **Integration**: Test all feature combinations work together

### Feature Flag Best Practices

- Keep the default feature set minimal
- Make features composable (they should work together)
- Use descriptive feature names
- Document dependencies between features
- Provide examples for each feature combination

## Troubleshooting

### Compilation Errors

```bash
# If mcp or sqlite features fail:
error[E0433]: failed to resolve: unresolved import
```

**Solution**: Use working features only:
```bash
cargo build --features cli
```

### Missing Dependencies

```bash
# If CLI commands fail:
error: no such command: 'claude-sdk-rs'
```

**Solution**: Install with CLI features:
```bash
cargo install claude-sdk-rs --features cli
```

### Runtime Errors

```bash
# If features don't behave as expected:
cargo build --features cli --verbose
```

**Check**: Ensure the feature is actually enabled in the build.

## Migration Guide

### From Full Features to Working Features

If you were using:
```toml
claude-sdk-rs = { version = "1.0", features = ["full"] }
```

Change to:
```toml
claude-sdk-rs = { version = "1.0", features = ["cli", "analytics"] }
```

### From SQLite to In-Memory Sessions

Replace:
```rust
// This won't compile currently
use claude_sdk_rs::sqlite::SqliteSessionManager;
```

With:
```rust
// Use built-in session management
use claude_sdk_rs::{Client, Config};
let client = Client::new(Config::default());
// Sessions are managed automatically
```

## Roadmap

### Planned Fixes

- [ ] Fix SQLite feature compilation errors
- [ ] Resolve MCP feature import issues  
- [ ] Add more granular feature flags
- [ ] Improve feature documentation
- [ ] Add feature-specific integration tests

### Future Features

- [ ] Redis session storage
- [ ] Advanced analytics exporters
- [ ] Plugin system
- [ ] Custom protocol support

---

For the most up-to-date feature status, check the [GitHub Issues](https://github.com/bredmond1019/claude-sdk-rust/issues) page.