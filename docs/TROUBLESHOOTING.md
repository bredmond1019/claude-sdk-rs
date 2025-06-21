# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when developing with and using the Claude AI SDK.

**Last verified: 2025-06-19**
**Development status**: Updated with current build and development issues

## Table of Contents

- [Error Code Reference](#error-code-reference)
- [Development Issues](#development-issues)
- [Feature Flag Problems](#feature-flag-problems)
- [Build and Compilation Issues](#build-and-compilation-issues)
- [Common Issues](#common-issues)
- [Installation Problems](#installation-problems)
- [Authentication Issues](#authentication-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Testing Issues](#testing-issues)
- [Debugging Tips](#debugging-tips)

## Error Code Reference

Each error in the Claude AI SDK has a unique code for easy reference:

| Code | Error Type | Description | Solution |
|------|------------|-------------|----------|
| C001 | BinaryNotFound | Claude CLI not installed or not in PATH | Install Claude CLI: `curl -fsSL https://claude.ai/install.sh \| sh` |
| C002 | SessionNotFound | Session ID doesn't exist | Create new session or use existing one |
| C003 | PermissionDenied | Tool access restricted | Add tool to `allowed_tools` in config |
| C004 | McpError | MCP server communication failed | Check MCP server status and connectivity |
| C005 | ConfigError | Invalid configuration | Review configuration parameters |
| C006 | InvalidInput | Input doesn't meet constraints | Validate input format and content |
| C007 | Timeout | Operation exceeded timeout | Increase timeout or optimize query |
| C008 | SerializationError | JSON parsing failed | Check response format and data |
| C009 | IoError | I/O operation failed | Check file permissions and disk space |
| C010 | ProcessError | CLI process failed | Check CLI installation and logs |
| C011 | StreamClosed | Streaming connection lost | Retry operation or check network |
| C012 | NotAuthenticated | Claude CLI not authenticated | Run `claude auth` to authenticate |
| C013 | RateLimitExceeded | Too many requests | Wait and retry with backoff |

## Development Issues

### Feature Compilation Errors

**Problem**: MCP and SQLite features fail to compile

**Symptoms**:
```
error[E0433]: failed to resolve: unresolved import
 --> src/mcp/config.rs:93:45
 |
93 |             load_balancing_strategy: crate::connection_pool::LoadBalancingStrategy::HealthBased,
```

**Solutions**:
1. Use working features only:
   ```bash
   cargo build --features cli
   cargo build --features analytics
   ```

2. Avoid broken features during development:
   ```bash
   # These will fail:
   # cargo build --features mcp
   # cargo build --features sqlite
   # cargo build --features full
   ```

3. Check feature status in [FEATURE_FLAGS.md](FEATURE_FLAGS.md)

### Clippy Warnings and Errors

**Problem**: Many clippy warnings and some errors

**Symptoms**:
```
error: this block is too nested
warning: unused imports
warning: missing documentation
```

**Solutions**:
1. Use clippy without strict mode during development:
   ```bash
   cargo clippy  # Shows warnings but doesn't fail
   ```

2. Avoid strict clippy during development:
   ```bash
   # This will fail with current code:
   # cargo clippy -- -D warnings
   ```

3. Use Makefile for development workflow:
   ```bash
   make dev  # Runs fmt, clippy (non-strict), and tests
   ```

### Test Timeouts

**Problem**: Tests take very long or timeout

**Symptoms**:
```
test runtime::telemetry::tests::test_error_rate_calculation has been running for over 60 seconds
```

**Solutions**:
1. Run unit tests only for faster development:
   ```bash
   cargo test --lib
   ```

2. Use timeout for full test runs:
   ```bash
   cargo test --timeout 300  # 5 minutes
   ```

3. Skip integration tests during development:
   ```bash
   cargo test --lib -- --skip integration
   ```

## Feature Flag Problems

### Unknown Feature Errors

**Problem**: Features don't exist or are broken

**Solutions**:
1. Check available features:
   ```bash
   grep -A 20 "\[features\]" Cargo.toml
   ```

2. Use only working features:
   ```toml
   # Working:
   claude-sdk-rs = { version = "1.0", features = ["cli"] }
   claude-sdk-rs = { version = "1.0", features = ["analytics"] }
   
   # Broken (avoid):
   # claude-sdk-rs = { version = "1.0", features = ["mcp"] }
   # claude-sdk-rs = { version = "1.0", features = ["sqlite"] }
   ```

3. See [FEATURE_FLAGS.md](FEATURE_FLAGS.md) for current status

## Build and Compilation Issues

### Dependency Compilation Errors

**Problem**: Long compilation times or dependency errors

**Solutions**:
1. Clean and rebuild:
   ```bash
   cargo clean
   cargo build
   ```

2. Update dependencies:
   ```bash
   cargo update
   ```

3. Use faster compilation checks:
   ```bash
   cargo check  # Faster than full build
   ```

### Binary Not Found During Build

**Problem**: CLI binary features require additional setup

**Solutions**:
1. Ensure you're building with CLI features:
   ```bash
   cargo build --features cli
   ```

2. Install the CLI binary:
   ```bash
   cargo install --path . --features cli
   ```

3. Check binary was created:
   ```bash
   ls target/debug/deps/claude_sdk_rs-*
   ```

## Common Issues

### Claude CLI not found (C001)

**Symptoms:**
```
Error: [C001] Claude Code not found in PATH
```

**Solutions:**
1. Install Claude CLI:
   ```bash
   curl -fsSL https://claude.ai/install.sh | sh
   ```

2. Verify installation:
   ```bash
   which claude
   claude --version
   ```

3. Add to PATH if needed:
   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```

### Authentication required (C012)

**Symptoms:**
```
Error: [C012] Claude CLI is not authenticated
```

**Solutions:**
1. Authenticate with Claude:
   ```bash
   claude auth
   ```

2. Follow the prompts to enter your API key

3. Verify authentication:
   ```bash
   claude api test
   ```

### Timeout errors (C007)

**Symptoms:**
```
Error: [C007] Operation timed out after 30s
```

**Solutions:**
1. Increase timeout in configuration:
   ```rust
   let client = Client::builder()
       .timeout_secs(120)  // 2 minutes
       .build();
   ```

2. Optimize your queries:
   - Break large requests into smaller ones
   - Use streaming for long responses
   - Simplify complex prompts

### Session not found (C002)

**Symptoms:**
```
Error: [C002] Session abc123 not found
```

**Solutions:**
1. Use the same client instance for related queries:
   ```rust
   let client = Client::new(Config::default());
   // Use this client for all queries in the conversation
   ```

2. Check if session exists:
   ```rust
   let sessions = SessionManager::new()?;
   if let Some(session) = sessions.get(session_id)? {
       // Session exists
   }
   ```

### Permission denied (C003)

**Symptoms:**
```
Error: [C003] Tool permission denied: filesystem
```

**Solutions:**
1. Enable required tools in configuration:
   ```rust
   let client = Client::builder()
       .allowed_tools(vec![
           "mcp__filesystem__read".to_string(),
           "mcp__filesystem__write".to_string(),
           "bash:ls".to_string(),  // Basic bash tools
           "bash:cat".to_string(),
       ])
       .build();
   ```

2. Check available tools:
   ```bash
   claude api tools list
   ```

3. Use basic bash tools instead of MCP (since MCP feature is broken):
   ```rust
   let config = Config::builder()
       .allowed_tools(vec!["bash:ls".to_string(), "bash:cat".to_string()])
       .build()?;
   ```

## Installation Problems

### Cargo build fails

**Problem:** Build errors when compiling the SDK

**Solutions:**
1. Ensure Rust 1.70+ is installed:
   ```bash
   rustc --version
   cargo --version
   ```

2. Update Rust:
   ```bash
   rustup update stable
   ```

3. Clean and rebuild:
   ```bash
   cargo clean
   cargo build
   ```

### Feature flag issues

**Problem:** Optional features not working or failing to compile

**Solutions:**
1. Use only working features in Cargo.toml:
   ```toml
   [dependencies]
   # Working features:
   claude-sdk-rs = { version = "1.0.0", features = ["cli"] }
   claude-sdk-rs = { version = "1.0.0", features = ["analytics"] }
   
   # Broken features (avoid):
   # claude-sdk-rs = { version = "1.0.0", features = ["mcp"] }
   # claude-sdk-rs = { version = "1.0.0", features = ["sqlite"] }
   ```

2. Verify feature compilation:
   ```bash
   cargo build --features cli      # This works
   cargo build --features analytics # This works
   # cargo build --features mcp     # This fails
   ```

3. Check feature status documentation:
   ```bash
   cat docs/FEATURE_FLAGS.md
   ```

## Authentication Issues

### Invalid API key

**Problem:** Authentication fails with valid-looking key

**Solutions:**
1. Verify key format and validity
2. Check for extra whitespace
3. Ensure key has necessary permissions
4. Try re-authenticating:
   ```bash
   claude auth --force
   ```

### Token expiration

**Problem:** Previously working auth stops working

**Solutions:**
1. Re-authenticate with Claude CLI
2. Check for API key rotation policies
3. Implement token refresh in your application

## Runtime Errors

### Process failures (C010)

**Problem:** Claude CLI process crashes or returns errors

**Solutions:**
1. Check Claude CLI logs:
   ```bash
   claude logs
   ```

2. Verify CLI version compatibility:
   ```bash
   claude --version
   ```

3. Try running CLI directly:
   ```bash
   echo "Hello" | claude api messages create
   ```

### Serialization errors (C008)

**Problem:** JSON parsing failures

**Solutions:**
1. Use appropriate StreamFormat:
   ```rust
   // For simple text responses
   .stream_format(StreamFormat::Text)
   
   // For structured data
   .stream_format(StreamFormat::Json)
   ```

2. Handle partial JSON in streaming:
   ```rust
   use serde_json::Value;
   
   match serde_json::from_str::<Value>(&partial) {
       Ok(json) => process(json),
       Err(_) => continue, // Wait for more data
   }
   ```

### Stream interruptions (C011)

**Problem:** Streaming responses get cut off

**Solutions:**
1. Implement retry logic:
   ```rust
   let mut retries = 3;
   while retries > 0 {
       match client.query(prompt).stream().await {
           Ok(stream) => break,
           Err(Error::StreamClosed) => {
               retries -= 1;
               tokio::time::sleep(Duration::from_secs(1)).await;
           }
           Err(e) => return Err(e),
       }
   }
   ```

2. Use connection pooling for better stability

## Performance Issues

### Slow response times

**Solutions:**
1. Use appropriate model for your needs:
   ```rust
   // Faster model for simple tasks
   .model("claude-3-haiku-20240307")
   
   // More capable model for complex tasks
   .model("claude-3-opus-20240229")
   ```

2. Enable response streaming:
   ```rust
   let mut stream = client.query(prompt).stream().await?;
   while let Some(chunk) = stream.next().await {
       process_chunk(chunk?);
   }
   ```

3. Optimize prompts:
   - Be specific and concise
   - Avoid unnecessary context
   - Use system prompts effectively

### High memory usage

**Solutions:**
1. Use memory-optimized configuration:
   ```rust
   use claude_ai_runtime::StreamConfig;
   
   let config = StreamConfig::memory_optimized();
   ```

2. Process messages immediately instead of buffering
3. Clear old sessions periodically

### Rate limiting (C013)

**Solutions:**
1. Implement exponential backoff:
   ```rust
   use tokio::time::{sleep, Duration};
   
   let mut delay = Duration::from_millis(100);
   for _ in 0..5 {
       match client.query(prompt).send().await {
           Ok(response) => return Ok(response),
           Err(Error::RateLimitExceeded) => {
               sleep(delay).await;
               delay *= 2;
           }
           Err(e) => return Err(e),
       }
   }
   ```

2. Track and respect rate limits
3. Use batch processing when possible

## Debugging Tips

### Enable debug logging

Set environment variables:
```bash
export RUST_LOG=claude_ai=debug
export RUST_BACKTRACE=1
```

### Capture raw CLI output

```rust
let client = Client::builder()
    .stream_format(StreamFormat::Text)
    .build();

let raw_output = client.query("test").send().await?;
println!("Raw: {}", raw_output);
```

### Use error recovery helpers

```rust
use claude_ai_core::Error;

match result {
    Err(e) if e.is_recoverable() => {
        // Retry logic for recoverable errors
    }
    Err(e) => {
        eprintln!("Fatal error {}: {}", e.code(), e);
    }
    Ok(v) => process(v),
}
```

### Monitor system resources

```bash
# Watch memory usage
watch -n 1 'ps aux | grep your-app'

# Monitor file descriptors
lsof -p $(pgrep your-app) | wc -l

# Check network connections
netstat -an | grep claude
```

### Common environment issues

1. **Proxy settings**: Ensure HTTP_PROXY and HTTPS_PROXY are set correctly
2. **Firewall**: Check that Claude API endpoints are accessible
3. **DNS**: Verify DNS resolution for api.anthropic.com
4. **SSL/TLS**: Ensure certificates are up to date

## Testing Issues

### Tests Running Too Long

**Problem**: Some tests run for over 60 seconds

**Solutions**:
1. Run unit tests only:
   ```bash
   cargo test --lib
   ```

2. Use timeouts:
   ```bash
   cargo test --timeout 300
   ```

3. Skip problematic tests:
   ```bash
   cargo test --lib -- --skip telemetry
   ```

### Integration Tests Failing

**Problem**: Integration tests require external dependencies

**Solutions**:
1. Ensure Claude CLI is installed and authenticated:
   ```bash
   claude --version
   claude auth
   ```

2. Run integration tests separately:
   ```bash
   cargo test --test '*'
   ```

3. Use environment variables for test configuration:
   ```bash
   RUST_LOG=debug cargo test --lib
   ```

## Development Workflow

### Recommended Development Commands

```bash
# Quick development cycle
cargo check                    # Fast compilation check
cargo test --lib              # Fast unit tests
cargo clippy                   # Non-strict linting
cargo fmt                      # Code formatting

# Full development cycle
make dev                       # Format, lint, test

# Before committing
cargo build --features cli     # Verify CLI builds
cargo run --example basic_usage # Verify examples work
```

### Common Development Environment Setup

```bash
# 1. Clone and setup
git clone https://github.com/bredmond1019/claude-sdk-rust.git
cd claude-sdk-rust

# 2. Verify prerequisites
rustc --version  # Should be 1.70+
claude --version # Should be installed
claude auth      # Should be authenticated

# 3. Test core functionality
cargo build
cargo test --lib

# 4. Test CLI functionality
cargo build --features cli
cargo run --example basic_usage

# 5. Use development tools
make help        # See all available commands
make dev         # Run development workflow
```

## Getting Help

If you're still experiencing issues:

1. **Check Current Status**: See [DEV_SETUP.md](../DEV_SETUP.md) and [FEATURE_FLAGS.md](FEATURE_FLAGS.md)
2. **Search Issues**: Look for similar problems on [GitHub Issues](https://github.com/bredmond1019/claude-sdk-rust/issues)
3. **Enable Debug Mode**: Set `RUST_LOG=debug` for detailed logs
4. **Use Working Features**: Stick to `cli` and `analytics` features
5. **Create an Issue**: Provide:
   - Error message with code
   - Minimal reproduction code
   - Environment details (OS, Rust version, Claude CLI version)
   - Feature flags used
   - Debug logs if available

### Development-Specific Help

For development issues:

1. **Check Build Status**: Verify which features currently compile
2. **Use Makefile**: Run `make help` for development commands
3. **Test Incrementally**: Start with core features, add CLI, avoid broken features
4. **Check Documentation**: Feature flags and development setup are documented

## Quick Reference Card

```rust
// Error handling pattern
use claude_sdk_rs::{Client, Config, Error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup with error handling
    let client = Client::new(Config::default());
    
    match client.query("Hello").send().await {
        Ok(response) => {
            println!("Success: {}", response);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            
            // Handle specific error types
            match e {
                Error::BinaryNotFound => {
                    eprintln!("Please install Claude CLI: https://claude.ai/code");
                }
                Error::Timeout => {
                    eprintln!("Try increasing timeout in config");
                }
                Error::ProcessError(msg) => {
                    eprintln!("Process error: {}", msg);
                }
                _ => {
                    eprintln!("Other error: {:?}", e);
                }
            }
        }
    }
    
    Ok(())
}
```

### Development Error Patterns

```bash
# Handle compilation errors
if cargo build --features cli; then
    echo "CLI feature works"
else
    echo "CLI feature has issues, using core only"
    cargo build
fi

# Handle test timeouts
timeout 300 cargo test --lib || echo "Tests took too long, running unit tests only"

# Handle clippy issues
cargo clippy 2>/dev/null || echo "Clippy has warnings, continuing development"
```