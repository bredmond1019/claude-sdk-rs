# Part 3: Configuration and Customization

The claude-sdk-rs SDK provides extensive configuration options to customize Claude's behavior, performance, and integration with your application. This tutorial covers all available configuration options and common patterns.

## Configuration Overview

Configuration in claude-sdk-rs is handled through the `Config` struct, which supports both default settings and builder pattern customization:

```rust
use claude_sdk_rs::{Client, Config, StreamFormat};

// Default configuration
let client = Client::new(Config::default());

// Custom configuration with builder pattern
let config = Config::builder()
    .model("claude-opus-4")
    .system_prompt("You are a helpful Rust programming assistant")
    .stream_format(StreamFormat::Json)
    .timeout_secs(60)
    .build();

let client = Client::new(config);
```

## Core Configuration Options

### Model Selection

Choose which Claude model to use based on your needs:

```rust
let config = Config::builder()
    .model("claude-3-opus-20240229")     // Most capable model
    .build();

// Other available models:
// - "claude-3-opus-20240229"    - Most capable, higher cost
// - "claude-3-sonnet-20240229"  - Balanced performance
// - "claude-3-haiku-20240307"   - Fastest, most cost-effective
```

**When to use each model:**
- **Opus**: Complex reasoning, code generation, detailed analysis
- **Sonnet**: Best balance of capability and cost for most applications
- **Haiku**: Quick queries, simple tasks, high-volume applications

### System Prompts

Set consistent context and behavior for your AI assistant:

```rust
let config = Config::builder()
    .system_prompt(
        "You are an expert Rust developer. Always provide idiomatic, \
         safe code examples with error handling. Explain your reasoning \
         and suggest best practices."
    )
    .build();
```

**System prompt tips:**
- Be specific about the role and expertise level
- Include output format preferences
- Mention any constraints or guidelines
- Keep it concise but comprehensive

### Stream Formats

Control response structure and metadata availability:

```rust
// Plain text (default) - simplest, no metadata
let config = Config::builder()
    .stream_format(StreamFormat::Text)
    .build();

// JSON - structured response with full metadata
let config = Config::builder()
    .stream_format(StreamFormat::Json)
    .build();

// StreamJSON - line-delimited JSON for streaming
let config = Config::builder()
    .stream_format(StreamFormat::StreamJson)
    .build();
```

### Timeouts and Performance

Configure timeouts based on your use case:

```rust
let config = Config::builder()
    .timeout_secs(30)      // Default: 30 seconds
    .max_tokens(4096)      // Limit response length
    .build();
```

**Timeout guidelines:**
- **30s**: Quick queries, simple questions
- **60s**: Code generation, analysis
- **120s+**: Complex reasoning, large document processing

## Tool Configuration

Control which tools Claude can access:

```rust
use claude_sdk_rs::{Config, ToolPermission};

let config = Config::builder()
    .allowed_tools(vec![
        // Built-in tools
        "bash".to_string(),
        "filesystem".to_string(),
        
        // MCP server tools (format: server__tool)
        "database__query".to_string(),
        "web__search".to_string(),
    ])
    .build();
```

### Tool Permission Patterns

For more advanced tool configuration, use the `ToolPermission` helper:

```rust
let config = Config::builder()
    .allowed_tools(vec![
        ToolPermission::bash("npm install").to_cli_format(),
        ToolPermission::bash("npm run").to_cli_format(),
        ToolPermission::mcp("filesystem", "*").to_cli_format(),
        ToolPermission::mcp("database", "read_only").to_cli_format(),
    ])
    .build();
```

## MCP (Model Context Protocol) Configuration

For advanced tool integration, configure MCP servers:

```rust
use std::path::PathBuf;

let config = Config::builder()
    .mcp_config(PathBuf::from("./mcp-config.json"))
    .allowed_tools(vec![
        "database__query".to_string(),
        "filesystem__read".to_string(),
        "web__search".to_string(),
    ])
    .build();
```

Example `mcp-config.json`:
```json
{
  "servers": {
    "database": {
      "command": "mcp-database-server",
      "args": ["--connection-string", "postgresql://..."]
    },
    "filesystem": {
      "command": "mcp-filesystem-server",
      "args": ["--root", "/safe/directory"]
    }
  }
}
```

## Common Configuration Patterns

### Development Configuration

Optimal settings for development and testing:

```rust
fn dev_config() -> Config {
    Config::builder()
        .model("claude-3-opus-20240229")
        .stream_format(StreamFormat::Json)
        .system_prompt("You are a helpful development assistant")
        .timeout_secs(60)
        .allowed_tools(vec![
            "bash".to_string(),
            "filesystem".to_string(),
        ])
        .build()
}
```

### Production Configuration

Optimized for production environments:

```rust
fn production_config() -> Config {
    Config::builder()
        .model("claude-3-haiku-20240307")  // Cost-effective
        .stream_format(StreamFormat::Json)
        .timeout_secs(30)                     // Fast timeouts
        .max_tokens(2048)                     // Limit response size
        .allowed_tools(vec![])                // No tools for security
        .build()
}
```

### Streaming Chat Configuration

For real-time chat applications:

```rust
fn chat_config() -> Config {
    Config::builder()
        .model("claude-3-opus-20240229")
        .stream_format(StreamFormat::StreamJson)  // Enable streaming
        .system_prompt("You are a helpful assistant. Keep responses conversational and engaging.")
        .timeout_secs(120)
        .build()
}
```

### Code Assistant Configuration

Specialized for code-related tasks:

```rust
fn code_assistant_config() -> Config {
    Config::builder()
        .model("claude-3-opus-20240229")
        .stream_format(StreamFormat::Json)
        .system_prompt(
            "You are an expert software engineer. Always provide:\n\
             1. Working, idiomatic code\n\
             2. Proper error handling\n\
             3. Clear explanations\n\
             4. Security considerations"
        )
        .allowed_tools(vec![
            "bash".to_string(),
            "filesystem".to_string(),
        ])
        .timeout_secs(90)
        .build()
}
```

## Environment-Based Configuration

Load configuration from environment variables:

```rust
use std::env;

fn config_from_env() -> Config {
    let mut builder = Config::builder();
    
    if let Ok(model) = env::var("CLAUDE_MODEL") {
        builder = builder.model(model);
    }
    
    if let Ok(timeout) = env::var("CLAUDE_TIMEOUT") {
        if let Ok(timeout_secs) = timeout.parse::<u64>() {
            builder = builder.timeout_secs(timeout_secs);
        }
    }
    
    if let Ok(system_prompt) = env::var("CLAUDE_SYSTEM_PROMPT") {
        builder = builder.system_prompt(system_prompt);
    }
    
    builder.build()
}

// Usage
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = config_from_env();
    let client = Client::new(config);
    
    // ... rest of your application
    Ok(())
}
```

## Configuration Validation

Validate configurations before use:

```rust
impl Config {
    fn validate(&self) -> Result<(), String> {
        if let Some(timeout) = self.timeout_secs {
            if timeout == 0 {
                return Err("Timeout must be greater than 0".to_string());
            }
            if timeout > 600 {
                return Err("Timeout too large (max 600s)".to_string());
            }
        }
        
        if let Some(max_tokens) = self.max_tokens {
            if max_tokens == 0 {
                return Err("Max tokens must be greater than 0".to_string());
            }
        }
        
        Ok(())
    }
}

// Usage
let config = Config::builder()
    .timeout_secs(60)
    .max_tokens(4096)
    .build();

config.validate().expect("Invalid configuration");
```

## Performance Optimization

### Response Size Control

```rust
let config = Config::builder()
    .max_tokens(1024)              // Limit token usage
    .stream_format(StreamFormat::Text)  // Reduce parsing overhead
    .build();
```

### Timeout Optimization

```rust
// Fast responses for simple queries
let quick_config = Config::builder()
    .timeout_secs(15)
    .model("claude-3-haiku-20240307")
    .build();

// Longer timeouts for complex tasks
let complex_config = Config::builder()
    .timeout_secs(180)
    .model("claude-3-opus-20240229")
    .build();
```

## Security Considerations

### Restrictive Tool Access

```rust
let secure_config = Config::builder()
    .allowed_tools(vec![])  // No tools allowed
    .system_prompt("You can only provide information, not execute commands")
    .build();
```

### Sandboxed Environment

```rust
let sandboxed_config = Config::builder()
    .allowed_tools(vec![
        ToolPermission::bash("ls").to_cli_format(),
        ToolPermission::bash("cat").to_cli_format(),
        // Only allow specific, safe commands
    ])
    .mcp_config(PathBuf::from("./restricted-mcp.json"))
    .build();
```

## Testing Configuration

Create test-specific configurations:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    fn test_config() -> Config {
        Config::builder()
            .model("claude-3-haiku-20240307")  // Fastest for tests
            .timeout_secs(10)                     // Short timeout
            .stream_format(StreamFormat::Text)    // Simple output
            .build()
    }
    
    #[tokio::test]
    async fn test_basic_query() {
        let client = Client::new(test_config());
        let response = client.query("Hello").send().await;
        assert!(response.is_ok());
    }
}
```

## Next Steps

Now that you understand configuration, explore:

- **Part 4**: [Streaming Responses](04-streaming-responses.md) - Real-time response processing
- **Part 5**: [Tool Integration](05-tool-integration.md) - Working with external tools and MCP

## Configuration Best Practices

1. **Use environment variables** for deployment-specific settings
2. **Start with defaults** and customize only what you need
3. **Validate configurations** before creating clients
4. **Match timeouts to use cases** - short for simple queries, long for complex tasks
5. **Consider security** when enabling tools
6. **Test configurations** with your actual workloads

The configuration system is designed to grow with your application's needs!