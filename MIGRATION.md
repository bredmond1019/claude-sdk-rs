# Migration Guide

This guide helps you migrate from earlier versions of the `claude-sdk-rs` to the latest version with enhanced CLI features.

## Overview

The latest version introduces several new CLI flags and configuration options that enhance control over Claude AI interactions:

- **System Prompt Extension**: `append_system_prompt` for adding to existing prompts
- **Conversation Limits**: `max_turns` for controlling conversation length
- **Granular Tool Permissions**: Enhanced tool control with specific command formats
- **Permission Management**: `skip_permissions` for automated/secure environments

## Breaking Changes

### None

All new features are additive and backward compatible. Existing code will continue to work without modifications.

## New Features Migration

### 1. System Prompt Extension

**Before:**
```rust
let client = Client::builder()
    .system_prompt("You are a helpful assistant.")
    .build();
```

**After (Enhanced):**
```rust
// Option 1: Replace system prompt entirely (unchanged)
let client = Client::builder()
    .system_prompt("You are a helpful assistant.")
    .build();

// Option 2: Extend existing system prompt (NEW)
let client = Client::builder()
    .append_system_prompt("Additionally, be very concise.")
    .build();

// Note: Cannot use both system_prompt and append_system_prompt together
```

### 2. Conversation Control

**New Feature - Max Turns:**
```rust
// Limit conversations to 5 back-and-forth exchanges
let client = Client::builder()
    .max_turns(5)
    .timeout_secs(120)  // Increase timeout for longer conversations
    .build();
```

### 3. Enhanced Tool Permissions

**Before:**
```rust
let client = Client::builder()
    .allowed_tools(vec![
        "bash".to_string(),
        "mcp_server_tool".to_string(),
    ])
    .build();
```

**After (Enhanced):**
```rust
use claude_sdk_rs::ToolPermission;

// Option 1: Use the new ToolPermission enum (RECOMMENDED)
let client = Client::builder()
    .allowed_tools(vec![
        ToolPermission::bash("git").to_cli_format(),
        ToolPermission::bash("npm install").to_cli_format(),
        ToolPermission::mcp("filesystem", "read").to_cli_format(),
        ToolPermission::All.to_cli_format(),
    ])
    .build();

// Option 2: Use granular string formats directly
let client = Client::builder()
    .allowed_tools(vec![
        "Bash(git status)".to_string(),     // Specific bash command
        "bash:ls".to_string(),              // Legacy format (still supported)
        "mcp__database__query".to_string(), // MCP tool
        "*".to_string(),                    // All tools
    ])
    .build();

// Option 3: Original format still works
let client = Client::builder()
    .allowed_tools(vec![
        "bash".to_string(),
        "filesystem".to_string(),
    ])
    .build();
```

### 4. Tool Restrictions

**New Feature - Disallowed Tools:**
```rust
let client = Client::builder()
    .disallowed_tools(vec![
        "Bash(rm)".to_string(),           // Block dangerous commands
        "Bash(sudo)".to_string(),
        "mcp__system__shutdown".to_string(),
    ])
    .build();
```

### 5. Permission Prompts

**New Feature - Permission Control:**
```rust
// Default: Skip permission prompts (automated environment)
let automated_client = Client::builder()
    .skip_permissions(true)  // Default
    .build();

// Secure: Require permission prompts (interactive environment)
let secure_client = Client::builder()
    .skip_permissions(false)
    .build();
```

## Migration Examples

### Example 1: Basic Configuration Update

**Before:**
```rust
use claude_sdk_rs::{Client, StreamFormat};

let client = Client::builder()
    .model("claude-3-sonnet-20240229")
    .system_prompt("You are a code reviewer.")
    .stream_format(StreamFormat::Json)
    .timeout_secs(60)
    .build();
```

**After (with new features):**
```rust
use claude_sdk_rs::{Client, StreamFormat, ToolPermission};

let client = Client::builder()
    .model("claude-3-sonnet-20240229")
    .system_prompt("You are a code reviewer.")
    .append_system_prompt("Focus on security and performance.")  // NEW
    .stream_format(StreamFormat::Json)
    .timeout_secs(60)
    .max_turns(10)  // NEW: Limit review conversation
    .allowed_tools(vec![  // ENHANCED: Granular permissions
        ToolPermission::bash("git diff").to_cli_format(),
        ToolPermission::bash("cargo clippy").to_cli_format(),
        ToolPermission::mcp("filesystem", "read").to_cli_format(),
    ])
    .disallowed_tools(vec![  // NEW: Block dangerous operations
        "Bash(rm)".to_string(),
        "Bash(sudo)".to_string(),
    ])
    .skip_permissions(false)  // NEW: Require explicit permissions
    .build();
```

### Example 2: Tool Configuration Migration

**Before:**
```rust
let client = Client::builder()
    .allowed_tools(vec![
        "bash".to_string(),
        "filesystem".to_string(),
    ])
    .build();
```

**After (granular control):**
```rust
use claude_sdk_rs::ToolPermission;

let client = Client::builder()
    .allowed_tools(vec![
        // Specific bash commands only
        ToolPermission::bash("ls").to_cli_format(),
        ToolPermission::bash("cat").to_cli_format(),
        ToolPermission::bash("grep").to_cli_format(),
        
        // Read-only filesystem access
        ToolPermission::mcp("filesystem", "read").to_cli_format(),
        
        // Specific MCP server tools
        ToolPermission::mcp("database", "query").to_cli_format(),
    ])
    .disallowed_tools(vec![
        // Block dangerous commands even if bash is broadly allowed
        "Bash(rm)".to_string(),
        "Bash(chmod)".to_string(),
        "mcp__system__delete".to_string(),
    ])
    .build();
```

### Example 3: Session Management Enhancement

**Before:**
```rust
let client = Client::builder()
    .stream_format(StreamFormat::Json)
    .build();

// Multiple queries without control
let response1 = client.query("Start analysis").send().await?;
let response2 = client.query("Continue analysis").send().await?;
// ... potentially many more
```

**After (with conversation control):**
```rust
let client = Client::builder()
    .stream_format(StreamFormat::Json)
    .max_turns(5)  // NEW: Automatically limit conversation length
    .append_system_prompt("Keep responses focused and conclude quickly.")  // NEW
    .build();

// Conversation will automatically stop after 5 exchanges
let response1 = client.query("Start analysis").send().await?;
let response2 = client.query("Continue analysis").send().await?;
// ... max 5 total exchanges
```

## Validation Changes

### Enhanced Error Handling

The new features include comprehensive validation:

```rust
// These will now return validation errors:

// Invalid: Cannot use both system_prompt and append_system_prompt
let invalid_config = Client::builder()
    .system_prompt("Base prompt")
    .append_system_prompt("Extended prompt")  // ERROR
    .build();  // Returns Error::InvalidInput

// Invalid: Cannot allow and disallow the same tool
let invalid_config = Client::builder()
    .allowed_tools(vec!["Bash(ls)".to_string()])
    .disallowed_tools(vec!["Bash(ls)".to_string()])  // ERROR
    .build();  // Returns Error::InvalidInput

// Invalid: max_turns cannot be zero
let invalid_config = Client::builder()
    .max_turns(0)  // ERROR
    .build();  // Returns Error::InvalidInput
```

## Security Considerations

### 1. Permission Model Changes

**Old Behavior:**
- Tools were broadly allowed (e.g., "bash" allowed all commands)
- No permission prompts by default

**New Behavior:**
- Granular control over specific commands
- Permission prompts can be required
- Explicit disallow lists for dangerous operations

### 2. Recommended Security Settings

```rust
// For production/automated environments
let production_client = Client::builder()
    .allowed_tools(vec![
        // Only specific, safe commands
        ToolPermission::bash("git status").to_cli_format(),
        ToolPermission::bash("cargo check").to_cli_format(),
        ToolPermission::mcp("filesystem", "read").to_cli_format(),
    ])
    .disallowed_tools(vec![
        // Explicitly block dangerous operations
        "Bash(rm)".to_string(),
        "Bash(sudo)".to_string(),
        "Bash(chmod)".to_string(),
        "mcp__system__*".to_string(),
    ])
    .skip_permissions(true)  // Automated environment
    .max_turns(20)  // Prevent runaway conversations
    .build();

// For interactive/development environments
let development_client = Client::builder()
    .allowed_tools(vec![
        ToolPermission::All.to_cli_format(),  // Allow all tools
    ])
    .skip_permissions(false)  // Require user confirmation
    .build();
```

## Performance Considerations

### 1. Tool Permission Validation

The enhanced tool permission system includes validation that may add minimal overhead:

```rust
// This now includes validation for each tool permission
let client = Client::builder()
    .allowed_tools(vec![
        // Each of these is validated for format and security
        "Bash(command1)".to_string(),
        "Bash(command2)".to_string(),
        // ... many tools
    ])
    .build();  // Validation happens here
```

### 2. Conversation Limits

Using `max_turns` can help prevent resource exhaustion:

```rust
// Without max_turns: conversation could continue indefinitely
let unlimited_client = Client::builder().build();

// With max_turns: conversation automatically stops
let limited_client = Client::builder()
    .max_turns(10)
    .build();
```

## Testing Migration

### 1. Update Test Configurations

**Before:**
```rust
#[tokio::test]
async fn test_client_creation() {
    let client = Client::builder()
        .allowed_tools(vec!["bash".to_string()])
        .build();
    // ... test logic
}
```

**After:**
```rust
#[tokio::test]
async fn test_client_creation() {
    use claude_sdk_rs::ToolPermission;
    
    let client = Client::builder()
        .allowed_tools(vec![
            ToolPermission::bash("echo").to_cli_format()
        ])
        .max_turns(5)  // Limit test conversations
        .build();
    // ... test logic
}
```

### 2. Test New Features

```rust
#[tokio::test]
async fn test_new_features() {
    let client = Client::builder()
        .append_system_prompt("Test mode")
        .max_turns(3)
        .disallowed_tools(vec!["Bash(rm)".to_string()])
        .skip_permissions(true)
        .build();
    
    assert!(client.is_ok());
}

#[tokio::test]
async fn test_validation_errors() {
    // Test that invalid combinations are caught
    let result = Client::builder()
        .system_prompt("Base")
        .append_system_prompt("Extension")  // Should fail
        .build();
    
    assert!(result.is_err());
}
```

## Troubleshooting

### Common Migration Issues

1. **"Cannot use both system_prompt and append_system_prompt"**
   ```rust
   // Wrong:
   .system_prompt("Base")
   .append_system_prompt("Extension")
   
   // Correct:
   .append_system_prompt("Extension only")
   ```

2. **"Tool permission format error"**
   ```rust
   // Wrong:
   "Bash()"  // Empty command
   "mcp__server"  // Missing tool
   
   // Correct:
   "Bash(ls)"
   "mcp__server__tool"
   ```

3. **"Tool conflicts between allowed and disallowed"**
   ```rust
   // Wrong:
   .allowed_tools(vec!["Bash(ls)".to_string()])
   .disallowed_tools(vec!["Bash(ls)".to_string()])
   
   // Correct:
   .allowed_tools(vec!["Bash(ls)".to_string()])
   .disallowed_tools(vec!["Bash(rm)".to_string()])
   ```

### Getting Help

- Check the [examples/](examples/) directory for working code
- Review the [API documentation](https://docs.rs/claude-sdk-rs)
- File issues at [GitHub Issues](https://github.com/bredmond1019/claude-sdk-rust/issues)

## Summary

The migration to the latest version is straightforward:

1. **No breaking changes** - existing code continues to work
2. **Additive features** - new capabilities are opt-in
3. **Enhanced security** - granular permissions and validation
4. **Better control** - conversation limits and prompt extension

All new features are designed to enhance functionality while maintaining backward compatibility.