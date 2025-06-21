//! # Advanced Permissions Example
//!
//! This example demonstrates advanced tool permission features of the claude-sdk-rs SDK.
//! It shows how to:
//! - Configure allowed and disallowed tools
//! - Use the default skip permissions behavior
//! - Granular permission control
//! - Security-conscious permission settings
//! - Permission conflict handling
//!
//! Tool permissions allow you to control which tools Claude can access,
//! providing fine-grained security control for your applications.

use claude_sdk_rs::{Client, StreamFormat};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Claude SDK Advanced Permissions Example ===\n");

    // Example 1: Basic allowed and disallowed tools
    basic_tool_permissions().await?;

    // Example 2: Skip permissions behavior
    skip_permissions_examples().await?;

    // Example 3: Security-focused configurations
    security_focused_permissions().await?;

    // Example 4: Granular tool permissions
    granular_permissions_example().await?;

    // Example 5: Permission validation and conflicts
    permission_validation_examples().await?;

    println!("Advanced permissions example completed successfully!");
    Ok(())
}

/// Demonstrates basic allowed and disallowed tools configuration
async fn basic_tool_permissions() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Tool Permissions");
    println!("   Configuring allowed and disallowed tools\n");

    // Example 1a: Only allow safe tools
    println!("   Example 1a: Allowing only safe tools");
    let safe_client = Client::builder()
        .allowed_tools(vec![
            "mcp__calculator__add".to_string(),
            "mcp__calculator__subtract".to_string(),
            "mcp__search__web".to_string(),
        ])
        .stream_format(StreamFormat::Json)
        .build()?;

    let response = safe_client
        .query("Help me calculate 2 + 2")
        .send_full()
        .await?;

    println!("   Query: Help me calculate 2 + 2");
    println!("   Response: {}", response.content);
    if let Some(metadata) = response.metadata {
        println!("   Session ID: {}", metadata.session_id);
    }
    println!();

    // Example 1b: Disallow dangerous tools
    println!("   Example 1b: Disallowing dangerous tools");
    let restricted_client = Client::builder()
        .disallowed_tools(vec![
            "Bash(rm)".to_string(),
            "Bash(sudo)".to_string(),
            "mcp__filesystem__delete".to_string(),
            "mcp__network__external_request".to_string(),
        ])
        .stream_format(StreamFormat::Json)
        .build()?;

    let response2 = restricted_client
        .query("Help me with a safe calculation")
        .send_full()
        .await?;

    println!("   Query: Help me with a safe calculation");
    println!("   Response: {}", response2.content);
    println!();

    Ok(())
}

/// Demonstrates skip permissions behavior
async fn skip_permissions_examples() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Skip Permissions Behavior");
    println!("   Understanding the --dangerously-skip-permissions flag\n");

    // Example 2a: Default behavior (skip permissions enabled)
    println!("   Example 2a: Default behavior (permissions skipped)");
    let default_client = Client::builder()
        .stream_format(StreamFormat::Json)
        .build()?;

    println!("   Default skip_permissions setting: enabled");
    println!("   This means --dangerously-skip-permissions flag is added automatically");

    let response = default_client
        .query("What tools can you use?")
        .send_full()
        .await?;

    println!("   Query: What tools can you use?");
    println!("   Response: {}", response.content);
    println!();

    // Example 2b: Security-conscious mode (skip permissions disabled)
    println!("   Example 2b: Security-conscious mode (permissions required)");
    let secure_client = Client::builder()
        .skip_permissions(false) // Require permission prompts
        .stream_format(StreamFormat::Json)
        .build()?;

    println!("   Skip permissions setting: disabled");
    println!("   This means Claude will prompt for tool permissions when needed");

    let response2 = secure_client
        .query("What's a safe operation you can perform?")
        .send_full()
        .await?;

    println!("   Query: What's a safe operation you can perform?");
    println!("   Response: {}", response2.content);
    println!();

    Ok(())
}

/// Demonstrates security-focused permission configurations
async fn security_focused_permissions() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Security-Focused Permissions");
    println!("   Implementing security best practices\n");

    // Example 3a: Read-only environment
    println!("   Example 3a: Read-only environment configuration");
    let readonly_client = Client::builder()
        .allowed_tools(vec![
            "mcp__search__web".to_string(),
            "mcp__calculator__*".to_string(), // Allow all calculator tools
            "mcp__text__analyze".to_string(),
        ])
        .disallowed_tools(vec![
            "Bash(rm)".to_string(),
            "Bash(sudo)".to_string(),
            "mcp__filesystem__write".to_string(),
            "mcp__filesystem__delete".to_string(),
            "mcp__network__post".to_string(),
        ])
        .skip_permissions(false)
        .stream_format(StreamFormat::Json)
        .build()?;

    let response = readonly_client
        .query("Search for information about Rust programming language")
        .send_full()
        .await?;

    println!("   Query: Search for information about Rust programming language");
    println!("   Response: {}", response.content);
    println!();

    // Example 3b: Development environment with controlled access
    println!("   Example 3b: Development environment with controlled access");
    let dev_client = Client::builder()
        .allowed_tools(vec![
            "Bash(ls)".to_string(),
            "Bash(cat)".to_string(),
            "Bash(grep)".to_string(),
            "mcp__git__status".to_string(),
            "mcp__git__diff".to_string(),
        ])
        .disallowed_tools(vec![
            "Bash(rm)".to_string(),
            "Bash(sudo)".to_string(),
            "mcp__filesystem__delete".to_string(),
        ])
        .skip_permissions(false)
        .stream_format(StreamFormat::Json)
        .build()?;

    let response2 = dev_client
        .query("Help me examine the current directory structure")
        .send_full()
        .await?;

    println!("   Query: Help me examine the current directory structure");
    println!("   Response: {}", response2.content);
    println!();

    Ok(())
}

/// Demonstrates granular tool permissions
async fn granular_permissions_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Granular Tool Permissions");
    println!("   Fine-grained control over tool access\n");

    // Example 4a: Specific bash commands only
    println!("   Example 4a: Allowing specific bash commands only");
    let bash_limited_client = Client::builder()
        .allowed_tools(vec![
            "Bash(echo)".to_string(),
            "Bash(pwd)".to_string(),
            "Bash(date)".to_string(),
            "Bash(whoami)".to_string(),
        ])
        .disallowed_tools(vec![
            "Bash(rm)".to_string(),
            "Bash(sudo)".to_string(),
            "Bash(chmod)".to_string(),
        ])
        .stream_format(StreamFormat::Json)
        .build()?;

    let response = bash_limited_client
        .query("Show me the current date and user")
        .send_full()
        .await?;

    println!("   Query: Show me the current date and user");
    println!("   Response: {}", response.content);
    println!();

    // Example 4b: MCP server with specific tool access
    println!("   Example 4b: MCP server with specific tool restrictions");
    let mcp_specific_client = Client::builder()
        .allowed_tools(vec![
            "mcp__database__read".to_string(),
            "mcp__database__query".to_string(),
        ])
        .disallowed_tools(vec![
            "mcp__database__write".to_string(),
            "mcp__database__delete".to_string(),
            "mcp__database__admin".to_string(),
        ])
        .stream_format(StreamFormat::Json)
        .build()?;

    let response2 = mcp_specific_client
        .query("Query the database for user information")
        .send_full()
        .await?;

    println!("   Query: Query the database for user information");
    println!("   Response: {}", response2.content);
    println!();

    Ok(())
}

/// Demonstrates permission validation and conflict handling
async fn permission_validation_examples() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Permission Validation and Conflict Handling");
    println!("   Understanding how permission conflicts are handled\n");

    // Example 5a: Valid configuration
    println!("   Example 5a: Valid permission configuration");
    let valid_result = Client::builder()
        .allowed_tools(vec![
            "Bash(echo)".to_string(),
            "mcp__calculator__add".to_string(),
        ])
        .disallowed_tools(vec![
            "Bash(rm)".to_string(),
            "mcp__dangerous__delete".to_string(),
        ])
        .build();

    match valid_result {
        Ok(_) => println!("   ✓ Valid configuration accepted"),
        Err(e) => println!("   ✗ Unexpected error: {}", e),
    }

    // Example 5b: Configuration with conflicts (should fail)
    println!("   Example 5b: Configuration with tool conflicts");
    let conflict_result = Client::builder()
        .allowed_tools(vec![
            "Bash(ls)".to_string(),
            "mcp__calculator__add".to_string(),
        ])
        .disallowed_tools(vec![
            "Bash(ls)".to_string(),
            "mcp__dangerous__tool".to_string(),
        ]) // 'Bash(ls)' is in both
        .build();

    match conflict_result {
        Ok(_) => println!("   ⚠ Conflict not detected (unexpected)"),
        Err(e) => println!("   ✓ Conflict correctly detected: {}", e),
    }

    // Example 5c: Empty tool lists (valid)
    println!("   Example 5c: Empty tool lists");
    let empty_result = Client::builder()
        .allowed_tools(vec![])
        .disallowed_tools(vec![])
        .build();

    match empty_result {
        Ok(_) => println!("   ✓ Empty tool lists accepted"),
        Err(e) => println!("   ✗ Unexpected error with empty lists: {}", e),
    }
    println!();

    Ok(())
}

// Example output:
/*
=== Claude SDK Advanced Permissions Example ===

1. Basic Tool Permissions
   Configuring allowed and disallowed tools

   Example 1a: Allowing only safe tools
   Query: Help me calculate 2 + 2
   Response: I can help you with that calculation. 2 + 2 equals 4.
   Session ID: 550e8400-e29b-41d4-a716-446655440003

   Example 1b: Disallowing dangerous tools
   Query: Help me with a safe calculation
   Response: I'd be happy to help with calculations! What would you like me to compute?

2. Skip Permissions Behavior
   Understanding the --dangerously-skip-permissions flag

   Example 2a: Default behavior (permissions skipped)
   Default skip_permissions setting: enabled
   This means --dangerously-skip-permissions flag is added automatically
   Query: What tools can you use?
   Response: I have access to various tools for calculations, text analysis, and web searches...

   Example 2b: Security-conscious mode (permissions required)
   Skip permissions setting: disabled
   This means Claude will prompt for tool permissions when needed
   Query: What's a safe operation you can perform?
   Response: I can help with text analysis, basic calculations, and answering questions...

3. Security-Focused Permissions
   Implementing security best practices

   Example 3a: Read-only environment configuration
   Query: Search for information about Rust programming language
   Response: Rust is a systems programming language that emphasizes safety, speed, and concurrency...

   Example 3b: Development environment with controlled access
   Query: Help me examine the current directory structure
   Response: I can help you explore the directory structure using safe commands like ls and cat...

4. Granular Tool Permissions
   Fine-grained control over tool access

   Example 4a: Allowing specific bash commands only
   Query: Show me the current date and user
   Response: The current date is 2024-01-15 and the current user is developer.

   Example 4b: MCP server with specific tool restrictions
   Query: Query the database for user information
   Response: I can help query the database for user information using read-only operations...

5. Permission Validation and Conflict Handling
   Understanding how permission conflicts are handled

   Example 5a: Valid permission configuration
   ✓ Valid configuration accepted

   Example 5b: Configuration with tool conflicts
   ✓ Conflict correctly detected: Tool 'bash' cannot be both allowed and disallowed

   Example 5c: Empty tool lists
   ✓ Empty tool lists accepted

Advanced permissions example completed successfully!
*/
