//! # Example 04: Tool Integration
//!
//! This example demonstrates tool integration with the `claude-sdk-rs` SDK.
//! It shows how to:
//! - Configure allowed tools
//! - Use MCP and bash tools
//! - Handle tool results
//! - Build tool-enabled applications

use claude_sdk_rs::{Client, ToolPermission};

#[tokio::main]
async fn main() -> claude_sdk_rs::Result<()> {
    println!("=== Claude AI Tool Integration Example ===\n");

    // Example 1: Basic tool usage
    basic_tools().await?;

    // Example 2: File operations
    file_operations().await?;

    // Example 3: Development workflow
    dev_workflow().await?;

    // Example 4: Code review assistant
    code_review_assistant().await?;

    // Show helper functions
    println!("\n=== Helper Function Examples ===");
    let _readonly_client = safe_patterns::create_readonly_client();
    println!("Created read-only client for safe operations");

    let _dev_client = safe_patterns::create_dev_client();
    println!("Created development client with full tool access");

    Ok(())
}

/// Demonstrates basic tool usage
async fn basic_tools() -> claude_sdk_rs::Result<()> {
    println!("1. Basic Tool Usage");
    println!("   Using simple bash commands\n");

    let client = Client::builder()
        .timeout_secs(120) // 2 minute timeout for tool operations
        .allowed_tools(vec![
            ToolPermission::bash("date").to_cli_format(),
            ToolPermission::bash("whoami").to_cli_format(),
            ToolPermission::bash("pwd").to_cli_format(),
        ])
        .build()?;

    let response = client
        .query("What's the current date, who am I, and what directory am I in?")
        .send()
        .await?;

    println!("   Response:\n{}", indent_text(&response, 3));

    // Add delay to avoid rate limiting
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    Ok(())
}

/// Shows file operations with tools
async fn file_operations() -> claude_sdk_rs::Result<()> {
    println!("\n2. File Operations");
    println!("   Reading and analyzing files\n");

    let client = Client::builder()
        .timeout_secs(120) // 2 minute timeout for tool operations
        .system_prompt("You are a code analysis assistant")
        .allowed_tools(vec![
            ToolPermission::mcp("filesystem", "read").to_cli_format(),
            ToolPermission::bash("find . -name '*.rs' -type f | head -5").to_cli_format(),
        ])
        .build()?;

    let response = client
        .query("Find some Rust files and tell me what kind of project this is")
        .send()
        .await?;

    println!("   Analysis:\n{}", indent_text(&response, 3));

    // Add delay to avoid rate limiting
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    Ok(())
}

/// Demonstrates a development workflow
async fn dev_workflow() -> claude_sdk_rs::Result<()> {
    println!("\n3. Development Workflow");
    println!("   Using tools for development tasks\n");

    let client = Client::builder()
        .timeout_secs(180) // 3 minute timeout for longer operations like cargo check
        .system_prompt("You are a Rust development assistant")
        .allowed_tools(vec![
            // Version control
            ToolPermission::bash("git").to_cli_format(),
            // Build tools
            ToolPermission::bash("cargo").to_cli_format(),
            // File access
            ToolPermission::mcp("filesystem", "read").to_cli_format(),
        ])
        .build()?;

    // Check project status
    println!("   Checking project status...");
    let status_response = client
        .query("Check the git status and current branch")
        .send()
        .await?;

    println!("   Status:\n{}", indent_text(&status_response, 3));

    // Add delay between operations
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Analyze code
    println!("\n   Analyzing code structure...");
    let analysis_response = client
        .query("Run cargo check --workspace and tell me if there are any issues")
        .send()
        .await?;

    println!("   Analysis:\n{}", indent_text(&analysis_response, 3));

    Ok(())
}

/// Example: Building a code review assistant
async fn code_review_assistant() -> claude_sdk_rs::Result<()> {
    println!("\n4. Code Review Assistant");
    println!("   Building a tool-enabled code review helper\n");

    let client = Client::builder()
        .system_prompt(
            "You are a code review assistant. Analyze code for:\n\
             - Best practices\n\
             - Potential bugs\n\
             - Performance issues\n\
             - Security concerns",
        )
        .allowed_tools(vec![
            // Code analysis
            ToolPermission::bash("cargo clippy -- -W clippy::all").to_cli_format(),
            ToolPermission::bash("cargo fmt -- --check").to_cli_format(),
            // File access
            ToolPermission::mcp("filesystem", "read").to_cli_format(),
            // Git integration
            ToolPermission::bash("git diff").to_cli_format(),
        ])
        .build()?;

    let response = client
        .query("Review the recent changes in this project")
        .send()
        .await?;

    println!("   Code Review:\n{}", indent_text(&response, 3));

    Ok(())
}

/// Helper function to indent text
fn indent_text(text: &str, spaces: usize) -> String {
    let indent = " ".repeat(spaces);
    text.lines()
        .map(|line| format!("{}{}", indent, line))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Example: Safe tool execution patterns
mod safe_patterns {
    use claude_sdk_rs::{Client, ToolPermission};

    /// Only allow read-only operations
    pub fn create_readonly_client() -> Client {
        Client::builder()
            .allowed_tools(vec![
                // Read-only filesystem
                ToolPermission::mcp("filesystem", "read").to_cli_format(),
                // Safe bash commands
                ToolPermission::bash("ls").to_cli_format(),
                ToolPermission::bash("cat").to_cli_format(),
                ToolPermission::bash("grep").to_cli_format(),
                ToolPermission::bash("find").to_cli_format(),
                // Git read operations
                ToolPermission::bash("git status").to_cli_format(),
                ToolPermission::bash("git log").to_cli_format(),
                ToolPermission::bash("git diff").to_cli_format(),
            ])
            .build()
            .expect("Failed to build client")
    }

    /// Development environment with write access
    pub fn create_dev_client() -> Client {
        Client::builder()
            .system_prompt(
                "You are a development assistant. Be careful with destructive operations.",
            )
            .allowed_tools(vec![
                // Full filesystem access
                ToolPermission::mcp("filesystem", "*").to_cli_format(),
                // Development commands
                ToolPermission::bash("cargo build").to_cli_format(),
                ToolPermission::bash("cargo test").to_cli_format(),
                ToolPermission::bash("cargo run").to_cli_format(),
                // Git operations (be careful!)
                ToolPermission::bash("git add").to_cli_format(),
                ToolPermission::bash("git commit").to_cli_format(),
            ])
            .build()
            .expect("Failed to build client")
    }
}

// Example output:
/*
=== Claude AI Tool Integration Example ===

1. Basic Tool Usage
   Using simple bash commands

   Response:
   I'll help you get that information using the available tools.

   The current date is: Mon Jan 15 14:30:22 PST 2024
   You are logged in as: brandon
   Current directory: /Users/brandon/Documents/Projects/claude-sdk-rs/claude-interactive

2. File Operations
   Reading and analyzing files

   Analysis:
   I found several Rust files in this project. After examining them, this appears to be a Rust workspace project called "claude-sdk-rs" that provides:

   1. A core SDK for interacting with Claude AI (`claude-sdk-rs` crate)
   2. An interactive CLI tool (`claude-sdk-rs-interactive`)
   3. Core types and configurations (`claude-sdk-rs-core`)
   4. Runtime implementation (`claude-sdk-rs-runtime`)
   5. Additional features like MCP support and macros

   The project structure follows Rust workspace conventions with multiple crates providing different functionality for working with Claude AI programmatically.

3. Development Workflow
   Using tools for development tasks

   Checking project status...
   Status:
   You're currently on the `claude-interactive` branch. There is one modified file:
   - `claude-ai-interactive/src/analytics/analytics_test.rs` (staged for commit)

   The working directory is otherwise clean.

   Analyzing code structure...
   Analysis:
   I ran `cargo check` on the workspace and the good news is that all crates compile successfully! There are a few warnings about unused imports and dead code, but no errors:

   - An unused import in `cost/tracker.rs`
   - Some unused fields in the analytics module
   - A few unused methods that might be used in future features

   Overall, the codebase is in good shape and ready for development.
*/
