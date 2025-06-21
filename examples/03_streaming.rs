//! # Example 03: Streaming Responses
//!
//! This example demonstrates streaming responses with the `claude-sdk-rs` SDK.
//! It shows how to:
//! - Enable streaming for real-time responses
//! - Process different message types
//! - Handle tool calls in streams
//! - Track costs and tokens in real-time

use claude_sdk_rs::{Client, Message, StreamFormat};
use futures::StreamExt;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> claude_sdk_rs::Result<()> {
    println!("=== Claude AI Streaming Example ===\n");

    // Example 1: Basic streaming
    basic_streaming().await?;

    // Example 2: Streaming with metadata
    streaming_with_metadata().await?;

    // Example 3: Tool usage in streams
    streaming_with_tools().await?;

    // Example 4: Streaming with progress
    streaming_with_progress().await?;

    Ok(())
}

/// Demonstrates basic streaming responses
async fn basic_streaming() -> claude_sdk_rs::Result<()> {
    println!("1. Basic Streaming");
    println!("   Real-time response as it's generated\n");

    let client = Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .build()?;

    print!("   Claude> ");
    use std::io::{self, Write};
    io::stdout().flush().unwrap();

    let mut stream = client
        .query("Write a haiku about programming")
        .stream()
        .await?;

    while let Some(message) = stream.next().await {
        match message? {
            Message::Assistant { content, .. } => {
                print!("{}", content);
                io::stdout().flush().unwrap();
            }
            Message::Result { .. } => {
                println!("\n\n   ✓ Stream complete");
            }
            _ => {}
        }
    }

    Ok(())
}

/// Shows streaming with detailed metadata
async fn streaming_with_metadata() -> claude_sdk_rs::Result<()> {
    println!("\n2. Streaming with Metadata");
    println!("   Tracking costs and tokens in real-time\n");

    let client = Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .build()?;

    let mut stream = client
        .query("Explain the concept of ownership in Rust in 3 sentences")
        .stream()
        .await?;

    let mut _total_tokens = 0u64;
    let mut _total_cost = 0.0;

    println!("   Response:");
    while let Some(message) = stream.next().await {
        match message? {
            Message::Assistant { content, meta } => {
                print!("{}", content);
                io::stdout().flush().unwrap();

                // Track tokens and cost
                if let Some(tokens) = &meta.tokens_used {
                    _total_tokens = tokens.total;
                }
                if let Some(cost) = meta.cost_usd {
                    _total_cost = cost;
                }
            }
            Message::Result { stats, .. } => {
                println!("\n\n   Stream Statistics:");
                println!("   - Messages: {}", stats.total_messages);
                println!("   - Total cost: ${:.6}", stats.total_cost_usd);
                println!("   - Total tokens: {}", stats.total_tokens.total);
                println!("   - Duration: {}ms", stats.total_duration_ms);
            }
            _ => {}
        }
    }

    Ok(())
}

/// Demonstrates streaming with tool usage
async fn streaming_with_tools() -> claude_sdk_rs::Result<()> {
    println!("\n3. Streaming with Tools");
    println!("   Showing tool execution in real-time\n");

    let client = Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .allowed_tools(vec![
            "mcp__filesystem__read".to_string(),
            "bash:ls".to_string(),
        ])
        .build()?;

    let mut stream = client
        .query("List the files in the current directory and explain what you see")
        .stream()
        .await?;

    println!("   Streaming response with tools:\n");

    while let Some(message) = stream.next().await {
        match message? {
            Message::Assistant { content, .. } => {
                print!("{}", content);
                io::stdout().flush().unwrap();
            }
            Message::Tool {
                name, parameters, ..
            } => {
                println!("\n   🔧 Tool: {} with params: {}", name, parameters);
            }
            Message::ToolResult { tool_name, .. } => {
                println!("   ✅ Tool {} completed", tool_name);
                // Could show truncated result if needed
                print!("\n   ");
            }
            Message::Result { stats, .. } => {
                println!(
                    "\n\n   Completed with {} total messages",
                    stats.total_messages
                );
            }
            _ => {}
        }
    }

    Ok(())
}

/// Helper to show progress indicator
async fn streaming_with_progress() -> claude_sdk_rs::Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tokio::time::{sleep, Duration};

    let client = Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .build()?;

    let is_streaming = Arc::new(AtomicBool::new(true));
    let is_streaming_clone = is_streaming.clone();

    // Spawn progress indicator
    tokio::spawn(async move {
        let spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        let mut i = 0;

        while is_streaming_clone.load(Ordering::Relaxed) {
            print!("\r{} Thinking...", spinner[i]);
            io::stdout().flush().unwrap();
            i = (i + 1) % spinner.len();
            sleep(Duration::from_millis(100)).await;
        }
        print!("\r                    \r"); // Clear spinner
    });

    let mut stream = client
        .query("What are the main principles of functional programming?")
        .stream()
        .await?;

    // Wait a bit to show spinner
    sleep(Duration::from_millis(500)).await;

    while let Some(message) = stream.next().await {
        match message? {
            Message::Assistant { content, .. } => {
                if is_streaming.load(Ordering::Relaxed) {
                    is_streaming.store(false, Ordering::Relaxed);
                    sleep(Duration::from_millis(150)).await; // Let spinner clear
                }
                print!("{}", content);
                io::stdout().flush().unwrap();
            }
            Message::Result { .. } => break,
            _ => {}
        }
    }

    println!();
    Ok(())
}

// Example output:
/*
=== Claude AI Streaming Example ===

1. Basic Streaming
   Real-time response as it's generated

   Claude> Code flows like water,
   Logic branches endlessly—
   Bugs bloom in spring rain.

   ✓ Stream complete

2. Streaming with Metadata
   Tracking costs and tokens in real-time

   Response:
   Ownership in Rust is a memory management system where each value has a single owner that is responsible for cleaning it up. When ownership is transferred to another variable, the original variable can no longer access the value, preventing double-free errors. This system enables memory safety without garbage collection by enforcing strict rules at compile time.

   Stream Statistics:
   - Messages: 2
   - Total cost: $0.000234
   - Total tokens: 89
   - Duration: 1523ms

3. Streaming with Tools
   Showing tool execution in real-time

   Streaming response with tools:

   🔧 Tool: bash:ls with params: {}
   ✅ Tool bash:ls completed

   I can see the following files in the current directory:

   - `Cargo.toml` - The workspace configuration file
   - `README.md` - Project documentation
   - `src/` - Source code directory
   - `examples/` - Example applications
   - `target/` - Build artifacts (compiled code)

   This appears to be a Rust project with a standard Cargo workspace structure. The presence of an examples directory suggests this is a library with usage examples.

   Completed with 4 total messages
*/
