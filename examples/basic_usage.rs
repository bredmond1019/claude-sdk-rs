//! # Basic Usage Example
//!
//! This example demonstrates the fundamental usage of the claude-sdk-rs SDK.
//! It shows how to:
//! - Initialize a client with the default configuration
//! - Send simple queries
//! - Handle responses
//! - Work with different response formats
//!
//! This uses the SDK functionality from the main crate.

use claude_sdk_rs::{Client, StreamFormat};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Claude SDK Basic Usage Example ===\n");

    // Example 1: Simple text query with default configuration
    simple_query().await?;

    // Example 2: Query with custom configuration
    configured_query().await?;

    // Example 3: Different response formats
    response_formats().await?;

    println!("Basic usage example completed successfully!");
    Ok(())
}

/// Demonstrates the simplest way to query Claude using the SDK
async fn simple_query() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Simple Query Example");
    println!("   Using default configuration for a basic question\n");

    // Create client with default configuration
    let client = Client::builder().build()?;

    // Send a simple query
    let response = client
        .query("What is the capital of France? Please keep the answer brief.")
        .send()
        .await?;

    println!("   Question: What is the capital of France?");
    println!("   Answer: {}\n", response);

    Ok(())
}

/// Shows how to configure the client with custom settings
async fn configured_query() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Configured Query Example");
    println!("   Using custom model and system prompt\n");

    // Build client with custom configuration
    let client = Client::builder()
        .system_prompt("You are a helpful geography teacher. Keep answers concise and educational.")
        .timeout_secs(30)
        .build()?;

    let response = client
        .query("Name three major rivers in Europe and their countries")
        .send()
        .await?;

    println!("   Question: Name three major rivers in Europe and their countries");
    println!("   Answer: {}\n", response);

    Ok(())
}

/// Demonstrates different response formats
async fn response_formats() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Response Format Examples");
    println!("   Demonstrating Text vs JSON formats\n");

    // Text format - raw output
    let text_client = Client::builder()
        .stream_format(StreamFormat::Text)
        .build()?;

    println!("   a) Text Format (raw CLI output):");
    let text_response = text_client
        .query("List three programming languages")
        .send()
        .await?;
    println!("   {}\n", text_response);

    // JSON format - structured data
    let json_client = Client::builder()
        .stream_format(StreamFormat::Json)
        .build()?;

    println!("   b) JSON Format (structured response):");
    let json_response = json_client
        .query("List three programming languages")
        .send_full()
        .await?;

    println!("   Content: {}", json_response.content);
    if let Some(metadata) = json_response.metadata {
        println!("   Session ID: {}", metadata.session_id);
        if let Some(cost) = metadata.cost_usd {
            println!("   Cost: ${:.6}", cost);
        }
    }
    println!();

    Ok(())
}

// Example output:
/*
=== Claude SDK Basic Usage Example ===

1. Simple Query Example
   Using default configuration for a basic question

   Question: What is the capital of France?
   Answer: The capital of France is Paris.

2. Configured Query Example
   Using custom model and system prompt

   Question: Name three major rivers in Europe and their countries
   Answer: Three major rivers in Europe are:
   1. The Danube - flows through Germany, Austria, Hungary, and other countries
   2. The Rhine - flows through Switzerland, Germany, and the Netherlands
   3. The Volga - the longest river in Europe, flows through Russia

3. Response Format Examples
   Demonstrating Text vs JSON formats

   a) Text Format (raw CLI output):
   Here are three popular programming languages:
   1. Python
   2. JavaScript
   3. Rust

   b) JSON Format (structured response):
   Content: Here are three popular programming languages:
   1. Python
   2. JavaScript
   3. Rust
   Session ID: 550e8400-e29b-41d4-a716-446655440000
   Cost: $0.000234

Basic usage example completed successfully!
*/
