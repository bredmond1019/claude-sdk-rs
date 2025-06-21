//! # Example 01: Basic SDK Usage
//!
//! This example demonstrates the fundamental usage of the `claude-sdk-rs` SDK.
//! It shows how to:
//! - Initialize a client
//! - Send simple queries
//! - Handle responses
//! - Work with different response formats
//!
//! This uses ONLY the `claude-sdk-rs` crate, not `claude-interactive`.

use claude_sdk_rs::{Client, StreamFormat};

#[tokio::main]
async fn main() -> claude_sdk_rs::Result<()> {
    println!("=== Claude AI Basic SDK Example ===\n");

    // Example 1: Simple text query
    simple_query().await?;

    // Example 2: Query with custom configuration
    configured_query().await?;

    // Example 3: Full response with metadata
    full_response_query().await?;

    // Example 4: Different stream formats
    stream_format_comparison().await?;

    Ok(())
}

/// Demonstrates the simplest way to query Claude
async fn simple_query() -> claude_sdk_rs::Result<()> {
    println!("1. Simple Query Example");
    println!("   Using default configuration for a basic question\n");

    let client = Client::builder().build()?;

    let response = client
        .query("What is the capital of France?")
        .send()
        .await?;

    println!("   Question: What is the capital of France?");
    println!("   Answer: {}\n", response);

    Ok(())
}

/// Shows how to configure the client with custom settings
async fn configured_query() -> claude_sdk_rs::Result<()> {
    println!("2. Configured Query Example");
    println!("   Using custom model and system prompt\n");

    let client = Client::builder()
        .model("claude-3-sonnet-20240229")
        .system_prompt("You are a helpful geography teacher. Keep answers concise.")
        .timeout_secs(30)
        .build()?;

    let response = client
        .query("Name three major rivers in Europe")
        .send()
        .await?;

    println!("   Question: Name three major rivers in Europe");
    println!("   Answer: {}\n", response);

    Ok(())
}

/// Demonstrates getting full response metadata
async fn full_response_query() -> claude_sdk_rs::Result<()> {
    println!("3. Full Response with Metadata");
    println!("   Accessing cost, tokens, and timing information\n");

    let client = Client::builder()
        .stream_format(StreamFormat::Json)
        .build()?;

    let response = client
        .query("Explain quantum computing in one paragraph")
        .send_full()
        .await?;

    println!("   Question: Explain quantum computing in one paragraph");
    println!("   Answer: {}", response.content);

    if let Some(metadata) = response.metadata {
        println!("\n   Metadata:");
        if let Some(cost) = metadata.cost_usd {
            println!("   - Cost: ${:.6}", cost);
        }
        if let Some(duration) = metadata.duration_ms {
            println!("   - Duration: {}ms", duration);
        }
        if let Some(tokens) = metadata.tokens_used {
            if let (Some(input), Some(output)) = (tokens.input_tokens, tokens.output_tokens) {
                let total = input + output;
                println!(
                    "   - Tokens: {} input, {} output, {} total",
                    input, output, total
                );
            }
        }
    }

    // Note: ClaudeResponse doesn't have stats field - that's in streaming messages

    println!();
    Ok(())
}

/// Compares different stream format behaviors
async fn stream_format_comparison() -> claude_sdk_rs::Result<()> {
    println!("4. Stream Format Comparison");
    println!("   Demonstrating Text vs Json formats\n");

    // Text format - raw output
    let text_client = Client::builder()
        .stream_format(StreamFormat::Text)
        .build()?;

    println!("   a) Text Format (raw CLI output):");
    let text_response = text_client.query("List three colors").send().await?;
    println!("   {}\n", text_response);

    // JSON format - structured data
    let json_client = Client::builder()
        .stream_format(StreamFormat::Json)
        .build()?;

    println!("   b) JSON Format (structured response):");
    let json_response = json_client.query("List three colors").send_full().await?;

    println!("   Content: {}", json_response.content);
    if let Some(metadata) = json_response.metadata {
        println!("   Session ID: {}", metadata.session_id);
    }

    Ok(())
}

// Example output:
/*
=== Claude AI Basic SDK Example ===

1. Simple Query Example
   Using default configuration for a basic question

   Question: What is the capital of France?
   Answer: The capital of France is Paris.

2. Configured Query Example
   Using custom model and system prompt

   Question: Name three major rivers in Europe
   Answer: Three major rivers in Europe are: the Danube, the Rhine, and the Volga.

3. Full Response with Metadata
   Accessing cost, tokens, and timing information

   Question: Explain quantum computing in one paragraph
   Answer: Quantum computing is a revolutionary computing paradigm that harnesses the principles of quantum mechanics to process information in fundamentally different ways than classical computers. Unlike traditional computers that use bits representing either 0 or 1, quantum computers use quantum bits (qubits) that can exist in superposition, representing both 0 and 1 simultaneously. This property, along with quantum entanglement and interference, allows quantum computers to perform certain calculations exponentially faster than classical computers, particularly for problems involving optimization, cryptography, and simulation of quantum systems. While still in early stages of development, quantum computing holds immense promise for solving complex problems in fields ranging from drug discovery and materials science to artificial intelligence and financial modeling.

   Metadata:
   - Cost: $0.000834
   - Duration: 2341ms
   - Tokens: 125 input, 142 output, 267 total

   Conversation Stats:
   - Total messages: 2
   - Total cost: $0.000834
   - Total duration: 2341ms

4. Stream Format Comparison
   Demonstrating Text vs Json formats

   a) Text Format (raw CLI output):
   Here are three colors:
   1. Blue
   2. Red
   3. Green

   b) JSON Format (structured response):
   Content: Here are three colors:
   1. Blue
   2. Red
   3. Green
   Session ID: 550e8400-e29b-41d4-a716-446655440000
*/
