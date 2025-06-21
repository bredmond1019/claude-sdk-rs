//! # Error Handling Example
//!
//! This example demonstrates comprehensive error handling patterns
//! when using the claude-sdk-rs SDK. It shows how to:
//! - Handle different types of errors gracefully
//! - Implement retry logic for transient failures
//! - Validate configurations before use
//! - Provide meaningful error messages to users
//! - Recover from network and timeout issues
//!
//! Proper error handling is crucial for building reliable applications.

use claude_sdk_rs::{Client, Error as ClaudeError, Message, StreamFormat};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Claude SDK Error Handling Example ===\n");

    // Example 1: Basic error handling
    basic_error_handling().await;

    // Example 2: Configuration validation
    configuration_validation().await;

    // Example 3: Timeout handling
    timeout_handling().await;

    // Example 4: Retry logic for transient failures
    retry_logic().await;

    // Example 5: Streaming error recovery
    streaming_error_recovery().await;

    println!("Error handling example completed successfully!");
    Ok(())
}

/// Demonstrates basic error handling patterns
async fn basic_error_handling() {
    println!("1. Basic Error Handling");
    println!("   Handling common error scenarios gracefully\n");

    // Test with a potentially problematic query
    let client = Client::builder().build().expect("Failed to build client");

    println!("   Testing with normal query:");
    match client.query("What is 2 + 2?").send().await {
        Ok(response) => println!("   ✓ Success: {}", response),
        Err(e) => println!("   ✗ Error: {}", e),
    }

    println!("\n   Testing with very long query (might timeout):");
    let long_query = "x".repeat(10000); // Very long query that might cause issues

    match client.query(&long_query).send().await {
        Ok(response) => println!("   ✓ Success: {} characters", response.len()),
        Err(e) => {
            println!("   ✗ Error occurred: {}", e);
            // Classify the error type
            match &e {
                _ if e.to_string().contains("timeout") => {
                    println!("   → This appears to be a timeout error");
                }
                _ if e.to_string().contains("process") => {
                    println!("   → This appears to be a process execution error");
                }
                _ => {
                    println!("   → This is another type of error");
                }
            }
        }
    }
    println!();
}

/// Demonstrates configuration validation
async fn configuration_validation() {
    println!("2. Configuration Validation");
    println!("   Validating configurations before use\n");

    // Test various configuration scenarios
    let configs = vec![("Valid config", 30), ("Zero timeout", 0), ("Large timeout", 3600)];

    for (description, timeout) in configs {
        println!("   Testing {}: ", description);

        // Try to build client with the timeout
        match Client::builder().timeout_secs(timeout).build() {
            Ok(client) => match client.query("Hello").send().await {
                Ok(_) => println!("   ✓ Configuration works"),
                Err(e) => {
                    println!("   ✗ Configuration issue: {}", e);
                    provide_config_suggestions(&e);
                }
            },
            Err(e) => {
                println!("   ✗ Failed to build client: {}", e);
                provide_config_suggestions(&e);
            }
        }
    }
    println!();
}

/// Demonstrates timeout handling with different timeout values
async fn timeout_handling() {
    println!("3. Timeout Handling");
    println!("   Testing different timeout scenarios\n");

    // Test with very short timeout (likely to fail)
    println!("   Testing with 1-second timeout:");
    let client = Client::builder()
        .timeout_secs(1)
        .build()
        .expect("Failed to build client");

    match client
        .query("Write a detailed essay about the history of computers (this might take a while)")
        .send()
        .await
    {
        Ok(response) => println!("   ✓ Completed within timeout: {} chars", response.len()),
        Err(e) => {
            println!("   ✗ Timeout occurred: {}", e);
            println!("   → Consider increasing timeout for complex queries");
        }
    }

    // Test with reasonable timeout
    println!("\n   Testing with 30-second timeout:");
    let client = Client::builder()
        .timeout_secs(30)
        .build()
        .expect("Failed to build client");

    match client.query("What is the capital of Japan?").send().await {
        Ok(response) => println!("   ✓ Completed successfully: {}", response),
        Err(e) => println!("   ✗ Error: {}", e),
    }
    println!();
}

/// Demonstrates retry logic for handling transient failures
async fn retry_logic() {
    println!("4. Retry Logic");
    println!("   Implementing automatic retry for transient failures\n");

    let client = Client::builder().build().expect("Failed to build client");
    let query = "What is machine learning?";

    println!("   Attempting query with retry logic:");

    match retry_with_backoff(&client, query, 3).await {
        Ok(response) => {
            println!("   ✓ Success after retries: {}", response);
        }
        Err(e) => {
            println!("   ✗ Failed after all retries: {}", e);
        }
    }
    println!();
}

/// Demonstrates error recovery in streaming scenarios
async fn streaming_error_recovery() {
    println!("5. Streaming Error Recovery");
    println!("   Handling errors in streaming responses\n");

    let client = Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .build()
        .expect("Failed to build client");

    println!("   Starting stream with error recovery:");

    match client
        .query("Tell me about the solar system")
        .stream()
        .await
    {
        Ok(mut stream) => {
            use futures::StreamExt;
            let mut content = String::new();
            let mut error_count = 0;

            while let Some(message) = stream.next().await {
                match message {
                    Ok(msg) => match msg {
                        Message::Assistant { content: text, .. } => {
                            content.push_str(&text);
                        }
                        Message::Result { .. } => {
                            break;
                        }
                        _ => {}
                    },
                    Err(e) => {
                        error_count += 1;
                        println!("   ⚠ Stream error {}: {}", error_count, e);

                        if error_count >= 3 {
                            println!("   ✗ Too many stream errors, stopping");
                            break;
                        }

                        // Brief pause before continuing
                        sleep(Duration::from_millis(100)).await;
                    }
                }
            }

            if !content.is_empty() {
                println!("   ✓ Stream completed with {} characters", content.len());
                if error_count > 0 {
                    println!("   → Recovered from {} stream errors", error_count);
                }
            }
        }
        Err(e) => {
            println!("   ✗ Failed to start stream: {}", e);
        }
    }
    println!();
}

/// Helper function to implement retry logic with exponential backoff
async fn retry_with_backoff(
    client: &Client,
    query: &str,
    max_retries: u32,
) -> Result<String, ClaudeError> {
    let mut last_error = None;

    for attempt in 1..=max_retries {
        println!("   → Attempt {} of {}", attempt, max_retries);

        match client.query(query).send().await {
            Ok(response) => {
                if attempt > 1 {
                    println!("   → Succeeded on retry!");
                }
                return Ok(response);
            }
            Err(e) => {
                println!("   → Attempt {} failed: {}", attempt, e);
                last_error = Some(e);

                if attempt < max_retries {
                    let delay = Duration::from_millis(1000 * (1 << (attempt - 1))); // Exponential backoff
                    println!("   → Waiting {:?} before retry...", delay);
                    sleep(delay).await;
                }
            }
        }
    }

    Err(last_error.unwrap())
}

/// Helper function to provide configuration suggestions based on errors
fn provide_config_suggestions(error: &ClaudeError) {
    let error_msg = error.to_string().to_lowercase();

    if error_msg.contains("timeout") {
        println!("   → Suggestion: Increase timeout_secs in your Config");
    } else if error_msg.contains("process") {
        println!("   → Suggestion: Check if Claude Code CLI is installed and authenticated");
    } else if error_msg.contains("binary") {
        println!("   → Suggestion: Install Claude Code CLI from https://claude.ai/code");
    } else if error_msg.contains("authentication") {
        println!("   → Suggestion: Run 'claude login' to authenticate");
    } else {
        println!("   → Suggestion: Check network connection and try again");
    }
}

// Example output:
/*
=== Claude SDK Error Handling Example ===

1. Basic Error Handling
   Handling common error scenarios gracefully

   Testing with normal query:
   ✓ Success: 2 + 2 equals 4.

   Testing with very long query (might timeout):
   ✗ Error occurred: Request timeout after 30 seconds
   → This appears to be a timeout error

2. Configuration Validation
   Validating configurations before use

   Testing Valid config:
   ✓ Configuration works
   Testing Zero timeout:
   ✗ Configuration issue: Invalid timeout value
   → Suggestion: Increase timeout_secs in your Config
   Testing Large timeout:
   ✓ Configuration works

3. Timeout Handling
   Testing different timeout scenarios

   Testing with 1-second timeout:
   ✗ Timeout occurred: Request timeout after 1 seconds
   → Consider increasing timeout for complex queries

   Testing with 30-second timeout:
   ✓ Completed successfully: The capital of Japan is Tokyo.

4. Retry Logic
   Implementing automatic retry for transient failures

   Attempting query with retry logic:
   → Attempt 1 of 3
   ✓ Success after retries: Machine learning is a subset of artificial intelligence...

5. Streaming Error Recovery
   Handling errors in streaming responses

   Starting stream with error recovery:
   ✓ Stream completed with 1234 characters

Error handling example completed successfully!
*/
