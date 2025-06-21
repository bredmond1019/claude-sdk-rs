//! # Configuration Example
//!
//! This example demonstrates various configuration options available
//! in the claude-sdk-rs SDK. It shows how to:
//! - Use different configuration patterns
//! - Set model preferences and system prompts
//! - Configure timeouts and stream formats
//! - Create reusable configuration templates
//! - Validate and test configurations
//!
//! Proper configuration allows you to customize the SDK behavior
//! for different use cases and requirements.

use claude_sdk_rs::{Client, Message, StreamFormat};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Claude SDK Configuration Example ===\n");

    // Example 1: Default configuration
    default_configuration().await?;

    // Example 2: Builder pattern configuration
    builder_pattern().await?;

    // Example 3: Model-specific configurations
    model_configurations().await?;

    // Example 4: System prompt configurations
    system_prompt_configurations().await?;

    // Example 5: Stream format configurations
    stream_format_configurations().await?;

    // Example 6: Timeout configurations
    timeout_configurations().await?;

    // Example 7: Configuration templates
    configuration_templates().await?;

    println!("Configuration example completed successfully!");
    Ok(())
}

/// Demonstrates default configuration usage
async fn default_configuration() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Default Configuration");
    println!("   Using Config::default() for quick setup\n");

    // Default configuration uses sensible defaults
    let client = Client::builder().build()?;

    println!("   Default config properties:");
    println!("   - Uses default Claude model");
    println!("   - 30-second timeout");
    println!("   - Text stream format");
    println!("   - No custom system prompt");

    let response = client
        .query("What's the default behavior of this SDK?")
        .send()
        .await?;

    println!("\n   Response: {}\n", response);
    Ok(())
}

/// Demonstrates the builder pattern for configuration
async fn builder_pattern() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Builder Pattern Configuration");
    println!("   Using Config::builder() for custom settings\n");

    // Build custom configuration step by step
    let client = Client::builder()
        .model("claude-3-opus-20240229")
        .system_prompt("You are a helpful coding assistant. Provide clear, concise answers.")
        .timeout_secs(45)
        .stream_format(StreamFormat::Json)
        .build()?;

    println!("   Custom config properties:");
    println!("   - Model: claude-3-opus-20240229");
    println!("   - System prompt: Custom coding assistant");
    println!("   - Timeout: 45 seconds");
    println!("   - Stream format: JSON");

    let response = client
        .query("How do I handle errors in Rust?")
        .send_full()
        .await?;

    println!("\n   Response: {}", response.content);
    if let Some(metadata) = response.metadata {
        println!("   Session ID: {}", metadata.session_id);
    }
    println!();

    Ok(())
}

/// Demonstrates model-specific configurations
async fn model_configurations() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Model-Specific Configurations");
    println!("   Configuring different Claude models\n");

    let models = vec![
        ("claude-3-opus-20240229", "Most capable model"),
        ("claude-3-sonnet-20240229", "Balanced performance"),
        ("claude-3-haiku-20240307", "Fastest model"),
    ];

    for (model, description) in models {
        println!("   Testing {}: {}", model, description);

        let client = Client::builder()
            .model(model)
            .system_prompt(&format!("You are using {}. Keep responses brief.", model))
            .build()?;

        match client.query("What model are you using?").send().await {
            Ok(response) => {
                println!("   Response: {}", response.trim());
            }
            Err(e) => {
                println!("   Error with {}: {}", model, e);
            }
        }
        println!();
    }

    Ok(())
}

/// Demonstrates system prompt configurations
async fn system_prompt_configurations() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. System Prompt Configurations");
    println!("   Using different system prompts for specialized behavior\n");

    let prompts = vec![
        (
            "Teacher",
            "You are a patient teacher. Explain concepts clearly with examples.",
            "Explain recursion in programming",
        ),
        (
            "Poet",
            "You are a creative poet. Respond in verse and metaphor.",
            "Describe a beautiful sunset",
        ),
        (
            "Technical Writer",
            "You are a technical writer. Use precise, formal language.",
            "Describe how HTTP works",
        ),
    ];

    for (role, system_prompt, query) in prompts {
        println!("   {} Role:", role);
        println!("   System prompt: {}", system_prompt);

        let client = Client::builder().system_prompt(system_prompt).build()?;

        let response = client.query(query).send().await?;
        println!("   Query: {}", query);
        println!("   Response: {}\n", response);
    }

    Ok(())
}

/// Demonstrates stream format configurations
async fn stream_format_configurations() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Stream Format Configurations");
    println!("   Comparing different stream formats\n");

    let query = "List three benefits of exercise";

    // Text format
    println!("   a) Text Format:");
    let text_client = Client::builder()
        .stream_format(StreamFormat::Text)
        .build()?;
    let text_response = text_client.query(query).send().await?;
    println!("   {}\n", text_response);

    // JSON format
    println!("   b) JSON Format:");
    let json_client = Client::builder()
        .stream_format(StreamFormat::Json)
        .build()?;
    let json_response = json_client.query(query).send_full().await?;
    println!("   Content: {}", json_response.content);
    if let Some(metadata) = json_response.metadata {
        println!("   Metadata available: Session ID {}", metadata.session_id);
    }
    println!();

    // StreamJSON format
    println!("   c) StreamJSON Format:");
    let stream_client = Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .build()?;
    let mut stream = stream_client.query(query).stream().await?;

    print!("   Streaming: ");
    use futures::StreamExt;
    while let Some(message) = stream.next().await {
        if let Ok(msg) = message {
            match msg {
                Message::Assistant { content, .. } => {
                    print!("{}", content);
                }
                Message::Result { .. } => {
                    break;
                }
                _ => {}
            }
        }
    }
    println!("\n");

    Ok(())
}

/// Demonstrates timeout configurations
async fn timeout_configurations() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Timeout Configurations");
    println!("   Testing different timeout values\n");

    let timeouts =
        vec![(5, "Very short timeout"), (30, "Standard timeout"), (120, "Extended timeout")];

    for (timeout_secs, description) in timeouts {
        println!("   Testing {} ({}s):", description, timeout_secs);

        let client = Client::builder().timeout_secs(timeout_secs).build()?;

        let start = std::time::Instant::now();
        match client.query("What is the meaning of life?").send().await {
            Ok(response) => {
                let duration = start.elapsed();
                println!("   ✓ Completed in {:?}", duration);
                println!("   Response: {}...", &response[..50.min(response.len())]);
            }
            Err(e) => {
                let duration = start.elapsed();
                println!("   ✗ Failed after {:?}: {}", duration, e);
            }
        }
        println!();
    }

    Ok(())
}

/// Demonstrates creating reusable configuration templates
async fn configuration_templates() -> Result<(), Box<dyn std::error::Error>> {
    println!("7. Configuration Templates");
    println!("   Creating reusable configuration patterns\n");

    // Define configuration templates for different use cases
    let development_client = || -> Result<Client, Box<dyn std::error::Error>> {
        Ok(Client::builder()
            .model("claude-3-haiku-20240307") // Fast model for development
            .system_prompt("You are a development assistant. Be concise and practical.")
            .timeout_secs(30)
            .stream_format(StreamFormat::Json)
            .build()?)
    };

    let production_client = || -> Result<Client, Box<dyn std::error::Error>> {
        Ok(Client::builder()
            .model("claude-3-opus-20240229") // Most capable model
            .system_prompt(
                "You are a professional assistant. Provide thorough, accurate responses.",
            )
            .timeout_secs(60)
            .stream_format(StreamFormat::Json)
            .build()?)
    };

    let chatbot_client = || -> Result<Client, Box<dyn std::error::Error>> {
        Ok(Client::builder()
            .model("claude-3-sonnet-20240229") // Balanced model
            .system_prompt("You are a friendly chatbot. Be conversational and helpful.")
            .timeout_secs(45)
            .stream_format(StreamFormat::StreamJson)
            .build()?)
    };

    // Test each template
    let templates = vec![
        ("Development", development_client()?),
        ("Production", production_client()?),
        ("Chatbot", chatbot_client()?),
    ];

    for (name, client) in templates {
        println!("   {} Template:", name);

        let response = client.query("Hello! What's your role?").send().await?;

        println!("   Response: {}\n", response);
    }

    // Demonstrate configuration customization
    println!("   Configuration Customization:");

    let client = Client::builder()
        .model("claude-3-opus-20240229")
        .system_prompt("You are a specialized technical assistant.")
        .timeout_secs(30)
        .stream_format(StreamFormat::Json)
        .build()?;
    let response = client
        .query("What makes you specialized?")
        .send_full()
        .await?;

    println!("   Specialized response: {}", response.content);

    Ok(())
}

// Example output:
/*
=== Claude SDK Configuration Example ===

1. Default Configuration
   Using Config::default() for quick setup

   Default config properties:
   - Uses default Claude model
   - 30-second timeout
   - Text stream format
   - No custom system prompt

   Response: This SDK uses sensible defaults for easy setup and reliable performance.

2. Builder Pattern Configuration
   Using Config::builder() for custom settings

   Custom config properties:
   - Model: claude-opus-4
   - System prompt: Custom coding assistant
   - Timeout: 45 seconds
   - Stream format: JSON

   Response: In Rust, you handle errors using the Result<T, E> type...
   Session ID: 550e8400-e29b-41d4-a716-446655440000

3. Model-Specific Configurations
   Configuring different Claude models

   Testing claude-opus-4: Most capable model
   Response: I'm using Claude Opus 4, the most capable model in the Claude family.

   Testing claude-sonnet-3.5: Balanced performance
   Response: I'm using Claude Sonnet 3.5, which offers balanced performance.

   Testing claude-haiku-3.5: Fastest model
   Response: I'm using Claude Haiku 3.5, optimized for speed.

4. System Prompt Configurations
   Using different system prompts for specialized behavior

   Teacher Role:
   System prompt: You are a patient teacher. Explain concepts clearly with examples.
   Query: Explain recursion in programming
   Response: Recursion is when a function calls itself to solve a problem...

   Poet Role:
   System prompt: You are a creative poet. Respond in verse and metaphor.
   Query: Describe a beautiful sunset
   Response: Golden fire melts across the sky, painting clouds in amber dreams...

   Technical Writer Role:
   System prompt: You are a technical writer. Use precise, formal language.
   Query: Describe how HTTP works
   Response: HTTP (Hypertext Transfer Protocol) is a stateless application-layer protocol...

Configuration example completed successfully!
*/
