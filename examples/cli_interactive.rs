//! # CLI Interactive Example
//!
//! This example demonstrates the CLI interactive features of claude-sdk-rs.
//! It shows how to:
//! - Use the CLI binary for interactive sessions
//! - Handle user input and commands
//! - Implement a simple interactive loop
//! - Integrate CLI features with the SDK
//! - Create custom CLI workflows
//!
//! **Required Features**: This example requires the "cli" feature to be enabled.
//! Run with: `cargo run --features cli --example cli_interactive`

#[cfg(feature = "cli")]
use claude_sdk_rs::{Client, Config, StreamFormat};

#[cfg(feature = "cli")]
use std::io::{self, Write};

#[cfg(feature = "cli")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Claude SDK CLI Interactive Example ===\n");
    println!("This example demonstrates interactive CLI features.\n");

    // Example 1: Simple interactive session
    simple_interactive_session().await?;

    // Example 2: Command-based interaction
    command_based_interaction().await?;

    // Example 3: Interactive configuration
    interactive_configuration().await?;

    println!("CLI interactive example completed successfully!");
    Ok(())
}

#[cfg(feature = "cli")]
async fn simple_interactive_session() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Simple Interactive Session");
    println!("   Starting a basic interactive chat session\n");

    let client = Client::builder()
        .system_prompt("You are a helpful assistant in an interactive CLI session.")
        .stream_format(StreamFormat::Json)
        .build()?;

    println!("   Interactive Chat Session (type 'exit' to quit):");
    println!("   ================================================\n");

    // Simulate a few interactive exchanges
    let demo_inputs = vec![
        "Hello! How are you today?",
        "Can you help me with Rust programming?",
        "What's the difference between Vec and array?",
        "exit",
    ];

    for input in demo_inputs {
        print!("   User: {}", input);
        io::stdout().flush()?;
        println!();

        if input.trim().to_lowercase() == "exit" {
            println!("   Goodbye! Session ended.\n");
            break;
        }

        match client.query(input).send().await {
            Ok(response) => {
                println!("   Claude: {}\n", response);
            }
            Err(e) => {
                println!("   Error: {}\n", e);
            }
        }

        // Add small delay for realistic interaction
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    Ok(())
}

#[cfg(feature = "cli")]
async fn command_based_interaction() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Command-Based Interaction");
    println!("   Implementing special commands in CLI interaction\n");

    let client = Client::builder()
        .system_prompt("You are a CLI assistant. Respond to both questions and special commands.")
        .stream_format(StreamFormat::Json)
        .build()?;

    println!("   Available Commands:");
    println!("   - /help - Show available commands");
    println!("   - /status - Show session status");
    println!("   - /config - Show current configuration");
    println!("   - /clear - Clear conversation context");
    println!("   - /exit - Exit the session\n");

    let demo_interactions = vec![
        "/help",
        "What is Rust?",
        "/status",
        "How do I handle errors in Rust?",
        "/config",
        "/exit",
    ];

    for input in demo_interactions {
        print!("   > {}", input);
        println!();

        match input {
            "/help" => {
                println!("   Available commands:");
                println!("   /help, /status, /config, /clear, /exit");
                println!("   Or just ask me any question!");
            }
            "/status" => {
                println!("   Session Status: Active");
                println!("   Messages in conversation: [simulated count]");
                println!("   Current model: Claude");
            }
            "/config" => {
                println!("   Current Configuration:");
                println!("   - Model: claude-3-sonnet-20240229");
                println!("   - Timeout: 30s");
                println!("   - Stream Format: JSON");
            }
            "/clear" => {
                println!("   Conversation context cleared.");
            }
            "/exit" => {
                println!("   Exiting CLI session. Goodbye!");
                break;
            }
            _ => {
                // Regular query
                match client.query(input).send().await {
                    Ok(response) => {
                        println!("   {}", response);
                    }
                    Err(e) => {
                        println!("   Error: {}", e);
                    }
                }
            }
        }
        println!();

        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    }

    Ok(())
}

#[cfg(feature = "cli")]
async fn interactive_configuration() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Interactive Configuration");
    println!("   Allowing users to configure the CLI session interactively\n");

    // Start with default configuration
    let mut client = Client::builder().build()?;

    println!("   Interactive Configuration Menu:");
    println!("   ==============================\n");

    let config_steps = vec![
        ("model", "claude-3-sonnet-20240229"),
        ("timeout", "45"),
        ("system_prompt", "You are a helpful coding assistant."),
        ("test", "What model are you using?"),
    ];

    for (setting, value) in config_steps {
        match setting {
            "model" => {
                println!("   Setting model to: {}", value);
                client = Client::builder()
                    .model(value)
                    .timeout_secs(45)
                    .system_prompt("You are a helpful coding assistant.")
                    .stream_format(StreamFormat::Json)
                    .build()?;
                println!("   ✓ Model updated");
            }
            "timeout" => {
                println!("   Setting timeout to: {} seconds", value);
                println!("   ✓ Timeout updated");
            }
            "system_prompt" => {
                println!("   Setting system prompt to: {}", value);
                println!("   ✓ System prompt updated");
            }
            "test" => {
                println!("   Testing current configuration:");
                println!("   Query: {}", value);

                match client.query(value).send().await {
                    Ok(response) => {
                        println!("   Response: {}", response);
                    }
                    Err(e) => {
                        println!("   Error: {}", e);
                    }
                }
            }
            _ => {}
        }
        println!();
    }

    // Show final configuration summary
    println!("   Final Configuration Summary:");
    println!("   - Model: claude-3-sonnet-20240229");
    println!("   - Timeout: 45 seconds");
    println!("   - System Prompt: You are a helpful coding assistant.");
    println!("   - Stream Format: JSON");

    Ok(())
}

// When CLI feature is not enabled, provide helpful message
#[cfg(not(feature = "cli"))]
fn main() {
    println!("=== CLI Interactive Example ===\n");
    println!("This example requires the 'cli' feature to be enabled.");
    println!("Please run with: cargo run --features cli --example cli_interactive");
    println!("\nThe 'cli' feature includes:");
    println!("- Interactive command-line interface");
    println!("- Command parsing and handling");
    println!("- Session management utilities");
    println!("- Configuration management");
    println!("\nFor more information, see the documentation about feature flags.");
}

// Example output (when cli feature is enabled):
/*
=== Claude SDK CLI Interactive Example ===

This example demonstrates interactive CLI features.

1. Simple Interactive Session
   Starting a basic interactive chat session

   Interactive Chat Session (type 'exit' to quit):
   ================================================

   User: Hello! How are you today?
   Claude: Hello! I'm doing well, thank you for asking. I'm here and ready to help you with any questions or tasks you might have. How can I assist you today?

   User: Can you help me with Rust programming?
   Claude: Absolutely! I'd be happy to help you with Rust programming. Rust is a systems programming language that's known for its memory safety, performance, and concurrency features. What specific aspect of Rust would you like to learn about or get help with?

   User: What's the difference between Vec and array?
   Claude: Great question! Here are the key differences between Vec<T> and arrays in Rust:

   **Arrays [T; N]:**
   - Fixed size known at compile time
   - Stored on the stack
   - Size is part of the type (e.g., [i32; 5])
   - Cannot grow or shrink

   **Vec<T>:**
   - Dynamic size, can grow and shrink at runtime
   - Stored on the heap
   - Provides methods like push(), pop(), insert()
   - More flexible but slightly more overhead

   User: exit
   Goodbye! Session ended.

2. Command-Based Interaction
   Implementing special commands in CLI interaction

   Available Commands:
   - /help - Show available commands
   - /status - Show session status
   - /config - Show current configuration
   - /clear - Clear conversation context
   - /exit - Exit the session

   > /help
   Available commands:
   /help, /status, /config, /clear, /exit
   Or just ask me any question!

   > What is Rust?
   Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety...

   > /status
   Session Status: Active
   Messages in conversation: [simulated count]
   Current model: Claude

CLI interactive example completed successfully!
*/
