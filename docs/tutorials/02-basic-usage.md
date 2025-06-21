# Part 2: Basic Usage and Response Modes

Now that you can make basic queries, let's explore the different ways to interact with Claude and handle responses.

## Three Response Modes

The claude-sdk-rs SDK offers three distinct response modes, each suited for different use cases:

### 1. Simple Text Mode (Default)

This is the simplest mode - just get the text response:

```rust
use claude_sdk_rs::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(Config::default());
    
    let response = client
        .query("Explain quantum computing in one sentence")
        .send()
        .await?;
    
    println!("Claude says: {}", response);
    Ok(())
}
```

**When to use**: Quick questions, simple integrations, when you only need the text content.

### 2. Full Metadata Mode (JSON)

Get structured response with metadata, costs, and session information:

```rust
use claude_sdk_rs::{Client, Config, StreamFormat};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .stream_format(StreamFormat::Json)
        .build();
    let client = Client::new(config);
    
    let response = client
        .query("What's the capital of France?")
        .send_full()  // Note: send_full() instead of send()
        .await?;
    
    println!("Response: {}", response.content);
    println!("Session ID: {}", response.meta.session_id);
    println!("Cost: ${:.4}", response.stats.total_cost_usd);
    println!("Tokens used: {}", response.stats.token_usage.total);
    
    Ok(())
}
```

**When to use**: When you need cost tracking, session management, or response metadata.

### 3. Streaming Mode (StreamJSON)

Process responses in real-time as they're generated:

```rust
use claude_sdk_rs::{Client, Config, StreamFormat, Message};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .stream_format(StreamFormat::StreamJson)
        .build();
    let client = Client::new(config);
    
    let mut stream = client
        .query("Write a short story about a robot")
        .stream()
        .await?;
    
    while let Some(message) = stream.next().await {
        match message? {
            Message::Assistant { content, .. } => {
                print!("{}", content);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Message::Result { stats, .. } => {
                println!("\n\nDone! Cost: ${:.4}", stats.total_cost_usd);
            }
            _ => {}
        }
    }
    
    Ok(())
}
```

**When to use**: Long responses, real-time output, interactive applications.

## Query Builder Options

The query builder provides several options for customizing your requests:

### System Prompts

Set context or instructions for the conversation:

```rust
let response = client
    .query("What's 2+2?")
    .system_prompt("You are a math tutor. Always explain your reasoning.")
    .send()
    .await?;
```

### Model Selection

Choose which Claude model to use:

```rust
let config = Config::builder()
    .model("claude-3-sonnet-20240229")  // Sonnet model
    .build();
let client = Client::new(config);

let response = client
    .query("Write a haiku about programming")
    .send()
    .await?;
```

### Combining Options

```rust
let response = client
    .query("Analyze this code for potential bugs")
    .system_prompt("You are an expert code reviewer")
    .send()
    .await?;
```

## Error Handling Patterns

Understanding and handling different error types:

```rust
use claude_sdk_rs::{Client, Config, Error};

#[tokio::main]
async fn main() {
    let client = Client::new(Config::default());
    
    match client.query("Hello").send().await {
        Ok(response) => {
            println!("Success: {}", response);
        }
        Err(Error::BinaryNotFound) => {
            eprintln!("Error: Claude CLI not found. Please install it first.");
        }
        Err(Error::Timeout) => {
            eprintln!("Error: Request timed out. Try increasing timeout.");
        }
        Err(Error::ProcessError(msg)) => {
            eprintln!("CLI Error: {}", msg);
        }
        Err(e) => {
            eprintln!("Other error: {:?}", e);
        }
    }
}
```

## Practical Examples

### Code Review Assistant

```rust
use claude_sdk_rs::{Client, Config};

async fn review_code(code: &str) -> Result<String, Box<dyn std::error::Error>> {
    let client = Client::new(Config::default());
    
    let prompt = format!(
        "Review this code for bugs, performance issues, and best practices:\n\n```rust\n{}\n```", 
        code
    );
    
    client
        .query(&prompt)
        .system_prompt("You are an expert Rust developer and code reviewer")
        .send()
        .await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let code = r#"
fn fibonacci(n: u32) -> u32 {
    if n <= 1 { n } else { fibonacci(n-1) + fibonacci(n-2) }
}
"#;
    
    let review = review_code(code).await?;
    println!("Code Review:\n{}", review);
    
    Ok(())
}
```

### Interactive Chat Loop

```rust
use claude_sdk_rs::{Client, Config, StreamFormat};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::builder()
        .stream_format(StreamFormat::Json)
        .build();
    let client = Client::new(config);
    
    println!("Claude Chat - Type 'quit' to exit");
    
    loop {
        print!("You: ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        
        if input == "quit" {
            break;
        }
        
        let response = client.query(input).send().await?;
        println!("Claude: {}\n", response);
    }
    
    Ok(())
}
```

## Performance Tips

1. **Reuse clients**: Create one client and use it for multiple queries
2. **Choose appropriate timeouts**: Longer queries need longer timeouts
3. **Use streaming for long responses**: Better user experience
4. **Handle errors gracefully**: Don't crash on network issues

## Next Steps

Ready to dive deeper? Check out:

- **Part 3**: [Configuration and Customization](03-configuration.md) - Learn about all the ways to customize your client
- **Part 4**: [Streaming Responses](04-streaming-responses.md) - Master real-time response processing

## Common Patterns Summary

```rust
// Quick text response
let text = client.query("Hello").send().await?;

// Full metadata
let full = client.query("Hello").send_full().await?;

// Streaming
let mut stream = client.query("Long response").stream().await?;
while let Some(msg) = stream.next().await { /* process */ }

// With options
let response = client
    .query("Question")
    .system_prompt("Context")
    .send()
    .await?;
```

These patterns will cover 90% of your use cases!