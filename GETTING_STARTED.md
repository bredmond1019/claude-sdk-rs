# Getting Started with claude-sdk-rs: A Comprehensive Guide

Welcome to the claude-sdk-rs Rust SDK! This comprehensive guide will walk you through everything you need to know to start building powerful AI-powered applications with Claude in Rust.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Core Concepts](#core-concepts)
4. [Step-by-Step Examples](#step-by-step-examples)
5. [Real-World Applications](#real-world-applications)
6. [Best Practices](#best-practices)
7. [Common Patterns](#common-patterns)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Next Steps](#next-steps)

## Introduction

The claude-sdk-rs SDK is a type-safe, async-first Rust library that transforms the Claude Code CLI into a powerful programmatic API. Whether you're building a chatbot, code analyzer, or content generation system, this SDK provides the tools you need.

### Why Use claude-sdk-rs?

- **Type Safety**: All interactions are strongly typed, catching errors at compile time
- **Performance**: Built on Tokio for efficient async operations
- **Flexibility**: Three response modes (text, JSON, streaming) for different use cases
- **Production Ready**: Comprehensive error handling, timeouts, and retry logic
- **Cost Tracking**: Built-in token and cost monitoring for budget management

## Prerequisites & Installation

### System Requirements

1. **Rust 1.70 or later**
   ```bash
   # Check your Rust version
   rustc --version
   
   # Update Rust if needed
   rustup update stable
   ```

2. **Claude Code CLI**
   ```bash
   # Install Claude CLI (macOS/Linux)
   curl -sSL https://claude.ai/install.sh | sh
   
   # Windows users: Download from https://github.com/anthropics/claude-code
   
   # Verify installation
   claude --version
   ```

3. **Authentication**
   ```bash
   # Login with your API key
   claude auth login
   
   # Verify authentication
   claude auth status
   ```

### Project Setup

1. **Create a new Rust project**
   ```bash
   cargo new my-claude-app
   cd my-claude-app
   ```

2. **Add dependencies to `Cargo.toml`**
   ```toml
   [dependencies]
   claude-sdk-rs = "1.0.0"
   tokio = { version = "1.40", features = ["full"] }
   # Optional: for environment variables
   dotenv = "0.15"
   # Optional: for JSON handling
   serde = { version = "1.0", features = ["derive"] }
   serde_json = "1.0"
   ```

3. **Create a basic main.rs**
   ```rust
   use claude_ai::{Client, Config};
   
   #[tokio::main]
   async fn main() -> claude_ai::Result<()> {
       println!("Claude AI SDK is ready!");
       Ok(())
   }
   ```

4. **Verify everything works**
   ```bash
   cargo build
   cargo run
   ```

## Core Concepts

Before diving into examples, let's understand the key concepts:

### 1. Client

The `Client` is your main interface to Claude:
```rust
// Default client
let client = Client::new(Config::default());

// Custom configuration
let client = Client::builder()
    .model("claude-sonnet-4-20250514")
    .timeout(60)
    .build();
```

### 2. Query Builder

Queries are built using a fluent API:
```rust
client.query("Your prompt here")
    .session_id("optional-session-id")
    .send()
    .await?
```

### 3. Response Modes

Three ways to get responses:
- `send()` - Simple text response
- `send_full()` - Response with metadata (cost, tokens, etc.)
- `stream()` - Real-time streaming output

### 4. Configuration

Configure behavior through `Config` or the builder pattern:
```rust
let config = Config::builder()
    .model("claude-opus-4-20250514")
    .system_prompt("You are a helpful assistant")
    .stream_format(StreamFormat::Json)
    .timeout(Duration::from_secs(30))
    .build()?;
```

## Step-by-Step Examples

### Example 1: Hello World

Let's start with the absolute basics:

```rust
use claude_ai::{Client, Config};

#[tokio::main]
async fn main() -> claude_ai::Result<()> {
    // Step 1: Create a client
    let client = Client::new(Config::default());
    
    // Step 2: Send a query
    let response = client
        .query("Say hello and introduce yourself!")
        .send()
        .await?;
    
    // Step 3: Display the response
    println!("Claude says: {}", response);
    
    Ok(())
}
```

**What's happening:**
1. We create a client with default settings
2. We send a simple query asking Claude to introduce itself
3. We get a text response and print it

### Example 2: Custom Configuration

Now let's customize Claude's behavior:

```rust
use claude_ai::{Client, StreamFormat};
use std::time::Duration;

#[tokio::main]
async fn main() -> claude_ai::Result<()> {
    // Build a client with specific settings
    let client = Client::builder()
        .model("claude-sonnet-4-20250514")  // Faster model
        .system_prompt("You are a Rust programming expert. Be concise.")
        .stream_format(StreamFormat::Json)   // Get structured responses
        .timeout(Duration::from_secs(45))    // 45 second timeout
        .build();
    
    // Ask a Rust-specific question
    let response = client
        .query("What are the key differences between &str and String in Rust?")
        .send_full()  // Get full response with metadata
        .await?;
    
    // Display response and metadata
    println!("Answer: {}", response.content);
    
    if let Some(metadata) = response.metadata {
        println!("\n--- Response Details ---");
        println!("Session ID: {}", metadata.session_id);
        
        if let Some(cost) = metadata.cost_usd {
            println!("Cost: ${:.6}", cost);
        }
        
        if let Some(tokens) = metadata.tokens_used {
            println!("Tokens used: {} input, {} output", 
                tokens.input_tokens.unwrap_or(0),
                tokens.output_tokens.unwrap_or(0)
            );
        }
    }
    
    Ok(())
}
```

### Example 3: Session Management

Maintain conversation context across multiple queries:

```rust
use claude_ai::{Client, Config};
use claude_ai_core::session::SessionId;

#[tokio::main]
async fn main() -> claude_ai::Result<()> {
    let client = Client::new(Config::default());
    
    // Start a new session
    let session_id = SessionId::new();
    
    // First message
    let response1 = client
        .query("My name is Alice and I'm learning Rust.")
        .session_id(&session_id)
        .send()
        .await?;
    
    println!("Claude: {}\n", response1);
    
    // Follow-up message in same session
    let response2 = client
        .query("What's my name and what am I learning?")
        .session_id(&session_id)
        .send()
        .await?;
    
    println!("Claude: {}", response2);
    
    Ok(())
}
```

### Example 4: Error Handling

Robust error handling for production applications:

```rust
use claude_ai::{Client, Config, Error};
use std::time::Duration;

#[tokio::main]
async fn main() {
    let client = Client::builder()
        .timeout(Duration::from_secs(5))  // Short timeout for demo
        .build();
    
    match execute_query(&client).await {
        Ok(response) => println!("Success: {}", response),
        Err(e) => handle_error(e),
    }
}

async fn execute_query(client: &Client) -> claude_ai::Result<String> {
    client
        .query("Explain quantum computing in detail")
        .send()
        .await
}

fn handle_error(error: Error) {
    match error {
        Error::Timeout => {
            eprintln!("Request timed out. Try a shorter query or increase timeout.");
        }
        Error::ClaudeNotAuthenticated => {
            eprintln!("Not authenticated. Run: claude auth login");
        }
        Error::ClaudeNotFound => {
            eprintln!("Claude CLI not found. Please install it first.");
        }
        Error::ProcessFailed { exit_code, stderr } => {
            eprintln!("Claude returned an error (code {}): {}", exit_code, stderr);
        }
        Error::SerializationError(e) => {
            eprintln!("Failed to parse response: {}", e);
        }
        _ => {
            eprintln!("Unexpected error: {:?}", error);
        }
    }
}
```

### Example 5: Streaming Responses

Get real-time output as Claude generates it:

```rust
use claude_ai::{Client, StreamFormat};
use futures::StreamExt;

#[tokio::main]
async fn main() -> claude_ai::Result<()> {
    let client = Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .build();
    
    println!("Asking Claude to write a story...\n");
    
    let mut stream = client
        .query("Write a short story about a robot learning to paint")
        .stream()
        .await?;
    
    // Process each chunk as it arrives
    while let Some(result) = stream.next().await {
        match result {
            Ok(message) => {
                // Print without newline for smooth streaming
                print!("{}", message.content);
                // Flush to ensure immediate output
                use std::io::{self, Write};
                io::stdout().flush().unwrap();
            }
            Err(e) => eprintln!("\nStream error: {}", e),
        }
    }
    
    println!("\n\nStory complete!");
    
    Ok(())
}
```

## Real-World Applications

### Application 1: Code Review Assistant

Build a code review tool that analyzes Rust code:

```rust
use claude_ai::{Client, Config};
use std::fs;

#[tokio::main]
async fn main() -> claude_ai::Result<()> {
    let client = Client::builder()
        .system_prompt("You are a Rust code reviewer. Focus on safety, performance, and idiomatic patterns.")
        .build();
    
    // Read code from file
    let code = fs::read_to_string("src/main.rs")?;
    
    let prompt = format!(
        "Please review this Rust code and provide feedback:\n\n```rust\n{}\n```\n\nFocus on:\n1. Memory safety issues\n2. Performance improvements\n3. Idiomatic Rust patterns\n4. Error handling",
        code
    );
    
    let review = client.query(&prompt).send().await?;
    
    println!("Code Review Results:\n{}", review);
    
    Ok(())
}
```

### Application 2: Interactive CLI Assistant

Create an interactive command-line assistant:

```rust
use claude_ai::{Client, Config};
use claude_ai_core::session::SessionId;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> claude_ai::Result<()> {
    let client = Client::builder()
        .system_prompt("You are a helpful CLI assistant. Be concise and practical.")
        .build();
    
    let session_id = SessionId::new();
    
    println!("Claude CLI Assistant (type 'quit' to exit)\n");
    
    loop {
        // Prompt for input
        print!("> ");
        io::stdout().flush()?;
        
        // Read user input
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        // Check for exit
        if input == "quit" || input == "exit" {
            println!("Goodbye!");
            break;
        }
        
        // Send to Claude with session context
        match client.query(input).session_id(&session_id).send().await {
            Ok(response) => println!("\n{}\n", response),
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    
    Ok(())
}
```

### Application 3: Batch Processing with Cost Tracking

Process multiple queries while tracking costs:

```rust
use claude_ai::{Client, StreamFormat};
use std::fs::File;
use std::io::{BufRead, BufReader};

#[tokio::main]
async fn main() -> claude_ai::Result<()> {
    let client = Client::builder()
        .stream_format(StreamFormat::Json)
        .model("claude-haiku-3-20250307")  // Cheaper model for batch processing
        .build();
    
    // Read queries from file
    let file = File::open("queries.txt")?;
    let reader = BufReader::new(file);
    
    let mut total_cost = 0.0;
    let mut total_tokens = 0;
    
    for (i, line) in reader.lines().enumerate() {
        let query = line?;
        if query.trim().is_empty() {
            continue;
        }
        
        println!("Processing query {}: {}", i + 1, query);
        
        match client.query(&query).send_full().await {
            Ok(response) => {
                println!("Response: {}\n", response.content);
                
                // Track costs
                if let Some(metadata) = response.metadata {
                    if let Some(cost) = metadata.cost_usd {
                        total_cost += cost;
                    }
                    if let Some(tokens) = metadata.tokens_used {
                        total_tokens += tokens.total_tokens.unwrap_or(0);
                    }
                }
            }
            Err(e) => eprintln!("Error processing query {}: {}", i + 1, e),
        }
    }
    
    println!("\n--- Batch Processing Complete ---");
    println!("Total cost: ${:.6}", total_cost);
    println!("Total tokens: {}", total_tokens);
    
    Ok(())
}
```

## Best Practices

### 1. Client Reuse

Always reuse clients when possible:

```rust
// Good: Create once, use many times
let client = Client::new(Config::default());
for query in queries {
    let response = client.query(&query).send().await?;
}

// Bad: Creating new client for each query
for query in queries {
    let client = Client::new(Config::default());  // Inefficient!
    let response = client.query(&query).send().await?;
}
```

### 2. Appropriate Response Modes

Choose the right response mode for your use case:

```rust
// Simple text response - when you just need the content
let text = client.query("...").send().await?;

// Full response - when you need metadata
let full = client.query("...").send_full().await?;

// Streaming - for long responses or real-time output
let stream = client.query("...").stream().await?;
```

### 3. Error Recovery

Implement retry logic for transient failures:

```rust
use std::time::Duration;
use tokio::time::sleep;

async fn query_with_retry(client: &Client, prompt: &str, max_retries: u32) -> claude_ai::Result<String> {
    let mut retries = 0;
    
    loop {
        match client.query(prompt).send().await {
            Ok(response) => return Ok(response),
            Err(e) if retries < max_retries => {
                eprintln!("Attempt {} failed: {}. Retrying...", retries + 1, e);
                retries += 1;
                sleep(Duration::from_secs(2_u64.pow(retries))).await;
            }
            Err(e) => return Err(e),
        }
    }
}
```

### 4. Cost Management

Monitor and control costs:

```rust
use claude_ai::{Client, StreamFormat};

struct CostTracker {
    budget: f64,
    spent: f64,
}

impl CostTracker {
    fn new(budget: f64) -> Self {
        Self { budget, spent: 0.0 }
    }
    
    async fn query(&mut self, client: &Client, prompt: &str) -> claude_ai::Result<String> {
        if self.spent >= self.budget {
            return Err(claude_ai::Error::Custom("Budget exceeded".to_string()));
        }
        
        let response = client.query(prompt).send_full().await?;
        
        if let Some(metadata) = response.metadata {
            if let Some(cost) = metadata.cost_usd {
                self.spent += cost;
            }
        }
        
        Ok(response.content)
    }
    
    fn remaining(&self) -> f64 {
        self.budget - self.spent
    }
}
```

## Common Patterns

### Pattern 1: Configuration from Environment

Load settings from environment variables:

```rust
use claude_ai::{Client, StreamFormat};
use std::env;

fn create_client_from_env() -> Client {
    Client::builder()
        .model(env::var("CLAUDE_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string()))
        .timeout(env::var("CLAUDE_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30))
        .build()
}
```

### Pattern 2: Prompt Templates

Create reusable prompt templates:

```rust
struct PromptTemplate {
    template: String,
}

impl PromptTemplate {
    fn new(template: &str) -> Self {
        Self { template: template.to_string() }
    }
    
    fn render(&self, vars: &[(&str, &str)]) -> String {
        let mut result = self.template.clone();
        for (key, value) in vars {
            result = result.replace(&format!("{{{}}}", key), value);
        }
        result
    }
}

// Usage
let template = PromptTemplate::new(
    "Analyze this {language} code for {focus}:\n\n```\n{code}\n```"
);

let prompt = template.render(&[
    ("language", "Rust"),
    ("focus", "memory safety"),
    ("code", &code_snippet),
]);
```

### Pattern 3: Response Caching

Cache responses to avoid redundant API calls:

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

struct CachedClient {
    client: Client,
    cache: Arc<Mutex<HashMap<String, String>>>,
}

impl CachedClient {
    fn new(client: Client) -> Self {
        Self {
            client,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    async fn query(&self, prompt: &str) -> claude_ai::Result<String> {
        // Check cache first
        {
            let cache = self.cache.lock().await;
            if let Some(cached) = cache.get(prompt) {
                return Ok(cached.clone());
            }
        }
        
        // Query Claude
        let response = self.client.query(prompt).send().await?;
        
        // Cache the response
        {
            let mut cache = self.cache.lock().await;
            cache.insert(prompt.to_string(), response.clone());
        }
        
        Ok(response)
    }
}
```

## Performance Optimization

### 1. Concurrent Queries

Process multiple queries in parallel:

```rust
use futures::future::join_all;

async fn batch_query(client: &Client, queries: Vec<String>) -> Vec<claude_ai::Result<String>> {
    let futures = queries.into_iter()
        .map(|q| client.query(&q).send())
        .collect::<Vec<_>>();
    
    join_all(futures).await
}
```

### 2. Model Selection

Choose the right model for your use case:

```rust
enum TaskType {
    Simple,      // Quick responses
    Complex,     // Deep analysis
    Creative,    // Creative writing
}

fn select_model(task: TaskType) -> &'static str {
    match task {
        TaskType::Simple => "claude-haiku-3-20250307",      // Fast and cheap
        TaskType::Complex => "claude-sonnet-4-20250514",    // Balanced
        TaskType::Creative => "claude-opus-4-20250514",     // Most capable
    }
}
```

### 3. Stream Processing

Process streaming responses efficiently:

```rust
use futures::StreamExt;
use tokio::io::{AsyncWrite, AsyncWriteExt};

async fn stream_to_file<W: AsyncWrite + Unpin>(
    client: &Client,
    prompt: &str,
    mut writer: W,
) -> claude_ai::Result<()> {
    let mut stream = client.query(prompt).stream().await?;
    
    while let Some(result) = stream.next().await {
        let message = result?;
        writer.write_all(message.content.as_bytes()).await?;
    }
    
    writer.flush().await?;
    Ok(())
}
```

## Troubleshooting

### Common Issues and Solutions

1. **"Binary not found" error**
   ```bash
   # Verify Claude CLI is installed
   which claude
   
   # Add to PATH if needed
   export PATH="$PATH:/path/to/claude"
   ```

2. **Authentication failures**
   ```bash
   # Check auth status
   claude auth status
   
   # Re-authenticate
   claude auth logout
   claude auth login
   ```

3. **Timeout errors**
   ```rust
   // Increase timeout for complex queries
   let client = Client::builder()
       .timeout(120)  // 2 minutes
       .build();
   ```

4. **JSON parsing errors**
   ```rust
   // Use Text format if JSON parsing fails
   let client = Client::builder()
       .stream_format(StreamFormat::Text)
       .build();
   ```

5. **Rate limiting**
   ```rust
   // Add delay between requests
   use tokio::time::{sleep, Duration};
   
   for query in queries {
       let response = client.query(&query).send().await?;
       sleep(Duration::from_millis(500)).await;
   }
   ```

### Debug Mode

Enable debug output for troubleshooting:

```rust
use claude_ai::{Client, Config};

// Set environment variable
std::env::set_var("RUST_LOG", "claude_ai=debug");

// Initialize logging
env_logger::init();

// Your code will now output debug information
```

## Next Steps

Now that you've mastered the basics, explore these advanced topics:

1. **[Tool Integration](docs/tutorials/05-tool-integration.md)** - Let Claude use filesystem, web search, and custom tools
2. **[Session Management](docs/tutorials/06-session-management.md)** - Build stateful applications with conversation history
3. **[Advanced Configuration](docs/tutorials/07-advanced-usage.md)** - Fine-tune Claude's behavior for specific use cases
4. **[MCP Servers](claude-sdk-rs-mcp/README.md)** - Extend Claude with Model Context Protocol servers

### Additional Resources

- **Examples**: Check `claude-sdk-rs/examples/` for complete working examples
- **API Reference**: See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for detailed API documentation
- **Performance Guide**: Read [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for optimization tips
- **Migration Guide**: See [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) for upgrading

### Community and Support

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share experiences
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Summary

You now have everything you need to build powerful AI applications with the claude-sdk-rs SDK. Remember:

- Start simple with basic queries
- Use the builder pattern for customization
- Choose appropriate response modes
- Implement proper error handling
- Monitor costs and performance
- Reuse clients for efficiency

Happy coding with Claude AI!