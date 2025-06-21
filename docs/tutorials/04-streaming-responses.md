# Part 4: Streaming Responses

Real-time response processing is one of the most powerful features of the claude-sdk-rs SDK. This tutorial covers everything you need to know about streaming responses, from basic setup to advanced patterns for building interactive applications.

## Why Use Streaming?

Streaming responses provide several advantages over traditional request-response patterns:

- **Better User Experience**: Users see output immediately as it's generated
- **Perceived Performance**: Applications feel more responsive
- **Real-time Feedback**: Progress indicators and live updates
- **Memory Efficiency**: Process responses incrementally without buffering everything
- **Interactivity**: Build chat interfaces and live coding assistants

## Basic Streaming Setup

To enable streaming, configure your client with `StreamFormat::StreamJson`:

```rust
use claude_sdk_rs::{Client, Config, StreamFormat, Message};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure client for streaming
    let config = Config::builder()
        .stream_format(StreamFormat::StreamJson)
        .model("claude-3-sonnet-20240229")
        .build();
    let client = Client::new(config);

    // Create a streaming query
    let mut stream = client
        .query("Write a short story about a programmer who discovers AI")
        .stream()
        .await?;

    // Process messages as they arrive
    while let Some(message) = stream.next().await {
        match message? {
            Message::Assistant { content, .. } => {
                print!("{}", content);
                // Flush to see output immediately
                use std::io::{self, Write};
                io::stdout().flush().unwrap();
            }
            Message::Result { stats, .. } => {
                println!("\n\nStream complete! Cost: ${:.4}", stats.total_cost_usd);
            }
            _ => {}
        }
    }

    Ok(())
}
```

## Understanding the Message Enum

The streaming API uses a rich `Message` enum to represent different types of events:

### Message Types

```rust
pub enum Message {
    Init { meta: MessageMeta },                    // Stream initialization
    User { content: String, meta: MessageMeta },   // User messages
    Assistant { content: String, meta: MessageMeta }, // Claude's responses
    System { content: String, meta: MessageMeta }, // System messages
    Tool { name: String, parameters: Value, meta: MessageMeta }, // Tool calls
    ToolResult { tool_name: String, result: Value, meta: MessageMeta }, // Tool outputs
    Result { meta: MessageMeta, stats: ConversationStats }, // Final statistics
}
```

### Message Metadata

Each message includes rich metadata:

```rust
pub struct MessageMeta {
    pub session_id: String,         // Unique session identifier
    pub timestamp: Option<SystemTime>, // When the message was created
    pub cost_usd: Option<f64>,      // Cost for this message
    pub duration_ms: Option<u64>,   // Processing time
    pub tokens_used: Option<TokenUsage>, // Token usage breakdown
}

pub struct TokenUsage {
    pub input: u64,    // Input tokens consumed
    pub output: u64,   // Output tokens generated
    pub total: u64,    // Total tokens used
}
```

## Message Processing Patterns

### Pattern 1: Content-Only Processing

For simple text streaming:

```rust
async fn stream_text_only(client: &Client, query: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut stream = client.query(query).stream().await?;
    let mut full_response = String::new();

    while let Some(message) = stream.next().await {
        if let Message::Assistant { content, .. } = message? {
            print!("{}", content);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            full_response.push_str(&content);
        }
    }

    Ok(full_response)
}
```

### Pattern 2: Complete Message Handling

Process all message types with full context:

```rust
async fn handle_all_messages(client: &Client, query: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = client.query(query).stream().await?;
    let mut session_id = String::new();

    while let Some(message) = stream.next().await {
        match message? {
            Message::Init { meta } => {
                session_id = meta.session_id.clone();
                println!("🚀 Starting session: {}", session_id);
            }
            Message::User { content, meta } => {
                println!("👤 User: {}", content);
                if let Some(cost) = meta.cost_usd {
                    println!("   💰 Cost: ${:.6}", cost);
                }
            }
            Message::Assistant { content, meta } => {
                print!("{}", content);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                
                // Optional: Track token usage in real-time
                if let Some(tokens) = &meta.tokens_used {
                    // Update progress indicators
                }
            }
            Message::System { content, .. } => {
                println!("🔧 System: {}", content);
            }
            Message::Tool { name, parameters, .. } => {
                println!("🛠️  Tool call: {} with {}", name, parameters);
            }
            Message::ToolResult { tool_name, result, .. } => {
                println!("📊 Tool {} result: {}", tool_name, result);
            }
            Message::Result { stats, .. } => {
                println!("\n✅ Complete!");
                println!("   Messages: {}", stats.total_messages);
                println!("   Cost: ${:.4}", stats.total_cost_usd);
                println!("   Duration: {}ms", stats.total_duration_ms);
                println!("   Tokens: {}", stats.total_tokens.total);
            }
        }
    }

    Ok(())
}
```

### Pattern 3: Structured Data Collection

Collect and organize streaming data:

```rust
use std::collections::VecDeque;

#[derive(Debug)]
struct StreamedConversation {
    session_id: String,
    messages: Vec<String>,
    tools_used: Vec<String>,
    final_stats: Option<ConversationStats>,
}

async fn collect_conversation(client: &Client, query: &str) -> Result<StreamedConversation, Box<dyn std::error::Error>> {
    let mut stream = client.query(query).stream().await?;
    let mut conversation = StreamedConversation {
        session_id: String::new(),
        messages: Vec::new(),
        tools_used: Vec::new(),
        final_stats: None,
    };

    while let Some(message) = stream.next().await {
        match message? {
            Message::Init { meta } => {
                conversation.session_id = meta.session_id;
            }
            Message::Assistant { content, .. } => {
                conversation.messages.push(content.clone());
                print!("{}", content);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Message::Tool { name, .. } => {
                conversation.tools_used.push(name);
            }
            Message::Result { stats, .. } => {
                conversation.final_stats = Some(stats);
            }
            _ => {}
        }
    }

    Ok(conversation)
}
```

## Error Handling in Streaming

Streaming operations can fail at various points. Here's how to handle errors gracefully:

### Basic Error Handling

```rust
async fn stream_with_error_handling(client: &Client, query: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = match client.query(query).stream().await {
        Ok(stream) => stream,
        Err(e) => {
            eprintln!("Failed to start stream: {}", e);
            return Err(e);
        }
    };

    while let Some(message_result) = stream.next().await {
        match message_result {
            Ok(Message::Assistant { content, .. }) => {
                print!("{}", content);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Ok(Message::Result { stats, .. }) => {
                println!("\nStream completed successfully!");
                println!("Cost: ${:.4}", stats.total_cost_usd);
                break;
            }
            Ok(_) => {} // Handle other message types
            Err(e) => {
                eprintln!("\nStream error: {}", e);
                // Decide whether to continue or abort
                match e {
                    claude_sdk_rs::Error::Timeout => {
                        eprintln!("Stream timed out, retrying...");
                        // Could implement retry logic here
                        break;
                    }
                    claude_sdk_rs::Error::ProcessError(_) => {
                        eprintln!("Process error, aborting stream");
                        return Err(e);
                    }
                    _ => {
                        eprintln!("Unknown error: {:?}", e);
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}
```

### Retry Logic for Streams

```rust
use tokio::time::{sleep, Duration};

async fn stream_with_retry(
    client: &Client, 
    query: &str, 
    max_retries: u32
) -> Result<(), Box<dyn std::error::Error>> {
    let mut attempts = 0;

    while attempts < max_retries {
        match try_stream(client, query).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                attempts += 1;
                eprintln!("Stream attempt {} failed: {}", attempts, e);
                
                if attempts >= max_retries {
                    return Err(e);
                }

                // Exponential backoff
                let delay = Duration::from_secs(2_u64.pow(attempts));
                sleep(delay).await;
            }
        }
    }

    Err(claude_sdk_rs::Error::ProcessError("Max retries exceeded".to_string()))
}

async fn try_stream(client: &Client, query: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = client.query(query).stream().await?;

    while let Some(message) = stream.next().await {
        match message? {
            Message::Assistant { content, .. } => {
                print!("{}", content);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Message::Result { .. } => break,
            _ => {}
        }
    }

    Ok(())
}
```

## Progress Indicators and UI Integration

### Simple Progress Indicator

```rust
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

struct ProgressTracker {
    tokens_received: Arc<Mutex<u64>>,
    is_complete: Arc<Mutex<bool>>,
}

impl ProgressTracker {
    fn new() -> Self {
        Self {
            tokens_received: Arc::new(Mutex::new(0)),
            is_complete: Arc::new(Mutex::new(false)),
        }
    }

    fn start_spinner(&self) {
        let tokens = Arc::clone(&self.tokens_received);
        let complete = Arc::clone(&self.is_complete);

        thread::spawn(move || {
            let spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            let mut i = 0;

            while !*complete.lock().unwrap() {
                let token_count = *tokens.lock().unwrap();
                print!("\r{} Processing... ({} tokens)", spinner_chars[i], token_count);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                
                i = (i + 1) % spinner_chars.len();
                thread::sleep(Duration::from_millis(100));
            }
            
            print!("\r✅ Complete!          \n");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        });
    }

    fn update_tokens(&self, count: u64) {
        *self.tokens_received.lock().unwrap() = count;
    }

    fn mark_complete(&self) {
        *self.is_complete.lock().unwrap() = true;
    }
}

async fn stream_with_progress(client: &Client, query: &str) -> Result<(), Box<dyn std::error::Error>> {
    let progress = ProgressTracker::new();
    progress.start_spinner();

    let mut stream = client.query(query).stream().await?;
    let mut total_tokens = 0u64;

    while let Some(message) = stream.next().await {
        match message? {
            Message::Assistant { content, meta } => {
                if let Some(tokens) = &meta.tokens_used {
                    total_tokens += tokens.output;
                    progress.update_tokens(total_tokens);
                }
                print!("{}", content);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Message::Result { .. } => {
                progress.mark_complete();
                break;
            }
            _ => {}
        }
    }

    // Give spinner time to update
    tokio::time::sleep(Duration::from_millis(150)).await;
    Ok(())
}
```

### Real-time Statistics Display

```rust
use std::time::Instant;

struct StreamStats {
    start_time: Instant,
    tokens_received: u64,
    messages_count: u64,
    current_cost: f64,
}

impl StreamStats {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            tokens_received: 0,
            messages_count: 0,
            current_cost: 0.0,
        }
    }

    fn update(&mut self, tokens: Option<&TokenUsage>, cost: Option<f64>) {
        if let Some(token_usage) = tokens {
            self.tokens_received += token_usage.output;
        }
        if let Some(c) = cost {
            self.current_cost += c;
        }
        self.messages_count += 1;
    }

    fn display_stats(&self) {
        let elapsed = self.start_time.elapsed();
        let tokens_per_sec = if elapsed.as_secs() > 0 {
            self.tokens_received as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        print!(
            "\r📊 {} tokens | {:.1} tok/s | ${:.4} | {}s",
            self.tokens_received,
            tokens_per_sec,
            self.current_cost,
            elapsed.as_secs()
        );
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }
}

async fn stream_with_stats(client: &Client, query: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = client.query(query).stream().await?;
    let mut stats = StreamStats::new();

    println!("Starting stream...");

    while let Some(message) = stream.next().await {
        match message? {
            Message::Assistant { content, meta } => {
                stats.update(meta.tokens_used.as_ref(), meta.cost_usd);
                stats.display_stats();
                
                // Move to new line for content
                println!();
                print!("{}", content);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Message::Result { stats: final_stats, .. } => {
                println!("\n\n✅ Stream Complete!");
                println!("Final Statistics:");
                println!("  Total Cost: ${:.4}", final_stats.total_cost_usd);
                println!("  Total Tokens: {}", final_stats.total_tokens.total);
                println!("  Duration: {}ms", final_stats.total_duration_ms);
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
```

## Advanced Streaming Patterns

### Chat Interface Implementation

```rust
use tokio::io::{self, AsyncBufReadExt, BufReader};

async fn interactive_streaming_chat(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Claude Streaming Chat");
    println!("Type 'quit' to exit, 'clear' to start new session");
    println!("=" * 50);

    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        print!("\n👤 You: ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        line.clear();
        if reader.read_line(&mut line).await? == 0 {
            break; // EOF
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input == "quit" {
            break;
        }
        if input == "clear" {
            println!("🔄 Starting new session...");
            continue;
        }

        print!("🤖 Claude: ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        // Stream the response
        let mut stream = client.query(input).stream().await?;
        let mut response_started = false;

        while let Some(message) = stream.next().await {
            match message? {
                Message::Assistant { content, .. } => {
                    if !response_started {
                        response_started = true;
                    }
                    print!("{}", content);
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }
                Message::Result { stats, .. } => {
                    if response_started {
                        println!();
                        println!("💰 Cost: ${:.4} | ⏱️  {}ms", 
                                stats.total_cost_usd, 
                                stats.total_duration_ms);
                    }
                    break;
                }
                _ => {}
            }
        }
    }

    println!("👋 Goodbye!");
    Ok(())
}
```

### Concurrent Stream Processing

```rust
use tokio::sync::mpsc;
use futures::stream::StreamExt;

#[derive(Debug)]
enum StreamEvent {
    Content(String),
    Progress(u64),
    Complete(ConversationStats),
    Error(Box<dyn std::error::Error + Send + Sync>),
}

async fn concurrent_stream_processing(
    client: &Client, 
    queries: Vec<&str>
) -> Result<(), Box<dyn std::error::Error>> {
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(100);

    // Spawn tasks for each query
    let handles: Vec<_> = queries.into_iter().enumerate().map(|(id, query)| {
        let client = client.clone();
        let tx = tx.clone();
        let query = query.to_string();

        tokio::spawn(async move {
            let mut stream = match client.query(&query).stream().await {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(e)).await;
                    return;
                }
            };

            while let Some(message) = stream.next().await {
                match message {
                    Ok(Message::Assistant { content, .. }) => {
                        let event = StreamEvent::Content(format!("[{}] {}", id, content));
                        if tx.send(event).await.is_err() {
                            break;
                        }
                    }
                    Ok(Message::Result { stats, .. }) => {
                        let _ = tx.send(StreamEvent::Complete(stats)).await;
                        break;
                    }
                    Err(e) => {
                        let _ = tx.send(StreamEvent::Error(e)).await;
                        break;
                    }
                    _ => {}
                }
            }
        })
    }).collect();

    // Drop sender to close channel when all tasks complete
    drop(tx);

    // Process events as they arrive
    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Content(content) => {
                print!("{}", content);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            StreamEvent::Progress(tokens) => {
                eprintln!("Progress: {} tokens processed", tokens);
            }
            StreamEvent::Complete(stats) => {
                eprintln!("Query completed! Cost: ${:.4}", stats.total_cost_usd);
            }
            StreamEvent::Error(e) => {
                eprintln!("Stream error: {}", e);
            }
        }
    }

    // Wait for all tasks to complete
    for handle in handles {
        let _ = handle.await;
    }

    Ok(())
}
```

## Performance Considerations

### Memory Management

```rust
use std::collections::VecDeque;

struct BoundedBuffer {
    buffer: VecDeque<String>,
    max_size: usize,
}

impl BoundedBuffer {
    fn new(max_size: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            max_size,
        }
    }

    fn push(&mut self, content: String) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(content);
    }

    fn get_recent(&self, count: usize) -> Vec<&String> {
        self.buffer.iter().rev().take(count).collect()
    }
}

async fn memory_efficient_streaming(client: &Client, query: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = client.query(query).stream().await?;
    let mut buffer = BoundedBuffer::new(100); // Keep last 100 messages

    while let Some(message) = stream.next().await {
        match message? {
            Message::Assistant { content, .. } => {
                buffer.push(content.clone());
                print!("{}", content);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                
                // Optional: Process buffer periodically
                if buffer.buffer.len() % 10 == 0 {
                    // Perform some processing on recent messages
                    let recent = buffer.get_recent(5);
                    // ... process recent messages
                }
            }
            Message::Result { .. } => break,
            _ => {}
        }
    }

    Ok(())
}
```

### Throughput Optimization

```rust
async fn high_throughput_streaming(client: &Client, query: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Configure client for maximum throughput
    let client = Client::builder()
        .stream_format(StreamFormat::StreamJson)
        .timeout_secs(300) // Longer timeout for large responses
        .build();

    let mut stream = client.query(query).stream().await?;
    let mut content_buffer = String::with_capacity(8192); // Pre-allocate buffer
    let mut flush_counter = 0;

    while let Some(message) = stream.next().await {
        match message? {
            Message::Assistant { content, .. } => {
                content_buffer.push_str(&content);
                flush_counter += 1;

                // Batch output for better performance
                if flush_counter >= 10 || content_buffer.len() > 1024 {
                    print!("{}", content_buffer);
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    content_buffer.clear();
                    flush_counter = 0;
                }
            }
            Message::Result { .. } => {
                // Flush any remaining content
                if !content_buffer.is_empty() {
                    print!("{}", content_buffer);
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
```

## Testing Streaming Applications

### Mock Streaming for Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    async fn create_mock_stream() -> impl futures::Stream<Item = Result<Message, Box<dyn std::error::Error>>> {
        use futures::stream;
        
        let messages = vec![
            Ok(Message::Init { 
                meta: MessageMeta {
                    session_id: "test-session".to_string(),
                    timestamp: None,
                    cost_usd: None,
                    duration_ms: None,
                    tokens_used: None,
                }
            }),
            Ok(Message::Assistant { 
                content: "Hello, ".to_string(),
                meta: MessageMeta {
                    session_id: "test-session".to_string(),
                    timestamp: None,
                    cost_usd: Some(0.001),
                    duration_ms: Some(100),
                    tokens_used: Some(TokenUsage { input: 10, output: 2, total: 12 }),
                }
            }),
            Ok(Message::Assistant { 
                content: "world!".to_string(),
                meta: MessageMeta {
                    session_id: "test-session".to_string(),
                    timestamp: None,
                    cost_usd: Some(0.001),
                    duration_ms: Some(50),
                    tokens_used: Some(TokenUsage { input: 0, output: 1, total: 1 }),
                }
            }),
        ];

        stream::iter(messages)
    }

    #[tokio::test]
    async fn test_stream_processing() {
        let mut stream = create_mock_stream().await;
        let mut content = String::new();
        let mut message_count = 0;

        while let Some(message) = stream.next().await {
            match message.unwrap() {
                Message::Assistant { content: msg_content, .. } => {
                    content.push_str(&msg_content);
                    message_count += 1;
                }
                _ => {}
            }
        }

        assert_eq!(content, "Hello, world!");
        assert_eq!(message_count, 2);
    }
}
```

## Common Pitfalls and Solutions

### 1. Blocking I/O in Async Context

**Problem**: Using synchronous I/O operations in async streams
```rust
// ❌ Bad: Blocks the async runtime
while let Some(message) = stream.next().await {
    if let Message::Assistant { content, .. } = message? {
        std::thread::sleep(Duration::from_millis(100)); // Blocks!
        println!("{}", content);
    }
}
```

**Solution**: Use async alternatives
```rust
// ✅ Good: Non-blocking
while let Some(message) = stream.next().await {
    if let Message::Assistant { content, .. } = message? {
        tokio::time::sleep(Duration::from_millis(100)).await;
        println!("{}", content);
    }
}
```

### 2. Memory Leaks with Long Streams

**Problem**: Accumulating all streaming data in memory
```rust
// ❌ Bad: Unbounded memory growth
let mut all_content = String::new();
while let Some(message) = stream.next().await {
    if let Message::Assistant { content, .. } = message? {
        all_content.push_str(&content); // Memory keeps growing!
    }
}
```

**Solution**: Process data incrementally or use bounded buffers
```rust
// ✅ Good: Process as you go
while let Some(message) = stream.next().await {
    if let Message::Assistant { content, .. } = message? {
        // Process immediately
        process_content_chunk(&content).await?;
        // Don't store unless necessary
    }
}
```

### 3. Ignoring Stream Errors

**Problem**: Not handling stream errors properly
```rust
// ❌ Bad: Errors are ignored
while let Some(message) = stream.next().await {
    let msg = message.unwrap(); // Will panic on error!
    // ...
}
```

**Solution**: Handle errors gracefully
```rust
// ✅ Good: Proper error handling
while let Some(message) = stream.next().await {
    match message {
        Ok(msg) => {
            // Process message
        }
        Err(e) => {
            eprintln!("Stream error: {}", e);
            // Decide whether to continue, retry, or abort
            match e {
                claude_sdk_rs::Error::Timeout => continue, // Maybe retry
                _ => break, // Abort on other errors
            }
        }
    }
}
```

## Next Steps

You now have comprehensive knowledge of streaming responses! Here's what to explore next:

- **Part 5**: [Tool Integration](05-tool-integration.md) - Learn about tool usage in streaming contexts
- **Part 6**: [Session Management](06-session-management.md) - Persistent conversations and session handling

## Summary

Streaming responses unlock powerful real-time interactions with Claude:

```rust
// Basic streaming pattern
let mut stream = client.query("Your question").stream().await?;
while let Some(message) = stream.next().await {
    match message? {
        Message::Assistant { content, .. } => print!("{}", content),
        Message::Result { stats, .. } => println!("Cost: ${:.4}", stats.total_cost_usd),
        _ => {}
    }
}

// With error handling and progress
let mut stream = client.query("Complex query").stream().await?;
while let Some(message_result) = stream.next().await {
    match message_result {
        Ok(Message::Assistant { content, meta }) => {
            print!("{}", content);
            if let Some(tokens) = &meta.tokens_used {
                update_progress(tokens.total);
            }
        }
        Ok(Message::Result { stats, .. }) => {
            println!("Complete! Cost: ${:.4}", stats.total_cost_usd);
            break;
        }
        Err(e) => {
            eprintln!("Stream error: {}", e);
            // Handle error appropriately
        }
        _ => {}
    }
}
```

Streaming responses transform static request-response patterns into dynamic, interactive experiences that feel natural and responsive to users!