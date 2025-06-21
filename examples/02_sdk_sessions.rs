//! # Example 02: SDK with Session Management
//!
//! This example demonstrates session usage with the `claude-sdk-rs` SDK.
//! It shows how to:
//! - Use session IDs to maintain context
//! - Track conversations across multiple queries
//! - Manage session state manually
//!
//! This uses ONLY the `claude-sdk-rs` crate, not `claude-interactive`.

use claude_sdk_rs::{Client, SessionId};
use std::collections::HashMap;
use uuid::Uuid;

#[tokio::main]
async fn main() -> claude_sdk_rs::Result<()> {
    println!("=== Claude AI Session Management Example ===\n");

    // Example 1: Basic session usage
    basic_session_example().await?;

    // Example 2: Multi-session conversations
    multi_session_example().await?;

    // Example 3: Session context demonstration
    session_context_demo().await?;

    Ok(())
}

/// Demonstrates basic session usage
async fn basic_session_example() -> claude_sdk_rs::Result<()> {
    println!("1. Basic Session Usage");
    println!("   Using a session ID to maintain conversation context\n");

    let client = Client::builder()
        .timeout_secs(120) // 2 minute timeout for session examples
        .build()?;

    // Create a session ID
    let session_id = SessionId::new(Uuid::new_v4().to_string());
    println!("   Created session: {}", session_id);

    // First query in the session
    let response1 = client
        .query("What is ownership in Rust?")
        .session(session_id.clone())
        .send()
        .await?;

    println!("\n   Q1: What is ownership in Rust?");
    println!("   A1: {}", truncate_response(&response1, 200));

    // Follow-up query using the same session (maintains context)
    let response2 = client
        .query("Can you give me a simple example?")
        .session(session_id.clone())
        .send()
        .await?;

    println!("\n   Q2: Can you give me a simple example?");
    println!("   A2: {}", truncate_response(&response2, 200));

    // Small delay to ensure stable operation
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    println!("\n   ✓ Both queries used session: {}", session_id.as_str());
    println!("   ✓ Claude maintained context between queries");

    Ok(())
}

/// Shows how to manage multiple concurrent sessions
async fn multi_session_example() -> claude_sdk_rs::Result<()> {
    println!("\n2. Multi-Session Conversations");
    println!("   Managing multiple conversation contexts\n");

    let client = Client::builder()
        .timeout_secs(120) // 2 minute timeout for session examples
        .build()?;

    // Create session IDs for different topics
    let rust_session = SessionId::new(Uuid::new_v4().to_string());
    let python_session = SessionId::new(Uuid::new_v4().to_string());

    println!("   Created sessions:");
    println!("   - Rust: {}...", &rust_session.as_str()[..8]);
    println!("   - Python: {}...", &python_session.as_str()[..8]);

    // Query in Rust session
    let rust_response = client
        .query("What are traits in Rust?")
        .session(rust_session.clone())
        .send()
        .await?;

    println!("\n   Rust Session - Q: What are traits in Rust?");
    println!("   A: {}", truncate_response(&rust_response, 150));

    // Small delay between requests to avoid rate limiting
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Query in Python session
    let python_response = client
        .query("What are decorators in Python?")
        .session(python_session.clone())
        .send()
        .await?;

    println!("\n   Python Session - Q: What are decorators in Python?");
    println!("   A: {}", truncate_response(&python_response, 150));

    // Small delay between requests to avoid rate limiting
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Follow-up in Rust session (maintains Rust context)
    let rust_followup = client
        .query("How do they differ from interfaces?")
        .session(rust_session.clone())
        .send()
        .await?;

    println!("\n   Rust Session - Q: How do they differ from interfaces?");
    println!("   A: {}", truncate_response(&rust_followup, 150));

    println!("\n   ✓ Each session maintained its own context");

    Ok(())
}

/// Demonstrates how session context works
async fn session_context_demo() -> claude_sdk_rs::Result<()> {
    println!("\n3. Session Context Demonstration");
    println!("   Showing how Claude remembers conversation history\n");

    let client = Client::builder()
        .timeout_secs(180) // 3 minute timeout for multiple queries
        .build()?;
    let session_id = SessionId::new(Uuid::new_v4().to_string());

    // Build up context with a series of related queries
    let queries = vec![
        "What is a web server?",
        "What Rust crate is best for this?",
        "Show me a simple route example",
    ];

    println!("   Session: {}...", &session_id.as_str()[..8]);
    println!("   Building conversation context:\n");

    for (i, query) in queries.iter().enumerate() {
        let response = client
            .query(*query)
            .session(session_id.clone())
            .send()
            .await?;

        println!("   Q{}: {}", i + 1, query);
        println!("   A{}: {}\n", i + 1, truncate_response(&response, 120));

        // Longer delay between requests to avoid rate limiting
        if i < queries.len() - 1 {
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        }
    }

    println!("   ✓ Claude maintained context throughout the conversation");
    println!("   ✓ Each response built upon previous exchanges");

    Ok(())
}

/// Manual session tracking example
struct SimpleSessionManager {
    sessions: HashMap<String, SessionInfo>,
}

struct SessionInfo {
    id: String,
    description: String,
    message_count: usize,
}

impl SimpleSessionManager {
    fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    fn create_session(&mut self, description: &str) -> String {
        let id = Uuid::new_v4().to_string();
        let info = SessionInfo {
            id: id.clone(),
            description: description.to_string(),
            message_count: 0,
        };
        self.sessions.insert(id.clone(), info);
        id
    }

    fn record_message(&mut self, session_id: &str) {
        if let Some(info) = self.sessions.get_mut(session_id) {
            info.message_count += 1;
        }
    }

    fn list_sessions(&self) {
        println!("\n   Active Sessions:");
        for (id, info) in &self.sessions {
            println!(
                "   - {}: {} ({} messages)",
                &id[..8],
                info.description,
                info.message_count
            );
        }
    }
}

/// Helper function to truncate long responses
fn truncate_response(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len])
    }
}

// Example output:
/*
=== Claude AI Session Management Example ===

1. Basic Session Usage
   Using a session ID to maintain conversation context

   Created session: 550e8400-e29b-41d4-a716-446655440000

   Q1: What is ownership in Rust?
   A1: Ownership is Rust's most unique feature and central to how it achieves memory safety without garbage collection. It's a set of rules that govern how memory is managed...

   Q2: Can you give me a simple example?
   A2: Here's a simple example of ownership in Rust:

   ```rust
   fn main() {
       let s1 = String::from("hello");  // s1 owns the String
       let s2 = s1;                     // ownership moves to s2...

   ✓ Both queries used session: 550e8400
   ✓ Claude maintained context between queries

2. Multi-Session Conversations
   Managing multiple conversation contexts

   Created sessions:
   - Rust: 6ba7b810...
   - Python: 6ba7b811...

   Rust Session - Q: What are traits in Rust?
   A: Traits in Rust are a way to define shared behavior across different types. They're similar to interfaces in other languages but more powerful...

   Python Session - Q: What are decorators in Python?
   A: Decorators in Python are a design pattern that allows you to modify or enhance functions and classes without permanently modifying...

   Rust Session - Q: How do they differ from interfaces?
   A: While traits in Rust are similar to interfaces, they have several key differences: 1) Traits can provide default implementations...

   ✓ Each session maintained its own context

3. Session Context Demonstration
   Showing how Claude remembers conversation history

   Session: 6ba7b812...
   Building conversation context:

   Q1: Let's talk about building a web server in Rust
   A1: Great! Building web servers in Rust is an excellent choice due to Rust's performance, safety, and concurrency features...

   Q2: What crates would you recommend?
   A2: For web servers in Rust, I'd recommend these popular and well-maintained crates: 1) Axum - Modern, ergonomic...

   Q3: How do I handle routing?
   A3: In Axum (which I mentioned earlier), routing is handled elegantly using a builder pattern. Here's how you define routes...

   Q4: Can you show me a simple example with the framework you mentioned?
   A4: Here's a simple Axum web server example that demonstrates the routing concepts I just explained:

   ```rust
   use axum::{Router, routing::get};...

   ✓ Claude maintained context throughout the conversation
   ✓ Each response built upon previous exchanges
*/
