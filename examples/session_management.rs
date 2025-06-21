//! # Session Management Example
//!
//! This example demonstrates session management capabilities of the claude-sdk-rs SDK.
//! It shows how to:
//! - Create and manage conversation sessions
//! - Use --continue flag for session continuation
//! - Use --resume flag to resume specific sessions
//! - Maintain context across multiple queries
//! - Switch between different sessions
//! - Persist and retrieve session data
//! - Implement session-based workflows
//!
//! Sessions allow you to maintain conversation context and build
//! more sophisticated multi-turn interactions with Claude.

use claude_sdk_rs::{Client, StreamFormat};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Claude SDK Session Management Example ===\n");

    // Example 1: Basic session usage
    basic_session_usage().await?;

    // Example 2: Multiple session management
    multiple_session_management().await?;

    // Example 3: Session-based conversation
    session_based_conversation().await?;

    // Example 4: Session context preservation
    session_context_preservation().await?;

    // Example 5: Session workflow patterns
    session_workflow_patterns().await?;

    // Example 6: Continue session functionality
    continue_session_example().await?;

    // Example 7: Resume specific session functionality
    resume_session_example().await?;

    println!("Session management example completed successfully!");
    Ok(())
}

/// Demonstrates basic session creation and usage
async fn basic_session_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Session Usage");
    println!("   Creating and using a single session\n");

    // Create client with JSON format to access session metadata
    let client = Client::builder()
        .stream_format(StreamFormat::Json)
        .build()?;

    println!("   Starting conversation with session tracking:");

    // First query - this will create a new session
    let response1 = client
        .query("Hello! My name is Alice. I'm a software engineer.")
        .send_full()
        .await?;

    println!("   Query 1: Hello! My name is Alice. I'm a software engineer.");
    println!("   Response 1: {}", response1.content);

    let session_id = response1
        .metadata
        .as_ref()
        .map(|m| m.session_id.clone())
        .unwrap_or_else(|| "unknown".to_string());

    println!("   Session ID: {}\n", session_id);

    // Second query - should maintain context from first query
    let response2 = client
        .query("What's my name and profession?")
        .send_full()
        .await?;

    println!("   Query 2: What's my name and profession?");
    println!("   Response 2: {}", response2.content);

    // Verify same session
    if let Some(metadata) = response2.metadata {
        println!(
            "   Session ID: {} (should be same as above)",
            metadata.session_id
        );
    }
    println!();

    Ok(())
}

/// Demonstrates managing multiple concurrent sessions
async fn multiple_session_management() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Multiple Session Management");
    println!("   Creating and managing multiple independent sessions\n");

    // Create separate clients for different sessions
    // Note: In practice, sessions are managed by the Claude CLI automatically
    // but we can demonstrate the concept by using different client instances
    // or by explicitly managing conversation context

    let mut sessions = HashMap::new();

    // Session 1: Math tutoring
    println!("   Creating Math Tutoring Session:");
    let math_client = Client::builder()
        .system_prompt("You are a math tutor. Help with mathematical concepts.")
        .stream_format(StreamFormat::Json)
        .build()?;

    let math_response = math_client
        .query("I need help understanding calculus derivatives.")
        .send_full()
        .await?;

    println!("   Math Query: I need help understanding calculus derivatives.");
    println!("   Math Response: {}", math_response.content);

    if let Some(metadata) = math_response.metadata {
        sessions.insert("math", metadata.session_id.clone());
        println!("   Math Session ID: {}", metadata.session_id);
    }
    println!();

    // Session 2: Creative writing
    println!("   Creating Creative Writing Session:");
    let writing_client = Client::builder()
        .system_prompt("You are a creative writing assistant. Help with storytelling.")
        .stream_format(StreamFormat::Json)
        .build()?;

    let writing_response = writing_client
        .query("I want to write a story about a dragon who loves to cook.")
        .send_full()
        .await?;

    println!("   Writing Query: I want to write a story about a dragon who loves to cook.");
    println!("   Writing Response: {}", writing_response.content);

    if let Some(metadata) = writing_response.metadata {
        sessions.insert("writing", metadata.session_id.clone());
        println!("   Writing Session ID: {}", metadata.session_id);
    }
    println!();

    // Continue conversations in each session
    println!("   Continuing Math Session:");
    let math_followup = math_client
        .query("Can you give me a simple example of a derivative?")
        .send_full()
        .await?;
    println!("   Math Follow-up: {}", math_followup.content);
    println!();

    println!("   Continuing Writing Session:");
    let writing_followup = writing_client
        .query("What should be the dragon's first cooking adventure?")
        .send_full()
        .await?;
    println!("   Writing Follow-up: {}", writing_followup.content);
    println!();

    // Display session summary
    println!("   Active Sessions:");
    for (name, session_id) in sessions {
        println!("   - {}: {}", name, session_id);
    }
    println!();

    Ok(())
}

/// Demonstrates a full session-based conversation
async fn session_based_conversation() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Session-Based Conversation");
    println!("   Building a multi-turn conversation with context\n");

    let client = Client::builder()
        .system_prompt("You are a helpful assistant. Remember details from our conversation.")
        .stream_format(StreamFormat::Json)
        .build()?;

    // Simulate a conversation about planning a trip
    let conversation = vec![
        "I'm planning a trip to Japan next month.",
        "I'm interested in both modern cities and traditional culture.",
        "What cities would you recommend for my trip?",
        "How many days should I spend in each city?",
        "What's the best way to travel between these cities?",
        "Can you summarize our travel plan?",
    ];

    println!("   Starting travel planning conversation:\n");

    for (i, query) in conversation.into_iter().enumerate() {
        println!("   Turn {}: {}", i + 1, query);

        let response = client.query(query).send_full().await?;
        println!("   Response: {}\n", response.content);

        // Add small delay to make conversation feel more natural
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    Ok(())
}

/// Demonstrates session context preservation across complex interactions
async fn session_context_preservation() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Session Context Preservation");
    println!("   Testing how well context is maintained across queries\n");

    let client = Client::builder()
        .system_prompt("You are a helpful assistant. Pay attention to details and context.")
        .stream_format(StreamFormat::Json)
        .build()?;

    // Set up initial context
    println!("   Setting up context:");
    let setup_queries = vec![
        "I'm working on a Rust project called 'weather-app'.",
        "It uses tokio for async operations and serde for JSON parsing.",
        "The main function fetches weather data from an API.",
    ];

    for query in setup_queries {
        println!("   Setup: {}", query);
        let response = client.query(query).send().await?;
        println!(
            "   Acknowledged: {}\n",
            response.split('.').next().unwrap_or(&response)
        );
    }

    // Test context preservation with specific questions
    println!("   Testing context preservation:");
    let test_queries = vec![
        "What's the name of my project?",
        "What async runtime am I using?",
        "What crate handles JSON in my project?",
        "Can you suggest error handling for the API calls?",
    ];

    for query in test_queries {
        println!("   Question: {}", query);
        let response = client.query(query).send().await?;
        println!("   Answer: {}\n", response);
    }

    Ok(())
}

/// Demonstrates advanced session workflow patterns
async fn session_workflow_patterns() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Session Workflow Patterns");
    println!("   Implementing common session-based workflows\n");

    // Pattern 1: Information gathering workflow
    println!("   a) Information Gathering Workflow:");
    information_gathering_workflow().await?;

    // Pattern 2: Iterative refinement workflow
    println!("   b) Iterative Refinement Workflow:");
    iterative_refinement_workflow().await?;

    // Pattern 3: Context switching workflow
    println!("   c) Context Switching Workflow:");
    context_switching_workflow().await?;

    Ok(())
}

/// Information gathering workflow pattern
async fn information_gathering_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder()
        .system_prompt("You are conducting an information gathering session. Ask follow-up questions when needed.")
        .stream_format(StreamFormat::Json)
        .build()?;

    println!("   Gathering information about a user's coding project:\n");

    let workflow_steps = vec![
        "I want to build a web application.",
        "I prefer using Rust for the backend.",
        "I need it to handle user authentication and a database.",
        "The frontend should be responsive and modern.",
        "Now please summarize what we've discussed and provide recommendations.",
    ];

    for (i, step) in workflow_steps.into_iter().enumerate() {
        println!("   Step {}: {}", i + 1, step);
        let response = client.query(step).send().await?;
        println!("   Response: {}\n", response);
    }

    Ok(())
}

/// Iterative refinement workflow pattern
async fn iterative_refinement_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder()
        .system_prompt("You help refine and improve ideas through iteration.")
        .stream_format(StreamFormat::Json)
        .build()?;

    println!("   Refining a function design through iterations:\n");

    let refinement_steps = vec![
        "I need a function to process user input for a CLI application.",
        "Actually, it should also validate the input format.",
        "And it should provide helpful error messages for invalid input.",
        "Can you add logging support to track processing steps?",
        "Now show me the final refined function with all improvements.",
    ];

    for (i, step) in refinement_steps.into_iter().enumerate() {
        println!("   Iteration {}: {}", i + 1, step);
        let response = client.query(step).send().await?;
        println!("   Response: {}\n", response);
    }

    Ok(())
}

/// Context switching workflow pattern
async fn context_switching_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder()
        .system_prompt("You can handle multiple topics and return to previous discussions.")
        .stream_format(StreamFormat::Json)
        .build()?;

    println!("   Demonstrating context switching:\n");

    let context_steps = vec![
        "Let's discuss Rust memory management.",
        "Actually, let me ask about Python list comprehensions first.",
        "Thanks! Now back to Rust - explain ownership rules.",
        "One more Python question - what about async/await?",
        "Finally, back to Rust - how does borrowing relate to what we discussed?",
    ];

    for (i, step) in context_steps.into_iter().enumerate() {
        println!("   Context {}: {}", i + 1, step);
        let response = client.query(step).send().await?;
        println!("   Response: {}\n", response);
    }

    Ok(())
}

/// Demonstrates the --continue flag for session continuation
async fn continue_session_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Continue Session Example");
    println!("   Using --continue flag to resume the most recent session\n");

    // Create a client that will continue the most recent session
    let continue_client = Client::builder()
        .stream_format(StreamFormat::Json)
        .continue_session() // This adds the --continue flag
        .build()?;

    println!("   Starting new conversation that will continue previous session:");

    let response = continue_client
        .query("Can you remind me what we were discussing?")
        .send_full()
        .await?;

    println!("   Query: Can you remind me what we were discussing?");
    println!("   Response: {}", response.content);

    if let Some(metadata) = response.metadata {
        println!("   Session ID: {}", metadata.session_id);
        if let Some(cost) = metadata.cost_usd {
            println!("   Cost: ${:.6}", cost);
        }
    }
    println!();

    // Continue the conversation
    let follow_up = continue_client
        .query("Let's continue from where we left off.")
        .send_full()
        .await?;

    println!("   Follow-up: Let's continue from where we left off.");
    println!("   Response: {}", follow_up.content);
    println!();

    Ok(())
}

/// Demonstrates the --resume flag for resuming specific sessions
async fn resume_session_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("7. Resume Specific Session Example");
    println!("   Using --resume flag to continue a specific session by ID\n");

    // First, create a session and get its ID
    println!("   Step 1: Creating a new session to get a session ID:");
    let initial_client = Client::builder()
        .stream_format(StreamFormat::Json)
        .system_prompt("You are helping plan a vacation to Italy.")
        .build()?;

    let initial_response = initial_client
        .query("I'm planning a 7-day trip to Italy. What cities should I visit?")
        .send_full()
        .await?;

    println!("   Initial Query: I'm planning a 7-day trip to Italy. What cities should I visit?");
    println!("   Response: {}", initial_response.content);

    let session_id = initial_response
        .metadata
        .as_ref()
        .map(|m| m.session_id.clone())
        .unwrap_or_else(|| "example_session_id".to_string());

    println!("   Session ID captured: {}\n", session_id);

    // Now simulate resuming that specific session later
    println!("   Step 2: Later, resuming the specific session:");
    let resume_client = Client::builder()
        .stream_format(StreamFormat::Json)
        .resume_session(&session_id) // This adds the --resume flag with session ID
        .build()?;

    let resume_response = resume_client
        .query("Based on our previous discussion, which city should I visit first?")
        .send_full()
        .await?;

    println!("   Resume Query: Based on our previous discussion, which city should I visit first?");
    println!("   Response: {}", resume_response.content);

    if let Some(metadata) = resume_response.metadata {
        println!("   Resumed Session ID: {}", metadata.session_id);
        println!("   Should match: {}", session_id);

        if metadata.session_id == session_id {
            println!("   ✓ Successfully resumed the correct session!");
        } else {
            println!("   ⚠ Session ID mismatch - this may be expected in some cases");
        }
    }
    println!();

    // Continue the resumed conversation
    let continued_response = resume_client
        .query("What about accommodation recommendations for the first city?")
        .send_full()
        .await?;

    println!("   Continued Query: What about accommodation recommendations for the first city?");
    println!("   Response: {}", continued_response.content);
    println!();

    Ok(())
}

// Example output:
/*
=== Claude SDK Session Management Example ===

1. Basic Session Usage
   Creating and using a single session

   Starting conversation with session tracking:
   Query 1: Hello! My name is Alice. I'm a software engineer.
   Response 1: Hello Alice! Nice to meet you. It's great to meet a fellow software engineer...
   Session ID: 550e8400-e29b-41d4-a716-446655440000

   Query 2: What's my name and profession?
   Response 2: Your name is Alice and you're a software engineer, as you mentioned earlier.
   Session ID: 550e8400-e29b-41d4-a716-446655440000 (should be same as above)

2. Multiple Session Management
   Creating and managing multiple independent sessions

   Creating Math Tutoring Session:
   Math Query: I need help understanding calculus derivatives.
   Math Response: I'd be happy to help you understand derivatives in calculus...
   Math Session ID: 550e8400-e29b-41d4-a716-446655440001

   Creating Creative Writing Session:
   Writing Query: I want to write a story about a dragon who loves to cook.
   Writing Response: What a delightful and unique concept! A culinary dragon opens up...
   Writing Session ID: 550e8400-e29b-41d4-a716-446655440002

   Continuing Math Session:
   Math Follow-up: Let's say we have f(x) = x². The derivative f'(x) = 2x...

   Continuing Writing Session:
   Writing Follow-up: For the dragon's first cooking adventure, consider...

   Active Sessions:
   - math: 550e8400-e29b-41d4-a716-446655440001
   - writing: 550e8400-e29b-41d4-a716-446655440002

Session management example completed successfully!
*/
