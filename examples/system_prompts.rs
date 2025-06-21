//! # System Prompts Example
//!
//! This example demonstrates system prompt control and conversation management features of the claude-sdk-rs SDK.
//! It shows how to:
//! - Use append_system_prompt to modify Claude's behavior
//! - Control conversation length with max_turns
//! - Combine system prompt modifications with turn limits
//! - Different system prompt scenarios (formatting, role-based, response style)
//! - Validation and error handling for system prompt configurations
//!
//! System prompt control allows you to dynamically adjust Claude's behavior without
//! replacing the entire system context, while max_turns provides conversation flow control.

use claude_sdk_rs::{Client, StreamFormat};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Claude SDK System Prompts Example ===\n");

    // Example 1: Basic append system prompt usage
    basic_append_system_prompt().await?;

    // Example 2: Max turns conversation control
    max_turns_conversation_control().await?;

    // Example 3: Combining system prompts with conversation limits
    combined_system_prompt_and_turns().await?;

    // Example 4: Different system prompt scenarios
    system_prompt_scenarios().await?;

    // Example 5: Validation and error handling
    validation_and_error_handling().await?;

    println!("System prompts example completed successfully!");
    Ok(())
}

/// Demonstrates basic append_system_prompt functionality
async fn basic_append_system_prompt() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Append System Prompt Usage");
    println!("   Modifying Claude's behavior with additional instructions\n");

    // Example 1a: Adding concise response instruction
    println!("   Example 1a: Requesting concise responses");
    let concise_client = Client::builder()
        .append_system_prompt("Always provide concise, one-sentence answers.")
        .stream_format(StreamFormat::Json)
        .build()?;

    let response = concise_client
        .query("Explain what Rust programming language is")
        .send_full()
        .await?;

    println!("   Query: Explain what Rust programming language is");
    println!("   Response: {}", response.content);
    if let Some(metadata) = response.metadata {
        println!("   Session ID: {}", metadata.session_id);
    }
    println!();

    // Example 1b: Adding formatting instructions
    println!("   Example 1b: Adding specific formatting requirements");
    let formatted_client = Client::builder()
        .append_system_prompt("Format all responses as numbered lists with exactly 3 points.")
        .stream_format(StreamFormat::Json)
        .build()?;

    let response2 = formatted_client
        .query("What are the benefits of using Rust?")
        .send_full()
        .await?;

    println!("   Query: What are the benefits of using Rust?");
    println!("   Response: {}", response2.content);
    println!();

    // Example 1c: Adding role-based instructions
    println!("   Example 1c: Adding role-based behavior modification");
    let expert_client = Client::builder()
        .append_system_prompt("You are now speaking as a senior software engineer with 10+ years of experience. Include practical insights.")
        .stream_format(StreamFormat::Json)
        .build()?;

    let response3 = expert_client
        .query("Should I use async/await in Rust?")
        .send_full()
        .await?;

    println!("   Query: Should I use async/await in Rust?");
    println!("   Response: {}", response3.content);
    println!();

    Ok(())
}

/// Demonstrates max_turns conversation control
async fn max_turns_conversation_control() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Max Turns Conversation Control");
    println!("   Limiting conversation length with max_turns parameter\n");

    // Example 2a: Single turn conversation
    println!("   Example 2a: Single turn conversation (max_turns = 1)");
    let single_turn_client = Client::builder()
        .max_turns(1)
        .stream_format(StreamFormat::Json)
        .build()?;

    let response = single_turn_client
        .query("Hello! Can you help me with Rust programming?")
        .send_full()
        .await?;

    println!("   Query: Hello! Can you help me with Rust programming?");
    println!("   Response: {}", response.content);
    if let Some(metadata) = response.metadata {
        println!("   Session ID: {} (max 1 turn)", metadata.session_id);
    }
    println!();

    // Example 2b: Short conversation limit
    println!("   Example 2b: Short conversation (max_turns = 3)");
    let short_conversation_client = Client::builder()
        .max_turns(3)
        .stream_format(StreamFormat::Json)
        .build()?;

    let response2 = short_conversation_client
        .query("Start a conversation about learning Rust")
        .send_full()
        .await?;

    println!("   Query: Start a conversation about learning Rust");
    println!("   Response: {}", response2.content);
    if let Some(metadata) = response2.metadata {
        println!("   Session ID: {} (max 3 turns)", metadata.session_id);
    }
    println!();

    // Example 2c: Controlled Q&A session
    println!("   Example 2c: Controlled Q&A session (max_turns = 5)");
    let qa_client = Client::builder()
        .max_turns(5)
        .append_system_prompt("You are conducting a brief Q&A session. Keep track of how many questions have been asked.")
        .stream_format(StreamFormat::Json)
        .build()?;

    let response3 = qa_client
        .query("I want to learn about Rust ownership. Can you start with the basics?")
        .send_full()
        .await?;

    println!("   Query: I want to learn about Rust ownership. Can you start with the basics?");
    println!("   Response: {}", response3.content);
    if let Some(metadata) = response3.metadata {
        println!(
            "   Session ID: {} (max 5 turns for Q&A)",
            metadata.session_id
        );
    }
    println!();

    Ok(())
}

/// Demonstrates combining system prompts with conversation limits
async fn combined_system_prompt_and_turns() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Combining System Prompts with Conversation Limits");
    println!("   Using both append_system_prompt and max_turns together\n");

    // Example 3a: Tutorial with limited turns
    println!("   Example 3a: Brief tutorial session (concise + 2 turns)");
    let tutorial_client = Client::builder()
        .append_system_prompt("You are providing a brief tutorial. Be concise but comprehensive. Mention that this is a limited session.")
        .max_turns(2)
        .stream_format(StreamFormat::Json)
        .build()?;

    let response = tutorial_client
        .query("Teach me about Rust error handling")
        .send_full()
        .await?;

    println!("   Query: Teach me about Rust error handling");
    println!("   Response: {}", response.content);
    println!();

    // Example 3b: Code review session with limits
    println!("   Example 3b: Code review session (expert role + 3 turns)");
    let review_client = Client::builder()
        .append_system_prompt("You are conducting a code review session. Be direct and focus on critical issues. This is a time-limited session.")
        .max_turns(3)
        .stream_format(StreamFormat::Json)
        .build()?;

    let response2 = review_client
        .query("Review this Rust code approach: using unwrap() everywhere for simplicity")
        .send_full()
        .await?;

    println!("   Query: Review this Rust code approach: using unwrap() everywhere for simplicity");
    println!("   Response: {}", response2.content);
    println!();

    // Example 3c: Quick consultation with specific output format
    println!("   Example 3c: Quick consultation (format + 1 turn)");
    let consultation_client = Client::builder()
        .append_system_prompt("Provide consultation in this exact format: PROBLEM: [brief], SOLUTION: [specific action], REASONING: [why]. This is a single-response consultation.")
        .max_turns(1)
        .stream_format(StreamFormat::Json)
        .build()?;

    let response3 = consultation_client
        .query("My Rust program is running slowly with large vectors")
        .send_full()
        .await?;

    println!("   Query: My Rust program is running slowly with large vectors");
    println!("   Response: {}", response3.content);
    println!();

    Ok(())
}

/// Demonstrates different system prompt scenarios
async fn system_prompt_scenarios() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Different System Prompt Scenarios");
    println!("   Various real-world use cases for system prompt modification\n");

    // Example 4a: Debugging assistant
    println!("   Example 4a: Debugging assistant mode");
    let debug_client = Client::builder()
        .append_system_prompt("You are now in debugging mode. Always ask for specific error messages, code snippets, and system details. Be methodical in your troubleshooting approach.")
        .stream_format(StreamFormat::Json)
        .build()?;

    let response = debug_client
        .query("My Rust code won't compile")
        .send_full()
        .await?;

    println!("   Query: My Rust code won't compile");
    println!("   Response: {}", response.content);
    println!();

    // Example 4b: API documentation generator
    println!("   Example 4b: API documentation generator");
    let doc_client = Client::builder()
        .append_system_prompt("Generate API documentation in standard Rust doc format. Include examples, parameters, return types, and potential errors for all functions discussed.")
        .stream_format(StreamFormat::Json)
        .build()?;

    let response2 = doc_client
        .query("Document a function that reads a file and returns its contents")
        .send_full()
        .await?;

    println!("   Query: Document a function that reads a file and returns its contents");
    println!("   Response: {}", response2.content);
    println!();

    // Example 4c: Performance analyzer
    println!("   Example 4c: Performance analysis mode");
    let perf_client = Client::builder()
        .append_system_prompt("You are analyzing code for performance issues. Always mention Big O complexity, memory usage patterns, and suggest specific Rust optimizations like using iterators, avoiding clones, or choosing appropriate data structures.")
        .max_turns(2)
        .stream_format(StreamFormat::Json)
        .build()?;

    let response3 = perf_client
        .query("Analyze the performance of nested loops processing a Vec<Vec<i32>>")
        .send_full()
        .await?;

    println!("   Query: Analyze the performance of nested loops processing a Vec<Vec<i32>>");
    println!("   Response: {}", response3.content);
    println!();

    // Example 4d: Learning assistant with progression
    println!("   Example 4d: Learning assistant with skill progression");
    let learning_client = Client::builder()
        .append_system_prompt("You are a learning assistant. Assess the user's current level from their questions and adjust your explanations accordingly. Start with fundamentals and build up complexity. Mention prerequisite concepts when needed.")
        .max_turns(4)
        .stream_format(StreamFormat::Json)
        .build()?;

    let response4 = learning_client
        .query("What are lifetimes in Rust?")
        .send_full()
        .await?;

    println!("   Query: What are lifetimes in Rust?");
    println!("   Response: {}", response4.content);
    println!();

    Ok(())
}

/// Demonstrates validation and error handling scenarios
async fn validation_and_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Validation and Error Handling");
    println!("   Testing system prompt and max_turns validation\n");

    // Example 5a: Valid configuration
    println!("   Example 5a: Valid system prompt and max_turns configuration");
    let valid_result = Client::builder()
        .append_system_prompt("Provide helpful and accurate responses.")
        .max_turns(5)
        .stream_format(StreamFormat::Json)
        .build();

    match valid_result {
        Ok(_) => println!("   ✓ Valid configuration accepted"),
        Err(e) => println!("   ✗ Unexpected error: {}", e),
    }

    // Example 5b: Testing max_turns validation (should fail with 0)
    println!("   Example 5b: Invalid max_turns value (0)");
    let invalid_turns_result = Client::builder().max_turns(0).build();

    match invalid_turns_result {
        Ok(_) => println!("   ⚠ Invalid turns value not detected (unexpected)"),
        Err(e) => println!("   ✓ Invalid turns correctly detected: {}", e),
    }

    // Example 5c: Testing system prompt length limits
    println!("   Example 5c: System prompt length validation");
    let long_prompt = "A".repeat(15000); // Exceeds MAX_SYSTEM_PROMPT_LENGTH (10,000)
    let long_prompt_result = Client::builder().append_system_prompt(long_prompt).build();

    match long_prompt_result {
        Ok(_) => println!("   ⚠ Long prompt not detected (unexpected)"),
        Err(e) => println!("   ✓ Long prompt correctly detected: {}", e),
    }

    // Example 5d: Testing system_prompt + append_system_prompt conflict
    println!("   Example 5d: system_prompt + append_system_prompt conflict");
    let conflict_result = Client::builder()
        .system_prompt("You are a helpful assistant.")
        .append_system_prompt("Additionally, be concise.")
        .build();

    match conflict_result {
        Ok(_) => println!("   ⚠ Conflict not detected (unexpected)"),
        Err(e) => println!("   ✓ Conflict correctly detected: {}", e),
    }

    // Example 5e: Valid max_turns values
    println!("   Example 5e: Various valid max_turns values");

    let valid_turns = [1, 3, 10, 50, 100];
    for turns in valid_turns.iter() {
        let result = Client::builder().max_turns(*turns).build();

        match result {
            Ok(_) => println!("   ✓ max_turns({}) accepted", turns),
            Err(e) => println!("   ✗ max_turns({}) rejected: {}", turns, e),
        }
    }

    // Example 5f: Empty append_system_prompt (should be valid)
    println!("   Example 5f: Empty append_system_prompt");
    let empty_prompt_result = Client::builder().append_system_prompt("").build();

    match empty_prompt_result {
        Ok(_) => println!("   ✓ Empty append_system_prompt accepted"),
        Err(e) => println!("   ✗ Empty append_system_prompt rejected: {}", e),
    }
    println!();

    Ok(())
}

// Example output:
/*
=== Claude SDK System Prompts Example ===

1. Basic Append System Prompt Usage
   Modifying Claude's behavior with additional instructions

   Example 1a: Requesting concise responses
   Query: Explain what Rust programming language is
   Response: Rust is a systems programming language focused on safety, speed, and concurrency without garbage collection.
   Session ID: 550e8400-e29b-41d4-a716-446655440003

   Example 1b: Adding specific formatting requirements
   Query: What are the benefits of using Rust?
   Response: 1. Memory safety without garbage collection through ownership system
2. Zero-cost abstractions providing high performance
3. Excellent concurrency support with thread safety guarantees

   Example 1c: Adding role-based behavior modification
   Query: Should I use async/await in Rust?
   Response: From my experience, async/await in Rust is excellent for I/O-bound applications like web servers or network clients. However, avoid it for CPU-bound tasks where it adds overhead without benefits. Start with tokio for the ecosystem, and consider the learning curve of futures and lifetimes in async contexts.

2. Max Turns Conversation Control
   Limiting conversation length with max_turns parameter

   Example 2a: Single turn conversation (max_turns = 1)
   Query: Hello! Can you help me with Rust programming?
   Response: Hello! Yes, I'd be happy to help you with Rust programming. What specific topic or question do you have?
   Session ID: 550e8400-e29b-41d4-a716-446655440004 (max 1 turn)

   Example 2b: Short conversation (max_turns = 3)
   Query: Start a conversation about learning Rust
   Response: Great choice! Rust is an excellent language to learn. What's your programming background? Are you coming from languages like Python, JavaScript, C++, or is this your first systems language?
   Session ID: 550e8400-e29b-41d4-a716-446655440005 (max 3 turns)

   Example 2c: Controlled Q&A session (max_turns = 5)
   Query: I want to learn about Rust ownership. Can you start with the basics?
   Response: Perfect! This is question 1 of our 5-turn Q&A session on Rust ownership.

Ownership is Rust's unique approach to memory management. Every value has an "owner" (a variable), and when the owner goes out of scope, the value is dropped. Key rules: 1) Each value has exactly one owner, 2) When the owner goes out of scope, the value is dropped, 3) Ownership can be transferred (moved) between variables.

What specific aspect would you like me to explain next?
   Session ID: 550e8400-e29b-41d4-a716-446655440006 (max 5 turns for Q&A)

3. Combining System Prompts with Conversation Limits
   Using both append_system_prompt and max_turns together

   Example 3a: Brief tutorial session (concise + 2 turns)
   Query: Teach me about Rust error handling
   Response: In this limited tutorial session, here are Rust's key error handling concepts:

**Result<T, E>**: For recoverable errors. Use `match`, `if let`, or `?` operator.
**panic!**: For unrecoverable errors. Use `unwrap()` sparingly.
**Option<T>**: For nullable values. Similar handling to Result.

Key pattern: `fn might_fail() -> Result<String, Error>` then use `?` to propagate errors up the call stack.

Since our session is limited, focus on Result and the `?` operator first - they're your main tools.

   Example 3b: Code review session (expert role + 3 turns)
   Query: Review this Rust code approach: using unwrap() everywhere for simplicity
   Response: **CRITICAL ISSUE**: Using `unwrap()` everywhere is dangerous and not simple - it's reckless.

**Problems**:
- Causes panics on any error, crashing your program
- Makes debugging harder (no context about what failed)
- Violates Rust's safety principles

**Fix immediately**:
- Replace `unwrap()` with proper error handling using `?`, `match`, or `expect("meaningful message")`
- Return `Result<T, E>` from functions that can fail
- Use `expect()` only when you're certain something won't fail

This is a time-limited session - prioritize fixing file I/O and network operations first, as they fail most often.

   Example 3c: Quick consultation (format + 1 turn)
   Query: My Rust program is running slowly with large vectors
   Response: PROBLEM: Large vector operations causing performance bottleneck in Rust program.

SOLUTION: 1) Use iterators instead of loops (.iter().map() vs for loops), 2) Preallocate capacity with Vec::with_capacity(), 3) Consider parallel processing with rayon crate for CPU-intensive operations, 4) Profile with cargo-flamegraph to identify exact bottlenecks.

REASONING: Iterators are zero-cost abstractions that often optimize better than manual loops, preallocation prevents repeated memory allocations during growth, and parallel processing utilizes multiple cores for large datasets. Profiling ensures you're optimizing the right parts.

4. Different System Prompt Scenarios
   Various real-world use cases for system prompt modification

   Example 4a: Debugging assistant mode
   Query: My Rust code won't compile
   Response: I need more specific information to help you debug effectively:

1. **What's the exact error message?** Please share the complete compiler output, including error codes (like E0308, E0277).

2. **Share the problematic code snippet** - even just the function or struct where the error occurs.

3. **What Rust version are you using?** Run `rustc --version`.

4. **What were you trying to accomplish?** Understanding your intent helps identify the root cause.

5. **When did this start happening?** Did it work before, or is this new code?

The Rust compiler gives excellent error messages with suggestions - please share them so I can walk you through the solution step by step.

   Example 4b: API documentation generator
   Query: Document a function that reads a file and returns its contents
   Response: ```rust
/// Reads the entire contents of a file into a string.
///
/// This function opens the file at the specified path and reads all of its contents
/// into memory as a UTF-8 encoded string.
///
/// # Parameters
///
/// * `path` - A path-like object that can be converted to a `Path`. This includes
///   `&str`, `String`, `&Path`, and `PathBuf`.
///
/// # Returns
///
/// * `Result<String, std::io::Error>` - On success, returns `Ok(String)` containing
///   the file contents. On failure, returns `Err(std::io::Error)` with details about
///   what went wrong.
///
/// # Errors
///
/// This function will return an error if:
/// * The file does not exist
/// * The current process lacks permissions to read the file
/// * The file contains invalid UTF-8 data
/// * Any I/O error occurs during reading
///
/// # Examples
///
/// ```rust
/// use std::fs;
///
/// // Read a configuration file
/// match read_file_contents("config.toml") {
///     Ok(contents) => println!("Config: {}", contents),
///     Err(e) => eprintln!("Failed to read config: {}", e),
/// }
///
/// // Using the ? operator in a function that returns Result
/// fn load_config() -> Result<String, std::io::Error> {
///     let contents = read_file_contents("app.config")?;
///     Ok(contents)
/// }
/// ```
///
/// # See Also
///
/// * [`std::fs::read_to_string`] - Standard library function with same functionality
/// * [`std::fs::File::open`] - For more control over file opening
/// * [`BufReader`] - For reading large files efficiently
pub fn read_file_contents<P: AsRef<Path>>(path: P) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)
}
```

   Example 4c: Performance analysis mode
   Query: Analyze the performance of nested loops processing a Vec<Vec<i32>>
   Response: **Performance Analysis: Nested Loop on Vec<Vec<i32>>**

**Time Complexity**: O(n*m) where n = outer vector length, m = average inner vector length
**Space Complexity**: O(1) additional space (not counting input)

**Performance Issues**:
1. **Cache Misses**: Inner vectors may not be contiguous in memory
2. **Double Indirection**: Two pointer dereferences per element access
3. **Bounds Checking**: Rust checks bounds on each access unless optimized away

**Rust-Specific Optimizations**:
1. **Use Iterators**: `vec.iter().flat_map(|inner| inner.iter())` - often faster than manual loops
2. **Flatten Structure**: Consider `Vec<i32>` with calculated indices instead of `Vec<Vec<i32>>`
3. **Parallel Processing**: Use `rayon::par_iter()` for CPU-intensive operations
4. **Slice Operations**: Use `&[i32]` instead of `Vec<i32>` for inner vectors if possible

**Better Data Structure**: For frequent random access, consider `ndarray` crate or flattened `Vec<i32>` with `(row * cols + col)` indexing.

This is a time-limited analysis session. Profile with `cargo bench` to measure actual impact before optimizing.

   Example 4d: Learning assistant with skill progression
   Query: What are lifetimes in Rust?
   Response: Based on your question, I'll start with the fundamentals and build up your understanding.

**Prerequisites**: You should understand ownership and borrowing basics first. Lifetimes work together with these concepts.

**What Lifetimes Are**: Lifetimes are Rust's way of ensuring that references are valid for as long as needed. They prevent "dangling pointers" - references to memory that's already been freed.

**Simple Example**:
```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

The `'a` tells Rust: "the returned reference will live as long as both input references."

**Why They Matter**: Without lifetimes, Rust couldn't guarantee memory safety. They're usually inferred automatically, but sometimes you need to specify them explicitly.

**Next Steps**: Once you're comfortable with this basic concept, we can explore lifetime elision rules (when Rust infers them automatically) and more complex scenarios.

What part would you like me to explain further? The syntax, the reasoning behind them, or a specific example you're struggling with?

5. Validation and Error Handling
   Testing system prompt and max_turns validation

   Example 5a: Valid system prompt and max_turns configuration
   ✓ Valid configuration accepted

   Example 5b: Invalid max_turns value (0)
   ✓ Invalid turns correctly detected: Max turns must be greater than 0

   Example 5c: System prompt length validation
   ✓ Long prompt correctly detected: Append system prompt exceeds maximum length of 10000 characters (got 15000)

   Example 5d: system_prompt + append_system_prompt conflict
   ✓ Conflict correctly detected: Cannot use both system_prompt and append_system_prompt simultaneously

   Example 5e: Various valid max_turns values
   ✓ max_turns(1) accepted
   ✓ max_turns(3) accepted
   ✓ max_turns(10) accepted
   ✓ max_turns(50) accepted
   ✓ max_turns(100) accepted

   Example 5f: Empty append_system_prompt
   ✓ Empty append_system_prompt accepted

System prompts example completed successfully!
*/
