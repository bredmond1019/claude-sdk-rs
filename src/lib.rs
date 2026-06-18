//! # claude-sdk-rs - Rust SDK for Claude Code
//!
//! A type-safe, async-first Rust SDK that wraps the [Claude Code](https://docs.claude.com/claude-code)
//! CLI binary in a fluent, strongly-typed API. It gives you structured responses, streaming,
//! session continuity, and tool control from Rust — all async on tokio.
//!
//! This is a pure library crate. It shells out to the `claude` binary (`claude -p`) and maps a
//! typed [`Config`] onto CLI flags (`--output-format`, `--model`, `--system-prompt`, `--resume`,
//! allowed/disallowed tool lists, …), then parses the structured output back into Rust types.
//! The core mechanism lives in `src/runtime/process.rs`.
//!
//! ## Features
//!
//! ### Always available
//! - **Async-first API**: Built on tokio for high-performance async operations
//! - **Type-safe responses**: Structured data types for all responses
//! - **Multiple response formats**: Text, JSON, and streaming JSON support
//! - **Session management**: Persistent conversation contexts
//! - **Error handling**: Comprehensive error types with rich context and telemetry
//! - **Configuration**: Flexible configuration with a builder pattern
//!
//! ### Optional: `sqlite` - SQLite Storage Backend
//! Adds SQLite-backed persistent session storage:
//! ```toml
//! [dependencies]
//! claude-sdk-rs = { version = "2", features = ["sqlite"] }
//! ```
//!
//! ## Architecture Overview
//!
//! The SDK is organized into two modules:
//!
//! ### Core Module (`core`)
//! Contains fundamental types and configuration:
//! - [`Config`] - SDK configuration with builder pattern
//! - [`Error`] - Comprehensive error handling
//! - [`Message`] - Request/response message types
//! - [`Session`] - Conversation session management
//! - [`StreamFormat`] - Response format configuration
//!
//! ### Runtime Module (`runtime`)
//! Handles Claude Code CLI interaction:
//! - [`Client`] - Main API client
//! - [`QueryBuilder`] - Fluent query construction
//! - [`MessageStream`] - Streaming response handling
//!
//! ## Quick Start Examples
//!
//! ### Basic SDK Usage
//! ```rust,no_run
//! use claude_sdk_rs::{Client, Config};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), claude_sdk_rs::Error> {
//!     // Create client with default configuration
//!     let client = Client::new(Config::default());
//!
//!     // Send a simple query
//!     let response = client
//!         .query("Write a hello world program in Rust")
//!         .send()
//!         .await?;
//!
//!     println!("Claude's response: {}", response);
//!     Ok(())
//! }
//! ```
//!
//! ### Advanced Configuration
//! ```rust,no_run
//! use claude_sdk_rs::{Client, Config, StreamFormat};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), claude_sdk_rs::Error> {
//!     // Build custom configuration
//!     let config = Config::builder()
//!         .model("claude-3-sonnet-20240229")
//!         .system_prompt("You are a helpful coding assistant.")
//!         .timeout_secs(60)
//!         .stream_format(StreamFormat::Json)
//!         .build();
//!
//!     let client = Client::new(config);
//!
//!     // Get full response with metadata
//!     let response = client
//!         .query("Explain Rust ownership")
//!         .send_full()
//!         .await?;
//!
//!     println!("Response: {}", response.content);
//!     if let Some(metadata) = response.metadata {
//!         println!("Cost: ${:.6}", metadata.cost_usd.unwrap_or(0.0));
//!         println!("Session: {}", metadata.session_id);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ### Streaming Responses
//! ```rust,no_run
//! use claude_sdk_rs::{Client, Config, StreamFormat};
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), claude_sdk_rs::Error> {
//!     let client = Client::builder()
//!         .stream_format(StreamFormat::StreamJson)
//!         .build();
//!
//!     let mut stream = client
//!         .query("Write a short story about a robot")
//!         .stream()
//!         .await?;
//!
//!     // Process streaming response
//!     while let Some(message) = stream.next().await {
//!         match message {
//!             Ok(msg) => {
//!                 if let Some(content) = msg.content {
//!                     print!("{}", content);
//!                 }
//!                 if msg.stop_reason.is_some() {
//!                     break;
//!                 }
//!             }
//!             Err(e) => eprintln!("Stream error: {}", e),
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ### Session Management
//! ```rust,no_run
//! use claude_sdk_rs::{Client, Config, StreamFormat};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), claude_sdk_rs::Error> {
//!     let client = Client::builder()
//!         .stream_format(StreamFormat::Json) // Needed for session metadata
//!         .build();
//!
//!     // Start a conversation
//!     let response1 = client
//!         .query("Hello! My name is Alice and I'm learning Rust.")
//!         .send_full()
//!         .await?;
//!
//!     println!("Response 1: {}", response1.content);
//!
//!     // Continue in the same session - Claude remembers context
//!     let response2 = client
//!         .query("What's my name?")
//!         .send_full()
//!         .await?;
//!
//!     println!("Response 2: {}", response2.content);
//!     // Should respond with "Alice"
//!     Ok(())
//! }
//! ```
//!
//! ## Error Handling
//!
//! The SDK provides comprehensive error handling with the [`Error`] enum:
//!
//! ```rust,no_run
//! use claude_sdk_rs::{Client, Config, Error};
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = Client::new(Config::default());
//!
//!     match client.query("Hello").send().await {
//!         Ok(response) => println!("Success: {}", response),
//!         Err(Error::ProcessError(e)) => {
//!             eprintln!("Claude CLI process error: {}", e);
//!         }
//!         Err(Error::SerializationError(e)) => {
//!             eprintln!("JSON parsing error: {}", e);
//!         }
//!         Err(Error::BinaryNotFound) => {
//!             eprintln!("Claude CLI not found. Please install from https://claude.ai/code");
//!         }
//!         Err(Error::Timeout) => {
//!             eprintln!("Request timed out. Consider increasing timeout_secs in Config");
//!         }
//!         Err(e) => eprintln!("Other error: {}", e),
//!     }
//! }
//! ```
//!
//! ## Performance and Best Practices
//!
//! ### Configuration Optimization
//! - Use appropriate timeout values for your use case
//! - Choose the right [`StreamFormat`] for your needs:
//!   - `Text`: Fastest, raw output
//!   - `Json`: Structured data with metadata
//!   - `StreamJson`: Real-time streaming with metadata
//!
//! ### Error Handling
//! - Always handle [`Error::BinaryNotFound`] to guide users to install Claude CLI
//! - Implement retry logic for transient failures
//! - Use appropriate timeout values
//!
//! ### Resource Management
//! - Reuse [`Client`] instances when possible
//! - Process streaming responses incrementally for large outputs
//! - Use [`Config::builder()`] to avoid unnecessary allocations
//!
//! ### Security Considerations
//! - Never log or store API responses containing sensitive data
//! - Use environment variables for configuration in production
//! - Validate user input before sending to Claude
//!
//! ## Compatibility and Requirements
//!
//! - **Rust Version**: 1.70 or later (MSRV)
//! - **Claude Code CLI**: Must be installed and authenticated
//! - **Runtime**: Requires tokio async runtime
//! - **Platforms**: Linux, macOS, Windows
//!
//! ## Examples
//!
//! For more comprehensive examples, see the `examples/` directory:
//! - `01_basic_sdk.rs` - Simple SDK usage patterns
//! - `02_sdk_sessions.rs` - Multi-turn conversations
//! - `03_streaming.rs` - Real-time streaming responses
//! - `04_tools.rs` - Tool permissions and control
//! - `configuration.rs` - Advanced configuration options
//! - `error_handling.rs` - Comprehensive error handling
//! - `session_persistence.rs` - SQLite-based session storage (requires `sqlite` feature)

#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

// Core modules
pub mod core;
pub mod runtime;

// Re-export core types for convenience
pub use crate::core::{
    ClaudeResponse, Config, Cost, Error, Message, MessageMeta, MessageType, ResponseMetadata,
    Result, Session, SessionId, SessionManager, StreamFormat, TokenUsage, ToolPermission,
};

// Re-export runtime types
pub use crate::runtime::{Client, MessageStream, QueryBuilder};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{Client, Config, Error, Message, MessageType, Result, StreamFormat};
    pub use futures::StreamExt;
}
