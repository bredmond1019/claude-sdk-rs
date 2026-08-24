//! Core types, configuration, and error handling for the Claude AI SDK.

/// Configuration types and builders for Claude AI client
pub mod config;
/// Error types and result helpers for the Claude AI SDK
pub mod error;
/// Message types and structures for Claude AI conversations
pub mod message;
/// Session management for persistent conversations
pub mod session;
/// Core types and response structures for the Claude AI SDK
pub mod types;

pub use config::{validate_query, Config, StreamFormat};
pub use error::{Error, ErrorCode, Result};
pub use message::{ConversationStats, Message, MessageMeta, MessageType, TokenUsage};
pub use session::{
    Session, SessionBuilder, SessionId, SessionManager, SessionStorage, StorageBackend,
};
pub use types::{ClaudeCliResponse, ClaudeResponse, Cost, ResponseMetadata, ToolPermission};

#[cfg(test)]
mod config_test;
