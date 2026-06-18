use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Types of messages in a Claude AI conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    /// Initialization message for a new session
    Init,
    /// Message from the user
    User,
    /// Response from Claude AI assistant
    Assistant,
    /// Final result message with statistics
    Result,
    /// System message for context setting
    System,
    /// Tool invocation message
    Tool,
    /// Result from a tool invocation
    ToolResult,
}

/// Metadata associated with a message
///
/// Contains contextual information about a message including timing, cost,
/// and resource usage. Not all fields are populated for every message type.
///
/// # Examples
///
/// ```rust
/// # use claude_sdk_rs::core::message::{MessageMeta, TokenUsage};
/// # use std::time::SystemTime;
/// let meta = MessageMeta {
///     session_id: "session-123".to_string(),
///     timestamp: Some(SystemTime::now()),
///     cost_usd: Some(0.0015),
///     duration_ms: Some(1200),
///     tokens_used: Some(TokenUsage {
///         input: 50,
///         output: 100,
///         total: 150,
///     }),
/// };
///
/// // Check if this was an expensive message
/// if let Some(cost) = meta.cost_usd {
///     if cost > 0.01 {
///         println!("Warning: High cost message: ${:.4}", cost);
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMeta {
    /// Unique identifier for the session this message belongs to
    pub session_id: String,
    /// Timestamp when the message was created
    pub timestamp: Option<SystemTime>,
    /// Cost of processing this specific message in USD
    pub cost_usd: Option<f64>,
    /// Time taken to process this message in milliseconds
    pub duration_ms: Option<u64>,
    /// Token usage for this specific message
    pub tokens_used: Option<TokenUsage>,
}

/// Token usage statistics for a message
///
/// Tracks the number of tokens consumed by Claude for processing input and
/// generating output. Useful for monitoring costs and staying within limits.
///
/// # Examples
///
/// ```rust
/// # use claude_sdk_rs::core::message::TokenUsage;
/// let usage = TokenUsage {
///     input: 150,
///     output: 500,
///     total: 650,
/// };
///
/// // Calculate approximate cost (example rates)
/// let input_cost = usage.input as f64 * 0.000003;  // $3/million tokens
/// let output_cost = usage.output as f64 * 0.000015; // $15/million tokens
/// let total_cost = input_cost + output_cost;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of input tokens processed
    pub input: u64,
    /// Number of output tokens generated
    pub output: u64,
    /// Total tokens used (input + output)
    pub total: u64,
}

/// Represents a message in a Claude AI conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(missing_docs)] // meta fields are flattened and self-explanatory
pub enum Message {
    /// Initialization message when starting a session
    Init {
        #[serde(flatten)]
        meta: MessageMeta,
    },
    /// User input message
    User {
        /// The user's message content
        content: String,
        #[serde(flatten)]
        meta: MessageMeta,
    },
    /// Claude AI's response
    Assistant {
        /// The assistant's response content
        content: String,
        #[serde(flatten)]
        meta: MessageMeta,
    },
    /// Final result with conversation statistics
    Result {
        #[serde(flatten)]
        meta: MessageMeta,
        /// Statistics about the conversation
        stats: ConversationStats,
    },
    /// System message for setting context
    System {
        /// System prompt or context
        content: String,
        #[serde(flatten)]
        meta: MessageMeta,
    },
    /// Tool invocation request
    Tool {
        /// Name of the tool to invoke
        name: String,
        /// Parameters for the tool
        parameters: serde_json::Value,
        #[serde(flatten)]
        meta: MessageMeta,
    },
    /// Result from a tool execution
    ToolResult {
        /// Name of the tool that was executed
        tool_name: String,
        /// Result of the tool execution
        result: serde_json::Value,
        #[serde(flatten)]
        meta: MessageMeta,
    },
}

/// Statistics for an entire conversation
///
/// Provides aggregate metrics for a complete conversation session, including
/// message counts, costs, duration, and token usage. This is typically included
/// in the final `Result` message of a stream.
///
/// # Examples
///
/// ```rust
/// # use claude_sdk_rs::core::message::{ConversationStats, TokenUsage};
/// let stats = ConversationStats {
///     total_messages: 10,
///     total_cost_usd: 0.045,
///     total_duration_ms: 3500,
///     total_tokens: TokenUsage {
///         input: 1500,
///         output: 2500,
///         total: 4000,
///     },
/// };
///
/// println!("Conversation summary:");
/// println!("  Messages: {}", stats.total_messages);
/// println!("  Cost: ${:.4}", stats.total_cost_usd);
/// println!("  Duration: {:.1}s", stats.total_duration_ms as f64 / 1000.0);
/// println!("  Tokens: {} ({}in/{}out)",
///     stats.total_tokens.total,
///     stats.total_tokens.input,
///     stats.total_tokens.output
/// );
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationStats {
    /// Total number of messages exchanged in the conversation
    pub total_messages: u64,
    /// Total cost of the conversation in USD
    pub total_cost_usd: f64,
    /// Total processing time in milliseconds
    pub total_duration_ms: u64,
    /// Aggregate token usage for the entire conversation
    pub total_tokens: TokenUsage,
}

impl Message {
    /// Get the type of this message
    pub fn message_type(&self) -> MessageType {
        match self {
            Message::Init { .. } => MessageType::Init,
            Message::User { .. } => MessageType::User,
            Message::Assistant { .. } => MessageType::Assistant,
            Message::Result { .. } => MessageType::Result,
            Message::System { .. } => MessageType::System,
            Message::Tool { .. } => MessageType::Tool,
            Message::ToolResult { .. } => MessageType::ToolResult,
        }
    }

    /// Get the metadata associated with this message
    pub fn meta(&self) -> &MessageMeta {
        match self {
            Message::Init { meta, .. }
            | Message::User { meta, .. }
            | Message::Assistant { meta, .. }
            | Message::Result { meta, .. }
            | Message::System { meta, .. }
            | Message::Tool { meta, .. }
            | Message::ToolResult { meta, .. } => meta,
        }
    }

    /// Get a string representation of the message content
    pub fn content(&self) -> String {
        match self {
            Message::User { content, .. }
            | Message::Assistant { content, .. }
            | Message::System { content, .. } => content.clone(),
            Message::Tool {
                name, parameters, ..
            } => {
                format!("Tool: {name} with parameters: {parameters}")
            }
            Message::ToolResult {
                tool_name, result, ..
            } => {
                format!("Tool result from {tool_name}: {result}")
            }
            Message::Init { .. } => "Session initialized".to_string(),
            Message::Result { stats, .. } => {
                format!(
                    "Conversation stats: {} messages, ${:.6} total cost",
                    stats.total_messages, stats.total_cost_usd
                )
            }
        }
    }
}
