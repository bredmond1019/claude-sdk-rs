use serde::{Deserialize, Serialize};

/// Raw response from Claude CLI in JSON format
///
/// This represents the direct JSON response from the Claude CLI tool.
/// Most users should use [`ClaudeResponse`] instead, which provides
/// a more convenient interface.
///
/// # Examples
///
/// ```rust
/// use claude_sdk_rs_core::ClaudeCliResponse;
/// use serde_json;
///
/// // This would typically come from parsing Claude CLI output
/// let json = r#"{
///     "type": "assistant_response",
///     "subtype": "completion",
///     "cost_usd": 0.001234,
///     "is_error": false,
///     "duration_ms": 1500,
///     "duration_api_ms": 1200,
///     "num_turns": 1,
///     "result": "Hello, world!",
///     "total_cost": 0.001234,
///     "session_id": "session_123"
/// }"#;
///
/// let response: ClaudeCliResponse = serde_json::from_str(json)?;
/// assert_eq!(response.result, "Hello, world!");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeCliResponse {
    /// Type of response (e.g., `"assistant_response"`)
    #[serde(rename = "type")]
    pub response_type: String,

    /// Subtype providing more specific classification
    pub subtype: String,

    /// Cost of this specific request in USD
    pub cost_usd: Option<f64>,

    /// Whether this response represents an error
    pub is_error: bool,

    /// Total duration including processing time
    pub duration_ms: u64,

    /// API-specific duration (excluding local processing)
    pub duration_api_ms: Option<u64>,

    /// Number of turns in the conversation
    pub num_turns: u32,

    /// The actual text result from Claude
    pub result: String,

    /// Total accumulated cost for the session
    pub total_cost: Option<f64>,

    /// Unique identifier for this session
    pub session_id: String,
}

/// High-level response from Claude with convenient access to content and metadata
///
/// This is the primary response type returned by the Claude AI SDK. It provides
/// both the text content and optional metadata like costs, session information,
/// and raw JSON for advanced use cases.
///
/// # Examples
///
/// ```rust
/// use claude_sdk_rs_core::ClaudeResponse;
///
/// // Simple text response
/// let response = ClaudeResponse::text("Hello, world!".to_string());
/// assert_eq!(response.content, "Hello, world!");
/// assert!(response.metadata.is_none());
///
/// // Response with metadata (typically created by the SDK)
/// let json = serde_json::json!({
///     "session_id": "session_123",
///     "cost_usd": 0.001,
/// });
/// let response = ClaudeResponse::with_json("Response text".to_string(), json);
/// assert!(response.metadata.is_some());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeResponse {
    /// The main text content from Claude's response
    ///
    /// This is the primary result that most applications will use.
    pub content: String,

    /// Raw JSON response from Claude CLI for advanced parsing
    ///
    /// Contains the complete, unprocessed JSON from Claude CLI.
    /// Useful for accessing fields not covered by the structured metadata
    /// or for implementing custom parsing logic.
    pub raw_json: Option<serde_json::Value>,

    /// Structured metadata when available
    ///
    /// Provides convenient access to common metadata fields like costs,
    /// session IDs, and token usage. Only present when using JSON output formats.
    pub metadata: Option<ResponseMetadata>,
}

/// Metadata extracted from Claude responses
///
/// Contains structured information about the response such as costs,
/// timing, token usage, and session details.
///
/// # Examples
///
/// ```rust
/// use claude_sdk_rs_core::ResponseMetadata;
///
/// // Accessing metadata from a response
/// # let response = claude_sdk_rs_core::ClaudeResponse::text("test".to_string());
/// if let Some(metadata) = &response.metadata {
///     if let Some(cost) = metadata.cost_usd {
///         println!("Request cost: ${:.6}", cost);
///     }
///     println!("Session: {}", metadata.session_id);
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Unique identifier for the session this response belongs to
    pub session_id: String,

    /// Cost of this specific request in USD, if available
    pub cost_usd: Option<f64>,

    /// Total duration of the request in milliseconds, if available
    pub duration_ms: Option<u64>,

    /// Detailed token usage information, if available
    pub tokens_used: Option<TokenUsage>,

    /// The model that generated this response, if available
    pub model: Option<String>,
}

/// Token usage statistics for a Claude request
///
/// Provides detailed information about token consumption, which is useful
/// for understanding costs and optimizing requests.
///
/// # Examples
///
/// ```rust
/// use claude_sdk_rs_core::TokenUsage;
///
/// # let response = claude_sdk_rs_core::ClaudeResponse::text("test".to_string());
/// if let Some(metadata) = &response.metadata {
///     if let Some(tokens) = &metadata.tokens_used {
///         if let (Some(input), Some(output)) = (tokens.input_tokens, tokens.output_tokens) {
///             println!("Used {} input tokens and {} output tokens", input, output);
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of input tokens processed
    pub input_tokens: Option<u64>,

    /// Number of output tokens generated
    pub output_tokens: Option<u64>,

    /// Tokens used for cache creation (if applicable)
    pub cache_creation_input_tokens: Option<u64>,

    /// Tokens read from cache (if applicable)
    pub cache_read_input_tokens: Option<u64>,
}

impl ClaudeResponse {
    /// Create a simple text response without metadata
    ///
    /// Use this for creating responses when you only have the text content
    /// and don't need to include metadata or raw JSON data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs_core::ClaudeResponse;
    ///
    /// let response = ClaudeResponse::text("Hello, world!".to_string());
    /// assert_eq!(response.content, "Hello, world!");
    /// assert!(response.raw_json.is_none());
    /// assert!(response.metadata.is_none());
    /// ```
    pub fn text(content: String) -> Self {
        Self {
            content,
            raw_json: None,
            metadata: None,
        }
    }

    /// Create a response with full JSON data and extracted metadata
    ///
    /// This constructor is typically used internally by the SDK when parsing
    /// JSON responses from Claude CLI. It automatically extracts metadata
    /// from the raw JSON for convenient access.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs_core::ClaudeResponse;
    /// use serde_json::json;
    ///
    /// let raw_json = json!({
    ///     "session_id": "test_session",
    ///     "cost_usd": 0.001,
    ///     "duration_ms": 1500
    /// });
    ///
    /// let response = ClaudeResponse::with_json(
    ///     "Hello, world!".to_string(),
    ///     raw_json
    /// );
    ///
    /// assert_eq!(response.content, "Hello, world!");
    /// assert!(response.raw_json.is_some());
    /// assert!(response.metadata.is_some());
    /// ```
    pub fn with_json(content: String, raw_json: serde_json::Value) -> Self {
        let metadata = Self::extract_metadata(&raw_json);
        Self {
            content,
            raw_json: Some(raw_json),
            metadata,
        }
    }

    /// Extract structured metadata from raw JSON response
    ///
    /// This method parses the raw JSON to extract commonly used metadata
    /// fields like session ID, cost, duration, and token usage.
    ///
    /// Returns `None` if the JSON doesn't contain the required `session_id` field.
    fn extract_metadata(json: &serde_json::Value) -> Option<ResponseMetadata> {
        let session_id = json.get("session_id")?.as_str()?.to_string();

        Some(ResponseMetadata {
            session_id,
            cost_usd: json.get("cost_usd").and_then(serde_json::Value::as_f64),
            duration_ms: json.get("duration_ms").and_then(serde_json::Value::as_u64),
            tokens_used: json
                .get("message")
                .and_then(|m| m.get("usage"))
                .map(|usage| TokenUsage {
                    input_tokens: usage
                        .get("input_tokens")
                        .and_then(serde_json::Value::as_u64),
                    output_tokens: usage
                        .get("output_tokens")
                        .and_then(serde_json::Value::as_u64),
                    cache_creation_input_tokens: usage
                        .get("cache_creation_input_tokens")
                        .and_then(serde_json::Value::as_u64),
                    cache_read_input_tokens: usage
                        .get("cache_read_input_tokens")
                        .and_then(serde_json::Value::as_u64),
                }),
            model: json
                .get("message")
                .and_then(|m| m.get("model"))
                .and_then(|v| v.as_str())
                .map(String::from),
        })
    }
}

/// Tool permission specification for controlling what tools Claude can access
///
/// This enum defines the different types of tools that Claude can be granted
/// permission to use, providing fine-grained control over capabilities.
///
/// # Examples
///
/// ```rust
/// use claude_sdk_rs_core::ToolPermission;
///
/// // Allow specific MCP server tool
/// let mcp_tool = ToolPermission::mcp("database", "query");
/// assert_eq!(mcp_tool.to_cli_format(), "mcp__database__query");
///
/// // Allow specific bash command
/// let bash_tool = ToolPermission::bash("ls");
/// assert_eq!(bash_tool.to_cli_format(), "bash:ls");
///
/// // Allow all tools
/// let all_tools = ToolPermission::All;
/// assert_eq!(all_tools.to_cli_format(), "*");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolPermission {
    /// Permission for a specific MCP (Model Context Protocol) tool
    ///
    /// Grants access to a specific tool on a specific MCP server.
    /// Use "*" as the tool name to allow all tools on the server.
    Mcp {
        /// Name of the MCP server
        server: String,
        /// Name of the specific tool, or "*" for all tools
        tool: String,
    },

    /// Permission for a specific bash command
    ///
    /// Grants access to execute a specific bash command.
    /// This provides fine-grained control over shell access.
    Bash {
        /// The specific bash command to allow
        command: String,
    },

    /// Permission for all available tools
    ///
    /// Grants unrestricted access to all tools. Use with caution
    /// in production environments.
    All,
}

impl ToolPermission {
    /// Create a new MCP tool permission
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs_core::ToolPermission;
    ///
    /// // Allow specific tool
    /// let tool = ToolPermission::mcp("database", "query");
    ///
    /// // Allow all tools on a server
    /// let all_tools = ToolPermission::mcp("filesystem", "*");
    /// ```
    pub fn mcp(server: impl Into<String>, tool: impl Into<String>) -> Self {
        Self::Mcp {
            server: server.into(),
            tool: tool.into(),
        }
    }

    /// Create a new bash command permission
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs_core::ToolPermission;
    ///
    /// let permission = ToolPermission::bash("ls");
    /// assert_eq!(permission.to_cli_format(), "bash:ls");
    /// ```
    pub fn bash(command: impl Into<String>) -> Self {
        Self::Bash {
            command: command.into(),
        }
    }

    /// Convert to the CLI format string expected by Claude Code
    ///
    /// This method formats the permission for use with the Claude CLI's
    /// `--allowed-tools` parameter.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs_core::ToolPermission;
    ///
    /// assert_eq!(
    ///     ToolPermission::mcp("server", "tool").to_cli_format(),
    ///     "mcp__server__tool"
    /// );
    /// assert_eq!(
    ///     ToolPermission::bash("ls").to_cli_format(),
    ///     "bash:ls"
    /// );
    /// assert_eq!(
    ///     ToolPermission::All.to_cli_format(),
    ///     "*"
    /// );
    /// ```
    pub fn to_cli_format(&self) -> String {
        match self {
            Self::Mcp { server, tool } => {
                if tool == "*" {
                    format!("mcp__{server}__*")
                } else {
                    format!("mcp__{server}__{tool}")
                }
            }
            Self::Bash { command } => format!("bash:{command}"),
            Self::All => "*".to_string(),
        }
    }
}

/// Represents a cost in USD
///
/// This is a simple wrapper around a floating-point cost value that provides
/// convenient methods for cost calculations and aggregation.
///
/// # Examples
///
/// ```rust
/// use claude_sdk_rs_core::Cost;
///
/// let cost1 = Cost::new(0.001);
/// let cost2 = Cost::new(0.002);
/// let total = cost1.add(&cost2);
///
/// assert_eq!(total.usd, 0.003);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Cost {
    /// Cost amount in USD
    pub usd: f64,
}

impl Cost {
    /// Create a new cost with the specified USD amount
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs_core::Cost;
    ///
    /// let cost = Cost::new(0.001234);
    /// assert_eq!(cost.usd, 0.001234);
    /// ```
    pub fn new(usd: f64) -> Self {
        Self { usd }
    }

    /// Create a zero-cost instance
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs_core::Cost;
    ///
    /// let cost = Cost::zero();
    /// assert_eq!(cost.usd, 0.0);
    /// ```
    pub fn zero() -> Self {
        Self { usd: 0.0 }
    }

    /// Add this cost to another cost and return the sum
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs_core::Cost;
    ///
    /// let cost1 = Cost::new(0.001);
    /// let cost2 = Cost::new(0.002);
    /// let total = cost1.add(&cost2);
    ///
    /// assert_eq!(total.usd, 0.003);
    /// ```
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            usd: self.usd + other.usd,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_claude_cli_response_with_optional_costs() {
        // Test 1: JSON with cost fields present
        let json_with_cost = r#"{
            "type": "assistant_response",
            "subtype": "completion",
            "cost_usd": 0.001234,
            "is_error": false,
            "duration_ms": 1500,
            "duration_api_ms": 1200,
            "num_turns": 1,
            "result": "Hello, world!",
            "total_cost": 0.001234,
            "session_id": "session_123"
        }"#;

        let response: ClaudeCliResponse = serde_json::from_str(json_with_cost).unwrap();
        assert_eq!(response.cost_usd, Some(0.001_234));
        assert_eq!(response.total_cost, Some(0.001_234));

        // Test 2: JSON without cost fields
        let json_without_cost = r#"{
            "type": "assistant_response",
            "subtype": "completion",
            "is_error": false,
            "duration_ms": 1500,
            "duration_api_ms": 1200,
            "num_turns": 1,
            "result": "Hello, world!",
            "session_id": "session_123"
        }"#;

        let response: ClaudeCliResponse = serde_json::from_str(json_without_cost).unwrap();
        assert_eq!(response.cost_usd, None);
        assert_eq!(response.total_cost, None);

        // Test 3: JSON with null cost values
        let json_null_cost = r#"{
            "type": "assistant_response",
            "subtype": "completion",
            "cost_usd": null,
            "is_error": false,
            "duration_ms": 1500,
            "num_turns": 1,
            "result": "Hello, world!",
            "total_cost": null,
            "session_id": "session_123"
        }"#;

        let response: ClaudeCliResponse = serde_json::from_str(json_null_cost).unwrap();
        assert_eq!(response.cost_usd, None);
        assert_eq!(response.total_cost, None);
    }
}
