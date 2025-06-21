use crate::core::Error;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

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
    /// `--allowed-tools` parameter. The format follows Claude CLI conventions:
    /// - Bash commands: `Bash(command)` e.g., `Bash(ls)`, `Bash(git status)`
    /// - MCP tools: `mcp__server__tool` e.g., `mcp__database__query`
    /// - All tools: `*`
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
    ///     "Bash(ls)"
    /// );
    /// assert_eq!(
    ///     ToolPermission::bash("git status").to_cli_format(),
    ///     "Bash(git status)"
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
            Self::Bash { command } => format!("Bash({command})"),
            Self::All => "*".to_string(),
        }
    }

    /// Parse a granular permission string into a ToolPermission
    ///
    /// Supports multiple formats for tool permissions:
    /// - `Bash(command)` - Bash command permission (e.g., `Bash(ls)`, `Bash(git status)`)
    /// - `bash:command` - Alternative bash format (converted to `Bash(command)`)
    /// - `mcp__server__tool` - MCP tool permission
    /// - `*` - All tools permission
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs_core::ToolPermission;
    /// use std::str::FromStr;
    ///
    /// // Bash command formats
    /// let bash1 = ToolPermission::from_str("Bash(ls)").unwrap();
    /// let bash2 = ToolPermission::from_str("bash:ls").unwrap();
    /// assert_eq!(bash1, bash2);
    ///
    /// // MCP tool format
    /// let mcp = ToolPermission::from_str("mcp__database__query").unwrap();
    /// assert_eq!(mcp, ToolPermission::mcp("database", "query"));
    ///
    /// // All tools
    /// let all = ToolPermission::from_str("*").unwrap();
    /// assert_eq!(all, ToolPermission::All);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the permission string doesn't match any known format.
    pub fn parse_granular(permission_str: &str) -> Result<Self, Error> {
        let trimmed = permission_str.trim();

        // Handle all tools
        if trimmed == "*" {
            return Ok(Self::All);
        }

        // Handle Bash(command) format
        if trimmed.starts_with("Bash(") && trimmed.ends_with(')') {
            let command = trimmed[5..trimmed.len() - 1].trim();
            if command.is_empty() {
                return Err(Error::InvalidInput(
                    "Bash command cannot be empty in permission string".to_string(),
                ));
            }
            return Ok(Self::Bash {
                command: command.to_string(),
            });
        }

        // Handle bash:command format (legacy)
        if trimmed.starts_with("bash:") {
            let command = trimmed[5..].trim();
            if command.is_empty() {
                return Err(Error::InvalidInput(
                    "Bash command cannot be empty in permission string".to_string(),
                ));
            }
            return Ok(Self::Bash {
                command: command.to_string(),
            });
        }

        // Handle mcp__server__tool format
        if trimmed.starts_with("mcp__") {
            // Remove the "mcp__" prefix and split the remaining part
            let remaining = &trimmed[5..];

            // Find the first "__" to separate server from tool
            if let Some(separator_pos) = remaining.find("__") {
                let server = remaining[..separator_pos].trim();
                let tool = remaining[separator_pos + 2..].trim();

                if server.is_empty() || tool.is_empty() {
                    return Err(Error::InvalidInput(
                        "MCP server and tool names cannot be empty".to_string(),
                    ));
                }

                return Ok(Self::Mcp {
                    server: server.to_string(),
                    tool: tool.to_string(),
                });
            }
            return Err(Error::InvalidInput(format!(
                "Invalid MCP permission format '{}'. Expected 'mcp__server__tool'",
                trimmed
            )));
        }

        // Unknown format
        Err(Error::InvalidInput(format!(
            "Unknown tool permission format '{}'. Supported formats: 'Bash(command)', 'bash:command', 'mcp__server__tool', '*'",
            trimmed
        )))
    }

    /// Validate that this permission is properly formatted
    ///
    /// Checks that:
    /// - Bash commands are not empty
    /// - MCP server and tool names are not empty
    /// - No invalid characters are present
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs_core::ToolPermission;
    ///
    /// // Valid permissions
    /// assert!(ToolPermission::bash("ls").validate().is_ok());
    /// assert!(ToolPermission::mcp("server", "tool").validate().is_ok());
    /// assert!(ToolPermission::All.validate().is_ok());
    ///
    /// // Invalid permissions would be caught during construction
    /// // but this method can validate existing instances
    /// ```
    pub fn validate(&self) -> Result<(), Error> {
        match self {
            Self::Bash { command } => {
                if command.trim().is_empty() {
                    return Err(Error::InvalidInput(
                        "Bash command cannot be empty".to_string(),
                    ));
                }
                // Check for potentially dangerous characters that might break CLI parsing
                if command.contains('\n') || command.contains('\r') {
                    return Err(Error::InvalidInput(
                        "Bash commands cannot contain newline characters".to_string(),
                    ));
                }
                Ok(())
            }
            Self::Mcp { server, tool } => {
                if server.trim().is_empty() {
                    return Err(Error::InvalidInput(
                        "MCP server name cannot be empty".to_string(),
                    ));
                }
                if tool.trim().is_empty() {
                    return Err(Error::InvalidInput(
                        "MCP tool name cannot be empty".to_string(),
                    ));
                }
                // Check for invalid characters in MCP names
                if server.contains("__") || tool.contains("__") {
                    return Err(Error::InvalidInput(
                        "MCP server and tool names cannot contain '__'".to_string(),
                    ));
                }
                Ok(())
            }
            Self::All => Ok(()),
        }
    }
}

impl FromStr for ToolPermission {
    type Err = Error;

    /// Parse a string into a ToolPermission
    ///
    /// This implementation uses the `parse_granular` method to support
    /// multiple permission string formats.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs_core::ToolPermission;
    /// use std::str::FromStr;
    ///
    /// let permission = ToolPermission::from_str("Bash(ls)")?;
    /// assert_eq!(permission, ToolPermission::bash("ls"));
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_granular(s)
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

    #[test]
    fn test_tool_permission_cli_format() {
        // Test bash command formatting
        assert_eq!(ToolPermission::bash("ls").to_cli_format(), "Bash(ls)");
        assert_eq!(
            ToolPermission::bash("git status").to_cli_format(),
            "Bash(git status)"
        );
        assert_eq!(
            ToolPermission::bash("npm install").to_cli_format(),
            "Bash(npm install)"
        );

        // Test MCP tool formatting
        assert_eq!(
            ToolPermission::mcp("database", "query").to_cli_format(),
            "mcp__database__query"
        );
        assert_eq!(
            ToolPermission::mcp("filesystem", "*").to_cli_format(),
            "mcp__filesystem__*"
        );

        // Test all tools
        assert_eq!(ToolPermission::All.to_cli_format(), "*");
    }

    #[test]
    fn test_tool_permission_granular_parsing() {
        use std::str::FromStr;

        // Test Bash(command) format parsing
        let bash_ls = ToolPermission::from_str("Bash(ls)").unwrap();
        assert_eq!(bash_ls, ToolPermission::bash("ls"));

        let bash_git = ToolPermission::from_str("Bash(git status)").unwrap();
        assert_eq!(bash_git, ToolPermission::bash("git status"));

        let bash_npm = ToolPermission::from_str("Bash(npm install)").unwrap();
        assert_eq!(bash_npm, ToolPermission::bash("npm install"));

        // Test bash:command format (legacy) parsing
        let legacy_bash = ToolPermission::from_str("bash:ls").unwrap();
        assert_eq!(legacy_bash, ToolPermission::bash("ls"));

        let legacy_complex = ToolPermission::from_str("bash:git status").unwrap();
        assert_eq!(legacy_complex, ToolPermission::bash("git status"));

        // Test MCP format parsing
        let mcp_tool = ToolPermission::from_str("mcp__database__query").unwrap();
        assert_eq!(mcp_tool, ToolPermission::mcp("database", "query"));

        let mcp_wildcard = ToolPermission::from_str("mcp__filesystem__*").unwrap();
        assert_eq!(mcp_wildcard, ToolPermission::mcp("filesystem", "*"));

        // Test all tools
        let all_tools = ToolPermission::from_str("*").unwrap();
        assert_eq!(all_tools, ToolPermission::All);

        // Test whitespace handling
        let with_spaces = ToolPermission::from_str("  Bash(ls)  ").unwrap();
        assert_eq!(with_spaces, ToolPermission::bash("ls"));
    }

    #[test]
    fn test_tool_permission_parsing_errors() {
        use std::str::FromStr;

        // Test invalid Bash format
        assert!(ToolPermission::from_str("Bash()").is_err()); // Empty command
        assert!(ToolPermission::from_str("Bash(").is_err()); // Unclosed parenthesis
        assert!(ToolPermission::from_str("bash:").is_err()); // Empty legacy command

        // Test invalid MCP format
        assert!(ToolPermission::from_str("mcp__").is_err()); // Incomplete
        assert!(ToolPermission::from_str("mcp__server").is_err()); // Missing tool
        assert!(ToolPermission::from_str("mcp__server__").is_err()); // Empty tool
        assert!(ToolPermission::from_str("mcp____tool").is_err()); // Empty server

        // Test unknown formats
        assert!(ToolPermission::from_str("unknown_format").is_err());
        assert!(ToolPermission::from_str("").is_err());
        assert!(ToolPermission::from_str("Shell(ls)").is_err()); // Wrong tool name
    }

    #[test]
    fn test_tool_permission_validation() {
        // Test valid permissions
        assert!(ToolPermission::bash("ls").validate().is_ok());
        assert!(ToolPermission::bash("git status").validate().is_ok());
        assert!(ToolPermission::mcp("server", "tool").validate().is_ok());
        assert!(ToolPermission::mcp("server", "*").validate().is_ok());
        assert!(ToolPermission::All.validate().is_ok());

        // Test that we can create invalid permissions and catch them in validation
        // (This tests the validation logic even though construction should prevent these)
        let invalid_bash = ToolPermission::Bash {
            command: "".to_string(), // Empty command
        };
        assert!(invalid_bash.validate().is_err());

        let newline_bash = ToolPermission::Bash {
            command: "ls\nrm -rf /".to_string(), // Newline in command
        };
        assert!(newline_bash.validate().is_err());

        let invalid_mcp_server = ToolPermission::Mcp {
            server: "".to_string(), // Empty server
            tool: "tool".to_string(),
        };
        assert!(invalid_mcp_server.validate().is_err());

        let invalid_mcp_tool = ToolPermission::Mcp {
            server: "server".to_string(),
            tool: "".to_string(), // Empty tool
        };
        assert!(invalid_mcp_tool.validate().is_err());

        let invalid_mcp_separator = ToolPermission::Mcp {
            server: "ser__ver".to_string(), // Contains separator
            tool: "tool".to_string(),
        };
        assert!(invalid_mcp_separator.validate().is_err());
    }

    #[test]
    fn test_tool_permission_roundtrip() {
        use std::str::FromStr;

        // Test that parsing and formatting are consistent
        let permissions = vec![
            "Bash(ls)",
            "Bash(git status)",
            "Bash(npm install --save)",
            "mcp__database__query",
            "mcp__filesystem__read",
            "mcp__server__*",
            "*",
        ];

        for permission_str in permissions {
            let parsed = ToolPermission::from_str(permission_str).unwrap();
            let formatted = parsed.to_cli_format();

            // Parse the formatted version and ensure it's the same
            let reparsed = ToolPermission::from_str(&formatted).unwrap();
            assert_eq!(
                parsed, reparsed,
                "Roundtrip failed for '{}': parsed={:?}, formatted='{}', reparsed={:?}",
                permission_str, parsed, formatted, reparsed
            );
        }
    }

    #[test]
    fn test_tool_permission_legacy_format_conversion() {
        use std::str::FromStr;

        // Test that legacy bash:command format gets converted properly
        let legacy_formats = vec![
            ("bash:ls", "Bash(ls)"),
            ("bash:git status", "Bash(git status)"),
            ("bash:npm install", "Bash(npm install)"),
        ];

        for (legacy, expected_cli) in legacy_formats {
            let parsed = ToolPermission::from_str(legacy).unwrap();
            let cli_format = parsed.to_cli_format();
            assert_eq!(
                cli_format, expected_cli,
                "Legacy format '{}' should convert to '{}'",
                legacy, expected_cli
            );
        }
    }
}
