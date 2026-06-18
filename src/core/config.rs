use crate::core::error::Error;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// Validation constants
const MAX_QUERY_LENGTH: usize = 100_000;
const MAX_SYSTEM_PROMPT_LENGTH: usize = 10_000;
const MIN_TIMEOUT_SECS: u64 = 1;
const MAX_TIMEOUT_SECS: u64 = 3600; // 1 hour
const MAX_TOKENS_LIMIT: usize = 200_000;
const MAX_TOOL_NAME_LENGTH: usize = 100;

/// Configuration options for Claude AI client
///
/// The `Config` struct holds all configuration options for the Claude AI client,
/// including model selection, system prompts, tool permissions, and output formatting.
///
/// # Examples
///
/// ```rust
/// use claude_sdk_rs::core::{Config, StreamFormat};
/// use std::path::PathBuf;
///
/// // Default configuration
/// let config = Config::default();
///
/// // Custom configuration with builder pattern
/// let config = Config::builder()
///     .model("claude-3-opus-20240229")
///     .system_prompt("You are a helpful Rust programming assistant")
///     .stream_format(StreamFormat::Json)
///     .timeout_secs(60)
///     .allowed_tools(vec!["bash".to_string(), "filesystem".to_string()])
///     .build();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Optional system prompt to set the assistant's behavior and context
    ///
    /// This prompt is sent with every request to provide consistent context
    /// and instructions to Claude about how it should respond.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// Claude model to use for requests
    ///
    /// Available models include:
    /// - `claude-3-opus-20240229` - Most capable model
    /// - `claude-3-sonnet-20240229` - Balanced performance and cost
    /// - `claude-3-haiku-20240307` - Fastest and most cost-effective
    ///
    /// If not specified, Claude CLI will use its default model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Path to Model Context Protocol (MCP) configuration file
    ///
    /// MCP allows Claude to interact with external tools and data sources.
    /// This should point to a valid MCP config file containing server definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_config_path: Option<PathBuf>,

    /// List of tools that Claude is allowed to use
    ///
    /// Tools are specified using the format `server_name__tool_name` for MCP tools
    /// or simple names like `bash` for built-in tools. An empty list or `None`
    /// means all available tools are allowed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let tools = vec![
    ///     "bash".to_string(),
    ///     "filesystem".to_string(),
    ///     "mcp_server__database_query".to_string(),
    /// ];
    /// ```
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,

    /// Output format for Claude CLI responses
    ///
    /// - `Text`: Plain text output (default)
    /// - `Json`: Structured JSON with metadata
    /// - `StreamJson`: Line-delimited JSON messages for streaming
    #[serde(default)]
    pub stream_format: StreamFormat,

    /// Whether to run Claude CLI in non-interactive mode
    ///
    /// When `true` (default), Claude CLI won't prompt for user input,
    /// making it suitable for programmatic use.
    #[serde(default)]
    pub non_interactive: bool,

    /// Enable verbose output from Claude CLI
    ///
    /// When `true`, additional debugging information will be included
    /// in the CLI output. Useful for troubleshooting.
    #[serde(default)]
    pub verbose: bool,

    /// Maximum number of tokens to generate in the response
    ///
    /// If not specified, Claude will use its default token limit.
    /// Setting this can help control response length and costs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,

    /// Timeout in seconds for Claude CLI execution (default: 30s)
    ///
    /// How long to wait for Claude CLI to respond before timing out.
    /// Increase this for complex queries that might take longer to process.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,

    /// Whether to continue the last session
    ///
    /// When `true`, adds the `--continue` flag to resume the most recent
    /// conversation session. This allows for conversation continuity
    /// across multiple API calls.
    #[serde(default)]
    pub continue_session: bool,

    /// Session ID to resume
    ///
    /// When set, adds the `--resume` flag with the specified session ID
    /// to continue a specific conversation session. This allows for
    /// resuming conversations from a particular point.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume_session_id: Option<String>,

    /// Additional system prompt to append
    ///
    /// When set, adds the `--append-system-prompt` flag to extend the
    /// existing system prompt. Cannot be used together with `system_prompt`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub append_system_prompt: Option<String>,

    /// List of tools that Claude is NOT allowed to use
    ///
    /// Tools are specified using the same format as `allowed_tools`.
    /// This provides fine-grained control over tool restrictions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disallowed_tools: Option<Vec<String>>,

    /// Maximum number of conversation turns
    ///
    /// When set, limits the conversation to the specified number of
    /// back-and-forth exchanges. Useful for controlling conversation length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,

    /// Whether to skip permission prompts (default: true)
    ///
    /// When `true`, adds the `--dangerously-skip-permissions` flag to
    /// bypass tool permission prompts. This is enabled by default for
    /// programmatic use but can be disabled for additional security.
    #[serde(default = "default_skip_permissions")]
    pub skip_permissions: bool,

    /// Security validation strictness level
    ///
    /// Controls how strictly user input is validated for potential security threats.
    /// - `Strict`: Blocks most special characters and patterns
    /// - `Balanced`: Context-aware validation (default)
    /// - `Relaxed`: Only blocks obvious attack patterns
    /// - `Disabled`: No security validation (use with caution)
    #[serde(default)]
    pub security_level: SecurityLevel,
}

/// Output format for Claude CLI responses
///
/// Controls how the Claude CLI formats its output, affecting both parsing
/// and the amount of metadata available in responses.
///
/// # Examples
///
/// ```rust
/// use claude_sdk_rs::core::StreamFormat;
///
/// // For simple text responses (default)
/// let format = StreamFormat::Text;
///
/// // For structured responses with metadata
/// let format = StreamFormat::Json;
///
/// // For streaming applications with line-delimited JSON
/// let format = StreamFormat::StreamJson;
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum StreamFormat {
    /// Plain text output without metadata
    ///
    /// This is the default format. Claude CLI returns only the text content
    /// of the response, making it simple to use but without access to
    /// metadata like costs, session IDs, or token usage.
    #[default]
    Text,

    /// Structured JSON output with full metadata
    ///
    /// Claude CLI returns a complete JSON object containing the response text
    /// along with metadata such as:
    /// - Session ID
    /// - Cost information
    /// - Token usage statistics
    /// - Timing information
    Json,

    /// Line-delimited JSON messages for streaming
    ///
    /// Each line contains a separate JSON message, allowing for real-time
    /// processing of the response as it's generated. Useful for implementing
    /// streaming interfaces or progress indicators.
    StreamJson,
}

/// Default value for skip_permissions field
fn default_skip_permissions() -> bool {
    true
}

/// Security validation strictness level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SecurityLevel {
    /// Strict security - blocks most special characters and patterns
    Strict,
    /// Balanced security - context-aware validation (default)
    #[default]
    Balanced,
    /// Relaxed security - only blocks obvious attack patterns
    Relaxed,
    /// Disabled - no security validation (use with caution)
    Disabled,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            system_prompt: None,
            model: None,
            mcp_config_path: None,
            allowed_tools: None,
            stream_format: StreamFormat::default(),
            non_interactive: true,
            verbose: false,
            max_tokens: None,
            timeout_secs: Some(30), // Default 30 second timeout
            continue_session: false,
            resume_session_id: None,
            append_system_prompt: None,
            disallowed_tools: None,
            max_turns: None,
            skip_permissions: default_skip_permissions(),
            security_level: SecurityLevel::default(),
        }
    }
}

impl Config {
    /// Create a new configuration builder
    ///
    /// The builder pattern provides a fluent interface for creating
    /// configurations with custom settings.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::{Config, StreamFormat};
    ///
    /// let config = Config::builder()
    ///     .model("claude-3-opus-20240229")
    ///     .system_prompt("You are a helpful assistant")
    ///     .stream_format(StreamFormat::Json)
    ///     .timeout_secs(120)
    ///     .build();
    /// ```
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::new()
    }

    /// Validate the configuration
    ///
    /// Checks all configuration values for validity according to defined limits
    /// and constraints. Returns an error if any validation fails.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if:
    /// - System prompt exceeds maximum length
    /// - Timeout is outside valid range
    /// - Max tokens exceeds limit
    /// - Tool names are invalid
    pub fn validate(&self) -> Result<(), Error> {
        // Validate system prompt length
        if let Some(prompt) = &self.system_prompt {
            if prompt.len() > MAX_SYSTEM_PROMPT_LENGTH {
                return Err(Error::InvalidInput(format!(
                    "System prompt exceeds maximum length of {} characters (got {})",
                    MAX_SYSTEM_PROMPT_LENGTH,
                    prompt.len()
                )));
            }

            // Check for potentially malicious content
            if contains_malicious_patterns(prompt) {
                return Err(Error::InvalidInput(
                    "System prompt contains potentially malicious content".to_string(),
                ));
            }
        }

        // Validate timeout
        if let Some(timeout) = self.timeout_secs {
            if timeout < MIN_TIMEOUT_SECS || timeout > MAX_TIMEOUT_SECS {
                return Err(Error::InvalidInput(format!(
                    "Timeout must be between {} and {} seconds (got {})",
                    MIN_TIMEOUT_SECS, MAX_TIMEOUT_SECS, timeout
                )));
            }
        }

        // Validate max tokens
        if let Some(max_tokens) = self.max_tokens {
            if max_tokens == 0 || max_tokens > MAX_TOKENS_LIMIT {
                return Err(Error::InvalidInput(format!(
                    "Max tokens must be between 1 and {} (got {})",
                    MAX_TOKENS_LIMIT, max_tokens
                )));
            }
        }

        // Validate allowed tools with granular permission parsing
        if let Some(tools) = &self.allowed_tools {
            for tool in tools {
                if tool.is_empty() || tool.len() > MAX_TOOL_NAME_LENGTH {
                    return Err(Error::InvalidInput(format!(
                        "Tool name length must be between 1 and {} characters (got '{}')",
                        MAX_TOOL_NAME_LENGTH, tool
                    )));
                }

                // Use enhanced granular permission parsing and validation
                if let Err(e) = crate::core::types::ToolPermission::parse_granular(tool) {
                    return Err(Error::InvalidInput(format!(
                        "Invalid tool permission format: '{}'. Error: {}",
                        tool, e
                    )));
                }
            }
        }

        // Validate disallowed tools with granular permission parsing
        if let Some(tools) = &self.disallowed_tools {
            for tool in tools {
                if tool.is_empty() || tool.len() > MAX_TOOL_NAME_LENGTH {
                    return Err(Error::InvalidInput(format!(
                        "Disallowed tool name length must be between 1 and {} characters (got '{}')",
                        MAX_TOOL_NAME_LENGTH, tool
                    )));
                }

                // Use enhanced granular permission parsing and validation
                if let Err(e) = crate::core::types::ToolPermission::parse_granular(tool) {
                    return Err(Error::InvalidInput(format!(
                        "Invalid disallowed tool permission format: '{}'. Error: {}",
                        tool, e
                    )));
                }
            }
        }

        // Validate MCP config path
        if let Some(path) = &self.mcp_config_path {
            if path.as_os_str().is_empty() {
                return Err(Error::InvalidInput(
                    "MCP config path cannot be empty".to_string(),
                ));
            }
        }

        // Validate max_turns is positive when set
        if let Some(turns) = self.max_turns {
            if turns == 0 {
                return Err(Error::InvalidInput(
                    "Max turns must be greater than 0".to_string(),
                ));
            }
        }

        // Validate no conflicts between allowed_tools and disallowed_tools
        if let (Some(allowed), Some(disallowed)) = (&self.allowed_tools, &self.disallowed_tools) {
            for tool in disallowed {
                if allowed.contains(tool) {
                    return Err(Error::InvalidInput(format!(
                        "Tool '{}' cannot be both allowed and disallowed",
                        tool
                    )));
                }
            }
        }

        // Validate system_prompt and append_system_prompt aren't both set
        if self.system_prompt.is_some() && self.append_system_prompt.is_some() {
            return Err(Error::InvalidInput(
                "Cannot use both system_prompt and append_system_prompt simultaneously".to_string(),
            ));
        }

        // Validate append_system_prompt length
        if let Some(prompt) = &self.append_system_prompt {
            if prompt.len() > MAX_SYSTEM_PROMPT_LENGTH {
                return Err(Error::InvalidInput(format!(
                    "Append system prompt exceeds maximum length of {} characters (got {})",
                    MAX_SYSTEM_PROMPT_LENGTH,
                    prompt.len()
                )));
            }

            if contains_malicious_patterns(prompt) {
                return Err(Error::InvalidInput(
                    "Append system prompt contains potentially malicious content".to_string(),
                ));
            }
        }

        // Validate session ID format for resume functionality
        if let Some(session_id) = &self.resume_session_id {
            if session_id.is_empty() {
                return Err(Error::InvalidInput(
                    "Resume session ID cannot be empty".to_string(),
                ));
            }

            if session_id.len() > 100 {
                return Err(Error::InvalidInput(
                    "Resume session ID exceeds maximum length of 100 characters".to_string(),
                ));
            }

            // Basic format validation - session IDs should be alphanumeric with limited special chars
            if !session_id
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
            {
                return Err(Error::InvalidInput(
                    "Resume session ID contains invalid characters. Only alphanumeric, underscore, and hyphen are allowed".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Builder for creating `Config` instances with fluent configuration
///
/// The `ConfigBuilder` provides a convenient way to construct configuration
/// objects using the builder pattern. All methods are chainable and return
/// `self` for fluent composition.
///
/// # Examples
///
/// ```rust
/// use claude_sdk_rs::core::{Config, StreamFormat};
///
/// let config = Config::builder()
///     .model("claude-3-sonnet-20240229")
///     .system_prompt("You are an expert Rust developer")
///     .stream_format(StreamFormat::Json)
///     .max_tokens(4096)
///     .timeout_secs(60)
///     .allowed_tools(vec!["bash".to_string(), "filesystem".to_string()])
///     .build();
/// ```
pub struct ConfigBuilder {
    config: Config,
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigBuilder {
    /// Create a new configuration builder with default settings
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    /// Set the system prompt for the assistant
    ///
    /// The system prompt provides context and instructions that influence
    /// how Claude responds to all queries in a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    ///
    /// let config = Config::builder()
    ///     .system_prompt("You are a helpful Rust programming assistant")
    ///     .build();
    /// ```
    #[must_use]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    /// Set the Claude model to use
    ///
    /// Specify which Claude model should handle the requests. Different models
    /// have different capabilities, speed, and cost characteristics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    ///
    /// let config = Config::builder()
    ///     .model("claude-3-opus-20240229")  // Most capable
    ///     .build();
    /// ```
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.model = Some(model.into());
        self
    }

    /// Set the path to the MCP (Model Context Protocol) configuration file
    ///
    /// MCP allows Claude to interact with external tools and data sources.
    /// The config file should contain server definitions and tool configurations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    /// use std::path::PathBuf;
    ///
    /// let config = Config::builder()
    ///     .mcp_config(PathBuf::from("./mcp-config.json"))
    ///     .build();
    /// ```
    #[must_use]
    pub fn mcp_config(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.mcp_config_path = Some(path.into());
        self
    }

    /// Set the list of allowed tools
    ///
    /// Controls which tools Claude can access during execution. Use this
    /// to restrict capabilities for security or to focus on specific tool sets.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    ///
    /// let config = Config::builder()
    ///     .allowed_tools(vec![
    ///         "bash".to_string(),
    ///         "filesystem".to_string(),
    ///         "calculator".to_string(),
    ///     ])
    ///     .build();
    /// ```
    #[must_use]
    pub fn allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.config.allowed_tools = Some(tools);
        self
    }

    /// Set the output format for Claude CLI responses
    ///
    /// Choose between plain text, structured JSON, or streaming JSON formats
    /// depending on your application's needs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::{Config, StreamFormat};
    ///
    /// let config = Config::builder()
    ///     .stream_format(StreamFormat::Json)
    ///     .build();
    /// ```
    #[must_use]
    pub fn stream_format(mut self, format: StreamFormat) -> Self {
        self.config.stream_format = format;
        self
    }

    /// Set whether to run in non-interactive mode
    ///
    /// When `true`, Claude CLI won't prompt for user input, making it
    /// suitable for programmatic use. This is usually `true` for SDK usage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    ///
    /// let config = Config::builder()
    ///     .non_interactive(true)
    ///     .build();
    /// ```
    #[must_use]
    pub fn non_interactive(mut self, non_interactive: bool) -> Self {
        self.config.non_interactive = non_interactive;
        self
    }

    /// Set the maximum number of tokens to generate
    ///
    /// Limits the length of Claude's responses. Useful for controlling
    /// costs and ensuring responses fit within expected bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    ///
    /// let config = Config::builder()
    ///     .max_tokens(2048)  // Limit to 2K tokens
    ///     .build();
    /// ```
    #[must_use]
    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Set the timeout in seconds for Claude CLI execution
    ///
    /// How long to wait for Claude CLI to respond before giving up.
    /// Increase for complex queries or slow network conditions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    ///
    /// let config = Config::builder()
    ///     .timeout_secs(120)  // 2 minute timeout
    ///     .build();
    /// ```
    #[must_use]
    pub fn timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.config.timeout_secs = Some(timeout_secs);
        self
    }

    /// Set whether to enable verbose output from Claude CLI
    ///
    /// When `true`, additional debugging information will be included
    /// in the CLI output. Useful for troubleshooting.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    ///
    /// let config = Config::builder()
    ///     .verbose(true)  // Enable verbose output
    ///     .build();
    /// ```
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Enable session continuation
    ///
    /// When enabled, adds the `--continue` flag to resume the most recent
    /// conversation session, allowing for conversation continuity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    ///
    /// let config = Config::builder()
    ///     .continue_session()
    ///     .build();
    /// ```
    #[must_use]
    pub fn continue_session(mut self) -> Self {
        self.config.continue_session = true;
        self
    }

    /// Set a specific session ID to resume
    ///
    /// When set, adds the `--resume` flag with the specified session ID
    /// to continue a specific conversation session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    ///
    /// let config = Config::builder()
    ///     .resume_session("session_123".to_string())
    ///     .build();
    /// ```
    #[must_use]
    pub fn resume_session(mut self, session_id: String) -> Self {
        self.config.resume_session_id = Some(session_id);
        self
    }

    /// Set an additional system prompt to append
    ///
    /// When set, adds the `--append-system-prompt` flag to extend the
    /// existing system prompt. Cannot be used with `system_prompt`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    ///
    /// let config = Config::builder()
    ///     .append_system_prompt("Additionally, be concise in your responses.")
    ///     .build();
    /// ```
    #[must_use]
    pub fn append_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.append_system_prompt = Some(prompt.into());
        self
    }

    /// Set the list of disallowed tools
    ///
    /// Controls which tools Claude cannot access during execution.
    /// Provides fine-grained control over tool restrictions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    ///
    /// let config = Config::builder()
    ///     .disallowed_tools(vec![
    ///         "bash".to_string(),
    ///         "filesystem".to_string(),
    ///     ])
    ///     .build();
    /// ```
    #[must_use]
    pub fn disallowed_tools(mut self, tools: Vec<String>) -> Self {
        self.config.disallowed_tools = Some(tools);
        self
    }

    /// Set the maximum number of conversation turns
    ///
    /// Limits the conversation to the specified number of back-and-forth
    /// exchanges. Useful for controlling conversation length.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    ///
    /// let config = Config::builder()
    ///     .max_turns(10)
    ///     .build();
    /// ```
    #[must_use]
    pub fn max_turns(mut self, turns: u32) -> Self {
        self.config.max_turns = Some(turns);
        self
    }

    /// Set whether to skip permission prompts
    ///
    /// When `true` (default), adds the `--dangerously-skip-permissions` flag
    /// to bypass tool permission prompts. Set to `false` for additional security.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::Config;
    ///
    /// let config = Config::builder()
    ///     .skip_permissions(false)  // Require permission prompts
    ///     .build();
    /// ```
    #[must_use]
    pub fn skip_permissions(mut self, skip: bool) -> Self {
        self.config.skip_permissions = skip;
        self
    }

    /// Set the security validation level
    ///
    /// Controls how strictly user input is validated for potential security threats.
    /// - `Strict`: Blocks most special characters and patterns
    /// - `Balanced`: Context-aware validation (default)
    /// - `Relaxed`: Only blocks obvious attack patterns
    /// - `Disabled`: No security validation (use with caution)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::{Config, SecurityLevel};
    ///
    /// let config = Config::builder()
    ///     .security_level(SecurityLevel::Relaxed)
    ///     .build();
    /// ```
    #[must_use]
    pub fn security_level(mut self, level: SecurityLevel) -> Self {
        self.config.security_level = level;
        self
    }

    /// Build the final configuration
    ///
    /// Consumes the builder and returns the constructed `Config` instance.
    /// Validates the configuration before returning it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use claude_sdk_rs::core::{Config, StreamFormat};
    ///
    /// let config = Config::builder()
    ///     .model("claude-3-sonnet-20240229")
    ///     .stream_format(StreamFormat::Json)
    ///     .timeout_secs(60)
    ///     .build()
    ///     .expect("valid configuration");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `Error::ConfigError` if the configuration is invalid
    pub fn build(self) -> Result<Config, Error> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Validate query input
///
/// Checks that a query string meets all validation requirements including
/// length limits and content validation.
///
/// # Errors
///
/// Returns `Error::ConfigError` if:
/// - Query exceeds maximum length
/// - Query contains malicious content
/// - Query is empty
pub fn validate_query(query: &str) -> Result<(), Error> {
    validate_query_with_security_level(query, SecurityLevel::default())
}

/// Validate query input with specific security level
///
/// Checks that a query string meets all validation requirements including
/// length limits and content validation based on the specified security level.
///
/// # Errors
///
/// Returns `Error::ConfigError` if:
/// - Query exceeds maximum length
/// - Query contains malicious content (based on security level)
/// - Query is empty
pub fn validate_query_with_security_level(query: &str, security_level: SecurityLevel) -> Result<(), Error> {
    if query.is_empty() {
        return Err(Error::InvalidInput("Query cannot be empty".to_string()));
    }

    if query.len() > MAX_QUERY_LENGTH {
        return Err(Error::InvalidInput(format!(
            "Query exceeds maximum length of {} characters (got {})",
            MAX_QUERY_LENGTH,
            query.len()
        )));
    }

    match security_level {
        SecurityLevel::Disabled => Ok(()),
        SecurityLevel::Relaxed => {
            if contains_obvious_malicious_patterns(query) {
                return Err(Error::InvalidInput(
                    "Query contains potentially malicious content".to_string(),
                ));
            }
            Ok(())
        }
        SecurityLevel::Balanced => {
            if contains_malicious_patterns(query) {
                return Err(Error::InvalidInput(
                    "Query contains potentially malicious content".to_string(),
                ));
            }
            Ok(())
        }
        SecurityLevel::Strict => {
            if contains_strict_malicious_patterns(query) {
                return Err(Error::InvalidInput(
                    "Query contains potentially malicious content".to_string(),
                ));
            }
            Ok(())
        }
    }
}

/// Check if a string contains potentially malicious patterns
fn contains_malicious_patterns(text: &str) -> bool {
    // Early return for obviously safe patterns
    if text.chars().all(|c| c.is_alphanumeric() || c.is_whitespace() || c == '-' || c == '_' || c == '.') {
        return false;
    }

    let lower_text = text.to_lowercase();
    
    // Check for definite malicious patterns (keep these strict)
    let strict_patterns = [
        // Script injection
        "<script",
        "javascript:",
        "onclick=",
        "onerror=",
        // Path traversal
        "../",
        "..\\",
        // SQL injection with context
        "' or '",
        "\" or \"",
        "'; drop",
        "\"; drop",
        "' or 1=1",
        "\" or 1=1",
        // Null bytes
        "\0",
    ];
    
    if strict_patterns.iter().any(|pattern| lower_text.contains(pattern)) {
        return true;
    }
    
    // For command injection patterns, use more context-aware checks
    // Check if multiple suspicious patterns appear together
    let command_indicators = [
        ("$(", ")"),
        ("${", "}"),
        ("&&", ""),
        ("||", ""),
        (";", ""),
        ("|", ""),
    ];
    
    let mut suspicious_count = 0;
    for (start, _) in &command_indicators {
        if text.contains(start) {
            suspicious_count += 1;
        }
    }
    
    // Flag if multiple command injection patterns are present
    // or if && or || appear (common command chaining)
    // or if $( or ${ appear (command substitution)
    // or if ; followed by dangerous commands
    if suspicious_count >= 2 || text.contains("&&") || text.contains("||") 
       || text.contains("$(") || text.contains("${") {
        return true;
    }
    
    // Check for semicolon or pipe followed by dangerous commands
    if text.contains(';') || text.contains('|') {
        let dangerous_commands = ["rm", "del", "format", "fdisk", "chmod", "chown", "malicious_command"];
        for cmd in &dangerous_commands {
            if text.contains(&format!("; {}", cmd)) || text.contains(&format!(";{}", cmd)) ||
               text.contains(&format!("| {}", cmd)) || text.contains(&format!("|{}", cmd)) {
                return true;
            }
        }
    }
    
    // Check for backticks only if they appear to be command substitution
    if text.contains('`') {
        // Allow single backticks for markdown code formatting
        let backtick_count = text.matches('`').count();
        if backtick_count >= 2 || text.contains("``") {
            // Check if it looks like command substitution
            if text.contains("`$") || text.contains("`;") || text.contains("`|") {
                return true;
            }
        }
    }
    
    // Check for redirection only in suspicious contexts
    if (text.contains('>') || text.contains('<')) && 
       (text.contains("2>") || text.contains("&>") || text.contains("1>") || 
        text.contains("<(") || text.contains(">(")) {
        return true;
    }
    
    false
}

/// Check for only the most obvious malicious patterns (relaxed mode)
fn contains_obvious_malicious_patterns(text: &str) -> bool {
    let lower_text = text.to_lowercase();
    
    // Only check for very obvious attack patterns
    let obvious_patterns = [
        // Script injection
        "<script",
        "javascript:",
        // Path traversal
        "../",
        "..\\",
        // SQL injection with clear intent
        "'; drop table",
        "\"; drop table",
        "' or 1=1--",
        "\" or 1=1--",
        // Null bytes
        "\0",
    ];
    
    obvious_patterns.iter().any(|pattern| lower_text.contains(pattern))
}

/// Check using strict patterns (strict mode) 
fn contains_strict_malicious_patterns(text: &str) -> bool {
    // Block if text contains any special characters commonly used in attacks
    let has_special_chars = text.chars().any(|c| matches!(c, 
        '`' | '$' | '{' | '}' | '(' | ')' | '[' | ']' | 
        ';' | '|' | '&' | '<' | '>' | '"' | '\'' | '\\' | 
        '\n' | '\r' | '\t' | '\0'
    ));
    
    if has_special_chars {
        return true;
    }
    
    // Also check for encoded patterns
    let lower_text = text.to_lowercase();
    let encoded_patterns = [
        "%3c", // <
        "%3e", // >
        "%22", // "
        "%27", // '
        "%2e%2e", // ..
        "%00", // null byte
    ];
    
    encoded_patterns.iter().any(|pattern| lower_text.contains(pattern))
}
