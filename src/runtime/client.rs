use crate::core::{
    validate_query, ClaudeCliResponse, ClaudeResponse, Config, Result, SessionId, StreamFormat,
};
use crate::runtime::{process::execute_claude, stream::MessageStream};
use std::sync::Arc;

/// Helper function to extract text from an assistant message
fn extract_text_from_message(msg: &serde_json::Value, result: &mut String) {
    if let Some(message) = msg.get("message") {
        if let Some(content_array) = message.get("content").and_then(|v| v.as_array()) {
            for content_item in content_array {
                if content_item.get("type").and_then(|v| v.as_str()) == Some("text") {
                    if let Some(text) = content_item.get("text").and_then(|v| v.as_str()) {
                        result.push_str(text);
                    }
                }
            }
        }
    }
}

/// High-level client for interacting with Claude Code CLI
///
/// The `Client` provides a type-safe, async interface to Claude Code with support
/// for different output formats, configuration options, and both simple and advanced
/// response handling.
///
/// # Examples
///
/// Basic usage:
/// ```rust,no_run
/// # use crate::core::*;
/// # use claude_sdk_rs_runtime::Client;
/// # #[tokio::main]
/// # async fn main() -> crate::core::Result<()> {
/// let client = Client::new(Config::default());
/// let response = client.query("Hello").send().await?;
/// println!("{}", response);
/// # Ok(())
/// # }
/// ```
///
/// With configuration:
/// ```rust,no_run
/// # use crate::core::*;
/// # use claude_sdk_rs_runtime::Client;
/// # #[tokio::main]
/// # async fn main() -> crate::core::Result<()> {
/// let client = Client::builder()
///     .model("claude-3-opus-20240229")
///     .stream_format(StreamFormat::Json)
///     .timeout_secs(60)
///     .build();
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Client {
    config: Arc<Config>,
}

impl Client {
    /// Create a new client with the given configuration
    pub fn new(config: Config) -> Self {
        Self {
            config: Arc::new(config),
        }
    }

    /// Create a new client builder for fluent configuration
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Create a query builder for the given query string
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use crate::core::*;
    /// # use claude_sdk_rs_runtime::Client;
    /// # #[tokio::main]
    /// # async fn main() -> crate::core::Result<()> {
    /// let client = Client::new(Config::default());
    /// let response = client
    ///     .query("Explain Rust ownership")
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn query(&self, query: impl Into<String>) -> QueryBuilder {
        QueryBuilder::new(self.clone(), query.into())
    }

    /// Send a query and return just the text content (backwards compatible)
    ///
    /// This is the simplest way to get a response from Claude. For access to
    /// metadata, costs, and raw JSON, use [`send_full`](Self::send_full).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use crate::core::*;
    /// # use claude_sdk_rs_runtime::Client;
    /// # #[tokio::main]
    /// # async fn main() -> crate::core::Result<()> {
    /// let client = Client::new(Config::default());
    /// let answer = client.send("What is 2 + 2?").await?;
    /// assert_eq!(answer.trim(), "4");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send(&self, query: &str) -> Result<String> {
        validate_query(query)?;
        let response = self.send_full(query).await?;
        Ok(response.content)
    }

    /// Send a query and return the full response with metadata and raw JSON
    ///
    /// This method provides access to the complete response from Claude Code,
    /// including metadata like costs, session IDs, and the raw JSON for
    /// advanced parsing or storage.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use crate::core::*;
    /// # use claude_sdk_rs_runtime::Client;
    /// # #[tokio::main]
    /// # async fn main() -> crate::core::Result<()> {
    /// let client = Client::builder()
    ///     .stream_format(StreamFormat::Json)
    ///     .build();
    ///
    /// let response = client.send_full("Hello").await?;
    /// println!("Content: {}", response.content);
    ///
    /// if let Some(metadata) = &response.metadata {
    ///     println!("Cost: ${:.6}", metadata.cost_usd.unwrap_or(0.0));
    ///     println!("Session: {}", metadata.session_id);
    /// }
    ///
    /// // Access raw JSON for custom parsing
    /// if let Some(raw) = &response.raw_json {
    ///     // Custom field extraction
    ///     let custom_field = raw.get("custom_field");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send_full(&self, query: &str) -> Result<ClaudeResponse> {
        validate_query(query)?;
        let output = execute_claude(&self.config, query).await?;

        // Parse response based on format
        match self.config.stream_format {
            StreamFormat::Text => Ok(ClaudeResponse::text(output.trim().to_string())),
            StreamFormat::Json => {
                // Parse the JSON response from claude CLI
                let json_value: serde_json::Value = serde_json::from_str(&output)?;
                let claude_response: ClaudeCliResponse =
                    serde_json::from_value(json_value.clone())?;
                Ok(ClaudeResponse::with_json(
                    claude_response.result,
                    json_value,
                ))
            }
            StreamFormat::StreamJson => {
                // For stream-json, we need to parse multiple JSON lines
                let mut result = String::new();
                let mut all_json = Vec::new();

                for line in output.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    // Try to parse as a message
                    if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
                        all_json.push(msg.clone());

                        // Check if it's an assistant message and extract text
                        if msg.get("type").and_then(|v| v.as_str()) == Some("assistant") {
                            extract_text_from_message(&msg, &mut result);
                        }
                    }
                }

                // Return the response with all JSON messages as an array
                let raw_json = serde_json::Value::Array(all_json);
                Ok(ClaudeResponse::with_json(result, raw_json))
            }
        }
    }
}

/// Builder for creating `Client` instances with fluent configuration
///
/// The `ClientBuilder` provides a convenient way to construct client instances
/// using the builder pattern. All methods are chainable and return `self` for
/// fluent composition.
///
/// # Examples
///
/// ```rust,no_run
/// # use crate::core::*;
/// # use claude_sdk_rs_runtime::Client;
/// let client = Client::builder()
///     .model("claude-3-sonnet-20240229")
///     .system_prompt("You are a helpful assistant")
///     .stream_format(StreamFormat::Json)
///     .timeout_secs(60)
///     .build();
/// ```
pub struct ClientBuilder {
    config: Config,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ClientBuilder {
    /// Create a new client builder with default configuration
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    /// Set the configuration directly
    ///
    /// This allows you to use a pre-built `Config` instance instead of
    /// configuring individual options.
    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Set the system prompt for the assistant
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use claude_sdk_rs_runtime::Client;
    /// let client = Client::builder()
    ///     .system_prompt("You are a Rust expert")
    ///     .build();
    /// ```
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    /// Set the Claude model to use
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use claude_sdk_rs_runtime::Client;
    /// let client = Client::builder()
    ///     .model("claude-3-opus-20240229")
    ///     .build();
    /// ```
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.model = Some(model.into());
        self
    }

    /// Set the list of allowed tools
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use claude_sdk_rs_runtime::Client;
    /// let client = Client::builder()
    ///     .allowed_tools(vec!["bash".to_string(), "filesystem".to_string()])
    ///     .build();
    /// ```
    pub fn allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.config.allowed_tools = Some(tools);
        self
    }

    /// Set the output format for responses
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use claude_sdk_rs_runtime::Client;
    /// # use crate::core::StreamFormat;
    /// let client = Client::builder()
    ///     .stream_format(StreamFormat::Json)
    ///     .build();
    /// ```
    pub fn stream_format(mut self, format: StreamFormat) -> Self {
        self.config.stream_format = format;
        self
    }

    /// Enable or disable verbose output
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use claude_sdk_rs_runtime::Client;
    /// let client = Client::builder()
    ///     .verbose(true)
    ///     .build();
    /// ```
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set the timeout in seconds
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use claude_sdk_rs_runtime::Client;
    /// let client = Client::builder()
    ///     .timeout_secs(120)  // 2 minute timeout
    ///     .build();
    /// ```
    pub fn timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.config.timeout_secs = Some(timeout_secs);
        self
    }

    /// Enable session continuation (--continue flag)
    ///
    /// When enabled, the client will use the --continue flag to resume
    /// the most recent conversation session.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use claude_sdk_rs_runtime::Client;
    /// let client = Client::builder()
    ///     .continue_session()
    ///     .build();
    /// ```
    pub fn continue_session(mut self) -> Self {
        self.config.continue_session = true;
        self
    }

    /// Resume a specific session by ID (--resume flag)
    ///
    /// When set, the client will use the --resume flag with the specified
    /// session ID to continue a specific conversation session.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use claude_sdk_rs_runtime::Client;
    /// let client = Client::builder()
    ///     .resume_session("session_123")
    ///     .build();
    /// ```
    pub fn resume_session(mut self, session_id: impl Into<String>) -> Self {
        self.config.resume_session_id = Some(session_id.into());
        self
    }

    /// Set the list of disallowed tools
    ///
    /// Controls which tools Claude cannot access during execution.
    /// Provides fine-grained control over tool restrictions.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use claude_sdk_rs_runtime::Client;
    /// let client = Client::builder()
    ///     .disallowed_tools(vec!["bash".to_string(), "filesystem".to_string()])
    ///     .build();
    /// ```
    pub fn disallowed_tools(mut self, tools: Vec<String>) -> Self {
        self.config.disallowed_tools = Some(tools);
        self
    }

    /// Set whether to skip permission prompts (default: true)
    ///
    /// When `true` (default), adds the `--dangerously-skip-permissions` flag
    /// to bypass tool permission prompts. Set to `false` for additional security.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use claude_sdk_rs_runtime::Client;
    /// let client = Client::builder()
    ///     .skip_permissions(false)  // Require permission prompts
    ///     .build();
    /// ```
    pub fn skip_permissions(mut self, skip: bool) -> Self {
        self.config.skip_permissions = skip;
        self
    }

    /// Set an additional system prompt to append
    ///
    /// When set, adds the `--append-system-prompt` flag to extend the
    /// existing system prompt. Cannot be used with `system_prompt`.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use claude_sdk_rs_runtime::Client;
    /// let client = Client::builder()
    ///     .append_system_prompt("Additionally, be concise in your responses.")
    ///     .build();
    /// ```
    pub fn append_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.append_system_prompt = Some(prompt.into());
        self
    }

    /// Set the maximum number of conversation turns
    ///
    /// Limits the conversation to the specified number of back-and-forth
    /// exchanges. Useful for controlling conversation length.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use claude_sdk_rs_runtime::Client;
    /// let client = Client::builder()
    ///     .max_turns(10)
    ///     .build();
    /// ```
    pub fn max_turns(mut self, turns: u32) -> Self {
        self.config.max_turns = Some(turns);
        self
    }

    /// Build the final client instance
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use claude_sdk_rs_runtime::Client;
    /// # use crate::core::StreamFormat;
    /// let client = Client::builder()
    ///     .model("claude-3-sonnet-20240229")
    ///     .stream_format(StreamFormat::Json)
    ///     .build()
    ///     .expect("valid configuration");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid
    pub fn build(self) -> Result<Client> {
        self.config.validate()?;
        Ok(Client::new(self.config))
    }
}

/// Builder for constructing and executing Claude queries
///
/// The `QueryBuilder` provides a fluent interface for configuring queries
/// before sending them to Claude. It supports different response formats
/// and execution modes.
///
/// # Examples
///
/// ```rust,no_run
/// # use crate::core::*;
/// # use claude_sdk_rs_runtime::Client;
/// # #[tokio::main]
/// # async fn main() -> crate::core::Result<()> {
/// # let client = Client::new(Config::default());
/// // Simple query
/// let response = client
///     .query("What is Rust?")
///     .send()
///     .await?;
///
/// // Query with session and custom format
/// let response = client
///     .query("Continue the conversation")
///     .session("my-session".to_string())
///     .format(StreamFormat::Json)
///     .send_full()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct QueryBuilder {
    client: Client,
    query: String,
    session_id: Option<SessionId>,
    format: Option<StreamFormat>,
}

impl QueryBuilder {
    /// Create a new query builder (internal use)
    fn new(client: Client, query: String) -> Self {
        Self {
            client,
            query,
            session_id: None,
            format: None,
        }
    }

    /// Specify a session ID for this query
    ///
    /// This allows the query to be part of an ongoing conversation
    /// with maintained context.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use crate::core::*;
    /// # use claude_sdk_rs_runtime::Client;
    /// # #[tokio::main]
    /// # async fn main() -> crate::core::Result<()> {
    /// # let client = Client::new(Config::default());
    /// let response = client
    ///     .query("Remember this: the key is 42")
    ///     .session("my-session".to_string())
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn session(mut self, session_id: SessionId) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Override the output format for this specific query
    ///
    /// This allows you to use a different format than the client's
    /// default configuration for this specific query.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use crate::core::*;
    /// # use claude_sdk_rs_runtime::Client;
    /// # #[tokio::main]
    /// # async fn main() -> crate::core::Result<()> {
    /// # let client = Client::new(Config::default());
    /// let response = client
    ///     .query("What is the weather?")
    ///     .format(StreamFormat::Json)
    ///     .send_full()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn format(mut self, format: StreamFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Send the query and return just the text content
    ///
    /// This is the simplest way to get a response from Claude,
    /// returning only the text without metadata.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use crate::core::*;
    /// # use claude_sdk_rs_runtime::Client;
    /// # #[tokio::main]
    /// # async fn main() -> crate::core::Result<()> {
    /// # let client = Client::new(Config::default());
    /// let answer = client
    ///     .query("What is 2 + 2?")
    ///     .send()
    ///     .await?;
    /// println!("Answer: {}", answer);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send(self) -> Result<String> {
        self.client.send(&self.query).await
    }

    /// Send the query and return the full response with metadata
    ///
    /// This provides access to cost information, session IDs, token usage,
    /// and the raw JSON response for advanced use cases.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use crate::core::*;
    /// # use claude_sdk_rs_runtime::Client;
    /// # #[tokio::main]
    /// # async fn main() -> crate::core::Result<()> {
    /// # let client = Client::new(Config::default());
    /// let response = client
    ///     .query("Explain quantum computing")
    ///     .send_full()
    ///     .await?;
    ///
    /// println!("Response: {}", response.content);
    /// if let Some(metadata) = &response.metadata {
    ///     if let Some(cost) = metadata.cost_usd {
    ///         println!("Cost: ${:.6}", cost);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send_full(self) -> Result<ClaudeResponse> {
        self.client.send_full(&self.query).await
    }

    /// Send the query and return a stream of messages
    ///
    /// This allows for real-time processing of Claude's response as it's
    /// being generated, useful for implementing streaming UIs.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use crate::core::*;
    /// # use claude_sdk_rs_runtime::Client;
    /// # use futures::StreamExt;
    /// # #[tokio::main]
    /// # async fn main() -> crate::core::Result<()> {
    /// # let client = Client::new(Config::default());
    /// let mut stream = client
    ///     .query("Write a short story")
    ///     .stream()
    ///     .await?;
    ///
    /// while let Some(message_result) = stream.next().await {
    ///     match message_result {
    ///         Ok(message) => {
    ///             // Process each message as it arrives
    ///             println!("Message: {:?}", message);
    ///         }
    ///         Err(e) => eprintln!("Stream error: {}", e),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stream(self) -> Result<MessageStream> {
        use crate::runtime::process::execute_claude_streaming;

        let format = self.format.unwrap_or(self.client.config.stream_format);

        // Use real streaming by calling the new streaming execute function
        let line_receiver = execute_claude_streaming(&self.client.config, &self.query).await?;

        // Convert the line stream to a message stream
        Ok(MessageStream::from_line_stream(line_receiver, format).await)
    }

    /// Send the query and parse the response as JSON
    ///
    /// This is a convenience method for when you expect Claude to return
    /// structured data that can be deserialized into a specific type.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use crate::core::*;
    /// # use claude_sdk_rs_runtime::Client;
    /// # use serde::Deserialize;
    /// # #[tokio::main]
    /// # async fn main() -> crate::core::Result<()> {
    /// #[derive(Deserialize)]
    /// struct WeatherData {
    ///     temperature: f64,
    ///     humidity: f64,
    /// }
    ///
    /// # let client = Client::new(Config::default());
    /// let weather: WeatherData = client
    ///     .query("Return weather data as JSON: {\"temperature\": 22.5, \"humidity\": 65}")
    ///     .parse_output()
    ///     .await?;
    ///
    /// println!("Temperature: {}°C", weather.temperature);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn parse_output<T: serde::de::DeserializeOwned>(self) -> Result<T> {
        let response = self.send().await?;
        serde_json::from_str(&response).map_err(Into::into)
    }
}
