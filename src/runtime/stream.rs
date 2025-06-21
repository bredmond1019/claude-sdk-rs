use crate::core::message::{ConversationStats, TokenUsage};
use crate::core::{Error, Message, Result, StreamFormat};
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tracing::{debug, error};

/// Stream of messages from Claude AI
///
/// `MessageStream` provides real-time access to Claude's responses as they are generated,
/// enabling streaming user interfaces and progressive content display. It implements the
/// `Stream` trait for easy integration with async Rust applications.
///
/// # Examples
///
/// ```rust,no_run
/// # use claude_sdk_rs_runtime::{Client, MessageStream};
/// # use crate::core::{Config, Message, Result};
/// # use futures::StreamExt;
/// # #[tokio::main]
/// # async fn main() -> Result<()> {
/// let client = Client::new(Config::default());
/// let mut stream = client.query("Write a story").stream().await?;
///
/// // Process messages as they arrive
/// while let Some(result) = stream.next().await {
///     match result {
///         Ok(Message::Assistant { content, .. }) => {
///             print!("{}", content); // Print content incrementally
///         }
///         Ok(Message::Result { stats, .. }) => {
///             println!("\nCompleted with {} tokens", stats.total_tokens);
///         }
///         Err(e) => eprintln!("Stream error: {}", e),
///         _ => {} // Handle other message types
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Stream Behavior
///
/// - Messages arrive in real-time as Claude generates the response
/// - Assistant messages may be split across multiple stream items
/// - The stream ends with a Result message containing statistics
/// - Errors are propagated through the stream rather than terminating it
///
/// # Error Handling
///
/// Errors can occur at any point in the stream. Common error scenarios:
/// - Network interruptions
/// - Invalid JSON parsing (for JSON formats)
/// - Process termination
/// - Timeout exceeded
pub struct MessageStream {
    receiver: mpsc::Receiver<Result<Message>>,
}

impl MessageStream {
    /// Create a new MessageStream from a channel receiver
    ///
    /// This is typically called internally by the Client. The format parameter
    /// is reserved for future use but currently not utilized.
    pub fn new(receiver: mpsc::Receiver<Result<Message>>, _format: StreamFormat) -> Self {
        Self { receiver }
    }

    /// Create a MessageStream from a line receiver and format
    ///
    /// This function takes a receiver of raw output lines from the Claude CLI
    /// and converts them into a stream of parsed Messages based on the format.
    pub async fn from_line_stream(
        mut line_receiver: mpsc::Receiver<Result<String>>,
        format: StreamFormat,
    ) -> Self {
        let config = crate::runtime::stream_config::get_stream_config();
        let (tx, rx) = mpsc::channel(config.channel_buffer_size);

        tokio::spawn(async move {
            let config = crate::runtime::stream_config::get_stream_config();
            let parser = MessageParser::new(format);
            let mut accumulated_content = String::with_capacity(config.string_capacity);

            while let Some(line_result) = line_receiver.recv().await {
                match line_result {
                    Ok(line) => {
                        debug!("Received line: {}", line);

                        match format {
                            StreamFormat::Text => {
                                // For text format, each line is part of the assistant's response
                                accumulated_content.push_str(&line);
                                accumulated_content.push('\n');

                                // Send incremental updates
                                let message = Message::Assistant {
                                    content: line,
                                    meta: crate::core::MessageMeta {
                                        session_id: "stream-session".to_string(),
                                        timestamp: Some(std::time::SystemTime::now()),
                                        cost_usd: None,
                                        duration_ms: None,
                                        tokens_used: None,
                                    },
                                };

                                if tx.send(Ok(message)).await.is_err() {
                                    debug!("Message receiver dropped");
                                    break;
                                }
                            }
                            StreamFormat::Json => {
                                // For JSON format, we expect a single JSON object at the end
                                accumulated_content.push_str(&line);
                                accumulated_content.push('\n');
                            }
                            StreamFormat::StreamJson => {
                                // For StreamJson, each line should be a separate JSON message
                                if let Ok(Some(message)) = parser.parse_line(&line) {
                                    if tx.send(Ok(message)).await.is_err() {
                                        debug!("Message receiver dropped");
                                        break;
                                    }
                                } else if !line.trim().is_empty() {
                                    debug!("Failed to parse line as message: {}", line);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if tx.send(Err(e)).await.is_err() {
                            debug!("Error receiver dropped");
                        }
                        break;
                    }
                }
            }

            // Handle final processing for non-streaming formats
            match format {
                StreamFormat::Json => {
                    // Try to parse the accumulated content as a single JSON response
                    if !accumulated_content.trim().is_empty() {
                        if let Ok(Some(message)) =
                            parser.parse_accumulated_json(&accumulated_content)
                        {
                            let _ = tx.send(Ok(message)).await;
                        }
                    }
                }
                StreamFormat::Text => {
                    // Send a final message indicating completion
                    let final_message = Message::Result {
                        meta: crate::core::MessageMeta {
                            session_id: "stream-session".to_string(),
                            timestamp: Some(std::time::SystemTime::now()),
                            cost_usd: None,
                            duration_ms: None,
                            tokens_used: None,
                        },
                        stats: ConversationStats {
                            total_messages: 1,
                            total_cost_usd: 0.0,
                            total_duration_ms: 0,
                            total_tokens: TokenUsage {
                                input: 0,
                                output: 0,
                                total: 0,
                            },
                        },
                    };
                    let _ = tx.send(Ok(final_message)).await;
                }
                StreamFormat::StreamJson => {
                    // StreamJson messages are sent as they arrive, no final processing needed
                }
            }
        });

        Self { receiver: rx }
    }

    /// Collects all messages from the stream and returns the full response as a single string.
    pub async fn collect_full_response(mut self) -> Result<String> {
        let config = crate::runtime::stream_config::get_stream_config();
        let mut response = String::with_capacity(config.string_capacity);

        while let Some(result) = self.next().await {
            match result? {
                Message::Assistant { content, .. } => {
                    response.push_str(&content);
                }
                Message::Result { .. } => {
                    // End of conversation
                    break;
                }
                _ => {}
            }
        }

        Ok(response)
    }
}

impl Stream for MessageStream {
    type Item = Result<Message>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

/// Parses streaming messages from Claude based on the configured format.
pub struct MessageParser {
    format: StreamFormat,
}

impl MessageParser {
    /// Creates a new message parser for the specified format.
    pub fn new(format: StreamFormat) -> Self {
        Self { format }
    }

    /// Parses a single line of output into a Message, returning None if the line should be skipped.
    pub fn parse_line(&self, line: &str) -> Result<Option<Message>> {
        match self.format {
            StreamFormat::Text => {
                // Text format doesn't have structured messages
                Ok(None)
            }
            StreamFormat::Json | StreamFormat::StreamJson => {
                if line.trim().is_empty() {
                    return Ok(None);
                }

                match serde_json::from_str::<Message>(line) {
                    Ok(message) => Ok(Some(message)),
                    Err(e) => {
                        error!("Failed to parse message: {}, line: {}", e, line);
                        Err(Error::SerializationError(e))
                    }
                }
            }
        }
    }

    /// Parse accumulated JSON content (for Json format)
    pub fn parse_accumulated_json(&self, content: &str) -> Result<Option<Message>> {
        if content.trim().is_empty() {
            return Ok(None);
        }

        // Try to parse as a direct message first
        if let Ok(message) = serde_json::from_str::<Message>(content) {
            return Ok(Some(message));
        }

        // If that fails, try to parse as a Claude CLI response and extract the result
        if let Ok(cli_response) = serde_json::from_str::<crate::core::ClaudeCliResponse>(content) {
            let message = Message::Assistant {
                content: cli_response.result,
                meta: crate::core::MessageMeta {
                    session_id: "json-response".to_string(),
                    timestamp: Some(std::time::SystemTime::now()),
                    cost_usd: None,
                    duration_ms: None,
                    tokens_used: None,
                },
            };
            return Ok(Some(message));
        }

        // If both fail, create a text message from the raw content
        let message = self.parse_text_response(content);
        Ok(Some(message))
    }

    /// Parses plain text into a Message structure for non-JSON responses.
    pub fn parse_text_response(&self, text: &str) -> Message {
        // For text format, create a simple assistant message
        Message::Assistant {
            content: text.to_string(),
            meta: crate::core::MessageMeta {
                session_id: "text-response".to_string(),
                timestamp: Some(std::time::SystemTime::now()),
                cost_usd: None,
                duration_ms: None,
                tokens_used: None,
            },
        }
    }
}
