//! Core command execution engine
//!
//! This module provides the CommandRunner for executing Claude commands with session context,
//! supporting both streaming and non-streaming responses, and integrating with the session
//! management system.

use crate::cli::execution::{ExecutionContext, ExecutionResult};
use crate::cli::session::{SessionId, SessionManager};
use crate::{cli::error::InteractiveError, cli::error::Result};
use crate::{Client, Config, StreamFormat};
use futures::StreamExt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// Configuration for command execution
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Timeout for command execution
    pub timeout: Duration,
    /// Whether to enable streaming responses
    pub streaming: bool,
    /// Stream format to use
    pub stream_format: StreamFormat,
    /// Maximum retries for failed commands
    pub max_retries: usize,
    /// Delay between retries
    pub retry_delay: Duration,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            streaming: true,
            stream_format: StreamFormat::StreamJson,
            max_retries: 3,
            retry_delay: Duration::from_secs(1),
        }
    }
}

/// Core command execution engine that integrates with Claude AI SDK
pub struct CommandRunner {
    client: Client,
    session_manager: Arc<tokio::sync::RwLock<SessionManager>>,
    config: RunnerConfig,
}

impl CommandRunner {
    /// Create a new command runner with default configuration
    pub fn new(session_manager: Arc<tokio::sync::RwLock<SessionManager>>) -> Result<Self> {
        let config = Config::builder()
            .stream_format(StreamFormat::StreamJson)
            .timeout_secs(30)
            .build();

        let client = Client::new(config?);

        Ok(Self {
            client,
            session_manager,
            config: RunnerConfig::default(),
        })
    }

    /// Create a new command runner with custom configuration
    pub fn with_config(
        session_manager: Arc<tokio::sync::RwLock<SessionManager>>,
        config: RunnerConfig,
    ) -> Result<Self> {
        let claude_config = Config::builder()
            .stream_format(config.stream_format.clone())
            .timeout_secs(config.timeout.as_secs())
            .build();

        let client = Client::new(claude_config?);

        Ok(Self {
            client,
            session_manager,
            config,
        })
    }

    /// Execute a command with the given context
    pub async fn execute(&self, context: ExecutionContext) -> Result<ExecutionResult> {
        let start_time = Instant::now();

        debug!(
            command = %context.command_name,
            args = ?context.args,
            session_id = ?context.session_id,
            "Starting command execution"
        );

        // Prepare the command message
        let command_text = self.build_command_text(&context)?;

        // Execute with retries
        let mut last_error = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                warn!(
                    "Retrying command execution (attempt {}/{})",
                    attempt + 1,
                    self.config.max_retries + 1
                );
                tokio::time::sleep(self.config.retry_delay).await;
            }

            match self.execute_with_timeout(&command_text, &context).await {
                Ok(result) => {
                    let duration = start_time.elapsed();

                    // Update session metadata if session is provided
                    if let Some(session_id) = context.session_id {
                        if let Err(e) = self
                            .update_session_after_execution(session_id, &result)
                            .await
                        {
                            warn!("Failed to update session metadata: {}", e);
                        }
                    }

                    info!(
                        command = %context.command_name,
                        duration_ms = duration.as_millis(),
                        success = result.success,
                        cost = result.cost,
                        "Command execution completed"
                    );

                    return Ok(ExecutionResult {
                        output: result.output,
                        cost: result.cost,
                        duration,
                        success: result.success,
                    });
                }
                Err(e) => {
                    last_error = Some(e);
                    if !self.should_retry(&last_error.as_ref().unwrap()) {
                        break;
                    }
                }
            }
        }

        let duration = start_time.elapsed();
        let error = last_error.unwrap();

        error!(
            command = %context.command_name,
            duration_ms = duration.as_millis(),
            error = %error,
            "Command execution failed after retries"
        );

        Err(error)
    }

    /// Execute a command and handle streaming responses
    pub async fn execute_streaming<F>(
        &self,
        context: ExecutionContext,
        mut output_handler: F,
    ) -> Result<ExecutionResult>
    where
        F: FnMut(String) -> Result<()>,
    {
        let start_time = Instant::now();

        debug!(
            command = %context.command_name,
            args = ?context.args,
            session_id = ?context.session_id,
            "Starting streaming command execution"
        );

        let command_text = self.build_command_text(&context)?;

        // Use streaming query builder
        let mut stream = timeout(
            self.config.timeout,
            self.client.query(&command_text).stream(),
        )
        .await
        .map_err(|_| InteractiveError::Timeout(self.config.timeout.as_secs()))?
        .map_err(|e| InteractiveError::execution(format!("Failed to start streaming: {}", e)))?;

        let mut full_output = String::new();
        let mut total_cost = 0.0;

        // Process streaming responses
        while let Some(response) = stream.next().await {
            match response {
                Ok(msg) => {
                    let content = msg.content();
                    full_output.push_str(&content);

                    // Call the output handler
                    output_handler(content)?;

                    // Accumulate cost if available
                    let meta = msg.meta();
                    if let Some(cost) = meta.cost_usd {
                        total_cost += cost;
                    }
                }
                Err(e) => {
                    error!("Stream error: {}", e);
                    return Err(InteractiveError::execution(format!(
                        "Streaming error: {}",
                        e
                    )));
                }
            }
        }

        let duration = start_time.elapsed();
        let result = ExecutionResult {
            output: full_output,
            cost: if total_cost > 0.0 {
                Some(total_cost)
            } else {
                None
            },
            duration,
            success: true,
        };

        // Update session metadata if session is provided
        if let Some(session_id) = context.session_id {
            if let Err(e) = self
                .update_session_after_execution(session_id, &result)
                .await
            {
                warn!("Failed to update session metadata: {}", e);
            }
        }

        info!(
            command = %context.command_name,
            duration_ms = duration.as_millis(),
            cost = result.cost,
            "Streaming command execution completed"
        );

        Ok(result)
    }

    /// Build the command text from context
    fn build_command_text(&self, context: &ExecutionContext) -> Result<String> {
        if context.args.is_empty() {
            Ok(context.command_name.clone())
        } else {
            Ok(format!(
                "{} {}",
                context.command_name,
                context.args.join(" ")
            ))
        }
    }

    /// Execute command with timeout
    async fn execute_with_timeout(
        &self,
        command_text: &str,
        _context: &ExecutionContext,
    ) -> Result<ExecutionResult> {
        let result = if self.config.streaming {
            timeout(
                self.config.timeout,
                self.execute_streaming_internal(command_text),
            )
            .await
        } else {
            timeout(
                self.config.timeout,
                self.execute_simple_internal(command_text),
            )
            .await
        };

        result.map_err(|_| InteractiveError::Timeout(self.config.timeout.as_secs()))?
    }

    /// Internal streaming execution
    async fn execute_streaming_internal(&self, command_text: &str) -> Result<ExecutionResult> {
        let mut stream = self
            .client
            .query(command_text)
            .stream()
            .await
            .map_err(|e| {
                InteractiveError::execution(format!("Failed to start streaming: {}", e))
            })?;

        let mut full_output = String::new();
        let mut total_cost = 0.0;

        while let Some(response) = stream.next().await {
            match response {
                Ok(msg) => {
                    full_output.push_str(&msg.content());

                    // Accumulate cost if available
                    let meta = msg.meta();
                    if let Some(cost) = meta.cost_usd {
                        total_cost += cost;
                    }
                }
                Err(e) => {
                    return Err(InteractiveError::execution(format!(
                        "Streaming error: {}",
                        e
                    )));
                }
            }
        }

        Ok(ExecutionResult {
            output: full_output,
            cost: if total_cost > 0.0 {
                Some(total_cost)
            } else {
                None
            },
            duration: Duration::default(), // Will be set by caller
            success: true,
        })
    }

    /// Internal simple execution
    async fn execute_simple_internal(&self, command_text: &str) -> Result<ExecutionResult> {
        let response =
            self.client.send(command_text).await.map_err(|e| {
                InteractiveError::execution(format!("Failed to send command: {}", e))
            })?;

        Ok(ExecutionResult {
            output: response,
            cost: None,                    // Simple responses don't include usage data
            duration: Duration::default(), // Will be set by caller
            success: true,
        })
    }

    /// Update session metadata after successful execution
    async fn update_session_after_execution(
        &self,
        session_id: SessionId,
        result: &ExecutionResult,
    ) -> Result<()> {
        // For now, we'll just log that we would update the session
        // The actual implementation would need the session manager to have this method
        debug!(
            "Would update session {} with command result: cost={:?}, duration={:?}",
            session_id, result.cost, result.duration
        );
        Ok(())
    }

    /// Check if an error should trigger a retry
    fn should_retry(&self, error: &InteractiveError) -> bool {
        error.is_retryable()
    }

    /// Get current configuration
    pub fn config(&self) -> &RunnerConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: RunnerConfig) -> Result<()> {
        // Rebuild Claude client with new config
        let claude_config = Config::builder()
            .stream_format(config.stream_format.clone())
            .timeout_secs(config.timeout.as_secs())
            .build();

        self.client = Client::new(claude_config?);

        self.config = config;
        Ok(())
    }
}

impl Default for CommandRunner {
    fn default() -> Self {
        // Create a default session manager for default case
        // In practice, this should be constructed with a proper session manager
        let session_manager = Arc::new(tokio::sync::RwLock::new(
            SessionManager::with_default_storage()
                .expect("Failed to create default session manager"),
        ));

        Self::new(session_manager).expect("Failed to create default command runner")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::session::storage::{JsonFileStorage, StorageConfig};
    use tempfile::tempdir;
    use tokio::time::Duration;

    async fn create_test_runner() -> Result<CommandRunner> {
        let temp_dir = tempdir().unwrap();
        let config = StorageConfig {
            data_dir: temp_dir.path().to_path_buf(),
            sessions_file: "test_sessions.json".to_string(),
            current_session_file: "test_current.json".to_string(),
        };

        let storage = Arc::new(JsonFileStorage::new(config));
        let session_manager = Arc::new(tokio::sync::RwLock::new(SessionManager::new(storage)));

        CommandRunner::new(session_manager)
    }

    #[tokio::test]
    async fn test_runner_creation() -> Result<()> {
        let runner = create_test_runner().await?;
        assert_eq!(runner.config.timeout, Duration::from_secs(30));
        assert!(runner.config.streaming);
        Ok(())
    }

    #[tokio::test]
    async fn test_command_text_building() -> Result<()> {
        let runner = create_test_runner().await?;

        let context = ExecutionContext {
            session_id: None,
            command_name: "test".to_string(),
            args: vec!["arg1".to_string(), "arg2".to_string()],
            parallel: false,
            agent_count: 1,
        };

        let command_text = runner.build_command_text(&context)?;
        assert_eq!(command_text, "test arg1 arg2");

        Ok(())
    }

    #[tokio::test]
    async fn test_config_update() -> Result<()> {
        let mut runner = create_test_runner().await?;

        let new_config = RunnerConfig {
            timeout: Duration::from_secs(60),
            streaming: false,
            stream_format: StreamFormat::Json,
            max_retries: 5,
            retry_delay: Duration::from_secs(2),
        };

        runner.set_config(new_config.clone())?;
        assert_eq!(runner.config.timeout, Duration::from_secs(60));
        assert!(!runner.config.streaming);
        assert_eq!(runner.config.max_retries, 5);

        Ok(())
    }
}
