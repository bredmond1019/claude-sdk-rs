use thiserror::Error;

/// Comprehensive error type for the claude-sdk-rs CLI
#[derive(Error, Debug)]
pub enum InteractiveError {
    #[error("Command discovery error: {0}")]
    CommandDiscovery(String),

    #[error("Command not found: {0}")]
    CommandNotFound(String),

    #[error("Session error: {0}")]
    Session(String),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Parallel execution error: {0}")]
    ParallelExecution(String),

    #[error("Cost tracking error: {0}")]
    CostTracking(String),

    #[error("History error: {0}")]
    History(String),

    #[error("Output formatting error: {0}")]
    OutputFormatting(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Timeout after {0} seconds")]
    Timeout(u64),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Claude SDK error: {0}")]
    ClaudeSDK(#[from] crate::core::error::Error),

    #[error("UUID error: {0}")]
    Uuid(#[from] uuid::Error),

    #[error("File watcher error: {0}")]
    FileWatcher(#[from] notify::Error),

    #[error("Async task error: {0}")]
    AsyncTask(#[from] tokio::task::JoinError),

    #[error("UTF-8 conversion error: {0}")]
    Utf8Conversion(#[from] std::string::FromUtf8Error),
}

impl InteractiveError {
    /// Create a command discovery error
    pub fn command_discovery<S: Into<String>>(msg: S) -> Self {
        Self::CommandDiscovery(msg.into())
    }

    /// Create a session error
    pub fn session<S: Into<String>>(msg: S) -> Self {
        Self::Session(msg.into())
    }

    /// Create an execution error
    pub fn execution<S: Into<String>>(msg: S) -> Self {
        Self::Execution(msg.into())
    }

    /// Create a cost tracking error
    pub fn cost_tracking<S: Into<String>>(msg: S) -> Self {
        Self::CostTracking(msg.into())
    }

    /// Create a history error
    pub fn history<S: Into<String>>(msg: S) -> Self {
        Self::History(msg.into())
    }

    /// Create an invalid input error
    pub fn invalid_input<S: Into<String>>(msg: S) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a session not found error
    pub fn session_not_found<S: Into<String>>(id: S) -> Self {
        Self::SessionNotFound(id.into())
    }

    /// Check if this error should be retried
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout(_) | Self::Io(_) | Self::ClaudeSDK(_) | Self::AsyncTask(_)
        )
    }

    /// Get user-friendly error message with suggestions
    pub fn user_message(&self) -> String {
        match self {
            Self::CommandNotFound(cmd) => {
                format!("Command '{}' not found. Run 'claude-interactive list' to see available commands.", cmd)
            }
            Self::SessionNotFound(id) => {
                format!("Session '{}' not found. Run 'claude-interactive session list' to see active sessions.", id)
            }
            Self::PermissionDenied(msg) => {
                format!("Permission denied: {}. Check your Claude CLI authentication with 'claude auth'.", msg)
            }
            Self::Timeout(secs) => {
                format!("Operation timed out after {} seconds. Try using --timeout flag with a larger value.", secs)
            }
            Self::ClaudeSDK(err) => {
                format!(
                    "Claude API error: {}. Check your network connection and API key.",
                    err
                )
            }
            _ => self.to_string(),
        }
    }
}

/// Convenient result type for the interactive CLI
pub type Result<T> = std::result::Result<T, InteractiveError>;

/// Trait for converting errors to user-friendly messages
pub trait UserFriendlyError {
    fn user_message(&self) -> String;
}

impl UserFriendlyError for InteractiveError {
    fn user_message(&self) -> String {
        self.user_message()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_error_creation_helpers() {
        let cmd_discovery_err = InteractiveError::command_discovery("Failed to discover commands");
        match cmd_discovery_err {
            InteractiveError::CommandDiscovery(msg) => {
                assert_eq!(msg, "Failed to discover commands");
            }
            _ => panic!("Expected CommandDiscovery error"),
        }

        let session_err = InteractiveError::session("Session initialization failed");
        match session_err {
            InteractiveError::Session(msg) => {
                assert_eq!(msg, "Session initialization failed");
            }
            _ => panic!("Expected Session error"),
        }

        let execution_err = InteractiveError::execution("Command execution failed");
        match execution_err {
            InteractiveError::Execution(msg) => {
                assert_eq!(msg, "Command execution failed");
            }
            _ => panic!("Expected Execution error"),
        }

        let cost_err = InteractiveError::cost_tracking("Failed to track costs");
        match cost_err {
            InteractiveError::CostTracking(msg) => {
                assert_eq!(msg, "Failed to track costs");
            }
            _ => panic!("Expected CostTracking error"),
        }

        let history_err = InteractiveError::history("History operation failed");
        match history_err {
            InteractiveError::History(msg) => {
                assert_eq!(msg, "History operation failed");
            }
            _ => panic!("Expected History error"),
        }

        let invalid_input_err = InteractiveError::invalid_input("Invalid command arguments");
        match invalid_input_err {
            InteractiveError::InvalidInput(msg) => {
                assert_eq!(msg, "Invalid command arguments");
            }
            _ => panic!("Expected InvalidInput error"),
        }

        let session_not_found_err = InteractiveError::session_not_found("session-123");
        match session_not_found_err {
            InteractiveError::SessionNotFound(id) => {
                assert_eq!(id, "session-123");
            }
            _ => panic!("Expected SessionNotFound error"),
        }
    }

    #[test]
    fn test_error_retry_logic() {
        // Retryable errors
        let timeout_err = InteractiveError::Timeout(30);
        assert!(timeout_err.is_retryable());

        let io_err =
            InteractiveError::Io(io::Error::new(io::ErrorKind::TimedOut, "Network timeout"));
        assert!(io_err.is_retryable());

        // Non-retryable errors
        let invalid_input_err = InteractiveError::InvalidInput("Bad arguments".to_string());
        assert!(!invalid_input_err.is_retryable());

        let permission_err = InteractiveError::PermissionDenied("Access denied".to_string());
        assert!(!permission_err.is_retryable());

        let command_not_found_err = InteractiveError::CommandNotFound("unknown-cmd".to_string());
        assert!(!command_not_found_err.is_retryable());

        let session_not_found_err =
            InteractiveError::SessionNotFound("missing-session".to_string());
        assert!(!session_not_found_err.is_retryable());
    }

    #[test]
    fn test_user_friendly_error_messages() {
        let command_not_found = InteractiveError::CommandNotFound("analyze".to_string());
        let message = command_not_found.user_message();
        assert!(message.contains("analyze"));
        assert!(message.contains("not found"));
        assert!(message.contains("claude-interactive list"));

        let session_not_found = InteractiveError::SessionNotFound("test-session".to_string());
        let message = session_not_found.user_message();
        assert!(message.contains("test-session"));
        assert!(message.contains("not found"));
        assert!(message.contains("claude-interactive session list"));

        let permission_denied = InteractiveError::PermissionDenied("API access denied".to_string());
        let message = permission_denied.user_message();
        assert!(message.contains("Permission denied"));
        assert!(message.contains("API access denied"));
        assert!(message.contains("claude auth"));

        let timeout = InteractiveError::Timeout(60);
        let message = timeout.user_message();
        assert!(message.contains("60 seconds"));
        assert!(message.contains("timeout"));
        assert!(message.contains("--timeout"));

        // Generic error should fall back to Display
        let generic_err = InteractiveError::Configuration("Invalid config".to_string());
        let message = generic_err.user_message();
        assert_eq!(message, "Configuration error: Invalid config");
    }

    #[test]
    fn test_error_type_conversions() {
        // Test From implementations
        let io_error = io::Error::new(io::ErrorKind::NotFound, "File not found");
        let interactive_error: InteractiveError = io_error.into();
        match interactive_error {
            InteractiveError::Io(_) => {} // Success
            _ => panic!("Expected Io error"),
        }

        // Test serialization error by creating an actual JSON error
        let json_result: std::result::Result<i32, serde_json::Error> =
            serde_json::from_str("invalid json");
        let json_error = json_result.unwrap_err();
        let interactive_error: InteractiveError = json_error.into();
        match interactive_error {
            InteractiveError::Serialization(_) => {} // Success
            _ => panic!("Expected Serialization error"),
        }

        // Test UUID error by parsing invalid UUID
        let uuid_result = uuid::Uuid::parse_str("invalid-uuid");
        let uuid_error = uuid_result.unwrap_err();
        let interactive_error: InteractiveError = uuid_error.into();
        match interactive_error {
            InteractiveError::Uuid(_) => {} // Success
            _ => panic!("Expected Uuid error"),
        }

        let utf8_error = String::from_utf8(vec![0xFF, 0xFE]).unwrap_err();
        let interactive_error: InteractiveError = utf8_error.into();
        match interactive_error {
            InteractiveError::Utf8Conversion(_) => {} // Success
            _ => panic!("Expected Utf8Conversion error"),
        }
    }

    #[test]
    fn test_error_display_messages() {
        let errors = vec![
            (
                InteractiveError::CommandDiscovery("Discovery failed".to_string()),
                "Command discovery error: Discovery failed",
            ),
            (
                InteractiveError::CommandNotFound("unknown".to_string()),
                "Command not found: unknown",
            ),
            (
                InteractiveError::Session("Session error".to_string()),
                "Session error: Session error",
            ),
            (
                InteractiveError::SessionNotFound("missing".to_string()),
                "Session not found: missing",
            ),
            (
                InteractiveError::Execution("Exec failed".to_string()),
                "Execution error: Exec failed",
            ),
            (
                InteractiveError::ParallelExecution("Parallel failed".to_string()),
                "Parallel execution error: Parallel failed",
            ),
            (
                InteractiveError::CostTracking("Cost error".to_string()),
                "Cost tracking error: Cost error",
            ),
            (
                InteractiveError::History("History error".to_string()),
                "History error: History error",
            ),
            (
                InteractiveError::OutputFormatting("Format error".to_string()),
                "Output formatting error: Format error",
            ),
            (
                InteractiveError::Configuration("Config error".to_string()),
                "Configuration error: Config error",
            ),
            (
                InteractiveError::PermissionDenied("Access denied".to_string()),
                "Permission denied: Access denied",
            ),
            (
                InteractiveError::InvalidInput("Bad input".to_string()),
                "Invalid input: Bad input",
            ),
            (InteractiveError::Timeout(30), "Timeout after 30 seconds"),
        ];

        for (error, expected_message) in errors {
            assert_eq!(error.to_string(), expected_message);
        }
    }

    #[test]
    fn test_user_friendly_error_trait() {
        let error = InteractiveError::CommandNotFound("test-cmd".to_string());
        let user_message = error.user_message();

        // Test the trait implementation
        let user_friendly: &dyn UserFriendlyError = &error;
        let trait_message = user_friendly.user_message();

        assert_eq!(user_message, trait_message);
        assert!(trait_message.contains("test-cmd"));
        assert!(trait_message.contains("not found"));
    }

    #[test]
    fn test_result_type_alias() {
        // Test that our Result type alias works correctly
        fn test_function() -> Result<String> {
            Ok("success".to_string())
        }

        fn test_function_error() -> Result<String> {
            Err(InteractiveError::InvalidInput("test error".to_string()))
        }

        let success_result = test_function();
        assert!(success_result.is_ok());
        assert_eq!(success_result.unwrap(), "success");

        let error_result = test_function_error();
        assert!(error_result.is_err());
        match error_result.unwrap_err() {
            InteractiveError::InvalidInput(msg) => {
                assert_eq!(msg, "test error");
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_error_categorization() {
        // Test various error categories
        let command_errors = vec![
            InteractiveError::CommandDiscovery("".to_string()),
            InteractiveError::CommandNotFound("".to_string()),
            InteractiveError::Execution("".to_string()),
            InteractiveError::ParallelExecution("".to_string()),
        ];

        let session_errors = vec![
            InteractiveError::Session("".to_string()),
            InteractiveError::SessionNotFound("".to_string()),
        ];

        let system_errors = vec![
            InteractiveError::Io(io::Error::new(io::ErrorKind::Other, "")),
            InteractiveError::Timeout(0),
        ];

        let input_errors = vec![
            InteractiveError::InvalidInput("".to_string()),
            InteractiveError::Configuration("".to_string()),
            InteractiveError::Serialization(serde_json::from_str::<i32>("invalid").unwrap_err()),
        ];

        // All command errors should not be retryable (except timeout scenarios)
        for error in command_errors {
            assert!(!error.is_retryable());
        }

        // Session errors should not be retryable
        for error in session_errors {
            assert!(!error.is_retryable());
        }

        // Most system errors should be retryable
        assert!(system_errors[0].is_retryable()); // IO error
        assert!(system_errors[1].is_retryable()); // Timeout

        // Input errors should not be retryable
        for error in input_errors {
            assert!(!error.is_retryable());
        }
    }

    #[tokio::test]
    async fn test_async_task_error_retry() {
        // Test AsyncTask error conversion and retry logic
        let handle = tokio::spawn(async {
            panic!("Test panic");
        });

        let join_result = handle.await;
        assert!(join_result.is_err());

        let join_error = join_result.unwrap_err();
        let interactive_error: InteractiveError = join_error.into();

        match interactive_error {
            InteractiveError::AsyncTask(_) => {
                assert!(interactive_error.is_retryable());
            }
            _ => panic!("Expected AsyncTask error"),
        }
    }

    #[test]
    fn test_complex_error_scenarios() {
        // Test chaining and complex error scenarios
        fn simulate_command_execution() -> Result<String> {
            // Simulate a command not found scenario
            Err(InteractiveError::command_discovery(
                "Commands directory not accessible",
            ))
        }

        fn simulate_session_workflow() -> Result<()> {
            let session_result = simulate_command_execution();
            match session_result {
                Ok(_) => Ok(()),
                Err(e) => {
                    // Transform discovery error into execution error
                    Err(InteractiveError::execution(format!(
                        "Failed to execute due to: {}",
                        e
                    )))
                }
            }
        }

        let workflow_result = simulate_session_workflow();
        assert!(workflow_result.is_err());

        let error = workflow_result.unwrap_err();
        match error {
            InteractiveError::Execution(msg) => {
                assert!(msg.contains("Failed to execute"));
                assert!(msg.contains("Commands directory not accessible"));
            }
            _ => panic!("Expected Execution error"),
        }

        // Test retry logic in a workflow
        fn should_retry_operation(error: &InteractiveError) -> bool {
            error.is_retryable()
        }

        let retryable_error = InteractiveError::Timeout(30);
        let non_retryable_error = InteractiveError::InvalidInput("Bad args".to_string());

        assert!(should_retry_operation(&retryable_error));
        assert!(!should_retry_operation(&non_retryable_error));
    }

    #[test]
    fn test_error_debug_formatting() {
        let error = InteractiveError::CommandNotFound("test-command".to_string());
        let debug_output = format!("{:?}", error);

        assert!(debug_output.contains("CommandNotFound"));
        assert!(debug_output.contains("test-command"));
    }

    #[test]
    fn test_timeout_error_specifics() {
        let timeout_30s = InteractiveError::Timeout(30);
        let timeout_60s = InteractiveError::Timeout(60);

        assert!(timeout_30s.is_retryable());
        assert!(timeout_60s.is_retryable());

        let msg_30s = timeout_30s.user_message();
        let msg_60s = timeout_60s.user_message();

        assert!(msg_30s.contains("30 seconds"));
        assert!(msg_60s.contains("60 seconds"));
        assert!(msg_30s.contains("--timeout"));
        assert!(msg_60s.contains("--timeout"));
    }
}
