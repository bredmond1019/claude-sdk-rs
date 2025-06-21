use std::fmt;
use thiserror::Error;

/// Error codes for Claude AI SDK operations
///
/// Each error type has a unique code that can be used for programmatic error handling
/// and troubleshooting. Error codes follow the pattern: `CXXX` where X is a digit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// `C001`: Claude binary not found
    BinaryNotFound = 1,
    /// `C002`: Session not found
    SessionNotFound = 2,
    /// `C003`: Permission denied
    PermissionDenied = 3,
    /// `C004`: MCP server error
    McpError = 4,
    /// `C005`: Configuration error
    ConfigError = 5,
    /// `C006`: Invalid input
    InvalidInput = 6,
    /// `C007`: Operation timeout
    Timeout = 7,
    /// `C008`: Serialization error
    SerializationError = 8,
    /// `C009`: I/O error
    IoError = 9,
    /// `C010`: Process execution error
    ProcessError = 10,
    /// `C011`: Stream closed
    StreamClosed = 11,
    /// `C012`: Not authenticated
    NotAuthenticated = 12,
    /// `C013`: Rate limit exceeded
    RateLimitExceeded = 13,
    /// `C014`: UTF-8 conversion error
    Utf8Error = 14,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C{:03}", *self as u16)
    }
}

/// Error types for Claude AI SDK operations
///
/// This enum covers all possible error conditions that can occur when using
/// the Claude AI SDK, from configuration issues to runtime execution problems.
/// Each error has an associated error code for easy reference.
///
/// # Examples
///
/// ```rust
/// use claude_sdk_rs_core::{Error, Result};
///
/// fn handle_error(result: Result<String>) {
///     match result {
///         Ok(response) => println!("Success: {}", response),
///         Err(Error::BinaryNotFound) => {
///             eprintln!("Error C001: Please install the Claude Code CLI first");
///         }
///         Err(Error::Timeout(seconds)) => {
///             eprintln!("Error C007: Request timed out after {} seconds", seconds);
///         }
///         Err(e) => eprintln!("Error {}: {}", e.code(), e),
///     }
/// }
/// ```
#[derive(Error, Debug)]
pub enum Error {
    /// Claude Code CLI binary not found in PATH `[C001]`
    ///
    /// This error occurs when the Claude Code CLI tool is not installed
    /// or not available in the system PATH. Install the CLI first.
    #[error("[{code}] Claude Code not found in PATH", code = ErrorCode::BinaryNotFound)]
    BinaryNotFound,

    /// Session with the given ID was not found `[C002]`
    ///
    /// Occurs when trying to access a session that doesn't exist or
    /// has been deleted.
    #[error("[{code}] Session {0} not found", code = ErrorCode::SessionNotFound)]
    SessionNotFound(String),

    /// Permission denied for the specified tool `[C003]`
    ///
    /// This error occurs when trying to use a tool that's restricted
    /// by the current configuration's `allowed_tools` setting.
    #[error("[{code}] Tool permission denied: {0}", code = ErrorCode::PermissionDenied)]
    PermissionDenied(String),

    /// Error from an MCP (Model Context Protocol) server `[C004]`
    ///
    /// Indicates a problem with an external MCP server that Claude
    /// was trying to communicate with.
    #[error("[{code}] MCP server error: {0}", code = ErrorCode::McpError)]
    McpError(String),

    /// Invalid configuration provided `[C005]`
    ///
    /// Occurs when the provided configuration has invalid or conflicting
    /// settings that prevent proper operation.
    #[error("[{code}] Invalid configuration: {0}", code = ErrorCode::ConfigError)]
    ConfigError(String),

    /// Invalid input provided to a function `[C006]`
    ///
    /// Indicates that the input parameters don't meet the expected
    /// format or constraints.
    #[error("[{code}] Invalid input: {0}", code = ErrorCode::InvalidInput)]
    InvalidInput(String),

    /// Operation timed out `[C007]`
    ///
    /// The operation took longer than the configured timeout period.
    /// Consider increasing the timeout for complex queries.
    #[error("[{code}] Operation timed out after {0}s", code = ErrorCode::Timeout)]
    Timeout(u64),

    /// JSON serialization or deserialization error `[C008]`
    ///
    /// Occurs when parsing JSON responses from Claude CLI or when
    /// serializing configuration to JSON format.
    #[error("[{code}] Serialization error: {0}", code = ErrorCode::SerializationError)]
    SerializationError(#[from] serde_json::Error),

    /// Input/output operation failed `[C009]`
    ///
    /// Covers file system operations, network operations, and other
    /// I/O related failures.
    #[error("[{code}] IO error: {0}", code = ErrorCode::IoError)]
    Io(#[from] std::io::Error),

    /// Process execution error `[C010]`
    ///
    /// Occurs when the Claude CLI process fails to execute properly
    /// or returns an unexpected exit code.
    #[error("[{code}] Process error: {0}", code = ErrorCode::ProcessError)]
    ProcessError(String),

    /// Stream was closed unexpectedly `[C011]`
    ///
    /// Happens during streaming operations when the connection to
    /// Claude CLI is interrupted.
    #[error("[{code}] Stream closed unexpectedly", code = ErrorCode::StreamClosed)]
    StreamClosed,

    /// Claude CLI is not authenticated `[C012]`
    ///
    /// The Claude CLI tool needs to be authenticated before use.
    /// Run `claude auth` to authenticate.
    #[error("[{code}] Claude CLI is not authenticated. Run 'claude auth' to authenticate.", code = ErrorCode::NotAuthenticated)]
    NotAuthenticated,

    /// Rate limit exceeded `[C013]`
    ///
    /// Too many requests have been made in a short period.
    /// Wait before retrying.
    #[error("[{code}] Rate limit exceeded. Please wait before retrying.", code = ErrorCode::RateLimitExceeded)]
    RateLimitExceeded,

    /// UTF-8 conversion error `[C014]`
    ///
    /// Occurs when trying to convert bytes to UTF-8 string but the data
    /// contains invalid UTF-8 sequences.
    #[error("[{code}] UTF-8 conversion error: {0}", code = ErrorCode::Utf8Error)]
    Utf8Error(#[from] std::string::FromUtf8Error),
}

impl Clone for Error {
    fn clone(&self) -> Self {
        match self {
            Error::BinaryNotFound => Error::BinaryNotFound,
            Error::SessionNotFound(s) => Error::SessionNotFound(s.clone()),
            Error::PermissionDenied(s) => Error::PermissionDenied(s.clone()),
            Error::McpError(s) => Error::McpError(s.clone()),
            Error::ConfigError(s) => Error::ConfigError(s.clone()),
            Error::InvalidInput(s) => Error::InvalidInput(s.clone()),
            Error::Timeout(secs) => Error::Timeout(*secs),
            Error::SerializationError(e) => Error::SerializationError(
                serde_json::from_str::<serde_json::Value>(&format!("\"{}\"", e.to_string()))
                    .unwrap_err(),
            ),
            Error::Io(e) => Error::Io(std::io::Error::new(e.kind(), e.to_string())),
            Error::ProcessError(s) => Error::ProcessError(s.clone()),
            Error::StreamClosed => Error::StreamClosed,
            Error::NotAuthenticated => Error::NotAuthenticated,
            Error::RateLimitExceeded => Error::RateLimitExceeded,
            Error::Utf8Error(e) => {
                Error::Utf8Error(std::string::String::from_utf8(e.as_bytes().to_vec()).unwrap_err())
            }
        }
    }
}

impl Error {
    /// Get the error code for this error
    ///
    /// Error codes are useful for programmatic error handling and
    /// for looking up specific troubleshooting steps.
    ///
    /// # Example
    ///
    /// ```rust
    /// use claude_sdk_rs_core::Error;
    ///
    /// let error = Error::BinaryNotFound;
    /// assert_eq!(error.code().to_string(), "C001");
    /// ```
    pub fn code(&self) -> ErrorCode {
        match self {
            Error::BinaryNotFound => ErrorCode::BinaryNotFound,
            Error::SessionNotFound(_) => ErrorCode::SessionNotFound,
            Error::PermissionDenied(_) => ErrorCode::PermissionDenied,
            Error::McpError(_) => ErrorCode::McpError,
            Error::ConfigError(_) => ErrorCode::ConfigError,
            Error::InvalidInput(_) => ErrorCode::InvalidInput,
            Error::Timeout(_) => ErrorCode::Timeout,
            Error::SerializationError(_) => ErrorCode::SerializationError,
            Error::Io(_) => ErrorCode::IoError,
            Error::ProcessError(_) => ErrorCode::ProcessError,
            Error::StreamClosed => ErrorCode::StreamClosed,
            Error::NotAuthenticated => ErrorCode::NotAuthenticated,
            Error::RateLimitExceeded => ErrorCode::RateLimitExceeded,
            Error::Utf8Error(_) => ErrorCode::Utf8Error,
        }
    }

    /// Check if this error is recoverable
    ///
    /// Some errors can be recovered from by retrying or changing
    /// configuration, while others indicate permanent failures.
    ///
    /// # Example
    ///
    /// ```rust
    /// use claude_sdk_rs_core::Error;
    ///
    /// let error = Error::Timeout(30);
    /// if error.is_recoverable() {
    ///     // Can retry with longer timeout
    /// }
    /// ```
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Error::Timeout(_)
                | Error::RateLimitExceeded
                | Error::StreamClosed
                | Error::Io(_)
                | Error::ProcessError(_)
        )
    }
}

// Conversion from CLI InteractiveError to core Error
#[cfg(feature = "cli")]
impl From<crate::cli::error::InteractiveError> for Error {
    fn from(err: crate::cli::error::InteractiveError) -> Self {
        match err {
            crate::cli::error::InteractiveError::CommandDiscovery(msg) => Error::ProcessError(msg),
            crate::cli::error::InteractiveError::CommandNotFound(msg) => {
                Error::ProcessError(format!("Command not found: {}", msg))
            }
            crate::cli::error::InteractiveError::Session(msg) => Error::SessionNotFound(msg),
            crate::cli::error::InteractiveError::SessionNotFound(id) => Error::SessionNotFound(id),
            crate::cli::error::InteractiveError::Execution(msg) => Error::ProcessError(msg),
            crate::cli::error::InteractiveError::ParallelExecution(msg) => {
                Error::ProcessError(format!("Parallel execution: {}", msg))
            }
            crate::cli::error::InteractiveError::CostTracking(msg) => {
                Error::ProcessError(format!("Cost tracking: {}", msg))
            }
            crate::cli::error::InteractiveError::History(msg) => {
                Error::ProcessError(format!("History: {}", msg))
            }
            crate::cli::error::InteractiveError::OutputFormatting(msg) => {
                Error::ProcessError(format!("Output formatting: {}", msg))
            }
            crate::cli::error::InteractiveError::Configuration(msg) => Error::ConfigError(msg),
            crate::cli::error::InteractiveError::PermissionDenied(msg) => {
                Error::PermissionDenied(msg)
            }
            crate::cli::error::InteractiveError::InvalidInput(msg) => Error::InvalidInput(msg),
            crate::cli::error::InteractiveError::Timeout(secs) => Error::Timeout(secs),
            crate::cli::error::InteractiveError::Io(err) => Error::Io(err),
            crate::cli::error::InteractiveError::Serialization(err) => {
                Error::SerializationError(err)
            }
            crate::cli::error::InteractiveError::ClaudeSDK(err) => err,
            crate::cli::error::InteractiveError::Uuid(_) => {
                Error::InvalidInput("UUID parsing error".to_string())
            }
            crate::cli::error::InteractiveError::FileWatcher(err) => Error::Io(
                std::io::Error::new(std::io::ErrorKind::Other, err.to_string()),
            ),
            crate::cli::error::InteractiveError::AsyncTask(err) => {
                Error::ProcessError(format!("Async task: {}", err))
            }
            crate::cli::error::InteractiveError::Utf8Conversion(err) => Error::Utf8Error(err),
        }
    }
}

/// Type alias for Results with Claude AI SDK errors
///
/// This is a convenience type alias that uses [`enum@Error`] as the error type.
/// Most functions in the SDK return this Result type.
///
/// # Examples
///
/// ```rust
/// use claude_sdk_rs_core::Result;
///
/// fn example_function() -> Result<String> {
///     Ok("success".to_string())
/// }
/// ```
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        // Test that each error returns the correct code
        assert_eq!(Error::BinaryNotFound.code(), ErrorCode::BinaryNotFound);
        assert_eq!(
            Error::SessionNotFound("test".to_string()).code(),
            ErrorCode::SessionNotFound
        );
        assert_eq!(
            Error::PermissionDenied("tool".to_string()).code(),
            ErrorCode::PermissionDenied
        );
        assert_eq!(
            Error::McpError("error".to_string()).code(),
            ErrorCode::McpError
        );
        assert_eq!(
            Error::ConfigError("invalid".to_string()).code(),
            ErrorCode::ConfigError
        );
        assert_eq!(
            Error::InvalidInput("bad".to_string()).code(),
            ErrorCode::InvalidInput
        );
        assert_eq!(Error::Timeout(30).code(), ErrorCode::Timeout);
        assert_eq!(
            Error::ProcessError("failed".to_string()).code(),
            ErrorCode::ProcessError
        );
        assert_eq!(Error::StreamClosed.code(), ErrorCode::StreamClosed);
        assert_eq!(Error::NotAuthenticated.code(), ErrorCode::NotAuthenticated);
        assert_eq!(
            Error::RateLimitExceeded.code(),
            ErrorCode::RateLimitExceeded
        );
    }

    #[test]
    fn test_error_code_display() {
        assert_eq!(ErrorCode::BinaryNotFound.to_string(), "C001");
        assert_eq!(ErrorCode::SessionNotFound.to_string(), "C002");
        assert_eq!(ErrorCode::PermissionDenied.to_string(), "C003");
        assert_eq!(ErrorCode::McpError.to_string(), "C004");
        assert_eq!(ErrorCode::ConfigError.to_string(), "C005");
        assert_eq!(ErrorCode::InvalidInput.to_string(), "C006");
        assert_eq!(ErrorCode::Timeout.to_string(), "C007");
        assert_eq!(ErrorCode::SerializationError.to_string(), "C008");
        assert_eq!(ErrorCode::IoError.to_string(), "C009");
        assert_eq!(ErrorCode::ProcessError.to_string(), "C010");
        assert_eq!(ErrorCode::StreamClosed.to_string(), "C011");
        assert_eq!(ErrorCode::NotAuthenticated.to_string(), "C012");
        assert_eq!(ErrorCode::RateLimitExceeded.to_string(), "C013");
    }

    #[test]
    fn test_error_messages_include_codes() {
        let error = Error::BinaryNotFound;
        assert!(error.to_string().contains("[C001]"));

        let error = Error::Timeout(30);
        assert!(error.to_string().contains("[C007]"));
        assert!(error.to_string().contains("30s"));

        let error = Error::NotAuthenticated;
        assert!(error.to_string().contains("[C012]"));
        assert!(error.to_string().contains("claude auth"));
    }

    #[test]
    fn test_is_recoverable() {
        // Recoverable errors
        assert!(Error::Timeout(30).is_recoverable());
        assert!(Error::RateLimitExceeded.is_recoverable());
        assert!(Error::StreamClosed.is_recoverable());
        assert!(Error::ProcessError("temp failure".to_string()).is_recoverable());

        // Non-recoverable errors
        assert!(!Error::BinaryNotFound.is_recoverable());
        assert!(!Error::ConfigError("invalid".to_string()).is_recoverable());
        assert!(!Error::InvalidInput("bad".to_string()).is_recoverable());
        assert!(!Error::NotAuthenticated.is_recoverable());
        assert!(!Error::PermissionDenied("denied".to_string()).is_recoverable());
        assert!(
            !Error::Utf8Error(std::string::String::from_utf8(vec![0xFF]).unwrap_err())
                .is_recoverable()
        );
    }

    #[test]
    fn test_error_conversions() {
        // Test From implementations
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let error: Error = io_error.into();
        assert_eq!(error.code(), ErrorCode::IoError);

        let json_error = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let error: Error = json_error.into();
        assert_eq!(error.code(), ErrorCode::SerializationError);
    }

    #[test]
    fn test_error_code_ordering() {
        // Ensure error codes are sequential and unique
        let codes = vec![
            ErrorCode::BinaryNotFound as u16,
            ErrorCode::SessionNotFound as u16,
            ErrorCode::PermissionDenied as u16,
            ErrorCode::McpError as u16,
            ErrorCode::ConfigError as u16,
            ErrorCode::InvalidInput as u16,
            ErrorCode::Timeout as u16,
            ErrorCode::SerializationError as u16,
            ErrorCode::IoError as u16,
            ErrorCode::ProcessError as u16,
            ErrorCode::StreamClosed as u16,
            ErrorCode::NotAuthenticated as u16,
            ErrorCode::RateLimitExceeded as u16,
        ];

        // Check sequential ordering
        for i in 0..codes.len() {
            assert_eq!(codes[i], (i + 1) as u16);
        }
    }
}
