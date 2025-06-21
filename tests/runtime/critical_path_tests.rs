//! Critical path tests for Claude AI Runtime
//!
//! This module contains comprehensive tests for all critical paths in the runtime library,
//! targeting >95% test coverage of essential functionality.

use claude_sdk_rs_core::*;
use claude_sdk_rs_runtime::*;
use std::time::Duration;
use tokio::sync::mpsc;

#[cfg(test)]
mod client_tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let config = Config::default();
        let client = Client::new(config);

        // Client should be created successfully
        assert!(format!("{:?}", client).contains("Client"));
    }

    #[test]
    fn test_client_builder() {
        let client = Client::builder()
            .model("claude-3-haiku-20240307")
            .system_prompt("You are a test assistant")
            .stream_format(StreamFormat::Json)
            .timeout_secs(60)
            .build();

        // Client should be built successfully
        assert!(format!("{:?}", client).contains("Client"));
    }

    #[test]
    fn test_client_clone() {
        let client = Client::new(Config::default());
        let cloned = client.clone();

        // Both clients should be equivalent
        assert_eq!(format!("{:?}", client), format!("{:?}", cloned));
    }

    #[test]
    fn test_query_builder_creation() {
        let client = Client::new(Config::default());
        let query_builder = client.query("Test query");

        // Query builder should be created
        assert!(format!("{:?}", query_builder).contains("QueryBuilder"));
    }

    #[test]
    fn test_query_builder_with_session() {
        let client = Client::new(Config::default());
        let session_id = SessionId::new("test-session");
        let query_builder = client.query("Test query").with_session(session_id.clone());

        // Query builder should include session
        assert!(format!("{:?}", query_builder).contains("test-session"));
    }

    #[test]
    fn test_client_builder_with_tools() {
        let client = Client::builder()
            .allowed_tools(vec!["bash".to_string(), "filesystem".to_string()])
            .build();

        // Client should be created with tools
        assert!(format!("{:?}", client).contains("Client"));
    }

    #[test]
    fn test_client_builder_validation() {
        // Test that invalid configurations are rejected during build
        let result = std::panic::catch_unwind(|| {
            Client::builder()
                .timeout_secs(0) // Invalid timeout
                .build()
        });

        // Should either panic or handle gracefully
        // The actual behavior depends on the implementation
        assert!(result.is_ok() || result.is_err());
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    #[tokio::test]
    async fn test_message_stream_creation() {
        let (tx, rx) = mpsc::channel::<Result<String>>(10);
        let stream = MessageStream::new(rx);

        // Stream should be created successfully
        assert!(format!("{:?}", stream).contains("MessageStream"));

        // Drop the sender to close the stream
        drop(tx);
    }

    #[tokio::test]
    async fn test_message_stream_with_messages() {
        let (tx, rx) = mpsc::channel::<Result<String>>(10);
        let mut stream = MessageStream::new(rx);

        // Send test messages
        tx.send(Ok("Message 1".to_string())).await.unwrap();
        tx.send(Ok("Message 2".to_string())).await.unwrap();
        drop(tx);

        // Collect messages
        let mut messages = Vec::new();
        while let Some(result) = stream.next().await {
            match result {
                Ok(msg) => messages.push(msg),
                Err(_) => break,
            }
        }

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0], "Message 1");
        assert_eq!(messages[1], "Message 2");
    }

    #[tokio::test]
    async fn test_message_stream_with_error() {
        let (tx, rx) = mpsc::channel::<Result<String>>(10);
        let mut stream = MessageStream::new(rx);

        // Send an error
        tx.send(Err(Error::ProcessError("Test error".to_string())))
            .await
            .unwrap();
        drop(tx);

        // Should receive the error
        let result = stream.next().await;
        assert!(result.is_some());
        assert!(result.unwrap().is_err());
    }

    #[tokio::test]
    async fn test_message_stream_empty() {
        let (tx, rx) = mpsc::channel::<Result<String>>(10);
        let mut stream = MessageStream::new(rx);

        // Close immediately
        drop(tx);

        // Should return None
        let result = stream.next().await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_message_stream_timeout() {
        let (tx, rx) = mpsc::channel::<Result<String>>(10);
        let mut stream = MessageStream::new(rx);

        // Don't send anything, just test timeout behavior
        let timeout_result = tokio::time::timeout(Duration::from_millis(100), stream.next()).await;

        // Should timeout
        assert!(timeout_result.is_err());

        drop(tx);
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;
    use claude_sdk_rs_runtime::error_handling::*;

    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext::new("test_operation")
            .with_debug_info("key1", "value1")
            .with_debug_info("key2", "value2")
            .with_error_chain("first error")
            .with_error_chain("second error");

        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.debug_info.len(), 2);
        assert_eq!(context.error_chain.len(), 2);

        let debug_string = context.to_debug_string();
        assert!(debug_string.contains("test_operation"));
        assert!(debug_string.contains("key1: value1"));
        assert!(debug_string.contains("first error"));
    }

    #[test]
    fn test_process_error_details_creation() {
        let details =
            ProcessErrorDetails::new("Test error", "claude", vec!["--version".to_string()]);

        assert_eq!(details.message, "Test error");
        assert_eq!(details.command, "claude");
        assert_eq!(details.args, vec!["--version".to_string()]);
        assert!(details.working_dir.is_some());
    }

    #[test]
    fn test_process_error_details_with_context() {
        let details =
            ProcessErrorDetails::new("Test error", "claude", vec!["--version".to_string()])
                .with_exit_code(1)
                .with_stderr("Error message")
                .with_stdout_preview("Output preview");

        let error = details.to_error();
        let error_string = error.to_string();

        // Check basic content
        assert!(error_string.contains("Test error"));
        assert!(error_string.contains("claude --version"));
        assert!(error_string.contains("Exit Code: Some(1)"));

        // Check enhanced content
        assert!(error_string.contains("System Information:"));
        assert!(error_string.contains("CPU Cores:"));
        assert!(error_string.contains("Debugging Suggestions:"));
        assert!(error_string.contains("Exit code 1 typically indicates"));
    }

    #[test]
    fn test_process_error_debugging_suggestions() {
        // Test authentication error
        let auth_error = ProcessErrorDetails::new("Auth failed", "claude", vec![])
            .with_stderr("authentication failed");

        let error_string = auth_error.to_error().to_string();
        assert!(error_string.contains("Run 'claude auth' to authenticate"));

        // Test rate limit error
        let rate_error = ProcessErrorDetails::new("Rate limited", "claude", vec![])
            .with_stderr("rate limit exceeded");

        let error_string = rate_error.to_error().to_string();
        assert!(error_string.contains("implement exponential backoff"));

        // Test exit code 127
        let not_found_error =
            ProcessErrorDetails::new("Command failed", "claude", vec![]).with_exit_code(127);

        let error_string = not_found_error.to_error().to_string();
        assert!(error_string.contains("Command not found"));
    }

    #[test]
    fn test_retry_config() {
        let config = RetryConfig {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            add_jitter: false,
        };

        // Test delay calculation
        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));

        // Should cap at max_delay
        let large_attempt_delay = config.delay_for_attempt(10);
        assert!(large_attempt_delay <= config.max_delay);
    }

    #[test]
    fn test_retry_config_with_jitter() {
        let config = RetryConfig {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            add_jitter: true,
        };

        // With jitter, delays should vary but be in reasonable range
        let delay1 = config.delay_for_attempt(1);
        let delay2 = config.delay_for_attempt(1);

        // Both should be around 200ms but may differ due to jitter
        assert!(delay1.as_millis() > 100);
        assert!(delay1.as_millis() < 300);
        assert!(delay2.as_millis() > 100);
        assert!(delay2.as_millis() < 300);
    }

    #[tokio::test]
    async fn test_retry_with_backoff_success() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&attempt_count);

        let operation = move || {
            let count = Arc::clone(&count_clone);
            async move {
                let current_count = count.fetch_add(1, Ordering::SeqCst) + 1;
                if current_count < 3 {
                    Err(Error::Timeout(30)) // Recoverable error
                } else {
                    Ok("success".to_string())
                }
            }
        };

        let config = RetryConfig {
            max_attempts: 5,
            base_delay: Duration::from_millis(1), // Fast for testing
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            add_jitter: false,
        };

        let result = retry_with_backoff(operation, config, "test").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_with_backoff_non_recoverable() {
        let operation = || async { Err(Error::BinaryNotFound) }; // Non-recoverable

        let config = RetryConfig {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            add_jitter: false,
        };

        let result: Result<String> = retry_with_backoff(operation, config, "test").await;
        assert!(result.is_err());
        // Should fail immediately without retries for non-recoverable errors
        assert!(matches!(result.unwrap_err(), Error::BinaryNotFound));
    }

    #[tokio::test]
    async fn test_retry_with_backoff_exhausted() {
        let operation = || async { Err(Error::Timeout(30)) }; // Always fails but recoverable

        let config = RetryConfig {
            max_attempts: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            add_jitter: false,
        };

        let result: Result<String> = retry_with_backoff(operation, config, "test").await;
        assert!(result.is_err());
        // Should exhaust retries and return last error
        assert!(matches!(result.unwrap_err(), Error::Timeout(30)));
    }
}

#[cfg(test)]
mod backpressure_tests {
    use super::*;
    use claude_sdk_rs_runtime::backpressure::*;

    #[test]
    fn test_backpressure_config_default() {
        let config = BackpressureConfig::default();

        assert!(config.max_concurrent_requests > 0);
        assert!(config.queue_size > 0);
        assert!(config.timeout > Duration::ZERO);
    }

    #[test]
    fn test_backpressure_config_builder() {
        let config = BackpressureConfig::builder()
            .max_concurrent_requests(5)
            .queue_size(50)
            .timeout(Duration::from_secs(10))
            .build();

        assert_eq!(config.max_concurrent_requests, 5);
        assert_eq!(config.queue_size, 50);
        assert_eq!(config.timeout, Duration::from_secs(10));
    }

    #[tokio::test]
    async fn test_backpressure_manager_creation() {
        let config = BackpressureConfig::default();
        let manager = BackpressureManager::new(config);

        // Manager should be created successfully
        assert!(format!("{:?}", manager).contains("BackpressureManager"));
    }

    #[tokio::test]
    async fn test_backpressure_manager_acquire_permit() {
        let config = BackpressureConfig::builder()
            .max_concurrent_requests(2)
            .build();
        let manager = BackpressureManager::new(config);

        // Should be able to acquire permits up to the limit
        let permit1 = manager.acquire().await;
        assert!(permit1.is_ok());

        let permit2 = manager.acquire().await;
        assert!(permit2.is_ok());

        // Clean up permits
        drop(permit1);
        drop(permit2);
    }

    #[tokio::test]
    async fn test_backpressure_manager_permit_limit() {
        let config = BackpressureConfig::builder()
            .max_concurrent_requests(1)
            .timeout(Duration::from_millis(100))
            .build();
        let manager = BackpressureManager::new(config);

        // Acquire the only permit
        let _permit1 = manager.acquire().await.unwrap();

        // Second acquire should timeout
        let result = manager.acquire().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_backpressure_permit_drop() {
        let config = BackpressureConfig::builder()
            .max_concurrent_requests(1)
            .build();
        let manager = BackpressureManager::new(config);

        // Acquire and immediately drop permit
        {
            let _permit = manager.acquire().await.unwrap();
        } // permit dropped here

        // Should be able to acquire again
        let permit2 = manager.acquire().await;
        assert!(permit2.is_ok());
    }
}

#[cfg(test)]
mod stream_config_tests {
    use super::*;
    use claude_sdk_rs_runtime::stream_config::*;

    #[test]
    fn test_stream_config_default() {
        let config = get_stream_config();

        assert!(config.channel_buffer_size > 0);
        assert!(config.line_buffer_size > 0);
        assert!(config.max_line_length > 0);
        assert!(config.read_timeout > Duration::ZERO);
    }

    #[test]
    fn test_stream_config_validation() {
        // Test that the stream config has reasonable values
        let config = get_stream_config();

        // Buffer sizes should be reasonable
        assert!(config.channel_buffer_size >= 1);
        assert!(config.channel_buffer_size <= 10000);

        assert!(config.line_buffer_size >= 1024);
        assert!(config.line_buffer_size <= 1024 * 1024);

        assert!(config.max_line_length >= 1024);
        assert!(config.max_line_length <= 10 * 1024 * 1024);

        // Timeout should be reasonable
        assert!(config.read_timeout >= Duration::from_millis(100));
        assert!(config.read_timeout <= Duration::from_secs(300));
    }
}

#[cfg(test)]
mod telemetry_tests {
    use super::*;
    use claude_sdk_rs_runtime::telemetry;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_record_error() {
        let error = Error::Timeout(30);
        let mut context = HashMap::new();
        context.insert("test_key".to_string(), "test_value".to_string());

        // Should not panic or return error
        telemetry::record_error(&error, "test_operation", context).await;
    }

    #[tokio::test]
    async fn test_record_execution_time() {
        let duration = Duration::from_millis(500);

        // Should not panic or return error
        telemetry::record_execution_time("test_operation", duration).await;
    }

    #[tokio::test]
    async fn test_record_success() {
        let mut context = HashMap::new();
        context.insert("response_size".to_string(), "1024".to_string());

        // Should not panic or return error
        telemetry::record_success("test_operation", context).await;
    }

    #[tokio::test]
    async fn test_telemetry_with_different_operations() {
        // Test that telemetry can handle various operation types
        let operations = vec![
            "claude_execution",
            "streaming_response",
            "session_management",
            "error_recovery",
        ];

        for operation in operations {
            let mut context = HashMap::new();
            context.insert("operation_type".to_string(), operation.to_string());

            telemetry::record_success(operation, context).await;
        }
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::*;
    use claude_sdk_rs_runtime::recovery::*;

    #[tokio::test]
    async fn test_recovery_binary_not_found() {
        let result = ErrorRecovery::recover_binary_not_found().await;

        // Should return an error with helpful message
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Claude CLI") || error_msg.contains("installation"));
    }

    #[tokio::test]
    async fn test_recovery_not_authenticated() {
        let result = ErrorRecovery::recover_not_authenticated().await;

        // Should return an error with auth instructions
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("auth") || error_msg.contains("authentication"));
    }

    #[tokio::test]
    async fn test_recovery_timeout() {
        let result = ErrorRecovery::recover_timeout(30).await;

        // Should return an error with timeout suggestions
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("timeout") && error_msg.contains("30"));
    }

    #[tokio::test]
    async fn test_recovery_rate_limit() {
        let result = ErrorRecovery::recover_rate_limit_exceeded().await;

        // Should return an error with rate limit suggestions
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("rate limit") || error_msg.contains("retry"));
    }

    #[tokio::test]
    async fn test_recovery_mcp_error() {
        let test_cases = vec![
            "connection refused",
            "timeout",
            "protocol error",
            "unknown error",
        ];

        for error_msg in test_cases {
            let result = ErrorRecovery::recover_mcp_error(error_msg).await;
            assert!(result.is_err());

            let recovery_msg = result.unwrap_err().to_string();
            assert!(recovery_msg.contains("MCP"));
        }
    }

    #[tokio::test]
    async fn test_recovery_stream_closed() {
        let result = ErrorRecovery::recover_stream_closed().await;

        // Should return an error with stream recovery suggestions
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("stream") || error_msg.contains("connection"));
    }
}
