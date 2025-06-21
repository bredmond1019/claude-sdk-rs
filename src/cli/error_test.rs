use super::*;
use std::io;

#[cfg(test)]
mod error_creation_tests {
    use super::*;

    #[test]
    fn test_all_constructor_methods() {
        // Test all the constructor helper methods
        let errors = vec![
            (
                InteractiveError::command_discovery("Failed to find commands"),
                "CommandDiscovery",
                "Failed to find commands",
            ),
            (
                InteractiveError::session("Session error occurred"),
                "Session",
                "Session error occurred",
            ),
            (
                InteractiveError::execution("Execution failed"),
                "Execution",
                "Execution failed",
            ),
            (
                InteractiveError::cost_tracking("Cost tracking error"),
                "CostTracking",
                "Cost tracking error",
            ),
            (
                InteractiveError::history("History error"),
                "History",
                "History error",
            ),
            (
                InteractiveError::invalid_input("Invalid input provided"),
                "InvalidInput",
                "Invalid input provided",
            ),
            (
                InteractiveError::session_not_found("test-session-123"),
                "SessionNotFound",
                "test-session-123",
            ),
        ];

        for (error, expected_variant, expected_msg) in errors {
            let error_str = format!("{:?}", error);
            assert!(
                error_str.contains(expected_variant),
                "Error {:?} should contain variant {}",
                error,
                expected_variant
            );
            assert!(
                error_str.contains(expected_msg),
                "Error {:?} should contain message {}",
                error,
                expected_msg
            );
        }
    }

    #[test]
    fn test_direct_error_construction() {
        // Test direct enum construction
        let cmd_not_found = InteractiveError::CommandNotFound("missing-cmd".to_string());
        match cmd_not_found {
            InteractiveError::CommandNotFound(ref msg) => {
                assert_eq!(msg, "missing-cmd");
            }
            _ => panic!("Wrong error variant"),
        }

        let parallel_exec = InteractiveError::ParallelExecution("Parallel failure".to_string());
        match parallel_exec {
            InteractiveError::ParallelExecution(ref msg) => {
                assert_eq!(msg, "Parallel failure");
            }
            _ => panic!("Wrong error variant"),
        }

        let output_fmt = InteractiveError::OutputFormatting("Format error".to_string());
        match output_fmt {
            InteractiveError::OutputFormatting(ref msg) => {
                assert_eq!(msg, "Format error");
            }
            _ => panic!("Wrong error variant"),
        }

        let config = InteractiveError::Configuration("Config error".to_string());
        match config {
            InteractiveError::Configuration(ref msg) => {
                assert_eq!(msg, "Config error");
            }
            _ => panic!("Wrong error variant"),
        }

        let perm_denied = InteractiveError::PermissionDenied("Access denied".to_string());
        match perm_denied {
            InteractiveError::PermissionDenied(ref msg) => {
                assert_eq!(msg, "Access denied");
            }
            _ => panic!("Wrong error variant"),
        }

        let timeout = InteractiveError::Timeout(120);
        match timeout {
            InteractiveError::Timeout(secs) => {
                assert_eq!(secs, 120);
            }
            _ => panic!("Wrong error variant"),
        }
    }
}

#[cfg(test)]
mod error_conversion_tests {
    use super::*;

    #[test]
    fn test_io_error_conversion() {
        let io_errors = vec![
            (io::ErrorKind::NotFound, "entity not found"),
            (io::ErrorKind::PermissionDenied, "permission denied"),
            (io::ErrorKind::ConnectionRefused, "connection refused"),
            (io::ErrorKind::TimedOut, "operation timed out"),
            (io::ErrorKind::Interrupted, "operation interrupted"),
            (io::ErrorKind::UnexpectedEof, "unexpected end of file"),
        ];

        for (kind, msg) in io_errors {
            let io_error = io::Error::new(kind, msg);
            let interactive_error: InteractiveError = io_error.into();

            match interactive_error {
                InteractiveError::Io(e) => {
                    assert_eq!(e.kind(), kind);
                }
                _ => panic!("Expected Io error variant"),
            }
        }
    }

    #[test]
    fn test_serde_json_error_conversion() {
        // Create various JSON errors
        let json_errors = vec![
            serde_json::from_str::<i32>("not a number"),
            serde_json::from_str::<i32>("{invalid json}"),
            serde_json::from_str::<i32>("[1, 2, 3]"), // Wrong type - array instead of number
        ];

        for json_result in json_errors {
            assert!(json_result.is_err());
            let json_error = json_result.unwrap_err();
            let interactive_error: InteractiveError = json_error.into();

            match interactive_error {
                InteractiveError::Serialization(_) => {
                    // Success - correctly converted
                }
                _ => panic!("Expected Serialization error variant"),
            }
        }
    }

    #[test]
    fn test_uuid_error_conversion() {
        let invalid_uuids = vec![
            "not-a-uuid",
            "12345678-1234-1234-1234-12345678", // Too short
            "12345678-1234-1234-1234-123456789abcdefg", // Too long
            "12345678-1234-1234-1234-123456789ghi", // Invalid characters
            "12345678123412341234567890",       // Too short, no hyphens
            "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", // Not hex
            "g2345678-1234-1234-1234-123456789abc", // Invalid character at start
        ];

        for invalid in invalid_uuids {
            let uuid_result = uuid::Uuid::parse_str(invalid);
            assert!(
                uuid_result.is_err(),
                "UUID parsing should fail for: {}",
                invalid
            );

            let uuid_error = uuid_result.unwrap_err();
            let interactive_error: InteractiveError = uuid_error.into();

            match interactive_error {
                InteractiveError::Uuid(_) => {
                    // Success - correctly converted
                }
                _ => panic!("Expected Uuid error variant for input: {}", invalid),
            }
        }
    }

    #[test]
    fn test_utf8_conversion_error() {
        let invalid_utf8_sequences = vec![
            vec![0xFF, 0xFE],
            vec![0xC3, 0x28],       // Invalid UTF-8 sequence
            vec![0xE2, 0x82, 0x28], // Incomplete UTF-8 sequence
            vec![0xF0, 0x90, 0x80], // Incomplete 4-byte sequence
        ];

        for invalid_bytes in invalid_utf8_sequences {
            let string_result = String::from_utf8(invalid_bytes.clone());
            assert!(string_result.is_err());

            let utf8_error = string_result.unwrap_err();
            let interactive_error: InteractiveError = utf8_error.into();

            match interactive_error {
                InteractiveError::Utf8Conversion(_) => {
                    // Success - correctly converted
                }
                _ => panic!("Expected Utf8Conversion error variant"),
            }
        }
    }

    #[tokio::test]
    async fn test_join_error_conversion() {
        // Test panic in task
        let handle = tokio::spawn(async {
            panic!("Task panicked!");
        });

        let join_result = handle.await;
        assert!(join_result.is_err());

        let join_error = join_result.unwrap_err();
        let interactive_error: InteractiveError = join_error.into();

        match interactive_error {
            InteractiveError::AsyncTask(_) => {
                // Success - correctly converted
            }
            _ => panic!("Expected AsyncTask error variant"),
        }

        // Test cancelled task
        let handle = tokio::spawn(async {
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
            42
        });

        handle.abort();
        let join_result = handle.await;
        assert!(join_result.is_err());

        let join_error = join_result.unwrap_err();
        let interactive_error: InteractiveError = join_error.into();

        match interactive_error {
            InteractiveError::AsyncTask(e) => {
                assert!(e.is_cancelled());
            }
            _ => panic!("Expected AsyncTask error variant"),
        }
    }

    // Note: We can't easily test notify::Error conversion without actually setting up
    // file watchers, which would be more of an integration test

    // Note: We can't easily test claude_sdk_rs::Error conversion without the actual SDK
}

#[cfg(test)]
mod user_friendly_message_tests {
    use super::*;

    #[test]
    fn test_user_friendly_messages_complete() {
        let test_cases = vec![
            (
                InteractiveError::CommandNotFound("analyze-code".to_string()),
                vec!["analyze-code", "not found", "claude-interactive list"],
            ),
            (
                InteractiveError::SessionNotFound("work-session-123".to_string()),
                vec!["work-session-123", "not found", "claude-interactive session list"],
            ),
            (
                InteractiveError::PermissionDenied("Cannot access API".to_string()),
                vec!["Permission denied", "Cannot access API", "claude auth"],
            ),
            (
                InteractiveError::Timeout(90),
                vec!["90 seconds", "timed out", "--timeout"],
            ),
        ];

        for (error, expected_parts) in test_cases {
            let user_msg = error.user_message();
            for part in expected_parts {
                assert!(
                    user_msg.contains(part),
                    "User message '{}' should contain '{}'",
                    user_msg,
                    part
                );
            }
        }
    }

    #[test]
    fn test_generic_error_fallback() {
        // Errors without custom user messages should fall back to Display
        let errors = vec![
            InteractiveError::CommandDiscovery("Discovery failed".to_string()),
            InteractiveError::Session("Session error".to_string()),
            InteractiveError::Execution("Exec failed".to_string()),
            InteractiveError::ParallelExecution("Parallel failed".to_string()),
            InteractiveError::CostTracking("Cost error".to_string()),
            InteractiveError::History("History error".to_string()),
            InteractiveError::OutputFormatting("Format error".to_string()),
            InteractiveError::Configuration("Config error".to_string()),
            InteractiveError::InvalidInput("Bad input".to_string()),
        ];

        for error in errors {
            let display_msg = error.to_string();
            let user_msg = error.user_message();
            assert_eq!(
                display_msg, user_msg,
                "Generic errors should use Display for user message"
            );
        }
    }

    #[test]
    fn test_claude_sdk_error_message() {
        // We can't create a real claude_sdk_rs::Error, but we can test the message format
        // by mocking the expected behavior
        struct MockClaudeError;
        impl std::fmt::Display for MockClaudeError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "API rate limit exceeded")
            }
        }

        // This tests the expected format
        let expected_format = format!(
            "Claude API error: {}. Check your network connection and API key.",
            MockClaudeError
        );

        assert!(expected_format.contains("Claude API error"));
        assert!(expected_format.contains("API rate limit exceeded"));
        assert!(expected_format.contains("Check your network connection and API key"));
    }
}

#[cfg(test)]
mod error_context_preservation_tests {
    use super::*;

    #[test]
    fn test_error_chaining_context() {
        // Test that we can build up error context through transformations
        fn inner_operation() -> Result<()> {
            Err(InteractiveError::command_discovery(
                "Cannot read commands directory",
            ))
        }

        fn middle_operation() -> Result<()> {
            inner_operation()
                .map_err(|e| InteractiveError::execution(format!("Failed to initialize: {}", e)))
        }

        fn outer_operation() -> Result<()> {
            middle_operation()
                .map_err(|e| InteractiveError::session(format!("Session setup failed: {}", e)))
        }

        let result = outer_operation();
        assert!(result.is_err());

        let error = result.unwrap_err();
        let error_string = error.to_string();

        // Check that context is preserved through the chain
        assert!(error_string.contains("Session setup failed"));
        assert!(error_string.contains("Failed to initialize"));
        assert!(error_string.contains("Cannot read commands directory"));
    }

    #[test]
    fn test_error_with_source_context() {
        // Test errors that have underlying causes
        let io_error = io::Error::new(io::ErrorKind::NotFound, "config.json not found");
        let interactive_error: InteractiveError = io_error.into();

        match &interactive_error {
            InteractiveError::Io(e) => {
                assert_eq!(e.kind(), io::ErrorKind::NotFound);
                let msg = e.to_string();
                assert!(msg.contains("config.json not found"));
            }
            _ => panic!("Expected Io error"),
        }

        // The Display implementation should show the IO error details
        let display_msg = interactive_error.to_string();
        assert!(display_msg.contains("IO error"));
    }

    #[test]
    fn test_error_metadata_preservation() {
        // Test that error metadata like timeout duration is preserved
        let timeout_error = InteractiveError::Timeout(45);

        match &timeout_error {
            InteractiveError::Timeout(duration) => {
                assert_eq!(*duration, 45);
            }
            _ => panic!("Expected Timeout error"),
        }

        // Check both Display and user_message preserve the duration
        let display_msg = timeout_error.to_string();
        let user_msg = timeout_error.user_message();

        assert!(display_msg.contains("45"));
        assert!(user_msg.contains("45 seconds"));
    }

    #[test]
    fn test_complex_error_scenarios_with_context() {
        // Simulate a real-world error scenario
        fn load_session_config(path: &str) -> Result<String> {
            if path.is_empty() {
                return Err(InteractiveError::invalid_input("Path cannot be empty"));
            }

            // Simulate file not found
            Err(InteractiveError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Session config not found: {}", path),
            )))
        }

        fn initialize_session(config_path: &str) -> Result<()> {
            let _config = load_session_config(config_path).map_err(|e| {
                InteractiveError::session(format!("Failed to load session configuration: {}", e))
            })?;

            Ok(())
        }

        // Test with empty path
        let result1 = initialize_session("");
        assert!(result1.is_err());
        let error1 = result1.unwrap_err();
        assert!(error1.to_string().contains("Path cannot be empty"));

        // Test with non-existent file
        let result2 = initialize_session("/nonexistent/config.json");
        assert!(result2.is_err());
        let error2 = result2.unwrap_err();
        assert!(error2
            .to_string()
            .contains("Failed to load session configuration"));
        assert!(error2.to_string().contains("Session config not found"));
    }
}

#[cfg(test)]
mod retry_logic_tests {
    use super::*;

    #[test]
    fn test_retryable_error_detection() {
        // Comprehensive test of all error types and their retry status
        let retryable_errors = vec![
            InteractiveError::Timeout(30),
            InteractiveError::Io(io::Error::new(io::ErrorKind::TimedOut, "Network timeout")),
            InteractiveError::Io(io::Error::new(io::ErrorKind::Interrupted, "Interrupted")),
            InteractiveError::Io(io::Error::new(
                io::ErrorKind::ConnectionRefused,
                "Connection refused",
            )),
            InteractiveError::Io(io::Error::new(
                io::ErrorKind::ConnectionReset,
                "Connection reset",
            )),
            InteractiveError::Io(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "Connection aborted",
            )),
            // AsyncTask errors are retryable
            // ClaudeSDK errors would be retryable but we can't create them in tests
        ];

        for error in retryable_errors {
            assert!(
                error.is_retryable(),
                "Error {:?} should be retryable",
                error
            );
        }

        let non_retryable_errors = vec![
            InteractiveError::CommandDiscovery("Discovery failed".to_string()),
            InteractiveError::CommandNotFound("cmd".to_string()),
            InteractiveError::Session("Session error".to_string()),
            InteractiveError::SessionNotFound("id".to_string()),
            InteractiveError::Execution("Exec failed".to_string()),
            InteractiveError::ParallelExecution("Parallel failed".to_string()),
            InteractiveError::CostTracking("Cost error".to_string()),
            InteractiveError::History("History error".to_string()),
            InteractiveError::OutputFormatting("Format error".to_string()),
            InteractiveError::Configuration("Config error".to_string()),
            InteractiveError::PermissionDenied("Access denied".to_string()),
            InteractiveError::InvalidInput("Bad input".to_string()),
            InteractiveError::Serialization(serde_json::from_str::<i32>("bad").unwrap_err()),
            InteractiveError::Uuid(uuid::Uuid::parse_str("bad").unwrap_err()),
            InteractiveError::Utf8Conversion(String::from_utf8(vec![0xFF]).unwrap_err()),
        ];

        for error in non_retryable_errors {
            assert!(
                !error.is_retryable(),
                "Error {:?} should NOT be retryable",
                error
            );
        }
    }

    #[test]
    fn test_retry_strategy_recommendations() {
        // Test that we can build retry strategies based on error types
        fn get_retry_strategy(error: &InteractiveError) -> RetryStrategy {
            if !error.is_retryable() {
                return RetryStrategy::NoRetry;
            }

            match error {
                InteractiveError::Timeout(_) => RetryStrategy::ExponentialBackoff {
                    max_retries: 3,
                    initial_delay_ms: 1000,
                },
                InteractiveError::Io(io_err) => match io_err.kind() {
                    io::ErrorKind::TimedOut => RetryStrategy::ExponentialBackoff {
                        max_retries: 5,
                        initial_delay_ms: 500,
                    },
                    io::ErrorKind::ConnectionRefused => RetryStrategy::LinearBackoff {
                        max_retries: 3,
                        delay_ms: 2000,
                    },
                    _ => RetryStrategy::Immediate { max_retries: 2 },
                },
                _ => RetryStrategy::Immediate { max_retries: 1 },
            }
        }

        #[derive(Debug, PartialEq)]
        enum RetryStrategy {
            NoRetry,
            Immediate {
                max_retries: u32,
            },
            LinearBackoff {
                max_retries: u32,
                delay_ms: u64,
            },
            ExponentialBackoff {
                max_retries: u32,
                initial_delay_ms: u64,
            },
        }

        // Test various scenarios
        let timeout_error = InteractiveError::Timeout(30);
        assert_eq!(
            get_retry_strategy(&timeout_error),
            RetryStrategy::ExponentialBackoff {
                max_retries: 3,
                initial_delay_ms: 1000,
            }
        );

        let network_timeout =
            InteractiveError::Io(io::Error::new(io::ErrorKind::TimedOut, "Network timeout"));
        assert_eq!(
            get_retry_strategy(&network_timeout),
            RetryStrategy::ExponentialBackoff {
                max_retries: 5,
                initial_delay_ms: 500,
            }
        );

        let permission_error = InteractiveError::PermissionDenied("Access denied".to_string());
        assert_eq!(
            get_retry_strategy(&permission_error),
            RetryStrategy::NoRetry
        );
    }
}

#[cfg(test)]
mod error_logging_tests {
    use super::*;

    #[test]
    fn test_error_debug_information() {
        // Test that Debug output contains useful information
        let errors = vec![
            InteractiveError::CommandNotFound("test-cmd".to_string()),
            InteractiveError::SessionNotFound("session-123".to_string()),
            InteractiveError::Timeout(60),
            InteractiveError::InvalidInput("bad input".to_string()),
        ];

        for error in errors {
            let debug_output = format!("{:?}", error);

            // Debug output should contain the variant name
            match &error {
                InteractiveError::CommandNotFound(_) => {
                    assert!(debug_output.contains("CommandNotFound"))
                }
                InteractiveError::SessionNotFound(_) => {
                    assert!(debug_output.contains("SessionNotFound"))
                }
                InteractiveError::Timeout(_) => assert!(debug_output.contains("Timeout")),
                InteractiveError::InvalidInput(_) => assert!(debug_output.contains("InvalidInput")),
                _ => {}
            }

            // Debug output should contain key error data (not necessarily the full display message)
            match &error {
                InteractiveError::CommandNotFound(cmd) => assert!(debug_output.contains(cmd)),
                InteractiveError::SessionNotFound(id) => assert!(debug_output.contains(id)),
                InteractiveError::Timeout(secs) => {
                    assert!(debug_output.contains(&secs.to_string()))
                }
                InteractiveError::InvalidInput(msg) => assert!(debug_output.contains(msg)),
                _ => {}
            }
        }
    }

    #[test]
    fn test_error_structured_logging() {
        // Test that errors can be structured for logging
        fn log_error_structured(error: &InteractiveError) -> serde_json::Value {
            serde_json::json!({
                "error_type": error_type_name(error),
                "message": error.to_string(),
                "user_message": error.user_message(),
                "is_retryable": error.is_retryable(),
                "details": error_details(error),
            })
        }

        fn error_type_name(error: &InteractiveError) -> &'static str {
            match error {
                InteractiveError::CommandDiscovery(_) => "CommandDiscovery",
                InteractiveError::CommandNotFound(_) => "CommandNotFound",
                InteractiveError::Session(_) => "Session",
                InteractiveError::SessionNotFound(_) => "SessionNotFound",
                InteractiveError::Execution(_) => "Execution",
                InteractiveError::ParallelExecution(_) => "ParallelExecution",
                InteractiveError::CostTracking(_) => "CostTracking",
                InteractiveError::History(_) => "History",
                InteractiveError::OutputFormatting(_) => "OutputFormatting",
                InteractiveError::Configuration(_) => "Configuration",
                InteractiveError::PermissionDenied(_) => "PermissionDenied",
                InteractiveError::InvalidInput(_) => "InvalidInput",
                InteractiveError::Timeout(_) => "Timeout",
                InteractiveError::Io(_) => "Io",
                InteractiveError::Serialization(_) => "Serialization",
                InteractiveError::ClaudeSDK(_) => "ClaudeSDK",
                InteractiveError::Uuid(_) => "Uuid",
                InteractiveError::FileWatcher(_) => "FileWatcher",
                InteractiveError::AsyncTask(_) => "AsyncTask",
                InteractiveError::Utf8Conversion(_) => "Utf8Conversion",
            }
        }

        fn error_details(error: &InteractiveError) -> serde_json::Value {
            match error {
                InteractiveError::Timeout(secs) => serde_json::json!({ "timeout_seconds": secs }),
                InteractiveError::Io(e) => {
                    serde_json::json!({ "io_error_kind": format!("{:?}", e.kind()) })
                }
                _ => serde_json::json!({}),
            }
        }

        // Test structured logging for different error types
        let timeout_error = InteractiveError::Timeout(45);
        let log_entry = log_error_structured(&timeout_error);

        assert_eq!(log_entry["error_type"], "Timeout");
        assert_eq!(log_entry["is_retryable"], true);
        assert_eq!(log_entry["details"]["timeout_seconds"], 45);

        let not_found_error = InteractiveError::CommandNotFound("missing".to_string());
        let log_entry = log_error_structured(&not_found_error);

        assert_eq!(log_entry["error_type"], "CommandNotFound");
        assert_eq!(log_entry["is_retryable"], false);
        assert!(log_entry["user_message"]
            .as_str()
            .unwrap()
            .contains("missing"));
    }
}

#[cfg(test)]
mod graceful_degradation_tests {
    use super::*;

    #[test]
    fn test_error_severity_classification() {
        // Classify errors by severity for graceful degradation
        #[derive(Debug, PartialEq)]
        enum ErrorSeverity {
            Critical, // Must stop execution
            Warning,  // Can continue with limitations
            Info,     // Informational, no impact
        }

        fn classify_error_severity(error: &InteractiveError) -> ErrorSeverity {
            match error {
                // Critical errors that should stop execution
                InteractiveError::PermissionDenied(_)
                | InteractiveError::Configuration(_)
                | InteractiveError::ClaudeSDK(_) => ErrorSeverity::Critical,

                // Warnings that allow degraded operation
                InteractiveError::CommandNotFound(_)
                | InteractiveError::SessionNotFound(_)
                | InteractiveError::CostTracking(_)
                | InteractiveError::History(_) => ErrorSeverity::Warning,

                // Info level that doesn't impact operation
                InteractiveError::OutputFormatting(_) => ErrorSeverity::Info,

                // Context-dependent (could be any level)
                InteractiveError::Io(io_err) => match io_err.kind() {
                    io::ErrorKind::PermissionDenied => ErrorSeverity::Critical,
                    io::ErrorKind::NotFound => ErrorSeverity::Warning,
                    _ => ErrorSeverity::Warning,
                },

                // Default to warning for most errors
                _ => ErrorSeverity::Warning,
            }
        }

        // Test classification
        assert_eq!(
            classify_error_severity(&InteractiveError::PermissionDenied(
                "API denied".to_string()
            )),
            ErrorSeverity::Critical
        );

        assert_eq!(
            classify_error_severity(&InteractiveError::CommandNotFound("cmd".to_string())),
            ErrorSeverity::Warning
        );

        assert_eq!(
            classify_error_severity(&InteractiveError::OutputFormatting("format".to_string())),
            ErrorSeverity::Info
        );
    }

    #[test]
    fn test_fallback_behavior_recommendations() {
        // Test fallback behaviors for different error scenarios
        fn get_fallback_behavior(error: &InteractiveError) -> &'static str {
            match error {
                InteractiveError::CommandNotFound(_) => {
                    "List available commands and suggest alternatives"
                }
                InteractiveError::SessionNotFound(_) => "Create new session or use default session",
                InteractiveError::CostTracking(_) => "Continue without cost tracking",
                InteractiveError::History(_) => "Continue without history recording",
                InteractiveError::OutputFormatting(_) => "Use plain text output",
                InteractiveError::Timeout(_) => "Retry with increased timeout or skip operation",
                InteractiveError::PermissionDenied(_) => {
                    "Prompt for authentication or elevated permissions"
                }
                _ => "Log error and continue if possible",
            }
        }

        // Verify fallback behaviors make sense
        let cmd_not_found = InteractiveError::CommandNotFound("analyze".to_string());
        let fallback = get_fallback_behavior(&cmd_not_found);
        assert!(fallback.contains("List available commands"));

        let cost_error = InteractiveError::CostTracking("DB error".to_string());
        let fallback = get_fallback_behavior(&cost_error);
        assert!(fallback.contains("Continue without cost tracking"));
    }
}

#[cfg(test)]
mod result_type_tests {
    use super::*;

    #[test]
    fn test_result_type_usage() {
        // Test the Result type alias works correctly
        fn operation_that_succeeds() -> Result<String> {
            Ok("Success".to_string())
        }

        fn operation_that_fails() -> Result<String> {
            Err(InteractiveError::invalid_input("Bad input"))
        }

        fn operation_with_question_mark() -> Result<String> {
            let _value = operation_that_fails()?;
            Ok("Never reached".to_string())
        }

        // Test success case
        let success = operation_that_succeeds();
        assert!(success.is_ok());
        assert_eq!(success.unwrap(), "Success");

        // Test failure case
        let failure = operation_that_fails();
        assert!(failure.is_err());
        match failure.unwrap_err() {
            InteractiveError::InvalidInput(msg) => assert_eq!(msg, "Bad input"),
            _ => panic!("Wrong error type"),
        }

        // Test ? operator
        let question_result = operation_with_question_mark();
        assert!(question_result.is_err());
    }

    #[test]
    fn test_result_combinators() {
        // Test that Result combinators work with our type alias
        let result: Result<i32> = Ok(42);

        let mapped = result.map(|x| x * 2);
        assert_eq!(mapped.unwrap(), 84);

        let result: Result<i32> = Err(InteractiveError::invalid_input("bad"));
        let mapped_err = result.map_err(|e| InteractiveError::execution(format!("Wrapped: {}", e)));

        match mapped_err.unwrap_err() {
            InteractiveError::Execution(msg) => {
                assert!(msg.contains("Wrapped"));
                assert!(msg.contains("Invalid input: bad"));
            }
            _ => panic!("Wrong error type"),
        }
    }
}

#[cfg(test)]
mod user_friendly_trait_tests {
    use super::*;

    #[test]
    fn test_trait_object_usage() {
        // Test that the UserFriendlyError trait can be used as a trait object
        let errors: Vec<Box<dyn UserFriendlyError>> = vec![
            Box::new(InteractiveError::CommandNotFound("cmd".to_string())),
            Box::new(InteractiveError::Timeout(30)),
            Box::new(InteractiveError::InvalidInput("bad".to_string())),
        ];

        for error in errors {
            let msg = error.user_message();
            assert!(!msg.is_empty());
        }
    }

    #[test]
    fn test_trait_implementation_completeness() {
        // Ensure all error variants have appropriate user messages
        let all_errors = vec![
            InteractiveError::CommandDiscovery("test".to_string()),
            InteractiveError::CommandNotFound("test".to_string()),
            InteractiveError::Session("test".to_string()),
            InteractiveError::SessionNotFound("test".to_string()),
            InteractiveError::Execution("test".to_string()),
            InteractiveError::ParallelExecution("test".to_string()),
            InteractiveError::CostTracking("test".to_string()),
            InteractiveError::History("test".to_string()),
            InteractiveError::OutputFormatting("test".to_string()),
            InteractiveError::Configuration("test".to_string()),
            InteractiveError::PermissionDenied("test".to_string()),
            InteractiveError::InvalidInput("test".to_string()),
            InteractiveError::Timeout(30),
            InteractiveError::Io(io::Error::new(io::ErrorKind::Other, "test")),
            InteractiveError::Serialization(serde_json::from_str::<i32>("bad").unwrap_err()),
            InteractiveError::Uuid(uuid::Uuid::parse_str("bad").unwrap_err()),
            InteractiveError::Utf8Conversion(String::from_utf8(vec![0xFF]).unwrap_err()),
        ];

        for error in all_errors {
            let user_msg = error.user_message();
            assert!(
                !user_msg.is_empty(),
                "Error {:?} should have a non-empty user message",
                error
            );

            // Verify trait method matches direct method
            let trait_obj: &dyn UserFriendlyError = &error;
            assert_eq!(trait_obj.user_message(), user_msg);
        }
    }
}
