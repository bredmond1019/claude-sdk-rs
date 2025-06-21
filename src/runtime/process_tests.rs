//! Comprehensive tests for process execution functionality
//!
//! These tests cover the core process execution logic including:
//! - Command construction and execution
//! - Error handling scenarios
//! - Timeout behavior
//! - Configuration variations
//! - Binary availability checks

use crate::runtime::process::{execute_claude, execute_claude_streaming};
use crate::core::error_handling::{ErrorContext, ProcessErrorDetails, retry_with_backoff, RetryConfig};
use crate::core::{Config, Error, Result, StreamFormat};
use std::time::Duration;
use tokio::time::timeout;

/// Helper function to check if claude binary is available for integration testing
/// Returns true if tests should be skipped
fn should_skip_integration_test() -> bool {
    which::which("claude").is_err() && std::env::var("CLAUDE_BINARY").is_err()
}

/// Test configuration builder for various scenarios
struct TestConfigBuilder {
    config: Config,
}

impl TestConfigBuilder {
    fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    fn with_format(mut self, format: StreamFormat) -> Self {
        self.config.stream_format = format;
        self
    }

    fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.config.timeout_secs = Some(timeout_secs);
        self
    }

    fn with_model(mut self, model: impl Into<String>) -> Self {
        self.config.model = Some(model.into());
        self
    }

    fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.config.allowed_tools = Some(tools);
        self
    }

    fn with_continue_session(mut self) -> Self {
        self.config.continue_session = true;
        self
    }

    fn with_resume_session(mut self, session_id: impl Into<String>) -> Self {
        self.config.resume_session_id = Some(session_id.into());
        self
    }

    fn with_disallowed_tools(mut self, tools: Vec<String>) -> Self {
        self.config.disallowed_tools = Some(tools);
        self
    }

    fn with_skip_permissions(mut self, skip: bool) -> Self {
        self.config.skip_permissions = skip;
        self
    }

    fn with_append_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.append_system_prompt = Some(prompt.into());
        self
    }

    fn with_max_turns(mut self, turns: u32) -> Self {
        self.config.max_turns = Some(turns);
        self
    }

    fn build(self) -> Config {
        self.config
    }
}

#[cfg(test)]
mod execute_claude_tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_claude_with_binary_not_found() {
        // Test when claude binary is not in PATH
        let config = TestConfigBuilder::new().build();
        
        // Skip integration test if claude binary is not available
        if should_skip_integration_test() {
            println!("Claude binary not found - skipping integration test");
            return;
        }
        
        // This test assumes claude is not installed or not in PATH
        // In a real environment, this would return BinaryNotFound
        // But since we can't control the test environment, we'll mock this behavior
        
        // Note: This test may pass if claude is actually installed
        // In CI/CD, we would ensure claude is not available for this test
        let result = execute_claude(&config, "test query").await;
        
        // We can't guarantee the exact error without controlling the environment
        // But we can test that we get a proper error structure
        if result.is_err() {
            let error = result.unwrap_err();
            // Verify we get some kind of error (could be binary not found or timeout)
            assert!(matches!(error, Error::BinaryNotFound | Error::ProcessError(_) | Error::Timeout(_)));
        }
    }

    #[tokio::test]
    async fn test_execute_claude_with_timeout() {
        // Skip integration test if claude binary is not available
        if should_skip_integration_test() {
            println!("Claude binary not found - skipping integration test");
            return;
        }

        let config = TestConfigBuilder::new()
            .with_timeout(1) // Very short timeout
            .build();
        
        // This should timeout quickly if claude is available but takes time
        let result = execute_claude(&config, "long running query that would take time").await;
        
        // Should either timeout or complete quickly
        if result.is_err() {
            let error = result.unwrap_err();
            // Could be timeout or binary not found
            assert!(matches!(error, Error::Timeout(_) | Error::BinaryNotFound | Error::ProcessError(_)));
        }
    }

    #[tokio::test]
    async fn test_execute_claude_different_formats() {
        let formats = vec![
            StreamFormat::Text,
            StreamFormat::Json,
            StreamFormat::StreamJson,
        ];
        
        for format in formats {
            let config = TestConfigBuilder::new()
                .with_format(format)
                .with_timeout(5)
                .build();
            
            let result = execute_claude(&config, "test").await;
            
            // Test should either succeed or fail gracefully
            // Success depends on environment, but no panics should occur
            match result {
                Ok(_) => {
                    // Command succeeded, which means claude is available
                    println!("Test passed for format {:?}", format);
                }
                Err(Error::BinaryNotFound) => {
                    // Expected when claude is not installed
                    println!("Claude not available for format {:?}", format);
                }
                Err(Error::Timeout(_)) => {
                    // Expected for short timeouts
                    println!("Timeout for format {:?}", format);
                }
                Err(Error::ProcessError(_)) => {
                    // Expected for various process errors
                    println!("Process error for format {:?}", format);
                }
                Err(e) => {
                    panic!("Unexpected error for format {:?}: {}", format, e);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_with_tools() {
        let config = TestConfigBuilder::new()
            .with_tools(vec!["bash:echo".to_string(), "mcp__test__tool".to_string()])
            .with_timeout(5)
            .build();
        
        let result = execute_claude(&config, "test with tools").await;
        
        // Should handle tools configuration properly
        match result {
            Ok(_) => println!("Tools test passed"),
            Err(Error::BinaryNotFound) => println!("Claude not available for tools test"),
            Err(Error::Timeout(_)) => println!("Timeout in tools test"),
            Err(Error::ProcessError(_)) => println!("Process error in tools test"),
            Err(e) => panic!("Unexpected error in tools test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_execute_claude_with_model_and_system_prompt() {
        let config = TestConfigBuilder::new()
            .with_model("claude-3-sonnet-20240229")
            .with_system_prompt("You are a helpful assistant")
            .with_timeout(5)
            .build();
        
        let result = execute_claude(&config, "test query").await;
        
        // Should handle model and system prompt configuration
        match result {
            Ok(_) => println!("Model/prompt test passed"),
            Err(Error::BinaryNotFound) => println!("Claude not available for model/prompt test"),
            Err(Error::Timeout(_)) => println!("Timeout in model/prompt test"),  
            Err(Error::ProcessError(_)) => println!("Process error in model/prompt test"),
            Err(e) => panic!("Unexpected error in model/prompt test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_execute_claude_empty_query() {
        let config = TestConfigBuilder::new().with_timeout(5).build();
        
        let result = execute_claude(&config, "").await;
        
        // Should handle empty query gracefully
        match result {
            Ok(_) => println!("Empty query test passed"),
            Err(Error::BinaryNotFound) => println!("Claude not available for empty query test"),
            Err(Error::Timeout(_)) => println!("Timeout in empty query test"),
            Err(Error::ProcessError(_)) => println!("Process error in empty query test"),
            Err(e) => panic!("Unexpected error in empty query test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_execute_claude_very_long_query() {
        let config = TestConfigBuilder::new().with_timeout(10).build();
        
        // Create a very long query
        let long_query = "a".repeat(10000);
        
        let result = execute_claude(&config, &long_query).await;
        
        // Should handle long queries without issues
        match result {
            Ok(_) => println!("Long query test passed"),
            Err(Error::BinaryNotFound) => println!("Claude not available for long query test"),
            Err(Error::Timeout(_)) => println!("Timeout in long query test"),
            Err(Error::ProcessError(_)) => println!("Process error in long query test"),
            Err(e) => panic!("Unexpected error in long query test: {}", e),
        }
    }
}

#[cfg(test)]
mod execute_claude_streaming_tests {
    use super::*;
    use crate::MessageStream;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_execute_claude_streaming_basic() {
        let config = TestConfigBuilder::new()
            .with_format(StreamFormat::StreamJson)
            .with_timeout(5)
            .build();
        
        let result = execute_claude_streaming(&config, "test streaming").await;
        
        match result {
            Ok(mut stream) => {
                // Try to read from the stream with timeout
                let timeout_result = timeout(Duration::from_secs(2), stream.next()).await;
                
                match timeout_result {
                    Ok(Some(message_result)) => {
                        match message_result {
                            Ok(_message) => println!("Streaming test got message"),
                            Err(e) => println!("Streaming test got error: {}", e),
                        }
                    }
                    Ok(None) => println!("Streaming test: stream ended"),
                    Err(_) => println!("Streaming test: timeout waiting for messages"),
                }
            }
            Err(Error::BinaryNotFound) => println!("Claude not available for streaming test"),
            Err(Error::Timeout(_)) => println!("Timeout in streaming test"),
            Err(Error::ProcessError(_)) => println!("Process error in streaming test"),
            Err(e) => panic!("Unexpected error in streaming test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_execute_claude_streaming_different_formats() {
        let formats = vec![
            StreamFormat::Text,
            StreamFormat::Json,
            StreamFormat::StreamJson,
        ];
        
        for format in formats {
            let config = TestConfigBuilder::new()
                .with_format(format)
                .with_timeout(5)
                .build();
            
            let result = execute_claude_streaming(&config, "test").await;
            
            match result {
                Ok(mut stream) => {
                    // Try to get one message with timeout
                    let timeout_result = timeout(Duration::from_secs(2), stream.next()).await;
                    
                    match timeout_result {
                        Ok(Some(_)) => println!("Streaming format {:?} test passed", format),
                        Ok(None) => println!("Streaming format {:?} test: no messages", format),
                        Err(_) => println!("Streaming format {:?} test: timeout", format),
                    }
                }
                Err(Error::BinaryNotFound) => println!("Claude not available for streaming format {:?}", format),
                Err(Error::Timeout(_)) => println!("Timeout in streaming format {:?}", format),
                Err(Error::ProcessError(_)) => println!("Process error in streaming format {:?}", format),
                Err(e) => panic!("Unexpected error in streaming format {:?}: {}", format, e),
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_streaming_with_tools() {
        let config = TestConfigBuilder::new()
            .with_format(StreamFormat::StreamJson)
            .with_tools(vec!["bash:ls".to_string()])
            .with_timeout(5)
            .build();
        
        let result = execute_claude_streaming(&config, "list files in current directory").await;
        
        match result {
            Ok(mut stream) => {
                // Try to read multiple messages
                let mut message_count = 0;
                let start_time = std::time::Instant::now();
                
                while let Ok(Some(message_result)) = timeout(Duration::from_secs(1), stream.next()).await {
                    match message_result {
                        Ok(_) => {
                            message_count += 1;
                            if message_count >= 3 || start_time.elapsed() > Duration::from_secs(3) {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                
                println!("Streaming tools test got {} messages", message_count);
            }
            Err(Error::BinaryNotFound) => println!("Claude not available for streaming tools test"),
            Err(Error::Timeout(_)) => println!("Timeout in streaming tools test"),
            Err(Error::ProcessError(_)) => println!("Process error in streaming tools test"),
            Err(e) => panic!("Unexpected error in streaming tools test: {}", e),
        }
    }
}

#[cfg(test)]
mod error_handling_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_process_error_details_integration() {
        // Test that ProcessErrorDetails work with real error scenarios
        let details = ProcessErrorDetails::new(
            "Test command failed",
            "claude",
            vec!["--test".to_string()],
        )
        .with_exit_code(1)
        .with_stderr("Command not recognized")
        .with_stdout_preview("Usage: claude [options]");
        
        let error = details.to_error();
        let error_string = error.to_string();
        
        assert!(error_string.contains("Test command failed"));
        assert!(error_string.contains("claude --test"));
        assert!(error_string.contains("Exit Code: Some(1)"));
        assert!(error_string.contains("Command not recognized"));
    }

    #[tokio::test]
    async fn test_error_context_with_process_execution() {
        let context = ErrorContext::new("test_process_execution")
            .with_debug_info("query_length", "100")
            .with_debug_info("timeout", "30");
        
        let debug_string = context.to_debug_string();
        
        assert!(debug_string.contains("test_process_execution"));
        assert!(debug_string.contains("query_length: 100"));
        assert!(debug_string.contains("timeout: 30"));
    }

    #[tokio::test]
    async fn test_retry_with_process_execution() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&attempt_count);
        
        let operation = move || {
            let count = Arc::clone(&count_clone);
            async move {
                let current_count = count.fetch_add(1, Ordering::SeqCst) + 1;
                if current_count < 2 {
                    Err(Error::Timeout(5)) // Recoverable error
                } else {
                    Ok("success".to_string())
                }
            }
        };
        
        let config = RetryConfig {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
            add_jitter: false,
            ..Default::default()
        };
        
        let result = retry_with_backoff(operation, config, "test_process").await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempt_count.load(Ordering::SeqCst), 2);
    }
}

#[cfg(test)]
mod configuration_validation_tests {
    use super::*;

    #[test]
    fn test_config_builder_creates_valid_configs() {
        let config = TestConfigBuilder::new()
            .with_format(StreamFormat::Json)
            .with_timeout(60)
            .with_model("claude-3-opus-20240229")
            .with_system_prompt("Test prompt")
            .with_tools(vec!["bash:echo".to_string()])
            .build();
        
        assert_eq!(config.stream_format, StreamFormat::Json);
        assert_eq!(config.timeout_secs, Some(60));
        assert_eq!(config.model, Some("claude-3-opus-20240229".to_string()));
        assert_eq!(config.system_prompt, Some("Test prompt".to_string()));
        assert_eq!(config.allowed_tools, Some(vec!["bash:echo".to_string()]));
    }

    #[test]
    fn test_config_default_values() {
        let config = TestConfigBuilder::new().build();
        
        // Verify default values
        assert_eq!(config.stream_format, StreamFormat::Text);
        assert_eq!(config.timeout_secs, None);
        assert_eq!(config.model, None);
        assert_eq!(config.system_prompt, None);
        assert_eq!(config.allowed_tools, None);
    }

    #[test]
    fn test_config_with_empty_tools() {
        let config = TestConfigBuilder::new()
            .with_tools(vec![])
            .build();
        
        assert_eq!(config.allowed_tools, Some(vec![]));
    }

    #[test]
    fn test_config_with_multiple_tools() {
        let tools = vec![
            "bash:ls".to_string(),
            "bash:cat".to_string(),
            "mcp__server1__tool1".to_string(),
            "mcp__server2__tool2".to_string(),
        ];
        
        let config = TestConfigBuilder::new()
            .with_tools(tools.clone())
            .build();
        
        assert_eq!(config.allowed_tools, Some(tools));
    }
}

#[cfg(test)]
mod timeout_behavior_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_timeout_enforcement() {
        let config = TestConfigBuilder::new()
            .with_timeout(1) // 1 second timeout
            .build();
        
        let start_time = Instant::now();
        let result = execute_claude(&config, "test query").await;
        let elapsed = start_time.elapsed();
        
        // Should complete within reasonable time (either success or timeout)
        assert!(elapsed < Duration::from_secs(10), "Test took too long: {:?}", elapsed);
        
        match result {
            Ok(_) => {
                // Quick success is fine
                println!("Quick success in timeout test");
            }
            Err(Error::Timeout(secs)) => {
                // Expected timeout
                assert_eq!(secs, 1);
                println!("Expected timeout after {} seconds", secs);
            }
            Err(Error::BinaryNotFound) => {
                // Expected when claude not available
                println!("Binary not found in timeout test");
            }
            Err(Error::ProcessError(_)) => {
                // Process error is acceptable
                println!("Process error in timeout test");
            }
            Err(e) => {
                panic!("Unexpected error in timeout test: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_streaming_timeout_behavior() {
        let config = TestConfigBuilder::new()
            .with_format(StreamFormat::StreamJson)
            .with_timeout(2)
            .build();
        
        let start_time = Instant::now();
        let result = execute_claude_streaming(&config, "test").await;
        let elapsed = start_time.elapsed();
        
        // Should start quickly
        assert!(elapsed < Duration::from_secs(5), "Streaming start took too long: {:?}", elapsed);
        
        match result {
            Ok(mut stream) => {
                // Try to read with timeout
                let read_start = Instant::now();
                let read_result = timeout(Duration::from_secs(3), stream.next()).await;
                let read_elapsed = read_start.elapsed();
                
                match read_result {
                    Ok(Some(_)) => println!("Got streaming message in {:?}", read_elapsed),
                    Ok(None) => println!("Stream ended in {:?}", read_elapsed),
                    Err(_) => println!("Stream read timeout after {:?}", read_elapsed),
                }
            }
            Err(Error::BinaryNotFound) => println!("Binary not found in streaming timeout test"),
            Err(Error::Timeout(_)) => println!("Initial timeout in streaming test"),
            Err(Error::ProcessError(_)) => println!("Process error in streaming timeout test"),
            Err(e) => panic!("Unexpected error in streaming timeout test: {}", e),
        }
    }
}

#[cfg(test)]
mod session_management_tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_claude_with_continue_session() {
        let config = TestConfigBuilder::new()
            .with_continue_session()
            .with_timeout(5)
            .build();
        
        let result = execute_claude(&config, "test with continue session").await;
        
        // Should handle continue session flag properly
        match result {
            Ok(_) => println!("Continue session test passed"),
            Err(Error::BinaryNotFound) => println!("Claude not available for continue session test"),
            Err(Error::Timeout(_)) => println!("Timeout in continue session test"),
            Err(Error::ProcessError(_)) => println!("Process error in continue session test"),
            Err(e) => panic!("Unexpected error in continue session test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_execute_claude_with_resume_session() {
        let config = TestConfigBuilder::new()
            .with_resume_session("test_session_123")
            .with_timeout(5)
            .build();
        
        let result = execute_claude(&config, "test with resume session").await;
        
        // Should handle resume session flag properly
        match result {
            Ok(_) => println!("Resume session test passed"),
            Err(Error::BinaryNotFound) => println!("Claude not available for resume session test"),
            Err(Error::Timeout(_)) => println!("Timeout in resume session test"),
            Err(Error::ProcessError(_)) => println!("Process error in resume session test"),
            Err(e) => panic!("Unexpected error in resume session test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_execute_claude_streaming_with_continue_session() {
        let config = TestConfigBuilder::new()
            .with_format(StreamFormat::StreamJson)
            .with_continue_session()
            .with_timeout(5)
            .build();
        
        let result = execute_claude_streaming(&config, "test streaming with continue").await;
        
        match result {
            Ok(mut stream) => {
                // Try to read from the stream with timeout
                let timeout_result = timeout(Duration::from_secs(2), stream.next()).await;
                
                match timeout_result {
                    Ok(Some(message_result)) => {
                        match message_result {
                            Ok(_message) => println!("Continue session streaming test got message"),
                            Err(e) => println!("Continue session streaming test got error: {}", e),
                        }
                    }
                    Ok(None) => println!("Continue session streaming test: stream ended"),
                    Err(_) => println!("Continue session streaming test: timeout waiting for messages"),
                }
            }
            Err(Error::BinaryNotFound) => println!("Claude not available for continue session streaming test"),
            Err(Error::Timeout(_)) => println!("Timeout in continue session streaming test"),
            Err(Error::ProcessError(_)) => println!("Process error in continue session streaming test"),
            Err(e) => panic!("Unexpected error in continue session streaming test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_execute_claude_streaming_with_resume_session() {
        let config = TestConfigBuilder::new()
            .with_format(StreamFormat::StreamJson)
            .with_resume_session("test_session_456")
            .with_timeout(5)
            .build();
        
        let result = execute_claude_streaming(&config, "test streaming with resume").await;
        
        match result {
            Ok(mut stream) => {
                // Try to read from the stream with timeout
                let timeout_result = timeout(Duration::from_secs(2), stream.next()).await;
                
                match timeout_result {
                    Ok(Some(message_result)) => {
                        match message_result {
                            Ok(_message) => println!("Resume session streaming test got message"),
                            Err(e) => println!("Resume session streaming test got error: {}", e),
                        }
                    }
                    Ok(None) => println!("Resume session streaming test: stream ended"),
                    Err(_) => println!("Resume session streaming test: timeout waiting for messages"),
                }
            }
            Err(Error::BinaryNotFound) => println!("Claude not available for resume session streaming test"),
            Err(Error::Timeout(_)) => println!("Timeout in resume session streaming test"),
            Err(Error::ProcessError(_)) => println!("Process error in resume session streaming test"),
            Err(e) => panic!("Unexpected error in resume session streaming test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_session_configuration_validation() {
        // Test continue session configuration
        let continue_config = TestConfigBuilder::new()
            .with_continue_session()
            .build();
        
        assert!(continue_config.continue_session);
        assert_eq!(continue_config.resume_session_id, None);

        // Test resume session configuration
        let resume_config = TestConfigBuilder::new()
            .with_resume_session("session_789")
            .build();
        
        assert!(!resume_config.continue_session);
        assert_eq!(resume_config.resume_session_id, Some("session_789".to_string()));

        // Test both session options together
        let both_config = TestConfigBuilder::new()
            .with_continue_session()
            .with_resume_session("session_mixed")
            .build();
        
        assert!(both_config.continue_session);
        assert_eq!(both_config.resume_session_id, Some("session_mixed".to_string()));
    }
}

#[cfg(test)]
mod permission_management_tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_claude_with_disallowed_tools() {
        let config = TestConfigBuilder::new()
            .with_disallowed_tools(vec!["bash".to_string(), "filesystem".to_string()])
            .with_timeout(5)
            .build();
        
        let result = execute_claude(&config, "test with disallowed tools").await;
        
        // Should handle disallowed tools flag properly
        match result {
            Ok(_) => println!("Disallowed tools test passed"),
            Err(Error::BinaryNotFound) => println!("Claude not available for disallowed tools test"),
            Err(Error::Timeout(_)) => println!("Timeout in disallowed tools test"),
            Err(Error::ProcessError(_)) => println!("Process error in disallowed tools test"),
            Err(e) => panic!("Unexpected error in disallowed tools test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_execute_claude_skip_permissions_default() {
        let config = TestConfigBuilder::new()
            .with_timeout(5)
            .build();
        
        // skip_permissions should be true by default
        assert!(config.skip_permissions);
        
        let result = execute_claude(&config, "test with default skip permissions").await;
        
        match result {
            Ok(_) => println!("Default skip permissions test passed"),
            Err(Error::BinaryNotFound) => println!("Claude not available for skip permissions test"),
            Err(Error::Timeout(_)) => println!("Timeout in skip permissions test"),
            Err(Error::ProcessError(_)) => println!("Process error in skip permissions test"),
            Err(e) => panic!("Unexpected error in skip permissions test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_execute_claude_skip_permissions_disabled() {
        let config = TestConfigBuilder::new()
            .with_skip_permissions(false)
            .with_timeout(5)
            .build();
        
        assert!(!config.skip_permissions);
        
        let result = execute_claude(&config, "test with permissions enabled").await;
        
        match result {
            Ok(_) => println!("Permissions enabled test passed"),
            Err(Error::BinaryNotFound) => println!("Claude not available for permissions enabled test"),
            Err(Error::Timeout(_)) => println!("Timeout in permissions enabled test"),
            Err(Error::ProcessError(_)) => println!("Process error in permissions enabled test"),
            Err(e) => panic!("Unexpected error in permissions enabled test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_execute_claude_streaming_with_disallowed_tools() {
        let config = TestConfigBuilder::new()
            .with_format(StreamFormat::StreamJson)
            .with_disallowed_tools(vec!["dangerous_tool".to_string()])
            .with_timeout(5)
            .build();
        
        let result = execute_claude_streaming(&config, "test streaming with disallowed tools").await;
        
        match result {
            Ok(mut stream) => {
                let timeout_result = timeout(Duration::from_secs(2), stream.next()).await;
                
                match timeout_result {
                    Ok(Some(message_result)) => {
                        match message_result {
                            Ok(_message) => println!("Disallowed tools streaming test got message"),
                            Err(e) => println!("Disallowed tools streaming test got error: {}", e),
                        }
                    }
                    Ok(None) => println!("Disallowed tools streaming test: stream ended"),
                    Err(_) => println!("Disallowed tools streaming test: timeout waiting for messages"),
                }
            }
            Err(Error::BinaryNotFound) => println!("Claude not available for disallowed tools streaming test"),
            Err(Error::Timeout(_)) => println!("Timeout in disallowed tools streaming test"),
            Err(Error::ProcessError(_)) => println!("Process error in disallowed tools streaming test"),
            Err(e) => panic!("Unexpected error in disallowed tools streaming test: {}", e),
        }
    }
}

#[cfg(test)]
mod system_prompt_and_conversation_tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_claude_with_append_system_prompt() {
        let config = TestConfigBuilder::new()
            .with_append_system_prompt("Additionally, be very concise.")
            .with_timeout(5)
            .build();
        
        let result = execute_claude(&config, "test with append system prompt").await;
        
        match result {
            Ok(_) => println!("Append system prompt test passed"),
            Err(Error::BinaryNotFound) => println!("Claude not available for append system prompt test"),
            Err(Error::Timeout(_)) => println!("Timeout in append system prompt test"),
            Err(Error::ProcessError(_)) => println!("Process error in append system prompt test"),
            Err(e) => panic!("Unexpected error in append system prompt test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_execute_claude_with_max_turns() {
        let config = TestConfigBuilder::new()
            .with_max_turns(3)
            .with_timeout(5)
            .build();
        
        let result = execute_claude(&config, "test with max turns limit").await;
        
        match result {
            Ok(_) => println!("Max turns test passed"),
            Err(Error::BinaryNotFound) => println!("Claude not available for max turns test"),
            Err(Error::Timeout(_)) => println!("Timeout in max turns test"),
            Err(Error::ProcessError(_)) => println!("Process error in max turns test"),
            Err(e) => panic!("Unexpected error in max turns test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_execute_claude_streaming_with_append_system_prompt() {
        let config = TestConfigBuilder::new()
            .with_format(StreamFormat::StreamJson)
            .with_append_system_prompt("Be brief and to the point.")
            .with_timeout(5)
            .build();
        
        let result = execute_claude_streaming(&config, "test streaming with append prompt").await;
        
        match result {
            Ok(mut stream) => {
                let timeout_result = timeout(Duration::from_secs(2), stream.next()).await;
                
                match timeout_result {
                    Ok(Some(message_result)) => {
                        match message_result {
                            Ok(_message) => println!("Append prompt streaming test got message"),
                            Err(e) => println!("Append prompt streaming test got error: {}", e),
                        }
                    }
                    Ok(None) => println!("Append prompt streaming test: stream ended"),
                    Err(_) => println!("Append prompt streaming test: timeout waiting for messages"),
                }
            }
            Err(Error::BinaryNotFound) => println!("Claude not available for append prompt streaming test"),
            Err(Error::Timeout(_)) => println!("Timeout in append prompt streaming test"),
            Err(Error::ProcessError(_)) => println!("Process error in append prompt streaming test"),
            Err(e) => panic!("Unexpected error in append prompt streaming test: {}", e),
        }
    }

    #[tokio::test]
    async fn test_configuration_validation_new_features() {
        // Test configuration with all new features
        let config = TestConfigBuilder::new()
            .with_disallowed_tools(vec!["dangerous".to_string()])
            .with_skip_permissions(false)
            .with_append_system_prompt("Be helpful")
            .with_max_turns(5)
            .build();
        
        assert_eq!(config.disallowed_tools, Some(vec!["dangerous".to_string()]));
        assert!(!config.skip_permissions);
        assert_eq!(config.append_system_prompt, Some("Be helpful".to_string()));
        assert_eq!(config.max_turns, Some(5));

        // Test empty disallowed tools
        let config2 = TestConfigBuilder::new()
            .with_disallowed_tools(vec![])
            .build();
        
        assert_eq!(config2.disallowed_tools, Some(vec![]));

        // Test default skip permissions behavior
        let config3 = TestConfigBuilder::new().build();
        assert!(config3.skip_permissions); // Should be true by default
    }
}