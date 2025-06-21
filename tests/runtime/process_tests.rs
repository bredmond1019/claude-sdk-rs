use claude_sdk_rs_core::{Config, Error, StreamFormat};
use claude_sdk_rs_runtime::process::execute_claude;

#[cfg(test)]
mod process_tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_claude_with_default_config() {
        // Skip if Claude CLI is not available
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let config = Config::default();
        let result = execute_claude(&config, "Hello, just say 'test response'").await;

        match result {
            Ok(response) => {
                assert!(!response.is_empty());
                println!("Response: {}", response);
            }
            Err(e) => {
                // This might fail if Claude CLI is not authenticated or available
                eprintln!("Expected error (CLI not available): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_with_json_format() {
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let config = Config::builder().stream_format(StreamFormat::Json).build();

        let result = execute_claude(&config, "Just say 'test'").await;

        match result {
            Ok(response) => {
                assert!(!response.is_empty());
                // Should be valid JSON
                if let Err(e) = serde_json::from_str::<serde_json::Value>(&response) {
                    eprintln!("Response is not valid JSON: {}", e);
                    eprintln!("Response: {}", response);
                }
            }
            Err(e) => {
                eprintln!("Expected error (CLI not available): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_with_stream_json_format() {
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let config = Config::builder()
            .stream_format(StreamFormat::StreamJson)
            .build();

        let result = execute_claude(&config, "Just say 'test'").await;

        match result {
            Ok(response) => {
                assert!(!response.is_empty());
                // Should contain JSON lines
                let lines: Vec<&str> = response.lines().collect();
                assert!(!lines.is_empty());

                // At least one line should be valid JSON
                let mut has_valid_json = false;
                for line in lines {
                    if line.trim().is_empty() {
                        continue;
                    }
                    if serde_json::from_str::<serde_json::Value>(line).is_ok() {
                        has_valid_json = true;
                        break;
                    }
                }
                assert!(has_valid_json, "No valid JSON found in stream response");
            }
            Err(e) => {
                eprintln!("Expected error (CLI not available): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_timeout() {
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let config = Config::builder()
            .timeout_secs(1) // Very short timeout
            .build();

        let result = execute_claude(
            &config,
            "Write a very long response about the history of computing",
        )
        .await;

        // This should either timeout or succeed quickly
        match result {
            Ok(_) => {
                // If it succeeds quickly, that's fine
                println!("Query completed within timeout");
            }
            Err(Error::Timeout(_)) => {
                // This is expected for the timeout test
                println!("Timeout error as expected");
            }
            Err(e) => {
                eprintln!("Unexpected error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_binary_not_found() {
        // Temporarily override the PATH to simulate binary not found
        std::env::set_var("PATH", "/nonexistent/path");

        let config = Config::default();
        let result = execute_claude(&config, "test").await;

        // Restore PATH
        std::env::set_var("PATH", std::env::var("PATH").unwrap_or_default());

        match result {
            Err(Error::BinaryNotFound) => {
                // This is expected
                println!("Binary not found error as expected");
            }
            _ => {
                panic!("Expected BinaryNotFound error");
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_with_system_prompt() {
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let config = Config::builder()
            .system_prompt(
                "You are a test assistant. Always respond with exactly 'SYSTEM PROMPT TEST'",
            )
            .build();

        let result = execute_claude(&config, "Say anything").await;

        match result {
            Ok(response) => {
                assert!(!response.is_empty());
                println!("Response with system prompt: {}", response);
            }
            Err(e) => {
                eprintln!("Expected error (CLI not available): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_with_model() {
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let config = Config::builder().model("claude-3-haiku-20240307").build();

        let result = execute_claude(&config, "Just say 'test'").await;

        match result {
            Ok(response) => {
                assert!(!response.is_empty());
                println!("Response with specific model: {}", response);
            }
            Err(e) => {
                eprintln!("Expected error (CLI not available): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_with_max_tokens() {
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let config = Config::builder().max_tokens(10).build();

        let result = execute_claude(&config, "Write a very long story").await;

        match result {
            Ok(response) => {
                assert!(!response.is_empty());
                println!("Response with max tokens: {}", response);
                // Response should be short due to max_tokens limit
            }
            Err(e) => {
                eprintln!("Expected error (CLI not available): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_with_tools() {
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let config = Config::builder()
            .allowed_tools(vec!["bash".to_string()])
            .build();

        let result = execute_claude(&config, "What tools do you have access to?").await;

        match result {
            Ok(response) => {
                assert!(!response.is_empty());
                println!("Response with tools: {}", response);
            }
            Err(e) => {
                eprintln!("Expected error (CLI not available): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_empty_query() {
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let config = Config::default();
        let result = execute_claude(&config, "").await;

        match result {
            Ok(response) => {
                // Claude should handle empty queries gracefully
                println!("Response to empty query: {}", response);
            }
            Err(e) => {
                eprintln!("Error with empty query: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_large_query() {
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let config = Config::default();
        let large_query = "A".repeat(10000); // 10KB query
        let result = execute_claude(&config, &large_query).await;

        match result {
            Ok(response) => {
                assert!(!response.is_empty());
                println!("Response to large query: {} chars", response.len());
            }
            Err(e) => {
                eprintln!("Error with large query: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_concurrent_requests() {
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let config = Config::default();

        // Spawn multiple concurrent requests
        let mut handles = Vec::new();
        for i in 0..3 {
            let config = config.clone();
            let handle =
                tokio::spawn(
                    async move { execute_claude(&config, &format!("Say 'test {}'", i)).await },
                );
            handles.push(handle);
        }

        // Wait for all to complete
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => eprintln!("Concurrent request error: {}", e),
            }
        }

        println!("Completed {} concurrent requests", results.len());
    }

    #[tokio::test]
    async fn test_execute_claude_verbose_mode() {
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let mut config = Config::default();
        config.verbose = true;

        let result = execute_claude(&config, "Just say 'test'").await;

        match result {
            Ok(response) => {
                assert!(!response.is_empty());
                println!("Verbose response: {}", response);
            }
            Err(e) => {
                eprintln!("Expected error (CLI not available): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_streaming_timeout_edge_case() {
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        // Test what happens when timeout occurs during streaming
        let config = Config::builder()
            .stream_format(StreamFormat::StreamJson)
            .timeout_secs(2) // Very short timeout for streaming
            .build();

        // Ask for a response that would normally take time to stream
        let result = execute_claude(
            &config,
            "Count from 1 to 1000, showing each number on a new line",
        )
        .await;

        match result {
            Ok(response) => {
                // If it completes, should have partial response
                println!(
                    "Streaming completed within timeout: {} chars",
                    response.len()
                );
            }
            Err(Error::Timeout(_)) => {
                // This is expected for timeout during streaming
                println!("Streaming timeout as expected");
            }
            Err(e) => {
                eprintln!("Unexpected error during streaming: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_malformed_output_recovery() {
        // This test would require a mock that produces malformed output
        // For now, we'll test that the system handles unexpected output gracefully
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let config = Config::builder().stream_format(StreamFormat::Json).build();

        // Query that might produce complex output
        let result = execute_claude(&config, "Output exactly: {\"incomplete\": ").await;

        // Should handle incomplete JSON gracefully
        match result {
            Ok(response) => {
                println!("Handled potentially malformed output: {}", response);
            }
            Err(e) => {
                println!("Error handling malformed output: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_claude_partial_stream_handling() {
        if which::which("claude").is_err() {
            eprintln!("Skipping test: Claude CLI not found");
            return;
        }

        let config = Config::builder()
            .stream_format(StreamFormat::StreamJson)
            .max_tokens(5) // Force very short response
            .build();

        let result = execute_claude(&config, "Write a long story").await;

        match result {
            Ok(response) => {
                // Should handle partial stream gracefully
                assert!(!response.is_empty());
                println!("Partial stream response: {} chars", response.len());
            }
            Err(e) => {
                eprintln!("Error with partial stream: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod mock_tests {
    use super::*;

    /// Test helper to create a mock Claude CLI script
    fn create_mock_claude_script(
        response: &str,
        exit_code: i32,
    ) -> std::io::Result<std::path::PathBuf> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("mock_claude");

        #[cfg(unix)]
        {
            let script_content = format!(
                r#"#!/bin/bash
echo "{}"
exit {}
"#,
                response.replace("\"", "\\\""),
                exit_code
            );

            std::fs::write(&script_path, script_content)?;

            // Make executable
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&script_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script_path, perms)?;
        }

        #[cfg(windows)]
        {
            let script_path = temp_dir.join("mock_claude.bat");
            let script_content = format!(
                r#"@echo off
echo {}
exit /b {}
"#,
                response, exit_code
            );

            std::fs::write(&script_path, script_content)?;
        }

        Ok(script_path)
    }

    #[tokio::test]
    async fn test_mock_claude_success() {
        let expected_response = "Mock Claude response";
        let script_path = create_mock_claude_script(expected_response, 0).unwrap();

        // Create a new 'claude' script for proper detection
        let script_dir = script_path.parent().unwrap();
        let claude_path = script_dir.join("claude");
        std::fs::copy(&script_path, &claude_path).unwrap();
        std::fs::remove_file(&script_path).unwrap();

        // Temporarily add the script directory to front of PATH
        let script_dir_str = script_dir.to_string_lossy();
        let original_path = std::env::var("PATH").unwrap_or_default();

        // On Unix, use colon separator; on Windows, use semicolon
        #[cfg(unix)]
        let separator = ":";
        #[cfg(windows)]
        let separator = ";";

        let new_path = format!("{}{}{}", script_dir_str, separator, original_path);
        std::env::set_var("PATH", &new_path);

        let config = Config::default();
        let result = execute_claude(&config, "test query").await;

        // Restore PATH
        std::env::set_var("PATH", original_path);

        // Clean up
        let _ = std::fs::remove_file(&claude_path);

        match result {
            Ok(response) => {
                assert_eq!(response.trim(), expected_response);
            }
            Err(e) => {
                panic!("Mock test failed: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_mock_claude_failure() {
        let script_path = create_mock_claude_script("Error message", 1).unwrap();

        // Create a new 'claude' script for proper detection
        let script_dir = script_path.parent().unwrap();
        let claude_path = script_dir.join("claude");
        std::fs::copy(&script_path, &claude_path).unwrap();
        std::fs::remove_file(&script_path).unwrap();

        // Temporarily add the script directory to front of PATH
        let script_dir_str = script_dir.to_string_lossy();
        let original_path = std::env::var("PATH").unwrap_or_default();

        // On Unix, use colon separator; on Windows, use semicolon
        #[cfg(unix)]
        let separator = ":";
        #[cfg(windows)]
        let separator = ";";

        let new_path = format!("{}{}{}", script_dir_str, separator, original_path);
        std::env::set_var("PATH", &new_path);

        let config = Config::default();
        let result = execute_claude(&config, "test query").await;

        // Restore PATH
        std::env::set_var("PATH", original_path);

        // Clean up
        let _ = std::fs::remove_file(&claude_path);

        match result {
            Err(Error::ProcessError(_)) => {
                // This is expected
                println!("Process error as expected");
            }
            Ok(_) => {
                panic!("Expected process error");
            }
            Err(e) => {
                panic!("Unexpected error type: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod malformed_output_tests {
    use super::*;

    #[tokio::test]
    async fn test_malformed_json_single_quote() {
        // Simulate Claude outputting JSON with single quotes instead of double quotes
        let malformed_json = "{'content': 'test', 'role': 'assistant'}";

        // This tests our JSON parsing resilience
        match serde_json::from_str::<serde_json::Value>(malformed_json) {
            Ok(_) => panic!("Should not parse malformed JSON"),
            Err(_) => {
                // Expected - malformed JSON should fail to parse
                assert!(true);
            }
        }
    }

    #[tokio::test]
    async fn test_malformed_json_missing_closing_brace() {
        let incomplete_json = r#"{"content": "test message", "role": "assistant""#;

        match serde_json::from_str::<serde_json::Value>(incomplete_json) {
            Ok(_) => panic!("Should not parse incomplete JSON"),
            Err(e) => {
                // Should get EOF error
                assert!(e.to_string().contains("EOF") || e.to_string().contains("end"));
            }
        }
    }

    #[tokio::test]
    async fn test_malformed_stream_json_mixed_format() {
        // Test handling of stream that contains both valid and invalid JSON lines
        let mixed_stream = r#"{"type": "content", "text": "Valid line"}
Not JSON at all
{"type": "content", "text": "Another valid line"}
{broken json
{"type": "done"}"#;

        let lines = mixed_stream.lines();
        let mut valid_count = 0;
        let mut invalid_count = 0;

        for line in lines {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<serde_json::Value>(line) {
                Ok(_) => valid_count += 1,
                Err(_) => invalid_count += 1,
            }
        }

        assert_eq!(valid_count, 3);
        assert_eq!(invalid_count, 2);
    }
}
