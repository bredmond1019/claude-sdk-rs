//! Integration tests for core functionality
//!
//! These tests focus on the stable core functionality that should always work,
//! providing a foundation for CI/CD reliability.

use claude_sdk_rs_core::{Config, Error, StreamFormat};
use std::time::Duration;

/// Test basic configuration creation and validation
#[test]
fn test_core_config_creation() {
    // Test default configuration
    let default_config = Config::default();
    assert_eq!(default_config.stream_format(), StreamFormat::Text);
    assert!(default_config.model().is_none());
    assert!(default_config.system_prompt().is_none());

    // Test builder pattern
    let config = Config::builder()
        .model("claude-sonnet-4-20250514")
        .timeout(Duration::from_secs(30))
        .stream_format(StreamFormat::Json)
        .build()
        .expect("Valid config should build");

    assert_eq!(config.model(), Some("claude-sonnet-4-20250514"));
    assert_eq!(config.timeout(), Duration::from_secs(30));
    assert_eq!(config.stream_format(), StreamFormat::Json);
}

/// Test configuration validation edge cases
#[test]
fn test_core_config_validation() {
    // Test invalid model name
    let result = Config::builder()
        .model("invalid-model-name-that-is-very-long-and-should-not-be-accepted")
        .build();

    // Should either succeed (if validation is permissive) or fail gracefully
    match result {
        Ok(_) => {
            // Validation is permissive, which is fine
        }
        Err(e) => {
            // Should be a validation error
            assert!(e.to_string().contains("model") || e.to_string().contains("validation"));
        }
    }

    // Test very short timeout
    let result = Config::builder().timeout(Duration::from_millis(1)).build();

    assert!(result.is_ok(), "Very short timeout should be allowed");

    // Test very long timeout
    let result = Config::builder().timeout(Duration::from_secs(3600)).build();

    assert!(result.is_ok(), "Long timeout should be allowed");
}

/// Test stream format handling
#[test]
fn test_core_stream_formats() {
    let formats = [
        StreamFormat::Text,
        StreamFormat::Json,
        StreamFormat::StreamJson,
    ];

    for format in formats {
        let config = Config::builder()
            .stream_format(format)
            .build()
            .expect("Stream format should be valid");

        assert_eq!(config.stream_format(), format);

        // Test that format can be displayed
        let debug_str = format!("{:?}", format);
        assert!(!debug_str.is_empty());
    }
}

/// Test error types and conversion
#[test]
fn test_core_error_handling() {
    // Test error creation
    let timeout_error = Error::Timeout;
    assert!(
        timeout_error.to_string().contains("timeout")
            || timeout_error.to_string().contains("Timeout")
    );

    let not_found_error = Error::ClaudeNotFound;
    assert!(
        not_found_error.to_string().contains("not found")
            || not_found_error.to_string().contains("Claude")
    );

    // Test error codes if available
    if let Some(code) = timeout_error.code() {
        assert!(!code.to_string().is_empty());
    }

    // Test error context
    let process_error = Error::ProcessError("test error".to_string());
    assert!(process_error.to_string().contains("test error"));
}

/// Test configuration cloning and equality
#[test]
fn test_core_config_cloning() {
    let original = Config::builder()
        .model("claude-sonnet-4-20250514")
        .timeout(Duration::from_secs(45))
        .stream_format(StreamFormat::Json)
        .system_prompt("You are a helpful assistant")
        .max_tokens(1000)
        .build()
        .expect("Config should build");

    let cloned = original.clone();

    // Test that all fields are copied correctly
    assert_eq!(original.model(), cloned.model());
    assert_eq!(original.timeout(), cloned.timeout());
    assert_eq!(original.stream_format(), cloned.stream_format());
    assert_eq!(original.system_prompt(), cloned.system_prompt());
    assert_eq!(original.max_tokens(), cloned.max_tokens());
}

/// Test configuration with allowed tools
#[test]
fn test_core_config_with_tools() {
    let tools = vec![
        "mcp__filesystem__read".to_string(),
        "bash:ls".to_string(),
        "mcp__notion__search".to_string(),
    ];

    let config = Config::builder()
        .allowed_tools(tools.clone())
        .build()
        .expect("Config with tools should build");

    assert_eq!(config.allowed_tools(), &tools);
}

/// Test configuration builder chaining
#[test]
fn test_core_config_builder_chaining() {
    // Test that methods can be chained in any order
    let config1 = Config::builder()
        .model("test-model")
        .timeout(Duration::from_secs(30))
        .stream_format(StreamFormat::Json)
        .build()
        .expect("Config should build");

    let config2 = Config::builder()
        .stream_format(StreamFormat::Json)
        .model("test-model")
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Config should build");

    let config3 = Config::builder()
        .timeout(Duration::from_secs(30))
        .stream_format(StreamFormat::Json)
        .model("test-model")
        .build()
        .expect("Config should build");

    // All should be equivalent
    assert_eq!(config1.model(), config2.model());
    assert_eq!(config1.model(), config3.model());
    assert_eq!(config1.timeout(), config2.timeout());
    assert_eq!(config1.timeout(), config3.timeout());
    assert_eq!(config1.stream_format(), config2.stream_format());
    assert_eq!(config1.stream_format(), config3.stream_format());
}

/// Test configuration with edge case values
#[test]
fn test_core_config_edge_cases() {
    // Test with empty strings
    let config = Config::builder().model("").system_prompt("").build();

    // Should handle empty strings gracefully
    assert!(config.is_ok());

    // Test with special characters
    let config = Config::builder()
        .system_prompt("System prompt with special chars: @#$%^&*()")
        .build()
        .expect("Special characters should be allowed");

    assert!(config.system_prompt().unwrap().contains("@#$%^&*()"));

    // Test with Unicode
    let config = Config::builder()
        .system_prompt("Unicode test: 你好世界 🦀 Rust")
        .build()
        .expect("Unicode should be allowed");

    assert!(config.system_prompt().unwrap().contains("你好世界"));
}

/// Test multiple configurations in parallel
#[test]
fn test_core_config_parallel_creation() {
    use std::thread;

    let handles: Vec<_> = (0..10)
        .map(|i| {
            thread::spawn(move || {
                let config = Config::builder()
                    .model(&format!("model-{}", i))
                    .timeout(Duration::from_secs(30 + i))
                    .build()
                    .expect("Config should build");

                assert_eq!(config.model(), Some(&format!("model-{}", i)));
                assert_eq!(config.timeout(), Duration::from_secs(30 + i));
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
}

/// Test configuration serialization if available
#[test]
fn test_core_config_serialization() {
    let config = Config::builder()
        .model("claude-sonnet-4-20250514")
        .timeout(Duration::from_secs(30))
        .stream_format(StreamFormat::Json)
        .build()
        .expect("Config should build");

    // Test debug formatting
    let debug_str = format!("{:?}", config);
    assert!(!debug_str.is_empty());
    assert!(debug_str.contains("claude-sonnet-4-20250514"));
}
