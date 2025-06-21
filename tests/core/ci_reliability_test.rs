//! CI/CD reliability tests for core functionality
//!
//! These tests are designed to be simple, fast, and reliable for CI/CD pipelines.
//! They focus on core functionality that should always work.

use claude_sdk_rs_core::{Config, Error, StreamFormat};
use serde::de::Error as SerdeError;

/// Test that basic config construction always works
#[test]
fn test_config_construction_reliability() {
    // Test default construction
    let default_config = Config::default();
    assert_eq!(default_config.stream_format, StreamFormat::Text);

    // Test builder construction that should always work
    let result = Config::builder()
        .model("claude-sonnet-4-20250514")
        .timeout_secs(30)
        .build();

    assert!(
        result.is_ok(),
        "Basic config construction should always succeed"
    );

    if let Ok(config) = result {
        assert_eq!(config.model, Some("claude-sonnet-4-20250514".to_string()));
        assert_eq!(config.timeout_secs, Some(30));
    }
}

/// Test that all stream formats are available and working
#[test]
fn test_stream_formats_reliability() {
    let formats = [
        StreamFormat::Text,
        StreamFormat::Json,
        StreamFormat::StreamJson,
    ];

    for format in formats {
        // Test construction
        let result = Config::builder().stream_format(format).build();

        assert!(result.is_ok(), "Stream format {:?} should be valid", format);

        if let Ok(config) = result {
            assert_eq!(config.stream_format, format);
        }

        // Test debug formatting
        let debug_str = format!("{:?}", format);
        assert!(
            !debug_str.is_empty(),
            "Format should have debug representation"
        );
    }
}

/// Test error type basic functionality
#[test]
fn test_error_types_reliability() {
    // Test that we can create different error types
    let process_error = Error::ProcessError("test error message".to_string());
    let serialization_error = Error::SerializationError(serde_json::Error::io(
        std::io::Error::new(std::io::ErrorKind::Other, "json error"),
    ));

    // Test that errors can be formatted
    let process_debug = format!("{:?}", process_error);
    let serialization_debug = format!("{:?}", serialization_error);

    assert!(!process_debug.is_empty());
    assert!(!serialization_debug.is_empty());
    assert!(process_debug.contains("test error message"));
    assert!(serialization_debug.contains("json error"));
}

/// Test config builder with various valid inputs
#[test]
fn test_config_builder_reliability() {
    // Test minimal config
    let minimal = Config::builder().build();
    assert!(minimal.is_ok(), "Minimal config should build");

    // Test config with model only
    let with_model = Config::builder().model("claude-haiku-3-20250307").build();
    assert!(with_model.is_ok(), "Config with model should build");

    // Test config with timeout only
    let with_timeout = Config::builder().timeout_secs(60).build();
    assert!(with_timeout.is_ok(), "Config with timeout should build");

    // Test config with system prompt
    let with_prompt = Config::builder()
        .system_prompt("You are a helpful assistant")
        .build();
    assert!(
        with_prompt.is_ok(),
        "Config with system prompt should build"
    );

    // Test config with all common fields
    let full_config = Config::builder()
        .model("claude-sonnet-4-20250514")
        .timeout_secs(45)
        .stream_format(StreamFormat::Json)
        .system_prompt("Test prompt")
        .max_tokens(1000)
        .build();

    assert!(full_config.is_ok(), "Full config should build");

    if let Ok(config) = full_config {
        assert_eq!(config.model, Some("claude-sonnet-4-20250514".to_string()));
        assert_eq!(config.timeout_secs, Some(45));
        assert_eq!(config.stream_format, StreamFormat::Json);
        assert_eq!(config.system_prompt, Some("Test prompt".to_string()));
        assert_eq!(config.max_tokens, Some(1000));
    }
}

/// Test config cloning and equality
#[test]
fn test_config_cloning_reliability() {
    let original = Config::builder()
        .model("test-model")
        .timeout_secs(30)
        .stream_format(StreamFormat::Json)
        .system_prompt("Test prompt")
        .build()
        .expect("Config should build");

    let cloned = original.clone();

    // Test that fields are correctly cloned
    assert_eq!(original.model, cloned.model);
    assert_eq!(original.timeout_secs, cloned.timeout_secs);
    assert_eq!(original.stream_format, cloned.stream_format);
    assert_eq!(original.system_prompt, cloned.system_prompt);
}

/// Test that config validation handles edge cases gracefully
#[test]
fn test_config_validation_reliability() {
    // Test with empty model string
    let empty_model = Config::builder().model("").build();

    // Should either succeed or fail gracefully
    match empty_model {
        Ok(_) => {
            // Empty model is allowed
        }
        Err(_) => {
            // Empty model is rejected, which is also fine
        }
    }

    // Test with very large timeout
    let large_timeout = Config::builder().timeout_secs(3600).build();

    assert!(large_timeout.is_ok(), "Large timeout should be allowed");

    // Test with very small timeout
    let small_timeout = Config::builder().timeout_secs(1).build();

    assert!(small_timeout.is_ok(), "Small timeout should be allowed");
}

/// Test that config can handle special characters and Unicode
#[test]
fn test_config_unicode_reliability() {
    let unicode_config = Config::builder()
        .system_prompt("Unicode test: 你好世界 🦀 Rust programming")
        .model("claude-sonnet-4-20250514")
        .build();

    assert!(
        unicode_config.is_ok(),
        "Unicode in system prompt should be supported"
    );

    if let Ok(config) = unicode_config {
        assert!(config.system_prompt.unwrap().contains("你好世界"));
    }
}

/// Test parallel config creation for thread safety
#[test]
fn test_config_parallel_creation_reliability() {
    use std::thread;

    let handles: Vec<_> = (0..5)
        .map(|i| {
            thread::spawn(move || {
                Config::builder()
                    .model(&format!("model-{}", i))
                    .timeout_secs(30 + i as u64)
                    .build()
                    .expect("Config should build in parallel")
            })
        })
        .collect();

    let configs: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("Thread should complete"))
        .collect();

    assert_eq!(configs.len(), 5);

    for (i, config) in configs.iter().enumerate() {
        assert_eq!(config.model, Some(format!("model-{}", i)));
        assert_eq!(config.timeout_secs, Some(30 + i as u64));
    }
}

/// Smoke test for basic functionality
#[test]
fn test_smoke_test() {
    // This test should always pass and validates basic compilation
    assert!(true);

    // Test that we can import and use basic types
    let _config = Config::default();
    let _format = StreamFormat::Text;
    let _error = Error::ProcessError("test".to_string());

    // Test that debug formatting works
    assert!(!format!("{:?}", _config).is_empty());
    assert!(!format!("{:?}", _format).is_empty());
    assert!(!format!("{:?}", _error).is_empty());
}
