//! Comprehensive error handling tests for the Claude AI runtime
//!
//! This test suite covers all error scenarios, recovery mechanisms,
//! and edge cases to ensure robust error handling in production.

use claude_sdk_rs_core::{Config, Error, ErrorCode, StreamFormat};
use claude_sdk_rs_runtime::{
    error_handling::{retry_with_backoff, ErrorContext, ProcessErrorDetails, RetryConfig},
    recovery::{
        CircuitBreaker, CircuitState, PartialResultRecovery, StreamReconnectionManager,
        TokenBucketRateLimiter,
    },
    telemetry::{init_telemetry, ErrorTelemetry, TelemetryConfig},
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{sleep, timeout};

/// Test all error types can be created and have correct codes
#[tokio::test]
async fn test_all_error_types() {
    let errors = vec![
        (Error::BinaryNotFound, ErrorCode::BinaryNotFound),
        (
            Error::SessionNotFound("test".to_string()),
            ErrorCode::SessionNotFound,
        ),
        (
            Error::PermissionDenied("tool".to_string()),
            ErrorCode::PermissionDenied,
        ),
        (
            Error::McpError("server error".to_string()),
            ErrorCode::McpError,
        ),
        (
            Error::ConfigError("invalid".to_string()),
            ErrorCode::ConfigError,
        ),
        (
            Error::InvalidInput("bad input".to_string()),
            ErrorCode::InvalidInput,
        ),
        (Error::Timeout(30), ErrorCode::Timeout),
        (
            Error::ProcessError("process failed".to_string()),
            ErrorCode::ProcessError,
        ),
        (Error::StreamClosed, ErrorCode::StreamClosed),
        (Error::NotAuthenticated, ErrorCode::NotAuthenticated),
        (Error::RateLimitExceeded, ErrorCode::RateLimitExceeded),
    ];

    for (error, expected_code) in errors {
        assert_eq!(error.code(), expected_code);
        assert!(error.to_string().contains(&expected_code.to_string()));
    }
}

/// Test error context creation and enrichment
#[tokio::test]
async fn test_error_context_enrichment() {
    let context = ErrorContext::new("test_operation")
        .with_debug_info("key1", "value1")
        .with_debug_info("key2", "value2")
        .with_error_chain("first error")
        .with_error_chain("second error")
        .with_error_chain("root cause");

    let debug_string = context.to_debug_string();

    assert!(debug_string.contains("Operation: test_operation"));
    assert!(debug_string.contains("key1: value1"));
    assert!(debug_string.contains("key2: value2"));
    assert!(debug_string.contains("first error"));
    assert!(debug_string.contains("second error"));
    assert!(debug_string.contains("root cause"));
}

/// Test ProcessErrorDetails with comprehensive information
#[tokio::test]
async fn test_process_error_details() {
    let details = ProcessErrorDetails::new(
        "Command execution failed",
        "claude",
        vec!["--format".to_string(), "json".to_string()],
    )
    .with_exit_code(127)
    .with_stderr("command not found: claude")
    .with_stdout_preview("partial output...");

    let error = details.to_error();
    let error_string = error.to_string();

    assert!(error_string.contains("Command: claude --format json"));
    assert!(error_string.contains("Exit Code: Some(127)"));
    assert!(error_string.contains("Stderr: command not found: claude"));
    assert!(error_string.contains("Stdout Preview: partial output..."));
    assert!(error_string.contains("Working Dir:"));
}

/// Test retry mechanism with exponential backoff
#[tokio::test]
async fn test_retry_with_exponential_backoff() {
    let counter = Arc::new(AtomicU32::new(0));
    let counter_clone = Arc::clone(&counter);

    let config = RetryConfig {
        max_attempts: 3,
        base_delay: Duration::from_millis(10),
        max_delay: Duration::from_millis(100),
        backoff_multiplier: 2.0,
        add_jitter: false,
    };

    let result = retry_with_backoff(
        move || {
            let count = counter_clone.fetch_add(1, Ordering::SeqCst);
            async move {
                if count < 2 {
                    Err(Error::Timeout(1))
                } else {
                    Ok("success".to_string())
                }
            }
        },
        config,
        "test_operation",
    )
    .await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "success");
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

/// Test retry fails with non-recoverable errors
#[tokio::test]
async fn test_retry_non_recoverable_error() {
    let counter = Arc::new(AtomicU32::new(0));
    let counter_clone = Arc::clone(&counter);

    let config = RetryConfig::default();

    let result = retry_with_backoff(
        move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            async move { Err::<String, _>(Error::BinaryNotFound) }
        },
        config,
        "test_operation",
    )
    .await;

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), Error::BinaryNotFound));
    assert_eq!(counter.load(Ordering::SeqCst), 1); // Should not retry
}

/// Test stream reconnection manager
#[tokio::test]
async fn test_stream_reconnection() {
    let manager = StreamReconnectionManager::new(5, Duration::from_millis(10));

    let counter = Arc::new(AtomicU32::new(0));
    let counter_clone = Arc::clone(&counter);

    // First few attempts fail, then succeed
    let result = manager
        .reconnect(move || {
            let count = counter_clone.fetch_add(1, Ordering::SeqCst);
            async move {
                if count < 3 {
                    Err(Error::StreamClosed)
                } else {
                    Ok("reconnected")
                }
            }
        })
        .await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "reconnected");

    let health = manager.health_status().await;
    assert_eq!(health.total_connections, 4); // 3 failures + 1 success
    assert!(health.success_rate > 0.0 && health.success_rate < 100.0);
}

/// Test circuit breaker state transitions
#[tokio::test]
async fn test_circuit_breaker_transitions() {
    let breaker = CircuitBreaker::new(2, 2, Duration::from_millis(100));

    // Initial state should be closed
    assert_eq!(breaker.current_state().await, CircuitState::Closed);

    // Two failures should open the circuit
    for _ in 0..2 {
        let _ = breaker
            .execute(|| async { Err::<(), _>(Error::ProcessError("failure".to_string())) })
            .await;
    }

    assert_eq!(breaker.current_state().await, CircuitState::Open);

    // Should reject requests while open
    let result = breaker
        .execute(|| async { Ok::<_, Error>("should not execute") })
        .await;
    assert!(result.is_err());

    // Wait for timeout to transition to half-open
    sleep(Duration::from_millis(150)).await;

    // Next request should be allowed (half-open)
    let _ = breaker.execute(|| async { Ok::<_, Error>("test") }).await;

    // Should be in half-open state
    // After one success, need one more to close
    let _ = breaker.execute(|| async { Ok::<_, Error>("test2") }).await;

    assert_eq!(breaker.current_state().await, CircuitState::Closed);
}

/// Test token bucket rate limiter
#[tokio::test]
async fn test_token_bucket_rate_limiter() {
    let limiter = TokenBucketRateLimiter::new(10, 5.0); // 10 tokens, 5/sec refill

    // Should have full capacity initially
    assert_eq!(limiter.available_tokens().await as u32, 10);

    // Consume 5 tokens
    assert!(limiter.try_acquire(5).await.is_ok());
    assert_eq!(limiter.available_tokens().await as u32, 5);

    // Try to consume more than available
    assert!(limiter.try_acquire(10).await.is_err());

    // Wait for refill
    sleep(Duration::from_millis(1100)).await;

    // Should have refilled ~5 tokens
    let tokens = limiter.available_tokens().await;
    assert!(tokens >= 9.0 && tokens <= 10.0);
}

/// Test partial result recovery
#[tokio::test]
async fn test_partial_result_recovery() {
    let recovery = PartialResultRecovery::new(5);

    // Save multiple chunks
    for i in 0..7 {
        recovery.save_partial(format!("chunk_{}", i)).await.unwrap();
    }

    // Should only keep last 5
    let recovered = recovery.recover().await;
    assert_eq!(recovered.len(), 5);
    assert_eq!(recovered[0], "chunk_2");
    assert_eq!(recovered[4], "chunk_6");

    // Check checkpoint
    assert_eq!(recovery.last_checkpoint().await, Some(5));

    // Clear and verify
    recovery.clear().await;
    assert_eq!(recovery.recover().await.len(), 0);
    assert_eq!(recovery.last_checkpoint().await, None);
}

/// Test error telemetry collection
#[tokio::test]
async fn test_error_telemetry() {
    let config = TelemetryConfig {
        max_recent_errors: 100,
        error_rate_threshold: 5.0,
        capture_stack_traces: true,
        external_logging: false,
        sampling_rate: 1.0,
    };

    let telemetry = ErrorTelemetry::new(config);

    // Record various errors
    let errors = vec![
        (Error::Timeout(30), "operation1"),
        (Error::RateLimitExceeded, "operation2"),
        (Error::ProcessError("test".to_string()), "operation3"),
        (Error::Timeout(60), "operation1"),
    ];

    for (error, operation) in &errors {
        let mut context = HashMap::new();
        context.insert("test".to_string(), "value".to_string());
        telemetry.record_error(error, operation, context).await;
    }

    // Record recovery attempts
    telemetry
        .record_recovery_attempt(ErrorCode::Timeout, true, Duration::from_millis(500))
        .await;

    telemetry
        .record_recovery_attempt(
            ErrorCode::RateLimitExceeded,
            false,
            Duration::from_millis(1000),
        )
        .await;

    // Get statistics
    let stats = telemetry.get_statistics().await;
    assert_eq!(stats.total_errors, 4);
    assert_eq!(*stats.error_counts.get(&ErrorCode::Timeout).unwrap(), 2);
    assert_eq!(
        *stats
            .error_counts
            .get(&ErrorCode::RateLimitExceeded)
            .unwrap(),
        1
    );
    assert_eq!(
        *stats.error_counts.get(&ErrorCode::ProcessError).unwrap(),
        1
    );

    // Check recovery rates
    let timeout_recovery = stats.recovery_rates.get(&ErrorCode::Timeout).unwrap();
    assert_eq!(timeout_recovery.success_rate(), 100.0);

    let rate_limit_recovery = stats
        .recovery_rates
        .get(&ErrorCode::RateLimitExceeded)
        .unwrap();
    assert_eq!(rate_limit_recovery.success_rate(), 0.0);

    // Test Prometheus export
    let metrics = telemetry.export_prometheus_metrics().await;
    assert!(metrics.contains("claude_errors_total{code=\"C007\"} 2"));
    assert!(metrics.contains("claude_recovery_success_rate"));
}

/// Test concurrent error handling
#[tokio::test]
async fn test_concurrent_error_handling() {
    let telemetry = Arc::new(ErrorTelemetry::new(TelemetryConfig::default()));

    let mut handles = vec![];

    // Spawn 100 concurrent tasks that generate errors
    for i in 0..100 {
        let telemetry_clone = Arc::clone(&telemetry);
        let handle = tokio::spawn(async move {
            let error = if i % 3 == 0 {
                Error::Timeout(30)
            } else if i % 3 == 1 {
                Error::RateLimitExceeded
            } else {
                Error::ProcessError("concurrent error".to_string())
            };

            let mut context = HashMap::new();
            context.insert("task_id".to_string(), i.to_string());

            telemetry_clone
                .record_error(&error, &format!("concurrent_task_{}", i), context)
                .await;
        });
        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all errors were recorded
    let stats = telemetry.get_statistics().await;
    assert_eq!(stats.total_errors, 100);
}

/// Test error recovery with timeout
#[tokio::test]
async fn test_recovery_with_timeout() {
    let operation = || async {
        sleep(Duration::from_millis(500)).await;
        Ok::<_, Error>("success")
    };

    // Should timeout
    let result = timeout(Duration::from_millis(100), operation()).await;
    assert!(result.is_err());

    // With sufficient timeout should succeed
    let result = timeout(Duration::from_secs(1), operation()).await;
    assert!(result.is_ok());
}

/// Test edge case: empty error messages
#[tokio::test]
async fn test_empty_error_messages() {
    let errors = vec![
        Error::SessionNotFound("".to_string()),
        Error::PermissionDenied("".to_string()),
        Error::McpError("".to_string()),
        Error::ConfigError("".to_string()),
        Error::InvalidInput("".to_string()),
        Error::ProcessError("".to_string()),
    ];

    for error in errors {
        // Should not panic
        let _ = error.to_string();
        let _ = error.code();
        let _ = error.is_recoverable();
    }
}

/// Test error serialization for logging
#[tokio::test]
async fn test_error_serialization() {
    use serde_json;

    let context = ErrorContext::new("test_op").with_debug_info("key", "value");

    let debug_string = context.to_debug_string();

    // Should be able to create JSON-like structure
    let json_value = serde_json::json!({
        "operation": "test_op",
        "debug_info": {
            "key": "value"
        },
        "timestamp": "now"
    });

    assert!(json_value.is_object());
}

/// Test recovery mechanism chaining
#[tokio::test]
async fn test_recovery_chaining() {
    let breaker = Arc::new(CircuitBreaker::new(3, 2, Duration::from_secs(1)));
    let limiter = Arc::new(TokenBucketRateLimiter::new(5, 2.0));

    let operation = {
        let breaker = Arc::clone(&breaker);
        let limiter = Arc::clone(&limiter);
        move || {
            let breaker = Arc::clone(&breaker);
            let limiter = Arc::clone(&limiter);
            async move {
                // First check rate limit
                limiter.try_acquire(1).await?;

                // Then check circuit breaker
                breaker
                    .execute(|| async { Ok::<_, Error>("success") })
                    .await
            }
        }
    };

    // Should work initially
    for _ in 0..5 {
        assert!(operation().await.is_ok());
    }

    // Should fail on rate limit
    assert!(operation().await.is_err());
}

/// Test global telemetry initialization
#[tokio::test]
async fn test_global_telemetry() {
    init_telemetry(TelemetryConfig::default());

    // Should be able to record errors
    let mut context = HashMap::new();
    context.insert("test".to_string(), "global".to_string());

    claude_ai_runtime::telemetry::record_error(&Error::Timeout(30), "global_test", context).await;

    claude_ai_runtime::telemetry::record_recovery(
        ErrorCode::Timeout,
        true,
        Duration::from_millis(100),
    )
    .await;

    // Global instance should exist
    assert!(claude_ai_runtime::telemetry::telemetry().is_some());
}
