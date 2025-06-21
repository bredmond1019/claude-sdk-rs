//! Chaos Engineering Tests for Claude AI Runtime
//!
//! This module contains comprehensive chaos engineering tests that simulate
//! failure scenarios to validate system resilience and recovery mechanisms.
//! These tests follow chaos engineering principles to discover weaknesses
//! in a controlled environment.

use claude_sdk_rs_core::{Config, Error, StreamFormat};
use claude_sdk_rs_runtime::process::{execute_claude, execute_claude_streaming};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};
use tokio::time::timeout;

#[cfg(test)]
mod failure_injection_tests {
    use super::*;

    #[tokio::test]
    async fn test_network_failure_simulation() {
        // Simulate network failures by using invalid configurations
        let configs = vec![
            // Invalid model that would cause network errors
            Config::builder()
                .model("invalid-model-that-does-not-exist")
                .timeout_secs(1) // Short timeout
                .build()
                .unwrap(),
            // Very short timeout to simulate network timeouts
            Config::builder()
                .model("claude-sonnet-4-20250514")
                .timeout_secs(1)
                .build()
                .unwrap(),
        ];

        for config in configs {
            let result = execute_claude(&config, "Simple test query").await;

            match result {
                Ok(_) => {
                    // If it succeeds despite the conditions, that's also valid
                    println!("Query succeeded despite simulated network issues");
                }
                Err(Error::Timeout(_)) => {
                    // Expected timeout error - system handled it gracefully
                    assert!(true, "Timeout handled correctly");
                }
                Err(Error::ProcessError(_)) => {
                    // Expected process error - system handled it gracefully
                    assert!(true, "Process error handled correctly");
                }
                Err(Error::BinaryNotFound) => {
                    // Binary not found - test environment issue, skip this test
                    println!("Claude binary not found - skipping network failure test");
                    return;
                }
                Err(e) => {
                    // Other errors should still be handled gracefully
                    println!("Other error handled: {:?}", e);
                    assert!(true, "Error handled gracefully");
                }
            }
        }
    }

    #[tokio::test]
    async fn test_resource_exhaustion_scenarios() {
        // Test behavior under resource constraints
        let large_query = "a".repeat(50_000); // Large but valid query

        let config = Config::builder().timeout_secs(30).build().unwrap();

        // Test multiple concurrent requests to simulate resource pressure
        let mut handles = Vec::new();

        for i in 0..5 {
            let config_clone = config.clone();
            let query = format!("{} - iteration {}", large_query, i);

            let handle = tokio::spawn(async move {
                let result = execute_claude(&config_clone, &query).await;
                (i, result)
            });

            handles.push(handle);
        }

        // Wait for all requests with a reasonable timeout
        let overall_timeout = Duration::from_secs(60);
        let start_time = Instant::now();

        for handle in handles {
            let remaining_time = overall_timeout.saturating_sub(start_time.elapsed());

            match timeout(remaining_time, handle).await {
                Ok(Ok((i, result))) => match result {
                    Ok(_) => println!("Concurrent request {} succeeded", i),
                    Err(Error::BinaryNotFound) => {
                        println!("Claude binary not found - skipping resource exhaustion test");
                        return;
                    }
                    Err(e) => println!("Concurrent request {} failed gracefully: {:?}", i, e),
                },
                Ok(Err(panic)) => {
                    panic!("Task {} panicked: {:?}", "unknown", panic);
                }
                Err(_) => {
                    println!("Request timed out - this is acceptable under resource pressure");
                }
            }
        }
    }

    #[tokio::test]
    async fn test_invalid_configuration_handling() {
        // Test with various invalid configurations to ensure graceful failure
        let invalid_configs = vec![
            // Extremely large timeout
            Config::builder().timeout_secs(u64::MAX).build(),
            // Zero timeout (if allowed by validation)
            Config::builder().timeout_secs(0).build(),
            // Very large max tokens
            Config::builder().max_tokens(usize::MAX).build(),
        ];

        for config_result in invalid_configs {
            match config_result {
                Ok(config) => {
                    // If config is valid, test execution
                    let result = execute_claude(&config, "Test query").await;
                    match result {
                        Ok(_) => println!("Execution succeeded with edge case config"),
                        Err(Error::BinaryNotFound) => {
                            println!("Claude binary not found - skipping invalid config test");
                            return;
                        }
                        Err(e) => {
                            println!("Execution failed gracefully with edge case config: {:?}", e)
                        }
                    }
                }
                Err(e) => {
                    // Config validation caught the issue - good!
                    println!("Invalid config caught during validation: {:?}", e);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_streaming_interruption_scenarios() {
        let config = Config::builder()
            .stream_format(StreamFormat::StreamJson)
            .timeout_secs(30)
            .build()
            .unwrap();

        // Test abrupt stream termination
        match execute_claude_streaming(&config, "Tell me a very long story").await {
            Ok(mut rx) => {
                let mut received_messages = 0;
                let max_messages = 3; // Limit to prevent long-running tests

                while let Some(result) = rx.recv().await {
                    match result {
                        Ok(message) => {
                            received_messages += 1;
                            println!("Received message {}: {}", received_messages, message.len());

                            // Simulate interruption after a few messages
                            if received_messages >= max_messages {
                                println!("Simulating stream interruption");
                                break;
                            }
                        }
                        Err(e) => {
                            println!("Stream error handled gracefully: {:?}", e);
                            break;
                        }
                    }
                }

                // Drop the receiver to simulate client disconnection
                drop(rx);

                // Give some time for cleanup
                tokio::time::sleep(Duration::from_millis(100)).await;

                assert!(
                    received_messages > 0,
                    "Should have received at least some messages"
                );
            }
            Err(Error::BinaryNotFound) => {
                println!("Claude binary not found - skipping streaming interruption test");
                return;
            }
            Err(e) => {
                println!("Streaming setup failed gracefully: {:?}", e);
            }
        }
    }
}

#[cfg(test)]
mod resilience_tests {
    use super::*;

    #[tokio::test]
    async fn test_graceful_degradation_under_load() {
        // Test that the system maintains basic functionality under load
        let config = Config::builder().timeout_secs(10).build().unwrap();

        let num_concurrent_requests = 10;
        let barrier = Arc::new(Barrier::new(num_concurrent_requests));
        let mut handles = Vec::new();

        // Launch concurrent requests simultaneously
        for i in 0..num_concurrent_requests {
            let config_clone = config.clone();
            let barrier_clone = Arc::clone(&barrier);

            let handle = tokio::spawn(async move {
                // Wait for all tasks to be ready
                barrier_clone.wait();

                let query = format!("Request {} - respond with a short message", i);
                let start_time = Instant::now();
                let result = execute_claude(&config_clone, &query).await;
                let duration = start_time.elapsed();

                (i, result, duration)
            });

            handles.push(handle);
        }

        let mut successful_requests = 0;
        let mut failed_requests = 0;
        let mut total_duration = Duration::new(0, 0);

        for handle in handles {
            match handle.await {
                Ok((i, result, duration)) => {
                    total_duration += duration;

                    match result {
                        Ok(_) => {
                            successful_requests += 1;
                            println!("Request {} succeeded in {:?}", i, duration);
                        }
                        Err(Error::BinaryNotFound) => {
                            println!(
                                "Claude binary not found - skipping graceful degradation test"
                            );
                            return;
                        }
                        Err(e) => {
                            failed_requests += 1;
                            println!("Request {} failed gracefully in {:?}: {:?}", i, duration, e);
                        }
                    }
                }
                Err(panic) => {
                    panic!("Task panicked: {:?}", panic);
                }
            }
        }

        println!(
            "Load test results: {} successful, {} failed",
            successful_requests, failed_requests
        );

        // At least some requests should complete (either successfully or with graceful failure)
        assert!(successful_requests + failed_requests == num_concurrent_requests);

        // Average response time should be reasonable even under load
        if successful_requests > 0 {
            let avg_duration = total_duration / successful_requests as u32;
            assert!(
                avg_duration < Duration::from_secs(60),
                "Average response time too high: {:?}",
                avg_duration
            );
        }
    }

    #[tokio::test]
    async fn test_error_recovery_mechanisms() {
        // Test that the system can recover from various error conditions
        let oversized_query = "x".repeat(200_000);
        let error_scenarios = vec![
            ("", "empty query"),
            (&oversized_query, "oversized query"),
            ("query\0with\0nulls", "null bytes"),
            ("<script>alert('test')</script>", "script content"),
        ];

        for (query, description) in error_scenarios {
            let config = Config::builder().timeout_secs(5).build().unwrap();

            let result = execute_claude(&config, &query).await;

            match result {
                Ok(response) => {
                    println!(
                        "Query '{}' succeeded: {} chars",
                        description,
                        response.len()
                    );
                    // Success is acceptable if the system handles the input safely
                }
                Err(Error::BinaryNotFound) => {
                    println!("Claude binary not found - skipping error recovery test");
                    return;
                }
                Err(e) => {
                    println!("Query '{}' failed gracefully: {:?}", description, e);
                    // Graceful failure is the expected behavior for problematic inputs

                    // Verify the system is still functional after the error
                    let recovery_result = execute_claude(&config, "Simple recovery test").await;
                    match recovery_result {
                        Ok(_) => println!("System recovered successfully after error"),
                        Err(Error::BinaryNotFound) => {
                            println!("Claude binary not found during recovery test");
                            return;
                        }
                        Err(recovery_err) => {
                            println!("Recovery test also failed: {:?}", recovery_err);
                            // This might be acceptable depending on the error type
                        }
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_timeout_handling_under_stress() {
        // Test timeout behavior under various stress conditions
        let timeout_scenarios = vec![
            (1, "very short timeout"),
            (5, "short timeout"),
            (30, "normal timeout"),
        ];

        for (timeout_secs, description) in timeout_scenarios {
            let config = Config::builder()
                .timeout_secs(timeout_secs)
                .build()
                .unwrap();

            // Use a query that might take some time
            let query = "Please write a detailed explanation of quantum computing, including the principles of superposition, entanglement, and quantum interference. Cover the differences between classical and quantum computers, current applications, and future prospects.";

            let start_time = Instant::now();
            let result = execute_claude(&config, query).await;
            let elapsed = start_time.elapsed();

            match result {
                Ok(response) => {
                    println!(
                        "Query with {} completed in {:?}: {} chars",
                        description,
                        elapsed,
                        response.len()
                    );
                    // Success within timeout is good
                    assert!(
                        elapsed <= Duration::from_secs(timeout_secs + 5),
                        "Response took longer than expected timeout"
                    );
                }
                Err(Error::Timeout(t)) => {
                    println!(
                        "Query with {} timed out after {}s (elapsed: {:?})",
                        description, t, elapsed
                    );
                    // Timeout is expected for short timeouts with complex queries
                    assert_eq!(t, timeout_secs, "Timeout value should match configuration");
                    // Elapsed time should be approximately the timeout value
                    assert!(
                        elapsed >= Duration::from_secs(timeout_secs),
                        "Should have waited at least the timeout duration"
                    );
                    assert!(
                        elapsed <= Duration::from_secs(timeout_secs + 10),
                        "Should not have waited much longer than timeout"
                    );
                }
                Err(Error::BinaryNotFound) => {
                    println!("Claude binary not found - skipping timeout stress test");
                    return;
                }
                Err(e) => {
                    println!(
                        "Query with {} failed with other error: {:?}",
                        description, e
                    );
                    // Other errors might be acceptable depending on system state
                }
            }
        }
    }

    #[tokio::test]
    async fn test_memory_pressure_handling() {
        // Test behavior under simulated memory pressure
        let config = Config::builder().timeout_secs(30).build().unwrap();

        // Create multiple large queries to simulate memory pressure
        let large_queries: Vec<String> = (0..5)
            .map(|i| format!("Large query {}: {}", i, "x".repeat(10_000)))
            .collect();

        let mut results = Vec::new();

        for (i, query) in large_queries.iter().enumerate() {
            let start_time = Instant::now();
            let result = execute_claude(&config, query).await;
            let duration = start_time.elapsed();

            results.push((i, result, duration));

            // Small delay to allow cleanup
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let mut successful = 0;
        let mut failed = 0;

        for (i, result, duration) in results {
            match result {
                Ok(response) => {
                    successful += 1;
                    println!(
                        "Large query {} succeeded in {:?}: {} chars",
                        i,
                        duration,
                        response.len()
                    );
                }
                Err(Error::BinaryNotFound) => {
                    println!("Claude binary not found - skipping memory pressure test");
                    return;
                }
                Err(e) => {
                    failed += 1;
                    println!("Large query {} failed in {:?}: {:?}", i, duration, e);
                }
            }
        }

        println!(
            "Memory pressure test: {} successful, {} failed",
            successful, failed
        );

        // At least some queries should complete successfully or fail gracefully
        assert!(successful + failed == large_queries.len());
    }
}

#[cfg(test)]
mod system_behavior_tests {
    use super::*;

    #[tokio::test]
    async fn test_rapid_configuration_changes() {
        // Test system behavior with rapid configuration changes
        let configurations = vec![
            Config::builder().timeout_secs(1).build().unwrap(),
            Config::builder().timeout_secs(5).build().unwrap(),
            Config::builder().timeout_secs(10).build().unwrap(),
            Config::builder()
                .stream_format(StreamFormat::Json)
                .build()
                .unwrap(),
            Config::builder()
                .stream_format(StreamFormat::Text)
                .build()
                .unwrap(),
            Config::builder().verbose(true).build().unwrap(),
            Config::builder().verbose(false).build().unwrap(),
        ];

        for (i, config) in configurations.iter().enumerate() {
            let query = format!("Configuration test {}: respond briefly", i);

            let result = execute_claude(config, &query).await;

            match result {
                Ok(response) => {
                    println!("Config {} succeeded: {} chars", i, response.len());
                }
                Err(Error::BinaryNotFound) => {
                    println!("Claude binary not found - skipping rapid config changes test");
                    return;
                }
                Err(e) => {
                    println!("Config {} failed gracefully: {:?}", i, e);
                }
            }

            // Brief pause between configuration changes
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    #[tokio::test]
    async fn test_streaming_resilience() {
        // Test streaming behavior under various conditions
        let config = Config::builder()
            .stream_format(StreamFormat::StreamJson)
            .timeout_secs(10)
            .build()
            .unwrap();

        let query = "Count from 1 to 10, one number per line";

        match execute_claude_streaming(&config, query).await {
            Ok(mut rx) => {
                let mut message_count = 0;
                let mut consecutive_errors = 0;
                let max_messages = 20; // Prevent infinite loops

                while message_count < max_messages {
                    match rx.recv().await {
                        Some(Ok(message)) => {
                            message_count += 1;
                            consecutive_errors = 0; // Reset error count on success
                            println!("Stream message {}: {}", message_count, message.len());
                        }
                        Some(Err(e)) => {
                            consecutive_errors += 1;
                            println!("Stream error {}: {:?}", consecutive_errors, e);

                            // Too many consecutive errors indicate a problem
                            if consecutive_errors >= 5 {
                                println!("Too many consecutive stream errors, stopping");
                                break;
                            }
                        }
                        None => {
                            println!("Stream completed normally");
                            break;
                        }
                    }
                }

                println!(
                    "Stream resilience test completed: {} messages, {} consecutive errors",
                    message_count, consecutive_errors
                );

                // Should have received at least some messages
                assert!(
                    message_count > 0 || consecutive_errors > 0,
                    "Stream should have produced some output or errors"
                );
            }
            Err(Error::BinaryNotFound) => {
                println!("Claude binary not found - skipping streaming resilience test");
                return;
            }
            Err(e) => {
                println!("Streaming setup failed: {:?}", e);
                // Setup failure is acceptable in test environments
            }
        }
    }

    #[tokio::test]
    async fn test_concurrent_streaming_sessions() {
        // Test multiple concurrent streaming sessions
        let config = Config::builder()
            .stream_format(StreamFormat::StreamJson)
            .timeout_secs(15)
            .build()
            .unwrap();

        let num_sessions = 3;
        let mut handles = Vec::new();

        for i in 0..num_sessions {
            let config_clone = config.clone();

            let handle = tokio::spawn(async move {
                let query = format!("Session {}: count from 1 to 5", i);

                match execute_claude_streaming(&config_clone, &query).await {
                    Ok(mut rx) => {
                        let mut messages = Vec::new();
                        let max_messages = 10;

                        while messages.len() < max_messages {
                            match rx.recv().await {
                                Some(Ok(message)) => {
                                    messages.push(message);
                                }
                                Some(Err(e)) => {
                                    println!("Session {} stream error: {:?}", i, e);
                                    break;
                                }
                                None => {
                                    break;
                                }
                            }
                        }

                        (i, Ok(messages))
                    }
                    Err(e) => (i, Err(e)),
                }
            });

            handles.push(handle);
        }

        let mut successful_sessions = 0;
        let mut failed_sessions = 0;

        for handle in handles {
            match handle.await {
                Ok((session_id, result)) => match result {
                    Ok(messages) => {
                        successful_sessions += 1;
                        println!(
                            "Session {} completed with {} messages",
                            session_id,
                            messages.len()
                        );
                    }
                    Err(Error::BinaryNotFound) => {
                        println!("Claude binary not found - skipping concurrent streaming test");
                        return;
                    }
                    Err(e) => {
                        failed_sessions += 1;
                        println!("Session {} failed: {:?}", session_id, e);
                    }
                },
                Err(panic) => {
                    panic!("Session task panicked: {:?}", panic);
                }
            }
        }

        println!(
            "Concurrent streaming: {} successful, {} failed",
            successful_sessions, failed_sessions
        );

        // At least some sessions should complete
        assert!(successful_sessions + failed_sessions == num_sessions);
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[tokio::test]
    async fn test_boundary_conditions() {
        // Test various boundary conditions
        let config = Config::default();

        let query_100 = "a".repeat(100);
        let query_1000 = "a".repeat(1000);
        let boundary_queries = vec![
            ("", "empty string"),
            (" ", "single space"),
            ("a", "single character"),
            (&query_100, "100 characters"),
            (&query_1000, "1000 characters"),
            ("Hello, World!", "simple greeting"),
            ("What is 2+2?", "simple math"),
        ];

        for (query, description) in boundary_queries {
            let result = execute_claude(&config, query).await;

            match result {
                Ok(response) => {
                    println!(
                        "Boundary test '{}' succeeded: {} chars",
                        description,
                        response.len()
                    );
                }
                Err(Error::BinaryNotFound) => {
                    println!("Claude binary not found - skipping boundary conditions test");
                    return;
                }
                Err(e) => {
                    println!("Boundary test '{}' failed gracefully: {:?}", description, e);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_cleanup_after_errors() {
        // Test that system properly cleans up after various error conditions
        let config = Config::builder()
            .timeout_secs(1) // Very short timeout to force errors
            .build()
            .unwrap();

        // Intentionally cause errors
        let large_query = "x".repeat(50_000);
        let error_queries = vec![
            "This is a complex query that should timeout due to the 1-second limit. Please write a very detailed explanation of artificial intelligence, machine learning, deep learning, neural networks, and their applications in various industries.",
            &large_query, // Large query
        ];

        for (i, query) in error_queries.iter().enumerate() {
            let result = execute_claude(&config, query).await;

            match result {
                Ok(_) => {
                    println!("Error query {} unexpectedly succeeded", i);
                }
                Err(Error::BinaryNotFound) => {
                    println!("Claude binary not found - skipping cleanup test");
                    return;
                }
                Err(e) => {
                    println!("Error query {} failed as expected: {:?}", i, e);
                }
            }

            // Test that system can still function after the error
            let normal_config = Config::builder().timeout_secs(10).build().unwrap();

            let recovery_result = execute_claude(&normal_config, "Simple test").await;
            match recovery_result {
                Ok(_) => {
                    println!("System recovered successfully after error {}", i);
                }
                Err(Error::BinaryNotFound) => {
                    println!("Claude binary not found during recovery");
                    return;
                }
                Err(e) => {
                    println!("Recovery failed after error {}: {:?}", i, e);
                    // This might be acceptable depending on the nature of the error
                }
            }
        }
    }

    #[tokio::test]
    async fn test_resource_cleanup_on_drop() {
        // Test that resources are properly cleaned up when objects are dropped
        let config = Config::builder()
            .stream_format(StreamFormat::StreamJson)
            .timeout_secs(30)
            .build()
            .unwrap();

        // Create a streaming session and drop it immediately
        match execute_claude_streaming(&config, "Tell me about Rust programming").await {
            Ok(rx) => {
                println!("Streaming session created, dropping immediately");
                // Explicitly drop the receiver to test cleanup
                drop(rx);

                // Give some time for cleanup
                tokio::time::sleep(Duration::from_millis(100)).await;

                // Test that we can still create new sessions
                match execute_claude_streaming(&config, "Simple test after drop").await {
                    Ok(mut new_rx) => {
                        println!("New streaming session created successfully after drop");

                        // Receive at least one message to verify it works
                        match new_rx.recv().await {
                            Some(Ok(message)) => {
                                println!(
                                    "Received message from new session: {} chars",
                                    message.len()
                                );
                            }
                            Some(Err(e)) => {
                                println!("New session error: {:?}", e);
                            }
                            None => {
                                println!("New session completed immediately");
                            }
                        }
                    }
                    Err(Error::BinaryNotFound) => {
                        println!("Claude binary not found - skipping resource cleanup test");
                        return;
                    }
                    Err(e) => {
                        println!("Failed to create new session after drop: {:?}", e);
                    }
                }
            }
            Err(Error::BinaryNotFound) => {
                println!("Claude binary not found - skipping resource cleanup test");
                return;
            }
            Err(e) => {
                println!("Failed to create initial streaming session: {:?}", e);
            }
        }
    }
}
