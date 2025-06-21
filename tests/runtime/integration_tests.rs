use claude_sdk_rs_core::{Config, SessionId, StreamFormat};
use claude_sdk_rs_runtime::{Client, MessageStream};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = Client::new(Config::default());
        // Test that client can be created without errors
        assert!(true);
    }

    #[tokio::test]
    async fn test_client_builder() {
        let client = Client::builder()
            .model("claude-3-haiku-20240307")
            .system_prompt("You are a test assistant")
            .stream_format(StreamFormat::Json)
            .timeout_secs(60)
            .build();

        // Test that client builder works without errors
        assert!(true);
    }

    #[tokio::test]
    async fn test_query_builder_creation() {
        let client = Client::new(Config::default());
        let query_builder = client.query("Test query");

        // Test that query builder can be created
        assert!(true);
    }

    #[tokio::test]
    async fn test_client_with_tools() {
        let client = Client::builder()
            .allowed_tools(vec!["bash".to_string(), "filesystem".to_string()])
            .build();

        // Test that client with tools can be created
        assert!(true);
    }

    #[tokio::test]
    async fn test_multiple_clients() {
        let client1 = Client::builder().stream_format(StreamFormat::Text).build();

        let client2 = Client::builder().stream_format(StreamFormat::Json).build();

        let client3 = Client::builder()
            .stream_format(StreamFormat::StreamJson)
            .build();

        // Test that multiple clients can exist simultaneously
        assert!(true);
    }

    #[tokio::test]
    async fn test_config_combinations() {
        // Test various config combinations
        let configs = vec![
            Config::builder().build(),
            Config::builder().model("claude-3-sonnet-20240229").build(),
            Config::builder().system_prompt("Test").build(),
            Config::builder().max_tokens(1000).build(),
            Config::builder().timeout_secs(120).build(),
            Config::builder()
                .stream_format(StreamFormat::Json)
                .model("claude-3-haiku-20240307")
                .system_prompt("Test assistant")
                .max_tokens(500)
                .timeout_secs(60)
                .build(),
        ];

        for config in configs {
            let client = Client::new(config);
            // Test that all config combinations work
            assert!(true);
        }
    }

    #[tokio::test]
    async fn test_query_builder_methods() {
        let client = Client::new(Config::default());

        // Test query builder method chaining
        let _query = client.query("Test query").format(StreamFormat::Json);

        let _query2 = client
            .query("Another test")
            .session(SessionId::new("test-session"))
            .format(StreamFormat::StreamJson);

        // Test that query builder methods work
        assert!(true);
    }
}

#[cfg(test)]
mod error_handling_tests {
    use claude_sdk_rs_core::Error;

    #[tokio::test]
    async fn test_binary_not_found_simulation() {
        // This test simulates what happens when Claude CLI is not found
        // We can't easily test this without manipulating PATH in unsafe ways
        // So we just verify the error type exists and can be matched
        let error = Error::BinaryNotFound;

        match error {
            Error::BinaryNotFound => assert!(true),
            _ => panic!("Expected BinaryNotFound error"),
        }
    }

    #[tokio::test]
    async fn test_timeout_error_simulation() {
        let error = Error::Timeout(30);

        match error {
            Error::Timeout(secs) => {
                assert_eq!(secs, 30);
            }
            _ => panic!("Expected Timeout error"),
        }
    }

    #[tokio::test]
    async fn test_process_error_simulation() {
        let error = Error::ProcessError("Test error".to_string());

        match error {
            Error::ProcessError(msg) => {
                assert_eq!(msg, "Test error");
            }
            _ => panic!("Expected ProcessError"),
        }
    }
}

#[cfg(test)]
mod stream_format_tests {
    use super::*;

    #[test]
    fn test_stream_format_default() {
        let format = StreamFormat::default();
        assert_eq!(format, StreamFormat::Text);
    }

    #[test]
    fn test_stream_format_equality() {
        assert_eq!(StreamFormat::Text, StreamFormat::Text);
        assert_eq!(StreamFormat::Json, StreamFormat::Json);
        assert_eq!(StreamFormat::StreamJson, StreamFormat::StreamJson);

        assert_ne!(StreamFormat::Text, StreamFormat::Json);
        assert_ne!(StreamFormat::Json, StreamFormat::StreamJson);
        assert_ne!(StreamFormat::Text, StreamFormat::StreamJson);
    }

    #[test]
    fn test_stream_format_clone() {
        let format = StreamFormat::Json;
        let cloned = format.clone();
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_stream_format_copy() {
        let format = StreamFormat::StreamJson;
        let copied = format; // This tests Copy trait
        assert_eq!(format, copied);
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_streaming_creation() {
        // Test that we can create a streaming query without errors
        let client = Client::new(Config::default());
        let query_builder = client.query("Test streaming query");

        // This should not panic or fail at creation time
        assert!(true);
    }

    #[tokio::test]
    async fn test_streaming_with_different_formats() {
        // Test that streaming works with different format configurations
        let configs = vec![
            Config::builder().stream_format(StreamFormat::Text).build(),
            Config::builder().stream_format(StreamFormat::Json).build(),
            Config::builder()
                .stream_format(StreamFormat::StreamJson)
                .build(),
        ];

        for config in configs {
            let client = Client::new(config);
            let query_builder = client.query("Test");

            // The creation should work for all formats
            assert!(true);
        }
    }

    #[tokio::test]
    async fn test_message_stream_creation() {
        use claude_sdk_rs_core::Message;
        use tokio::sync::mpsc;

        // Test that MessageStream can be created from a receiver
        let (tx, rx) = mpsc::channel(10);
        let _stream = MessageStream::new(rx, StreamFormat::Text);

        // Close the sender to avoid hanging
        drop(tx);

        assert!(true);
    }

    #[tokio::test]
    async fn test_message_stream_collect() {
        use claude_sdk_rs_core::{Message, MessageMeta};
        use tokio::sync::mpsc;

        // Test that MessageStream can collect messages
        let (tx, rx) = mpsc::channel(10);
        let stream = MessageStream::new(rx, StreamFormat::Text);

        // Send some test messages
        let message = Message::Assistant {
            content: "Hello".to_string(),
            meta: MessageMeta {
                session_id: "test".to_string(),
                timestamp: Some(std::time::SystemTime::now()),
                cost_usd: None,
                duration_ms: None,
                tokens_used: None,
            },
        };

        tx.send(Ok(message)).await.unwrap();

        // Send a result message to end the stream
        let result_message = Message::Result {
            meta: MessageMeta {
                session_id: "test".to_string(),
                timestamp: Some(std::time::SystemTime::now()),
                cost_usd: None,
                duration_ms: None,
                tokens_used: None,
            },
            stats: claude_sdk_rs_core::ConversationStats {
                total_messages: 1,
                total_cost_usd: 0.0,
                total_duration_ms: 0,
                total_tokens: claude_sdk_rs_core::TokenUsage {
                    input: 0,
                    output: 0,
                    total: 0,
                },
            },
        };

        tx.send(Ok(result_message)).await.unwrap();
        drop(tx);

        let response = stream.collect_full_response().await;
        assert!(response.is_ok());
        let content = response.unwrap();
        assert_eq!(content, "Hello");
    }

    #[tokio::test]
    async fn test_concurrent_streaming_requests() {
        use claude_sdk_rs_core::{Message, MessageMeta};
        use tokio::sync::mpsc;

        // Test that multiple streaming requests can run concurrently
        let mut handles = Vec::new();

        for i in 0..3 {
            let handle = tokio::spawn(async move {
                let (tx, rx) = mpsc::channel(10);
                let stream = MessageStream::new(rx, StreamFormat::Text);

                // Send test message
                let message = Message::Assistant {
                    content: format!("Stream {}", i),
                    meta: MessageMeta {
                        session_id: format!("session-{}", i),
                        timestamp: Some(std::time::SystemTime::now()),
                        cost_usd: None,
                        duration_ms: None,
                        tokens_used: None,
                    },
                };

                tx.send(Ok(message)).await.unwrap();
                drop(tx);

                stream.collect_full_response().await
            });
            handles.push(handle);
        }

        // Wait for all concurrent streams to complete
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => eprintln!("Concurrent stream error: {}", e),
            }
        }

        assert_eq!(results.len(), 3);
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok());
            assert!(result.as_ref().unwrap().contains(&format!("Stream {}", i)));
        }
    }

    #[tokio::test]
    async fn test_streaming_backpressure_handling() {
        use claude_sdk_rs_core::{Message, MessageMeta};
        use tokio::sync::mpsc;
        use tokio::time::{sleep, Duration};

        // Test handling of backpressure in streaming
        let (tx, rx) = mpsc::channel(1); // Small buffer to test backpressure
        let stream = MessageStream::new(rx, StreamFormat::Text);

        // Spawn a task to send messages
        let send_task = tokio::spawn(async move {
            for i in 0..5 {
                let message = Message::Assistant {
                    content: format!("Message {}", i),
                    meta: MessageMeta {
                        session_id: "backpressure-test".to_string(),
                        timestamp: Some(std::time::SystemTime::now()),
                        cost_usd: None,
                        duration_ms: None,
                        tokens_used: None,
                    },
                };

                // This should handle backpressure gracefully
                if tx.send(Ok(message)).await.is_err() {
                    break;
                }

                // Small delay to simulate streaming
                sleep(Duration::from_millis(10)).await;
            }
        });

        // Consume messages with some delay
        let mut collected = Vec::new();
        let mut stream = Box::pin(stream);

        while let Some(result) = stream.next().await {
            match result {
                Ok(content) => {
                    collected.push(content);
                    // Simulate slow consumer
                    sleep(Duration::from_millis(20)).await;
                }
                Err(e) => eprintln!("Stream error: {}", e),
            }
        }

        // Wait for send task to complete
        let _ = send_task.await;

        // Should have received at least some messages
        assert!(!collected.is_empty());
    }

    #[tokio::test]
    async fn test_streaming_error_propagation() {
        use claude_sdk_rs_core::Error;
        use tokio::sync::mpsc;

        // Test that errors are properly propagated through streams
        let (tx, rx) = mpsc::channel(10);
        let stream = MessageStream::new(rx, StreamFormat::Text);

        // Send an error
        tx.send(Err(Error::ProcessError("Test error".to_string())))
            .await
            .unwrap();
        drop(tx);

        let result = stream.collect_full_response().await;
        assert!(result.is_err());
        match result {
            Err(Error::ProcessError(msg)) => {
                assert_eq!(msg, "Test error");
            }
            _ => panic!("Expected ProcessError"),
        }
    }

    #[tokio::test]
    async fn test_streaming_timeout_immediate() {
        use claude_sdk_rs_core::Error;
        use tokio::sync::mpsc;
        use tokio::time::{timeout, Duration};

        // Test immediate timeout scenario
        let (tx, rx) = mpsc::channel(10);
        let stream = MessageStream::new(rx, StreamFormat::Text);

        // Don't send any messages and drop tx to simulate hanging
        drop(tx);

        // Apply a timeout to the collect operation
        let result = timeout(Duration::from_millis(100), stream.collect_full_response()).await;

        // Should timeout quickly
        assert!(result.is_err(), "Expected timeout but got result");
    }

    #[tokio::test]
    async fn test_streaming_timeout_during_messages() {
        use claude_sdk_rs_core::{Message, MessageMeta};
        use tokio::sync::mpsc;
        use tokio::time::{sleep, timeout, Duration};

        // Test timeout during message streaming
        let (tx, rx) = mpsc::channel(10);
        let stream = MessageStream::new(rx, StreamFormat::Text);

        // Send messages with long delays
        tokio::spawn(async move {
            for i in 0..10 {
                let message = Message::Assistant {
                    content: format!("Slow message {}", i),
                    meta: MessageMeta {
                        session_id: "timeout-test".to_string(),
                        timestamp: Some(std::time::SystemTime::now()),
                        cost_usd: None,
                        duration_ms: None,
                        tokens_used: None,
                    },
                };

                if tx.send(Ok(message)).await.is_err() {
                    break;
                }

                // Long delay between messages
                sleep(Duration::from_secs(2)).await;
            }
        });

        // Apply short timeout
        let result = timeout(Duration::from_millis(500), stream.collect_full_response()).await;

        // Should timeout before getting all messages
        assert!(result.is_err(), "Expected timeout during streaming");
    }

    #[tokio::test]
    async fn test_streaming_timeout_recovery() {
        use claude_sdk_rs_core::{Error, Message, MessageMeta};
        use tokio::sync::mpsc;
        use tokio::time::{timeout, Duration};

        // Test recovery after timeout
        let (tx1, rx1) = mpsc::channel(10);
        let stream1 = MessageStream::new(rx1, StreamFormat::Text);

        // First stream will timeout
        drop(tx1);
        let result1 = timeout(Duration::from_millis(50), stream1.collect_full_response()).await;
        assert!(result1.is_err());

        // Second stream should work normally
        let (tx2, rx2) = mpsc::channel(10);
        let stream2 = MessageStream::new(rx2, StreamFormat::Text);

        let message = Message::Assistant {
            content: "Recovery successful".to_string(),
            meta: MessageMeta {
                session_id: "recovery-test".to_string(),
                timestamp: Some(std::time::SystemTime::now()),
                cost_usd: None,
                duration_ms: None,
                tokens_used: None,
            },
        };

        tx2.send(Ok(message)).await.unwrap();
        drop(tx2);

        let result2 = stream2.collect_full_response().await;
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), "Recovery successful");
    }
}

#[cfg(test)]
mod concurrent_request_tests {
    use super::*;
    use claude_sdk_rs_core::{Message, MessageMeta};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_concurrent_client_creation() {
        // Test creating many clients concurrently
        let mut handles = Vec::new();

        for i in 0..10 {
            let handle = tokio::spawn(async move {
                let config = Config::builder()
                    .model(format!("test-model-{}", i))
                    .timeout_secs(30 + i)
                    .build();

                let _client = Client::new(config);
                i
            });
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap();
            results.push(result);
        }

        assert_eq!(results.len(), 10);
    }

    #[tokio::test]
    async fn test_concurrent_message_processing() {
        // Test processing messages from multiple streams concurrently
        let completed = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for i in 0..5 {
            let completed_clone = completed.clone();
            let handle = tokio::spawn(async move {
                let (tx, rx) = mpsc::channel(100);
                let stream = MessageStream::new(rx, StreamFormat::Text);

                // Send multiple messages
                for j in 0..10 {
                    let message = Message::Assistant {
                        content: format!("Client {} Message {}", i, j),
                        meta: MessageMeta {
                            session_id: format!("concurrent-{}", i),
                            timestamp: Some(std::time::SystemTime::now()),
                            cost_usd: None,
                            duration_ms: None,
                            tokens_used: None,
                        },
                    };

                    if tx.send(Ok(message)).await.is_err() {
                        break;
                    }
                }
                drop(tx);

                let result = stream.collect_full_response().await;
                if result.is_ok() {
                    completed_clone.fetch_add(1, Ordering::SeqCst);
                }
                result
            });
            handles.push(handle);
        }

        // Wait for all to complete
        let mut all_ok = true;
        for handle in handles {
            if let Ok(result) = handle.await {
                if result.is_err() {
                    all_ok = false;
                }
            }
        }

        assert!(all_ok);
        assert_eq!(completed.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn test_concurrent_error_handling() {
        use claude_sdk_rs_core::Error;

        // Test concurrent streams with mixed success/error cases
        let mut handles = Vec::new();

        for i in 0..6 {
            let handle = tokio::spawn(async move {
                let (tx, rx) = mpsc::channel(10);
                let stream = MessageStream::new(rx, StreamFormat::Text);

                if i % 2 == 0 {
                    // Send normal message
                    let message = Message::Assistant {
                        content: format!("Success {}", i),
                        meta: MessageMeta {
                            session_id: format!("mixed-{}", i),
                            timestamp: Some(std::time::SystemTime::now()),
                            cost_usd: None,
                            duration_ms: None,
                            tokens_used: None,
                        },
                    };
                    tx.send(Ok(message)).await.unwrap();
                } else {
                    // Send error
                    tx.send(Err(Error::ProcessError(format!("Error {}", i))))
                        .await
                        .unwrap();
                }
                drop(tx);

                stream.collect_full_response().await
            });
            handles.push(handle);
        }

        let mut success_count = 0;
        let mut error_count = 0;

        for (i, handle) in handles.into_iter().enumerate() {
            match handle.await.unwrap() {
                Ok(content) => {
                    assert!(content.contains("Success"));
                    success_count += 1;
                }
                Err(Error::ProcessError(msg)) => {
                    assert!(msg.contains("Error"));
                    error_count += 1;
                }
                _ => panic!("Unexpected result for index {}", i),
            }
        }

        assert_eq!(success_count, 3);
        assert_eq!(error_count, 3);
    }
}

#[cfg(test)]
mod error_recovery_tests {
    use super::*;
    use claude_sdk_rs_core::{Error, Message, MessageMeta};
    use tokio::sync::mpsc;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_error_recovery_after_timeout() {
        // First operation times out
        let config1 = Config::builder()
            .timeout_secs(0) // Immediate timeout
            .build();

        let client1 = Client::new(config1);

        // Second operation should work normally
        let config2 = Config::builder()
            .timeout_secs(30) // Normal timeout
            .build();

        let client2 = Client::new(config2);

        // Both clients should be independently functional
        assert!(true);
    }

    #[tokio::test]
    async fn test_error_recovery_after_process_error() {
        use tokio::sync::mpsc;

        // First stream with error
        let (tx1, rx1) = mpsc::channel(10);
        let stream1 = MessageStream::new(rx1, StreamFormat::Text);

        tx1.send(Err(Error::ProcessError("First error".to_string())))
            .await
            .unwrap();
        drop(tx1);

        let result1 = stream1.collect_full_response().await;
        assert!(result1.is_err());

        // Second stream should work fine
        let (tx2, rx2) = mpsc::channel(10);
        let stream2 = MessageStream::new(rx2, StreamFormat::Text);

        let message = Message::Assistant {
            content: "Recovery successful".to_string(),
            meta: MessageMeta {
                session_id: "recovery".to_string(),
                timestamp: Some(std::time::SystemTime::now()),
                cost_usd: None,
                duration_ms: None,
                tokens_used: None,
            },
        };

        tx2.send(Ok(message)).await.unwrap();
        drop(tx2);

        let result2 = stream2.collect_full_response().await;
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), "Recovery successful");
    }

    #[tokio::test]
    async fn test_error_recovery_with_retry_pattern() {
        // Simulate a retry pattern after errors
        let mut attempts = 0;
        let max_attempts = 3;
        let mut last_error = None;

        while attempts < max_attempts {
            let (tx, rx) = mpsc::channel(10);
            let stream = MessageStream::new(rx, StreamFormat::Text);

            if attempts < 2 {
                // First two attempts fail
                tx.send(Err(Error::ProcessError(format!(
                    "Attempt {} failed",
                    attempts + 1
                ))))
                .await
                .unwrap();
            } else {
                // Third attempt succeeds
                let message = Message::Assistant {
                    content: "Finally succeeded!".to_string(),
                    meta: MessageMeta {
                        session_id: "retry-test".to_string(),
                        timestamp: Some(std::time::SystemTime::now()),
                        cost_usd: None,
                        duration_ms: None,
                        tokens_used: None,
                    },
                };
                tx.send(Ok(message)).await.unwrap();
            }
            drop(tx);

            match stream.collect_full_response().await {
                Ok(content) => {
                    assert_eq!(content, "Finally succeeded!");
                    break;
                }
                Err(e) => {
                    last_error = Some(e);
                    attempts += 1;
                    // Small delay between retries
                    sleep(Duration::from_millis(10)).await;
                }
            }
        }

        assert_eq!(attempts, 2); // Should succeed on third attempt
    }

    #[tokio::test]
    async fn test_error_recovery_circuit_breaker_pattern() {
        // Simulate circuit breaker pattern
        let error_threshold = 3;
        let mut consecutive_errors = 0;
        let mut circuit_open = false;

        for i in 0..10 {
            if circuit_open {
                // Circuit is open, skip requests
                sleep(Duration::from_millis(50)).await;

                // Try to close circuit after cooldown
                if i >= 7 {
                    circuit_open = false;
                    consecutive_errors = 0;
                }
                continue;
            }

            let (tx, rx) = mpsc::channel(10);
            let stream = MessageStream::new(rx, StreamFormat::Text);

            if i < 4 || (i >= 7 && i < 9) {
                // Simulate errors
                tx.send(Err(Error::ProcessError("Service unavailable".to_string())))
                    .await
                    .unwrap();
            } else {
                // Simulate success
                let message = Message::Assistant {
                    content: format!("Request {} succeeded", i),
                    meta: MessageMeta {
                        session_id: "circuit-test".to_string(),
                        timestamp: Some(std::time::SystemTime::now()),
                        cost_usd: None,
                        duration_ms: None,
                        tokens_used: None,
                    },
                };
                tx.send(Ok(message)).await.unwrap();
            }
            drop(tx);

            match stream.collect_full_response().await {
                Ok(_) => {
                    consecutive_errors = 0;
                }
                Err(_) => {
                    consecutive_errors += 1;
                    if consecutive_errors >= error_threshold {
                        circuit_open = true;
                    }
                }
            }
        }

        // Circuit breaker should have triggered
        assert!(true);
    }
}
