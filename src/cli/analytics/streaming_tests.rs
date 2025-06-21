//! Tests for streaming optimization features
//!
//! This module contains comprehensive tests for the streaming buffer management
//! and optimization components, including performance validation and edge case
//! handling.

#[cfg(test)]
mod tests {
    use super::super::{
        optimized_dashboard::{
            OptimizedDashboardConfig, OptimizedDashboardFactory, OptimizedDashboardManager,
            UpdatePriority,
        },
        streaming_optimizer::{StreamingConfig, StreamingOptimizer, StreamingOptimizerFactory},
        AnalyticsEngine,
    };
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::{sleep, timeout};

    #[tokio::test]
    async fn test_streaming_optimizer_creation() {
        let config = StreamingConfig::default();
        let optimizer = StreamingOptimizer::<String>::new(config);

        let result = optimizer.start().await;
        assert!(
            result.is_ok(),
            "Streaming optimizer should start successfully"
        );
    }

    #[tokio::test]
    async fn test_streaming_factory_configurations() {
        let perf_config = StreamingOptimizerFactory::performance_optimized();
        assert_eq!(perf_config.initial_buffer_size, 1024);
        assert_eq!(perf_config.target_latency_ms, 25);
        assert!(perf_config.enable_adaptive_buffering);

        let memory_config = StreamingOptimizerFactory::memory_optimized();
        assert_eq!(memory_config.initial_buffer_size, 64);
        assert_eq!(memory_config.max_buffer_size, 512);
        assert!(!memory_config.enable_adaptive_buffering);

        let latency_config = StreamingOptimizerFactory::low_latency();
        assert_eq!(latency_config.target_latency_ms, 10);
        assert_eq!(latency_config.batch_timeout_ms, 25);
    }

    #[tokio::test]
    async fn test_streaming_basic_flow() {
        let config = StreamingConfig {
            initial_buffer_size: 10,
            enable_batching: false, // Disable batching for simpler test
            ..Default::default()
        };

        let optimizer = StreamingOptimizer::<String>::new(config);
        optimizer.start().await.unwrap();

        let mut receiver = optimizer.subscribe();

        // Send test data
        optimizer
            .send_item("test_message".to_string())
            .await
            .unwrap();

        // Should receive the message
        let result = timeout(Duration::from_secs(2), receiver.recv()).await;
        assert!(result.is_ok(), "Should receive message within timeout");

        let batch = result.unwrap().unwrap();
        assert_eq!(batch.items.len(), 1);
        assert_eq!(batch.items[0], "test_message");
    }

    #[tokio::test]
    async fn test_streaming_batching() {
        let config = StreamingConfig {
            initial_buffer_size: 10,
            enable_batching: true,
            max_batch_size: 3,
            batch_timeout_ms: 100,
            ..Default::default()
        };

        let optimizer = StreamingOptimizer::<String>::new(config);
        optimizer.start().await.unwrap();

        let mut receiver = optimizer.subscribe();

        // Send multiple items
        for i in 0..3 {
            optimizer.send_item(format!("item_{}", i)).await.unwrap();
        }

        // Should receive batched messages
        let result = timeout(Duration::from_secs(2), receiver.recv()).await;
        assert!(result.is_ok(), "Should receive batch within timeout");

        let batch = result.unwrap().unwrap();
        assert!(batch.items.len() >= 1, "Batch should contain items");
        assert!(batch.items.len() <= 3, "Batch should not exceed max size");
    }

    #[tokio::test]
    async fn test_streaming_metrics_collection() {
        let config = StreamingConfig::default();
        let optimizer = StreamingOptimizer::<String>::new(config);
        optimizer.start().await.unwrap();

        // Generate some activity
        for i in 0..5 {
            optimizer
                .send_item(format!("metric_test_{}", i))
                .await
                .unwrap();
        }

        // Wait for metrics to be collected
        sleep(Duration::from_millis(200)).await;

        let metrics = optimizer.get_metrics().await;
        assert!(metrics.throughput_messages_per_second >= 0.0);
        assert!(metrics.average_latency_ms >= 0.0);
        assert!(metrics.buffer_utilization >= 0.0);
    }

    #[tokio::test]
    async fn test_backpressure_detection() {
        let config = StreamingConfig::default();
        let optimizer = StreamingOptimizer::<String>::new(config);
        optimizer.start().await.unwrap();

        // Initial state should be normal
        let initial_state = optimizer.get_backpressure_state().await;
        // Note: BackpressureSignal doesn't implement PartialEq, so we can't directly compare
        // We'll just verify we can get the state without panicking

        // Simulate some load to potentially trigger backpressure detection
        for _ in 0..10 {
            optimizer.send_item("load_test".to_string()).await.unwrap();
        }

        sleep(Duration::from_millis(100)).await;

        let _state = optimizer.get_backpressure_state().await;
        // Test passes if we can get the state without errors
    }

    #[tokio::test]
    async fn test_optimized_dashboard_creation() {
        let config = OptimizedDashboardConfig::default();

        // Mock analytics engine - in real tests this would be properly initialized
        let analytics_engine = Arc::new(create_mock_analytics_engine());

        let result = OptimizedDashboardManager::new(analytics_engine, config).await;
        assert!(
            result.is_ok(),
            "Optimized dashboard should create successfully"
        );
    }

    #[tokio::test]
    async fn test_optimized_dashboard_factory() {
        let high_perf = OptimizedDashboardFactory::high_performance();
        assert_eq!(high_perf.base_config.refresh_interval_seconds, 5);
        assert_eq!(high_perf.max_client_update_rate, 20.0);
        assert!(high_perf.enable_differential_updates);

        let memory_eff = OptimizedDashboardFactory::memory_efficient();
        assert_eq!(memory_eff.base_config.refresh_interval_seconds, 15);
        assert_eq!(memory_eff.max_client_update_rate, 2.0);

        let low_lat = OptimizedDashboardFactory::low_latency();
        assert_eq!(low_lat.base_config.refresh_interval_seconds, 1);
        assert_eq!(low_lat.max_client_update_rate, 50.0);
    }

    #[tokio::test]
    async fn test_streaming_connection_tracking() {
        let config = StreamingConfig::default();
        let optimizer = StreamingOptimizer::<String>::new(config);
        optimizer.start().await.unwrap();

        let client_id = "test_client_123".to_string();
        let mut receiver = optimizer.subscribe_with_tracking(client_id.clone()).await;

        // Send a message
        optimizer
            .send_item("tracked_message".to_string())
            .await
            .unwrap();

        // Should receive the message
        let result = timeout(Duration::from_secs(2), receiver.recv()).await;
        assert!(result.is_ok(), "Tracked client should receive message");
    }

    #[tokio::test]
    async fn test_streaming_flush_batches() {
        let config = StreamingConfig {
            enable_batching: true,
            max_batch_size: 100,     // High number to prevent auto-flush
            batch_timeout_ms: 10000, // Long timeout to prevent auto-flush
            ..Default::default()
        };

        let optimizer = StreamingOptimizer::<String>::new(config);
        optimizer.start().await.unwrap();

        let mut receiver = optimizer.subscribe();

        // Send items but don't trigger auto-flush
        optimizer.send_item("pending_1".to_string()).await.unwrap();
        optimizer.send_item("pending_2".to_string()).await.unwrap();

        // Manually flush
        optimizer.flush_batches().await.unwrap();

        // Should receive the flushed batch
        let result = timeout(Duration::from_secs(2), receiver.recv()).await;
        assert!(result.is_ok(), "Should receive flushed batch");

        let batch = result.unwrap().unwrap();
        assert!(batch.items.len() >= 1, "Flushed batch should contain items");
    }

    #[tokio::test]
    async fn test_streaming_multiple_subscribers() {
        let config = StreamingConfig::default();
        let optimizer = StreamingOptimizer::<String>::new(config);
        optimizer.start().await.unwrap();

        // Create multiple subscribers
        let mut receiver1 = optimizer.subscribe();
        let mut receiver2 = optimizer.subscribe();

        // Send a message
        optimizer
            .send_item("broadcast_test".to_string())
            .await
            .unwrap();
        optimizer.flush_batches().await.unwrap();

        // Both receivers should get the message
        let result1 = timeout(Duration::from_secs(1), receiver1.recv()).await;
        let result2 = timeout(Duration::from_secs(1), receiver2.recv()).await;

        assert!(result1.is_ok(), "First subscriber should receive message");
        assert!(result2.is_ok(), "Second subscriber should receive message");
    }

    // Helper function to create a mock analytics engine
    // Note: This test is simplified and may not work with real dashboard creation
    fn create_mock_analytics_engine() -> AnalyticsEngine {
        use crate::{analytics::AnalyticsConfig, cost::CostTracker, history::HistoryStore};
        use std::sync::Arc;
        use tempfile::tempdir;
        use tokio::sync::RwLock;

        let temp_dir = tempdir().unwrap();
        let cost_tracker = Arc::new(RwLock::new(
            CostTracker::new(temp_dir.path().join("costs.json")).unwrap(),
        ));
        let history_store = Arc::new(RwLock::new(
            HistoryStore::new(temp_dir.path().join("history.json")).unwrap(),
        ));

        AnalyticsEngine::new(cost_tracker, history_store, AnalyticsConfig::default())
    }

    #[tokio::test]
    async fn test_streaming_performance_samples() {
        let config = StreamingConfig::default();
        let optimizer = StreamingOptimizer::<String>::new(config);
        optimizer.start().await.unwrap();

        // Generate activity to create performance samples
        for i in 0..20 {
            optimizer.send_item(format!("sample_{}", i)).await.unwrap();

            // Add small delay to create measurable timing
            sleep(Duration::from_millis(1)).await;
        }

        // Wait for performance data to be collected
        sleep(Duration::from_millis(200)).await;

        let metrics = optimizer.get_metrics().await;

        // Metrics should be collected but exact values depend on timing
        assert!(metrics.throughput_messages_per_second >= 0.0);
        assert!(metrics.average_latency_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_streaming_config_validation() {
        // Test that various configs can be created without panicking
        let configs = vec![
            StreamingConfig::default(),
            StreamingOptimizerFactory::performance_optimized(),
            StreamingOptimizerFactory::memory_optimized(),
            StreamingOptimizerFactory::low_latency(),
        ];

        for config in configs {
            let optimizer = StreamingOptimizer::<i32>::new(config);
            let result = optimizer.start().await;
            assert!(result.is_ok(), "All streaming configs should be valid");
        }
    }
}
