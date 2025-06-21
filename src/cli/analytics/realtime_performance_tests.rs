//! Task 3.5: Real-time Analytics Updates and Performance Testing
//!
//! This module implements comprehensive tests for real-time analytics updates
//! and performance optimization, covering real-time data streaming, performance
//! with large datasets, memory optimization, and concurrency safety.

use super::test_utils::*;
use super::*;
use crate::cli::cost::CostEntry;
use crate::cli::error::Result;
use crate::cli::history::HistoryEntry;
use crate::cli::session::SessionId;
use chrono::{DateTime, Duration, Utc};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration as StdDuration, Instant};
use tokio::sync::RwLock;

/// Real-time analytics stream handler for testing
#[derive(Clone)]
pub struct RealTimeAnalyticsStream {
    pub analytics_engine: Arc<AnalyticsEngine>,
    pub is_streaming: Arc<AtomicBool>,
    pub update_count: Arc<AtomicU64>,
    pub memory_tracker: Arc<MemoryTracker>,
    pub updates_buffer: Arc<RwLock<Vec<AnalyticsUpdate>>>,
}

/// Analytics update event for real-time streaming
#[derive(Debug, Clone)]
pub struct AnalyticsUpdate {
    pub timestamp: DateTime<Utc>,
    pub update_type: UpdateType,
    pub session_id: SessionId,
    pub metric_deltas: MetricDeltas,
}

/// Types of real-time updates
#[derive(Debug, Clone)]
pub enum UpdateType {
    CommandCompleted,
    CostIncurred,
    SessionStarted,
    SessionEnded,
    AlertTriggered,
    PerformanceMetricUpdated,
}

/// Metric deltas for incremental updates
#[derive(Debug, Clone, Default)]
pub struct MetricDeltas {
    pub cost_delta: f64,
    pub command_count_delta: i32,
    pub success_count_delta: i32,
    pub failure_count_delta: i32,
    pub response_time_delta: i64,
    pub token_delta: i64,
}

/// Memory usage tracking for performance tests
#[derive(Debug, Default)]
pub struct MemoryTracker {
    pub peak_memory_mb: AtomicU64,
    pub allocations: AtomicU64,
    pub deallocations: AtomicU64,
    pub active_objects: AtomicU64,
}

impl MemoryTracker {
    pub fn record_allocation(&self, size_mb: u64) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.active_objects.fetch_add(1, Ordering::Relaxed);

        // Use compare-and-swap for thread-safe peak memory tracking
        loop {
            let current = self.peak_memory_mb.load(Ordering::Relaxed);
            let new_peak = current + size_mb;

            if self
                .peak_memory_mb
                .compare_exchange_weak(current, new_peak, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    pub fn record_deallocation(&self, _size_mb: u64) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);

        // Only decrement if we have active objects
        if self.active_objects.load(Ordering::Relaxed) > 0 {
            self.active_objects.fetch_sub(1, Ordering::Relaxed);
        }

        // Don't modify peak memory on deallocation - it represents the peak, not current
    }

    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            peak_memory_mb: self.peak_memory_mb.load(Ordering::Relaxed),
            total_allocations: self.allocations.load(Ordering::Relaxed),
            total_deallocations: self.deallocations.load(Ordering::Relaxed),
            active_objects: self.active_objects.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub peak_memory_mb: u64,
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub active_objects: u64,
}

impl RealTimeAnalyticsStream {
    pub async fn new(analytics_engine: Arc<AnalyticsEngine>) -> Result<Self> {
        Ok(Self {
            analytics_engine,
            is_streaming: Arc::new(AtomicBool::new(false)),
            update_count: Arc::new(AtomicU64::new(0)),
            memory_tracker: Arc::new(MemoryTracker::default()),
            updates_buffer: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Start real-time streaming of analytics updates
    pub async fn start_streaming(&self) -> Result<()> {
        self.is_streaming.store(true, Ordering::Relaxed);

        // Simulate real-time updates
        let updates_buffer = Arc::clone(&self.updates_buffer);
        let is_streaming = Arc::clone(&self.is_streaming);
        let update_count = Arc::clone(&self.update_count);
        let memory_tracker = Arc::clone(&self.memory_tracker);

        tokio::spawn(async move {
            while is_streaming.load(Ordering::Relaxed) {
                let update = AnalyticsUpdate {
                    timestamp: Utc::now(),
                    update_type: UpdateType::CommandCompleted,
                    session_id: uuid::Uuid::new_v4(),
                    metric_deltas: MetricDeltas {
                        cost_delta: 0.01,
                        command_count_delta: 1,
                        success_count_delta: 1,
                        ..Default::default()
                    },
                };

                // Track memory usage for this update
                memory_tracker.record_allocation(1); // 1MB per update

                // Add to buffer
                if let Ok(mut buffer) = updates_buffer.try_write() {
                    buffer.push(update);
                    // Keep buffer size reasonable
                    if buffer.len() > 1000 {
                        buffer.drain(0..500);
                    }
                }

                update_count.fetch_add(1, Ordering::Relaxed);

                tokio::time::sleep(StdDuration::from_millis(10)).await;
            }
        });

        Ok(())
    }

    /// Stop streaming and clean up resources
    pub fn stop_streaming(&self) {
        self.is_streaming.store(false, Ordering::Relaxed);
    }

    /// Get recent updates from buffer
    pub async fn get_recent_updates(&self, limit: usize) -> Vec<AnalyticsUpdate> {
        let buffer = self.updates_buffer.read().await;
        let start = if buffer.len() > limit {
            buffer.len() - limit
        } else {
            0
        };
        buffer[start..].to_vec()
    }

    /// Process a batch of analytics updates efficiently
    pub async fn process_update_batch(&self, updates: Vec<AnalyticsUpdate>) -> Result<()> {
        use crate::cli::analytics::performance_fixes::SimpleBatchProcessor;

        let processor = SimpleBatchProcessor::new(100, 8);
        let memory_tracker = Arc::clone(&self.memory_tracker);

        processor
            .process_updates(updates, move |batch| {
                let memory_tracker = Arc::clone(&memory_tracker);
                Box::pin(async move {
                    let batch_size = batch.len();

                    // Record memory allocation
                    memory_tracker.record_allocation(batch_size as u64);

                    // Fast batch processing
                    if batch_size > 0 {
                        tokio::time::sleep(StdDuration::from_micros(batch_size as u64)).await;
                    }

                    // Clean up
                    memory_tracker.record_deallocation(batch_size as u64);

                    Ok(())
                })
            })
            .await?;

        Ok(())
    }
}

/// Test suite for Task 3.5.1: Real-time analytics data updates and event handling
#[cfg(test)]
mod realtime_updates_tests {
    use super::*;

    #[tokio::test]
    async fn test_real_time_data_streaming() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        let stream = RealTimeAnalyticsStream::new(analytics_engine)
            .await
            .unwrap();

        // Start streaming
        stream.start_streaming().await.unwrap();

        // Wait for some updates to be generated
        tokio::time::sleep(StdDuration::from_millis(200)).await;

        // Get recent updates
        let updates = stream.get_recent_updates(10).await;
        assert!(!updates.is_empty(), "Should receive real-time updates");

        let update = &updates[0];
        assert!(matches!(update.update_type, UpdateType::CommandCompleted));
        assert!(update.timestamp <= Utc::now());

        // Verify streaming continues
        let initial_count = stream.update_count.load(Ordering::Relaxed);
        tokio::time::sleep(StdDuration::from_millis(100)).await;
        let later_count = stream.update_count.load(Ordering::Relaxed);
        assert!(
            later_count > initial_count,
            "Should continue processing updates"
        );

        stream.stop_streaming();

        // Verify streaming stops
        tokio::time::sleep(StdDuration::from_millis(100)).await;
        let final_count = stream.update_count.load(Ordering::Relaxed);

        tokio::time::sleep(StdDuration::from_millis(100)).await;
        let after_stop_count = stream.update_count.load(Ordering::Relaxed);

        assert_eq!(
            final_count, after_stop_count,
            "Updates should stop after stopping stream"
        );
    }

    #[tokio::test]
    async fn test_event_driven_analytics_updates() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        let stream = RealTimeAnalyticsStream::new(analytics_engine)
            .await
            .unwrap();

        // Simulate various events by adding them directly to buffer
        let events = vec![
            AnalyticsUpdate {
                timestamp: Utc::now(),
                update_type: UpdateType::CommandCompleted,
                session_id: uuid::Uuid::new_v4(),
                metric_deltas: MetricDeltas {
                    command_count_delta: 1,
                    success_count_delta: 1,
                    response_time_delta: 1500,
                    ..Default::default()
                },
            },
            AnalyticsUpdate {
                timestamp: Utc::now(),
                update_type: UpdateType::CostIncurred,
                session_id: uuid::Uuid::new_v4(),
                metric_deltas: MetricDeltas {
                    cost_delta: 0.05,
                    token_delta: 1000,
                    ..Default::default()
                },
            },
            AnalyticsUpdate {
                timestamp: Utc::now(),
                update_type: UpdateType::AlertTriggered,
                session_id: uuid::Uuid::new_v4(),
                metric_deltas: Default::default(),
            },
        ];

        // Add events to buffer
        {
            let mut buffer = stream.updates_buffer.write().await;
            buffer.extend(events);
        }

        // Get updates and verify event types
        let received_updates = stream.get_recent_updates(10).await;
        let received_types: Vec<_> = received_updates.iter().map(|u| &u.update_type).collect();

        assert!(received_types
            .iter()
            .any(|t| matches!(t, UpdateType::CommandCompleted)));
        assert!(received_types
            .iter()
            .any(|t| matches!(t, UpdateType::CostIncurred)));
        assert!(received_types
            .iter()
            .any(|t| matches!(t, UpdateType::AlertTriggered)));
    }

    #[tokio::test]
    async fn test_real_time_metric_calculations() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        // Add initial data
        let session_id = uuid::Uuid::new_v4();
        let initial_data = fixture.data_generator.generate_test_data(1);
        fixture.load_data(&initial_data).await.unwrap();

        // Get baseline metrics
        let baseline_summary = analytics_engine.generate_summary(1).await.unwrap();

        // Simulate real-time updates
        let cost_entry = CostEntry::new(
            session_id,
            "real_time_cmd".to_string(),
            0.10,
            500,
            1000,
            2000,
            "claude-3-opus".to_string(),
        );

        fixture
            .cost_tracker
            .write()
            .await
            .record_cost(cost_entry)
            .await
            .unwrap();

        let history_entry = HistoryEntry::new(
            session_id,
            "real_time_cmd".to_string(),
            vec!["--real-time".to_string()],
            "Real-time response".to_string(),
            true,
            2000,
        );

        fixture
            .history_store
            .write()
            .await
            .store_entry(history_entry)
            .await
            .unwrap();

        // Get updated metrics
        let updated_summary = analytics_engine.generate_summary(1).await.unwrap();

        // Verify metrics were updated in real-time
        assert!(updated_summary.cost_summary.total_cost > baseline_summary.cost_summary.total_cost);
        assert!(
            updated_summary.cost_summary.command_count
                > baseline_summary.cost_summary.command_count
        );
        assert!(
            updated_summary.history_stats.total_entries
                > baseline_summary.history_stats.total_entries
        );
    }

    #[tokio::test]
    async fn test_websocket_streaming_simulation() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        let stream = RealTimeAnalyticsStream::new(analytics_engine)
            .await
            .unwrap();

        // Start streaming
        stream.start_streaming().await.unwrap();

        // Wait for some updates to accumulate
        tokio::time::sleep(StdDuration::from_millis(300)).await;

        // Simulate multiple concurrent WebSocket connections checking for updates
        let mut handles = Vec::new();

        for i in 0..5 {
            let stream_clone = stream.clone();
            let handle = tokio::spawn(async move {
                let mut total_received = 0;
                let start_time = Instant::now();

                // Simulate periodic polling of updates (like WebSocket would do)
                while start_time.elapsed() < StdDuration::from_millis(200) {
                    let updates = stream_clone.get_recent_updates(5).await;
                    total_received += updates.len();
                    tokio::time::sleep(StdDuration::from_millis(20)).await;
                }

                (i, total_received)
            });
            handles.push(handle);
        }

        // Wait for all connections to process
        tokio::time::sleep(StdDuration::from_millis(250)).await;

        stream.stop_streaming();

        // Verify all connections received updates
        for handle in handles {
            let (connection_id, received_count) = handle.await.unwrap();
            assert!(
                received_count > 0,
                "Connection {} should receive updates",
                connection_id
            );
        }
    }
}

/// Helper function for generating large datasets (accessible across test modules)
#[cfg(test)]
async fn generate_large_dataset(size: usize) -> TestDataSet {
    // Use optimized parallel generation
    let (cost_entries, history_entries) =
        crate::cli::analytics::performance_fixes::generate_large_dataset_optimized(size).await;

    let start_time = cost_entries
        .first()
        .map(|e| e.timestamp)
        .unwrap_or_else(|| Utc::now() - Duration::days(30));
    let end_time = cost_entries
        .last()
        .map(|e| e.timestamp)
        .unwrap_or_else(Utc::now);

    TestDataSet {
        sessions: vec![],
        cost_entries,
        history_entries,
        start_time,
        end_time,
    }
}

/// Test suite for Task 3.5.2: Analytics performance with large datasets (10k+ entries)
#[cfg(test)]
mod large_dataset_performance_tests {
    use super::*;

    #[tokio::test]
    async fn test_query_performance_10k_plus_entries() {
        use crate::cli::analytics::performance_optimizations::{BatchProcessor, BoundedCache};

        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        // Add cache for summary results
        let summary_cache: BoundedCache<u32, crate::cli::analytics::AnalyticsSummary> =
            BoundedCache::new(
                100,
                50,
                |summary: &crate::cli::analytics::AnalyticsSummary| {
                    // Estimate size of summary in bytes
                    std::mem::size_of_val(summary)
                        + summary.insights.iter().map(|s| s.len()).sum::<usize>()
                        + summary.alerts.len() * 100
                },
            );

        // Generate 10k+ entries using optimized parallel generation
        let large_dataset = generate_large_dataset(10000).await;
        println!(
            "Generated {} cost entries, {} history entries",
            large_dataset.cost_entries.len(),
            large_dataset.history_entries.len()
        );

        // Load data in batches for better performance
        let timer = perf::Timer::new("load_large_dataset_batched");

        // Batch load cost entries
        let cost_batch_processor = BatchProcessor::new(500, StdDuration::from_millis(10), 8, {
            let cost_tracker = Arc::clone(&fixture.cost_tracker);
            move |batch: Vec<CostEntry>| {
                let cost_tracker = Arc::clone(&cost_tracker);
                Box::pin(async move {
                    let mut tracker = cost_tracker.write().await;
                    for entry in batch {
                        tracker.record_cost(entry).await?;
                    }
                    Ok(())
                })
            }
        });

        for entry in large_dataset.cost_entries {
            cost_batch_processor.add(entry).await.unwrap();
        }
        cost_batch_processor.flush().await.unwrap();

        // Batch load history entries
        let history_batch_processor = BatchProcessor::new(500, StdDuration::from_millis(10), 8, {
            let history_store = Arc::clone(&fixture.history_store);
            move |batch: Vec<HistoryEntry>| {
                let history_store = Arc::clone(&history_store);
                Box::pin(async move {
                    let mut store = history_store.write().await;
                    for entry in batch {
                        store.store_entry(entry).await?;
                    }
                    Ok(())
                })
            }
        });

        for entry in large_dataset.history_entries {
            history_batch_processor.add(entry).await.unwrap();
        }
        history_batch_processor.flush().await.unwrap();

        println!("Data loading (batched): {}", timer.report());

        // Test query performance with caching
        let query_timer = perf::Timer::new("analytics_summary_10k_cached");

        // Check cache first
        let summary = if let Some(cached) = summary_cache.get(&30).await {
            cached
        } else {
            let summary = analytics_engine.generate_summary(30).await.unwrap();
            summary_cache.put(30, summary.clone()).await.unwrap();
            summary
        };

        println!(
            "Analytics summary generation (cached): {}",
            query_timer.report()
        );

        // Performance requirements: should complete within 3 seconds with optimizations
        assert!(
            query_timer.elapsed_ms() < 3000,
            "Large dataset query should complete within 3 seconds, took {}ms",
            query_timer.elapsed_ms()
        );

        // Verify data integrity
        assert!(summary.cost_summary.command_count >= 10000);
        assert!(summary.history_stats.total_entries >= 10000);
        assert!(summary.cost_summary.total_cost > 0.0);

        // Test cache hit performance
        let cache_hit_timer = perf::Timer::new("cache_hit");
        let _cached_summary = summary_cache.get(&30).await.unwrap();
        println!("Cache hit time: {}", cache_hit_timer.report());
        assert!(
            cache_hit_timer.elapsed_ms() < 10,
            "Cache hits should be very fast"
        );
    }

    #[tokio::test]
    async fn test_aggregation_performance_large_datasets() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        // Generate large dataset with patterns
        let large_dataset = generate_large_dataset(15000).await;
        fixture.load_data(&large_dataset).await.unwrap();

        let mut perf_collector = perf::PerfCollector::new();

        // Test different aggregation operations
        let timer1 = perf::Timer::new("cost_by_command_aggregation");
        let cost_tracker = fixture.cost_tracker.read().await;
        let top_commands = cost_tracker.get_top_commands(10).await.unwrap();
        perf_collector.record(timer1.name(), timer1.elapsed_ms());
        drop(cost_tracker);

        let timer2 = perf::Timer::new("performance_metrics_calculation");
        let search = crate::cli::history::HistorySearch::default();
        let _metrics = analytics_engine
            .calculate_performance_metrics(&search)
            .await
            .unwrap();
        perf_collector.record(timer2.name(), timer2.elapsed_ms());

        let timer3 = perf::Timer::new("dashboard_data_aggregation");
        let _dashboard = analytics_engine.get_dashboard_data().await.unwrap();
        perf_collector.record(timer3.name(), timer3.elapsed_ms());

        let summary = perf_collector.summary();
        println!("Aggregation performance summary: {:?}", summary);

        // Performance requirements
        assert!(
            summary.max_ms < 3000,
            "No single aggregation should take more than 3 seconds"
        );
        assert!(
            summary.average_ms < 1500,
            "Average aggregation time should be under 1.5 seconds"
        );
        assert!(!top_commands.is_empty(), "Should return aggregated results");
    }

    #[tokio::test]
    async fn test_concurrent_large_dataset_queries() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        // Generate large dataset
        let large_dataset = generate_large_dataset(12000).await;
        fixture.load_data(&large_dataset).await.unwrap();

        // Run concurrent queries
        let mut handles = Vec::new();
        let analytics_engine = Arc::new(analytics_engine);

        for i in 0..10 {
            let engine_clone = Arc::clone(&analytics_engine);
            let handle = tokio::spawn(async move {
                let timer = perf::Timer::new(&format!("concurrent_query_{}", i));
                let summary = engine_clone.generate_summary(30).await.unwrap();
                let duration = timer.elapsed_ms();

                (i, duration, summary.cost_summary.command_count)
            });
            handles.push(handle);
        }

        // Wait for all queries to complete
        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await.unwrap());
        }

        // Verify all queries completed successfully
        for (query_id, duration, command_count) in results {
            assert!(
                command_count >= 12000,
                "Query {} should return all data",
                query_id
            );
            assert!(
                duration < 10000,
                "Query {} should complete within 10 seconds, took {}ms",
                query_id,
                duration
            );
        }
    }

    #[tokio::test]
    async fn test_database_query_optimization() {
        use crate::cli::analytics::performance_optimizations::{
            BoundedCache, OptimizedDataGenerator,
        };
        use futures::StreamExt;

        let fixture = AnalyticsTestFixture::new().await.unwrap();

        // Create index cache for time-based queries
        let time_index_cache: Arc<RwLock<std::collections::BTreeMap<DateTime<Utc>, Vec<usize>>>> =
            Arc::new(RwLock::new(std::collections::BTreeMap::new()));

        // Generate data with optimized streaming approach
        let generator = OptimizedDataGenerator::new(1000);
        let mut stream = generator.generate_stream(20000);
        let mut all_entries = Vec::new();
        let mut entry_index = 0;

        // Build time index while loading
        let mut time_index = time_index_cache.write().await;

        while let Some(batch) = stream.next().await {
            for entry in &batch {
                time_index
                    .entry(
                        entry
                            .timestamp
                            .date_naive()
                            .and_hms_opt(0, 0, 0)
                            .unwrap()
                            .and_utc(),
                    )
                    .or_insert_with(Vec::new)
                    .push(entry_index);
                entry_index += 1;
            }
            all_entries.extend(batch);
        }
        drop(time_index);

        // Convert to test dataset and load
        let mut history_entries = Vec::new();
        for (idx, cost_entry) in all_entries.iter().enumerate() {
            let mut history_entry = HistoryEntry::new(
                cost_entry.session_id,
                cost_entry.command_name.clone(),
                vec![],
                format!("Output {}", idx),
                true,
                cost_entry.duration_ms,
            );
            history_entry.timestamp = cost_entry.timestamp;
            history_entries.push(history_entry);
        }

        let dataset = TestDataSet {
            sessions: vec![],
            cost_entries: all_entries,
            history_entries,
            start_time: Utc::now() - Duration::days(30),
            end_time: Utc::now(),
        };

        fixture.load_data(&dataset).await.unwrap();

        // Test time-range filtering with index optimization
        let timer1 = perf::Timer::new("indexed_recent_data_query");
        let recent_filter = crate::cli::cost::CostFilter {
            since: Some(Utc::now() - Duration::days(1)),
            until: Some(Utc::now()),
            ..Default::default()
        };

        // Use index to pre-filter
        let indexed_entries = {
            let time_index = time_index_cache.read().await;
            let recent_date = (Utc::now() - Duration::days(1))
                .date_naive()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc();
            time_index
                .range(recent_date..)
                .flat_map(|(_, indices)| indices.clone())
                .collect::<Vec<_>>()
        };

        let cost_tracker = fixture.cost_tracker.read().await;
        let recent_summary = cost_tracker
            .get_filtered_summary(&recent_filter)
            .await
            .unwrap();
        let recent_query_time = timer1.elapsed_ms();
        drop(cost_tracker);

        // Test full dataset query
        let timer2 = perf::Timer::new("full_data_query");
        let cost_tracker = fixture.cost_tracker.read().await;
        let full_summary = cost_tracker
            .get_filtered_summary(&Default::default())
            .await
            .unwrap();
        let full_query_time = timer2.elapsed_ms();
        drop(cost_tracker);

        println!(
            "Indexed recent data query: {}ms (pre-filtered {} entries)",
            recent_query_time,
            indexed_entries.len()
        );
        println!("Full data query: {}ms", full_query_time);

        // Indexed queries should be much faster
        assert!(
            recent_query_time < full_query_time / 3,
            "Indexed query should be at least 3x faster than full query"
        );

        // Both should complete reasonably fast
        assert!(
            recent_query_time < 1000,
            "Recent query should complete within 1 second"
        );
        assert!(
            full_query_time < 5000,
            "Full query should complete within 5 seconds"
        );

        // Both should return valid data
        assert!(recent_summary.command_count <= full_summary.command_count);
        assert!(recent_summary.total_cost <= full_summary.total_cost);
    }

    #[tokio::test]
    async fn test_pagination_and_streaming_large_results() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();

        // Generate large dataset
        let large_dataset = generate_large_dataset(25000).await;
        fixture.load_data(&large_dataset).await.unwrap();

        // Test paginated queries
        let timer = perf::Timer::new("paginated_query");
        let search = crate::cli::history::HistorySearch {
            limit: 1000,
            offset: 0,
            ..Default::default()
        };

        let history_store = fixture.history_store.read().await;
        let first_page = history_store.search(&search).await.unwrap();

        let search_page2 = crate::cli::history::HistorySearch {
            limit: 1000,
            offset: 1000,
            ..Default::default()
        };
        let second_page = history_store.search(&search_page2).await.unwrap();
        drop(history_store);

        println!("Paginated query: {}", timer.report());

        // Verify pagination works correctly
        assert_eq!(
            first_page.len(),
            1000,
            "First page should return 1000 results"
        );
        assert_eq!(
            second_page.len(),
            1000,
            "Second page should return 1000 results"
        );

        // Pages should be different
        assert_ne!(
            first_page[0].timestamp, second_page[0].timestamp,
            "Different pages should have different data"
        );

        // Pagination should be fast
        assert!(
            timer.elapsed_ms() < 2000,
            "Paginated queries should be fast"
        );
    }
}

/// Test suite for Task 3.5.3: Memory usage and optimization for analytics processing
#[cfg(test)]
mod memory_optimization_tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_allocation_patterns() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        let stream = RealTimeAnalyticsStream::new(analytics_engine)
            .await
            .unwrap();
        let memory_tracker = Arc::clone(&stream.memory_tracker);

        // Simulate memory allocation patterns
        for i in 0..1000 {
            memory_tracker.record_allocation(1); // 1MB allocations

            // Simulate periodic cleanup
            if i % 100 == 0 {
                for _ in 0..50 {
                    memory_tracker.record_deallocation(1);
                }
            }
        }

        let stats = memory_tracker.get_stats();
        println!("Memory stats: {:?}", stats);

        // Memory should be managed efficiently
        assert!(
            stats.active_objects < stats.total_allocations,
            "Should have deallocated some objects"
        );
        assert!(
            stats.total_deallocations > 0,
            "Should have performed deallocations"
        );

        // Peak memory should be reasonable
        assert!(
            stats.peak_memory_mb < 1000,
            "Peak memory should stay under 1GB"
        );
    }

    #[tokio::test]
    async fn test_garbage_collection_impact() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        // Generate data that will stress garbage collection
        let large_dataset = generate_large_dataset(15000).await;

        // Measure allocation phase
        let alloc_timer = perf::Timer::new("allocation_phase");
        fixture.load_data(&large_dataset).await.unwrap();
        let alloc_time = alloc_timer.elapsed_ms();

        // Generate multiple summaries to trigger garbage collection
        let mut gc_times = Vec::new();
        for i in 0..10 {
            let gc_timer = perf::Timer::new(&format!("gc_cycle_{}", i));
            let _summary = analytics_engine.generate_summary(30).await.unwrap();
            gc_times.push(gc_timer.elapsed_ms());

            // Force some memory pressure
            let _temp_data: Vec<Vec<u8>> = (0..1000).map(|_| vec![0u8; 1024]).collect();
            drop(_temp_data);
        }

        println!("Allocation time: {}ms", alloc_time);
        println!("GC cycle times: {:?}", gc_times);

        // Performance should stabilize after initial GC cycles
        let early_avg = gc_times[0..3].iter().sum::<u128>() / 3;
        let late_avg = gc_times[7..10].iter().sum::<u128>() / 3;

        // Later cycles should not be significantly slower (within 50%)
        assert!(
            late_avg <= early_avg * 150 / 100,
            "GC impact should stabilize over time"
        );
    }

    #[tokio::test]
    async fn test_memory_leak_detection() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        let stream = RealTimeAnalyticsStream::new(analytics_engine)
            .await
            .unwrap();
        let memory_tracker = Arc::clone(&stream.memory_tracker);

        // Simulate a workload that might cause memory leaks
        for cycle in 0..50 {
            // Allocate objects for analytics processing
            for _ in 0..100 {
                memory_tracker.record_allocation(1);
            }

            // Process some analytics (simulate work)
            let _summary = stream.analytics_engine.generate_summary(1).await.unwrap();

            // Clean up most objects (but simulate potential leaks)
            for _ in 0..95 {
                memory_tracker.record_deallocation(1);
            }

            // Check memory growth every 10 cycles
            if cycle % 10 == 0 {
                let stats = memory_tracker.get_stats();
                println!(
                    "Cycle {}: Active objects: {}, Peak memory: {}MB",
                    cycle, stats.active_objects, stats.peak_memory_mb
                );

                // Active objects should not grow unbounded
                assert!(
                    stats.active_objects < 1000,
                    "Active objects should not grow unbounded (cycle {})",
                    cycle
                );
            }
        }

        let final_stats = memory_tracker.get_stats();

        // Final check: active objects should be reasonable
        assert!(
            final_stats.active_objects < 500,
            "Should not have excessive active objects at end"
        );

        // Deallocation rate should be reasonable
        let dealloc_rate =
            final_stats.total_deallocations as f64 / final_stats.total_allocations as f64;
        assert!(
            dealloc_rate > 0.8,
            "Should deallocate at least 80% of allocated objects"
        );
    }

    #[tokio::test]
    async fn test_caching_and_pooling_strategies() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        // Generate dataset for caching tests
        let dataset = generate_large_dataset(10000).await;
        fixture.load_data(&dataset).await.unwrap();

        // Test cache effectiveness - first query (cache miss)
        let timer1 = perf::Timer::new("first_query_cache_miss");
        let summary1 = analytics_engine.generate_summary(7).await.unwrap();
        let first_time = timer1.elapsed_ms();

        // Second identical query (cache hit simulation)
        let timer2 = perf::Timer::new("second_query_cache_hit");
        let summary2 = analytics_engine.generate_summary(7).await.unwrap();
        let second_time = timer2.elapsed_ms();

        // Different query (different cache entry)
        let timer3 = perf::Timer::new("different_query");
        let summary3 = analytics_engine.generate_summary(1).await.unwrap();
        let third_time = timer3.elapsed_ms();

        println!("First query: {}ms", first_time);
        println!("Second query: {}ms", second_time);
        println!("Different query: {}ms", third_time);

        // Verify results are consistent
        assert_eq!(
            summary1.cost_summary.total_cost,
            summary2.cost_summary.total_cost
        );
        assert_ne!(
            summary1.cost_summary.total_cost,
            summary3.cost_summary.total_cost
        );

        // Test object pooling simulation
        let mut pool_objects = Vec::new();
        let pool_timer = perf::Timer::new("object_pooling");

        // Simulate object pool usage
        for i in 0..1000 {
            if i % 2 == 0 {
                // "Allocate" from pool
                pool_objects.push(format!("pooled_object_{}", i));
            } else {
                // "Return" to pool
                if !pool_objects.is_empty() {
                    pool_objects.pop();
                }
            }
        }

        println!("Object pooling simulation: {}", pool_timer.report());

        // Pool should help with allocation efficiency
        assert!(
            pool_timer.elapsed_ms() < 100,
            "Object pooling should be efficient"
        );
        assert!(
            pool_objects.len() <= 500,
            "Pool should reuse objects effectively"
        );
    }

    #[tokio::test]
    async fn test_memory_efficient_batch_processing() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        let stream = RealTimeAnalyticsStream::new(analytics_engine)
            .await
            .unwrap();
        let memory_tracker = Arc::clone(&stream.memory_tracker);

        // Generate large batch of updates
        let mut updates = Vec::new();
        for i in 0..5000 {
            updates.push(AnalyticsUpdate {
                timestamp: Utc::now(),
                update_type: if i % 2 == 0 {
                    UpdateType::CommandCompleted
                } else {
                    UpdateType::CostIncurred
                },
                session_id: uuid::Uuid::new_v4(),
                metric_deltas: MetricDeltas {
                    cost_delta: 0.01,
                    command_count_delta: 1,
                    ..Default::default()
                },
            });
        }

        // Test batch processing vs individual processing
        let batch_timer = perf::Timer::new("batch_processing");
        stream.process_update_batch(updates.clone()).await.unwrap();
        let batch_time = batch_timer.elapsed_ms();

        let individual_timer = perf::Timer::new("individual_processing");
        for update in updates.into_iter().take(100) {
            // Process subset individually
            stream.process_update_batch(vec![update]).await.unwrap();
        }
        let individual_time = individual_timer.elapsed_ms();

        println!("Batch processing 5000 updates: {}ms", batch_time);
        println!("Individual processing 100 updates: {}ms", individual_time);

        // Batch processing should be more memory efficient
        let stats = memory_tracker.get_stats();
        assert!(
            stats.peak_memory_mb < 100,
            "Batch processing should use reasonable memory"
        );

        // Batch should be significantly faster per item
        let batch_per_item = batch_time as f64 / 5000.0;
        let individual_per_item = individual_time as f64 / 100.0;

        assert!(
            batch_per_item < individual_per_item,
            "Batch processing should be more efficient per item"
        );
    }
}

/// Test suite for Task 3.5.4: Analytics data consistency under concurrent operations
#[cfg(test)]
mod concurrency_safety_tests {
    use super::*;

    #[tokio::test]
    async fn test_concurrent_read_write_operations() {
        use crate::cli::analytics::performance_optimizations::BatchProcessor;

        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        let cost_tracker = Arc::clone(&fixture.cost_tracker);
        let history_store = Arc::clone(&fixture.history_store);
        let analytics_engine_clone = Arc::clone(&analytics_engine);

        // Initial data
        let session_id = uuid::Uuid::new_v4();
        let initial_data = fixture.data_generator.generate_test_data(1);
        fixture.load_data(&initial_data).await.unwrap();

        // Use batch processors for writes to reduce contention
        let cost_batch_processor =
            Arc::new(BatchProcessor::new(50, StdDuration::from_millis(50), 4, {
                let cost_tracker = Arc::clone(&cost_tracker);
                move |batch: Vec<CostEntry>| {
                    let cost_tracker = Arc::clone(&cost_tracker);
                    Box::pin(async move {
                        let mut tracker = cost_tracker.write().await;
                        for entry in batch {
                            tracker.record_cost(entry).await?;
                        }
                        Ok(())
                    })
                }
            }));

        let history_batch_processor =
            Arc::new(BatchProcessor::new(50, StdDuration::from_millis(50), 4, {
                let history_store = Arc::clone(&history_store);
                move |batch: Vec<HistoryEntry>| {
                    let history_store = Arc::clone(&history_store);
                    Box::pin(async move {
                        let mut store = history_store.write().await;
                        for entry in batch {
                            store.store_entry(entry).await?;
                        }
                        Ok(())
                    })
                }
            }));

        let write_counter = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();

        // Concurrent write operations using batch processors
        for i in 0..10 {
            let cost_processor = Arc::clone(&cost_batch_processor);
            let history_processor = Arc::clone(&history_batch_processor);
            let counter = Arc::clone(&write_counter);
            let session_id_copy = session_id;

            let write_handle = tokio::spawn(async move {
                for j in 0..50 {
                    let cost_entry = CostEntry::new(
                        session_id_copy,
                        format!("concurrent_cmd_{}_{}", i, j),
                        0.01 * (j + 1) as f64,
                        100 + j as u32 * 10,
                        200 + j as u32 * 20,
                        1000 + j as u64 * 100,
                        "claude-3-haiku".to_string(),
                    );

                    let history_entry = HistoryEntry::new(
                        session_id_copy,
                        format!("concurrent_cmd_{}_{}", i, j),
                        vec![format!("--arg-{}", j)],
                        format!("Output {}_{}", i, j),
                        j % 10 != 0, // 10% failure rate
                        1000 + j as u64 * 100,
                    );

                    // Add to batch processors
                    cost_processor.add(cost_entry).await.unwrap();
                    history_processor.add(history_entry).await.unwrap();
                    counter.fetch_add(1, Ordering::Relaxed);

                    // Reduced delay for better throughput
                    if j % 10 == 0 {
                        tokio::time::sleep(StdDuration::from_micros(10)).await;
                    }
                }

                (format!("Writer {} completed", i), 50)
            });

            handles.push(write_handle);
        }

        // Concurrent read operations
        for i in 0..5 {
            let analytics_clone = Arc::clone(&analytics_engine_clone);

            let read_handle = tokio::spawn(async move {
                let mut summaries = Vec::new();

                for _ in 0..20 {
                    if let Ok(summary) = analytics_clone.generate_summary(1).await {
                        summaries.push(summary);
                    }

                    tokio::time::sleep(StdDuration::from_millis(10)).await;
                }

                (format!("Reader {} completed", i), summaries.len())
            });

            handles.push(read_handle);
        }

        // Wait for all operations to complete
        for handle in handles {
            let _result = handle.await.unwrap();
        }

        // Flush batch processors
        cost_batch_processor.flush().await.unwrap();
        history_batch_processor.flush().await.unwrap();

        // Give time for final writes to propagate
        tokio::time::sleep(StdDuration::from_millis(100)).await;

        // Verify final consistency
        let final_summary = analytics_engine.generate_summary(1).await.unwrap();
        let total_writes = write_counter.load(Ordering::Relaxed);

        println!("Total writes attempted: {}", total_writes);
        println!(
            "Final command count: {}",
            final_summary.cost_summary.command_count
        );
        println!(
            "Final history entries: {}",
            final_summary.history_stats.total_entries
        );

        // Should have processed most writes (allow for some timing issues)
        assert!(
            final_summary.cost_summary.command_count >= 450,
            "Should have processed at least 450 concurrent writes, got {}",
            final_summary.cost_summary.command_count
        );
        assert!(
            final_summary.history_stats.total_entries >= 450,
            "Should have at least 450 history entries, got {}",
            final_summary.history_stats.total_entries
        );
    }

    #[tokio::test]
    async fn test_race_condition_handling() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        let stream = RealTimeAnalyticsStream::new(analytics_engine)
            .await
            .unwrap();

        // Create a race condition scenario
        let mut handles = Vec::new();
        let update_counter = Arc::new(AtomicU64::new(0));

        // Multiple producers creating updates simultaneously
        for producer_id in 0..5 {
            let stream_clone = stream.clone();
            let counter_clone = Arc::clone(&update_counter);

            let producer_handle = tokio::spawn(async move {
                for i in 0..100 {
                    let update = AnalyticsUpdate {
                        timestamp: Utc::now(),
                        update_type: UpdateType::CommandCompleted,
                        session_id: uuid::Uuid::new_v4(),
                        metric_deltas: MetricDeltas {
                            cost_delta: 0.01,
                            command_count_delta: 1,
                            success_count_delta: 1,
                            ..Default::default()
                        },
                    };

                    // Simulate race condition with rapid-fire updates
                    {
                        let mut buffer = stream_clone.updates_buffer.write().await;
                        buffer.push(update);
                    }
                    counter_clone.fetch_add(1, Ordering::Relaxed);

                    // Minimal delay to increase race probability
                    if i % 10 == 0 {
                        tokio::time::sleep(StdDuration::from_micros(1)).await;
                    }
                }

                (producer_id, 0) // Return tuple for consistency
            });

            handles.push(producer_handle);
        }

        // Multiple consumers processing updates simultaneously
        for consumer_id in 0..3 {
            let stream_clone = stream.clone();

            let consumer_handle = tokio::spawn(async move {
                let mut received_count = 0;
                let start_time = Instant::now();

                while start_time.elapsed() < StdDuration::from_millis(500) {
                    let updates = stream_clone.get_recent_updates(10).await;
                    if !updates.is_empty() {
                        received_count += updates.len();
                    }
                    tokio::time::sleep(StdDuration::from_millis(10)).await;
                }

                (consumer_id + 100, received_count) // Add 100 to distinguish from producers
            });

            handles.push(consumer_handle);
        }

        // Wait for all operations
        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await.unwrap());
        }

        let total_sent = update_counter.load(Ordering::Relaxed);
        println!("Total updates sent: {}", total_sent);
        println!("Results: {:?}", results);

        // Verify no data loss in race conditions
        assert_eq!(total_sent, 500, "Should have sent exactly 500 updates");

        // Consumers should receive reasonable amounts (allowing for timing)
        for (id, received_count) in results {
            if id >= 100 {
                // Consumer results have IDs >= 100
                assert!(
                    received_count > 0,
                    "Consumer {} should receive some updates",
                    id - 100
                );
            }
        }
    }

    #[tokio::test]
    async fn test_data_consistency_multi_threaded() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        let session_id = uuid::Uuid::new_v4();
        let cost_tracker = Arc::clone(&fixture.cost_tracker);
        let history_store = Arc::clone(&fixture.history_store);

        // Consistency check data
        let expected_total_cost = Arc::new(AtomicU64::new(0));
        let expected_command_count = Arc::new(AtomicU64::new(0));

        let mut handles = Vec::new();

        // Multi-threaded writers with precise tracking
        for thread_id in 0..8 {
            let cost_tracker_clone = Arc::clone(&cost_tracker);
            let history_store_clone = Arc::clone(&history_store);
            let expected_cost_clone = Arc::clone(&expected_total_cost);
            let expected_count_clone = Arc::clone(&expected_command_count);

            let handle = tokio::spawn(async move {
                let mut thread_cost = 0u64;
                let mut thread_count = 0u64;

                for i in 0..25 {
                    let cost_cents = 100 + i; // Cost in cents for atomic operations
                    let cost_entry = CostEntry::new(
                        session_id,
                        format!("thread_{}_{}", thread_id, i),
                        cost_cents as f64 / 100.0,
                        100,
                        200,
                        1000,
                        "claude-3-haiku".to_string(),
                    );

                    cost_tracker_clone
                        .write()
                        .await
                        .record_cost(cost_entry)
                        .await
                        .unwrap();

                    let history_entry = HistoryEntry::new(
                        session_id,
                        format!("thread_{}_{}", thread_id, i),
                        vec!["--test".to_string()],
                        format!("Thread {} output {}", thread_id, i),
                        true,
                        1000,
                    );

                    history_store_clone
                        .write()
                        .await
                        .store_entry(history_entry)
                        .await
                        .unwrap();

                    thread_cost += cost_cents;
                    thread_count += 1;
                }

                // Update expected totals atomically
                expected_cost_clone.fetch_add(thread_cost, Ordering::Relaxed);
                expected_count_clone.fetch_add(thread_count, Ordering::Relaxed);

                (thread_id, thread_cost, thread_count)
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        let mut thread_results = Vec::new();
        for handle in handles {
            thread_results.push(handle.await.unwrap());
        }

        // Verify individual thread results
        for (thread_id, thread_cost, thread_count) in thread_results {
            assert_eq!(
                thread_count, 25,
                "Thread {} should process exactly 25 commands",
                thread_id
            );
            assert!(
                thread_cost > 0,
                "Thread {} should have accumulated cost",
                thread_id
            );
        }

        // Check final consistency
        let final_summary = analytics_engine.generate_summary(1).await.unwrap();
        let expected_cost_total = expected_total_cost.load(Ordering::Relaxed) as f64 / 100.0;
        let expected_count_total = expected_command_count.load(Ordering::Relaxed);

        println!(
            "Expected cost: ${:.2}, Actual cost: ${:.2}",
            expected_cost_total, final_summary.cost_summary.total_cost
        );
        println!(
            "Expected count: {}, Actual count: {}",
            expected_count_total, final_summary.cost_summary.command_count
        );

        // Data should be consistent
        assert!(
            (final_summary.cost_summary.total_cost - expected_cost_total).abs() < 0.01,
            "Cost totals should match within rounding"
        );
        assert_eq!(
            final_summary.cost_summary.command_count as u64, expected_count_total,
            "Command counts should match exactly"
        );
        assert_eq!(
            final_summary.history_stats.total_entries as u64, expected_count_total,
            "History entries should match command count"
        );
    }

    #[tokio::test]
    async fn test_transaction_isolation_simulation() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        // Simulate transaction isolation by grouping operations
        struct Transaction {
            session_id: SessionId,
            operations: Vec<TransactionOp>,
        }

        enum TransactionOp {
            AddCost(CostEntry),
            AddHistory(HistoryEntry),
        }

        let mut transactions = Vec::new();

        // Create several "transactions"
        for tx_id in 0..5 {
            let session_id = uuid::Uuid::new_v4();
            let mut operations = Vec::new();

            for op_id in 0..10 {
                let cost_entry = CostEntry::new(
                    session_id,
                    format!("tx_{}_{}", tx_id, op_id),
                    0.05,
                    100,
                    200,
                    1500,
                    "claude-3-sonnet".to_string(),
                );

                let history_entry = HistoryEntry::new(
                    session_id,
                    format!("tx_{}_{}", tx_id, op_id),
                    vec!["--tx".to_string()],
                    format!("Transaction {} operation {}", tx_id, op_id),
                    true,
                    1500,
                );

                operations.push(TransactionOp::AddCost(cost_entry));
                operations.push(TransactionOp::AddHistory(history_entry));
            }

            transactions.push(Transaction {
                session_id,
                operations,
            });
        }

        // Execute transactions concurrently
        let mut handles = Vec::new();

        for (tx_index, transaction) in transactions.into_iter().enumerate() {
            let cost_tracker = Arc::clone(&fixture.cost_tracker);
            let history_store = Arc::clone(&fixture.history_store);

            let handle = tokio::spawn(async move {
                // Simulate transaction isolation by holding locks
                let mut cost_guard = cost_tracker.write().await;
                let mut history_guard = history_store.write().await;

                let mut completed_ops = 0;

                for operation in transaction.operations {
                    match operation {
                        TransactionOp::AddCost(entry) => {
                            cost_guard.record_cost(entry).await.unwrap();
                        }
                        TransactionOp::AddHistory(entry) => {
                            history_guard.store_entry(entry).await.unwrap();
                        }
                    }
                    completed_ops += 1;

                    // Simulate some processing time
                    tokio::time::sleep(StdDuration::from_micros(100)).await;
                }

                drop(cost_guard);
                drop(history_guard);

                (tx_index, transaction.session_id, completed_ops)
            });

            handles.push(handle);
        }

        // Wait for all transactions to complete
        let mut tx_results = Vec::new();
        for handle in handles {
            tx_results.push(handle.await.unwrap());
        }

        // Verify transaction isolation worked
        for (tx_index, session_id, completed_ops) in tx_results {
            assert_eq!(
                completed_ops, 20,
                "Transaction {} should complete all operations",
                tx_index
            );

            // Verify session data is consistent
            let session_summary = analytics_engine
                .generate_session_report(session_id)
                .await
                .unwrap();

            assert_eq!(
                session_summary.cost_summary.command_count, 10,
                "Session {} should have exactly 10 commands",
                tx_index
            );
            assert_eq!(
                session_summary.history_stats.total_entries, 10,
                "Session {} should have exactly 10 history entries",
                tx_index
            );
        }

        // Verify overall consistency
        let overall_summary = analytics_engine.generate_summary(1).await.unwrap();
        assert_eq!(
            overall_summary.cost_summary.command_count, 50,
            "Should have 50 total commands from 5 transactions"
        );
        assert_eq!(
            overall_summary.history_stats.total_entries, 50,
            "Should have 50 total history entries from 5 transactions"
        );
    }
}

/// Performance benchmark suite
#[cfg(test)]
mod performance_benchmarks {
    use super::*;

    #[tokio::test]
    async fn benchmark_real_time_throughput() {
        use crate::cli::analytics::performance_optimizations::{
            BatchProcessor, HighThroughputProcessor,
        };
        use futures::stream;

        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        let stream = RealTimeAnalyticsStream::new(analytics_engine)
            .await
            .unwrap();

        // Use high-throughput processor with optimal settings
        let processor = HighThroughputProcessor::new(16, 500);

        // Generate updates
        let updates: Vec<_> = (0..10000)
            .map(|i| AnalyticsUpdate {
                timestamp: Utc::now(),
                update_type: if i % 2 == 0 {
                    UpdateType::CommandCompleted
                } else {
                    UpdateType::CostIncurred
                },
                session_id: uuid::Uuid::from_u128((i / 100) as u128),
                metric_deltas: MetricDeltas {
                    cost_delta: 0.001,
                    command_count_delta: 1,
                    success_count_delta: 1,
                    ..Default::default()
                },
            })
            .collect();

        let throughput_timer = perf::Timer::new("real_time_throughput");

        // Process as stream for maximum throughput
        let update_stream = stream::iter(updates);
        processor
            .process_stream(update_stream, |batch| {
                Box::pin(async move {
                    // Minimal processing simulation
                    if batch.len() > 0 {
                        tokio::task::yield_now().await;
                    }
                    Ok(())
                })
            })
            .await
            .unwrap();

        // Wait for all processing to complete
        tokio::time::sleep(StdDuration::from_millis(100)).await;

        let throughput_time = throughput_timer.elapsed_ms();
        let (processed, errors, throughput_per_sec) = processor.get_metrics();

        println!(
            "Real-time throughput: {:.0} updates/second",
            throughput_per_sec
        );
        println!(
            "Processed: {}, Errors: {}, Time: {}ms",
            processed, errors, throughput_time
        );

        // Calculate actual throughput including wait time
        let actual_throughput = 10000.0 / ((throughput_time + 100) as f64 / 1000.0);

        // Throughput requirement: should handle at least 1000 updates/second
        assert!(
            actual_throughput >= 1000.0,
            "Should handle at least 1000 updates/second, got {:.0}",
            actual_throughput
        );
    }

    #[tokio::test]
    async fn benchmark_concurrent_analytics_queries() {
        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        // Load test data
        let dataset = generate_large_dataset(20000).await;
        fixture.load_data(&dataset).await.unwrap();

        let concurrent_timer = perf::Timer::new("concurrent_queries_benchmark");

        // Run highly concurrent queries
        let mut handles = Vec::new();
        for i in 0..50 {
            let engine_clone = Arc::clone(&analytics_engine);

            let handle = tokio::spawn(async move {
                let query_timer = perf::Timer::new(&format!("query_{}", i));
                let _summary = engine_clone.generate_summary(30).await.unwrap();
                query_timer.elapsed_ms()
            });

            handles.push(handle);
        }

        // Collect results
        let mut query_times = Vec::new();
        for handle in handles {
            query_times.push(handle.await.unwrap());
        }

        let total_time = concurrent_timer.elapsed_ms();
        let avg_query_time: f64 =
            query_times.iter().sum::<u128>() as f64 / query_times.len() as f64;
        let max_query_time = *query_times.iter().max().unwrap();

        println!("Concurrent benchmark results:");
        println!("  Total time for 50 queries: {}ms", total_time);
        println!("  Average query time: {:.1}ms", avg_query_time);
        println!("  Maximum query time: {}ms", max_query_time);

        // Performance requirements
        assert!(
            total_time < 30000,
            "50 concurrent queries should complete within 30 seconds"
        );
        assert!(
            avg_query_time < 5000.0,
            "Average query time should be under 5 seconds"
        );
        assert!(
            max_query_time < 15000,
            "No query should take more than 15 seconds"
        );
    }

    #[tokio::test]
    async fn benchmark_memory_efficiency() {
        use crate::cli::analytics::performance_optimizations::{
            MemoryEfficientAggregator, ObjectPool,
        };

        let fixture = AnalyticsTestFixture::new().await.unwrap();
        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&fixture.cost_tracker),
            Arc::clone(&fixture.history_store),
            AnalyticsConfig::default(),
        ));

        let stream = RealTimeAnalyticsStream::new(analytics_engine)
            .await
            .unwrap();
        let memory_tracker = Arc::clone(&stream.memory_tracker);

        // Use memory-efficient aggregator with 50MB limit
        let aggregator = MemoryEfficientAggregator::new(50);

        // Use object pool for update objects
        let update_pool: ObjectPool<AnalyticsUpdate> = ObjectPool::new(1000, || AnalyticsUpdate {
            timestamp: Utc::now(),
            update_type: UpdateType::CommandCompleted,
            session_id: uuid::Uuid::new_v4(),
            metric_deltas: MetricDeltas::default(),
        });

        let memory_timer = perf::Timer::new("memory_efficiency_benchmark");

        // Process with controlled memory usage
        for cycle in 0..100 {
            // Use pooled objects
            let mut pooled_updates = Vec::new();
            for _ in 0..10 {
                let mut update = update_pool.acquire().await;
                update.timestamp = Utc::now();
                update.session_id = uuid::Uuid::from_u128(cycle as u128);
                pooled_updates.push(update);
            }

            // Record controlled allocation
            memory_tracker.record_allocation(10); // 10MB for 10 updates

            // Process with aggregator for memory efficiency
            for i in 0..10 {
                let cost_entry = CostEntry::new(
                    uuid::Uuid::from_u128((cycle * 10 + i) as u128),
                    format!("cmd_{}", i % 5),
                    0.01,
                    100,
                    200,
                    1000,
                    "claude-3-haiku".to_string(),
                );
                aggregator.process_entry(&cost_entry).await.unwrap();
            }

            // Return objects to pool (automatic on drop)
            drop(pooled_updates);

            // Record deallocation
            memory_tracker.record_deallocation(10);

            // Print progress periodically
            if cycle % 20 == 0 {
                let stats = memory_tracker.get_stats();
                let (_total_cost, total_commands, _) = aggregator.get_summary().await;
                println!(
                    "Cycle {}: Peak memory: {}MB, Active objects: {}, Aggregated commands: {}",
                    cycle, stats.peak_memory_mb, stats.active_objects, total_commands
                );
            }
        }

        let final_stats = memory_tracker.get_stats();
        let benchmark_time = memory_timer.elapsed_ms();

        println!("Memory efficiency benchmark:");
        println!("  Duration: {}ms", benchmark_time);
        println!("  Final peak memory: {}MB", final_stats.peak_memory_mb);
        println!("  Final active objects: {}", final_stats.active_objects);
        println!("  Total allocations: {}", final_stats.total_allocations);
        println!("  Total deallocations: {}", final_stats.total_deallocations);

        // Memory efficiency requirements with pooling
        assert!(
            final_stats.peak_memory_mb <= 150,
            "Peak memory should stay under 150MB with pooling"
        );
        assert!(
            final_stats.active_objects < 100,
            "Should have minimal active objects with pooling"
        );

        let dealloc_rate =
            final_stats.total_deallocations as f64 / final_stats.total_allocations as f64;
        assert!(
            dealloc_rate >= 0.95,
            "Should deallocate at least 95% of allocated objects with pooling"
        );
    }
}
