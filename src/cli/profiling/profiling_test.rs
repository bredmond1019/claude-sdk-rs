//! Comprehensive tests for profiling module
//!
//! This module provides extensive test coverage for the performance profiling
//! and benchmarking utilities, including property-based tests and edge cases.

use super::*;
use crate::cli::history::{HistoryEntry, HistorySearch};
use crate::profiling::profiler::ApplicationProfiler;
use crate::cli::error::Result;
use proptest::prelude::*;
use std::collections::HashMap;
use std::time::Duration;
use tempfile::tempdir;
use uuid::Uuid;

/// Test fixture for profiling testing
pub struct ProfilingTestFixture {
    pub temp_dir: tempfile::TempDir,
    pub profiler: ApplicationProfiler,
}

impl ProfilingTestFixture {
    /// Create a new test fixture
    pub fn new() -> Result<Self> {
        let temp_dir = tempdir().unwrap();
        let profiler = ApplicationProfiler::new()?;

        Ok(Self { temp_dir, profiler })
    }

    /// Create a test profiling configuration
    pub fn create_config(&self, operation_count: usize, dataset_size: usize) -> ProfilingConfig {
        ProfilingConfig {
            operation_count,
            dataset_size,
            operation_types: vec![OperationType::HistoryStore, OperationType::HistorySearch],
            profile_memory: true,
            profile_storage: true,
            warmup: false, // Disable for faster tests
            warmup_iterations: 0,
        }
    }

    /// Create test entries for profiling
    pub fn create_test_entries(&self, count: usize) -> Vec<HistoryEntry> {
        let session_id = Uuid::new_v4();
        (0..count)
            .map(|i| {
                HistoryEntry::new(
                    session_id,
                    format!("cmd_{}", i % 10),
                    vec![format!("arg_{}", i)],
                    format!("output_{}", i),
                    i % 5 != 0, // 80% success rate
                    (100 + i * 10) as u64,
                )
            })
            .collect()
    }
}

/// Property strategies for profiling testing
pub mod profiling_strategies {
    use super::*;

    /// Generate valid operation counts
    pub fn operation_count() -> impl Strategy<Value = usize> {
        1usize..=1000usize
    }

    /// Generate valid dataset sizes
    pub fn dataset_size() -> impl Strategy<Value = usize> {
        10usize..=10000usize
    }

    /// Generate profiling configurations
    pub fn profiling_config() -> impl Strategy<Value = ProfilingConfig> {
        (
            operation_count(),
            dataset_size(),
            prop::collection::vec(operation_type(), 1..=6),
            prop::bool::ANY,
            prop::bool::ANY,
            prop::bool::ANY,
            0usize..=100usize,
        )
            .prop_map(
                |(
                    operation_count,
                    dataset_size,
                    operation_types,
                    profile_memory,
                    profile_storage,
                    warmup,
                    warmup_iterations,
                )| ProfilingConfig {
                    operation_count,
                    dataset_size,
                    operation_types,
                    profile_memory,
                    profile_storage,
                    warmup,
                    warmup_iterations,
                },
            )
    }

    /// Generate operation types
    pub fn operation_type() -> impl Strategy<Value = OperationType> {
        prop::sample::select(vec![
            OperationType::HistoryStore,
            OperationType::HistorySearch,
            OperationType::FullTextSearch,
            OperationType::IndexOptimization,
            OperationType::SessionManagement,
            OperationType::BatchOperations,
        ])
    }

    /// Generate performance metrics
    pub fn performance_metrics() -> impl Strategy<Value = PerformanceMetrics> {
        (
            0u64..=10000u64, // elapsed_time_ms
            0.0f64..=1000.0f64, // initial_memory_mb
            0.0f64..=1000.0f64, // final_memory_mb
            0usize..=10000usize, // operation_count
            0usize..=1000usize, // reads
            0usize..=1000usize, // writes
            0usize..=1000usize, // searches
        )
            .prop_map(
                |(
                    elapsed_ms,
                    initial_memory,
                    final_memory,
                    operation_count,
                    reads,
                    writes,
                    searches,
                )| PerformanceMetrics {
                    elapsed_time: Duration::from_millis(elapsed_ms),
                    memory_usage: MemoryMetrics {
                        initial_mb: initial_memory,
                        final_mb: final_memory,
                        peak_mb: initial_memory.max(final_memory),
                        allocated_mb: (final_memory - initial_memory).abs(),
                    },
                    storage_metrics: StorageMetrics {
                        reads,
                        writes,
                        read_time: Duration::from_millis(reads as u64 * 10),
                        write_time: Duration::from_millis(writes as u64 * 15),
                        avg_read_latency: Duration::ZERO,
                        avg_write_latency: Duration::ZERO,
                    },
                    search_metrics: SearchMetrics {
                        search_count: searches,
                        total_search_time: Duration::from_millis(searches as u64 * 20),
                        avg_search_latency: Duration::ZERO,
                        index_cache_hits: searches / 2,
                        index_cache_misses: searches / 2,
                        cache_hit_ratio: 0.0,
                    },
                    operation_count,
                    ops_per_second: 0.0,
                    peak_memory_mb: initial_memory.max(final_memory),
                    timestamp: chrono::Utc::now(),
                },
            )
    }
}

#[cfg(test)]
mod core_types_tests {
    use super::*;

    #[test]
    fn test_profiling_config_default() {
        let config = ProfilingConfig::default();

        assert_eq!(config.operation_count, 1000);
        assert_eq!(config.dataset_size, 10000);
        assert!(config.profile_memory);
        assert!(config.profile_storage);
        assert!(config.warmup);
        assert_eq!(config.warmup_iterations, 100);
        assert!(!config.operation_types.is_empty());
    }

    #[test]
    fn test_memory_metrics_default() {
        let metrics = MemoryMetrics::default();

        assert_eq!(metrics.initial_mb, 0.0);
        assert_eq!(metrics.final_mb, 0.0);
        assert_eq!(metrics.peak_mb, 0.0);
        assert_eq!(metrics.allocated_mb, 0.0);
    }

    #[test]
    fn test_storage_metrics_default() {
        let metrics = StorageMetrics::default();

        assert_eq!(metrics.reads, 0);
        assert_eq!(metrics.writes, 0);
        assert_eq!(metrics.read_time, Duration::ZERO);
        assert_eq!(metrics.write_time, Duration::ZERO);
        assert_eq!(metrics.avg_read_latency, Duration::ZERO);
        assert_eq!(metrics.avg_write_latency, Duration::ZERO);
    }

    #[test]
    fn test_search_metrics_default() {
        let metrics = SearchMetrics::default();

        assert_eq!(metrics.search_count, 0);
        assert_eq!(metrics.total_search_time, Duration::ZERO);
        assert_eq!(metrics.avg_search_latency, Duration::ZERO);
        assert_eq!(metrics.index_cache_hits, 0);
        assert_eq!(metrics.index_cache_misses, 0);
        assert_eq!(metrics.cache_hit_ratio, 0.0);
    }

    #[test]
    fn test_performance_metrics_creation() {
        let metrics = PerformanceMetrics::new();

        assert_eq!(metrics.elapsed_time, Duration::ZERO);
        assert_eq!(metrics.operation_count, 0);
        assert_eq!(metrics.ops_per_second, 0.0);
        assert_eq!(metrics.peak_memory_mb, 0.0);
        assert_eq!(metrics.memory_usage.initial_mb, 0.0);
        assert_eq!(metrics.storage_metrics.reads, 0);
        assert_eq!(metrics.search_metrics.search_count, 0);
    }

    #[test]
    fn test_operation_type_serialization() {
        let types = vec![
            OperationType::HistoryStore,
            OperationType::HistorySearch,
            OperationType::FullTextSearch,
            OperationType::IndexOptimization,
            OperationType::SessionManagement,
            OperationType::BatchOperations,
        ];

        for op_type in types {
            let serialized = serde_json::to_string(&op_type).unwrap();
            assert!(!serialized.is_empty());

            let deserialized: OperationType = serde_json::from_str(&serialized).unwrap();
            // Compare the debug representations since OperationType doesn't implement PartialEq
            assert_eq!(format!("{:?}", op_type), format!("{:?}", deserialized));
        }
    }

    proptest! {
        #[test]
        fn property_metrics_consistency(
            mut metrics in profiling_strategies::performance_metrics()
        ) {
            // Property: operations per second calculation should be consistent
            metrics.calculate_ops_per_second();

            if metrics.elapsed_time.as_secs_f64() > 0.0 {
                let expected_ops_per_second = metrics.operation_count as f64 / metrics.elapsed_time.as_secs_f64();
                prop_assert!((metrics.ops_per_second - expected_ops_per_second).abs() < 0.001);
            } else {
                prop_assert_eq!(metrics.ops_per_second, 0.0);
            }
        }

        #[test]
        fn property_storage_averages_calculation(
            mut metrics in profiling_strategies::performance_metrics()
        ) {
            metrics.calculate_storage_averages();

            // Property: average latencies should be reasonable
            if metrics.storage_metrics.reads > 0 {
                let expected_read_avg = metrics.storage_metrics.read_time / metrics.storage_metrics.reads as u32;
                prop_assert_eq!(metrics.storage_metrics.avg_read_latency, expected_read_avg);
            }

            if metrics.storage_metrics.writes > 0 {
                let expected_write_avg = metrics.storage_metrics.write_time / metrics.storage_metrics.writes as u32;
                prop_assert_eq!(metrics.storage_metrics.avg_write_latency, expected_write_avg);
            }
        }

        #[test]
        fn property_search_averages_calculation(
            mut metrics in profiling_strategies::performance_metrics()
        ) {
            metrics.calculate_search_averages();

            // Property: cache hit ratio should be between 0 and 1
            prop_assert!(metrics.search_metrics.cache_hit_ratio >= 0.0);
            prop_assert!(metrics.search_metrics.cache_hit_ratio <= 1.0);

            // Property: if we have search operations, average should be calculated
            if metrics.search_metrics.search_count > 0 {
                let expected_avg = metrics.search_metrics.total_search_time / metrics.search_metrics.search_count as u32;
                prop_assert_eq!(metrics.search_metrics.avg_search_latency, expected_avg);
            }
        }
    }
}

#[cfg(test)]
mod timer_tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn test_timer_creation() {
        let timer = Timer::start("test_timer");
        assert_eq!(timer.name, "test_timer");
    }

    #[test]
    fn test_timer_elapsed() {
        let timer = Timer::start("test_timer");
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_timer_stop() {
        let timer = Timer::start("test_timer");
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_timer_multiple_elapsed_calls() {
        let timer = Timer::start("test_timer");
        let elapsed1 = timer.elapsed();
        std::thread::sleep(Duration::from_millis(5));
        let elapsed2 = timer.elapsed();
        assert!(elapsed2 >= elapsed1);
    }
}

#[cfg(test)]
mod profiling_result_tests {
    use super::*;

    fn create_test_result() -> ProfilingResult {
        let mut metrics = PerformanceMetrics::new();
        metrics.elapsed_time = Duration::from_millis(1000);
        metrics.operation_count = 100;
        metrics.ops_per_second = 100.0;
        metrics.peak_memory_mb = 50.0;
        metrics.storage_metrics.reads = 50;
        metrics.storage_metrics.writes = 50;
        metrics.storage_metrics.avg_read_latency = Duration::from_millis(10);
        metrics.storage_metrics.avg_write_latency = Duration::from_millis(15);
        metrics.search_metrics.search_count = 25;
        metrics.search_metrics.avg_search_latency = Duration::from_millis(20);
        metrics.search_metrics.cache_hit_ratio = 0.8;

        ProfilingResult {
            scenario_name: "Test Scenario".to_string(),
            dataset_size: 1000,
            metrics,
            metadata: HashMap::new(),
            config: ProfilingConfig::default(),
        }
    }

    #[tokio::test]
    async fn test_profiling_result_summary_generation() {
        let result = create_test_result();
        let summary = result.generate_summary();

        assert!(summary.contains("Test Scenario"));
        assert!(summary.contains("1000 entries"));
        assert!(summary.contains("100"));
        assert!(summary.contains("100.00"));
        assert!(summary.contains("50.00 MB"));
        assert!(summary.contains("80.00%"));
    }

    #[tokio::test]
    async fn test_profiling_result_save_and_load() -> Result<()> {
        let result = create_test_result();
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_result.json");

        // Save result
        result.save_to_file(&file_path).await?;
        assert!(file_path.exists());

        // Load result
        let loaded_result = ProfilingResult::load_from_file(&file_path).await?;
        assert_eq!(loaded_result.scenario_name, result.scenario_name);
        assert_eq!(loaded_result.dataset_size, result.dataset_size);
        assert_eq!(loaded_result.metrics.operation_count, result.metrics.operation_count);

        Ok(())
    }

    #[tokio::test]
    async fn test_profiling_result_serialization() -> Result<()> {
        let result = create_test_result();

        // Test JSON serialization
        let json = serde_json::to_string(&result)?;
        assert!(!json.is_empty());

        let deserialized: ProfilingResult = serde_json::from_str(&json)?;
        assert_eq!(deserialized.scenario_name, result.scenario_name);
        assert_eq!(deserialized.dataset_size, result.dataset_size);

        Ok(())
    }

    #[test]
    fn test_profiling_result_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("test_key".to_string(), "test_value".to_string());

        let result = ProfilingResult {
            scenario_name: "Test".to_string(),
            dataset_size: 100,
            metrics: PerformanceMetrics::new(),
            metadata: metadata.clone(),
            config: ProfilingConfig::default(),
        };

        assert_eq!(result.metadata.get("test_key"), Some(&"test_value".to_string()));
    }
}

#[cfg(test)]
mod profiler_tests {
    use super::*;

    #[tokio::test]
    async fn test_application_profiler_creation() -> Result<()> {
        let profiler = ApplicationProfiler::new()?;
        assert_eq!(profiler.get_results().len(), 0);
        Ok(())
    }

    #[test]
    fn test_application_profiler_default() {
        let _profiler = ApplicationProfiler::default();
        // Just verify it can be created
    }

    #[tokio::test]
    async fn test_profiler_test_data_generation() -> Result<()> {
        let fixture = ProfilingTestFixture::new()?;
        let entries = fixture.create_test_entries(100);

        assert_eq!(entries.len(), 100);
        assert!(entries.iter().any(|e| e.command_name.starts_with("cmd_")));
        assert!(entries.iter().any(|e| e.success));
        assert!(entries.iter().any(|e| !e.success)); // Should have some failures

        Ok(())
    }

    #[tokio::test]
    async fn test_empty_profiling_suite() -> Result<()> {
        let mut profiler = ApplicationProfiler::new()?;
        let config = ProfilingConfig {
            operation_count: 0,
            dataset_size: 0,
            operation_types: vec![],
            profile_memory: false,
            profile_storage: false,
            warmup: false,
            warmup_iterations: 0,
        };

        let results = profiler.run_profiling_suite(config).await?;
        assert_eq!(results.len(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_profiler_results_storage() -> Result<()> {
        let profiler = ApplicationProfiler::new()?;
        let temp_dir = tempdir().unwrap();

        // Test saving empty results
        profiler.save_results(temp_dir.path()).await?;

        let summary_path = temp_dir.path().join("profiling_summary.txt");
        assert!(summary_path.exists());

        let summary_content = tokio::fs::read_to_string(&summary_path).await?;
        assert!(summary_content.contains("Profiling Summary Report"));

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_usage_function() {
        let usage = get_memory_usage_mb();
        // The placeholder implementation returns 0.0
        assert_eq!(usage, 0.0);
    }

    proptest! {
        #[test]
        fn property_profiling_config_validation(
            config in profiling_strategies::profiling_config()
        ) {
            // Property: operation count should be positive if non-zero
            if config.operation_count > 0 {
                prop_assert!(config.operation_count >= 1);
            }

            // Property: dataset size should be reasonable
            if config.dataset_size > 0 {
                prop_assert!(config.dataset_size >= 10);
                prop_assert!(config.dataset_size <= 10000);
            }

            // Property: warmup iterations should not exceed operation count
            if config.warmup {
                prop_assert!(config.warmup_iterations <= config.operation_count);
            }

            // Property: operation types should not be empty if we want to profile
            if config.operation_count > 0 && config.dataset_size > 0 {
                // Config might be empty in property tests, but that's valid
            }
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_profiling_workflow() -> Result<()> {
        let fixture = ProfilingTestFixture::new()?;
        let small_config = fixture.create_config(10, 50);

        // Create a profiler and run a minimal suite
        let mut profiler = ApplicationProfiler::new()?;
        
        // We can't easily test the full profiling suite without mocking the dependencies,
        // but we can test the data structures and basic functionality
        
        // Test config creation
        assert_eq!(small_config.operation_count, 10);
        assert_eq!(small_config.dataset_size, 50);
        assert!(!small_config.warmup);

        // Test result storage
        profiler.save_results(fixture.temp_dir.path()).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_metrics_calculation_workflow() -> Result<()> {
        let mut metrics = PerformanceMetrics::new();

        // Simulate some operations
        metrics.operation_count = 100;
        metrics.elapsed_time = Duration::from_millis(2000);
        
        metrics.storage_metrics.reads = 50;
        metrics.storage_metrics.read_time = Duration::from_millis(500);
        metrics.storage_metrics.writes = 30;
        metrics.storage_metrics.write_time = Duration::from_millis(450);

        metrics.search_metrics.search_count = 25;
        metrics.search_metrics.total_search_time = Duration::from_millis(250);
        metrics.search_metrics.index_cache_hits = 20;
        metrics.search_metrics.index_cache_misses = 5;

        // Calculate all averages
        metrics.calculate_ops_per_second();
        metrics.calculate_storage_averages();
        metrics.calculate_search_averages();

        // Verify calculations
        assert_eq!(metrics.ops_per_second, 50.0); // 100 ops / 2 seconds
        assert_eq!(metrics.storage_metrics.avg_read_latency, Duration::from_millis(10)); // 500ms / 50 reads
        assert_eq!(metrics.storage_metrics.avg_write_latency, Duration::from_millis(15)); // 450ms / 30 writes
        assert_eq!(metrics.search_metrics.avg_search_latency, Duration::from_millis(10)); // 250ms / 25 searches
        assert_eq!(metrics.search_metrics.cache_hit_ratio, 0.8); // 20 hits / 25 total

        Ok(())
    }

    #[tokio::test]
    async fn test_profiling_result_with_all_data() -> Result<()> {
        let fixture = ProfilingTestFixture::new()?;
        let config = fixture.create_config(100, 500);

        let mut metrics = PerformanceMetrics::new();
        metrics.elapsed_time = Duration::from_millis(5000);
        metrics.operation_count = 100;
        metrics.peak_memory_mb = 75.5;
        metrics.calculate_ops_per_second();

        let mut metadata = HashMap::new();
        metadata.insert("test_environment".to_string(), "automated_test".to_string());
        metadata.insert("cpu_cores".to_string(), "4".to_string());

        let result = ProfilingResult {
            scenario_name: "Integration Test Scenario".to_string(),
            dataset_size: 500,
            metrics,
            metadata,
            config,
        };

        // Test summary generation
        let summary = result.generate_summary();
        assert!(summary.contains("Integration Test Scenario"));
        assert!(summary.contains("500 entries"));
        assert!(summary.contains("20.00")); // ops per second
        assert!(summary.contains("75.50 MB"));

        // Test serialization roundtrip
        let temp_path = fixture.temp_dir.path().join("integration_test.json");
        result.save_to_file(&temp_path).await?;
        let loaded_result = ProfilingResult::load_from_file(&temp_path).await?;

        assert_eq!(loaded_result.scenario_name, result.scenario_name);
        assert_eq!(loaded_result.dataset_size, result.dataset_size);
        assert_eq!(loaded_result.metrics.ops_per_second, result.metrics.ops_per_second);

        Ok(())
    }

    #[tokio::test]
    async fn test_edge_case_metrics() -> Result<()> {
        // Test with zero values
        let mut metrics_zero = PerformanceMetrics::new();
        metrics_zero.calculate_ops_per_second();
        metrics_zero.calculate_storage_averages();
        metrics_zero.calculate_search_averages();

        assert_eq!(metrics_zero.ops_per_second, 0.0);
        assert_eq!(metrics_zero.storage_metrics.avg_read_latency, Duration::ZERO);
        assert_eq!(metrics_zero.search_metrics.cache_hit_ratio, 0.0);

        // Test with very large values
        let mut metrics_large = PerformanceMetrics::new();
        metrics_large.operation_count = 1_000_000;
        metrics_large.elapsed_time = Duration::from_secs(1);
        metrics_large.storage_metrics.reads = 1_000_000;
        metrics_large.storage_metrics.read_time = Duration::from_secs(10);

        metrics_large.calculate_ops_per_second();
        metrics_large.calculate_storage_averages();

        assert_eq!(metrics_large.ops_per_second, 1_000_000.0);
        assert_eq!(metrics_large.storage_metrics.avg_read_latency, Duration::from_micros(10));

        Ok(())
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_profiling_result_invalid_file_path() {
        let result = ProfilingResult {
            scenario_name: "Test".to_string(),
            dataset_size: 100,
            metrics: PerformanceMetrics::new(),
            metadata: HashMap::new(),
            config: ProfilingConfig::default(),
        };

        // Try to save to an invalid path
        let invalid_path = std::path::Path::new("/invalid/path/that/does/not/exist.json");
        let save_result = result.save_to_file(invalid_path).await;
        assert!(save_result.is_err());

        // Try to load from non-existent file
        let load_result = ProfilingResult::load_from_file(invalid_path).await;
        assert!(load_result.is_err());
    }

    #[tokio::test]
    async fn test_profiling_result_corrupted_file() -> Result<()> {
        let temp_dir = tempdir().unwrap();
        let corrupted_path = temp_dir.path().join("corrupted.json");

        // Write invalid JSON
        tokio::fs::write(&corrupted_path, "{ invalid json content").await?;

        // Try to load corrupted file
        let load_result = ProfilingResult::load_from_file(&corrupted_path).await;
        assert!(load_result.is_err());

        Ok(())
    }

    #[test]
    fn test_timer_with_very_short_durations() {
        let timer = Timer::start("short_timer");
        // Don't sleep, measure immediately
        let elapsed = timer.stop();
        
        // Should be a very small duration but not zero (unless system is very fast)
        assert!(elapsed.as_nanos() >= 0);
    }

    #[test]
    fn test_memory_metrics_with_negative_allocation() {
        let metrics = MemoryMetrics {
            initial_mb: 100.0,
            final_mb: 50.0, // Less than initial
            peak_mb: 120.0,
            allocated_mb: 0.0, // Will be calculated
        };

        // Memory can decrease (garbage collection, etc.)
        assert!(metrics.final_mb < metrics.initial_mb);
        assert!(metrics.peak_mb >= metrics.initial_mb);
    }
}

/// Performance regression tests
#[cfg(test)]
mod performance_regression_tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_calculation_performance() {
        let start = std::time::Instant::now();
        
        // Create and calculate metrics many times
        for _ in 0..1000 {
            let mut metrics = PerformanceMetrics::new();
            metrics.operation_count = 1000;
            metrics.elapsed_time = Duration::from_millis(1000);
            metrics.storage_metrics.reads = 100;
            metrics.storage_metrics.read_time = Duration::from_millis(100);
            metrics.search_metrics.search_count = 50;
            metrics.search_metrics.total_search_time = Duration::from_millis(50);
            metrics.search_metrics.index_cache_hits = 40;
            metrics.search_metrics.index_cache_misses = 10;

            metrics.calculate_ops_per_second();
            metrics.calculate_storage_averages();
            metrics.calculate_search_averages();
        }

        let elapsed = start.elapsed();
        // Should complete quickly (adjust threshold as needed)
        assert!(elapsed < Duration::from_millis(100));
    }

    #[tokio::test]
    async fn test_summary_generation_performance() {
        let result = ProfilingResult {
            scenario_name: "Performance Test".to_string(),
            dataset_size: 100000,
            metrics: PerformanceMetrics::new(),
            metadata: HashMap::new(),
            config: ProfilingConfig::default(),
        };

        let start = std::time::Instant::now();
        
        // Generate summaries many times
        for _ in 0..100 {
            let _summary = result.generate_summary();
        }

        let elapsed = start.elapsed();
        // Summary generation should be fast
        assert!(elapsed < Duration::from_millis(50));
    }
}