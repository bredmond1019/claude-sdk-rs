//! Performance profiling and benchmarking utilities
//!
//! This module provides tools for profiling the application with large datasets,
//! measuring performance bottlenecks, and generating performance reports.

pub mod benchmark;
pub mod metrics;
pub mod profiler;

// #[cfg(test)]
// pub mod profiling_test; // Temporarily disabled

use crate::cli::error::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance metrics collected during profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total elapsed time for the operation
    pub elapsed_time: Duration,
    /// Memory usage statistics
    pub memory_usage: MemoryMetrics,
    /// Database/storage operation metrics
    pub storage_metrics: StorageMetrics,
    /// Search operation metrics
    pub search_metrics: SearchMetrics,
    /// Number of operations performed
    pub operation_count: usize,
    /// Operations per second
    pub ops_per_second: f64,
    /// Peak memory usage during the operation
    pub peak_memory_mb: f64,
    /// Timestamp when metrics were collected
    pub timestamp: DateTime<Utc>,
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Initial memory usage in MB
    pub initial_mb: f64,
    /// Final memory usage in MB
    pub final_mb: f64,
    /// Peak memory usage in MB
    pub peak_mb: f64,
    /// Memory allocated during operation in MB
    pub allocated_mb: f64,
}

/// Storage operation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetrics {
    /// Number of read operations
    pub reads: usize,
    /// Number of write operations
    pub writes: usize,
    /// Total time spent on read operations
    pub read_time: Duration,
    /// Total time spent on write operations
    pub write_time: Duration,
    /// Average read latency
    pub avg_read_latency: Duration,
    /// Average write latency
    pub avg_write_latency: Duration,
}

/// Search operation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    /// Number of search operations
    pub search_count: usize,
    /// Total time spent on searches
    pub total_search_time: Duration,
    /// Average search latency
    pub avg_search_latency: Duration,
    /// Number of index cache hits
    pub index_cache_hits: usize,
    /// Number of index cache misses
    pub index_cache_misses: usize,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

/// Profiling result for a specific test scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingResult {
    /// Name of the test scenario
    pub scenario_name: String,
    /// Dataset size used for testing
    pub dataset_size: usize,
    /// Performance metrics collected
    pub metrics: PerformanceMetrics,
    /// Additional metadata about the test
    pub metadata: HashMap<String, String>,
    /// Test configuration parameters
    pub config: ProfilingConfig,
}

/// Configuration for profiling tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Number of operations to perform
    pub operation_count: usize,
    /// Size of dataset to generate/use
    pub dataset_size: usize,
    /// Type of operations to profile
    pub operation_types: Vec<OperationType>,
    /// Whether to include memory profiling
    pub profile_memory: bool,
    /// Whether to include storage profiling
    pub profile_storage: bool,
    /// Whether to warm up caches before profiling
    pub warmup: bool,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
}

/// Types of operations to profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    /// History entry storage operations
    HistoryStore,
    /// History search operations
    HistorySearch,
    /// Full-text search operations
    FullTextSearch,
    /// Index optimization operations
    IndexOptimization,
    /// Session management operations
    SessionManagement,
    /// Batch operations
    BatchOperations,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            operation_count: 1000,
            dataset_size: 10000,
            operation_types: vec![
                OperationType::HistoryStore,
                OperationType::HistorySearch,
                OperationType::FullTextSearch,
            ],
            profile_memory: true,
            profile_storage: true,
            warmup: true,
            warmup_iterations: 100,
        }
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            initial_mb: 0.0,
            final_mb: 0.0,
            peak_mb: 0.0,
            allocated_mb: 0.0,
        }
    }
}

impl Default for StorageMetrics {
    fn default() -> Self {
        Self {
            reads: 0,
            writes: 0,
            read_time: Duration::ZERO,
            write_time: Duration::ZERO,
            avg_read_latency: Duration::ZERO,
            avg_write_latency: Duration::ZERO,
        }
    }
}

impl Default for SearchMetrics {
    fn default() -> Self {
        Self {
            search_count: 0,
            total_search_time: Duration::ZERO,
            avg_search_latency: Duration::ZERO,
            index_cache_hits: 0,
            index_cache_misses: 0,
            cache_hit_ratio: 0.0,
        }
    }
}

impl PerformanceMetrics {
    /// Create new performance metrics with current timestamp
    pub fn new() -> Self {
        Self {
            elapsed_time: Duration::ZERO,
            memory_usage: MemoryMetrics::default(),
            storage_metrics: StorageMetrics::default(),
            search_metrics: SearchMetrics::default(),
            operation_count: 0,
            ops_per_second: 0.0,
            peak_memory_mb: 0.0,
            timestamp: Utc::now(),
        }
    }

    /// Calculate operations per second based on elapsed time and operation count
    pub fn calculate_ops_per_second(&mut self) {
        if self.elapsed_time.as_secs_f64() > 0.0 {
            self.ops_per_second = self.operation_count as f64 / self.elapsed_time.as_secs_f64();
        }
    }

    /// Calculate average latencies for storage operations
    pub fn calculate_storage_averages(&mut self) {
        if self.storage_metrics.reads > 0 {
            self.storage_metrics.avg_read_latency =
                self.storage_metrics.read_time / self.storage_metrics.reads as u32;
        }
        if self.storage_metrics.writes > 0 {
            self.storage_metrics.avg_write_latency =
                self.storage_metrics.write_time / self.storage_metrics.writes as u32;
        }
    }

    /// Calculate search metrics averages
    pub fn calculate_search_averages(&mut self) {
        if self.search_metrics.search_count > 0 {
            self.search_metrics.avg_search_latency =
                self.search_metrics.total_search_time / self.search_metrics.search_count as u32;
        }

        let total_cache_operations =
            self.search_metrics.index_cache_hits + self.search_metrics.index_cache_misses;
        if total_cache_operations > 0 {
            self.search_metrics.cache_hit_ratio =
                self.search_metrics.index_cache_hits as f64 / total_cache_operations as f64;
        }
    }
}

impl ProfilingResult {
    /// Generate a human-readable summary of the profiling results
    pub fn generate_summary(&self) -> String {
        format!(
            "Profiling Results: {}\n\
            Dataset Size: {} entries\n\
            Operations: {}\n\
            Total Time: {:?}\n\
            Ops/Second: {:.2}\n\
            Peak Memory: {:.2} MB\n\
            Storage Reads: {} (avg: {:?})\n\
            Storage Writes: {} (avg: {:?})\n\
            Search Operations: {} (avg: {:?})\n\
            Cache Hit Ratio: {:.2}%",
            self.scenario_name,
            self.dataset_size,
            self.metrics.operation_count,
            self.metrics.elapsed_time,
            self.metrics.ops_per_second,
            self.metrics.peak_memory_mb,
            self.metrics.storage_metrics.reads,
            self.metrics.storage_metrics.avg_read_latency,
            self.metrics.storage_metrics.writes,
            self.metrics.storage_metrics.avg_write_latency,
            self.metrics.search_metrics.search_count,
            self.metrics.search_metrics.avg_search_latency,
            self.metrics.search_metrics.cache_hit_ratio * 100.0
        )
    }

    /// Save profiling results to a JSON file
    pub async fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }

    /// Load profiling results from a JSON file
    pub async fn load_from_file(path: &std::path::Path) -> Result<Self> {
        let content = tokio::fs::read_to_string(path).await?;
        let result: ProfilingResult = serde_json::from_str(&content)?;
        Ok(result)
    }
}

/// Utility function to get current memory usage in MB
pub fn get_memory_usage_mb() -> f64 {
    // This is a simplified implementation
    // In a real implementation, you might use system-specific APIs
    // or crates like `sysinfo` for more accurate memory measurements
    0.0 // Placeholder
}

/// Timer utility for measuring operation durations
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    /// Start a new timer with the given name
    pub fn start(name: &str) -> Self {
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }

    /// Stop the timer and return the elapsed duration
    pub fn stop(self) -> Duration {
        let elapsed = self.start.elapsed();
        // log::debug!("Timer '{}' elapsed: {:?}", self.name, elapsed);
        elapsed
    }

    /// Get elapsed time without stopping the timer
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}
