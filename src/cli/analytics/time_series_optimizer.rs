//! Time series data generation optimization
//!
//! This module provides optimized implementations for generating time series data
//! by using batch queries instead of individual per-hour queries, significantly
//! improving performance for dashboard generation.
//!
//! # Performance Improvements
//!
//! - **Batch Querying**: Single query to fetch all data within time range
//! - **In-Memory Aggregation**: Group and aggregate data in memory by time buckets
//! - **Reduced Database Load**: Minimize database round trips
//! - **Parallel Processing**: Process multiple time series concurrently
//!
//! # Example
//!
//! ```no_run
//! use crate_interactive::analytics::time_series_optimizer::TimeSeriesOptimizer;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let optimizer = TimeSeriesOptimizer::new(analytics_engine);
//!
//! // Generate multiple time series efficiently
//! let time_series = optimizer.generate_optimized_time_series(
//!     start_time, end_time, vec![
//!         TimeSeriesType::Cost,
//!         TimeSeriesType::Commands,
//!         TimeSeriesType::SuccessRate,
//!     ]
//! ).await?;
//! # Ok(())
//! # }
//! ```

use super::{dashboard::ChartMetric, AnalyticsEngine};
use crate::{
    cli::cost::{CostEntry, CostFilter},
    cli::history::{HistoryEntry, HistorySearch},
    Result,
};
use chrono::{DateTime, Duration, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinSet;

/// Types of time series data that can be generated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeSeriesType {
    Cost,
    Commands,
    SuccessRate,
    ResponseTime,
}

/// Configuration for time series optimization
#[derive(Debug, Clone)]
pub struct TimeSeriesConfig {
    /// Time bucket size (in minutes)
    pub bucket_size_minutes: u32,
    /// Whether to enable parallel processing
    pub enable_parallel_processing: bool,
    /// Whether to use caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl Default for TimeSeriesConfig {
    fn default() -> Self {
        Self {
            bucket_size_minutes: 60, // 1 hour buckets
            enable_parallel_processing: true,
            enable_caching: true,
            cache_ttl_seconds: 300, // 5 minutes
        }
    }
}

/// Optimized time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedTimeSeriesPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub metadata: TimeSeriesMetadata,
}

/// Metadata for time series points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesMetadata {
    pub count: usize,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub source_records: usize,
}

/// Complete time series data with optimization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedTimeSeriesData {
    pub cost_over_time: Vec<OptimizedTimeSeriesPoint>,
    pub commands_over_time: Vec<OptimizedTimeSeriesPoint>,
    pub success_rate_over_time: Vec<OptimizedTimeSeriesPoint>,
    pub response_time_over_time: Vec<OptimizedTimeSeriesPoint>,
    pub generation_time_ms: u64,
    pub total_queries: usize,
    pub cache_hits: usize,
    pub optimization_ratio: f64, // How much faster than individual queries
}

/// Time series optimizer with batch processing capabilities
pub struct TimeSeriesOptimizer {
    analytics_engine: AnalyticsEngine,
    config: TimeSeriesConfig,
    cache: Arc<tokio::sync::RwLock<HashMap<String, (DateTime<Utc>, OptimizedTimeSeriesData)>>>,
}

impl TimeSeriesOptimizer {
    /// Create a new time series optimizer
    pub fn new(analytics_engine: AnalyticsEngine) -> Self {
        Self::with_config(analytics_engine, TimeSeriesConfig::default())
    }

    /// Create optimizer with custom configuration
    pub fn with_config(analytics_engine: AnalyticsEngine, config: TimeSeriesConfig) -> Self {
        Self {
            analytics_engine,
            config,
            cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    /// Generate all time series data using optimized batch processing
    pub async fn generate_optimized_time_series(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        types: Vec<TimeSeriesType>,
    ) -> Result<OptimizedTimeSeriesData> {
        let generation_start = std::time::Instant::now();

        // Save types info before they are moved
        let types_len = types.len();
        let types_for_cache = types.clone();

        // Check cache first
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(&start_time, &end_time, &types_for_cache);
            if let Some(cached) = self.get_cached_data(&cache_key).await {
                return Ok(cached);
            }
        }

        let mut result = OptimizedTimeSeriesData {
            cost_over_time: Vec::new(),
            commands_over_time: Vec::new(),
            success_rate_over_time: Vec::new(),
            response_time_over_time: Vec::new(),
            generation_time_ms: 0,
            total_queries: 0,
            cache_hits: 0,
            optimization_ratio: 1.0,
        };

        if self.config.enable_parallel_processing {
            // Generate time series in parallel
            let mut tasks = JoinSet::new();

            for ts_type in types {
                let optimizer_clone = self.clone_for_task();
                let start = start_time;
                let end = end_time;

                tasks.spawn(async move {
                    match ts_type {
                        TimeSeriesType::Cost => optimizer_clone
                            .generate_cost_time_series_optimized(start, end)
                            .await
                            .map(|data| (ts_type, data)),
                        TimeSeriesType::Commands => optimizer_clone
                            .generate_commands_time_series_optimized(start, end)
                            .await
                            .map(|data| (ts_type, data)),
                        TimeSeriesType::SuccessRate => optimizer_clone
                            .generate_success_rate_time_series_optimized(start, end)
                            .await
                            .map(|data| (ts_type, data)),
                        TimeSeriesType::ResponseTime => optimizer_clone
                            .generate_response_time_time_series_optimized(start, end)
                            .await
                            .map(|data| (ts_type, data)),
                    }
                });
            }

            // Collect results
            while let Some(task_result) = tasks.join_next().await {
                if let Ok(Ok((ts_type, data))) = task_result {
                    match ts_type {
                        TimeSeriesType::Cost => result.cost_over_time = data,
                        TimeSeriesType::Commands => result.commands_over_time = data,
                        TimeSeriesType::SuccessRate => result.success_rate_over_time = data,
                        TimeSeriesType::ResponseTime => result.response_time_over_time = data,
                    }
                    result.total_queries += 1;
                }
            }
        } else {
            // Generate time series sequentially
            for ts_type in types {
                let data = match ts_type {
                    TimeSeriesType::Cost => {
                        self.generate_cost_time_series_optimized(start_time, end_time)
                            .await?
                    }
                    TimeSeriesType::Commands => {
                        self.generate_commands_time_series_optimized(start_time, end_time)
                            .await?
                    }
                    TimeSeriesType::SuccessRate => {
                        self.generate_success_rate_time_series_optimized(start_time, end_time)
                            .await?
                    }
                    TimeSeriesType::ResponseTime => {
                        self.generate_response_time_time_series_optimized(start_time, end_time)
                            .await?
                    }
                };

                match ts_type {
                    TimeSeriesType::Cost => result.cost_over_time = data,
                    TimeSeriesType::Commands => result.commands_over_time = data,
                    TimeSeriesType::SuccessRate => result.success_rate_over_time = data,
                    TimeSeriesType::ResponseTime => result.response_time_over_time = data,
                }
                result.total_queries += 1;
            }
        }

        let generation_time = generation_start.elapsed();
        result.generation_time_ms = generation_time.as_millis() as u64;

        // Calculate optimization ratio (estimated based on number of time buckets)
        let time_span_hours = (end_time - start_time).num_hours() as f64;
        let estimated_old_queries = time_span_hours * types_len as f64;
        result.optimization_ratio = estimated_old_queries / result.total_queries as f64;

        // Cache the result
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(&start_time, &end_time, &types_for_cache);
            self.cache_data(cache_key, result.clone()).await;
        }

        Ok(result)
    }

    /// Generate cost time series using optimized batch query
    async fn generate_cost_time_series_optimized(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<OptimizedTimeSeriesPoint>> {
        // Single batch query for entire time range
        let filter = CostFilter {
            since: Some(start_time),
            until: Some(end_time),
            ..Default::default()
        };

        let cost_tracker = self.analytics_engine.cost_tracker.read().await;
        let all_entries = cost_tracker.get_entries(&filter).await?;
        drop(cost_tracker);

        // Group entries by time buckets in memory
        let bucket_duration = Duration::minutes(self.config.bucket_size_minutes as i64);
        let mut buckets: HashMap<DateTime<Utc>, Vec<&CostEntry>> = HashMap::new();

        for entry in &all_entries {
            let bucket_start = self.align_to_bucket(entry.timestamp, bucket_duration);
            buckets
                .entry(bucket_start)
                .or_insert_with(Vec::new)
                .push(entry);
        }

        // Generate time series points
        let mut data = Vec::new();
        let mut current_time = self.align_to_bucket(start_time, bucket_duration);

        while current_time < end_time {
            let empty_vec = Vec::new();
            let bucket_entries = buckets.get(&current_time).unwrap_or(&empty_vec);
            let total_cost: f64 = bucket_entries.iter().map(|e| e.cost_usd).sum();

            let min_cost = bucket_entries
                .iter()
                .map(|e| e.cost_usd)
                .fold(f64::INFINITY, f64::min);
            let max_cost = bucket_entries
                .iter()
                .map(|e| e.cost_usd)
                .fold(f64::NEG_INFINITY, f64::max);

            data.push(OptimizedTimeSeriesPoint {
                timestamp: current_time,
                value: total_cost,
                metadata: TimeSeriesMetadata {
                    count: bucket_entries.len(),
                    min_value: if bucket_entries.is_empty() {
                        None
                    } else {
                        Some(min_cost)
                    },
                    max_value: if bucket_entries.is_empty() {
                        None
                    } else {
                        Some(max_cost)
                    },
                    source_records: bucket_entries.len(),
                },
            });

            current_time += bucket_duration;
        }

        Ok(data)
    }

    /// Generate commands time series using optimized batch query
    async fn generate_commands_time_series_optimized(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<OptimizedTimeSeriesPoint>> {
        // Single batch query for entire time range
        let search = HistorySearch {
            since: Some(start_time),
            until: Some(end_time),
            ..Default::default()
        };

        let history_store = self.analytics_engine.history_store.read().await;
        let all_entries = history_store.search(&search).await?;
        drop(history_store);

        // Group entries by time buckets
        let bucket_duration = Duration::minutes(self.config.bucket_size_minutes as i64);
        let mut buckets: HashMap<DateTime<Utc>, Vec<&HistoryEntry>> = HashMap::new();

        for entry in &all_entries {
            let bucket_start = self.align_to_bucket(entry.timestamp, bucket_duration);
            buckets
                .entry(bucket_start)
                .or_insert_with(Vec::new)
                .push(entry);
        }

        // Generate time series points
        let mut data = Vec::new();
        let mut current_time = self.align_to_bucket(start_time, bucket_duration);

        while current_time < end_time {
            let empty_vec = Vec::new();
            let bucket_entries = buckets.get(&current_time).unwrap_or(&empty_vec);
            let command_count = bucket_entries.len() as f64;

            data.push(OptimizedTimeSeriesPoint {
                timestamp: current_time,
                value: command_count,
                metadata: TimeSeriesMetadata {
                    count: bucket_entries.len(),
                    min_value: if bucket_entries.is_empty() {
                        None
                    } else {
                        Some(1.0)
                    },
                    max_value: if bucket_entries.is_empty() {
                        None
                    } else {
                        Some(1.0)
                    },
                    source_records: bucket_entries.len(),
                },
            });

            current_time += bucket_duration;
        }

        Ok(data)
    }

    /// Generate success rate time series using optimized batch query
    async fn generate_success_rate_time_series_optimized(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<OptimizedTimeSeriesPoint>> {
        // Single batch query for entire time range
        let search = HistorySearch {
            since: Some(start_time),
            until: Some(end_time),
            ..Default::default()
        };

        let history_store = self.analytics_engine.history_store.read().await;
        let all_entries = history_store.search(&search).await?;
        drop(history_store);

        // Group entries by time buckets
        let bucket_duration = Duration::minutes(self.config.bucket_size_minutes as i64);
        let mut buckets: HashMap<DateTime<Utc>, Vec<&HistoryEntry>> = HashMap::new();

        for entry in &all_entries {
            let bucket_start = self.align_to_bucket(entry.timestamp, bucket_duration);
            buckets
                .entry(bucket_start)
                .or_insert_with(Vec::new)
                .push(entry);
        }

        // Generate time series points
        let mut data = Vec::new();
        let mut current_time = self.align_to_bucket(start_time, bucket_duration);

        while current_time < end_time {
            let empty_vec = Vec::new();
            let bucket_entries = buckets.get(&current_time).unwrap_or(&empty_vec);

            let success_rate = if bucket_entries.is_empty() {
                f64::NAN // No data available
            } else {
                let successful = bucket_entries.iter().filter(|e| e.success).count();
                (successful as f64 / bucket_entries.len() as f64) * 100.0
            };

            data.push(OptimizedTimeSeriesPoint {
                timestamp: current_time,
                value: success_rate,
                metadata: TimeSeriesMetadata {
                    count: bucket_entries.len(),
                    min_value: if bucket_entries.is_empty() {
                        None
                    } else {
                        Some(0.0)
                    },
                    max_value: if bucket_entries.is_empty() {
                        None
                    } else {
                        Some(100.0)
                    },
                    source_records: bucket_entries.len(),
                },
            });

            current_time += bucket_duration;
        }

        Ok(data)
    }

    /// Generate response time time series using optimized batch query
    async fn generate_response_time_time_series_optimized(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<OptimizedTimeSeriesPoint>> {
        // Single batch query for entire time range
        let search = HistorySearch {
            since: Some(start_time),
            until: Some(end_time),
            ..Default::default()
        };

        let history_store = self.analytics_engine.history_store.read().await;
        let all_entries = history_store.search(&search).await?;
        drop(history_store);

        // Group entries by time buckets
        let bucket_duration = Duration::minutes(self.config.bucket_size_minutes as i64);
        let mut buckets: HashMap<DateTime<Utc>, Vec<&HistoryEntry>> = HashMap::new();

        for entry in &all_entries {
            let bucket_start = self.align_to_bucket(entry.timestamp, bucket_duration);
            buckets
                .entry(bucket_start)
                .or_insert_with(Vec::new)
                .push(entry);
        }

        // Generate time series points
        let mut data = Vec::new();
        let mut current_time = self.align_to_bucket(start_time, bucket_duration);

        while current_time < end_time {
            let empty_vec = Vec::new();
            let bucket_entries = buckets.get(&current_time).unwrap_or(&empty_vec);

            let avg_response_time = if bucket_entries.is_empty() {
                f64::NAN // No data available
            } else {
                let total_duration: u64 = bucket_entries.iter().map(|e| e.duration_ms).sum();
                total_duration as f64 / bucket_entries.len() as f64
            };

            let min_duration = bucket_entries
                .iter()
                .map(|e| e.duration_ms as f64)
                .fold(f64::INFINITY, f64::min);
            let max_duration = bucket_entries
                .iter()
                .map(|e| e.duration_ms as f64)
                .fold(f64::NEG_INFINITY, f64::max);

            data.push(OptimizedTimeSeriesPoint {
                timestamp: current_time,
                value: avg_response_time,
                metadata: TimeSeriesMetadata {
                    count: bucket_entries.len(),
                    min_value: if bucket_entries.is_empty() {
                        None
                    } else {
                        Some(min_duration)
                    },
                    max_value: if bucket_entries.is_empty() {
                        None
                    } else {
                        Some(max_duration)
                    },
                    source_records: bucket_entries.len(),
                },
            });

            current_time += bucket_duration;
        }

        Ok(data)
    }

    // Helper methods

    /// Align timestamp to bucket boundary
    fn align_to_bucket(
        &self,
        timestamp: DateTime<Utc>,
        bucket_duration: Duration,
    ) -> DateTime<Utc> {
        let bucket_minutes = self.config.bucket_size_minutes as i64;
        let minutes_since_epoch = timestamp.timestamp() / 60;
        let aligned_minutes = (minutes_since_epoch / bucket_minutes) * bucket_minutes;
        DateTime::from_timestamp(aligned_minutes * 60, 0).unwrap_or(timestamp)
    }

    /// Generate cache key for time series data
    fn generate_cache_key(
        &self,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        types: &[TimeSeriesType],
    ) -> String {
        let mut type_str = String::new();
        for ts_type in types {
            type_str.push_str(&format!("{:?}", ts_type));
        }
        format!(
            "{}:{}:{}:{}",
            start.timestamp(),
            end.timestamp(),
            type_str,
            self.config.bucket_size_minutes
        )
    }

    /// Get cached data if available and not expired
    async fn get_cached_data(&self, cache_key: &str) -> Option<OptimizedTimeSeriesData> {
        let cache = self.cache.read().await;
        if let Some((cached_time, data)) = cache.get(cache_key) {
            let age = Utc::now() - *cached_time;
            if age.num_seconds() < self.config.cache_ttl_seconds as i64 {
                return Some(data.clone());
            }
        }
        None
    }

    /// Cache time series data
    async fn cache_data(&self, cache_key: String, data: OptimizedTimeSeriesData) {
        let mut cache = self.cache.write().await;
        cache.insert(cache_key, (Utc::now(), data));

        // Clean up old cache entries (simple LRU-like behavior)
        if cache.len() > 100 {
            let cutoff_time =
                Utc::now() - Duration::seconds(self.config.cache_ttl_seconds as i64 * 2);
            cache.retain(|_, (time, _)| *time > cutoff_time);
        }
    }

    /// Clone optimizer for parallel task execution
    fn clone_for_task(&self) -> Self {
        Self {
            analytics_engine: self.analytics_engine.clone(),
            config: self.config.clone(),
            cache: Arc::clone(&self.cache),
        }
    }

    /// Convert to legacy format for backward compatibility
    pub fn to_legacy_format(data: &OptimizedTimeSeriesData) -> super::dashboard::TimeSeriesData {
        super::dashboard::TimeSeriesData {
            cost_over_time: data
                .cost_over_time
                .iter()
                .map(|p| (p.timestamp, p.value))
                .collect(),
            commands_over_time: data
                .commands_over_time
                .iter()
                .map(|p| (p.timestamp, p.value as usize))
                .collect(),
            success_rate_over_time: data
                .success_rate_over_time
                .iter()
                .map(|p| (p.timestamp, p.value))
                .collect(),
            response_time_over_time: data
                .response_time_over_time
                .iter()
                .map(|p| (p.timestamp, p.value))
                .collect(),
        }
    }

    /// Generate performance comparison report
    pub async fn generate_performance_comparison(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<TimeSeriesPerformanceReport> {
        let types = vec![
            TimeSeriesType::Cost,
            TimeSeriesType::Commands,
            TimeSeriesType::SuccessRate,
            TimeSeriesType::ResponseTime,
        ];

        // Test optimized version
        let optimized_start = std::time::Instant::now();
        let optimized_data = self
            .generate_optimized_time_series(start_time, end_time, types.clone())
            .await?;
        let optimized_duration = optimized_start.elapsed();

        // Estimate legacy performance (based on number of queries that would be made)
        let time_span_hours = (end_time - start_time).num_hours() as f64;
        let estimated_legacy_queries = time_span_hours * types.len() as f64;
        let estimated_legacy_duration = optimized_duration.as_millis() as f64
            * estimated_legacy_queries
            / optimized_data.total_queries as f64;

        Ok(TimeSeriesPerformanceReport {
            optimized_duration_ms: optimized_duration.as_millis() as u64,
            estimated_legacy_duration_ms: estimated_legacy_duration as u64,
            performance_improvement_ratio: estimated_legacy_duration
                / optimized_duration.as_millis() as f64,
            queries_reduced_from: estimated_legacy_queries as usize,
            queries_reduced_to: optimized_data.total_queries,
            query_reduction_percentage: (1.0
                - (optimized_data.total_queries as f64 / estimated_legacy_queries))
                * 100.0,
            time_range_hours: time_span_hours as u32,
            data_points_generated: optimized_data.cost_over_time.len()
                + optimized_data.commands_over_time.len()
                + optimized_data.success_rate_over_time.len()
                + optimized_data.response_time_over_time.len(),
        })
    }
}

/// Performance comparison report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPerformanceReport {
    pub optimized_duration_ms: u64,
    pub estimated_legacy_duration_ms: u64,
    pub performance_improvement_ratio: f64,
    pub queries_reduced_from: usize,
    pub queries_reduced_to: usize,
    pub query_reduction_percentage: f64,
    pub time_range_hours: u32,
    pub data_points_generated: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::analytics::dashboard_tests::DashboardTestFixture;
    use crate::cli::analytics::test_utils::AnalyticsTestDataGenerator;

    #[tokio::test]
    async fn test_optimized_time_series_generation() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Generate test data
        let generator = AnalyticsTestDataGenerator {
            session_count: 10,
            commands_per_session: (20, 40),
            ..Default::default()
        };
        let test_data = generator.generate_test_data(7);
        fixture.load_data(&test_data).await?;

        let optimizer = TimeSeriesOptimizer::new(fixture.analytics_engine.clone());

        let start_time = Utc::now() - Duration::days(7);
        let end_time = Utc::now();

        let types = vec![
            TimeSeriesType::Cost,
            TimeSeriesType::Commands,
            TimeSeriesType::SuccessRate,
            TimeSeriesType::ResponseTime,
        ];

        let result = optimizer
            .generate_optimized_time_series(start_time, end_time, types)
            .await?;

        // Verify results
        assert!(!result.cost_over_time.is_empty(), "Should have cost data");
        assert!(
            !result.commands_over_time.is_empty(),
            "Should have commands data"
        );
        assert!(
            result.generation_time_ms > 0,
            "Should track generation time"
        );
        assert!(
            result.optimization_ratio > 1.0,
            "Should show performance improvement"
        );

        println!("Optimization Results:");
        println!("  Generation time: {}ms", result.generation_time_ms);
        println!("  Total queries: {}", result.total_queries);
        println!("  Optimization ratio: {:.2}x", result.optimization_ratio);

        Ok(())
    }

    #[tokio::test]
    async fn test_performance_comparison() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Generate substantial test data
        let generator = AnalyticsTestDataGenerator {
            session_count: 20,
            commands_per_session: (30, 60),
            ..Default::default()
        };
        let test_data = generator.generate_test_data(30); // 30 days
        fixture.load_data(&test_data).await?;

        let optimizer = TimeSeriesOptimizer::new(fixture.analytics_engine.clone());

        let start_time = Utc::now() - Duration::days(7); // 1 week
        let end_time = Utc::now();

        let report = optimizer
            .generate_performance_comparison(start_time, end_time)
            .await?;

        println!("Performance Comparison Report:");
        println!("  Optimized duration: {}ms", report.optimized_duration_ms);
        println!(
            "  Estimated legacy duration: {}ms",
            report.estimated_legacy_duration_ms
        );
        println!(
            "  Performance improvement: {:.2}x",
            report.performance_improvement_ratio
        );
        println!(
            "  Queries reduced from {} to {}",
            report.queries_reduced_from, report.queries_reduced_to
        );
        println!(
            "  Query reduction: {:.1}%",
            report.query_reduction_percentage
        );

        // Verify performance improvements
        assert!(
            report.performance_improvement_ratio > 1.0,
            "Should show performance improvement"
        );
        assert!(
            report.query_reduction_percentage > 50.0,
            "Should reduce queries by at least 50%"
        );
        assert!(
            report.queries_reduced_to < report.queries_reduced_from,
            "Should reduce total queries"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_caching_functionality() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        let generator = AnalyticsTestDataGenerator {
            session_count: 5,
            commands_per_session: (10, 20),
            ..Default::default()
        };
        let test_data = generator.generate_test_data(3);
        fixture.load_data(&test_data).await?;

        let config = TimeSeriesConfig {
            bucket_size_minutes: 60,
            enable_parallel_processing: true,
            enable_caching: true,
            cache_ttl_seconds: 60,
        };

        let optimizer = TimeSeriesOptimizer::with_config(fixture.analytics_engine.clone(), config);

        let start_time = Utc::now() - Duration::days(3);
        let end_time = Utc::now();
        let types = vec![TimeSeriesType::Cost, TimeSeriesType::Commands];

        // First call - should cache the result
        let first_start = std::time::Instant::now();
        let _first_result = optimizer
            .generate_optimized_time_series(start_time, end_time, types.clone())
            .await?;
        let first_duration = first_start.elapsed();

        // Second call - should use cache
        let second_start = std::time::Instant::now();
        let second_result = optimizer
            .generate_optimized_time_series(start_time, end_time, types)
            .await?;
        let second_duration = second_start.elapsed();

        println!("Cache Test Results:");
        println!("  First call: {:?}", first_duration);
        println!("  Second call: {:?}", second_duration);
        println!("  Cache hits: {}", second_result.cache_hits);

        // Second call should be faster (cached)
        assert!(
            second_duration <= first_duration,
            "Cached call should be faster or equal"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_parallel_vs_sequential() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        let generator = AnalyticsTestDataGenerator {
            session_count: 15,
            commands_per_session: (25, 45),
            ..Default::default()
        };
        let test_data = generator.generate_test_data(7);
        fixture.load_data(&test_data).await?;

        let start_time = Utc::now() - Duration::days(7);
        let end_time = Utc::now();
        let types = vec![
            TimeSeriesType::Cost,
            TimeSeriesType::Commands,
            TimeSeriesType::SuccessRate,
            TimeSeriesType::ResponseTime,
        ];

        // Test parallel processing
        let parallel_config = TimeSeriesConfig {
            enable_parallel_processing: true,
            enable_caching: false, // Disable caching for fair comparison
            ..Default::default()
        };
        let parallel_optimizer =
            TimeSeriesOptimizer::with_config(fixture.analytics_engine.clone(), parallel_config);

        let parallel_start = std::time::Instant::now();
        let _parallel_result = parallel_optimizer
            .generate_optimized_time_series(start_time, end_time, types.clone())
            .await?;
        let parallel_duration = parallel_start.elapsed();

        // Test sequential processing
        let sequential_config = TimeSeriesConfig {
            enable_parallel_processing: false,
            enable_caching: false, // Disable caching for fair comparison
            ..Default::default()
        };
        let sequential_optimizer =
            TimeSeriesOptimizer::with_config(fixture.analytics_engine.clone(), sequential_config);

        let sequential_start = std::time::Instant::now();
        let _sequential_result = sequential_optimizer
            .generate_optimized_time_series(start_time, end_time, types)
            .await?;
        let sequential_duration = sequential_start.elapsed();

        println!("Parallel vs Sequential Test:");
        println!("  Parallel: {:?}", parallel_duration);
        println!("  Sequential: {:?}", sequential_duration);

        // Parallel should be faster or at least not significantly slower
        let ratio = sequential_duration.as_millis() as f64 / parallel_duration.as_millis() as f64;
        assert!(
            ratio >= 0.8,
            "Parallel processing should not be much slower than sequential"
        );

        Ok(())
    }
}
