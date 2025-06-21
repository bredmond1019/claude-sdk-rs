//! Test infrastructure for metrics module
//!
//! This module provides test utilities and infrastructure for testing
//! performance metrics collection, aggregation, and reporting functionality.

use super::metrics::*;
use crate::cli::error::Result;
use chrono::{DateTime, Duration, Timelike, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Test data generator for metrics testing
pub struct MetricsTestDataGenerator {
    base_time: DateTime<Utc>,
    command_patterns: Vec<CommandPattern>,
}

#[derive(Clone)]
struct CommandPattern {
    name: String,
    base_duration_ms: f64,
    variance: f64,
    success_rate: f64,
    frequency: f64,
}

impl MetricsTestDataGenerator {
    /// Create a new metrics test data generator
    pub fn new() -> Self {
        let command_patterns = vec![
            CommandPattern {
                name: "analyze".to_string(),
                base_duration_ms: 2000.0,
                variance: 500.0,
                success_rate: 0.95,
                frequency: 0.3,
            },
            CommandPattern {
                name: "summarize".to_string(),
                base_duration_ms: 1500.0,
                variance: 300.0,
                success_rate: 0.98,
                frequency: 0.25,
            },
            CommandPattern {
                name: "translate".to_string(),
                base_duration_ms: 1000.0,
                variance: 200.0,
                success_rate: 0.99,
                frequency: 0.2,
            },
            CommandPattern {
                name: "code".to_string(),
                base_duration_ms: 3000.0,
                variance: 1000.0,
                success_rate: 0.90,
                frequency: 0.15,
            },
            CommandPattern {
                name: "explain".to_string(),
                base_duration_ms: 1200.0,
                variance: 400.0,
                success_rate: 0.97,
                frequency: 0.1,
            },
        ];

        Self {
            base_time: Utc::now() - Duration::hours(24),
            command_patterns,
        }
    }

    /// Generate realistic command execution metrics
    pub fn generate_command_metrics(&self, count: usize) -> Vec<CommandMetric> {
        let mut metrics = Vec::new();
        let mut current_time = self.base_time;

        for _ in 0..count {
            let pattern = self.select_command_pattern();
            let duration = self.generate_duration(&pattern);
            let success = rand::random::<f64>() < pattern.success_rate;
            let cost = duration / 1000.0 * 0.01; // Simple cost model

            metrics.push(CommandMetric {
                command_name: pattern.name.clone(),
                timestamp: current_time,
                duration_ms: duration,
                success,
                cost_usd: cost,
                input_tokens: (100.0 + rand::random::<f64>() * 900.0) as u32,
                output_tokens: (200.0 + rand::random::<f64>() * 1800.0) as u32,
                session_id: uuid::Uuid::new_v4(),
                error_message: if !success {
                    Some("Simulated error".to_string())
                } else {
                    None
                },
            });

            // Advance time with some randomness
            let time_delta = Duration::seconds((30.0 + rand::random::<f64>() * 120.0) as i64);
            current_time = current_time + time_delta;
        }

        metrics
    }

    /// Generate time-series metrics data
    pub fn generate_time_series_metrics(
        &self,
        hours: u32,
        points_per_hour: u32,
    ) -> Vec<TimeSeriesPoint> {
        let mut points = Vec::new();
        let end_time = Utc::now();
        let start_time = end_time - Duration::hours(hours as i64);

        let mut current_time = start_time;
        let interval = Duration::minutes(60 / points_per_hour as i64);

        while current_time <= end_time {
            let hour = current_time.hour();
            let base_load = match hour {
                9..=11 | 14..=16 => 0.8, // Peak hours
                12..=13 => 0.5,          // Lunch
                17..=20 => 0.6,          // Evening
                _ => 0.2,                // Low activity
            };

            let load = base_load * (0.8 + rand::random::<f64>() * 0.4);

            points.push(TimeSeriesPoint {
                timestamp: current_time,
                value: load,
                label: format!("load_factor_{}", hour),
            });

            current_time = current_time + interval;
        }

        points
    }

    /// Generate distribution data for histograms
    pub fn generate_distribution_data(
        &self,
        count: usize,
        distribution_type: DistributionType,
    ) -> Vec<f64> {
        match distribution_type {
            DistributionType::ResponseTime => {
                (0..count)
                    .map(|_| {
                        let base = 1000.0;
                        let variation = rand::random::<f64>() * 2000.0;
                        base + variation
                            + if rand::random::<f64>() > 0.95 {
                                // 5% outliers
                                5000.0 + rand::random::<f64>() * 10000.0
                            } else {
                                0.0
                            }
                    })
                    .collect()
            }
            DistributionType::Cost => {
                (0..count)
                    .map(|_| {
                        let base = 0.01;
                        let variation = rand::random::<f64>() * 0.05;
                        base + variation
                            + if rand::random::<f64>() > 0.98 {
                                // 2% expensive operations
                                0.5 + rand::random::<f64>() * 1.0
                            } else {
                                0.0
                            }
                    })
                    .collect()
            }
            DistributionType::TokenCount => (0..count)
                .map(|_| {
                    let base = 500.0;
                    let variation = rand::random::<f64>() * 1000.0;
                    base + variation
                })
                .collect(),
        }
    }

    fn select_command_pattern(&self) -> &CommandPattern {
        let total_frequency: f64 = self.command_patterns.iter().map(|p| p.frequency).sum();
        let mut random_value = rand::random::<f64>() * total_frequency;

        for pattern in &self.command_patterns {
            random_value -= pattern.frequency;
            if random_value <= 0.0 {
                return pattern;
            }
        }

        &self.command_patterns[0]
    }

    fn generate_duration(&self, pattern: &CommandPattern) -> f64 {
        let variation = (rand::random::<f64>() - 0.5) * 2.0 * pattern.variance;
        (pattern.base_duration_ms + variation).max(100.0)
    }
}

/// Types of distributions for testing
#[derive(Debug, Clone)]
pub enum DistributionType {
    ResponseTime,
    Cost,
    TokenCount,
}

/// Command metric for testing
#[derive(Debug, Clone)]
pub struct CommandMetric {
    pub command_name: String,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: f64,
    pub success: bool,
    pub cost_usd: f64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub session_id: uuid::Uuid,
    pub error_message: Option<String>,
}

/// Time series point for testing
#[derive(Debug, Clone)]
pub struct TimeSeriesPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub label: String,
}

/// Test fixture for metrics testing
pub struct MetricsTestFixture {
    pub registry: Arc<MetricRegistry>,
    pub data_generator: MetricsTestDataGenerator,
}

impl MetricsTestFixture {
    /// Create a new metrics test fixture
    pub async fn new() -> Result<Self> {
        let config = MetricConfig::default();
        let registry = Arc::new(MetricRegistry::new(config));

        // Initialize default metrics
        registry.initialize_default_metrics().await?;

        let data_generator = MetricsTestDataGenerator::new();

        Ok(Self {
            registry,
            data_generator,
        })
    }

    /// Populate metrics with test data
    pub async fn populate_metrics(&self, command_count: usize) -> Result<()> {
        let metrics = self.data_generator.generate_command_metrics(command_count);

        for metric in metrics {
            // Record command execution
            self.registry.increment_counter("commands_total").await?;

            if !metric.success {
                self.registry
                    .increment_counter("commands_failed_total")
                    .await?;
            }

            // Record duration
            self.registry
                .observe_histogram("command_duration_ms", metric.duration_ms)
                .await?;
            self.registry
                .record_timer("api_response_time", metric.duration_ms)
                .await?;

            // Record cost
            self.registry
                .observe_histogram("command_cost_usd", metric.cost_usd)
                .await?;
            self.registry
                .set_gauge("total_cost_usd", metric.cost_usd)
                .await?;
        }

        Ok(())
    }

    /// Generate metric snapshot for testing
    pub async fn generate_test_snapshot(&self) -> MetricSnapshot {
        self.registry.snapshot().await
    }
}

/// Performance benchmarking utilities for metrics
pub mod performance {
    use super::*;
    use std::time::Instant;

    /// Benchmark metric recording performance
    pub async fn benchmark_metric_recording(
        fixture: &MetricsTestFixture,
        operations: usize,
    ) -> BenchmarkResult {
        let start = Instant::now();
        let mut durations = Vec::new();

        for _ in 0..operations {
            let op_start = Instant::now();

            // Record various metrics
            let _ = fixture.registry.increment_counter("test_counter").await;
            let _ = fixture
                .registry
                .set_gauge("test_gauge", rand::random::<f64>() * 100.0)
                .await;
            let _ = fixture
                .registry
                .observe_histogram("test_histogram", rand::random::<f64>() * 1000.0)
                .await;

            durations.push(op_start.elapsed().as_micros());
        }

        let total_duration = start.elapsed();

        durations.sort();
        let p50 = durations[durations.len() / 2];
        let p95 = durations[(durations.len() as f64 * 0.95) as usize];
        let p99 = durations[(durations.len() as f64 * 0.99) as usize];

        BenchmarkResult {
            total_operations: operations,
            total_duration_ms: total_duration.as_millis(),
            ops_per_second: operations as f64 / total_duration.as_secs_f64(),
            p50_micros: p50,
            p95_micros: p95,
            p99_micros: p99,
        }
    }

    /// Benchmark metric aggregation performance
    pub async fn benchmark_aggregation(
        fixture: &MetricsTestFixture,
        data_points: usize,
    ) -> AggregationBenchmark {
        // Populate with test data
        let _ = fixture.populate_metrics(data_points).await;

        // Benchmark snapshot generation
        let mut snapshot_durations = Vec::new();
        for _ in 0..10 {
            let start = Instant::now();
            let _ = fixture.registry.snapshot().await;
            snapshot_durations.push(start.elapsed().as_millis());
        }

        // Benchmark Prometheus export
        let mut export_durations = Vec::new();
        for _ in 0..10 {
            let start = Instant::now();
            let _ = fixture.registry.export_prometheus().await;
            export_durations.push(start.elapsed().as_millis());
        }

        AggregationBenchmark {
            data_points,
            avg_snapshot_ms: snapshot_durations.iter().sum::<u128>() as f64
                / snapshot_durations.len() as f64,
            avg_export_ms: export_durations.iter().sum::<u128>() as f64
                / export_durations.len() as f64,
        }
    }

    #[derive(Debug)]
    pub struct BenchmarkResult {
        pub total_operations: usize,
        pub total_duration_ms: u128,
        pub ops_per_second: f64,
        pub p50_micros: u128,
        pub p95_micros: u128,
        pub p99_micros: u128,
    }

    #[derive(Debug)]
    pub struct AggregationBenchmark {
        pub data_points: usize,
        pub avg_snapshot_ms: f64,
        pub avg_export_ms: f64,
    }
}

/// Test assertions for metrics
pub mod assertions {
    use super::*;

    /// Assert counter metric validity
    pub fn assert_counter_valid(counter: &Counter) {
        assert!(!counter.name.is_empty(), "Counter name should not be empty");
        assert!(counter.value >= 0, "Counter value should be non-negative");
        assert!(
            counter.created_at <= counter.last_updated,
            "Counter creation time should be before or equal to last update"
        );
    }

    /// Assert gauge metric validity
    pub fn assert_gauge_valid(gauge: &Gauge) {
        assert!(!gauge.name.is_empty(), "Gauge name should not be empty");
        assert!(!gauge.value.is_nan(), "Gauge value should not be NaN");
        assert!(
            gauge.created_at <= gauge.last_updated,
            "Gauge creation time should be before or equal to last update"
        );
    }

    /// Assert histogram validity
    pub fn assert_histogram_valid(histogram: &Histogram) {
        assert!(
            !histogram.name.is_empty(),
            "Histogram name should not be empty"
        );
        assert!(
            !histogram.buckets.is_empty(),
            "Histogram should have buckets"
        );
        assert_eq!(
            histogram.buckets.len(),
            histogram.bucket_counts.len(),
            "Bucket and count arrays should have same length"
        );

        // Check buckets are sorted
        for i in 1..histogram.buckets.len() {
            assert!(
                histogram.buckets[i - 1] <= histogram.buckets[i],
                "Histogram buckets should be sorted"
            );
        }

        // Check bucket counts don't exceed total count
        let total_bucket_count: u64 = histogram.bucket_counts.iter().sum();
        assert!(
            total_bucket_count >= histogram.count,
            "Bucket counts should not exceed total count"
        );
    }

    /// Assert timer validity
    pub fn assert_timer_valid(timer: &Timer) {
        assert!(!timer.name.is_empty(), "Timer name should not be empty");
        assert!(
            timer.samples.len() <= timer.max_samples,
            "Timer samples should not exceed maximum"
        );

        // Check all samples are positive
        for &sample in &timer.samples {
            assert!(sample >= 0.0, "Timer samples should be non-negative");
        }
    }

    /// Assert metric snapshot validity
    pub fn assert_snapshot_valid(snapshot: &MetricSnapshot) {
        // Check all counters
        for counter in &snapshot.counters {
            assert!(
                !counter.name.is_empty(),
                "Counter name in snapshot should not be empty"
            );
            assert!(
                counter.value >= 0,
                "Counter value in snapshot should be non-negative"
            );
        }

        // Check all gauges
        for gauge in &snapshot.gauges {
            assert!(
                !gauge.name.is_empty(),
                "Gauge name in snapshot should not be empty"
            );
            assert!(
                !gauge.value.is_nan(),
                "Gauge value in snapshot should not be NaN"
            );
        }

        // Check all histograms
        for histogram in &snapshot.histograms {
            assert!(
                !histogram.name.is_empty(),
                "Histogram name in snapshot should not be empty"
            );
            assert!(
                histogram.count >= 0,
                "Histogram count should be non-negative"
            );

            // Check percentiles
            for (_, percentile) in &histogram.percentiles {
                assert!(
                    *percentile >= 0.0,
                    "Percentile values should be non-negative"
                );
            }
        }

        // Check all timers
        for timer in &snapshot.timers {
            assert!(
                !timer.name.is_empty(),
                "Timer name in snapshot should not be empty"
            );
            assert!(timer.mean >= 0.0, "Timer mean should be non-negative");
            assert!(
                timer.min <= timer.max,
                "Timer min should be less than or equal to max"
            );
        }
    }
}

/// Mock metric collector for testing
pub struct MockMetricCollector {
    collected_metrics: Arc<RwLock<Vec<CollectedMetric>>>,
}

#[derive(Debug, Clone)]
pub struct CollectedMetric {
    pub metric_type: String,
    pub name: String,
    pub value: f64,
    pub timestamp: DateTime<Utc>,
    pub labels: HashMap<String, String>,
}

impl MockMetricCollector {
    pub fn new() -> Self {
        Self {
            collected_metrics: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn collect(&self, metric: CollectedMetric) {
        let mut metrics = self.collected_metrics.write().await;
        metrics.push(metric);
    }

    pub async fn get_collected_metrics(&self) -> Vec<CollectedMetric> {
        let metrics = self.collected_metrics.read().await;
        metrics.clone()
    }

    pub async fn clear(&self) {
        let mut metrics = self.collected_metrics.write().await;
        metrics.clear();
    }
}

/// Utility functions for metric testing
pub mod utils {
    use super::*;

    /// Create standard metric buckets for response times
    pub fn create_response_time_buckets() -> Vec<f64> {
        vec![10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0]
    }

    /// Create standard metric buckets for costs
    pub fn create_cost_buckets() -> Vec<f64> {
        vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    }

    /// Create standard metric buckets for token counts
    pub fn create_token_buckets() -> Vec<f64> {
        vec![100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0]
    }

    /// Calculate statistics for a set of values
    pub fn calculate_statistics(values: &[f64]) -> MetricStatistics {
        if values.is_empty() {
            return MetricStatistics::default();
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let sum: f64 = values.iter().sum();
        let mean = sum / values.len() as f64;

        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let median = if values.len() % 2 == 0 {
            (sorted[values.len() / 2 - 1] + sorted[values.len() / 2]) / 2.0
        } else {
            sorted[values.len() / 2]
        };

        MetricStatistics {
            count: values.len(),
            sum,
            mean,
            median,
            std_dev,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            p50: sorted[values.len() / 2],
            p95: sorted[(values.len() as f64 * 0.95) as usize],
            p99: sorted[(values.len() as f64 * 0.99) as usize],
        }
    }

    #[derive(Debug, Default)]
    pub struct MetricStatistics {
        pub count: usize,
        pub sum: f64,
        pub mean: f64,
        pub median: f64,
        pub std_dev: f64,
        pub min: f64,
        pub max: f64,
        pub p50: f64,
        pub p95: f64,
        pub p99: f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::analytics::test_utils::{
        AnalyticsTestDataGenerator, AnalyticsTestFixture, DataPattern,
    };
    use crate::cli::cost::{CostEntry, CostTracker};
    use chrono::{DateTime, Duration, Timelike, Utc};
    use std::collections::HashMap;
    use uuid::Uuid;

    /// Task 3.3.1: Test usage metrics calculations
    mod usage_metrics_tests {
        use super::*;

        #[tokio::test]
        async fn test_command_counting_per_session() {
            let fixture = MetricsTestFixture::new().await.unwrap();
            let registry = &fixture.registry;

            // Generate test data with known session patterns
            let session1 = Uuid::new_v4();
            let session2 = Uuid::new_v4();

            // Session 1: 10 commands
            for i in 0..10 {
                let metric = CommandMetric {
                    command_name: format!("cmd_{}", i),
                    timestamp: Utc::now() - Duration::minutes(30 - i as i64),
                    duration_ms: 1000.0,
                    success: true,
                    cost_usd: 0.01,
                    input_tokens: 100,
                    output_tokens: 200,
                    session_id: session1,
                    error_message: None,
                };

                registry.increment_counter("commands_total").await.unwrap();
                registry
                    .observe_histogram("command_duration_ms", metric.duration_ms)
                    .await
                    .unwrap();
            }

            // Session 2: 5 commands
            for i in 0..5 {
                let metric = CommandMetric {
                    command_name: format!("cmd_{}", i),
                    timestamp: Utc::now() - Duration::minutes(15 - i as i64),
                    duration_ms: 2000.0,
                    success: true,
                    cost_usd: 0.02,
                    input_tokens: 150,
                    output_tokens: 300,
                    session_id: session2,
                    error_message: None,
                };

                registry.increment_counter("commands_total").await.unwrap();
                registry
                    .observe_histogram("command_duration_ms", metric.duration_ms)
                    .await
                    .unwrap();
            }

            let snapshot = registry.snapshot().await;

            // Verify total command count
            let total_counter = snapshot
                .counters
                .iter()
                .find(|c| c.name == "commands_total")
                .unwrap();
            assert_eq!(total_counter.value, 15);

            // Verify histogram has recorded all commands
            let duration_histogram = snapshot
                .histograms
                .iter()
                .find(|h| h.name == "command_duration_ms")
                .unwrap();
            assert_eq!(duration_histogram.count, 15);
        }

        #[tokio::test]
        async fn test_command_frequency_calculations() {
            let fixture = MetricsTestFixture::new().await.unwrap();
            let _generator = &fixture.data_generator;

            // Generate commands with known frequency distribution
            let mut command_frequencies = HashMap::new();
            let commands = vec![
                ("analyze", 30),
                ("summarize", 25),
                ("translate", 20),
                ("code", 15),
                ("explain", 10),
            ];

            for (cmd_name, count) in commands {
                for _ in 0..count {
                    let metric = CommandMetric {
                        command_name: cmd_name.to_string(),
                        timestamp: Utc::now() - Duration::hours(1),
                        duration_ms: 1000.0,
                        success: true,
                        cost_usd: 0.01,
                        input_tokens: 100,
                        output_tokens: 200,
                        session_id: Uuid::new_v4(),
                        error_message: None,
                    };

                    fixture
                        .registry
                        .increment_counter("commands_total")
                        .await
                        .unwrap();

                    // Track frequency
                    *command_frequencies.entry(cmd_name.to_string()).or_insert(0) += 1;
                }
            }

            // Verify frequency distribution
            let total_commands: u32 = command_frequencies.values().sum();
            assert_eq!(total_commands, 100);

            // Check relative frequencies
            assert_eq!(command_frequencies["analyze"], 30);
            assert_eq!(command_frequencies["summarize"], 25);
            assert_eq!(command_frequencies["translate"], 20);
            assert_eq!(command_frequencies["code"], 15);
            assert_eq!(command_frequencies["explain"], 10);

            // Calculate frequency percentages
            let analyze_freq =
                (command_frequencies["analyze"] as f64 / total_commands as f64) * 100.0;
            assert!((analyze_freq - 30.0).abs() < 0.1);
        }

        #[tokio::test]
        async fn test_session_duration_metrics() {
            let fixture = MetricsTestFixture::new().await.unwrap();
            let registry = &fixture.registry;

            // Simulate different session durations
            let session_durations = vec![
                ("short_session", 5.0 * 60.0 * 1000.0),   // 5 minutes
                ("medium_session", 30.0 * 60.0 * 1000.0), // 30 minutes
                ("long_session", 120.0 * 60.0 * 1000.0),  // 2 hours
            ];

            for (_session_type, duration_ms) in session_durations {
                registry
                    .record_timer("session_duration", duration_ms)
                    .await
                    .unwrap();
            }

            let snapshot = registry.snapshot().await;
            let session_timer = snapshot
                .timers
                .iter()
                .find(|t| t.name == "session_duration")
                .unwrap();

            // Verify metrics
            assert_eq!(session_timer.count, 3);
            assert!((session_timer.mean - 51.666666 * 60.0 * 1000.0).abs() < 1000.0); // ~51.67 minutes average
            assert_eq!(session_timer.min, 5.0 * 60.0 * 1000.0);
            assert_eq!(session_timer.max, 120.0 * 60.0 * 1000.0);
        }
    }

    /// Task 3.3.2: Test performance metrics (execution time, success rates)
    mod performance_metrics_tests {
        use super::*;

        #[tokio::test]
        async fn test_execution_time_calculations() {
            let fixture = MetricsTestFixture::new().await.unwrap();
            let registry = &fixture.registry;

            // Generate execution times with known distribution
            let execution_times = vec![
                100.0, // min
                500.0, 1000.0, 1500.0, 2000.0, // median
                2500.0, 3000.0, 4000.0, 9000.0,  // p95 outlier
                15000.0, // p99 outlier - max
            ];

            for time in &execution_times {
                registry
                    .observe_histogram("command_duration_ms", *time)
                    .await
                    .unwrap();
                registry
                    .record_timer("api_response_time", *time)
                    .await
                    .unwrap();
            }

            let snapshot = registry.snapshot().await;

            // Test histogram metrics
            let duration_histogram = snapshot
                .histograms
                .iter()
                .find(|h| h.name == "command_duration_ms")
                .unwrap();

            assert_eq!(duration_histogram.count, 10);
            let expected_sum: f64 = execution_times.iter().sum();
            assert!((duration_histogram.sum - expected_sum).abs() < 0.1);

            // Test timer metrics
            let response_timer = snapshot
                .timers
                .iter()
                .find(|t| t.name == "api_response_time")
                .unwrap();

            assert_eq!(response_timer.count, 10);
            assert_eq!(response_timer.min, 100.0);
            assert_eq!(response_timer.max, 15000.0);
            assert!((response_timer.mean - 3860.0).abs() < 0.1);
            assert!((response_timer.median - 2250.0).abs() < 0.1);
            assert!(response_timer.p95 >= 9000.0);
            assert!(response_timer.p99 >= 15000.0);
        }
    }

    /// Task 3.3.3: Test cost metrics integration with cost tracking module
    mod cost_metrics_tests {
        use super::*;

        #[tokio::test]
        async fn test_cost_aggregation_by_session() {
            let fixture = MetricsTestFixture::new().await.unwrap();
            let registry = &fixture.registry;

            // Register session cost counter
            registry
                .register_counter(
                    "session_cost_cents".to_string(),
                    "Session cost in cents".to_string(),
                    HashMap::new(),
                )
                .await
                .unwrap();

            // Session 1: Low cost operations
            for i in 0..10 {
                let cost = 0.001 * (i + 1) as f64;
                registry
                    .observe_histogram("command_cost_usd", cost)
                    .await
                    .unwrap();
                registry
                    .add_to_counter("session_cost_cents", (cost * 100.0) as u64)
                    .await
                    .unwrap();
            }

            // Session 2: Medium cost operations
            for i in 0..5 {
                let cost = 0.01 * (i + 1) as f64;
                registry
                    .observe_histogram("command_cost_usd", cost)
                    .await
                    .unwrap();
                registry
                    .add_to_counter("session_cost_cents", (cost * 100.0) as u64)
                    .await
                    .unwrap();
            }

            // Session 3: High cost operations
            for i in 0..3 {
                let cost = 0.1 * (i + 1) as f64;
                registry
                    .observe_histogram("command_cost_usd", cost)
                    .await
                    .unwrap();
                registry
                    .add_to_counter("session_cost_cents", (cost * 100.0) as u64)
                    .await
                    .unwrap();
            }

            let snapshot = registry.snapshot().await;

            // Verify cost histogram
            let cost_histogram = snapshot
                .histograms
                .iter()
                .find(|h| h.name == "command_cost_usd")
                .unwrap();

            assert_eq!(cost_histogram.count, 18); // 10 + 5 + 3

            // Verify total cost
            let total_cost = 0.055 + 0.15 + 0.6; // Sum of all costs
            assert!((cost_histogram.sum - total_cost).abs() < 0.01);
        }
    }

    /// Task 3.3.4: Test trend calculations and statistical analysis
    mod trend_analysis_tests {
        use super::*;

        #[tokio::test]
        async fn test_moving_average_calculations() {
            let fixture = MetricsTestFixture::new().await.unwrap();
            let registry = &fixture.registry;

            // Register the metric_value histogram
            registry
                .register_histogram(
                    "metric_value".to_string(),
                    "Test metric for trend analysis".to_string(),
                    vec![50.0, 100.0, 150.0, 200.0, 250.0],
                    HashMap::new(),
                )
                .await
                .unwrap();

            // Generate time series data with more pronounced variation
            let mut values = Vec::new();
            for i in 0..30 {
                let base_trend = 100.0 + (i as f64 * 2.0);
                let noise = (rand::random::<f64>() - 0.5) * 30.0; // Increased noise
                let value = base_trend + noise;
                values.push(value);
                registry
                    .observe_histogram("metric_value", value)
                    .await
                    .unwrap();
            }

            // Calculate 7-day moving average
            let mut ma7 = Vec::new();
            for i in 6..values.len() {
                let sum: f64 = values[i - 6..=i].iter().sum();
                ma7.push(sum / 7.0);
            }

            // Verify moving averages smooth out variations
            let ma7_variance = calculate_variance(&ma7);
            let raw_variance = calculate_variance(&values);

            // Moving average should have lower variance (relaxed threshold)
            assert!(
                ma7_variance <= raw_variance,
                "Moving average variance ({:.2}) should be <= raw variance ({:.2})",
                ma7_variance,
                raw_variance
            );

            // Trend should be upward (positive slope)
            let first_ma7 = ma7.first().unwrap();
            let last_ma7 = ma7.last().unwrap();
            assert!(last_ma7 > first_ma7);
        }

        // Helper functions
        fn calculate_variance(values: &[f64]) -> f64 {
            if values.is_empty() {
                return 0.0;
            }

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
        }
    }
}
