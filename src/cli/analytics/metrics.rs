//! Performance metrics collection and monitoring
//!
//! This module provides a comprehensive metrics system for tracking performance,
//! usage patterns, and system health in the Claude AI Interactive system.
//!
//! # Overview
//!
//! The metrics system supports four primary metric types:
//! - **Counters**: Monotonically increasing values (e.g., total API calls)
//! - **Gauges**: Values that can go up or down (e.g., active sessions)
//! - **Histograms**: Distribution of values (e.g., response times)
//! - **Timers**: Specialized histograms for timing operations
//!
//! # Architecture
//!
//! The metrics system consists of:
//! 1. **MetricRegistry**: Central storage for all metrics
//! 2. **MetricsEngine**: High-level API for metric operations
//! 3. **Collectors**: Background tasks that gather metrics
//! 4. **Exporters**: Interfaces to external monitoring systems
//!
//! # Example Usage
//!
//! ```no_run
//! use crate_interactive::analytics::{MetricsEngine, MetricConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create metrics engine
//! let config = MetricConfig::default();
//! let metrics = MetricsEngine::new(config);
//!
//! // Register and use a counter
//! metrics.register_counter(
//!     "api_calls_total",
//!     "Total number of API calls",
//!     vec![("service", "claude")]
//! ).await?;
//!
//! metrics.increment_counter(
//!     "api_calls_total",
//!     vec![("status", "success")]
//! ).await?;
//!
//! // Use a timer for performance tracking
//! let timer = metrics.start_timer("operation_duration").await?;
//! // ... perform operation ...
//! let duration = timer.stop().await?;
//! println!("Operation took {}ms", duration);
//! # Ok(())
//! # }
//! ```
//!
//! # Best Practices
//!
//! 1. Use consistent naming: `namespace_subsystem_name_unit`
//! 2. Keep cardinality low: avoid unique IDs in labels
//! 3. Pre-declare metrics during initialization
//! 4. Use appropriate metric types for each use case

use crate::{cli::error::InteractiveError, cli::error::Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tokio::sync::RwLock;

/// Metric registry for collecting and managing metrics
pub struct MetricRegistry {
    counters: RwLock<HashMap<String, Counter>>,
    gauges: RwLock<HashMap<String, Gauge>>,
    histograms: RwLock<HashMap<String, Histogram>>,
    timers: RwLock<HashMap<String, Timer>>,
    config: MetricConfig,
}

/// Metric collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricConfig {
    pub collection_interval_seconds: u64,
    pub retention_duration_hours: u32,
    pub enable_high_cardinality_metrics: bool,
    pub metric_buffer_size: usize,
}

impl Default for MetricConfig {
    fn default() -> Self {
        Self {
            collection_interval_seconds: 10,
            retention_duration_hours: 24,
            enable_high_cardinality_metrics: false,
            metric_buffer_size: 10000,
        }
    }
}

/// Counter metric - monotonically increasing value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counter {
    pub name: String,
    pub description: String,
    pub value: u64,
    pub labels: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

impl Counter {
    pub fn new(name: String, description: String, labels: HashMap<String, String>) -> Self {
        let now = Utc::now();
        Self {
            name,
            description,
            value: 0,
            labels,
            created_at: now,
            last_updated: now,
        }
    }

    pub fn increment(&mut self) {
        self.value += 1;
        self.last_updated = Utc::now();
    }

    pub fn add(&mut self, value: u64) {
        self.value += value;
        self.last_updated = Utc::now();
    }
}

/// Gauge metric - arbitrary value that can go up or down
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gauge {
    pub name: String,
    pub description: String,
    pub value: f64,
    pub labels: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

impl Gauge {
    pub fn new(name: String, description: String, labels: HashMap<String, String>) -> Self {
        let now = Utc::now();
        Self {
            name,
            description,
            value: 0.0,
            labels,
            created_at: now,
            last_updated: now,
        }
    }

    pub fn set(&mut self, value: f64) {
        self.value = value;
        self.last_updated = Utc::now();
    }

    pub fn add(&mut self, delta: f64) {
        self.value += delta;
        self.last_updated = Utc::now();
    }
}

/// Histogram metric - for measuring distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    pub name: String,
    pub description: String,
    pub buckets: Vec<f64>,
    pub bucket_counts: Vec<u64>,
    pub count: u64,
    pub sum: f64,
    pub labels: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

impl Histogram {
    pub fn new(
        name: String,
        description: String,
        buckets: Vec<f64>,
        labels: HashMap<String, String>,
    ) -> Self {
        let bucket_counts = vec![0; buckets.len()];
        let now = Utc::now();
        Self {
            name,
            description,
            buckets,
            bucket_counts,
            count: 0,
            sum: 0.0,
            labels,
            created_at: now,
            last_updated: now,
        }
    }

    pub fn observe(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.last_updated = Utc::now();

        for (i, &bucket) in self.buckets.iter().enumerate() {
            if value <= bucket {
                self.bucket_counts[i] += 1;
            }
        }
    }

    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.count == 0 {
            return None;
        }

        let target_count = (self.count as f64 * p / 100.0) as u64;
        let mut cumulative = 0;

        for (i, &count) in self.bucket_counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target_count {
                return Some(self.buckets[i]);
            }
        }

        self.buckets.last().copied()
    }
}

/// Timer metric - for measuring durations with statistical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timer {
    pub name: String,
    pub description: String,
    pub samples: VecDeque<f64>,
    pub max_samples: usize,
    pub labels: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

impl Timer {
    pub fn new(
        name: String,
        description: String,
        max_samples: usize,
        labels: HashMap<String, String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            name,
            description,
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            labels,
            created_at: now,
            last_updated: now,
        }
    }

    pub fn record(&mut self, duration_ms: f64) {
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(duration_ms);
        self.last_updated = Utc::now();
    }

    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }

    pub fn median(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    pub fn percentile(&self, p: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
        let clamped_index = (index.saturating_sub(1)).min(sorted.len() - 1);
        sorted[clamped_index]
    }

    pub fn min(&self) -> f64 {
        self.samples.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    }

    pub fn max(&self) -> f64 {
        self.samples
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    }
}

/// Metric snapshot for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSnapshot {
    pub timestamp: DateTime<Utc>,
    pub counters: Vec<CounterSnapshot>,
    pub gauges: Vec<GaugeSnapshot>,
    pub histograms: Vec<HistogramSnapshot>,
    pub timers: Vec<TimerSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterSnapshot {
    pub name: String,
    pub value: u64,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaugeSnapshot {
    pub name: String,
    pub value: f64,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramSnapshot {
    pub name: String,
    pub count: u64,
    pub sum: f64,
    pub percentiles: HashMap<String, f64>,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimerSnapshot {
    pub name: String,
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
    pub labels: HashMap<String, String>,
}

/// Core metrics calculation engine
pub struct MetricsEngine {
    registry: MetricRegistry,
    config: MetricConfig,
}

impl MetricsEngine {
    /// Create a new metrics engine
    pub fn new(config: MetricConfig) -> Self {
        Self {
            registry: MetricRegistry::new(config.clone()),
            config,
        }
    }

    /// Initialize with default metrics
    pub async fn initialize(&self) -> Result<()> {
        self.registry.initialize_default_metrics().await
    }

    /// Record a command metric
    pub async fn record_command_metric(&self, metric: CommandMetric) -> Result<()> {
        // Record in counter
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

        Ok(())
    }

    /// Calculate average response time
    pub async fn calculate_average_response_time(&self) -> f64 {
        let snapshot = self.registry.snapshot().await;
        if let Some(timer) = snapshot
            .timers
            .iter()
            .find(|t| t.name == "api_response_time")
        {
            timer.mean
        } else {
            0.0
        }
    }

    /// Calculate success rate percentage
    pub async fn calculate_success_rate(&self) -> f64 {
        let snapshot = self.registry.snapshot().await;

        let total = snapshot
            .counters
            .iter()
            .find(|c| c.name == "commands_total")
            .map(|c| c.value)
            .unwrap_or(0);

        let failed = snapshot
            .counters
            .iter()
            .find(|c| c.name == "commands_failed_total")
            .map(|c| c.value)
            .unwrap_or(0);

        if total > 0 {
            ((total - failed) as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate percentile for response times
    pub async fn calculate_percentile(&self, percentile: f64) -> f64 {
        let snapshot = self.registry.snapshot().await;
        if let Some(timer) = snapshot
            .timers
            .iter()
            .find(|t| t.name == "api_response_time")
        {
            match percentile {
                95.0 => timer.p95,
                99.0 => timer.p99,
                50.0 => timer.median,
                _ => timer.mean, // Default to mean for other percentiles
            }
        } else {
            0.0
        }
    }

    /// Get metrics snapshot
    pub async fn get_snapshot(&self) -> MetricSnapshot {
        self.registry.snapshot().await
    }

    /// Export metrics in Prometheus format
    pub async fn export_prometheus(&self) -> String {
        self.registry.export_prometheus().await
    }

    /// Get registry reference
    pub fn registry(&self) -> &MetricRegistry {
        &self.registry
    }
}

/// Command execution metric
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

impl CommandMetric {
    pub fn new(
        command_name: String,
        duration_ms: f64,
        success: bool,
        cost_usd: f64,
        input_tokens: u32,
        output_tokens: u32,
        session_id: uuid::Uuid,
    ) -> Self {
        Self {
            command_name,
            timestamp: Utc::now(),
            duration_ms,
            success,
            cost_usd,
            input_tokens,
            output_tokens,
            session_id,
            error_message: None,
        }
    }

    pub fn with_error(mut self, error: String) -> Self {
        self.error_message = Some(error);
        self.success = false;
        self
    }

    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }
}

/// Time series data point
#[derive(Debug, Clone)]
pub struct TimeSeriesPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub label: String,
}

impl TimeSeriesPoint {
    pub fn new(timestamp: DateTime<Utc>, value: f64, label: String) -> Self {
        Self {
            timestamp,
            value,
            label,
        }
    }
}

impl MetricRegistry {
    /// Create a new metric registry
    pub fn new(config: MetricConfig) -> Self {
        Self {
            counters: RwLock::new(HashMap::new()),
            gauges: RwLock::new(HashMap::new()),
            histograms: RwLock::new(HashMap::new()),
            timers: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Register a new counter
    pub async fn register_counter(
        &self,
        name: String,
        description: String,
        labels: HashMap<String, String>,
    ) -> Result<()> {
        let counter = Counter::new(name.clone(), description, labels);
        let mut counters = self.counters.write().await;
        counters.insert(name, counter);
        Ok(())
    }

    /// Increment a counter
    pub async fn increment_counter(&self, name: &str) -> Result<()> {
        let mut counters = self.counters.write().await;
        if let Some(counter) = counters.get_mut(name) {
            counter.increment();
        } else {
            return Err(InteractiveError::invalid_input(format!(
                "Counter '{}' not found",
                name
            )));
        }
        Ok(())
    }

    /// Add value to a counter
    pub async fn add_to_counter(&self, name: &str, value: u64) -> Result<()> {
        let mut counters = self.counters.write().await;
        if let Some(counter) = counters.get_mut(name) {
            counter.add(value);
        } else {
            return Err(InteractiveError::invalid_input(format!(
                "Counter '{}' not found",
                name
            )));
        }
        Ok(())
    }

    /// Register a new gauge
    pub async fn register_gauge(
        &self,
        name: String,
        description: String,
        labels: HashMap<String, String>,
    ) -> Result<()> {
        let gauge = Gauge::new(name.clone(), description, labels);
        let mut gauges = self.gauges.write().await;
        gauges.insert(name, gauge);
        Ok(())
    }

    /// Set gauge value
    pub async fn set_gauge(&self, name: &str, value: f64) -> Result<()> {
        let mut gauges = self.gauges.write().await;
        if let Some(gauge) = gauges.get_mut(name) {
            gauge.set(value);
        } else {
            return Err(InteractiveError::invalid_input(format!(
                "Gauge '{}' not found",
                name
            )));
        }
        Ok(())
    }

    /// Register a new histogram
    pub async fn register_histogram(
        &self,
        name: String,
        description: String,
        buckets: Vec<f64>,
        labels: HashMap<String, String>,
    ) -> Result<()> {
        let histogram = Histogram::new(name.clone(), description, buckets, labels);
        let mut histograms = self.histograms.write().await;
        histograms.insert(name, histogram);
        Ok(())
    }

    /// Observe value in histogram
    pub async fn observe_histogram(&self, name: &str, value: f64) -> Result<()> {
        let mut histograms = self.histograms.write().await;
        if let Some(histogram) = histograms.get_mut(name) {
            histogram.observe(value);
        } else {
            return Err(InteractiveError::invalid_input(format!(
                "Histogram '{}' not found",
                name
            )));
        }
        Ok(())
    }

    /// Register a new timer
    pub async fn register_timer(
        &self,
        name: String,
        description: String,
        labels: HashMap<String, String>,
    ) -> Result<()> {
        let timer = Timer::new(name.clone(), description, 1000, labels);
        let mut timers = self.timers.write().await;
        timers.insert(name, timer);
        Ok(())
    }

    /// Record duration in timer
    pub async fn record_timer(&self, name: &str, duration_ms: f64) -> Result<()> {
        let mut timers = self.timers.write().await;
        if let Some(timer) = timers.get_mut(name) {
            timer.record(duration_ms);
        } else {
            return Err(InteractiveError::invalid_input(format!(
                "Timer '{}' not found",
                name
            )));
        }
        Ok(())
    }

    /// Get current metric snapshot
    pub async fn snapshot(&self) -> MetricSnapshot {
        let counters = self.counters.read().await;
        let gauges = self.gauges.read().await;
        let histograms = self.histograms.read().await;
        let timers = self.timers.read().await;

        let counter_snapshots: Vec<CounterSnapshot> = counters
            .values()
            .map(|c| CounterSnapshot {
                name: c.name.clone(),
                value: c.value,
                labels: c.labels.clone(),
            })
            .collect();

        let gauge_snapshots: Vec<GaugeSnapshot> = gauges
            .values()
            .map(|g| GaugeSnapshot {
                name: g.name.clone(),
                value: g.value,
                labels: g.labels.clone(),
            })
            .collect();

        let histogram_snapshots: Vec<HistogramSnapshot> = histograms
            .values()
            .map(|h| {
                let mut percentiles = HashMap::new();
                percentiles.insert("50".to_string(), h.percentile(50.0).unwrap_or(0.0));
                percentiles.insert("95".to_string(), h.percentile(95.0).unwrap_or(0.0));
                percentiles.insert("99".to_string(), h.percentile(99.0).unwrap_or(0.0));

                HistogramSnapshot {
                    name: h.name.clone(),
                    count: h.count,
                    sum: h.sum,
                    percentiles,
                    labels: h.labels.clone(),
                }
            })
            .collect();

        let timer_snapshots: Vec<TimerSnapshot> = timers
            .values()
            .map(|t| TimerSnapshot {
                name: t.name.clone(),
                count: t.samples.len(),
                mean: t.mean(),
                median: t.median(),
                p95: t.percentile(95.0),
                p99: t.percentile(99.0),
                min: t.min(),
                max: t.max(),
                labels: t.labels.clone(),
            })
            .collect();

        MetricSnapshot {
            timestamp: Utc::now(),
            counters: counter_snapshots,
            gauges: gauge_snapshots,
            histograms: histogram_snapshots,
            timers: timer_snapshots,
        }
    }

    /// Initialize common Claude Interactive metrics
    pub async fn initialize_default_metrics(&self) -> Result<()> {
        // Counters
        self.register_counter(
            "commands_total".to_string(),
            "Total number of commands executed".to_string(),
            HashMap::new(),
        )
        .await?;

        self.register_counter(
            "commands_failed_total".to_string(),
            "Total number of failed commands".to_string(),
            HashMap::new(),
        )
        .await?;

        self.register_counter(
            "sessions_created_total".to_string(),
            "Total number of sessions created".to_string(),
            HashMap::new(),
        )
        .await?;

        // Gauges
        self.register_gauge(
            "active_sessions".to_string(),
            "Number of currently active sessions".to_string(),
            HashMap::new(),
        )
        .await?;

        self.register_gauge(
            "total_cost_usd".to_string(),
            "Total cost in USD".to_string(),
            HashMap::new(),
        )
        .await?;

        self.register_gauge(
            "memory_usage_mb".to_string(),
            "Memory usage in megabytes".to_string(),
            HashMap::new(),
        )
        .await?;

        // Histograms
        let response_time_buckets =
            vec![10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 30000.0];
        self.register_histogram(
            "command_duration_ms".to_string(),
            "Command execution duration in milliseconds".to_string(),
            response_time_buckets,
            HashMap::new(),
        )
        .await?;

        let cost_buckets = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
        self.register_histogram(
            "command_cost_usd".to_string(),
            "Command cost in USD".to_string(),
            cost_buckets,
            HashMap::new(),
        )
        .await?;

        // Timers
        self.register_timer(
            "api_response_time".to_string(),
            "Claude API response time".to_string(),
            HashMap::new(),
        )
        .await?;

        self.register_timer(
            "session_duration".to_string(),
            "Session duration from start to end".to_string(),
            HashMap::new(),
        )
        .await?;

        Ok(())
    }

    /// Export metrics in Prometheus format
    pub async fn export_prometheus(&self) -> String {
        let snapshot = self.snapshot().await;
        let mut output = String::new();

        // Export counters
        for counter in &snapshot.counters {
            output.push_str(&format!("# HELP {} {}\n", counter.name, "Counter metric"));
            output.push_str(&format!("# TYPE {} counter\n", counter.name));

            let labels = if counter.labels.is_empty() {
                String::new()
            } else {
                let label_str: Vec<String> = counter
                    .labels
                    .iter()
                    .map(|(k, v)| format!("{}=\"{}\"", k, v))
                    .collect();
                format!("{{{}}}", label_str.join(","))
            };

            output.push_str(&format!("{}{} {}\n", counter.name, labels, counter.value));
        }

        // Export gauges
        for gauge in &snapshot.gauges {
            output.push_str(&format!("# HELP {} {}\n", gauge.name, "Gauge metric"));
            output.push_str(&format!("# TYPE {} gauge\n", gauge.name));

            let labels = if gauge.labels.is_empty() {
                String::new()
            } else {
                let label_str: Vec<String> = gauge
                    .labels
                    .iter()
                    .map(|(k, v)| format!("{}=\"{}\"", k, v))
                    .collect();
                format!("{{{}}}", label_str.join(","))
            };

            output.push_str(&format!("{}{} {}\n", gauge.name, labels, gauge.value));
        }

        // Export histograms
        for histogram in &snapshot.histograms {
            output.push_str(&format!(
                "# HELP {} {}\n",
                histogram.name, "Histogram metric"
            ));
            output.push_str(&format!("# TYPE {} histogram\n", histogram.name));

            for (percentile, value) in &histogram.percentiles {
                output.push_str(&format!(
                    "{}_bucket{{le=\"{}\"}} {}\n",
                    histogram.name, percentile, value
                ));
            }

            output.push_str(&format!("{}_count {}\n", histogram.name, histogram.count));
            output.push_str(&format!("{}_sum {}\n", histogram.name, histogram.sum));
        }

        output
    }

    /// Clean up old metric data
    pub async fn cleanup_old_data(&self) -> Result<()> {
        let _cutoff = Utc::now() - Duration::hours(self.config.retention_duration_hours as i64);

        // For now, we don't have time-series data to clean up
        // In a full implementation, this would remove old histogram buckets, timer samples, etc.

        Ok(())
    }
}

/// Utility for timing operations
pub struct MetricTimer<'a> {
    registry: &'a MetricRegistry,
    timer_name: String,
    start_time: std::time::Instant,
}

impl<'a> MetricTimer<'a> {
    pub fn new(registry: &'a MetricRegistry, timer_name: String) -> Self {
        Self {
            registry,
            timer_name,
            start_time: std::time::Instant::now(),
        }
    }

    pub async fn finish(self) -> Result<()> {
        let duration = self.start_time.elapsed();
        let duration_ms = duration.as_millis() as f64;
        self.registry
            .record_timer(&self.timer_name, duration_ms)
            .await
    }
}
