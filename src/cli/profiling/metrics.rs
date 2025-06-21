//! Metrics collection and analysis utilities

use super::*;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Real-time metrics collector for ongoing performance monitoring
pub struct MetricsCollector {
    operation_times: Arc<Mutex<VecDeque<Duration>>>,
    memory_samples: Arc<Mutex<VecDeque<f64>>>,
    error_counts: Arc<Mutex<HashMap<String, usize>>>,
    max_samples: usize,
    collection_interval: Duration,
}

/// Collected metrics over a time period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectedMetrics {
    pub sample_count: usize,
    pub time_period: Duration,
    pub operation_stats: OperationStats,
    pub memory_stats: MemoryStats,
    pub error_stats: ErrorStats,
    pub percentiles: PercentileStats,
}

/// Statistical analysis of operation times
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    pub total_operations: usize,
    pub avg_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub median_duration: Duration,
    pub std_deviation: Duration,
    pub operations_per_second: f64,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub avg_memory_mb: f64,
    pub min_memory_mb: f64,
    pub max_memory_mb: f64,
    pub memory_variance: f64,
    pub peak_usage_times: Vec<DateTime<Utc>>,
}

/// Error occurrence statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStats {
    pub total_errors: usize,
    pub error_rate: f64,
    pub error_types: HashMap<String, usize>,
    pub most_common_error: Option<String>,
}

/// Percentile statistics for operation times
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PercentileStats {
    pub p50: Duration,
    pub p75: Duration,
    pub p90: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub time_period: Duration,
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub performance_degradation: Option<f64>,
    pub recommendations: Vec<String>,
}

/// Direction of performance trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(max_samples: usize, collection_interval: Duration) -> Self {
        Self {
            operation_times: Arc::new(Mutex::new(VecDeque::with_capacity(max_samples))),
            memory_samples: Arc::new(Mutex::new(VecDeque::with_capacity(max_samples))),
            error_counts: Arc::new(Mutex::new(HashMap::new())),
            max_samples,
            collection_interval,
        }
    }

    /// Record an operation duration
    pub fn record_operation(&self, duration: Duration) {
        let mut times = self.operation_times.lock().unwrap();
        if times.len() >= self.max_samples {
            times.pop_front();
        }
        times.push_back(duration);
    }

    /// Record memory usage sample
    pub fn record_memory_usage(&self, memory_mb: f64) {
        let mut samples = self.memory_samples.lock().unwrap();
        if samples.len() >= self.max_samples {
            samples.pop_front();
        }
        samples.push_back(memory_mb);
    }

    /// Record an error occurrence
    pub fn record_error(&self, error_type: &str) {
        let mut errors = self.error_counts.lock().unwrap();
        *errors.entry(error_type.to_string()).or_insert(0) += 1;
    }

    /// Collect and analyze all metrics
    pub fn collect_metrics(&self, time_period: Duration) -> CollectedMetrics {
        let operation_times = self.operation_times.lock().unwrap().clone();
        let memory_samples = self.memory_samples.lock().unwrap().clone();
        let error_counts = self.error_counts.lock().unwrap().clone();

        let operation_stats = self.calculate_operation_stats(&operation_times, time_period);
        let memory_stats = self.calculate_memory_stats(&memory_samples);
        let error_stats = self.calculate_error_stats(&error_counts, operation_times.len());
        let percentiles = self.calculate_percentiles(&operation_times);

        CollectedMetrics {
            sample_count: operation_times.len(),
            time_period,
            operation_stats,
            memory_stats,
            error_stats,
            percentiles,
        }
    }

    /// Calculate operation statistics
    fn calculate_operation_stats(
        &self,
        times: &VecDeque<Duration>,
        time_period: Duration,
    ) -> OperationStats {
        if times.is_empty() {
            return OperationStats {
                total_operations: 0,
                avg_duration: Duration::ZERO,
                min_duration: Duration::ZERO,
                max_duration: Duration::ZERO,
                median_duration: Duration::ZERO,
                std_deviation: Duration::ZERO,
                operations_per_second: 0.0,
            };
        }

        let total_operations = times.len();
        let total_duration: Duration = times.iter().sum();
        let avg_duration = total_duration / total_operations as u32;

        let min_duration = *times.iter().min().unwrap();
        let max_duration = *times.iter().max().unwrap();

        let mut sorted_times: Vec<Duration> = times.iter().cloned().collect();
        sorted_times.sort();
        let median_duration = sorted_times[sorted_times.len() / 2];

        // Calculate standard deviation
        let variance: f64 = times
            .iter()
            .map(|&d| {
                let diff = d.as_nanos() as f64 - avg_duration.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>()
            / total_operations as f64;

        let std_deviation = Duration::from_nanos(variance.sqrt() as u64);

        let operations_per_second = if time_period.as_secs_f64() > 0.0 {
            total_operations as f64 / time_period.as_secs_f64()
        } else {
            0.0
        };

        OperationStats {
            total_operations,
            avg_duration,
            min_duration,
            max_duration,
            median_duration,
            std_deviation,
            operations_per_second,
        }
    }

    /// Calculate memory statistics
    fn calculate_memory_stats(&self, samples: &VecDeque<f64>) -> MemoryStats {
        if samples.is_empty() {
            return MemoryStats {
                avg_memory_mb: 0.0,
                min_memory_mb: 0.0,
                max_memory_mb: 0.0,
                memory_variance: 0.0,
                peak_usage_times: Vec::new(),
            };
        }

        let avg_memory_mb = samples.iter().sum::<f64>() / samples.len() as f64;
        let min_memory_mb = samples.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_memory_mb = samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let memory_variance = samples
            .iter()
            .map(|&m| (m - avg_memory_mb).powi(2))
            .sum::<f64>()
            / samples.len() as f64;

        // Find peak usage times (simplified - would need timestamps in real implementation)
        let peak_usage_times = Vec::new();

        MemoryStats {
            avg_memory_mb,
            min_memory_mb,
            max_memory_mb,
            memory_variance,
            peak_usage_times,
        }
    }

    /// Calculate error statistics
    fn calculate_error_stats(
        &self,
        errors: &HashMap<String, usize>,
        total_operations: usize,
    ) -> ErrorStats {
        let total_errors: usize = errors.values().sum();
        let error_rate = if total_operations > 0 {
            total_errors as f64 / total_operations as f64
        } else {
            0.0
        };

        let most_common_error = errors
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(error_type, _)| error_type.clone());

        ErrorStats {
            total_errors,
            error_rate,
            error_types: errors.clone(),
            most_common_error,
        }
    }

    /// Calculate percentile statistics
    fn calculate_percentiles(&self, times: &VecDeque<Duration>) -> PercentileStats {
        if times.is_empty() {
            return PercentileStats {
                p50: Duration::ZERO,
                p75: Duration::ZERO,
                p90: Duration::ZERO,
                p95: Duration::ZERO,
                p99: Duration::ZERO,
            };
        }

        let mut sorted_times: Vec<Duration> = times.iter().cloned().collect();
        sorted_times.sort();

        let p50 = self.percentile(&sorted_times, 0.50);
        let p75 = self.percentile(&sorted_times, 0.75);
        let p90 = self.percentile(&sorted_times, 0.90);
        let p95 = self.percentile(&sorted_times, 0.95);
        let p99 = self.percentile(&sorted_times, 0.99);

        PercentileStats {
            p50,
            p75,
            p90,
            p95,
            p99,
        }
    }

    /// Calculate a specific percentile from sorted data
    fn percentile(&self, sorted_data: &[Duration], percentile: f64) -> Duration {
        if sorted_data.is_empty() {
            return Duration::ZERO;
        }

        let index = (percentile * (sorted_data.len() - 1) as f64) as usize;
        sorted_data[index.min(sorted_data.len() - 1)]
    }

    /// Analyze performance trends over time
    pub fn analyze_trends(&self, time_windows: &[CollectedMetrics]) -> PerformanceTrend {
        if time_windows.len() < 2 {
            return PerformanceTrend {
                time_period: Duration::ZERO,
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.0,
                performance_degradation: None,
                recommendations: vec!["Need more data points for trend analysis".to_string()],
            };
        }

        let total_time_period = time_windows.iter().map(|w| w.time_period).sum();

        // Analyze operation performance trend
        let ops_per_sec: Vec<f64> = time_windows
            .iter()
            .map(|w| w.operation_stats.operations_per_second)
            .collect();

        let trend_strength = self.calculate_trend_strength(&ops_per_sec);
        let trend_direction = self.determine_trend_direction(&ops_per_sec, trend_strength);

        // Calculate performance degradation if applicable
        let performance_degradation = if matches!(trend_direction, TrendDirection::Degrading) {
            let first = ops_per_sec.first().unwrap_or(&0.0);
            let last = ops_per_sec.last().unwrap_or(&0.0);
            if *first > 0.0 {
                Some((first - last) / first * 100.0)
            } else {
                None
            }
        } else {
            None
        };

        let recommendations = self.generate_recommendations(&trend_direction, time_windows);

        PerformanceTrend {
            time_period: total_time_period,
            trend_direction,
            trend_strength,
            performance_degradation,
            recommendations,
        }
    }

    /// Calculate trend strength using linear regression slope
    fn calculate_trend_strength(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let sum_x: f64 = x_values.iter().sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = x_values.iter().zip(values).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = x_values.iter().map(|x| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        slope.abs()
    }

    /// Determine trend direction based on values and strength
    fn determine_trend_direction(&self, values: &[f64], strength: f64) -> TrendDirection {
        if strength < 0.1 {
            return TrendDirection::Stable;
        }

        if values.len() < 2 {
            return TrendDirection::Stable;
        }

        let first_half: f64 =
            values[..values.len() / 2].iter().sum::<f64>() / (values.len() / 2) as f64;
        let second_half: f64 = values[values.len() / 2..].iter().sum::<f64>()
            / (values.len() - values.len() / 2) as f64;

        // Check for volatility
        let variance = values
            .iter()
            .map(|&v| (v - (values.iter().sum::<f64>() / values.len() as f64)).powi(2))
            .sum::<f64>()
            / values.len() as f64;

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let coefficient_of_variation = if mean > 0.0 {
            variance.sqrt() / mean
        } else {
            0.0
        };

        if coefficient_of_variation > 0.5 {
            return TrendDirection::Volatile;
        }

        if second_half > first_half * 1.05 {
            TrendDirection::Improving
        } else if second_half < first_half * 0.95 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        }
    }

    /// Generate performance recommendations based on trend analysis
    fn generate_recommendations(
        &self,
        direction: &TrendDirection,
        windows: &[CollectedMetrics],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        match direction {
            TrendDirection::Degrading => {
                recommendations.push("Performance is degrading over time".to_string());
                recommendations.push("Consider optimizing frequently used operations".to_string());
                recommendations
                    .push("Check for memory leaks or growing data structures".to_string());

                // Check for high error rates
                if let Some(latest) = windows.last() {
                    if latest.error_stats.error_rate > 0.05 {
                        recommendations.push(
                            "High error rate detected - investigate error causes".to_string(),
                        );
                    }

                    if latest.memory_stats.max_memory_mb > latest.memory_stats.avg_memory_mb * 2.0 {
                        recommendations.push(
                            "Memory usage spikes detected - consider memory optimization"
                                .to_string(),
                        );
                    }
                }
            }
            TrendDirection::Volatile => {
                recommendations.push("Performance is highly variable".to_string());
                recommendations
                    .push("Consider implementing caching or connection pooling".to_string());
                recommendations.push(
                    "Investigate external dependencies that may cause variability".to_string(),
                );
            }
            TrendDirection::Stable => {
                recommendations.push("Performance is stable".to_string());
                recommendations
                    .push("Consider load testing to identify scalability limits".to_string());
            }
            TrendDirection::Improving => {
                recommendations.push("Performance is improving - good trend!".to_string());
                recommendations.push("Monitor to ensure improvements are sustained".to_string());
            }
        }

        recommendations
    }

    /// Clear all collected metrics
    pub fn clear_metrics(&self) {
        self.operation_times.lock().unwrap().clear();
        self.memory_samples.lock().unwrap().clear();
        self.error_counts.lock().unwrap().clear();
    }

    /// Get current metrics snapshot without clearing
    pub fn snapshot(&self) -> CollectedMetrics {
        self.collect_metrics(self.collection_interval)
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new(10000, Duration::from_secs(60))
    }
}

/// Create a metrics collector optimized for performance monitoring
pub fn create_performance_monitor() -> MetricsCollector {
    MetricsCollector::new(50000, Duration::from_secs(300)) // 5-minute windows
}

/// Create a metrics collector for real-time monitoring
pub fn create_realtime_monitor() -> MetricsCollector {
    MetricsCollector::new(1000, Duration::from_secs(10)) // 10-second windows
}
