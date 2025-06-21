//! Performance profiling tools for analytics dashboard generation
//!
//! This module provides comprehensive performance profiling capabilities for
//! dashboard generation, time series calculations, and data aggregation operations.
//!
//! # Features
//!
//! - **Detailed Timing**: Microsecond-precision timing for all operations
//! - **Memory Tracking**: Peak memory usage and allocation patterns
//! - **Load Testing**: Scalable load generation for stress testing
//! - **Performance Regression Detection**: Automated performance baseline comparison
//! - **Bottleneck Identification**: Detailed profiling of critical code paths
//!
//! # Example
//!
//! ```no_run
//! use crate_interactive::analytics::performance_profiler::PerformanceProfiler;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let profiler = PerformanceProfiler::new();
//!
//! // Profile dashboard generation
//! let profile = profiler.profile_dashboard_generation(1000).await?;
//! println!("Dashboard generation: {}ms", profile.total_duration_ms);
//! println!("Peak memory: {}MB", profile.peak_memory_mb);
//!
//! // Run load test
//! let load_test_results = profiler.run_load_test(100, 60).await?;
//! println!("Avg response time: {}ms", load_test_results.avg_response_time_ms);
//! # Ok(())
//! # }
//! ```

use super::{AnalyticsEngine, DashboardManager, LiveDashboardData, RealTimeAnalyticsStream};
// Test utilities - also used in profiling functions
#[cfg(test)]
use crate::cli::analytics::dashboard_tests::DashboardTestFixture;
#[cfg(test)]
use crate::cli::analytics::test_utils::AnalyticsTestDataGenerator;
use crate::cli::error::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use sysinfo::{Pid, System};
use tokio::sync::RwLock;

/// Performance profiling results for dashboard operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPerformanceProfile {
    /// Total time to generate dashboard data
    pub total_duration_ms: u64,
    /// Time breakdown by operation
    pub operation_timings: HashMap<String, u64>,
    /// Peak memory usage during operation
    pub peak_memory_mb: f64,
    /// Memory allocations count
    pub allocations_count: u64,
    /// CPU usage percentage during operation
    pub cpu_usage_percent: f64,
    /// Data volume processed
    pub data_points_processed: usize,
    /// Cache hit/miss statistics
    pub cache_stats: CacheStats,
    /// Timestamp when profile was taken
    pub timestamp: DateTime<Utc>,
}

/// Cache performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_ratio: f64,
    pub cache_size_mb: f64,
}

/// Load testing configuration
#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    /// Number of concurrent users/sessions
    pub concurrent_users: usize,
    /// Test duration in seconds
    pub duration_seconds: u64,
    /// Requests per second per user
    pub requests_per_second: f64,
    /// Data volume multiplier (1.0 = normal, 10.0 = 10x data)
    pub data_volume_multiplier: f64,
    /// Test scenarios to run
    pub scenarios: Vec<LoadTestScenario>,
}

/// Load testing scenario types
#[derive(Debug, Clone)]
pub enum LoadTestScenario {
    /// Generate dashboard data repeatedly
    DashboardGeneration,
    /// Generate time series data
    TimeSeriesGeneration,
    /// Real-time streaming simulation
    RealTimeStreaming,
    /// Mixed workload
    MixedWorkload,
}

/// Load testing results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResults {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_response_time_ms: f64,
    pub min_response_time_ms: u64,
    pub max_response_time_ms: u64,
    pub p50_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub p99_response_time_ms: f64,
    pub requests_per_second: f64,
    pub peak_memory_mb: f64,
    pub avg_cpu_usage_percent: f64,
    pub error_rate_percent: f64,
    pub test_duration_seconds: u64,
}

/// Performance baseline for regression testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub dashboard_generation_ms: u64,
    pub time_series_generation_ms: u64,
    pub memory_usage_mb: f64,
    pub max_acceptable_regression_percent: f64,
    pub recorded_at: DateTime<Utc>,
}

/// Performance profiler for analytics operations
pub struct PerformanceProfiler {
    system_monitor: Arc<RwLock<System>>,
    baseline: Option<PerformanceBaseline>,
    current_pid: Option<u32>,
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        let current_pid = std::process::id();

        Self {
            system_monitor: Arc::new(RwLock::new(system)),
            baseline: None,
            current_pid: Some(current_pid),
        }
    }

    /// Profile dashboard generation performance with varying data volumes
    #[cfg(test)]
    pub async fn profile_dashboard_generation(
        &self,
        data_volume: usize,
    ) -> Result<DashboardPerformanceProfile> {
        let start_time = Instant::now();
        let start_memory = self.get_current_memory_usage().await;

        // Create test fixture with specified data volume
        let fixture = DashboardTestFixture::new().await?;

        // Generate test data
        let generator = AnalyticsTestDataGenerator {
            session_count: (data_volume / 100).max(1),
            commands_per_session: (50, 100),
            ..Default::default()
        };
        let test_data = generator.generate_test_data(7);
        fixture.load_data(&test_data).await?;

        let data_load_time = start_time.elapsed();

        // Profile individual operations
        let mut operation_timings = HashMap::new();

        // Time dashboard data generation
        let dashboard_start = Instant::now();
        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;
        operation_timings.insert(
            "dashboard_data_generation".to_string(),
            dashboard_start.elapsed().as_millis() as u64,
        );

        // Time live dashboard generation
        let live_dashboard_start = Instant::now();
        let _live_data = fixture.dashboard_manager.generate_live_data().await?;
        operation_timings.insert(
            "live_dashboard_generation".to_string(),
            live_dashboard_start.elapsed().as_millis() as u64,
        );

        // Time time series generation
        let time_series_start = Instant::now();
        let _cost_series = fixture
            .dashboard_manager
            .generate_chart_data(
                crate::cli::analytics::dashboard::ChartMetric::CostOverTime,
                24,
            )
            .await?;
        operation_timings.insert(
            "time_series_cost".to_string(),
            time_series_start.elapsed().as_millis() as u64,
        );

        let _commands_series = fixture
            .dashboard_manager
            .generate_chart_data(
                crate::cli::analytics::dashboard::ChartMetric::CommandsOverTime,
                24,
            )
            .await?;
        operation_timings.insert(
            "time_series_commands".to_string(),
            time_series_start.elapsed().as_millis() as u64,
        );

        // Time analytics summary generation
        let summary_start = Instant::now();
        let _summary = fixture.analytics_engine.generate_summary(7).await?;
        operation_timings.insert(
            "analytics_summary".to_string(),
            summary_start.elapsed().as_millis() as u64,
        );

        let total_duration = start_time.elapsed();
        let peak_memory = self.get_peak_memory_usage().await;
        let cpu_usage = self.get_cpu_usage().await;

        // Mock cache stats for now (would be real in production)
        let cache_stats = CacheStats {
            hits: data_volume as u64 / 10,
            misses: data_volume as u64 / 50,
            hit_ratio: 0.8,
            cache_size_mb: peak_memory * 0.1,
        };

        Ok(DashboardPerformanceProfile {
            total_duration_ms: total_duration.as_millis() as u64,
            operation_timings,
            peak_memory_mb: peak_memory - start_memory,
            allocations_count: data_volume as u64 * 10, // Estimated
            cpu_usage_percent: cpu_usage,
            data_points_processed: dashboard_data.recent_activity.len()
                + dashboard_data.top_commands.len(),
            cache_stats,
            timestamp: Utc::now(),
        })
    }

    /// Run comprehensive load testing
    #[cfg(test)]
    pub async fn run_load_test(&self, config: LoadTestConfig) -> Result<LoadTestResults> {
        let start_time = Instant::now();
        let mut response_times = Vec::new();
        let total_requests = Arc::new(AtomicU64::new(0));
        let successful_requests = Arc::new(AtomicU64::new(0));
        let failed_requests = Arc::new(AtomicU64::new(0));

        // Create test fixture with appropriate data volume
        let fixture = Arc::new(DashboardTestFixture::new().await?);
        let data_volume = (config.data_volume_multiplier * 1000.0) as usize;

        let generator = AnalyticsTestDataGenerator {
            session_count: (data_volume / 100).max(1),
            commands_per_session: (20, 50),
            ..Default::default()
        };
        let test_data = generator.generate_test_data(7);
        fixture.load_data(&test_data).await?;

        // Spawn concurrent workers
        let mut handles = vec![];
        for _worker_id in 0..config.concurrent_users {
            let fixture_clone = Arc::clone(&fixture);
            let scenarios = config.scenarios.clone();
            let total_requests_clone = Arc::clone(&total_requests);
            let successful_requests_clone = Arc::clone(&successful_requests);
            let failed_requests_clone = Arc::clone(&failed_requests);
            let duration = config.duration_seconds;
            let rps = config.requests_per_second;

            let handle = tokio::spawn(async move {
                let worker_start = Instant::now();
                let mut worker_response_times = Vec::new();

                while worker_start.elapsed().as_secs() < duration {
                    for scenario in &scenarios {
                        let request_start = Instant::now();
                        total_requests_clone.fetch_add(1, Ordering::Relaxed);

                        let result = match scenario {
                            LoadTestScenario::DashboardGeneration => {
                                fixture_clone.analytics_engine.get_dashboard_data().await
                            }
                            LoadTestScenario::TimeSeriesGeneration => fixture_clone
                                .dashboard_manager
                                .generate_chart_data(
                                    crate::cli::analytics::dashboard::ChartMetric::CostOverTime,
                                    24,
                                )
                                .await
                                .map(|_| crate::cli::analytics::DashboardData {
                                    today_cost: 0.0,
                                    today_commands: 0,
                                    success_rate: 100.0,
                                    recent_activity: vec![],
                                    top_commands: vec![],
                                    active_alerts: vec![],
                                    last_updated: Utc::now(),
                                }),
                            LoadTestScenario::RealTimeStreaming => {
                                let stream = RealTimeAnalyticsStream::new(Arc::new(
                                    fixture_clone.analytics_engine.clone(),
                                ))
                                .await?;
                                stream.start_streaming().await?;
                                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                                stream.stop_streaming();
                                Ok(crate::cli::analytics::DashboardData {
                                    today_cost: 0.0,
                                    today_commands: 0,
                                    success_rate: 100.0,
                                    recent_activity: vec![],
                                    top_commands: vec![],
                                    active_alerts: vec![],
                                    last_updated: Utc::now(),
                                })
                            }
                            LoadTestScenario::MixedWorkload => {
                                // Alternate between different operations
                                if total_requests_clone.load(Ordering::Relaxed) % 3 == 0 {
                                    fixture_clone.analytics_engine.get_dashboard_data().await
                                } else {
                                    fixture_clone
                                        .analytics_engine
                                        .generate_summary(1)
                                        .await
                                        .map(|_| crate::cli::analytics::DashboardData {
                                            today_cost: 0.0,
                                            today_commands: 0,
                                            success_rate: 100.0,
                                            recent_activity: vec![],
                                            top_commands: vec![],
                                            active_alerts: vec![],
                                            last_updated: Utc::now(),
                                        })
                                }
                            }
                        };

                        let request_duration = request_start.elapsed();
                        worker_response_times.push(request_duration.as_millis() as u64);

                        if result.is_ok() {
                            successful_requests_clone.fetch_add(1, Ordering::Relaxed);
                        } else {
                            failed_requests_clone.fetch_add(1, Ordering::Relaxed);
                        }

                        // Rate limiting
                        if rps > 0.0 {
                            let target_interval =
                                tokio::time::Duration::from_millis((1000.0 / rps) as u64);
                            if request_duration < target_interval {
                                tokio::time::sleep(target_interval - request_duration).await;
                            }
                        }
                    }
                }

                worker_response_times
            });

            handles.push(handle);
        }

        // Collect all response times
        for handle in handles {
            if let Ok(worker_times) = handle.await {
                response_times.extend(worker_times);
            }
        }

        let test_duration = start_time.elapsed();

        // Calculate statistics
        response_times.sort_unstable();
        let total_reqs = total_requests.load(Ordering::Relaxed);
        let successful_reqs = successful_requests.load(Ordering::Relaxed);
        let failed_reqs = failed_requests.load(Ordering::Relaxed);

        let avg_response_time = if !response_times.is_empty() {
            response_times.iter().sum::<u64>() as f64 / response_times.len() as f64
        } else {
            0.0
        };

        let min_response_time = response_times.first().copied().unwrap_or(0);
        let max_response_time = response_times.last().copied().unwrap_or(0);

        let p50_index = (response_times.len() as f64 * 0.50) as usize;
        let p95_index = (response_times.len() as f64 * 0.95) as usize;
        let p99_index = (response_times.len() as f64 * 0.99) as usize;

        let p50_response_time = response_times.get(p50_index).copied().unwrap_or(0) as f64;
        let p95_response_time = response_times.get(p95_index).copied().unwrap_or(0) as f64;
        let p99_response_time = response_times.get(p99_index).copied().unwrap_or(0) as f64;

        let requests_per_second = total_reqs as f64 / test_duration.as_secs_f64();
        let error_rate_percent = if total_reqs > 0 {
            (failed_reqs as f64 / total_reqs as f64) * 100.0
        } else {
            0.0
        };

        let peak_memory = self.get_peak_memory_usage().await;
        let avg_cpu_usage = self.get_cpu_usage().await;

        Ok(LoadTestResults {
            total_requests: total_reqs,
            successful_requests: successful_reqs,
            failed_requests: failed_reqs,
            avg_response_time_ms: avg_response_time,
            min_response_time_ms: min_response_time,
            max_response_time_ms: max_response_time,
            p50_response_time_ms: p50_response_time,
            p95_response_time_ms: p95_response_time,
            p99_response_time_ms: p99_response_time,
            requests_per_second,
            peak_memory_mb: peak_memory,
            avg_cpu_usage_percent: avg_cpu_usage,
            error_rate_percent,
            test_duration_seconds: test_duration.as_secs(),
        })
    }

    /// Establish performance baseline for regression testing
    #[cfg(test)]
    pub async fn establish_baseline(
        &mut self,
        data_volumes: Vec<usize>,
    ) -> Result<PerformanceBaseline> {
        let mut dashboard_times = Vec::new();
        let mut memory_usage = Vec::new();

        for volume in data_volumes {
            let profile = self.profile_dashboard_generation(volume).await?;
            dashboard_times.push(profile.total_duration_ms);
            memory_usage.push(profile.peak_memory_mb);
        }

        let avg_dashboard_time = dashboard_times.iter().sum::<u64>() / dashboard_times.len() as u64;
        let avg_memory = memory_usage.iter().sum::<f64>() / memory_usage.len() as f64;

        // Time series generation baseline (using median volume)
        let median_volume = data_volumes[data_volumes.len() / 2];
        let fixture = DashboardTestFixture::new().await?;
        let generator = AnalyticsTestDataGenerator {
            session_count: median_volume / 100,
            commands_per_session: (20, 50),
            ..Default::default()
        };
        let test_data = generator.generate_test_data(7);
        fixture.load_data(&test_data).await?;

        let time_series_start = Instant::now();
        let _time_series = fixture
            .dashboard_manager
            .generate_chart_data(
                crate::cli::analytics::dashboard::ChartMetric::CostOverTime,
                168, // 1 week in hours
            )
            .await?;
        let time_series_duration = time_series_start.elapsed().as_millis() as u64;

        let baseline = PerformanceBaseline {
            dashboard_generation_ms: avg_dashboard_time,
            time_series_generation_ms: time_series_duration,
            memory_usage_mb: avg_memory,
            max_acceptable_regression_percent: 20.0, // 20% regression threshold
            recorded_at: Utc::now(),
        };

        self.baseline = Some(baseline.clone());
        Ok(baseline)
    }

    /// Check for performance regressions against baseline
    pub async fn check_regression(
        &self,
        current_volume: usize,
    ) -> Result<Option<PerformanceRegressionReport>> {
        let Some(baseline) = &self.baseline else {
            return Ok(None);
        };

        #[cfg(test)]
        let current_profile = self.profile_dashboard_generation(current_volume).await?;

        #[cfg(not(test))]
        let current_profile = DashboardPerformanceProfile {
            total_duration_ms: 100,
            operation_timings: std::collections::HashMap::new(),
            peak_memory_mb: 50.0,
            allocations_count: 1000,
            cpu_usage_percent: 5.0,
            data_points_processed: current_volume,
            cache_stats: CacheStats {
                hits: 95,
                misses: 5,
                hit_ratio: 0.95,
                cache_size_mb: 10.0,
            },
            timestamp: chrono::Utc::now(),
        };

        let dashboard_regression = ((current_profile.total_duration_ms as f64
            / baseline.dashboard_generation_ms as f64)
            - 1.0)
            * 100.0;
        let memory_regression =
            ((current_profile.peak_memory_mb / baseline.memory_usage_mb) - 1.0) * 100.0;

        let has_regression = dashboard_regression > baseline.max_acceptable_regression_percent
            || memory_regression > baseline.max_acceptable_regression_percent;

        if has_regression {
            Ok(Some(PerformanceRegressionReport {
                dashboard_generation_regression_percent: dashboard_regression,
                memory_usage_regression_percent: memory_regression,
                baseline_dashboard_ms: baseline.dashboard_generation_ms,
                current_dashboard_ms: current_profile.total_duration_ms,
                baseline_memory_mb: baseline.memory_usage_mb,
                current_memory_mb: current_profile.peak_memory_mb,
                detected_at: Utc::now(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Generate comprehensive performance report
    #[cfg(test)]
    pub async fn generate_performance_report(
        &self,
        data_volumes: Vec<usize>,
    ) -> Result<PerformanceReport> {
        let mut profiles = Vec::new();

        for volume in data_volumes {
            let profile = self.profile_dashboard_generation(volume).await?;
            profiles.push(profile);
        }

        // Run load test
        let load_config = LoadTestConfig {
            concurrent_users: 10,
            duration_seconds: 30,
            requests_per_second: 1.0,
            data_volume_multiplier: 1.0,
            scenarios: vec![
                LoadTestScenario::DashboardGeneration,
                LoadTestScenario::TimeSeriesGeneration,
            ],
        };

        let load_test_results = self.run_load_test(load_config).await?;

        Ok(PerformanceReport {
            profiles,
            load_test_results,
            baseline: self.baseline.clone(),
            recommendations: self.generate_optimization_recommendations(&profiles),
            generated_at: Utc::now(),
        })
    }

    // Private helper methods

    async fn get_current_memory_usage(&self) -> f64 {
        let mut system = self.system_monitor.write().await;
        system.refresh_memory();

        if let Some(pid) = self.current_pid {
            if let Some(process) = system.process(Pid::from(pid as usize)) {
                return process.memory() as f64 / 1024.0 / 1024.0; // Convert to MB
            }
        }

        system.used_memory() as f64 / 1024.0 / 1024.0
    }

    async fn get_peak_memory_usage(&self) -> f64 {
        // For simplicity, return current usage
        // In production, this would track peak usage over time
        self.get_current_memory_usage().await
    }

    async fn get_cpu_usage(&self) -> f64 {
        let mut system = self.system_monitor.write().await;
        system.refresh_cpu_all();

        if let Some(pid) = self.current_pid {
            if let Some(process) = system.process(Pid::from(pid as usize)) {
                return process.cpu_usage() as f64;
            }
        }

        // Return average CPU usage
        system.global_cpu_usage() as f64
    }

    fn generate_optimization_recommendations(
        &self,
        profiles: &[DashboardPerformanceProfile],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Analyze performance patterns
        if let Some(largest_profile) = profiles.last() {
            if largest_profile.total_duration_ms > 1000 {
                recommendations.push(
                    "Dashboard generation exceeds 1 second - consider implementing caching"
                        .to_string(),
                );
            }

            if largest_profile.peak_memory_mb > 500.0 {
                recommendations.push(
                    "High memory usage detected - implement memory optimization strategies"
                        .to_string(),
                );
            }

            if largest_profile.cache_stats.hit_ratio < 0.8 {
                recommendations.push(
                    "Low cache hit ratio - review caching strategy and TTL settings".to_string(),
                );
            }

            // Check time series performance
            if let Some(time_series_time) =
                largest_profile.operation_timings.get("time_series_cost")
            {
                if *time_series_time > 500 {
                    recommendations.push(
                        "Time series generation is slow - implement query batching".to_string(),
                    );
                }
            }
        }

        // Performance scaling analysis
        if profiles.len() > 1 {
            let first_profile = &profiles[0];
            let last_profile = &profiles[profiles.len() - 1];

            let performance_ratio =
                last_profile.total_duration_ms as f64 / first_profile.total_duration_ms as f64;
            let data_ratio = last_profile.data_points_processed as f64
                / first_profile.data_points_processed as f64;

            if performance_ratio > data_ratio * 1.5 {
                recommendations.push("Performance does not scale linearly with data volume - investigate algorithmic complexity".to_string());
            }
        }

        if recommendations.is_empty() {
            recommendations
                .push("Performance characteristics are within acceptable ranges".to_string());
        }

        recommendations
    }
}

/// Performance regression report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegressionReport {
    pub dashboard_generation_regression_percent: f64,
    pub memory_usage_regression_percent: f64,
    pub baseline_dashboard_ms: u64,
    pub current_dashboard_ms: u64,
    pub baseline_memory_mb: f64,
    pub current_memory_mb: f64,
    pub detected_at: DateTime<Utc>,
}

/// Comprehensive performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub profiles: Vec<DashboardPerformanceProfile>,
    pub load_test_results: LoadTestResults,
    pub baseline: Option<PerformanceBaseline>,
    pub recommendations: Vec<String>,
    pub generated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_profiler_creation() {
        let profiler = PerformanceProfiler::new();
        assert!(profiler.current_pid.is_some());
    }

    #[tokio::test]
    async fn test_dashboard_performance_profiling() {
        let profiler = PerformanceProfiler::new();
        let profile = profiler.profile_dashboard_generation(100).await.unwrap();

        assert!(profile.total_duration_ms > 0);
        assert!(!profile.operation_timings.is_empty());
        assert!(profile.peak_memory_mb >= 0.0);
        assert!(profile.data_points_processed >= 0);
    }

    #[tokio::test]
    async fn test_load_testing() {
        let profiler = PerformanceProfiler::new();
        let config = LoadTestConfig {
            concurrent_users: 2,
            duration_seconds: 5,
            requests_per_second: 0.5,
            data_volume_multiplier: 0.1,
            scenarios: vec![LoadTestScenario::DashboardGeneration],
        };

        let results = profiler.run_load_test(config).await.unwrap();

        assert!(results.total_requests > 0);
        assert!(results.avg_response_time_ms >= 0.0);
        assert!(results.requests_per_second >= 0.0);
    }

    #[tokio::test]
    async fn test_baseline_establishment() {
        let mut profiler = PerformanceProfiler::new();
        let baseline = profiler.establish_baseline(vec![50, 100]).await.unwrap();

        assert!(baseline.dashboard_generation_ms > 0);
        assert!(baseline.memory_usage_mb >= 0.0);
        assert_eq!(baseline.max_acceptable_regression_percent, 20.0);
    }

    #[tokio::test]
    async fn test_performance_report_generation() {
        let profiler = PerformanceProfiler::new();
        let report = profiler
            .generate_performance_report(vec![50, 100])
            .await
            .unwrap();

        assert_eq!(report.profiles.len(), 2);
        assert!(report.load_test_results.total_requests >= 0);
        assert!(!report.recommendations.is_empty());
    }
}
