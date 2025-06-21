//! Performance baseline establishment and regression testing
//!
//! This module provides tools for establishing performance baselines and
//! detecting performance regressions in analytics operations.

use super::dashboard_tests::DashboardTestFixture;
use super::performance_profiler::*;
use super::test_utils::AnalyticsTestDataGenerator;
use crate::cli::error::Result;
use std::time::Instant;
use tokio;

/// Comprehensive performance baseline test suite
#[cfg(test)]
mod baseline_tests {
    use super::*;

    #[tokio::test]
    async fn test_establish_performance_baseline() -> Result<()> {
        let mut profiler = PerformanceProfiler::new();

        // Test with multiple data volumes to establish scaling characteristics
        let data_volumes = vec![100, 500, 1000, 2000];
        let baseline = profiler.establish_baseline(data_volumes).await?;

        // Verify baseline values are reasonable
        assert!(
            baseline.dashboard_generation_ms > 0,
            "Dashboard generation should take some time"
        );
        assert!(
            baseline.dashboard_generation_ms < 10000,
            "Dashboard generation should be under 10 seconds"
        );
        assert!(
            baseline.time_series_generation_ms > 0,
            "Time series generation should take some time"
        );
        assert!(baseline.memory_usage_mb > 0.0, "Should use some memory");
        assert!(
            baseline.memory_usage_mb < 1000.0,
            "Should not use excessive memory"
        );

        println!("Established baseline:");
        println!(
            "  Dashboard generation: {}ms",
            baseline.dashboard_generation_ms
        );
        println!(
            "  Time series generation: {}ms",
            baseline.time_series_generation_ms
        );
        println!("  Memory usage: {:.1}MB", baseline.memory_usage_mb);

        Ok(())
    }

    #[tokio::test]
    async fn test_performance_scaling_characteristics() -> Result<()> {
        let profiler = PerformanceProfiler::new();

        // Test performance across different data volumes
        let test_volumes = vec![100, 500, 1000, 2000, 5000];
        let mut profiles = Vec::new();

        for volume in test_volumes {
            let profile = profiler.profile_dashboard_generation(volume).await?;

            println!(
                "Volume {}: {}ms, {:.1}MB",
                volume, profile.total_duration_ms, profile.peak_memory_mb
            );

            profiles.push((volume, profile));
        }

        // Analyze scaling characteristics
        let first_profile = &profiles[0].1;
        let last_profile = &profiles[profiles.len() - 1].1;

        let time_scaling_factor =
            last_profile.total_duration_ms as f64 / first_profile.total_duration_ms as f64;
        let data_scaling_factor = profiles[profiles.len() - 1].0 as f64 / profiles[0].0 as f64;
        let memory_scaling_factor = last_profile.peak_memory_mb / first_profile.peak_memory_mb;

        println!("Scaling analysis:");
        println!("  Data scale factor: {:.2}x", data_scaling_factor);
        println!("  Time scale factor: {:.2}x", time_scaling_factor);
        println!("  Memory scale factor: {:.2}x", memory_scaling_factor);

        // Performance should scale sub-linearly or linearly with data volume
        assert!(
            time_scaling_factor <= data_scaling_factor * 2.0,
            "Performance should not degrade worse than 2x linear scaling"
        );

        // Memory should scale reasonably with data volume
        assert!(
            memory_scaling_factor <= data_scaling_factor * 1.5,
            "Memory usage should not grow worse than 1.5x linear scaling"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_operation_breakdown_profiling() -> Result<()> {
        let profiler = PerformanceProfiler::new();
        let profile = profiler.profile_dashboard_generation(1000).await?;

        println!("Operation breakdown:");
        for (operation, duration_ms) in &profile.operation_timings {
            let percentage = (*duration_ms as f64 / profile.total_duration_ms as f64) * 100.0;
            println!("  {}: {}ms ({:.1}%)", operation, duration_ms, percentage);
        }

        // Verify we have timing data for key operations
        assert!(
            profile
                .operation_timings
                .contains_key("dashboard_data_generation"),
            "Should have timing for dashboard data generation"
        );
        assert!(
            profile.operation_timings.contains_key("time_series_cost"),
            "Should have timing for time series generation"
        );
        assert!(
            profile.operation_timings.contains_key("analytics_summary"),
            "Should have timing for analytics summary"
        );

        // Verify total time is sum of parts (approximately)
        let sum_of_parts: u64 = profile.operation_timings.values().sum();
        let total_time = profile.total_duration_ms;

        // Allow for some overhead (total can be larger than sum of parts)
        assert!(
            total_time >= sum_of_parts,
            "Total time should be at least the sum of measured operations"
        );
        assert!(
            total_time <= sum_of_parts * 2,
            "Total time should not be more than 2x the sum of operations (accounting for overhead)"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_load_testing_scenarios() -> Result<()> {
        let profiler = PerformanceProfiler::new();

        // Test different load scenarios
        let scenarios = vec![
            (
                "light_load",
                LoadTestConfig {
                    concurrent_users: 2,
                    duration_seconds: 10,
                    requests_per_second: 0.5,
                    data_volume_multiplier: 0.5,
                    scenarios: vec![LoadTestScenario::DashboardGeneration],
                },
            ),
            (
                "medium_load",
                LoadTestConfig {
                    concurrent_users: 5,
                    duration_seconds: 15,
                    requests_per_second: 1.0,
                    data_volume_multiplier: 1.0,
                    scenarios: vec![
                        LoadTestScenario::DashboardGeneration,
                        LoadTestScenario::TimeSeriesGeneration,
                    ],
                },
            ),
            (
                "mixed_workload",
                LoadTestConfig {
                    concurrent_users: 3,
                    duration_seconds: 10,
                    requests_per_second: 1.0,
                    data_volume_multiplier: 1.0,
                    scenarios: vec![LoadTestScenario::MixedWorkload],
                },
            ),
        ];

        for (scenario_name, config) in scenarios {
            println!("\nRunning {} scenario...", scenario_name);
            let start_time = Instant::now();

            let results = profiler.run_load_test(config).await?;

            let test_duration = start_time.elapsed();

            println!("  Total requests: {}", results.total_requests);
            println!(
                "  Success rate: {:.1}%",
                (results.successful_requests as f64 / results.total_requests as f64) * 100.0
            );
            println!("  Avg response time: {:.1}ms", results.avg_response_time_ms);
            println!("  P95 response time: {:.1}ms", results.p95_response_time_ms);
            println!("  Requests/sec: {:.1}", results.requests_per_second);
            println!("  Peak memory: {:.1}MB", results.peak_memory_mb);
            println!("  Test duration: {:.1}s", test_duration.as_secs_f64());

            // Verify basic load test invariants
            assert!(results.total_requests > 0, "Should have made some requests");
            assert!(
                results.successful_requests > 0,
                "Should have some successful requests"
            );
            assert!(
                results.avg_response_time_ms > 0.0,
                "Should have measurable response times"
            );
            assert!(
                results.error_rate_percent < 50.0,
                "Error rate should be reasonable"
            );

            // Performance expectations
            assert!(
                results.avg_response_time_ms < 5000.0,
                "Average response time should be under 5 seconds"
            );
            assert!(
                results.p95_response_time_ms < 10000.0,
                "P95 response time should be under 10 seconds"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_usage_characteristics() -> Result<()> {
        let profiler = PerformanceProfiler::new();

        // Test memory usage with different data volumes
        let volumes = vec![100, 500, 1000, 2000];
        let mut memory_profiles = Vec::new();

        for volume in volumes {
            let profile = profiler.profile_dashboard_generation(volume).await?;
            memory_profiles.push((volume, profile.peak_memory_mb));

            println!(
                "Volume {}: {:.1}MB peak memory",
                volume, profile.peak_memory_mb
            );
        }

        // Analyze memory scaling
        let first_memory = memory_profiles[0].1;
        let last_memory = memory_profiles[memory_profiles.len() - 1].1;
        let memory_growth_factor = last_memory / first_memory;
        let data_growth_factor =
            memory_profiles[memory_profiles.len() - 1].0 as f64 / memory_profiles[0].0 as f64;

        println!("Memory scaling:");
        println!("  Data grew by: {:.2}x", data_growth_factor);
        println!("  Memory grew by: {:.2}x", memory_growth_factor);

        // Memory should not grow worse than linearly with data
        assert!(
            memory_growth_factor <= data_growth_factor * 1.5,
            "Memory should scale reasonably with data volume"
        );

        // Should not use excessive memory even with large datasets
        assert!(
            last_memory < 500.0,
            "Should not use more than 500MB even with large datasets"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_regression_detection() -> Result<()> {
        let mut profiler = PerformanceProfiler::new();

        // Establish baseline
        let baseline = profiler.establish_baseline(vec![500, 1000]).await?;

        // Test with same volume - should not detect regression
        let regression_check = profiler.check_regression(750).await?;

        if let Some(regression) = regression_check {
            println!("Regression detected:");
            println!(
                "  Dashboard: {:.1}% slower",
                regression.dashboard_generation_regression_percent
            );
            println!(
                "  Memory: {:.1}% more",
                regression.memory_usage_regression_percent
            );

            // If regression is detected, it should be within reasonable bounds for test variability
            assert!(
                regression.dashboard_generation_regression_percent < 50.0,
                "Regression should not be extreme for same volume test"
            );
        } else {
            println!("No regression detected (expected for baseline test)");
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_performance_report_generation() -> Result<()> {
        let profiler = PerformanceProfiler::new();

        let report = profiler
            .generate_performance_report(vec![200, 500, 1000])
            .await?;

        println!("Performance Report Generated:");
        println!("  Profiles: {}", report.profiles.len());
        println!(
            "  Load test requests: {}",
            report.load_test_results.total_requests
        );
        println!("  Recommendations: {}", report.recommendations.len());

        // Verify report structure
        assert_eq!(
            report.profiles.len(),
            3,
            "Should have 3 profiles for 3 volumes"
        );
        assert!(
            report.load_test_results.total_requests > 0,
            "Should have load test data"
        );
        assert!(
            !report.recommendations.is_empty(),
            "Should have recommendations"
        );

        // Print recommendations
        println!("Recommendations:");
        for (i, rec) in report.recommendations.iter().enumerate() {
            println!("  {}: {}", i + 1, rec);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_profiling() -> Result<()> {
        let profiler = std::sync::Arc::new(PerformanceProfiler::new());

        // Run multiple profiles concurrently
        let mut handles = Vec::new();
        for i in 0..5 {
            let profiler_clone = std::sync::Arc::clone(&profiler);
            let handle = tokio::spawn(async move {
                let volume = 200 + (i * 100);
                (i, profiler_clone.profile_dashboard_generation(volume).await)
            });
            handles.push(handle);
        }

        // Collect results
        let mut results = Vec::new();
        for handle in handles {
            if let Ok((id, result)) = handle.await {
                if let Ok(profile) = result {
                    results.push((id, profile));
                }
            }
        }

        assert_eq!(results.len(), 5, "All concurrent profiles should succeed");

        // Verify each profile has reasonable results
        for (id, profile) in results {
            println!(
                "Concurrent profile {}: {}ms, {:.1}MB",
                id, profile.total_duration_ms, profile.peak_memory_mb
            );

            assert!(
                profile.total_duration_ms > 0,
                "Profile {} should have measurable duration",
                id
            );
            assert!(
                profile.peak_memory_mb > 0.0,
                "Profile {} should have measurable memory usage",
                id
            );
        }

        Ok(())
    }
}

/// Real-world performance testing scenarios
#[cfg(test)]
mod realistic_scenarios {
    use super::*;

    #[tokio::test]
    async fn test_typical_daily_usage_scenario() -> Result<()> {
        let profiler = PerformanceProfiler::new();

        // Simulate typical daily usage: moderate sessions, recent data
        let profile = profiler.profile_dashboard_generation(800).await?;

        println!("Typical daily usage scenario:");
        println!("  Total time: {}ms", profile.total_duration_ms);
        println!("  Memory usage: {:.1}MB", profile.peak_memory_mb);
        println!("  Data points: {}", profile.data_points_processed);

        // Performance expectations for typical usage
        assert!(
            profile.total_duration_ms < 2000,
            "Typical usage should be under 2 seconds, got {}ms",
            profile.total_duration_ms
        );
        assert!(
            profile.peak_memory_mb < 100.0,
            "Typical usage should use under 100MB, got {:.1}MB",
            profile.peak_memory_mb
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_heavy_usage_scenario() -> Result<()> {
        let profiler = PerformanceProfiler::new();

        // Simulate heavy usage: many sessions, lots of historical data
        let profile = profiler.profile_dashboard_generation(3000).await?;

        println!("Heavy usage scenario:");
        println!("  Total time: {}ms", profile.total_duration_ms);
        println!("  Memory usage: {:.1}MB", profile.peak_memory_mb);
        println!("  Data points: {}", profile.data_points_processed);

        // Performance expectations for heavy usage
        assert!(
            profile.total_duration_ms < 10000,
            "Heavy usage should be under 10 seconds, got {}ms",
            profile.total_duration_ms
        );
        assert!(
            profile.peak_memory_mb < 500.0,
            "Heavy usage should use under 500MB, got {:.1}MB",
            profile.peak_memory_mb
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_enterprise_load_scenario() -> Result<()> {
        let profiler = PerformanceProfiler::new();

        // Simulate enterprise-level concurrent usage
        let config = LoadTestConfig {
            concurrent_users: 20,
            duration_seconds: 30,
            requests_per_second: 2.0,
            data_volume_multiplier: 2.0,
            scenarios: vec![
                LoadTestScenario::DashboardGeneration,
                LoadTestScenario::TimeSeriesGeneration,
                LoadTestScenario::MixedWorkload,
            ],
        };

        let results = profiler.run_load_test(config).await?;

        println!("Enterprise load scenario:");
        println!("  Total requests: {}", results.total_requests);
        println!(
            "  Success rate: {:.1}%",
            (results.successful_requests as f64 / results.total_requests as f64) * 100.0
        );
        println!("  Avg response time: {:.1}ms", results.avg_response_time_ms);
        println!("  P95 response time: {:.1}ms", results.p95_response_time_ms);
        println!("  P99 response time: {:.1}ms", results.p99_response_time_ms);
        println!("  Throughput: {:.1} req/s", results.requests_per_second);
        println!("  Peak memory: {:.1}MB", results.peak_memory_mb);
        println!("  Error rate: {:.1}%", results.error_rate_percent);

        // Enterprise performance expectations
        assert!(
            results.error_rate_percent < 5.0,
            "Enterprise scenario should have low error rate, got {:.1}%",
            results.error_rate_percent
        );
        assert!(
            results.p95_response_time_ms < 5000.0,
            "P95 response time should be under 5s for enterprise load, got {:.1}ms",
            results.p95_response_time_ms
        );
        assert!(
            results.requests_per_second > 10.0,
            "Should maintain good throughput under enterprise load, got {:.1} req/s",
            results.requests_per_second
        );

        Ok(())
    }
}
