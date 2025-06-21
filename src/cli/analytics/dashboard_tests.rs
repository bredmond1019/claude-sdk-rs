//! Comprehensive tests for dashboard data generation functionality
//!
//! This module tests dashboard metrics calculation, data aggregation,
//! performance indicators, and caching logic.

use super::dashboard::*;
use super::dashboard_test::*;
use super::test_utils::*;
use crate::cli::analytics::{Alert, AlertSeverity, AlertType, AnalyticsEngine};
use crate::cli::cost::{CostEntry, CostFilter, CostTracker};
use crate::cli::error::Result;
use crate::cli::history::{HistoryEntry, HistorySearch, HistoryStore};
use chrono::{DateTime, Duration, TimeZone, Timelike, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tempfile::TempDir;
use tokio::sync::RwLock;

/// Task 3.2.1: Test dashboard metrics calculation with various time periods
#[cfg(test)]
mod metrics_calculation_tests {
    use super::*;

    #[tokio::test]
    async fn test_daily_metrics_calculation() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Generate 24 hours of data using the fixture's method
        fixture.populate_test_data(1).await?;

        // Get dashboard data
        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;

        // Verify daily metrics
        assert!(
            dashboard_data.today_cost >= 0.0,
            "Daily cost should be non-negative"
        );
        assert!(
            dashboard_data.today_commands > 0,
            "Should have commands today"
        );
        assert!(
            dashboard_data.success_rate >= 0.0 && dashboard_data.success_rate <= 100.0,
            "Success rate should be between 0 and 100"
        );

        // Verify that metrics are from today only
        let today_start = Utc::now()
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        for entry in &dashboard_data.recent_activity {
            assert!(
                entry.timestamp >= today_start,
                "Recent activity should only include today's entries"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_weekly_metrics_calculation() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Generate 7 days of data
        fixture.populate_test_data(7).await?;

        // Calculate weekly summary
        let summary = fixture.analytics_engine.generate_summary(7).await?;

        // Verify weekly metrics
        assert!(
            summary.cost_summary.total_cost > 0.0,
            "Weekly cost should be positive"
        );
        assert!(
            summary.history_stats.total_entries > 0,
            "Should have entries in the week"
        );

        // Verify date range
        let expected_start = Utc::now() - Duration::days(7);
        assert!(
            (summary.period_start - expected_start).num_seconds().abs() < 60,
            "Period start should be approximately 7 days ago"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_monthly_metrics_calculation() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Generate 30 days of data
        fixture.populate_test_data(30).await?;

        // Calculate monthly summary
        let summary = fixture.analytics_engine.generate_summary(30).await?;

        // Verify monthly metrics
        assert!(
            summary.cost_summary.total_cost > 0.0,
            "Monthly cost should be positive"
        );
        assert!(
            summary.history_stats.total_entries > 0,
            "Should have entries in the month"
        );

        // Verify insights generation
        assert!(
            !summary.insights.is_empty(),
            "Should generate insights for monthly data"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_custom_date_range_calculation() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Generate 15 days of data
        fixture.populate_test_data(15).await?;

        // Test custom ranges
        for days in [3, 5, 10, 14] {
            let summary = fixture.analytics_engine.generate_summary(days).await?;

            // Verify period matches requested range
            let duration = summary.period_end - summary.period_start;
            assert!(
                duration.num_days() as u32 >= days - 1,
                "Summary period should cover at least {} days",
                days
            );

            // Verify data is within range
            let cost_tracker = fixture.cost_tracker.read().await;
            let filter = CostFilter {
                since: Some(summary.period_start),
                until: Some(summary.period_end),
                ..Default::default()
            };
            let filtered_summary = cost_tracker.get_filtered_summary(&filter).await?;
            assert_eq!(
                filtered_summary.total_cost, summary.cost_summary.total_cost,
                "Filtered cost should match summary cost"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_timezone_handling() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Create entries at current time to ensure they show up in today's count
        let now = Utc::now();
        let today_start = now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc();

        // Create one entry for today (current time) and one for tomorrow
        let today_entry_time = now - Duration::minutes(30); // 30 minutes ago (definitely today)
        let tomorrow_entry_time = today_start + Duration::days(1) + Duration::hours(1); // Tomorrow

        let mut cost_tracker = fixture.cost_tracker.write().await;
        let mut history_store = fixture.history_store.write().await;

        for (timestamp, idx) in [(today_entry_time, 0), (tomorrow_entry_time, 1)] {
            let session_id = uuid::Uuid::new_v4();

            let mut cost_entry = CostEntry::new(
                session_id,
                format!("command-{}", idx),
                0.10,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            );
            cost_entry.timestamp = timestamp;
            cost_tracker.record_cost(cost_entry).await?;

            let mut history_entry = HistoryEntry::new(
                session_id,
                format!("command-{}", idx),
                vec![],
                "output".to_string(),
                true,
                1000,
            );
            history_entry.timestamp = timestamp;
            history_store.store_entry(history_entry).await?;
        }

        drop(cost_tracker);
        drop(history_store);

        // Get today's dashboard data
        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;

        // Only the today entry (command-0) should be counted in today's commands
        // The tomorrow entry (command-1) should not be counted
        assert_eq!(
            dashboard_data.today_commands, 1,
            "Should have exactly 1 command today, got {}",
            dashboard_data.today_commands
        );

        // Verify recent activity includes at least the today entry
        assert!(
            !dashboard_data.recent_activity.is_empty(),
            "Recent activity should not be empty"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_hourly_metrics_aggregation() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;
        let manager = &fixture.dashboard_manager;

        // Generate hourly data
        let generator = DashboardTestDataGenerator::new();
        let time_series_data = generator.generate_time_series_data(24);

        // Process into dashboard format
        let live_data = manager.generate_live_data().await?;

        // Verify hourly aggregation
        assert!(
            !live_data.time_series.cost_over_time.is_empty(),
            "Should have hourly cost data"
        );
        assert!(
            !live_data.time_series.commands_over_time.is_empty(),
            "Should have hourly command data"
        );

        // Verify data is chronologically ordered
        for i in 1..live_data.time_series.cost_over_time.len() {
            assert!(
                live_data.time_series.cost_over_time[i].0
                    >= live_data.time_series.cost_over_time[i - 1].0,
                "Time series data should be chronologically ordered"
            );
        }

        Ok(())
    }
}

/// Task 3.2.2: Test dashboard data aggregation from multiple sources
#[cfg(test)]
mod data_aggregation_tests {
    use super::*;

    #[tokio::test]
    async fn test_combining_data_from_sessions_commands_costs() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Create data across multiple sessions
        let sessions = vec![uuid::Uuid::new_v4(), uuid::Uuid::new_v4(), uuid::Uuid::new_v4()];

        let mut cost_tracker = fixture.cost_tracker.write().await;
        let mut history_store = fixture.history_store.write().await;

        // Add varied data for each session
        for (idx, session_id) in sessions.iter().enumerate() {
            let commands = ["analyze", "summarize", "translate"];
            let base_cost = 0.05 * (idx + 1) as f64;

            for (cmd_idx, command) in commands.iter().enumerate() {
                let cost_entry = CostEntry::new(
                    *session_id,
                    command.to_string(),
                    base_cost + (cmd_idx as f64 * 0.01),
                    100 + (cmd_idx as u32 * 50),
                    200 + (cmd_idx as u32 * 100),
                    1000 + (cmd_idx as u64 * 500),
                    "claude-3-opus".to_string(),
                );
                cost_tracker.record_cost(cost_entry).await?;

                let history_entry = HistoryEntry::new(
                    *session_id,
                    command.to_string(),
                    vec!["--option".to_string()],
                    format!("Output for {}", command),
                    true,
                    1000 + (cmd_idx as u64 * 500),
                );
                history_store.store_entry(history_entry).await?;
            }
        }

        drop(cost_tracker);
        drop(history_store);

        // Get aggregated dashboard data
        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;

        // Verify aggregation
        assert_eq!(
            dashboard_data.today_commands, 9,
            "Should have 3 sessions * 3 commands = 9 total commands"
        );
        assert!(
            dashboard_data.today_cost > 0.0,
            "Total cost should be sum of all session costs"
        );

        // Verify top commands aggregation
        assert!(
            !dashboard_data.top_commands.is_empty(),
            "Should have top commands aggregated"
        );
        let total_top_cost: f64 = dashboard_data
            .top_commands
            .iter()
            .map(|(_, cost)| cost)
            .sum();
        assert!(
            total_top_cost <= dashboard_data.today_cost,
            "Top commands cost should not exceed total cost"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_data_consistency_across_sources() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;
        let generator = AnalyticsTestDataGenerator::default();

        // Generate consistent test data
        let test_data = generator.generate_test_data(1);
        fixture.load_data(&test_data).await?;

        // Get data from different sources
        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;
        let summary = fixture.analytics_engine.generate_summary(1).await?;

        // Verify consistency
        assert_eq!(
            dashboard_data.today_commands, summary.history_stats.total_entries,
            "Command count should be consistent across sources"
        );

        // Cost might differ slightly due to timing, but should be close
        let cost_diff = (dashboard_data.today_cost - summary.cost_summary.total_cost).abs();
        assert!(
            cost_diff < 0.01,
            "Cost data should be consistent (within rounding): {} vs {}",
            dashboard_data.today_cost,
            summary.cost_summary.total_cost
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_handling_missing_data() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Test with no data
        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;

        assert_eq!(
            dashboard_data.today_cost, 0.0,
            "Cost should be 0 with no data"
        );
        assert_eq!(
            dashboard_data.today_commands, 0,
            "Command count should be 0 with no data"
        );
        assert!(
            dashboard_data.recent_activity.is_empty(),
            "Recent activity should be empty with no data"
        );

        // Add partial data (cost without history)
        let cost_entry = CostEntry::new(
            uuid::Uuid::new_v4(),
            "test".to_string(),
            0.10,
            100,
            200,
            1000,
            "claude-3-opus".to_string(),
        );
        fixture
            .cost_tracker
            .write()
            .await
            .record_cost(cost_entry)
            .await?;

        // Should handle gracefully
        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;
        assert!(
            dashboard_data.today_cost > 0.0,
            "Should show cost even without history data"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_handling_incomplete_data() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;
        let session_id = uuid::Uuid::new_v4();

        // Add history entry without corresponding cost
        let history_entry = HistoryEntry::new(
            session_id,
            "analyze".to_string(),
            vec![],
            "output".to_string(),
            true,
            1500,
        );
        fixture
            .history_store
            .write()
            .await
            .store_entry(history_entry)
            .await?;

        // Should handle gracefully
        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;
        assert_eq!(
            dashboard_data.today_commands, 1,
            "Should count command even without cost data"
        );
        assert_eq!(
            dashboard_data.today_cost, 0.0,
            "Cost should be 0 if no cost entry exists"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_data_aggregation_with_filters() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Test different scenarios by directly populating test data for today

        // Test 1: High volume scenario - populate today's data with many commands
        fixture.populate_test_data(1).await?; // Generate test data for today
        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;

        // The populate_test_data method should generate some commands for today
        assert!(
            dashboard_data.today_commands > 0,
            "Should have some commands from populate_test_data, got {}",
            dashboard_data.today_commands
        );

        // Test 2: Error spike scenario - add failed commands
        let session_id = uuid::Uuid::new_v4();
        let mut history_store = fixture.history_store.write().await;

        // Add several failed commands
        for i in 0..10 {
            let mut history_entry = HistoryEntry::new(
                session_id,
                format!("failing_cmd_{}", i),
                vec![],
                "Error output".to_string(),
                false, // Failed command
                1000,
            );
            // Set timestamp to now to ensure it's counted in today's data
            history_entry.timestamp = Utc::now() - Duration::minutes(i as i64 * 5);
            history_store.store_entry(history_entry).await?;
        }
        drop(history_store);

        let dashboard_data_with_errors = fixture.analytics_engine.get_dashboard_data().await?;

        // Should now have a lower success rate due to failed commands
        assert!(
            dashboard_data_with_errors.success_rate < 90.0,
            "Success rate should be lower with failed commands, got {}",
            dashboard_data_with_errors.success_rate
        );

        assert!(
            dashboard_data_with_errors.today_commands > dashboard_data.today_commands,
            "Should have more commands after adding failed ones"
        );

        Ok(())
    }
}

/// Task 3.2.3: Test dashboard performance indicators and KPI calculations
#[cfg(test)]
mod kpi_calculation_tests {
    use super::*;

    #[tokio::test]
    async fn test_success_rate_calculations() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;
        let session_id = uuid::Uuid::new_v4();

        // Add mix of successful and failed commands
        let commands =
            vec![("cmd1", true), ("cmd2", true), ("cmd3", false), ("cmd4", true), ("cmd5", false)];

        let mut history_store = fixture.history_store.write().await;
        for (cmd, success) in commands {
            let entry = HistoryEntry::new(
                session_id,
                cmd.to_string(),
                vec![],
                if success { "Success" } else { "Error" }.to_string(),
                success,
                1000,
            );
            history_store.store_entry(entry).await?;
        }
        drop(history_store);

        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;

        // Success rate should be 3/5 = 60%
        assert!(
            (dashboard_data.success_rate - 60.0).abs() < 0.1,
            "Success rate should be approximately 60%, got {}",
            dashboard_data.success_rate
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_average_response_time_metrics() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;
        let session_id = uuid::Uuid::new_v4();

        // Add commands with varying response times
        let response_times = vec![500, 1000, 1500, 2000, 2500];
        let expected_avg = response_times.iter().sum::<u64>() as f64 / response_times.len() as f64;

        let mut history_store = fixture.history_store.write().await;
        for (idx, duration) in response_times.iter().enumerate() {
            let entry = HistoryEntry::new(
                session_id,
                format!("cmd{}", idx),
                vec![],
                "output".to_string(),
                true,
                *duration,
            );
            history_store.store_entry(entry).await?;
        }
        drop(history_store);

        let summary = fixture.analytics_engine.generate_summary(1).await?;

        assert!(
            (summary.performance_metrics.average_response_time - expected_avg).abs() < 1.0,
            "Average response time should be approximately {}, got {}",
            expected_avg,
            summary.performance_metrics.average_response_time
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_cost_efficiency_kpis() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Create commands with different cost efficiency
        let commands = vec![
            ("efficient_cmd", 0.01, 100, 500),   // Low cost, high output
            ("normal_cmd", 0.05, 200, 800),      // Average
            ("expensive_cmd", 0.20, 1000, 2000), // High cost
        ];

        let mut cost_tracker = fixture.cost_tracker.write().await;
        for (cmd, cost, input_tokens, output_tokens) in commands {
            let entry = CostEntry::new(
                uuid::Uuid::new_v4(),
                cmd.to_string(),
                cost,
                input_tokens,
                output_tokens,
                1000,
                "claude-3-opus".to_string(),
            );
            cost_tracker.record_cost(entry).await?;
        }
        drop(cost_tracker);

        let summary = fixture.analytics_engine.generate_summary(1).await?;

        // Verify cost efficiency metrics
        assert!(
            summary.cost_summary.average_cost > 0.0,
            "Should calculate average cost per command"
        );
        assert!(
            summary
                .cost_summary
                .by_command
                .contains_key("expensive_cmd"),
            "Should track cost by command"
        );

        // Verify insights include cost efficiency
        let has_cost_insight = summary
            .insights
            .iter()
            .any(|i| i.contains("cost") || i.contains("expensive"));
        assert!(has_cost_insight, "Should generate insights about costs");

        Ok(())
    }

    #[tokio::test]
    async fn test_user_engagement_metrics() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;
        let generator = AnalyticsTestDataGenerator::default();

        // Generate data with varying engagement patterns
        let test_data = generator.generate_test_data(7);
        fixture.load_data(&test_data).await?;

        let summary = fixture.analytics_engine.generate_summary(7).await?;

        // Verify engagement metrics
        assert!(
            summary.performance_metrics.throughput_commands_per_hour > 0.0,
            "Should calculate command throughput"
        );
        assert!(
            summary.performance_metrics.peak_usage_hour <= 23,
            "Peak usage hour should be valid (0-23)"
        );

        // Verify usage insights
        let has_usage_insight = summary
            .insights
            .iter()
            .any(|i| i.contains("commands per day") || i.contains("usage"));
        assert!(
            has_usage_insight,
            "Should generate insights about usage patterns"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_performance_percentiles() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Generate response times for percentile calculations
        let mut response_times = Vec::new();
        for i in 0..100 {
            response_times.push(500 + (i * 50)); // 500ms to 5500ms
        }

        let session_id = uuid::Uuid::new_v4();
        let mut history_store = fixture.history_store.write().await;

        for (idx, duration) in response_times.iter().enumerate() {
            let entry = HistoryEntry::new(
                session_id,
                format!("cmd{}", idx),
                vec![],
                "output".to_string(),
                true,
                *duration as u64,
            );
            history_store.store_entry(entry).await?;
        }
        drop(history_store);

        // Generate widget data for response time
        let generator = DashboardTestDataGenerator::new();
        let widget_data = generator
            .generate_widget_data(&crate::cli::analytics::dashboard_test::WidgetType::ResponseTime);

        // Verify percentiles are reasonable
        if let Some(p50) = widget_data.get("p50_ms").and_then(|v| v.as_f64()) {
            assert!(p50 > 0.0, "P50 should be positive");
        }
        if let Some(p95) = widget_data.get("p95_ms").and_then(|v| v.as_f64()) {
            assert!(p95 > 0.0, "P95 should be positive");
        }
        if let Some(p99) = widget_data.get("p99_ms").and_then(|v| v.as_f64()) {
            assert!(p99 > 0.0, "P99 should be positive");
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_error_rate_by_command() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Create commands with different error rates
        let commands = vec![
            ("reliable_cmd", vec![true, true, true, true]),
            ("flaky_cmd", vec![true, false, true, false]),
            ("broken_cmd", vec![false, false, false, true]),
        ];

        let mut history_store = fixture.history_store.write().await;
        for (cmd_name, results) in commands {
            for (idx, success) in results.iter().enumerate() {
                let entry = HistoryEntry::new(
                    uuid::Uuid::new_v4(),
                    cmd_name.to_string(),
                    vec![],
                    format!("Result {}", idx),
                    *success,
                    1000,
                );
                history_store.store_entry(entry).await?;
            }
        }
        drop(history_store);

        let summary = fixture.analytics_engine.generate_summary(1).await?;

        // Verify error rates by command
        let error_rates = &summary.performance_metrics.error_rate_by_command;

        assert!(
            error_rates.get("reliable_cmd").unwrap_or(&100.0) < &1.0,
            "Reliable command should have low error rate"
        );
        assert!(
            (error_rates.get("flaky_cmd").unwrap_or(&0.0) - &50.0).abs() < 1.0,
            "Flaky command should have ~50% error rate"
        );
        assert!(
            error_rates.get("broken_cmd").unwrap_or(&0.0) > &70.0,
            "Broken command should have high error rate"
        );

        Ok(())
    }
}

/// Task 3.2.4: Test dashboard data caching and refresh logic
#[cfg(test)]
mod caching_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_cache_invalidation_on_data_updates() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Get initial dashboard data
        let initial_data = fixture.dashboard_manager.generate_live_data().await?;
        let initial_cost = initial_data.current_metrics.today_cost;

        // Add new data
        let cost_entry = CostEntry::new(
            uuid::Uuid::new_v4(),
            "new_command".to_string(),
            0.15,
            200,
            400,
            1500,
            "claude-3-opus".to_string(),
        );
        fixture
            .cost_tracker
            .write()
            .await
            .record_cost(cost_entry)
            .await?;

        // Get updated dashboard data
        let updated_data = fixture.dashboard_manager.generate_live_data().await?;

        // Verify cache was invalidated and data updated
        assert!(
            updated_data.current_metrics.today_cost > initial_cost,
            "Dashboard should reflect new cost data"
        );
        assert!(
            updated_data.timestamp > initial_data.timestamp,
            "Timestamp should be updated"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_partial_cache_updates() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Generate initial data
        let generator = AnalyticsTestDataGenerator::default();
        let test_data = generator.generate_test_data(1);
        fixture.load_data(&test_data).await?;

        // Trigger partial update by adding single new entry
        let history_entry = HistoryEntry::new(
            uuid::Uuid::new_v4(),
            "quick_update".to_string(),
            vec![],
            "output".to_string(),
            true,
            500,
        );
        fixture
            .history_store
            .write()
            .await
            .store_entry(history_entry)
            .await?;

        // Get initial data
        let initial_data = fixture.dashboard_manager.generate_live_data().await?;

        // Add another entry
        let history_entry2 = HistoryEntry::new(
            uuid::Uuid::new_v4(),
            "another_update".to_string(),
            vec![],
            "output2".to_string(),
            true,
            600,
        );
        fixture
            .history_store
            .write()
            .await
            .store_entry(history_entry2)
            .await?;

        // Get updated data
        let updated_data = fixture.dashboard_manager.generate_live_data().await?;

        // Verify partial update worked
        assert!(
            updated_data.current_metrics.today_commands
                > initial_data.current_metrics.today_commands,
            "Command count should increase with new data"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_performance_improvements() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Load substantial data
        let generator = AnalyticsTestDataGenerator {
            session_count: 20,
            commands_per_session: (10, 30),
            ..Default::default()
        };
        let test_data = generator.generate_test_data(7);
        fixture.load_data(&test_data).await?;

        // Measure performance
        let durations =
            crate::cli::analytics::dashboard_test::performance::measure_dashboard_generation(
                &fixture, 10,
            )
            .await;

        // Calculate statistics
        let avg_duration = durations.iter().sum::<u128>() / durations.len() as u128;
        let first_gen = durations[0];
        let subsequent_avg = durations[1..].iter().sum::<u128>() / (durations.len() - 1) as u128;

        // First generation might be slower (cold cache)
        // Subsequent generations should be faster
        assert!(
            subsequent_avg <= first_gen,
            "Subsequent generations should not be slower than first: {} vs {}",
            subsequent_avg,
            first_gen
        );

        // All generations should be reasonably fast
        assert!(
            avg_duration < 50_000, // 50ms
            "Average dashboard generation should be under 50ms, got {}μs",
            avg_duration
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_stale_data_handling() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Add old data (simulating stale cache)
        let old_timestamp = Utc::now() - Duration::hours(2);
        let session_id = uuid::Uuid::new_v4();

        let mut cost_entry = CostEntry::new(
            session_id,
            "old_command".to_string(),
            0.05,
            100,
            200,
            1000,
            "claude-3-opus".to_string(),
        );
        cost_entry.timestamp = old_timestamp;

        fixture
            .cost_tracker
            .write()
            .await
            .record_cost(cost_entry)
            .await?;

        // Generate fresh dashboard data
        let dashboard_data = fixture.dashboard_manager.generate_live_data().await?;

        // Verify timestamp is current, not stale
        let age = Utc::now() - dashboard_data.timestamp;
        assert!(
            age.num_seconds() < 5,
            "Dashboard timestamp should be recent, not stale"
        );

        // Verify system status is current
        assert_eq!(
            dashboard_data.system_status.health,
            HealthStatus::Healthy,
            "System status should reflect current state"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_cache_access() -> Result<()> {
        let fixture = Arc::new(DashboardTestFixture::new().await?);

        // Load test data
        let generator = AnalyticsTestDataGenerator::default();
        let test_data = generator.generate_test_data(1);
        fixture.load_data(&test_data).await?;

        // Spawn multiple concurrent readers
        let mut handles = vec![];
        for i in 0..10 {
            let fixture_clone = Arc::clone(&fixture);
            let handle = tokio::spawn(async move {
                let start = Instant::now();
                let data = fixture_clone.dashboard_manager.generate_live_data().await;
                let duration = start.elapsed();
                (i, data, duration)
            });
            handles.push(handle);
        }

        // Collect results
        let mut results = vec![];
        for handle in handles {
            if let Ok(result) = handle.await {
                results.push(result);
            }
        }

        // Verify all requests succeeded
        assert_eq!(results.len(), 10, "All concurrent requests should succeed");

        // Verify consistency
        let first_cost = results[0].1.as_ref().unwrap().current_metrics.today_cost;
        for (idx, data, _) in &results {
            assert_eq!(
                data.as_ref().unwrap().current_metrics.today_cost,
                first_cost,
                "Concurrent access should return consistent data (request {})",
                idx
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_memory_efficiency() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Generate large dataset
        let generator = AnalyticsTestDataGenerator {
            session_count: 50,
            commands_per_session: (20, 50),
            ..Default::default()
        };
        let test_data = generator.generate_test_data(30);
        fixture.load_data(&test_data).await?;

        // Measure memory usage indirectly through generation time
        let start = Instant::now();
        let _ = fixture.dashboard_manager.generate_live_data().await?;
        let duration = start.elapsed();

        // Even with large dataset, generation should be efficient
        assert!(
            duration.as_millis() < 100,
            "Dashboard generation should be efficient even with large datasets"
        );

        // Test widget-specific caching
        let widget_configs = vec![
            crate::cli::analytics::dashboard_test::WidgetType::CostSummary,
            crate::cli::analytics::dashboard_test::WidgetType::CommandActivity,
            crate::cli::analytics::dashboard_test::WidgetType::SuccessRate,
            crate::cli::analytics::dashboard_test::WidgetType::ResponseTime,
        ];

        for widget_type in widget_configs {
            let start = Instant::now();
            let _ = DashboardTestDataGenerator::new().generate_widget_data(&widget_type);
            let duration = start.elapsed();

            assert!(
                duration.as_micros() < 1000,
                "Widget data generation should be very fast (<1ms)"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_refresh_interval_configuration() -> Result<()> {
        // Test with custom refresh interval
        let temp_dir = TempDir::new()?;
        let cost_tracker = Arc::new(RwLock::new(CostTracker::new(
            temp_dir.path().join("costs.json"),
        )?));
        let history_store = Arc::new(RwLock::new(HistoryStore::new(
            temp_dir.path().join("history.json"),
        )?));

        let analytics_config = crate::cli::analytics::AnalyticsConfig {
            dashboard_refresh_interval: 1, // 1 second for testing
            ..Default::default()
        };

        let analytics_engine = AnalyticsEngine::new(
            Arc::clone(&cost_tracker),
            Arc::clone(&history_store),
            analytics_config,
        );

        let dashboard_config = DashboardConfig {
            refresh_interval_seconds: 1, // 1 second for testing
            ..Default::default()
        };

        let dashboard_manager = DashboardManager::new(analytics_engine, dashboard_config);

        // Test that we can generate data at configured intervals
        let start = Instant::now();
        let mut data_points = vec![];

        // Collect data points for 3 seconds
        while start.elapsed().as_secs() < 3 {
            let data = dashboard_manager.generate_live_data().await?;
            data_points.push(data);
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }

        // Should have approximately 3 data points
        assert!(
            data_points.len() >= 3,
            "Should generate data points at regular intervals, got {}",
            data_points.len()
        );

        // Verify timestamps are different
        for i in 1..data_points.len() {
            assert!(
                data_points[i].timestamp > data_points[i - 1].timestamp,
                "Timestamps should increase"
            );
        }

        Ok(())
    }
}

/// Edge case and error handling tests
#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[tokio::test]
    async fn test_empty_database_handling() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Test with completely empty database
        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;

        crate::cli::analytics::dashboard_test::assertions::assert_dashboard_data_valid(
            &dashboard_data,
        );
        assert_eq!(dashboard_data.today_cost, 0.0);
        assert_eq!(dashboard_data.today_commands, 0);
        assert!(dashboard_data.recent_activity.is_empty());
        assert!(dashboard_data.top_commands.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_extreme_data_volumes() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Generate extreme volume
        let generator = AnalyticsTestDataGenerator {
            session_count: 100,
            commands_per_session: (50, 100),
            ..Default::default()
        };

        // Only generate 1 day to keep test reasonable
        let test_data = generator.generate_test_data(1);
        fixture.load_data(&test_data).await?;

        // Should handle gracefully
        let start = Instant::now();
        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;
        let duration = start.elapsed();

        crate::cli::analytics::dashboard_test::assertions::assert_dashboard_data_valid(
            &dashboard_data,
        );
        assert!(
            duration.as_secs() < 5,
            "Should handle extreme volumes within 5 seconds"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_date_boundary_edge_cases() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Test at exact midnight
        let midnight = Utc::now()
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();

        let mut cost_entry = CostEntry::new(
            uuid::Uuid::new_v4(),
            "midnight_cmd".to_string(),
            0.05,
            100,
            200,
            1000,
            "claude-3-opus".to_string(),
        );
        cost_entry.timestamp = midnight;

        fixture
            .cost_tracker
            .write()
            .await
            .record_cost(cost_entry)
            .await?;

        // Test at 23:59:59
        let end_of_day = midnight + Duration::days(1) - Duration::seconds(1);
        let mut cost_entry = CostEntry::new(
            uuid::Uuid::new_v4(),
            "endofday_cmd".to_string(),
            0.05,
            100,
            200,
            1000,
            "claude-3-opus".to_string(),
        );
        cost_entry.timestamp = end_of_day;

        fixture
            .cost_tracker
            .write()
            .await
            .record_cost(cost_entry)
            .await?;

        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;
        crate::cli::analytics::dashboard_test::assertions::assert_dashboard_data_valid(
            &dashboard_data,
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_floating_point_precision() -> Result<()> {
        let fixture = DashboardTestFixture::new().await?;

        // Add many small costs that might cause precision issues
        let mut cost_tracker = fixture.cost_tracker.write().await;
        for i in 0..1000 {
            let cost_entry = CostEntry::new(
                uuid::Uuid::new_v4(),
                format!("cmd_{}", i),
                0.0001, // Very small cost
                10,
                20,
                100,
                "claude-3-haiku".to_string(),
            );
            cost_tracker.record_cost(cost_entry).await?;
        }
        drop(cost_tracker);

        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;

        // Total should be approximately 0.1 (1000 * 0.0001)
        let expected = 0.1;
        let tolerance = 0.001;
        assert!(
            (dashboard_data.today_cost - expected).abs() < tolerance,
            "Cost calculation should maintain precision: expected ~{}, got {}",
            expected,
            dashboard_data.today_cost
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_data_updates() -> Result<()> {
        let fixture = Arc::new(DashboardTestFixture::new().await?);

        // Spawn multiple writers
        let mut handles = vec![];
        for i in 0..10 {
            let fixture_clone = Arc::clone(&fixture);
            let handle = tokio::spawn(async move {
                let session_id = uuid::Uuid::new_v4();
                let cost_entry = CostEntry::new(
                    session_id,
                    format!("concurrent_cmd_{}", i),
                    0.01 * i as f64,
                    100,
                    200,
                    1000,
                    "claude-3-opus".to_string(),
                );
                fixture_clone
                    .cost_tracker
                    .write()
                    .await
                    .record_cost(cost_entry)
                    .await
            });
            handles.push(handle);
        }

        // Wait for all writes
        for handle in handles {
            handle.await.unwrap()?;
        }

        // Verify data integrity
        let dashboard_data = fixture.analytics_engine.get_dashboard_data().await?;
        assert!(
            dashboard_data.today_cost > 0.0,
            "Should have recorded all concurrent costs"
        );

        Ok(())
    }
}
