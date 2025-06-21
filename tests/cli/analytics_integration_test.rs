//! Comprehensive integration tests for the analytics system
//!
//! These tests verify:
//! - End-to-end data flow from cost tracking to analytics display
//! - Real-time updates and streaming functionality
//! - Performance under load
//! - Data consistency across all modules
//! - Error handling and recovery

use chrono::{Duration, Utc};
use claude_sdk_rs_interactive::{
    analytics::{
        AlertSeverity, AlertType, AnalyticsConfig, AnalyticsEngine, DashboardConfig,
        DashboardManager, HealthStatus, MetricsEngine, RealTimeAnalyticsStream, ReportFormat,
        UpdateType,
    },
    cost::tracker::{AdvancedCostTracker, Budget, BudgetScope, CostAlert},
    cost::{CostEntry, CostFilter, CostTracker},
    history::{HistoryEntry, HistorySearch, HistoryStore},
    session::SessionId,
    Result,
};
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Test harness for analytics integration tests
struct AnalyticsTestHarness {
    temp_dir: TempDir,
    cost_tracker: Arc<RwLock<CostTracker>>,
    history_store: Arc<RwLock<HistoryStore>>,
    analytics_engine: Arc<AnalyticsEngine>,
    dashboard_manager: Arc<DashboardManager>,
    metrics_engine: Arc<MetricsEngine>,
    realtime_stream: Arc<RealTimeAnalyticsStream>,
}

impl AnalyticsTestHarness {
    async fn new() -> Result<Self> {
        let temp_dir = tempfile::tempdir()?;
        let base_path = temp_dir.path();

        // Initialize core components
        let cost_tracker = Arc::new(RwLock::new(CostTracker::new(base_path.join("costs.json"))?));

        let history_store = Arc::new(RwLock::new(HistoryStore::new(
            base_path.join("history.json"),
        )?));

        // Create analytics engine
        let analytics_config = AnalyticsConfig {
            enable_real_time_alerts: true,
            cost_alert_threshold: 5.0,
            report_schedule: claude_ai_interactive::analytics::ReportSchedule::Daily,
            retention_days: 90,
            dashboard_refresh_interval: 1, // Fast refresh for testing
        };

        let analytics_engine = Arc::new(AnalyticsEngine::new(
            Arc::clone(&cost_tracker),
            Arc::clone(&history_store),
            analytics_config,
        ));

        // Create dashboard manager
        let dashboard_config = DashboardConfig {
            refresh_interval_seconds: 1,
            max_recent_entries: 50,
            enable_live_updates: true,
            chart_time_range_hours: 24,
            enable_real_system_monitoring: true,
        };

        let dashboard_manager = Arc::new(DashboardManager::new(
            (*analytics_engine).clone(),
            dashboard_config,
        ));

        // Create metrics engine
        let metrics_engine = Arc::new(MetricsEngine::new(
            claude_ai_interactive::analytics::MetricConfig::default(),
        ));

        // Create real-time stream
        let realtime_stream =
            Arc::new(RealTimeAnalyticsStream::new(Arc::clone(&analytics_engine)).await?);

        Ok(Self {
            temp_dir,
            cost_tracker,
            history_store,
            analytics_engine,
            dashboard_manager,
            metrics_engine,
            realtime_stream,
        })
    }

    /// Simulate realistic daily usage patterns
    async fn simulate_daily_usage(&self, session_id: SessionId, day_offset: i64) -> Result<()> {
        let base_time = Utc::now() - Duration::days(day_offset);
        let commands = vec![
            ("analyze", 0.05, 150, 300, 2000),
            ("generate", 0.10, 200, 400, 3000),
            ("chat", 0.02, 50, 100, 1000),
            ("search", 0.01, 30, 60, 500),
            ("summarize", 0.08, 180, 360, 2500),
        ];

        // Simulate usage throughout the day with peak hours
        for hour in 0..24 {
            let usage_multiplier = match hour {
                9..=11 | 14..=16 => 3.0,          // Peak hours
                7..=8 | 12..=13 | 17..=18 => 2.0, // Medium usage
                19..=22 => 1.5,                   // Evening usage
                _ => 0.5,                         // Low usage
            };

            let commands_this_hour = (2.0 * usage_multiplier) as usize;

            for _ in 0..commands_this_hour {
                let (cmd_name, base_cost, input_tokens, output_tokens, duration_ms) =
                    commands[hour % commands.len()];

                let timestamp = base_time
                    .date_naive()
                    .and_hms_opt(
                        hour as u32,
                        rand::random::<u32>() % 60,
                        rand::random::<u32>() % 60,
                    )
                    .unwrap()
                    .and_utc();

                // Record cost entry
                let mut cost_entry = CostEntry::new(
                    session_id,
                    cmd_name.to_string(),
                    base_cost * usage_multiplier,
                    (input_tokens as f64 * usage_multiplier) as u32,
                    (output_tokens as f64 * usage_multiplier) as u32,
                    duration_ms,
                    "claude-3-opus".to_string(),
                );
                cost_entry.timestamp = timestamp;

                self.cost_tracker
                    .write()
                    .await
                    .record_cost(cost_entry.clone())
                    .await?;

                // Record history entry
                let mut history_entry = HistoryEntry::new(
                    session_id,
                    cmd_name.to_string(),
                    vec![format!("--arg{}", hour)],
                    format!("Output for {} at hour {}", cmd_name, hour),
                    rand::random::<f32>() > 0.05, // 95% success rate
                    duration_ms,
                )
                .with_cost(
                    base_cost * usage_multiplier,
                    (input_tokens as f64 * usage_multiplier) as u32,
                    (output_tokens as f64 * usage_multiplier) as u32,
                    "claude-3-opus".to_string(),
                );
                history_entry.timestamp = timestamp;

                self.history_store
                    .write()
                    .await
                    .store_entry(history_entry)
                    .await?;

                // Update metrics
                let metric = claude_ai_interactive::analytics::CommandMetric::new(
                    cmd_name.to_string(),
                    duration_ms as f64,
                    rand::random::<f32>() > 0.05,
                    base_cost * usage_multiplier,
                    (input_tokens as f64 * usage_multiplier) as u32,
                    (output_tokens as f64 * usage_multiplier) as u32,
                    session_id,
                );
                self.metrics_engine.record_command_metric(metric).await?;
            }
        }

        Ok(())
    }
}

#[tokio::test]
async fn test_end_to_end_data_flow() -> Result<()> {
    let harness = AnalyticsTestHarness::new().await?;
    let session_id = Uuid::new_v4();

    // Step 1: Record cost data
    let cost_entry = CostEntry::new(
        session_id,
        "test_command".to_string(),
        0.15,
        200,
        400,
        1500,
        "claude-3-opus".to_string(),
    );

    harness
        .cost_tracker
        .write()
        .await
        .record_cost(cost_entry.clone())
        .await?;

    // Step 2: Verify cost data flows to analytics engine
    let summary = harness.analytics_engine.generate_summary(1).await?;
    assert_eq!(summary.cost_summary.total_cost, 0.15);
    assert_eq!(summary.cost_summary.command_count, 1);

    // Step 3: Verify dashboard reflects the cost
    let dashboard_data = harness.dashboard_manager.generate_live_data().await?;
    assert!(dashboard_data.current_metrics.today_cost >= 0.15); // May include other test data

    // Step 4: Verify metrics are updated
    let snapshot = harness.metrics_engine.get_snapshot().await;
    assert!(snapshot
        .counters
        .iter()
        .any(|c| c.name == "commands_total" && c.value > 0));

    // Step 5: Generate and verify report includes the data
    let report_path = harness.temp_dir.path().join("test_report.json");
    harness
        .analytics_engine
        .export_analytics_report(&report_path, ReportFormat::Json, 1)
        .await?;

    let report_content = tokio::fs::read_to_string(&report_path).await?;
    assert!(report_content.contains("test_command"));
    assert!(report_content.contains("0.15"));

    Ok(())
}

#[tokio::test]
async fn test_real_time_updates() -> Result<()> {
    let harness = AnalyticsTestHarness::new().await?;
    let session_id = Uuid::new_v4();

    // Start real-time streaming
    harness.realtime_stream.start_streaming().await?;

    // Wait for stream to initialize
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Record multiple cost entries rapidly
    for i in 0..10 {
        let cost_entry = CostEntry::new(
            session_id,
            format!("stream_cmd_{}", i),
            0.01 * (i + 1) as f64,
            50 + i as u32 * 10,
            100 + i as u32 * 20,
            (500 + i * 100) as u64,
            "claude-3-opus".to_string(),
        );

        harness
            .cost_tracker
            .write()
            .await
            .record_cost(cost_entry)
            .await?;

        // Simulate real-time update
        let metric = claude_ai_interactive::analytics::CommandMetric::new(
            format!("stream_cmd_{}", i),
            (500 + i * 100) as f64,
            true,
            0.01 * (i + 1) as f64,
            50 + i as u32 * 10,
            100 + i as u32 * 20,
            session_id,
        );
        harness.metrics_engine.record_command_metric(metric).await?;
    }

    // Get recent updates from the stream
    let recent_updates = harness.realtime_stream.get_recent_updates(5).await;
    assert!(!recent_updates.is_empty());

    // Verify update count increased
    let update_count = harness
        .realtime_stream
        .update_count
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(update_count > 0);

    // Stop streaming
    harness.realtime_stream.stop_streaming();

    Ok(())
}

#[tokio::test]
async fn test_budget_alert_integration() -> Result<()> {
    let harness = AnalyticsTestHarness::new().await?;
    let session_id = Uuid::new_v4();

    // Create advanced cost tracker with budget
    // Note: In a real scenario, we'd use the same cost tracker instance
    let temp_tracker = CostTracker::new(harness.temp_dir.path().join("temp_costs.json"))?;
    let mut advanced_tracker = AdvancedCostTracker::new(temp_tracker);

    let budget = Budget {
        id: Uuid::new_v4().to_string(),
        name: "Test Budget".to_string(),
        limit_usd: 1.0,
        period_days: 30,
        scope: BudgetScope::Global,
        alert_thresholds: vec![50.0, 80.0, 95.0],
        created_at: Utc::now(),
    };

    advanced_tracker.add_budget(budget.clone())?;

    // Add cost alert
    let alert = CostAlert {
        id: Uuid::new_v4().to_string(),
        name: "High Cost Alert".to_string(),
        threshold_usd: 0.5,
        period_days: 1,
        enabled: true,
        notification_channels: vec!["dashboard".to_string()],
    };

    advanced_tracker.add_alert(alert)?;

    // Record costs that should trigger alerts
    for i in 0..15 {
        let cost_entry = CostEntry::new(
            session_id,
            format!("expensive_cmd_{}", i),
            0.10,
            200,
            400,
            2000,
            "claude-3-opus".to_string(),
        );

        harness
            .cost_tracker
            .write()
            .await
            .record_cost(cost_entry.clone())
            .await?;

        // Check budget status after each entry
        let violations = advanced_tracker.check_budget_limits(&cost_entry).await?;
        if !violations.is_empty() {
            // Budget exceeded - verify alert is generated
            let summary = harness.analytics_engine.generate_summary(1).await?;
            assert!(!summary.alerts.is_empty());
            break;
        }
    }

    // Verify alerts appear in dashboard
    let dashboard_data = harness.dashboard_manager.generate_live_data().await?;
    assert!(!dashboard_data.current_metrics.active_alerts.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_performance_under_load() -> Result<()> {
    let harness = AnalyticsTestHarness::new().await?;
    let start_time = std::time::Instant::now();

    // Create multiple sessions
    let sessions: Vec<SessionId> = (0..10).map(|_| Uuid::new_v4()).collect();

    // Wrap harness in Arc for sharing
    let harness = Arc::new(harness);

    // Simulate concurrent usage across sessions
    let mut handles = vec![];

    for (idx, session_id) in sessions.iter().enumerate() {
        let harness_clone = Arc::clone(&harness);
        let session_id = *session_id;

        let handle = tokio::spawn(async move {
            // Each session generates different amount of load
            for i in 0..100 {
                let cost_entry = CostEntry::new(
                    session_id,
                    format!("load_test_cmd_{}_{}", idx, i),
                    0.001 * (i + 1) as f64,
                    10 + i as u32,
                    20 + i as u32 * 2,
                    (100 + i * 10) as u64,
                    "claude-3-haiku".to_string(),
                );

                harness_clone
                    .cost_tracker
                    .write()
                    .await
                    .record_cost(cost_entry)
                    .await
                    .unwrap();

                // Add corresponding history entry
                let history_entry = HistoryEntry::new(
                    session_id,
                    format!("load_test_cmd_{}_{}", idx, i),
                    vec![],
                    format!("Output {}", i),
                    true,
                    (100 + i * 10) as u64,
                );

                harness_clone
                    .history_store
                    .write()
                    .await
                    .store_entry(history_entry)
                    .await
                    .unwrap();

                // Don't overwhelm the system
                if i % 10 == 0 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        handle.await?;
    }

    let elapsed = start_time.elapsed();

    // Verify all data was recorded correctly
    let summary = harness.analytics_engine.generate_summary(1).await?;
    assert_eq!(summary.cost_summary.command_count, 1000); // 10 sessions * 100 commands
    assert_eq!(summary.history_stats.total_entries, 1000);

    // Performance assertion - should complete in reasonable time
    assert!(
        elapsed.as_secs() < 30,
        "Load test took too long: {:?}",
        elapsed
    );

    // Verify no data loss or corruption
    for session_id in sessions {
        let session_summary = harness
            .cost_tracker
            .read()
            .await
            .get_session_summary(session_id)
            .await?;
        assert_eq!(session_summary.command_count, 100);
    }

    Ok(())
}

#[tokio::test]
async fn test_data_consistency_across_modules() -> Result<()> {
    let harness = AnalyticsTestHarness::new().await?;
    let session_id = Uuid::new_v4();

    // Record identical data in both cost tracker and history store
    let command_name = "consistency_test";
    let cost = 0.25;
    let input_tokens = 300;
    let output_tokens = 600;
    let duration_ms = 2000;

    // Record in cost tracker
    let cost_entry = CostEntry::new(
        session_id,
        command_name.to_string(),
        cost,
        input_tokens,
        output_tokens,
        duration_ms,
        "claude-3-opus".to_string(),
    );

    harness
        .cost_tracker
        .write()
        .await
        .record_cost(cost_entry)
        .await?;

    // Record in history store
    let history_entry = HistoryEntry::new(
        session_id,
        command_name.to_string(),
        vec!["--test".to_string()],
        "Test output".to_string(),
        true,
        duration_ms,
    )
    .with_cost(
        cost,
        input_tokens,
        output_tokens,
        "claude-3-opus".to_string(),
    );

    harness
        .history_store
        .write()
        .await
        .store_entry(history_entry)
        .await?;

    // Verify consistency in analytics summary
    let summary = harness.analytics_engine.generate_summary(1).await?;

    // Cost data
    assert_eq!(summary.cost_summary.total_cost, cost);
    assert_eq!(summary.cost_summary.command_count, 1);
    assert_eq!(
        summary.cost_summary.total_tokens,
        input_tokens + output_tokens
    );

    // History data
    assert_eq!(summary.history_stats.total_entries, 1);
    assert_eq!(summary.history_stats.successful_commands, 1);
    assert_eq!(summary.history_stats.failed_commands, 0);

    // Performance metrics
    assert_eq!(
        summary.performance_metrics.average_response_time,
        duration_ms as f64
    );
    assert_eq!(summary.performance_metrics.success_rate, 100.0);

    // Verify in dashboard
    let dashboard_data = harness.dashboard_manager.generate_live_data().await?;
    assert!(dashboard_data.current_metrics.today_cost >= cost);
    assert!(dashboard_data.current_metrics.today_commands >= 1);

    Ok(())
}

#[tokio::test]
async fn test_error_recovery_and_handling() -> Result<()> {
    let harness = AnalyticsTestHarness::new().await?;
    let session_id = Uuid::new_v4();

    // Simulate various error scenarios

    // 1. Record failed commands
    for i in 0..5 {
        let history_entry = HistoryEntry::new(
            session_id,
            format!("failing_cmd_{}", i),
            vec!["--fail".to_string()],
            "Error: Command failed".to_string(),
            false, // Failed
            500,
        );

        harness
            .history_store
            .write()
            .await
            .store_entry(history_entry)
            .await?;
    }

    // 2. Verify error metrics are tracked correctly
    let summary = harness.analytics_engine.generate_summary(1).await?;
    assert_eq!(summary.history_stats.failed_commands, 5);
    assert!(summary.performance_metrics.success_rate < 100.0);

    // 3. Verify error alerts are generated
    assert!(summary.alerts.iter().any(|a| matches!(
        a.alert_type,
        claude_ai_interactive::analytics::AlertType::ErrorRate
    )));

    // 4. Test recovery - add successful commands
    for i in 0..10 {
        let history_entry = HistoryEntry::new(
            session_id,
            format!("success_cmd_{}", i),
            vec!["--success".to_string()],
            "Success".to_string(),
            true,
            300,
        );

        harness
            .history_store
            .write()
            .await
            .store_entry(history_entry)
            .await?;
    }

    // 5. Verify metrics improve
    let updated_summary = harness.analytics_engine.generate_summary(1).await?;
    assert!(updated_summary.performance_metrics.success_rate > 50.0);

    Ok(())
}

#[tokio::test]
async fn test_daily_usage_pattern_simulation() -> Result<()> {
    let harness = AnalyticsTestHarness::new().await?;
    let session_id = Uuid::new_v4();

    // Simulate 7 days of usage
    for day in 0..7 {
        harness.simulate_daily_usage(session_id, day).await?;
    }

    // Analyze weekly patterns
    let weekly_summary = harness.analytics_engine.generate_summary(7).await?;

    // Verify realistic usage patterns
    assert!(weekly_summary.cost_summary.total_cost > 0.0);
    assert!(weekly_summary.cost_summary.command_count > 100); // Multiple commands per day
    assert!(weekly_summary.performance_metrics.peak_usage_hour >= 9); // Business hours
    assert!(weekly_summary.performance_metrics.peak_usage_hour <= 16);

    // Generate weekly report
    let report_path = harness.temp_dir.path().join("weekly_report.html");
    harness
        .analytics_engine
        .export_analytics_report(&report_path, ReportFormat::Html, 7)
        .await?;

    assert!(report_path.exists());

    // Verify cost trends
    let temp_tracker = CostTracker::new(harness.temp_dir.path().join("temp_costs2.json"))?;
    let advanced_tracker = AdvancedCostTracker::new(temp_tracker);
    let trend = advanced_tracker.analyze_cost_trend(7).await?;

    assert_eq!(trend.period_days, 7);
    assert!(!trend.daily_costs.is_empty());
    assert!(trend.projected_monthly_cost > 0.0);

    Ok(())
}

#[tokio::test]
async fn test_concurrent_operations() -> Result<()> {
    let harness = Arc::new(AnalyticsTestHarness::new().await?);
    let session_id = Uuid::new_v4();

    // Start real-time streaming
    harness.realtime_stream.start_streaming().await?;

    // Concurrent operations
    let mut handles = vec![];

    // Task 1: Continuous cost recording
    let h1 = {
        let harness = Arc::clone(&harness);
        tokio::spawn(async move {
            for i in 0..50 {
                let cost_entry = CostEntry::new(
                    session_id,
                    format!("concurrent_cmd_{}", i),
                    0.01,
                    50,
                    100,
                    200,
                    "claude-3-haiku".to_string(),
                );

                harness
                    .cost_tracker
                    .write()
                    .await
                    .record_cost(cost_entry)
                    .await
                    .unwrap();

                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        })
    };
    handles.push(h1);

    // Task 2: Dashboard updates
    let h2 = {
        let harness = Arc::clone(&harness);
        tokio::spawn(async move {
            for _ in 0..20 {
                let _data = harness
                    .dashboard_manager
                    .generate_live_data()
                    .await
                    .unwrap();
                tokio::time::sleep(tokio::time::Duration::from_millis(25)).await;
            }
        })
    };
    handles.push(h2);

    // Task 3: Report generation
    let h3 = {
        let harness = Arc::clone(&harness);
        tokio::spawn(async move {
            for i in 0..5 {
                let path = harness
                    .temp_dir
                    .path()
                    .join(format!("concurrent_report_{}.json", i));
                harness
                    .analytics_engine
                    .export_analytics_report(&path, ReportFormat::Json, 1)
                    .await
                    .unwrap();
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        })
    };
    handles.push(h3);

    // Task 4: Metrics updates
    let h4 = {
        let harness = Arc::clone(&harness);
        tokio::spawn(async move {
            for i in 0..30 {
                let metric = claude_ai_interactive::analytics::CommandMetric::new(
                    format!("metric_cmd_{}", i),
                    150.0,
                    true,
                    0.02,
                    50,
                    100,
                    session_id,
                );
                harness
                    .metrics_engine
                    .record_command_metric(metric)
                    .await
                    .unwrap();
                tokio::time::sleep(tokio::time::Duration::from_millis(15)).await;
            }
        })
    };
    handles.push(h4);

    // Wait for all tasks to complete
    for handle in handles {
        handle.await?;
    }

    // Stop streaming
    harness.realtime_stream.stop_streaming();

    // Verify data integrity after concurrent operations
    let final_summary = harness.analytics_engine.generate_summary(1).await?;
    assert!(final_summary.cost_summary.command_count >= 50);

    Ok(())
}

#[tokio::test]
async fn test_memory_usage_monitoring() -> Result<()> {
    let harness = AnalyticsTestHarness::new().await?;
    let session_id = Uuid::new_v4();

    // Start memory tracking
    let memory_stats_before = harness.realtime_stream.memory_tracker.get_stats();

    // Generate significant load
    for i in 0..100 {
        let cost_entry = CostEntry::new(
            session_id,
            format!("memory_test_cmd_{}", i),
            0.01,
            100,
            200,
            500,
            "claude-3-opus".to_string(),
        );

        harness
            .cost_tracker
            .write()
            .await
            .record_cost(cost_entry)
            .await?;

        // Process analytics update
        let update = claude_ai_interactive::analytics::AnalyticsUpdate {
            timestamp: Utc::now(),
            update_type: UpdateType::CommandCompleted,
            session_id,
            metric_deltas: claude_ai_interactive::analytics::MetricDeltas {
                cost_delta: 0.01,
                command_count_delta: 1,
                success_count_delta: 1,
                ..Default::default()
            },
        };

        harness
            .realtime_stream
            .process_update_batch(vec![update])
            .await?;
    }

    // Check memory stats after load
    let memory_stats_after = harness.realtime_stream.memory_tracker.get_stats();

    // Verify memory tracking is working
    assert!(memory_stats_after.total_allocations > memory_stats_before.total_allocations);
    assert!(memory_stats_after.peak_memory_mb > 0);

    // Verify memory cleanup
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    let memory_stats_final = harness.realtime_stream.memory_tracker.get_stats();
    assert!(memory_stats_final.total_deallocations > 0);

    Ok(())
}

#[tokio::test]
async fn test_session_tracking_integration() -> Result<()> {
    let harness = AnalyticsTestHarness::new().await?;

    // Create multiple sessions
    let session_ids: Vec<SessionId> = (0..3).map(|_| Uuid::new_v4()).collect();

    // Add different amounts of activity to each session
    for (idx, session_id) in session_ids.iter().enumerate() {
        let command_count = (idx + 1) * 10;
        let cost_per_command = 0.05 * (idx + 1) as f64;

        for i in 0..command_count {
            let cost_entry = CostEntry::new(
                *session_id,
                format!("session_{}_cmd_{}", idx, i),
                cost_per_command,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            );

            harness
                .cost_tracker
                .write()
                .await
                .record_cost(cost_entry)
                .await?;
        }
    }

    // Generate session reports for each
    for (idx, session_id) in session_ids.iter().enumerate() {
        let report = harness
            .analytics_engine
            .generate_session_report(*session_id)
            .await?;

        assert_eq!(report.session_id, *session_id);
        assert_eq!(report.cost_summary.command_count, (idx + 1) * 10);
        assert!(report.cost_summary.total_cost > 0.0);
    }

    // Verify global summary includes all sessions
    let global_summary = harness.analytics_engine.generate_summary(1).await?;
    assert_eq!(global_summary.cost_summary.command_count, 60); // 10 + 20 + 30

    Ok(())
}

#[tokio::test]
async fn test_alert_lifecycle() -> Result<()> {
    let harness = AnalyticsTestHarness::new().await?;
    let session_id = Uuid::new_v4();

    // Use the engine with default config (cost_alert_threshold is 5.0)
    // We'll just add more costs to exceed the threshold

    // Record costs to trigger alert (default threshold is 5.0)
    for i in 0..15 {
        let cost_entry = CostEntry::new(
            session_id,
            format!("alert_test_cmd_{}", i),
            0.50, // Higher cost to exceed 5.0 threshold
            50,
            100,
            500,
            "claude-3-opus".to_string(),
        );

        harness
            .cost_tracker
            .write()
            .await
            .record_cost(cost_entry)
            .await?;
    }

    // Check for alerts
    let summary = harness.analytics_engine.generate_summary(1).await?;
    let cost_alerts: Vec<_> = summary
        .alerts
        .iter()
        .filter(|a| matches!(a.alert_type, AlertType::CostThreshold))
        .collect();

    assert!(!cost_alerts.is_empty());
    assert_eq!(cost_alerts[0].resolved, false);

    // Verify alert appears in dashboard
    let dashboard_data = harness.dashboard_manager.generate_live_data().await?;
    assert!(dashboard_data
        .current_metrics
        .active_alerts
        .iter()
        .any(|a| matches!(a.alert_type, AlertType::CostThreshold)));

    Ok(())
}

#[tokio::test]
async fn test_report_format_generation() -> Result<()> {
    let harness = AnalyticsTestHarness::new().await?;
    let session_id = Uuid::new_v4();

    // Add some test data
    harness.simulate_daily_usage(session_id, 0).await?;

    // Test all report formats
    let formats = vec![
        (ReportFormat::Json, "report.json"),
        (ReportFormat::Html, "report.html"),
        (ReportFormat::Csv, "report.csv"),
        (ReportFormat::Markdown, "report.md"),
    ];

    for (format, filename) in formats {
        let path = harness.temp_dir.path().join(filename);
        harness
            .analytics_engine
            .export_analytics_report(&path, format.clone(), 1)
            .await?;

        assert!(path.exists());

        let content = tokio::fs::read_to_string(&path).await?;
        assert!(!content.is_empty());

        // Verify format-specific content
        match &format {
            ReportFormat::Json => {
                assert!(content.contains("cost_summary"));
                assert!(content.contains("performance_metrics"));
            }
            ReportFormat::Html => {
                assert!(content.contains("<html>"));
                assert!(content.contains("Analytics Report"));
            }
            ReportFormat::Csv => {
                assert!(content.contains("Metric,Value"));
            }
            ReportFormat::Markdown => {
                assert!(content.contains("# Claude AI Interactive"));
                assert!(content.contains("## Cost Summary"));
            }
            _ => {}
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_dashboard_health_monitoring() -> Result<()> {
    let harness = AnalyticsTestHarness::new().await?;

    // Get initial dashboard data and verify health status
    let live_data = harness.dashboard_manager.generate_live_data().await?;
    assert_eq!(live_data.system_status.health, HealthStatus::Healthy);

    // Simulate unhealthy conditions by adding many errors
    let session_id = Uuid::new_v4();
    for i in 0..20 {
        let history_entry = HistoryEntry::new(
            session_id,
            format!("error_cmd_{}", i),
            vec![],
            "Error".to_string(),
            false,
            100,
        );

        harness
            .history_store
            .write()
            .await
            .store_entry(history_entry)
            .await?;
    }

    // Check if health status reflects issues
    let summary = harness.analytics_engine.generate_summary(1).await?;
    assert!(!summary.alerts.is_empty());

    Ok(())
}

/// Helper function to generate realistic command patterns
fn generate_command_pattern(hour: u32) -> &'static str {
    match hour {
        6..=8 => "morning_routine",
        9..=11 => "analysis_work",
        12..=13 => "lunch_queries",
        14..=16 => "afternoon_tasks",
        17..=18 => "wrap_up",
        19..=21 => "evening_research",
        _ => "misc_queries",
    }
}

#[tokio::test]
async fn test_time_series_data_accuracy() -> Result<()> {
    let harness = AnalyticsTestHarness::new().await?;
    let session_id = Uuid::new_v4();

    // Record costs at specific times
    let base_time = Utc::now();
    for hour in 0..24 {
        let timestamp = base_time - Duration::hours(23 - hour);
        let cost = 0.05 * ((hour % 4) + 1) as f64;

        let mut cost_entry = CostEntry::new(
            session_id,
            generate_command_pattern(hour as u32).to_string(),
            cost,
            100,
            200,
            1000,
            "claude-3-opus".to_string(),
        );
        cost_entry.timestamp = timestamp;

        harness
            .cost_tracker
            .write()
            .await
            .record_cost(cost_entry)
            .await?;
    }

    // Get live dashboard data with time series
    let live_data = harness.dashboard_manager.generate_live_data().await?;

    // Verify time series data
    assert!(!live_data.time_series.cost_over_time.is_empty());
    assert!(!live_data.time_series.commands_over_time.is_empty());

    // Verify data points are in chronological order
    let mut prev_time = None;
    for (timestamp, _) in &live_data.time_series.cost_over_time {
        if let Some(prev) = prev_time {
            assert!(timestamp > &prev);
        }
        prev_time = Some(*timestamp);
    }

    Ok(())
}

// Add more integration tests as needed...

#[cfg(test)]
mod utils {
    use super::*;

    /// Helper to create test data with specific patterns
    pub async fn create_pattern_data(
        cost_tracker: &Arc<RwLock<CostTracker>>,
        pattern: &str,
        count: usize,
    ) -> Result<()> {
        let session_id = Uuid::new_v4();

        for i in 0..count {
            let (cost, tokens) = match pattern {
                "high_cost" => (1.0 + i as f64 * 0.5, 1000),
                "low_cost" => (0.01, 50),
                "variable" => (0.1 * ((i % 10) + 1) as f64, 100 * ((i % 5) + 1)),
                _ => (0.05, 100),
            };

            let cost_entry = CostEntry::new(
                session_id,
                format!("{}_cmd_{}", pattern, i),
                cost,
                tokens as u32,
                (tokens * 2) as u32,
                500 + i as u64 * 100,
                "claude-3-opus".to_string(),
            );

            cost_tracker.write().await.record_cost(cost_entry).await?;
        }

        Ok(())
    }
}
