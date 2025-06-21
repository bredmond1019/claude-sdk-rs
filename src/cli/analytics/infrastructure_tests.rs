//! Analytics Test Infrastructure Completion (Task 3.1)
//!
//! This module completes the analytics test infrastructure by providing:
//! - Enhanced test fixtures and utilities
//! - Integration test scenarios
//! - Performance and stress testing utilities
//! - Mock data generators for comprehensive testing

use super::*;
use crate::{
    cost::{CostEntry, CostTracker},
    history::{HistoryEntry, HistoryStore},
    session::SessionId,
};
use chrono::{DateTime, Datelike, Duration, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Comprehensive analytics test infrastructure
pub struct AnalyticsTestInfrastructure {
    pub engine: AnalyticsEngine,
    pub cost_tracker: Arc<RwLock<CostTracker>>,
    pub history_store: Arc<RwLock<HistoryStore>>,
    pub temp_dir: TempDir,
    pub config: AnalyticsConfig,
}

impl AnalyticsTestInfrastructure {
    /// Create a new test infrastructure instance
    pub async fn new() -> Self {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let config = AnalyticsConfig::default();

        let cost_tracker = Arc::new(RwLock::new(
            CostTracker::new(temp_dir.path().join("test_costs.json")).unwrap(),
        ));

        let history_store = Arc::new(RwLock::new(
            HistoryStore::new(temp_dir.path().join("test_history.json")).unwrap(),
        ));

        let engine = AnalyticsEngine::new(
            Arc::clone(&cost_tracker),
            Arc::clone(&history_store),
            config.clone(),
        );

        Self {
            engine,
            cost_tracker,
            history_store,
            temp_dir,
            config,
        }
    }

    /// Create infrastructure with custom config
    pub async fn with_config(config: AnalyticsConfig) -> Self {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");

        let cost_tracker = Arc::new(RwLock::new(
            CostTracker::new(temp_dir.path().join("test_costs.json")).unwrap(),
        ));

        let history_store = Arc::new(RwLock::new(
            HistoryStore::new(temp_dir.path().join("test_history.json")).unwrap(),
        ));

        let engine = AnalyticsEngine::new(
            Arc::clone(&cost_tracker),
            Arc::clone(&history_store),
            config.clone(),
        );

        Self {
            engine,
            cost_tracker,
            history_store,
            temp_dir,
            config,
        }
    }

    /// Add realistic test scenario data
    pub async fn populate_realistic_scenario(&self) -> Vec<SessionId> {
        let mut session_ids = Vec::new();

        // Create 5 different sessions with varying patterns
        for session_idx in 0..5 {
            let session_id = Uuid::new_v4();
            session_ids.push(session_id);

            // Add data for the last 30 days
            for day in 0..30 {
                let base_date = Utc::now() - Duration::days(30 - day);

                // Different sessions have different usage patterns
                let commands_per_day = match session_idx {
                    0 => 2 + (day % 3), // Regular user: 2-4 commands/day
                    1 => 8 + (day % 5), // Power user: 8-12 commands/day
                    2 => {
                        if day % 7 < 2 {
                            0
                        } else {
                            5
                        }
                    } // Weekend warrior
                    3 => {
                        if day < 10 {
                            15
                        } else {
                            2
                        }
                    } // Early adopter, then casual
                    4 => 1,             // Minimal user
                    _ => 3,
                };

                for cmd_idx in 0..commands_per_day {
                    self.add_realistic_command(session_id, base_date, session_idx, cmd_idx)
                        .await;
                }
            }
        }

        session_ids
    }

    /// Add a realistic command with proper cost and timing patterns
    async fn add_realistic_command(
        &self,
        session_id: SessionId,
        base_time: DateTime<Utc>,
        session_type: i64,
        cmd_idx: i64,
    ) {
        // Vary command timing throughout the day
        let hour_offset = match cmd_idx % 3 {
            0 => 9,  // Morning
            1 => 14, // Afternoon
            2 => 20, // Evening
            _ => 12,
        };

        let timestamp = base_time + Duration::hours(hour_offset) + Duration::minutes(cmd_idx * 15);

        // Realistic command patterns
        let (command_name, base_cost, success_rate, avg_duration) = match cmd_idx % 7 {
            0 => ("code_review", 0.15, 0.95, 3000),
            1 => ("generate_docs", 0.08, 0.90, 2500),
            2 => ("debug_issue", 0.25, 0.75, 8000),
            3 => ("refactor_code", 0.35, 0.85, 6000),
            4 => ("write_tests", 0.12, 0.92, 4000),
            5 => ("optimize_query", 0.18, 0.88, 5000),
            6 => ("create_api", 0.45, 0.80, 10000),
            _ => ("misc_task", 0.10, 0.90, 3000),
        };

        // Model selection based on task complexity
        let model = if base_cost > 0.3 {
            "claude-3-opus"
        } else if base_cost > 0.15 {
            "claude-3-sonnet"
        } else {
            "claude-3-haiku"
        };

        // Add variance based on session type
        let cost_multiplier = match session_type {
            0 => 1.0, // Regular user
            1 => 1.2, // Power user (more complex tasks)
            2 => 0.8, // Weekend warrior (simpler tasks)
            3 => {
                if timestamp.day() < 10 {
                    1.5
                } else {
                    0.7
                }
            } // Early adopter pattern
            4 => 0.6, // Minimal user (simple tasks)
            _ => 1.0,
        };

        let final_cost = base_cost * cost_multiplier;
        let duration = avg_duration + ((cmd_idx * 337) % 2000) as u64; // Add variance
        let success = (cmd_idx * 7 + session_type * 3) % 100 < (success_rate * 100.0) as i64;

        // Calculate realistic token counts based on cost
        let input_tokens = ((final_cost * 1000.0) as u32).max(50);
        let output_tokens = ((final_cost * 800.0) as u32).max(30);

        // Add cost entry
        let mut cost_entry = CostEntry::new(
            session_id,
            command_name.to_string(),
            final_cost,
            input_tokens,
            output_tokens,
            duration,
            model.to_string(),
        );
        cost_entry.timestamp = timestamp;

        self.cost_tracker
            .write()
            .await
            .record_cost(cost_entry)
            .await
            .unwrap();

        // Add history entry
        let output_content = if success {
            format!("Successfully completed {} task", command_name)
        } else {
            format!("Failed to complete {} - error occurred", command_name)
        };

        let mut history_entry = HistoryEntry::new(
            session_id,
            command_name.to_string(),
            vec!["--verbose".to_string()],
            output_content,
            success,
            duration,
        );
        history_entry.timestamp = timestamp;
        history_entry =
            history_entry.with_cost(final_cost, input_tokens, output_tokens, model.to_string());

        self.history_store
            .write()
            .await
            .store_entry(history_entry)
            .await
            .unwrap();
    }

    /// Add stress test data (large volume)
    pub async fn populate_stress_test_data(&self, days: u32, commands_per_day: usize) -> SessionId {
        let session_id = Uuid::new_v4();
        let start_date = Utc::now() - Duration::days(days as i64);

        for day in 0..days {
            let day_start = start_date + Duration::days(day as i64);

            for cmd_idx in 0..commands_per_day {
                let timestamp = day_start
                    + Duration::seconds((cmd_idx as i64 * 86400) / commands_per_day as i64);

                // Vary costs for realistic distribution
                let cost = 0.01 + (cmd_idx as f64 * 0.001) + (day as f64 * 0.005);
                let duration = 1000 + (cmd_idx as u64 * 100);
                let success = cmd_idx % 10 != 7; // 90% success rate

                let mut cost_entry = CostEntry::new(
                    session_id,
                    format!("stress_cmd_{}", cmd_idx % 20),
                    cost,
                    100 + (cmd_idx as u32 * 5),
                    150 + (cmd_idx as u32 * 7),
                    duration,
                    if cost > 0.1 {
                        "claude-3-opus"
                    } else {
                        "claude-3-haiku"
                    }
                    .to_string(),
                );
                cost_entry.timestamp = timestamp;

                self.cost_tracker
                    .write()
                    .await
                    .record_cost(cost_entry)
                    .await
                    .unwrap();

                let mut history_entry = HistoryEntry::new(
                    session_id,
                    format!("stress_cmd_{}", cmd_idx % 20),
                    vec![],
                    if success { "Success" } else { "Error" }.to_string(),
                    success,
                    duration,
                );
                history_entry.timestamp = timestamp;

                self.history_store
                    .write()
                    .await
                    .store_entry(history_entry)
                    .await
                    .unwrap();
            }
        }

        session_id
    }

    /// Generate test data with specific patterns for testing edge cases
    pub async fn populate_edge_case_data(&self) -> HashMap<String, SessionId> {
        let mut scenarios = HashMap::new();

        // Scenario 1: All failures
        let all_failures_session = Uuid::new_v4();
        scenarios.insert("all_failures".to_string(), all_failures_session);
        self.add_session_with_pattern(all_failures_session, 10, |_| (0.05, false, 2000))
            .await;

        // Scenario 2: Extremely expensive commands
        let expensive_session = Uuid::new_v4();
        scenarios.insert("expensive".to_string(), expensive_session);
        self.add_session_with_pattern(expensive_session, 5, |_| (10.0, true, 30000))
            .await;

        // Scenario 3: Very fast commands
        let fast_session = Uuid::new_v4();
        scenarios.insert("fast".to_string(), fast_session);
        self.add_session_with_pattern(fast_session, 50, |_| (0.001, true, 100))
            .await;

        // Scenario 4: Highly variable costs
        let variable_session = Uuid::new_v4();
        scenarios.insert("variable".to_string(), variable_session);
        self.add_session_with_pattern(variable_session, 20, |idx| {
            let cost = if idx % 2 == 0 { 0.01 } else { 1.0 };
            (cost, true, 2000)
        })
        .await;

        // Scenario 5: Peak hour concentration
        let peak_hour_session = Uuid::new_v4();
        scenarios.insert("peak_hour".to_string(), peak_hour_session);
        self.add_peak_hour_data(peak_hour_session).await;

        scenarios
    }

    /// Add session data with a specific pattern
    async fn add_session_with_pattern<F>(&self, session_id: SessionId, count: usize, pattern: F)
    where
        F: Fn(usize) -> (f64, bool, u64),
    {
        for i in 0..count {
            let (cost, success, duration) = pattern(i);
            let timestamp = Utc::now() - Duration::minutes((count - i) as i64 * 10);

            let mut cost_entry = CostEntry::new(
                session_id,
                format!("pattern_cmd_{}", i),
                cost,
                100,
                200,
                duration,
                "claude-3-opus".to_string(),
            );
            cost_entry.timestamp = timestamp;

            self.cost_tracker
                .write()
                .await
                .record_cost(cost_entry)
                .await
                .unwrap();

            let mut history_entry = HistoryEntry::new(
                session_id,
                format!("pattern_cmd_{}", i),
                vec![],
                if success { "Success" } else { "Error" }.to_string(),
                success,
                duration,
            );
            history_entry.timestamp = timestamp;

            self.history_store
                .write()
                .await
                .store_entry(history_entry)
                .await
                .unwrap();
        }
    }

    /// Add data concentrated in specific hours to test peak detection
    async fn add_peak_hour_data(&self, session_id: SessionId) {
        // Concentrate 70% of activity in hour 14 (2 PM)
        for hour in 0..24 {
            let commands_in_hour = if hour == 14 {
                20 // Peak hour
            } else if hour >= 9 && hour <= 17 {
                2 // Business hours
            } else {
                1 // Off hours
            };

            for cmd_idx in 0..commands_in_hour {
                let timestamp = Utc::now()
                    .date_naive()
                    .and_hms_opt(hour, cmd_idx as u32 * 3, 0)
                    .unwrap()
                    .and_utc();

                let mut cost_entry = CostEntry::new(
                    session_id,
                    format!("peak_cmd_{}_{}", hour, cmd_idx),
                    0.05,
                    100,
                    200,
                    2000,
                    "claude-3-sonnet".to_string(),
                );
                cost_entry.timestamp = timestamp;

                self.cost_tracker
                    .write()
                    .await
                    .record_cost(cost_entry)
                    .await
                    .unwrap();

                let mut history_entry = HistoryEntry::new(
                    session_id,
                    format!("peak_cmd_{}_{}", hour, cmd_idx),
                    vec![],
                    "Peak hour command".to_string(),
                    true,
                    2000,
                );
                history_entry.timestamp = timestamp;

                self.history_store
                    .write()
                    .await
                    .store_entry(history_entry)
                    .await
                    .unwrap();
            }
        }
    }

    /// Verify the integrity of test data
    pub async fn verify_data_integrity(&self) -> Result<DataIntegrityReport> {
        let cost_summary = self.cost_tracker.read().await.get_global_summary().await?;
        let history_stats = self.history_store.read().await.get_stats(None).await?;

        let issues = Vec::new();

        // TODO: Add specific integrity checks
        // - Verify cost entries match history entries where applicable
        // - Check for data consistency across different time ranges
        // - Validate that aggregations are mathematically correct

        Ok(DataIntegrityReport {
            cost_entries_count: cost_summary.command_count,
            history_entries_count: history_stats.total_entries,
            issues,
        })
    }

    /// Generate performance benchmark data
    pub async fn generate_performance_benchmark(&self) -> PerformanceBenchmark {
        let start_time = std::time::Instant::now();

        // Generate summary
        let summary_start = std::time::Instant::now();
        let summary = self.engine.generate_summary(30).await.unwrap();
        let summary_duration = summary_start.elapsed();

        // Generate dashboard data
        let dashboard_start = std::time::Instant::now();
        let _dashboard = self.engine.get_dashboard_data().await.unwrap();
        let dashboard_duration = dashboard_start.elapsed();

        // Generate session report
        let session_report_start = std::time::Instant::now();
        let _sessions = self
            .cost_tracker
            .read()
            .await
            .get_global_summary()
            .await
            .unwrap();
        let session_report_duration = session_report_start.elapsed();

        let total_duration = start_time.elapsed();

        PerformanceBenchmark {
            total_duration_ms: total_duration.as_millis() as u64,
            summary_generation_ms: summary_duration.as_millis() as u64,
            dashboard_generation_ms: dashboard_duration.as_millis() as u64,
            session_report_ms: session_report_duration.as_millis() as u64,
            data_points: summary.cost_summary.command_count,
        }
    }
}

/// Report of data integrity verification
#[derive(Debug)]
pub struct DataIntegrityReport {
    pub cost_entries_count: usize,
    pub history_entries_count: usize,
    pub issues: Vec<String>,
}

/// Performance benchmark results
#[derive(Debug)]
pub struct PerformanceBenchmark {
    pub total_duration_ms: u64,
    pub summary_generation_ms: u64,
    pub dashboard_generation_ms: u64,
    pub session_report_ms: u64,
    pub data_points: usize,
}

/// Test scenario builder for creating specific analytics test cases
pub struct AnalyticsScenarioBuilder {
    entries: Vec<CostEntry>,
    history_entries: Vec<HistoryEntry>,
    session_id: SessionId,
}

impl AnalyticsScenarioBuilder {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            history_entries: Vec::new(),
            session_id: Uuid::new_v4(),
        }
    }

    pub fn with_session(mut self, session_id: SessionId) -> Self {
        self.session_id = session_id;
        self
    }

    pub fn add_cost_spike(mut self, spike_cost: f64, duration_hours: i64) -> Self {
        let base_time = Utc::now() - Duration::hours(duration_hours);

        for hour in 0..duration_hours {
            let timestamp = base_time + Duration::hours(hour);
            let cost = if hour == duration_hours / 2 {
                spike_cost
            } else {
                spike_cost * 0.1
            };

            let mut entry = CostEntry::new(
                self.session_id,
                format!("spike_cmd_{}", hour),
                cost,
                200,
                400,
                3000,
                "claude-3-opus".to_string(),
            );
            entry.timestamp = timestamp;
            self.entries.push(entry);
        }

        self
    }

    pub fn add_error_burst(mut self, error_count: usize, total_commands: usize) -> Self {
        for i in 0..total_commands {
            let success = i >= error_count; // First N commands fail
            let timestamp = Utc::now() - Duration::minutes((total_commands - i) as i64 * 5);

            let entry = HistoryEntry::new(
                self.session_id,
                format!("burst_cmd_{}", i),
                vec![],
                if success { "Success" } else { "Error" }.to_string(),
                success,
                2000,
            );

            let mut history_entry = entry;
            history_entry.timestamp = timestamp;
            self.history_entries.push(history_entry);
        }

        self
    }

    pub fn build(self) -> (Vec<CostEntry>, Vec<HistoryEntry>) {
        (self.entries, self.history_entries)
    }
}

#[cfg(test)]
mod infrastructure_tests {
    use super::*;

    #[tokio::test]
    async fn test_infrastructure_creation() {
        let infrastructure = AnalyticsTestInfrastructure::new().await;

        // Verify infrastructure is properly initialized
        assert!(infrastructure.temp_dir.path().exists());

        // Test basic engine functionality
        let summary = infrastructure.engine.generate_summary(1).await.unwrap();
        assert_eq!(summary.cost_summary.total_cost, 0.0);
        assert_eq!(summary.history_stats.total_entries, 0);
    }

    #[tokio::test]
    async fn test_realistic_scenario_population() {
        let infrastructure = AnalyticsTestInfrastructure::new().await;
        let sessions = infrastructure.populate_realistic_scenario().await;

        assert_eq!(sessions.len(), 5);

        // Verify data was populated
        let summary = infrastructure.engine.generate_summary(30).await.unwrap();
        assert!(summary.cost_summary.total_cost > 0.0);
        assert!(summary.cost_summary.command_count > 0);
        assert!(summary.history_stats.total_entries > 0);

        // Verify different usage patterns
        assert!(summary.performance_metrics.peak_usage_hour < 24);
        assert!(summary.performance_metrics.success_rate > 0.0);
    }

    #[tokio::test]
    async fn test_stress_data_population() {
        let infrastructure = AnalyticsTestInfrastructure::new().await;
        let session_id = infrastructure.populate_stress_test_data(7, 100).await;

        // Verify large volume data
        let session_report = infrastructure
            .engine
            .generate_session_report(session_id)
            .await
            .unwrap();
        assert_eq!(session_report.history_stats.total_entries, 700); // 7 days * 100 commands
        assert!(session_report.cost_summary.total_cost > 0.0);
    }

    #[tokio::test]
    async fn test_edge_case_scenarios() {
        let infrastructure = AnalyticsTestInfrastructure::new().await;
        let scenarios = infrastructure.populate_edge_case_data().await;

        assert!(scenarios.contains_key("all_failures"));
        assert!(scenarios.contains_key("expensive"));
        assert!(scenarios.contains_key("fast"));
        assert!(scenarios.contains_key("variable"));
        assert!(scenarios.contains_key("peak_hour"));

        // Test all failures scenario
        let all_failures_session = scenarios["all_failures"];
        let report = infrastructure
            .engine
            .generate_session_report(all_failures_session)
            .await
            .unwrap();
        assert_eq!(report.history_stats.success_rate, 0.0);

        // Test expensive scenario
        let expensive_session = scenarios["expensive"];
        let expensive_report = infrastructure
            .engine
            .generate_session_report(expensive_session)
            .await
            .unwrap();
        assert!(expensive_report.cost_summary.total_cost > 30.0); // 5 commands * ~10.0 each
    }

    #[tokio::test]
    async fn test_scenario_builder() {
        let (cost_entries, history_entries) = AnalyticsScenarioBuilder::new()
            .add_cost_spike(5.0, 24)
            .add_error_burst(3, 10)
            .build();

        assert_eq!(cost_entries.len(), 24);
        assert_eq!(history_entries.len(), 10);

        // Verify spike pattern
        let max_cost = cost_entries.iter().map(|e| e.cost_usd).fold(0.0, f64::max);
        assert!(max_cost >= 5.0);

        // Verify error burst pattern
        let failures = history_entries.iter().filter(|e| !e.success).count();
        assert_eq!(failures, 3);
    }

    #[tokio::test]
    async fn test_performance_benchmark() {
        let infrastructure = AnalyticsTestInfrastructure::new().await;
        infrastructure.populate_realistic_scenario().await;

        let benchmark = infrastructure.generate_performance_benchmark().await;

        // Verify performance metrics are reasonable
        assert!(benchmark.total_duration_ms >= 0);
        assert!(benchmark.summary_generation_ms >= 0);
        assert!(benchmark.dashboard_generation_ms >= 0); // Can be 0 for very fast operations
        assert!(benchmark.data_points > 0);

        // Performance should be reasonable for test data size
        assert!(benchmark.summary_generation_ms < 5000); // Should complete in under 5 seconds
    }

    #[tokio::test]
    async fn test_custom_config_infrastructure() {
        let config = AnalyticsConfig {
            enable_real_time_alerts: true,
            cost_alert_threshold: 1.0, // Low threshold
            report_schedule: ReportSchedule::Daily,
            retention_days: 30,
            dashboard_refresh_interval: 60,
        };

        let infrastructure = AnalyticsTestInfrastructure::with_config(config.clone()).await;
        infrastructure.populate_realistic_scenario().await;

        let summary = infrastructure.engine.generate_summary(30).await.unwrap();

        // Should trigger alerts with low threshold
        assert!(!summary.alerts.is_empty());

        // Verify config is applied
        assert_eq!(infrastructure.config.cost_alert_threshold, 1.0);
    }
}
