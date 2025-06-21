//! Test infrastructure for dashboard functionality
//!
//! This module provides test fixtures, data generators, and utilities
//! specifically for dashboard testing scenarios.

use super::*;
use crate::cli::analytics::dashboard::{DashboardConfig, DashboardManager, HealthStatus};
use crate::cli::analytics::test_utils::{AnalyticsTestFixture, TestDataSet};
use crate::cli::analytics::{AnalyticsEngine, DashboardData};
use crate::cli::cost::{CostEntry, CostTracker};
use crate::cli::error::Result;
use crate::cli::history::{HistoryEntry, HistoryStore};
use chrono::{DateTime, Duration, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::RwLock;

/// Dashboard-specific test fixture
pub struct DashboardTestFixture {
    pub analytics_engine: AnalyticsEngine,
    pub dashboard_manager: DashboardManager,
    pub cost_tracker: Arc<RwLock<CostTracker>>,
    pub history_store: Arc<RwLock<HistoryStore>>,
    pub temp_dir: TempDir,
}

impl DashboardTestFixture {
    pub async fn new() -> Result<Self> {
        let base_fixture = AnalyticsTestFixture::new().await?;

        let analytics_engine = AnalyticsEngine::new(
            Arc::clone(&base_fixture.cost_tracker),
            Arc::clone(&base_fixture.history_store),
            Default::default(),
        );

        let dashboard_manager =
            DashboardManager::new(analytics_engine.clone(), DashboardConfig::default());

        Ok(Self {
            analytics_engine,
            dashboard_manager,
            cost_tracker: base_fixture.cost_tracker,
            history_store: base_fixture.history_store,
            temp_dir: base_fixture.temp_dir,
        })
    }

    /// Populate test data for a specified number of days
    pub async fn populate_test_data(&self, days: u32) -> Result<()> {
        let generator = DashboardTestDataGenerator::new();
        let test_data = generator.generate_dashboard_data(days);
        self.load_data(&test_data).await
    }

    /// Load test data into the fixture
    pub async fn load_data(&self, data: &TestDataSet) -> Result<()> {
        let mut cost_tracker = self.cost_tracker.write().await;
        let mut history_store = self.history_store.write().await;

        for entry in &data.cost_entries {
            cost_tracker.record_cost(entry.clone()).await?;
        }

        for entry in &data.history_entries {
            history_store.store_entry(entry.clone()).await?;
        }

        Ok(())
    }
}

/// Dashboard-specific data generator
pub struct DashboardTestDataGenerator {
    base_time: DateTime<Utc>,
}

impl DashboardTestDataGenerator {
    pub fn new() -> Self {
        Self {
            base_time: Utc::now() - Duration::days(30),
        }
    }

    /// Generate dashboard-specific test data
    pub fn generate_dashboard_data(&self, days: u32) -> TestDataSet {
        let mut data = TestDataSet {
            sessions: Vec::new(),
            cost_entries: Vec::new(),
            history_entries: Vec::new(),
            start_time: Utc::now() - Duration::days(days as i64),
            end_time: Utc::now(),
        };

        // Generate data for each day
        for day in 0..days {
            let day_offset = Duration::days(day as i64);
            let day_start = Utc::now() - Duration::days((days - day) as i64);

            // Generate hourly data points for realistic dashboard metrics
            for hour in 0..24 {
                let timestamp = day_start + Duration::hours(hour);
                let commands_this_hour = ((hour as f64 * 0.5).sin().abs() * 10.0) as usize + 1;

                for cmd_idx in 0..commands_this_hour {
                    let session_id = uuid::Uuid::new_v4();
                    let command_name = format!("cmd_{}_{}", day, cmd_idx);
                    let cost = 0.01 + (cmd_idx as f64 * 0.001);

                    // Create cost entry
                    let mut cost_entry = CostEntry::new(
                        session_id,
                        command_name.clone(),
                        cost,
                        100 + cmd_idx as u32 * 10,
                        200 + cmd_idx as u32 * 20,
                        1000 + cmd_idx as u64 * 100,
                        "claude-3-opus".to_string(),
                    );
                    cost_entry.timestamp = timestamp;
                    data.cost_entries.push(cost_entry);

                    // Create history entry
                    let mut history_entry = HistoryEntry::new(
                        session_id,
                        command_name,
                        vec![],
                        "output".to_string(),
                        cmd_idx % 10 != 9, // 90% success rate
                        1000 + cmd_idx as u64 * 100,
                    );
                    history_entry.timestamp = timestamp;
                    data.history_entries.push(history_entry);
                }
            }
        }

        data
    }

    /// Generate time series data for charts
    pub fn generate_time_series_data(&self, hours: u32) -> Vec<(DateTime<Utc>, f64)> {
        let mut series = Vec::new();
        let now = Utc::now();

        for hour in 0..hours {
            let timestamp = now - Duration::hours(hour as i64);
            let value = ((hour as f64 * 0.1).sin().abs() * 100.0) + 50.0;
            series.push((timestamp, value));
        }

        series.reverse();
        series
    }

    /// Generate widget-specific test data
    pub fn generate_widget_data(&self, widget_type: &WidgetType) -> HashMap<String, Value> {
        let mut data = HashMap::new();

        match widget_type {
            WidgetType::CostSummary => {
                data.insert("total_cost".to_string(), serde_json::json!(125.43));
                data.insert("today_cost".to_string(), serde_json::json!(12.34));
                data.insert("trend".to_string(), serde_json::json!("+15%"));
            }
            WidgetType::CommandActivity => {
                data.insert("total_commands".to_string(), serde_json::json!(1543));
                data.insert("active_sessions".to_string(), serde_json::json!(5));
                data.insert("commands_per_hour".to_string(), serde_json::json!(64.3));
            }
            WidgetType::SuccessRate => {
                data.insert("success_rate".to_string(), serde_json::json!(94.5));
                data.insert("failed_commands".to_string(), serde_json::json!(85));
                data.insert("total_commands".to_string(), serde_json::json!(1543));
            }
            WidgetType::ResponseTime => {
                data.insert("avg_response_ms".to_string(), serde_json::json!(850));
                data.insert("p50_ms".to_string(), serde_json::json!(650));
                data.insert("p95_ms".to_string(), serde_json::json!(1850));
                data.insert("p99_ms".to_string(), serde_json::json!(3200));
            }
        }

        data
    }
}

/// Widget types for dashboard testing
#[derive(Debug, Clone)]
pub enum WidgetType {
    CostSummary,
    CommandActivity,
    SuccessRate,
    ResponseTime,
}

/// Performance measurement utilities
pub mod performance {
    use super::*;
    use std::time::Instant;

    /// Measure dashboard generation performance
    pub async fn measure_dashboard_generation(
        fixture: &DashboardTestFixture,
        iterations: usize,
    ) -> Vec<u128> {
        let mut durations = Vec::new();

        for _ in 0..iterations {
            let start = Instant::now();
            let _ = fixture.dashboard_manager.generate_live_data().await;
            durations.push(start.elapsed().as_micros());
        }

        durations
    }
}

/// Assertion helpers for dashboard testing
pub mod assertions {
    use super::*;

    /// Assert that dashboard data is valid and consistent
    pub fn assert_dashboard_data_valid(data: &DashboardData) {
        assert!(
            data.today_cost >= 0.0,
            "Today's cost should be non-negative"
        );
        assert!(
            data.today_commands >= 0,
            "Today's commands should be non-negative"
        );
        assert!(
            data.success_rate >= 0.0 && data.success_rate <= 100.0,
            "Success rate should be between 0 and 100"
        );
    }
}

/// Performance timer for benchmarking
pub mod perf {
    use std::time::Instant;

    pub struct Timer {
        name: String,
        start: Instant,
    }

    impl Timer {
        pub fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                start: Instant::now(),
            }
        }

        pub fn elapsed_ms(&self) -> u128 {
            self.start.elapsed().as_millis()
        }

        pub fn report(&self) -> String {
            format!("{}: {}ms", self.name, self.elapsed_ms())
        }

        pub fn name(&self) -> &str {
            &self.name
        }
    }

    pub struct PerfCollector {
        measurements: Vec<(String, u128)>,
    }

    impl PerfCollector {
        pub fn new() -> Self {
            Self {
                measurements: Vec::new(),
            }
        }

        pub fn record(&mut self, name: &str, duration_ms: u128) {
            self.measurements.push((name.to_string(), duration_ms));
        }

        pub fn summary(&self) -> PerfSummary {
            let total: u128 = self.measurements.iter().map(|(_, d)| d).sum();
            let count = self.measurements.len();
            let average_ms = if count > 0 { total / count as u128 } else { 0 };
            let max_ms = self.measurements.iter().map(|(_, d)| *d).max().unwrap_or(0);
            let min_ms = self.measurements.iter().map(|(_, d)| *d).min().unwrap_or(0);

            PerfSummary {
                total_ms: total,
                average_ms,
                max_ms,
                min_ms,
                count,
            }
        }
    }

    #[derive(Debug)]
    pub struct PerfSummary {
        pub total_ms: u128,
        pub average_ms: u128,
        pub max_ms: u128,
        pub min_ms: u128,
        pub count: usize,
    }
}
