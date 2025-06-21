//! Shared test utilities for analytics module
//!
//! This module provides common test infrastructure, data generators,
//! and utilities that can be reused across different analytics test modules.

use crate::cli::analytics::metrics::{MetricConfig, MetricsEngine};
use crate::cli::cost::{CostEntry, CostTracker};
use crate::cli::error::Result;
use crate::cli::history::{HistoryEntry, HistoryStore};
use crate::cli::session::SessionId;
use chrono::{DateTime, Duration, Utc};
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::RwLock;

/// Common test data patterns
pub mod patterns {
    /// Command execution patterns for realistic test data
    pub const COMMANDS: &[&str] = &[
        "analyze",
        "summarize",
        "translate",
        "code",
        "explain",
        "chat",
        "review",
        "search",
        "generate",
        "format",
    ];

    /// Model patterns with relative costs
    pub const MODELS: &[(&str, f64)] =
        &[("claude-3-opus", 1.0), ("claude-3-sonnet", 0.2), ("claude-3-haiku", 0.05)];

    /// Time-based activity patterns (hour -> relative activity level)
    pub const HOURLY_ACTIVITY: &[(u32, f64)] = &[
        (0, 0.1),
        (1, 0.1),
        (2, 0.1),
        (3, 0.1),
        (4, 0.1),
        (5, 0.2),
        (6, 0.3),
        (7, 0.5),
        (8, 0.7),
        (9, 1.0),
        (10, 0.9),
        (11, 0.8),
        (12, 0.5),
        (13, 0.6),
        (14, 0.9),
        (15, 1.0),
        (16, 0.8),
        (17, 0.6),
        (18, 0.4),
        (19, 0.3),
        (20, 0.3),
        (21, 0.2),
        (22, 0.2),
        (23, 0.1),
    ];
}

/// Unified test data generator for analytics
pub struct AnalyticsTestDataGenerator {
    pub base_time: DateTime<Utc>,
    pub session_count: usize,
    pub commands_per_session: (usize, usize), // (min, max)
    pub success_rate: f64,
}

impl Default for AnalyticsTestDataGenerator {
    fn default() -> Self {
        Self {
            base_time: Utc::now() - Duration::days(30),
            session_count: 10,
            commands_per_session: (5, 20),
            success_rate: 0.95,
        }
    }
}

impl AnalyticsTestDataGenerator {
    /// Generate comprehensive test data over a time period
    pub fn generate_test_data(&self, days: u32) -> TestDataSet {
        let mut sessions = Vec::new();
        let mut cost_entries = Vec::new();
        let mut history_entries = Vec::new();

        let end_time = Utc::now();
        let start_time = end_time - Duration::days(days as i64);

        for day in 0..days {
            let day_start = start_time + Duration::days(day as i64);
            let sessions_today = 1 + (rand::random::<f64>() * self.session_count as f64) as usize;

            for session_idx in 0..sessions_today {
                let session_id = uuid::Uuid::new_v4();
                let session_hour = 8 + (rand::random::<f64>() * 10.0) as i64;
                let session_start = day_start + Duration::hours(session_hour);

                sessions.push(TestSession {
                    id: session_id,
                    start_time: session_start,
                    command_count: 0,
                });

                // Generate commands for this session
                let command_count = self.commands_per_session.0
                    + (rand::random::<f64>()
                        * (self.commands_per_session.1 - self.commands_per_session.0) as f64)
                        as usize;

                for cmd_idx in 0..command_count {
                    let timestamp = session_start + Duration::minutes(cmd_idx as i64 * 2);
                    let command =
                        patterns::COMMANDS[rand::random::<usize>() % patterns::COMMANDS.len()];
                    let (model, model_cost_factor) =
                        patterns::MODELS[rand::random::<usize>() % patterns::MODELS.len()];

                    // Generate realistic token counts
                    let input_tokens = 100 + (rand::random::<f64>() * 2000.0) as u32;
                    let output_tokens = 200 + (rand::random::<f64>() * 3000.0) as u32;
                    let cost = self.calculate_cost(input_tokens, output_tokens, model_cost_factor);
                    let duration = 500 + (rand::random::<f64>() * 5000.0) as u64;
                    let success = rand::random::<f64>() < self.success_rate;

                    // Create cost entry
                    let mut cost_entry = CostEntry::new(
                        session_id,
                        command.to_string(),
                        cost,
                        input_tokens,
                        output_tokens,
                        duration,
                        model.to_string(),
                    );
                    cost_entry.timestamp = timestamp;
                    cost_entries.push(cost_entry);

                    // Create history entry
                    let mut history_entry = HistoryEntry::new(
                        session_id,
                        command.to_string(),
                        vec!["--option".to_string(), "value".to_string()],
                        if success {
                            format!("Success output for {} command", command)
                        } else {
                            format!("Error: {} command failed", command)
                        },
                        success,
                        duration,
                    );
                    history_entry.timestamp = timestamp;
                    history_entries.push(history_entry);
                }

                // Update session command count
                if let Some(session) = sessions.last_mut() {
                    session.command_count = command_count;
                }
            }
        }

        TestDataSet {
            sessions,
            cost_entries,
            history_entries,
            start_time,
            end_time,
        }
    }

    /// Generate data with specific patterns for testing
    pub fn generate_pattern_data(&self, pattern: DataPattern) -> TestDataSet {
        match pattern {
            DataPattern::HighVolume => self.generate_high_volume_data(),
            DataPattern::BurstyTraffic => self.generate_bursty_traffic_data(),
            DataPattern::GradualGrowth => self.generate_gradual_growth_data(),
            DataPattern::ErrorSpike => self.generate_error_spike_data(),
            DataPattern::ModelMigration => self.generate_model_migration_data(),
        }
    }

    fn calculate_cost(&self, input_tokens: u32, output_tokens: u32, model_factor: f64) -> f64 {
        let base_input_cost = 0.01;
        let base_output_cost = 0.03;
        ((input_tokens as f64 * base_input_cost + output_tokens as f64 * base_output_cost) / 1000.0)
            * model_factor
    }

    fn generate_high_volume_data(&self) -> TestDataSet {
        let generator = Self {
            commands_per_session: self.commands_per_session,
            session_count: self.session_count,
            ..self.clone()
        };
        generator.generate_test_data(1)
    }

    fn generate_bursty_traffic_data(&self) -> TestDataSet {
        // Generate normal traffic with periodic bursts
        let mut data = self.generate_test_data(7);

        // Add burst periods
        for day in [1, 3, 5] {
            let burst_time = self.base_time + Duration::days(day) + Duration::hours(14);
            let session_id = uuid::Uuid::new_v4();

            // Add 100 commands in 30 minutes
            for i in 0..100 {
                let timestamp = burst_time + Duration::seconds((i as i64) * 18);
                let command = patterns::COMMANDS[i % patterns::COMMANDS.len()];

                let mut cost_entry = CostEntry::new(
                    session_id,
                    command.to_string(),
                    0.05,
                    500,
                    1000,
                    200,
                    "claude-3-haiku".to_string(),
                );
                cost_entry.timestamp = timestamp;
                data.cost_entries.push(cost_entry);
            }
        }

        data.update_time_range();
        data
    }

    fn generate_gradual_growth_data(&self) -> TestDataSet {
        let mut all_data = TestDataSet::empty();

        // Generate 30 days with increasing volume
        for week in 0..4 {
            let mut generator = Self {
                base_time: self.base_time + Duration::weeks(week as i64),
                session_count: 5 + week * 3,
                commands_per_session: (5 + week * 2, 20 + week * 5),
                ..self.clone()
            };

            let week_data = generator.generate_test_data(7);
            all_data.merge(week_data);
        }

        all_data
    }

    fn generate_error_spike_data(&self) -> TestDataSet {
        let mut generator = Self {
            success_rate: 0.5, // 50% error rate
            ..self.clone()
        };
        generator.generate_test_data(1)
    }

    fn generate_model_migration_data(&self) -> TestDataSet {
        let mut data = TestDataSet::empty();

        // First week: all opus
        for i in 0..100 {
            let entry = self.create_cost_entry("claude-3-opus", i);
            data.cost_entries.push(entry);
        }

        // Second week: migrating to sonnet
        for i in 0..100 {
            let model = if i < 30 {
                "claude-3-opus"
            } else {
                "claude-3-sonnet"
            };
            let mut entry = self.create_cost_entry(model, 100 + i);
            entry.timestamp = entry.timestamp + Duration::weeks(1);
            data.cost_entries.push(entry);
        }

        // Third week: all sonnet
        for i in 0..100 {
            let mut entry = self.create_cost_entry("claude-3-sonnet", 200 + i);
            entry.timestamp = entry.timestamp + Duration::weeks(2);
            data.cost_entries.push(entry);
        }

        data.update_time_range();
        data
    }

    fn create_cost_entry(&self, model: &str, index: usize) -> CostEntry {
        let mut entry = CostEntry::new(
            uuid::Uuid::new_v4(),
            patterns::COMMANDS[index % patterns::COMMANDS.len()].to_string(),
            0.05,
            500,
            1000,
            1000,
            model.to_string(),
        );
        entry.timestamp = self.base_time + Duration::minutes(index as i64 * 10);
        entry
    }
}

// Make generator cloneable for pattern generation
impl Clone for AnalyticsTestDataGenerator {
    fn clone(&self) -> Self {
        Self {
            base_time: self.base_time,
            session_count: self.session_count,
            commands_per_session: self.commands_per_session,
            success_rate: self.success_rate,
        }
    }
}

/// Data generation patterns for specific test scenarios
#[derive(Debug, Clone)]
pub enum DataPattern {
    HighVolume,
    BurstyTraffic,
    GradualGrowth,
    ErrorSpike,
    ModelMigration,
}

/// Container for generated test data
#[derive(Debug, Clone)]
pub struct TestDataSet {
    pub sessions: Vec<TestSession>,
    pub cost_entries: Vec<CostEntry>,
    pub history_entries: Vec<HistoryEntry>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
}

impl TestDataSet {
    fn empty() -> Self {
        Self {
            sessions: Vec::new(),
            cost_entries: Vec::new(),
            history_entries: Vec::new(),
            start_time: Utc::now(),
            end_time: Utc::now(),
        }
    }

    fn merge(&mut self, other: TestDataSet) {
        self.sessions.extend(other.sessions);
        self.cost_entries.extend(other.cost_entries);
        self.history_entries.extend(other.history_entries);
        self.update_time_range();
    }

    fn update_time_range(&mut self) {
        if let Some(min_time) = self.cost_entries.iter().map(|e| e.timestamp).min() {
            self.start_time = min_time;
        }

        if let Some(max_time) = self.cost_entries.iter().map(|e| e.timestamp).max() {
            self.end_time = max_time;
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestSession {
    pub id: SessionId,
    pub start_time: DateTime<Utc>,
    pub command_count: usize,
}

/// Base test fixture for analytics tests
pub struct AnalyticsTestFixture {
    pub temp_dir: TempDir,
    pub cost_tracker: Arc<RwLock<CostTracker>>,
    pub history_store: Arc<RwLock<HistoryStore>>,
    pub data_generator: AnalyticsTestDataGenerator,
}

impl AnalyticsTestFixture {
    /// Create a new test fixture
    pub async fn new() -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let cost_path = temp_dir.path().join("costs.json");
        let history_path = temp_dir.path().join("history.json");

        let cost_tracker = Arc::new(RwLock::new(CostTracker::new(cost_path)?));
        let history_store = Arc::new(RwLock::new(HistoryStore::new(history_path)?));
        let data_generator = AnalyticsTestDataGenerator::default();

        Ok(Self {
            temp_dir,
            cost_tracker,
            history_store,
            data_generator,
        })
    }

    /// Load test data into stores
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

    /// Get paths for test files
    pub fn test_paths(&self) -> TestPaths {
        TestPaths {
            temp_dir: self.temp_dir.path().to_path_buf(),
            costs_file: self.temp_dir.path().join("costs.json"),
            history_file: self.temp_dir.path().join("history.json"),
            reports_dir: self.temp_dir.path().join("reports"),
        }
    }
}

pub struct TestPaths {
    pub temp_dir: PathBuf,
    pub costs_file: PathBuf,
    pub history_file: PathBuf,
    pub reports_dir: PathBuf,
}

/// Performance measurement utilities
pub mod perf {
    use std::time::Instant;

    /// Measure operation duration
    pub struct Timer {
        start: Instant,
        name: String,
    }

    impl Timer {
        pub fn new(name: &str) -> Self {
            Self {
                start: Instant::now(),
                name: name.to_string(),
            }
        }

        pub fn elapsed_ms(&self) -> u128 {
            self.start.elapsed().as_millis()
        }

        pub fn name(&self) -> &str {
            &self.name
        }

        pub fn report(&self) -> String {
            format!("{}: {}ms", self.name, self.elapsed_ms())
        }
    }

    /// Collect performance metrics
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
            if self.measurements.is_empty() {
                return PerfSummary::default();
            }

            let total: u128 = self.measurements.iter().map(|(_, d)| d).sum();
            let count = self.measurements.len();
            let average = total / count as u128;

            let mut sorted: Vec<u128> = self.measurements.iter().map(|(_, d)| *d).collect();
            sorted.sort();

            PerfSummary {
                total_ms: total,
                average_ms: average,
                min_ms: sorted[0],
                max_ms: sorted[sorted.len() - 1],
                p50_ms: sorted[count / 2],
                p95_ms: sorted[(count as f64 * 0.95) as usize],
                count,
            }
        }
    }

    #[derive(Debug, Default)]
    pub struct PerfSummary {
        pub total_ms: u128,
        pub average_ms: u128,
        pub min_ms: u128,
        pub max_ms: u128,
        pub p50_ms: u128,
        pub p95_ms: u128,
        pub count: usize,
    }
}

/// Random data generation utilities
pub mod random {
    use rand::Rng;

    /// Generate random cost between min and max
    pub fn cost(min: f64, max: f64) -> f64 {
        let mut rng = rand::thread_rng();
        rng.gen_range(min..max)
    }

    /// Generate random token count
    pub fn tokens(min: u32, max: u32) -> u32 {
        let mut rng = rand::thread_rng();
        rng.gen_range(min..max)
    }

    /// Generate random duration in milliseconds
    pub fn duration_ms(min: u64, max: u64) -> u64 {
        let mut rng = rand::thread_rng();
        rng.gen_range(min..max)
    }

    /// Generate random success based on rate
    pub fn success(rate: f64) -> bool {
        let mut rng = rand::thread_rng();
        rng.gen::<f64>() < rate
    }

    /// Select random element from slice
    pub fn choose<'a, T>(items: &'a [T]) -> &'a T {
        let mut rng = rand::thread_rng();
        &items[rng.gen_range(0..items.len())]
    }
}

/// Create a test metrics engine for testing purposes
pub async fn create_test_metrics_engine() -> Result<MetricsEngine> {
    let config = MetricConfig::default();
    let engine = MetricsEngine::new(config);
    engine.initialize().await?;
    Ok(engine)
}

/// Create a test report manager for testing purposes
pub async fn create_test_report_manager() -> Result<crate::cli::analytics::reports::ReportManager> {
    use crate::cli::analytics::reports::{ReportConfig, ReportManager};
    use crate::cli::analytics::{AnalyticsConfig, AnalyticsEngine};

    let temp_dir = TempDir::new()?;
    let cost_path = temp_dir.path().join("costs.json");
    let history_path = temp_dir.path().join("history.json");
    let reports_dir = temp_dir.path().join("reports");

    tokio::fs::create_dir_all(&reports_dir).await?;

    let cost_tracker = Arc::new(RwLock::new(CostTracker::new(cost_path)?));
    let history_store = Arc::new(RwLock::new(HistoryStore::new(history_path)?));

    let analytics_config = AnalyticsConfig::default();
    let analytics_engine = AnalyticsEngine::new(
        Arc::clone(&cost_tracker),
        Arc::clone(&history_store),
        analytics_config,
    );

    let report_config = ReportConfig {
        output_directory: reports_dir,
        ..Default::default()
    };

    Ok(ReportManager::new(analytics_engine, report_config))
}
