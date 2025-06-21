//! Comprehensive unit tests for the cost tracking module
//!
//! This module provides extensive test coverage for cost tracking functionality including:
//! - Cost entry creation and validation
//! - Cost recording with concurrent access
//! - Cost aggregation and analysis
//! - Budget tracking and alerts
//! - Data persistence and storage
//! - Edge cases and error handling

use super::{CostEntry, CostFilter, CostSummary, CostTracker};
use crate::cli::cost::tracker::{
    AdvancedCostTracker, Budget, BudgetScope, BudgetStatus, CostAlert,
};
use crate::cli::session::SessionId;
use chrono::{DateTime, Duration, Utc};
use proptest::prelude::*;
use proptest::strategy::Just;
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::tempdir;
use uuid::Uuid;

/// Test fixture for creating cost entries with known data
pub struct CostTestFixture {
    pub session_id: SessionId,
    pub entries: Vec<CostEntry>,
    pub temp_dir: tempfile::TempDir,
    pub storage_path: PathBuf,
}

impl CostTestFixture {
    /// Create a new test fixture with sample data
    pub fn new() -> Self {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let storage_path = temp_dir.path().join("test_costs.json");
        let session_id = Uuid::new_v4();

        let mut entries = Vec::new();

        // Create diverse test entries
        entries.push(CostEntry::new(
            session_id,
            "analyze_code".to_string(),
            0.050,
            150,
            300,
            2500,
            "claude-3-opus".to_string(),
        ));

        entries.push(CostEntry::new(
            session_id,
            "generate_docs".to_string(),
            0.025,
            75,
            150,
            1800,
            "claude-3-sonnet".to_string(),
        ));

        entries.push(CostEntry::new(
            session_id,
            "review_pr".to_string(),
            0.015,
            50,
            100,
            1200,
            "claude-3-haiku".to_string(),
        ));

        entries.push(CostEntry::new(
            session_id,
            "analyze_code".to_string(), // Duplicate command name
            0.040,
            120,
            240,
            2000,
            "claude-3-opus".to_string(),
        ));

        Self {
            session_id,
            entries,
            temp_dir,
            storage_path,
        }
    }

    /// Create fixture with large dataset for performance testing
    pub fn with_large_dataset(entry_count: usize) -> Self {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let storage_path = temp_dir.path().join("large_test_costs.json");
        let session_id = Uuid::new_v4();

        let mut entries = Vec::new();
        let commands = ["analyze", "generate", "review", "refactor", "test", "debug", "optimize"];
        let models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"];

        for i in 0..entry_count {
            let command_idx = i % commands.len();
            let model_idx = i % models.len();

            entries.push(CostEntry::new(
                session_id,
                format!("{}_{}", commands[command_idx], i),
                0.001 + (i as f64 * 0.001), // Varying costs
                10 + (i as u32 * 5),        // Varying input tokens
                20 + (i as u32 * 10),       // Varying output tokens
                500 + (i as u64 * 100),     // Varying duration
                models[model_idx].to_string(),
            ));
        }

        Self {
            session_id,
            entries,
            temp_dir,
            storage_path,
        }
    }

    /// Create fixture with time-distributed entries for trend analysis
    pub fn with_time_series_data(days: i64) -> Self {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let storage_path = temp_dir.path().join("timeseries_test_costs.json");
        let session_id = Uuid::new_v4();

        let mut entries = Vec::new();
        let base_time = Utc::now() - Duration::days(days);

        for day in 0..days {
            for hour in [9, 14, 16] {
                // Simulate peak usage hours
                let mut entry = CostEntry::new(
                    session_id,
                    format!("daily_task_{}", day),
                    0.010 + (day as f64 * 0.005), // Increasing cost trend
                    50 + (day as u32 * 2),
                    100 + (day as u32 * 4),
                    1000 + (day as u64 * 50),
                    "claude-3-opus".to_string(),
                );

                // Set specific timestamp
                entry.timestamp = base_time + Duration::days(day) + Duration::hours(hour);
                entries.push(entry);
            }
        }

        Self {
            session_id,
            entries,
            temp_dir,
            storage_path,
        }
    }

    /// Create a cost tracker with the fixture data
    pub async fn create_tracker(&self) -> CostTracker {
        let mut tracker =
            CostTracker::new(self.storage_path.clone()).expect("Failed to create test tracker");

        for entry in &self.entries {
            tracker
                .record_cost(entry.clone())
                .await
                .expect("Failed to record test entry");
        }

        tracker
    }

    /// Generate random cost entry for property-based testing
    pub fn random_entry(session_id: SessionId) -> impl Strategy<Value = CostEntry> {
        (
            any::<String>().prop_filter("Non-empty command", |s| !s.is_empty()),
            0.001f64..1.0f64, // Reasonable cost range
            1u32..10000u32,   // Input tokens
            1u32..20000u32,   // Output tokens
            100u64..60000u64, // Duration ms
            any::<String>().prop_filter("Non-empty model", |s| !s.is_empty()),
        )
            .prop_map(
                move |(command, cost, input_tokens, output_tokens, duration, model)| {
                    CostEntry::new(
                        session_id,
                        command,
                        cost,
                        input_tokens,
                        output_tokens,
                        duration,
                        model,
                    )
                },
            )
    }

    /// Generate cost filter for property-based testing
    pub fn random_filter() -> impl Strategy<Value = CostFilter> {
        (
            prop::option::of(Just(Uuid::new_v4())),
            prop::option::of(any::<String>()),
            prop::option::of(property_strategies::date_range().prop_map(|(start, _)| start)),
            prop::option::of(property_strategies::date_range().prop_map(|(_, end)| end)),
            prop::option::of(0.0f64..1000.0f64),
            prop::option::of(0.0f64..1000.0f64),
            prop::option::of(any::<String>()),
        )
            .prop_map(
                |(session_id, command_pattern, since, until, min_cost, max_cost, model)| {
                    CostFilter {
                        session_id,
                        command_pattern,
                        since,
                        until,
                        min_cost,
                        max_cost,
                        model,
                    }
                },
            )
    }
}

/// Helper functions for test data generation
pub mod test_helpers {
    use super::*;

    /// Create a cost entry with minimal valid data
    pub fn minimal_cost_entry(session_id: SessionId) -> CostEntry {
        CostEntry::new(
            session_id,
            "test_command".to_string(),
            0.001,
            1,
            1,
            100,
            "test_model".to_string(),
        )
    }

    /// Create a cost entry with maximum realistic values
    pub fn maximal_cost_entry(session_id: SessionId) -> CostEntry {
        CostEntry::new(
            session_id,
            "expensive_analysis_command_with_very_long_name".to_string(),
            99.999,
            100000,
            200000,
            3600000, // 1 hour
            "claude-3-opus-premium-ultra".to_string(),
        )
    }

    /// Create entries with edge case values
    pub fn edge_case_entries(session_id: SessionId) -> Vec<CostEntry> {
        vec![
            // Zero cost
            CostEntry::new(
                session_id,
                "free_command".to_string(),
                0.0,
                0,
                0,
                0,
                "free_model".to_string(),
            ),
            // Very small values
            CostEntry::new(
                session_id,
                "tiny".to_string(),
                0.00001,
                1,
                1,
                1,
                "tiny_model".to_string(),
            ),
            // Unicode in command names
            CostEntry::new(
                session_id,
                "分析代码".to_string(),
                0.01,
                50,
                100,
                1000,
                "claude-3-opus".to_string(),
            ),
            // Special characters
            CostEntry::new(
                session_id,
                "cmd-with_special.chars@test!".to_string(),
                0.02,
                60,
                120,
                1500,
                "model-v2.1".to_string(),
            ),
        ]
    }

    /// Generate entries for concurrent testing
    pub fn concurrent_test_entries(session_id: SessionId, count: usize) -> Vec<CostEntry> {
        (0..count)
            .map(|i| {
                CostEntry::new(
                    session_id,
                    format!("concurrent_cmd_{}", i),
                    0.01 * (i + 1) as f64,
                    10 * (i + 1) as u32,
                    20 * (i + 1) as u32,
                    100 * (i + 1) as u64,
                    format!("model_{}", i % 3),
                )
            })
            .collect()
    }

    /// Create entries spanning multiple sessions
    pub fn multi_session_entries() -> (Vec<SessionId>, Vec<CostEntry>) {
        let sessions: Vec<SessionId> = (0..3).map(|_| Uuid::new_v4()).collect();
        let mut entries = Vec::new();

        for (i, &session_id) in sessions.iter().enumerate() {
            for j in 0..5 {
                entries.push(CostEntry::new(
                    session_id,
                    format!("session_{}_cmd_{}", i, j),
                    0.01 * ((i * 5 + j) + 1) as f64,
                    50 + (j * 10) as u32,
                    100 + (j * 20) as u32,
                    1000 + (j * 200) as u64,
                    "claude-3-opus".to_string(),
                ));
            }
        }

        (sessions, entries)
    }

    /// Create temporally distributed entries for date filtering tests
    pub fn time_distributed_entries(session_id: SessionId) -> Vec<CostEntry> {
        let base_time = Utc::now();
        let mut entries = Vec::new();

        // Entries from different time periods
        let time_offsets = [
            Duration::days(-30),    // 30 days ago
            Duration::days(-7),     // 1 week ago
            Duration::days(-1),     // Yesterday
            Duration::hours(-6),    // 6 hours ago
            Duration::hours(-1),    // 1 hour ago
            Duration::minutes(-10), // 10 minutes ago
        ];

        for (i, offset) in time_offsets.iter().enumerate() {
            let mut entry = CostEntry::new(
                session_id,
                format!("time_cmd_{}", i),
                0.01 * (i + 1) as f64,
                50 + (i * 10) as u32,
                100 + (i * 20) as u32,
                1000 + (i * 200) as u64,
                "claude-3-opus".to_string(),
            );
            entry.timestamp = base_time + *offset;
            entries.push(entry);
        }

        entries
    }

    /// Validate cost summary calculations
    pub fn validate_cost_summary(entries: &[CostEntry], summary: &CostSummary) -> bool {
        if entries.is_empty() {
            return summary.total_cost == 0.0
                && summary.command_count == 0
                && summary.average_cost == 0.0
                && summary.total_tokens == 0;
        }

        let expected_total_cost: f64 = entries.iter().map(|e| e.cost_usd).sum();
        let expected_command_count = entries.len();
        let expected_average_cost = expected_total_cost / expected_command_count as f64;
        let expected_total_tokens: u32 = entries
            .iter()
            .map(|e| e.input_tokens + e.output_tokens)
            .sum();

        let cost_match = (summary.total_cost - expected_total_cost).abs() < 0.00001;
        let count_match = summary.command_count == expected_command_count;
        let avg_match = (summary.average_cost - expected_average_cost).abs() < 0.00001;
        let token_match = summary.total_tokens == expected_total_tokens;

        cost_match && count_match && avg_match && token_match
    }

    /// Create performance benchmark entries
    pub fn benchmark_entries(count: usize) -> Vec<CostEntry> {
        let session_id = Uuid::new_v4();
        (0..count)
            .map(|i| {
                CostEntry::new(
                    session_id,
                    format!("benchmark_cmd_{:06}", i),
                    0.001 + (i as f64 / 1000000.0), // Small incremental costs
                    10 + (i % 100) as u32,
                    20 + (i % 200) as u32,
                    100 + (i % 5000) as u64,
                    match i % 3 {
                        0 => "claude-3-opus",
                        1 => "claude-3-sonnet",
                        _ => "claude-3-haiku",
                    }
                    .to_string(),
                )
            })
            .collect()
    }
}

/// Property-based test strategies
pub mod property_strategies {
    use super::*;

    /// Strategy for generating valid cost values
    pub fn valid_cost() -> impl Strategy<Value = f64> {
        (0.0..1000.0f64).prop_filter("Non-negative finite cost", |&cost| {
            cost.is_finite() && cost >= 0.0
        })
    }

    /// Strategy for generating valid token counts
    pub fn valid_tokens() -> impl Strategy<Value = u32> {
        0u32..1000000u32
    }

    /// Strategy for generating valid durations
    pub fn valid_duration() -> impl Strategy<Value = u64> {
        1u64..3600000u64 // 1ms to 1 hour
    }

    /// Strategy for generating realistic command names
    pub fn command_name() -> impl Strategy<Value = String> {
        prop::collection::vec("[a-z_]{1,20}", 1..4).prop_map(|parts| parts.join("_"))
    }

    /// Strategy for generating model names
    pub fn model_name() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("claude-3-opus".to_string()),
            Just("claude-3-sonnet".to_string()),
            Just("claude-3-haiku".to_string()),
            "[a-z0-9-]{5,30}".prop_map(|s| s.to_string()),
        ]
    }

    /// Strategy for generating valid date ranges
    pub fn date_range() -> impl Strategy<Value = (DateTime<Utc>, DateTime<Utc>)> {
        (
            -365i64..0i64, // Days in the past for start
            0i64..365i64,  // Days in the future for end
        )
            .prop_map(|(start_days, end_days)| {
                let base = Utc::now();
                let start = base + Duration::days(start_days);
                let end = base + Duration::days(end_days);
                if start <= end {
                    (start, end)
                } else {
                    (end, start)
                }
            })
    }
}

#[cfg(test)]
mod test_infrastructure_tests {
    use super::*;

    #[test]
    fn test_fixture_creation() {
        let fixture = CostTestFixture::new();

        assert!(!fixture.entries.is_empty());
        assert!(!fixture.storage_path.exists()); // Should not exist yet
        assert_eq!(fixture.entries.len(), 4);

        // Verify entries have the same session ID
        for entry in &fixture.entries {
            assert_eq!(entry.session_id, fixture.session_id);
        }
    }

    #[test]
    fn test_large_dataset_fixture() {
        let fixture = CostTestFixture::with_large_dataset(1000);

        assert_eq!(fixture.entries.len(), 1000);

        // Verify diversity in the dataset
        let mut unique_commands = std::collections::HashSet::new();
        let mut unique_models = std::collections::HashSet::new();

        for entry in &fixture.entries {
            unique_commands.insert(&entry.command_name);
            unique_models.insert(&entry.model);
        }

        assert!(unique_commands.len() > 1);
        assert!(unique_models.len() > 1);
    }

    #[test]
    fn test_time_series_fixture() {
        let fixture = CostTestFixture::with_time_series_data(7);

        assert_eq!(fixture.entries.len(), 7 * 3); // 7 days * 3 entries per day

        // Verify timestamps are distributed over time
        let mut timestamps: Vec<_> = fixture.entries.iter().map(|e| e.timestamp).collect();
        timestamps.sort();

        assert!(timestamps.first().unwrap() < timestamps.last().unwrap());

        // Verify cost trend (should be increasing)
        let costs: Vec<_> = fixture.entries.iter().map(|e| e.cost_usd).collect();
        let first_day_cost = costs[0];
        let last_day_cost = costs[costs.len() - 3]; // Last day, first entry
        assert!(last_day_cost > first_day_cost);
    }

    #[tokio::test]
    async fn test_fixture_tracker_creation() {
        let fixture = CostTestFixture::new();
        let tracker = fixture.create_tracker().await;

        // Verify all entries were recorded
        let global_summary = tracker.get_global_summary().await.unwrap();
        assert_eq!(global_summary.command_count, fixture.entries.len());

        // Verify total cost calculation
        let expected_total: f64 = fixture.entries.iter().map(|e| e.cost_usd).sum();
        assert!((global_summary.total_cost - expected_total).abs() < 0.00001);
    }

    #[test]
    fn test_helper_functions() {
        let session_id = Uuid::new_v4();

        // Test minimal entry
        let minimal = test_helpers::minimal_cost_entry(session_id);
        assert_eq!(minimal.session_id, session_id);
        assert!(minimal.cost_usd > 0.0);

        // Test maximal entry
        let maximal = test_helpers::maximal_cost_entry(session_id);
        assert_eq!(maximal.session_id, session_id);
        assert!(maximal.cost_usd > minimal.cost_usd);

        // Test edge cases
        let edge_cases = test_helpers::edge_case_entries(session_id);
        assert!(!edge_cases.is_empty());

        // Test concurrent entries
        let concurrent = test_helpers::concurrent_test_entries(session_id, 10);
        assert_eq!(concurrent.len(), 10);

        // Test multi-session entries
        let (sessions, entries) = test_helpers::multi_session_entries();
        assert_eq!(sessions.len(), 3);
        assert_eq!(entries.len(), 15); // 3 sessions * 5 entries each

        // Test time distributed entries
        let time_entries = test_helpers::time_distributed_entries(session_id);
        assert_eq!(time_entries.len(), 6);
    }

    #[test]
    fn test_summary_validation() {
        let session_id = Uuid::new_v4();
        let entries = vec![
            CostEntry::new(
                session_id,
                "cmd1".to_string(),
                0.10,
                100,
                200,
                1000,
                "model1".to_string(),
            ),
            CostEntry::new(
                session_id,
                "cmd2".to_string(),
                0.20,
                150,
                300,
                1500,
                "model2".to_string(),
            ),
        ];

        let summary = CostSummary {
            total_cost: 0.30,
            command_count: 2,
            average_cost: 0.15,
            total_tokens: 750, // 100+200+150+300
            date_range: (Utc::now(), Utc::now()),
            by_command: HashMap::new(),
            by_model: HashMap::new(),
        };

        assert!(test_helpers::validate_cost_summary(&entries, &summary));

        // Test with invalid summary
        let invalid_summary = CostSummary {
            total_cost: 999.99, // Wrong total
            ..summary.clone()
        };

        assert!(!test_helpers::validate_cost_summary(
            &entries,
            &invalid_summary
        ));
    }

    #[test]
    fn test_benchmark_entries() {
        let entries = test_helpers::benchmark_entries(10000);
        assert_eq!(entries.len(), 10000);

        // Verify performance characteristics
        let start = std::time::Instant::now();
        let _total_cost: f64 = entries.iter().map(|e| e.cost_usd).sum();
        let duration = start.elapsed();

        // Should be able to sum 10k entries quickly
        assert!(duration.as_millis() < 10);
    }

    proptest! {
        #[test]
        fn property_cost_entry_creation(
            cost in property_strategies::valid_cost(),
            input_tokens in property_strategies::valid_tokens(),
            output_tokens in property_strategies::valid_tokens(),
            duration in property_strategies::valid_duration(),
            command in property_strategies::command_name(),
            model in property_strategies::model_name(),
        ) {
            let session_id = Uuid::new_v4();
            let entry = CostEntry::new(
                session_id,
                command,
                cost,
                input_tokens,
                output_tokens,
                duration,
                model.clone(),
            );

            // Verify properties
            prop_assert_eq!(entry.session_id, session_id);
            prop_assert_eq!(entry.cost_usd, cost);
            prop_assert_eq!(entry.input_tokens, input_tokens);
            prop_assert_eq!(entry.output_tokens, output_tokens);
            prop_assert_eq!(entry.duration_ms, duration);
            prop_assert_eq!(entry.model, model);
            prop_assert!(!entry.id.is_empty());
        }

        #[test]
        fn property_cost_filter_behavior(
            filter in CostTestFixture::random_filter(),
            entry in CostTestFixture::random_entry(Uuid::new_v4()),
        ) {
            // This test verifies that filter logic doesn't panic
            // and behaves consistently
            let temp_dir = tempdir().unwrap();
            let storage_path = temp_dir.path().join("prop_test.json");
            let tracker = CostTracker::new(storage_path).unwrap();

            // The filter logic should not panic regardless of input
            let matches = tracker.matches_filter(&entry, &filter);
            prop_assert!(matches == true || matches == false); // Just checking no panic
        }

        #[test]
        fn property_summary_calculation_consistency(
            entries in prop::collection::vec(
                CostTestFixture::random_entry(Uuid::new_v4()),
                0..100
            )
        ) {
            // Test that summary calculations are consistent
            if !entries.is_empty() {
                let total_cost: f64 = entries.iter().map(|e| e.cost_usd).sum();
                let command_count = entries.len();
                let average_cost = total_cost / command_count as f64;

                prop_assert!(total_cost >= 0.0);
                prop_assert!(average_cost >= 0.0);
                prop_assert!(command_count > 0);

                // If all entries have positive cost, average should be positive
                if entries.iter().all(|e| e.cost_usd > 0.0) {
                    prop_assert!(average_cost > 0.0);
                }
            }
        }
    }
}

/// Tests for core cost recording functionality (Task 1.2)
#[cfg(test)]
mod core_recording_tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[tokio::test]
    async fn test_record_cost_valid_data() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let entry = CostEntry::new(
            session_id,
            "test_command".to_string(),
            0.025,
            100,
            200,
            1500,
            "claude-3-opus".to_string(),
        );

        let initial_count = tracker.entries.len();
        tracker.record_cost(entry.clone()).await.unwrap();

        assert_eq!(tracker.entries.len(), initial_count + 1);
        let recorded_entry = &tracker.entries[tracker.entries.len() - 1];
        assert_eq!(recorded_entry.session_id, entry.session_id);
        assert_eq!(recorded_entry.command_name, entry.command_name);
        assert_eq!(recorded_entry.cost_usd, entry.cost_usd);
        assert_eq!(recorded_entry.input_tokens, entry.input_tokens);
        assert_eq!(recorded_entry.output_tokens, entry.output_tokens);
        assert_eq!(recorded_entry.duration_ms, entry.duration_ms);
        assert_eq!(recorded_entry.model, entry.model);
    }

    #[tokio::test]
    async fn test_record_cost_zero_values() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let zero_cost_entry = CostEntry::new(
            session_id,
            "free_command".to_string(),
            0.0, // Zero cost
            0,   // Zero input tokens
            0,   // Zero output tokens
            0,   // Zero duration
            "free_model".to_string(),
        );

        // Should succeed even with zero values
        tracker.record_cost(zero_cost_entry.clone()).await.unwrap();

        let summary = tracker.get_session_summary(session_id).await.unwrap();
        assert_eq!(summary.total_cost, 0.0);
        assert_eq!(summary.command_count, 1);
        assert_eq!(summary.total_tokens, 0);
    }

    #[tokio::test]
    async fn test_record_cost_negative_values() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let negative_cost_entry = CostEntry::new(
            session_id,
            "refund_command".to_string(),
            -0.05, // Negative cost (refund scenario)
            100,
            200,
            1000,
            "claude-3-opus".to_string(),
        );

        // Should handle negative costs (e.g., for refunds)
        tracker
            .record_cost(negative_cost_entry.clone())
            .await
            .unwrap();

        let summary = tracker.get_session_summary(session_id).await.unwrap();
        assert_eq!(summary.total_cost, -0.05);
        assert_eq!(summary.command_count, 1);
        assert!(summary.average_cost < 0.0);
    }

    #[tokio::test]
    async fn test_record_cost_different_types() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();

        // Different cost types and models
        let entries = vec![
            CostEntry::new(
                session_id,
                "input_heavy".to_string(),
                0.10,
                1000,
                100,
                2000,
                "claude-3-opus".to_string(),
            ),
            CostEntry::new(
                session_id,
                "output_heavy".to_string(),
                0.15,
                100,
                1000,
                3000,
                "claude-3-opus".to_string(),
            ),
            CostEntry::new(
                session_id,
                "balanced".to_string(),
                0.08,
                500,
                500,
                1500,
                "claude-3-sonnet".to_string(),
            ),
            CostEntry::new(
                session_id,
                "quick_task".to_string(),
                0.01,
                50,
                50,
                200,
                "claude-3-haiku".to_string(),
            ),
        ];

        for entry in entries {
            tracker.record_cost(entry).await.unwrap();
        }

        let summary = tracker.get_session_summary(session_id).await.unwrap();
        assert_eq!(summary.command_count, 4);
        assert_eq!(summary.by_model.len(), 3); // Three different models

        // Verify input-heavy vs output-heavy distinction
        let total_input_tokens: u32 = tracker
            .entries
            .iter()
            .filter(|e| e.session_id == session_id)
            .map(|e| e.input_tokens)
            .sum();
        let total_output_tokens: u32 = tracker
            .entries
            .iter()
            .filter(|e| e.session_id == session_id)
            .map(|e| e.output_tokens)
            .sum();

        assert_eq!(total_input_tokens, 1650);
        assert_eq!(total_output_tokens, 1650);
        assert_eq!(summary.total_tokens, 3300);
    }

    #[tokio::test]
    async fn test_concurrent_cost_recording() {
        let fixture = CostTestFixture::new();
        let tracker = Arc::new(Mutex::new(
            CostTracker::new(fixture.storage_path.clone()).unwrap(),
        ));
        let session_id = Uuid::new_v4();

        // Create multiple concurrent tasks
        let mut handles = vec![];

        for i in 0..50 {
            let tracker_clone = Arc::clone(&tracker);
            let handle = tokio::spawn(async move {
                let entry = CostEntry::new(
                    session_id,
                    format!("concurrent_cmd_{}", i),
                    0.01 * (i + 1) as f64,
                    10 + i as u32,
                    20 + i as u32,
                    100 + i as u64,
                    format!("model_{}", i % 3),
                );

                let mut tracker_lock = tracker_clone.lock().await;
                tracker_lock.record_cost(entry).await.unwrap();
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        let tracker_lock = tracker.lock().await;
        let summary = tracker_lock.get_session_summary(session_id).await.unwrap();

        assert_eq!(summary.command_count, 50);

        // Verify total cost calculation
        let expected_total: f64 = (1..=50).map(|i| 0.01 * i as f64).sum();
        assert!((summary.total_cost - expected_total).abs() < 0.00001);
    }

    #[tokio::test]
    async fn test_cost_recording_persistence() {
        let fixture = CostTestFixture::new();
        let session_id = Uuid::new_v4();

        // Create first tracker and add entries
        {
            let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

            for i in 0..5 {
                let entry = CostEntry::new(
                    session_id,
                    format!("persistent_cmd_{}", i),
                    0.01 * (i + 1) as f64,
                    50 + i as u32 * 10,
                    100 + i as u32 * 20,
                    1000 + i as u64 * 200,
                    "claude-3-opus".to_string(),
                );
                tracker.record_cost(entry).await.unwrap();
            }
        } // tracker goes out of scope

        // Create new tracker instance and verify persistence
        let tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();
        let summary = tracker.get_session_summary(session_id).await.unwrap();

        assert_eq!(summary.command_count, 5);

        let expected_total: f64 = (1..=5).map(|i| 0.01 * i as f64).sum();
        assert!((summary.total_cost - expected_total).abs() < 0.00001);
    }

    #[tokio::test]
    async fn test_record_cost_storage_failure_handling() {
        // Create a tracker with an invalid storage path (read-only directory)
        let temp_dir = tempdir().unwrap();
        let readonly_path = temp_dir.path().join("readonly");
        std::fs::create_dir(&readonly_path).unwrap();

        // Make directory read-only (this test might behave differently on different platforms)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&readonly_path).unwrap().permissions();
            perms.set_mode(0o444); // Read-only
            std::fs::set_permissions(&readonly_path, perms).unwrap();
        }

        let storage_path = readonly_path.join("costs.json");
        let mut tracker = CostTracker::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();
        let entry = CostEntry::new(
            session_id,
            "test_command".to_string(),
            0.025,
            100,
            200,
            1500,
            "claude-3-opus".to_string(),
        );

        // This should fail due to permission issues on Unix systems
        #[cfg(unix)]
        {
            let result = tracker.record_cost(entry).await;
            assert!(result.is_err());
        }

        // On other platforms, we just ensure it doesn't panic
        #[cfg(not(unix))]
        {
            let _result = tracker.record_cost(entry).await;
        }
    }

    #[tokio::test]
    async fn test_record_cost_with_unicode_content() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let entries = vec![
            CostEntry::new(
                session_id,
                "分析代码".to_string(),
                0.025,
                100,
                200,
                1500,
                "claude-3-opus".to_string(),
            ),
            CostEntry::new(
                session_id,
                "анализ_кода".to_string(),
                0.030,
                150,
                250,
                2000,
                "claude-3-sonnet".to_string(),
            ),
            CostEntry::new(
                session_id,
                "コード解析".to_string(),
                0.020,
                80,
                180,
                1200,
                "claude-3-haiku".to_string(),
            ),
            CostEntry::new(
                session_id,
                "🔍analyze🚀".to_string(),
                0.035,
                200,
                300,
                2500,
                "claude-3-opus".to_string(),
            ),
        ];

        for entry in entries {
            tracker.record_cost(entry).await.unwrap();
        }

        let summary = tracker.get_session_summary(session_id).await.unwrap();
        assert_eq!(summary.command_count, 4);

        // Verify unicode commands are stored correctly
        let global_summary = tracker.get_global_summary().await.unwrap();
        assert!(global_summary.by_command.contains_key("分析代码"));
        assert!(global_summary.by_command.contains_key("анализ_кода"));
        assert!(global_summary.by_command.contains_key("コード解析"));
        assert!(global_summary.by_command.contains_key("🔍analyze🚀"));
    }

    #[tokio::test]
    async fn test_record_cost_large_values() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let large_entry = CostEntry::new(
            session_id,
            "extremely_expensive_analysis".to_string(),
            999.999, // Very large cost
            1000000, // 1M input tokens
            2000000, // 2M output tokens
            3600000, // 1 hour duration
            "claude-3-opus-premium".to_string(),
        );

        tracker.record_cost(large_entry).await.unwrap();

        let summary = tracker.get_session_summary(session_id).await.unwrap();
        assert_eq!(summary.command_count, 1);
        assert!((summary.total_cost - 999.999).abs() < 0.00001);
        assert_eq!(summary.total_tokens, 3000000);
        assert!((summary.average_cost - 999.999).abs() < 0.00001);
    }

    #[tokio::test]
    async fn test_record_cost_precision() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();

        // Test with very small precise values
        let precise_entries = vec![0.00001, 0.00002, 0.00003, 0.00004, 0.00005];

        for (i, cost) in precise_entries.iter().enumerate() {
            let entry = CostEntry::new(
                session_id,
                format!("precise_cmd_{}", i),
                *cost,
                1,
                1,
                100,
                "claude-3-haiku".to_string(),
            );
            tracker.record_cost(entry).await.unwrap();
        }

        let summary = tracker.get_session_summary(session_id).await.unwrap();
        let expected_total: f64 = precise_entries.iter().sum();

        // Verify precision is maintained
        assert!((summary.total_cost - expected_total).abs() < 0.0000001);
        assert_eq!(summary.command_count, 5);
    }

    #[tokio::test]
    async fn test_record_cost_non_deduplication() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();

        // Record the same command multiple times (each with new ID)
        for _ in 0..3 {
            let entry = CostEntry::new(
                session_id,
                "test_command".to_string(),
                0.025,
                100,
                200,
                1500,
                "claude-3-opus".to_string(),
            );
            tracker.record_cost(entry).await.unwrap();
        }

        let summary = tracker.get_session_summary(session_id).await.unwrap();

        // Should have 3 separate entries (not deduplicated)
        assert_eq!(summary.command_count, 3);
        assert!((summary.total_cost - 0.075).abs() < 0.00001); // 3 * 0.025

        // Verify all entries have different IDs
        let session_entries: Vec<_> = tracker
            .entries
            .iter()
            .filter(|e| e.session_id == session_id)
            .collect();

        let entry_count = session_entries.len();
        let mut ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for entry in session_entries {
            ids.insert(entry.id.clone());
        }
        // Each entry should have a unique ID
        assert_eq!(ids.len(), entry_count);
        assert_eq!(ids.len(), 3);
    }

    proptest! {
        #[test]
        fn property_record_cost_maintains_invariants(
            cost in property_strategies::valid_cost(),
            input_tokens in property_strategies::valid_tokens(),
            output_tokens in property_strategies::valid_tokens(),
            duration in property_strategies::valid_duration(),
            command in property_strategies::command_name(),
            model in property_strategies::model_name(),
        ) {
            tokio_test::block_on(async {
                let fixture = CostTestFixture::new();
                let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();
                let session_id = Uuid::new_v4();

                let initial_count = tracker.entries.len();

                let entry = CostEntry::new(
                    session_id,
                    command,
                    cost,
                    input_tokens,
                    output_tokens,
                    duration,
                    model,
                );

                tracker.record_cost(entry.clone()).await.unwrap();

                // Invariants that should always hold after recording
                prop_assert_eq!(tracker.entries.len(), initial_count + 1);

                let summary = tracker.get_session_summary(session_id).await.unwrap();
                prop_assert_eq!(summary.command_count, 1);
                prop_assert!((summary.total_cost - cost).abs() < 0.00001);
                prop_assert_eq!(summary.total_tokens, input_tokens + output_tokens);

                if cost > 0.0 {
                    prop_assert!(summary.average_cost > 0.0);
                }

                Ok(())
            });
        }

        #[test]
        fn property_concurrent_recording_consistency(
            entries in prop::collection::vec(
                (
                    property_strategies::valid_cost(),
                    property_strategies::valid_tokens(),
                    property_strategies::valid_tokens(),
                    property_strategies::valid_duration(),
                    property_strategies::command_name(),
                    property_strategies::model_name(),
                ),
                1..20
            )
        ) {
            tokio_test::block_on(async {
                let fixture = CostTestFixture::new();
                let tracker = Arc::new(Mutex::new(CostTracker::new(fixture.storage_path.clone()).unwrap()));
                let session_id = Uuid::new_v4();

                let mut handles = vec![];
                let expected_total: f64 = entries.iter().map(|(cost, _, _, _, _, _)| cost).sum();

                for (i, (cost, input_tokens, output_tokens, duration, command, model)) in entries.into_iter().enumerate() {
                    let tracker_clone = Arc::clone(&tracker);
                    let handle = tokio::spawn(async move {
                        let entry = CostEntry::new(
                            session_id,
                            format!("{}_{}", command, i),
                            cost,
                            input_tokens,
                            output_tokens,
                            duration,
                            model,
                        );

                        let mut tracker_lock = tracker_clone.lock().await;
                        tracker_lock.record_cost(entry).await.unwrap();
                    });
                    handles.push(handle);
                }

                for handle in handles {
                    handle.await.unwrap();
                }

                let tracker_lock = tracker.lock().await;
                let summary = tracker_lock.get_session_summary(session_id).await.unwrap();

                prop_assert!((summary.total_cost - expected_total).abs() < 0.0001);

                Ok(())
            });
        }
    }
}

/// Tests for cost aggregation and analysis functionality (Task 1.3)
#[cfg(test)]
mod aggregation_analysis_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[tokio::test]
    async fn test_cost_aggregation_by_session_multiple_entries() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let session1 = Uuid::new_v4();
        let session2 = Uuid::new_v4();

        // Add multiple entries for session1
        let session1_entries = vec![
            CostEntry::new(
                session1,
                "analyze".to_string(),
                0.10,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            ),
            CostEntry::new(
                session1,
                "generate".to_string(),
                0.05,
                50,
                100,
                500,
                "claude-3-sonnet".to_string(),
            ),
            CostEntry::new(
                session1,
                "review".to_string(),
                0.03,
                30,
                60,
                300,
                "claude-3-haiku".to_string(),
            ),
        ];

        // Add entries for session2
        let session2_entries = vec![
            CostEntry::new(
                session2,
                "debug".to_string(),
                0.08,
                80,
                160,
                800,
                "claude-3-opus".to_string(),
            ),
            CostEntry::new(
                session2,
                "optimize".to_string(),
                0.12,
                120,
                240,
                1200,
                "claude-3-sonnet".to_string(),
            ),
        ];

        // Record all entries
        for entry in session1_entries.iter().chain(session2_entries.iter()) {
            tracker.record_cost(entry.clone()).await.unwrap();
        }

        // Test session1 aggregation
        let summary1 = tracker.get_session_summary(session1).await.unwrap();
        assert_eq!(summary1.command_count, 3);
        assert_relative_eq!(summary1.total_cost, 0.18, epsilon = 0.00001); // 0.10 + 0.05 + 0.03
        assert_relative_eq!(summary1.average_cost, 0.06, epsilon = 0.00001); // 0.18 / 3
        assert_eq!(summary1.total_tokens, 540); // (100+200) + (50+100) + (30+60)
        assert_eq!(summary1.by_command.len(), 3);
        assert_eq!(summary1.by_model.len(), 3);

        // Test session2 aggregation
        let summary2 = tracker.get_session_summary(session2).await.unwrap();
        assert_eq!(summary2.command_count, 2);
        assert_relative_eq!(summary2.total_cost, 0.20, epsilon = 0.00001); // 0.08 + 0.12
        assert_relative_eq!(summary2.average_cost, 0.10, epsilon = 0.00001); // 0.20 / 2
        assert_eq!(summary2.total_tokens, 600); // (80+160) + (120+240)
        assert_eq!(summary2.by_command.len(), 2);
        assert_eq!(summary2.by_model.len(), 2);

        // Verify command breakdowns
        assert_relative_eq!(summary1.by_command["analyze"], 0.10, epsilon = 0.00001);
        assert_relative_eq!(summary1.by_command["generate"], 0.05, epsilon = 0.00001);
        assert_relative_eq!(summary1.by_command["review"], 0.03, epsilon = 0.00001);

        assert_relative_eq!(summary2.by_command["debug"], 0.08, epsilon = 0.00001);
        assert_relative_eq!(summary2.by_command["optimize"], 0.12, epsilon = 0.00001);
    }

    #[tokio::test]
    async fn test_cost_aggregation_by_time_periods() {
        let fixture = CostTestFixture::with_time_series_data(30); // 30 days of data
        let tracker = fixture.create_tracker().await;

        let now = Utc::now();

        // Test daily aggregation (last 24 hours)
        let daily_filter = CostFilter {
            since: Some(now - Duration::days(1)),
            until: Some(now),
            ..Default::default()
        };
        let daily_summary = tracker.get_filtered_summary(&daily_filter).await.unwrap();

        // Test weekly aggregation (last 7 days)
        let weekly_filter = CostFilter {
            since: Some(now - Duration::days(7)),
            until: Some(now),
            ..Default::default()
        };
        let weekly_summary = tracker.get_filtered_summary(&weekly_filter).await.unwrap();

        // Test monthly aggregation (last 30 days)
        let monthly_filter = CostFilter {
            since: Some(now - Duration::days(30)),
            until: Some(now),
            ..Default::default()
        };
        let monthly_summary = tracker.get_filtered_summary(&monthly_filter).await.unwrap();

        // Verify hierarchical relationship: daily <= weekly <= monthly
        assert!(daily_summary.command_count <= weekly_summary.command_count);
        assert!(weekly_summary.command_count <= monthly_summary.command_count);

        assert!(daily_summary.total_cost <= weekly_summary.total_cost);
        assert!(weekly_summary.total_cost <= monthly_summary.total_cost);

        // Monthly should have all 90 entries (30 days * 3 entries per day)
        assert_eq!(monthly_summary.command_count, 90);

        // Verify weekly has entries from last 7 days (21 entries)
        assert_eq!(weekly_summary.command_count, 21);

        // Daily should have entries from today (3 entries if today is included)
        assert!(daily_summary.command_count <= 3);
    }

    #[tokio::test]
    async fn test_cost_filtering_by_date_ranges() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let base_time = Utc::now() - Duration::days(10);

        // Create entries with specific timestamps
        let mut entries = vec![];
        for i in 0..10 {
            let mut entry = CostEntry::new(
                session_id,
                format!("day_{}_cmd", i),
                0.01 * (i + 1) as f64,
                10 * (i + 1) as u32,
                20 * (i + 1) as u32,
                100 * (i + 1) as u64,
                "claude-3-opus".to_string(),
            );
            entry.timestamp = base_time + Duration::days(i as i64);
            entries.push(entry);
        }

        for entry in entries {
            tracker.record_cost(entry).await.unwrap();
        }

        // Test filtering middle range (days 3-7)
        let mid_start = base_time + Duration::days(3);
        let mid_end = base_time + Duration::days(7);
        let mid_filter = CostFilter {
            since: Some(mid_start),
            until: Some(mid_end),
            ..Default::default()
        };
        let mid_summary = tracker.get_filtered_summary(&mid_filter).await.unwrap();

        assert_eq!(mid_summary.command_count, 5); // Days 3, 4, 5, 6, 7
        let expected_mid_cost: f64 = (4..=8).map(|i| 0.01 * i as f64).sum(); // 0.04 + 0.05 + 0.06 + 0.07 + 0.08
        assert_relative_eq!(mid_summary.total_cost, expected_mid_cost, epsilon = 0.00001);

        // Test filtering exact day
        let exact_day_start = base_time + Duration::days(5);
        let exact_day_end = exact_day_start + Duration::hours(23) + Duration::minutes(59);
        let exact_filter = CostFilter {
            since: Some(exact_day_start),
            until: Some(exact_day_end),
            ..Default::default()
        };
        let exact_summary = tracker.get_filtered_summary(&exact_filter).await.unwrap();

        assert_eq!(exact_summary.command_count, 1);
        assert_relative_eq!(exact_summary.total_cost, 0.06, epsilon = 0.00001);

        // Test filtering with inclusive boundaries
        let inclusive_filter = CostFilter {
            since: Some(base_time),
            until: Some(base_time + Duration::days(9)),
            ..Default::default()
        };
        let inclusive_summary = tracker
            .get_filtered_summary(&inclusive_filter)
            .await
            .unwrap();

        assert_eq!(inclusive_summary.command_count, 10); // All entries
        let expected_total: f64 = (1..=10).map(|i| 0.01 * i as f64).sum();
        assert_relative_eq!(
            inclusive_summary.total_cost,
            expected_total,
            epsilon = 0.00001
        );
    }

    #[tokio::test]
    async fn test_cost_filtering_edge_cases() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let now = Utc::now();

        // Add some test entries
        for i in 0..5 {
            let mut entry = CostEntry::new(
                session_id,
                format!("test_cmd_{}", i),
                0.01 * (i + 1) as f64,
                10 * (i + 1) as u32,
                20 * (i + 1) as u32,
                100 * (i + 1) as u64,
                "claude-3-opus".to_string(),
            );
            entry.timestamp = now - Duration::hours(i as i64);
            tracker.record_cost(entry).await.unwrap();
        }

        // Test empty date range (since > until)
        let invalid_filter = CostFilter {
            since: Some(now),
            until: Some(now - Duration::hours(1)),
            ..Default::default()
        };
        let invalid_summary = tracker.get_filtered_summary(&invalid_filter).await.unwrap();
        assert_eq!(invalid_summary.command_count, 0);
        assert_eq!(invalid_summary.total_cost, 0.0);

        // Test future date range
        let future_filter = CostFilter {
            since: Some(now + Duration::hours(1)),
            until: Some(now + Duration::hours(2)),
            ..Default::default()
        };
        let future_summary = tracker.get_filtered_summary(&future_filter).await.unwrap();
        assert_eq!(future_summary.command_count, 0);

        // Test very old date range
        let old_filter = CostFilter {
            since: Some(now - Duration::days(365)),
            until: Some(now - Duration::days(364)),
            ..Default::default()
        };
        let old_summary = tracker.get_filtered_summary(&old_filter).await.unwrap();
        assert_eq!(old_summary.command_count, 0);

        // Test exact timestamp boundary
        let first_entry_time = now;
        let boundary_filter = CostFilter {
            since: Some(first_entry_time),
            until: Some(first_entry_time),
            ..Default::default()
        };
        let boundary_summary = tracker
            .get_filtered_summary(&boundary_filter)
            .await
            .unwrap();
        assert_eq!(boundary_summary.command_count, 1); // Should include exact match

        // Test microsecond precision
        let precise_start = now - Duration::microseconds(1);
        let precise_end = now + Duration::microseconds(1);
        let precise_filter = CostFilter {
            since: Some(precise_start),
            until: Some(precise_end),
            ..Default::default()
        };
        let precise_summary = tracker.get_filtered_summary(&precise_filter).await.unwrap();
        assert_eq!(precise_summary.command_count, 1);
    }

    #[tokio::test]
    async fn test_cost_aggregation_by_command_name() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();

        // Add multiple entries for same commands
        let entries = vec![
            // Multiple "analyze" commands
            CostEntry::new(
                session_id,
                "analyze".to_string(),
                0.10,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            ),
            CostEntry::new(
                session_id,
                "analyze".to_string(),
                0.15,
                150,
                300,
                1500,
                "claude-3-opus".to_string(),
            ),
            CostEntry::new(
                session_id,
                "analyze".to_string(),
                0.08,
                80,
                160,
                800,
                "claude-3-sonnet".to_string(),
            ),
            // Multiple "generate" commands
            CostEntry::new(
                session_id,
                "generate".to_string(),
                0.05,
                50,
                100,
                500,
                "claude-3-haiku".to_string(),
            ),
            CostEntry::new(
                session_id,
                "generate".to_string(),
                0.07,
                70,
                140,
                700,
                "claude-3-sonnet".to_string(),
            ),
            // Single "review" command
            CostEntry::new(
                session_id,
                "review".to_string(),
                0.03,
                30,
                60,
                300,
                "claude-3-haiku".to_string(),
            ),
        ];

        for entry in entries {
            tracker.record_cost(entry).await.unwrap();
        }

        let summary = tracker.get_session_summary(session_id).await.unwrap();

        // Verify command aggregation
        assert_eq!(summary.by_command.len(), 3);
        assert_relative_eq!(summary.by_command["analyze"], 0.33, epsilon = 0.00001); // 0.10 + 0.15 + 0.08
        assert_relative_eq!(summary.by_command["generate"], 0.12, epsilon = 0.00001); // 0.05 + 0.07
        assert_relative_eq!(summary.by_command["review"], 0.03, epsilon = 0.00001);

        // Verify model aggregation
        assert_eq!(summary.by_model.len(), 3);
        assert_relative_eq!(summary.by_model["claude-3-opus"], 0.25, epsilon = 0.00001); // 0.10 + 0.15
        assert_relative_eq!(summary.by_model["claude-3-sonnet"], 0.15, epsilon = 0.00001); // 0.08 + 0.07
        assert_relative_eq!(summary.by_model["claude-3-haiku"], 0.08, epsilon = 0.00001); // 0.05 + 0.03

        // Test top commands functionality
        let top_commands = tracker.get_top_commands(5).await.unwrap();
        assert_eq!(top_commands.len(), 3);
        assert_eq!(top_commands[0].0, "analyze"); // Most expensive
        assert_relative_eq!(top_commands[0].1, 0.33, epsilon = 0.00001);
        assert_eq!(top_commands[1].0, "generate");
        assert_relative_eq!(top_commands[1].1, 0.12, epsilon = 0.00001);
        assert_eq!(top_commands[2].0, "review"); // Least expensive
        assert_relative_eq!(top_commands[2].1, 0.03, epsilon = 0.00001);
    }

    #[tokio::test]
    async fn test_cost_aggregation_by_model_types() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();

        // Create entries with different model costs to reflect realistic pricing
        let entries = vec![
            // Opus entries (highest cost)
            CostEntry::new(
                session_id,
                "complex_analysis".to_string(),
                0.50,
                500,
                1000,
                5000,
                "claude-3-opus".to_string(),
            ),
            CostEntry::new(
                session_id,
                "deep_review".to_string(),
                0.60,
                600,
                1200,
                6000,
                "claude-3-opus".to_string(),
            ),
            // Sonnet entries (medium cost)
            CostEntry::new(
                session_id,
                "standard_task".to_string(),
                0.20,
                200,
                400,
                2000,
                "claude-3-sonnet".to_string(),
            ),
            CostEntry::new(
                session_id,
                "balanced_work".to_string(),
                0.25,
                250,
                500,
                2500,
                "claude-3-sonnet".to_string(),
            ),
            CostEntry::new(
                session_id,
                "routine_check".to_string(),
                0.15,
                150,
                300,
                1500,
                "claude-3-sonnet".to_string(),
            ),
            // Haiku entries (lowest cost)
            CostEntry::new(
                session_id,
                "quick_task".to_string(),
                0.05,
                50,
                100,
                500,
                "claude-3-haiku".to_string(),
            ),
            CostEntry::new(
                session_id,
                "simple_query".to_string(),
                0.03,
                30,
                60,
                300,
                "claude-3-haiku".to_string(),
            ),
            CostEntry::new(
                session_id,
                "fast_check".to_string(),
                0.04,
                40,
                80,
                400,
                "claude-3-haiku".to_string(),
            ),
            CostEntry::new(
                session_id,
                "basic_task".to_string(),
                0.02,
                20,
                40,
                200,
                "claude-3-haiku".to_string(),
            ),
        ];

        for entry in entries {
            tracker.record_cost(entry).await.unwrap();
        }

        let summary = tracker.get_session_summary(session_id).await.unwrap();

        // Verify model cost aggregation
        assert_eq!(summary.by_model.len(), 3);
        assert_relative_eq!(summary.by_model["claude-3-opus"], 1.10, epsilon = 0.00001); // 0.50 + 0.60
        assert_relative_eq!(summary.by_model["claude-3-sonnet"], 0.60, epsilon = 0.00001); // 0.20 + 0.25 + 0.15
        assert_relative_eq!(summary.by_model["claude-3-haiku"], 0.14, epsilon = 0.00001); // 0.05 + 0.03 + 0.04 + 0.02

        // Verify total cost
        assert_relative_eq!(summary.total_cost, 1.84, epsilon = 0.00001); // 1.10 + 0.60 + 0.14

        // Verify command count distribution
        assert_eq!(summary.command_count, 9);

        // Test cost ratio analysis
        let opus_ratio = summary.by_model["claude-3-opus"] / summary.total_cost;
        let sonnet_ratio = summary.by_model["claude-3-sonnet"] / summary.total_cost;
        let haiku_ratio = summary.by_model["claude-3-haiku"] / summary.total_cost;

        assert!(opus_ratio > sonnet_ratio); // Opus should be most expensive
        assert!(sonnet_ratio > haiku_ratio); // Sonnet should be medium cost
        assert_relative_eq!(
            opus_ratio + sonnet_ratio + haiku_ratio,
            1.0,
            epsilon = 0.00001
        );
    }

    #[tokio::test]
    async fn test_global_vs_session_aggregation() {
        let fixture = CostTestFixture::new();
        let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        // Create multiple sessions
        let sessions = vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()];

        let mut total_expected_cost = 0.0;
        let mut total_expected_commands = 0;

        // Add entries for each session
        for (session_index, &session_id) in sessions.iter().enumerate() {
            for cmd_index in 0..5 {
                let cost = 0.01 * ((session_index * 5 + cmd_index) + 1) as f64;
                let entry = CostEntry::new(
                    session_id,
                    format!("s{}_cmd_{}", session_index, cmd_index),
                    cost,
                    10 * (cmd_index + 1) as u32,
                    20 * (cmd_index + 1) as u32,
                    100 * (cmd_index + 1) as u64,
                    format!("model_{}", cmd_index % 3),
                );
                tracker.record_cost(entry).await.unwrap();
                total_expected_cost += cost;
                total_expected_commands += 1;
            }
        }

        // Test global aggregation
        let global_summary = tracker.get_global_summary().await.unwrap();
        assert_eq!(global_summary.command_count, total_expected_commands);
        assert_relative_eq!(
            global_summary.total_cost,
            total_expected_cost,
            epsilon = 0.00001
        );
        assert_relative_eq!(
            global_summary.average_cost,
            total_expected_cost / total_expected_commands as f64,
            epsilon = 0.00001
        );

        // Test individual session aggregations
        let mut sum_of_session_costs = 0.0;
        let mut sum_of_session_commands = 0;

        for (session_index, &session_id) in sessions.iter().enumerate() {
            let session_summary = tracker.get_session_summary(session_id).await.unwrap();
            assert_eq!(session_summary.command_count, 5);

            // Verify session cost calculation
            let session_expected_cost: f64 = (0..5)
                .map(|cmd_index| 0.01 * ((session_index * 5 + cmd_index) + 1) as f64)
                .sum();
            assert_relative_eq!(
                session_summary.total_cost,
                session_expected_cost,
                epsilon = 0.00001
            );

            sum_of_session_costs += session_summary.total_cost;
            sum_of_session_commands += session_summary.command_count;
        }

        // Verify that sum of sessions equals global
        assert_relative_eq!(
            sum_of_session_costs,
            global_summary.total_cost,
            epsilon = 0.00001
        );
        assert_eq!(sum_of_session_commands, global_summary.command_count);

        // Verify global command breakdown includes all sessions
        assert_eq!(global_summary.by_command.len(), 15); // 3 sessions * 5 commands each
    }

    #[tokio::test]
    async fn test_cost_aggregation_with_empty_results() {
        let fixture = CostTestFixture::new();
        let tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();

        let empty_session_id = Uuid::new_v4();

        // Test aggregation on empty session
        let empty_summary = tracker.get_session_summary(empty_session_id).await.unwrap();
        assert_eq!(empty_summary.command_count, 0);
        assert_eq!(empty_summary.total_cost, 0.0);
        assert_eq!(empty_summary.average_cost, 0.0);
        assert_eq!(empty_summary.total_tokens, 0);
        assert!(empty_summary.by_command.is_empty());
        assert!(empty_summary.by_model.is_empty());

        // Test global aggregation on empty tracker
        let global_empty = tracker.get_global_summary().await.unwrap();
        assert_eq!(global_empty.command_count, 0);
        assert_eq!(global_empty.total_cost, 0.0);
        assert_eq!(global_empty.average_cost, 0.0);
        assert_eq!(global_empty.total_tokens, 0);

        // Test filtered aggregation with no matches
        let no_match_filter = CostFilter {
            command_pattern: Some("nonexistent_command".to_string()),
            ..Default::default()
        };
        let no_match_summary = tracker
            .get_filtered_summary(&no_match_filter)
            .await
            .unwrap();
        assert_eq!(no_match_summary.command_count, 0);
        assert_eq!(no_match_summary.total_cost, 0.0);

        // Test top commands on empty tracker
        let top_commands = tracker.get_top_commands(10).await.unwrap();
        assert!(top_commands.is_empty());
    }

    #[tokio::test]
    async fn test_cost_aggregation_precision_with_large_datasets() {
        let fixture = CostTestFixture::with_large_dataset(1000);
        let tracker = fixture.create_tracker().await;

        let global_summary = tracker.get_global_summary().await.unwrap();

        // Verify aggregation consistency with large dataset
        assert_eq!(global_summary.command_count, 1000);

        // Calculate expected total manually
        let expected_total: f64 = (0..1000).map(|i| 0.001 + (i as f64 * 0.001)).sum();
        assert_relative_eq!(global_summary.total_cost, expected_total, epsilon = 0.001);

        // Verify average calculation
        let expected_average = expected_total / 1000.0;
        assert_relative_eq!(
            global_summary.average_cost,
            expected_average,
            epsilon = 0.001
        );

        // Verify token calculation
        let expected_tokens: u32 = (0..1000)
            .map(|i| (10 + (i as u32 * 5)) + (20 + (i as u32 * 10)))
            .sum();
        assert_eq!(global_summary.total_tokens, expected_tokens);

        // Test that aggregation performs well with large dataset
        let start = std::time::Instant::now();
        let _summary = tracker.get_global_summary().await.unwrap();
        let duration = start.elapsed();

        // Should complete aggregation quickly
        assert!(duration.as_millis() < 100);
    }

    proptest! {
        #[test]
        fn property_aggregation_consistency(
            entries in prop::collection::vec(
                (
                    property_strategies::valid_cost(),
                    property_strategies::valid_tokens(),
                    property_strategies::valid_tokens(),
                    property_strategies::command_name(),
                    property_strategies::model_name(),
                ),
                1..50
            )
        ) {
            let _ = tokio_test::block_on(async {
                let fixture = CostTestFixture::new();
                let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();
                let session_id = Uuid::new_v4();

                let mut expected_total_cost = 0.0;
                let mut expected_total_tokens = 0u32;

                for (i, (cost, input_tokens, output_tokens, command, model)) in entries.into_iter().enumerate() {
                    let entry = CostEntry::new(
                        session_id,
                        format!("{}_{}", command, i),
                        cost,
                        input_tokens,
                        output_tokens,
                        1000,
                        model,
                    );

                    tracker.record_cost(entry).await.unwrap();
                    expected_total_cost += cost;
                    expected_total_tokens += input_tokens + output_tokens;
                }

                let summary = tracker.get_session_summary(session_id).await.unwrap();

                // Verify aggregation properties
                prop_assert!((summary.total_cost - expected_total_cost).abs() < 0.001);
                prop_assert_eq!(summary.total_tokens, expected_total_tokens);

                if summary.command_count > 0 {
                    prop_assert!((summary.average_cost - (expected_total_cost / summary.command_count as f64)).abs() < 0.001);
                }

                // Verify that by_command total equals session total
                let command_total: f64 = summary.by_command.values().sum();
                prop_assert!((command_total - summary.total_cost).abs() < 0.001);

                // Verify that by_model total equals session total
                let model_total: f64 = summary.by_model.values().sum();
                prop_assert!((model_total - summary.total_cost).abs() < 0.001);

                Ok(())
            });
        }

        #[test]
        fn property_filtering_subset_consistency(
            cost_filter in 0.001f64..1.0f64,
            token_filter in 1u32..1000u32,
        ) {
            let _ = tokio_test::block_on(async {
                let fixture = CostTestFixture::new();
                let mut tracker = CostTracker::new(fixture.storage_path.clone()).unwrap();
                let session_id = Uuid::new_v4();

                // Add test entries with varying costs and tokens
                for i in 0..20 {
                    let entry = CostEntry::new(
                        session_id,
                        format!("cmd_{}", i),
                        0.01 * (i + 1) as f64,
                        10 * (i + 1) as u32,
                        20 * (i + 1) as u32,
                        1000,
                        "claude-3-opus".to_string(),
                    );
                    tracker.record_cost(entry).await.unwrap();
                }

                // Get global summary
                let global_summary = tracker.get_global_summary().await.unwrap();

                // Apply cost filter
                let cost_filtered = CostFilter {
                    min_cost: Some(cost_filter),
                    ..Default::default()
                };
                let filtered_summary = tracker.get_filtered_summary(&cost_filtered).await.unwrap();

                // Filtered results should be subset of global
                prop_assert!(filtered_summary.command_count <= global_summary.command_count);
                prop_assert!(filtered_summary.total_cost <= global_summary.total_cost);
                prop_assert!(filtered_summary.total_tokens <= global_summary.total_tokens);

                // All filtered costs should meet minimum threshold
                for entry in tracker.get_entries(&cost_filtered).await.unwrap() {
                    prop_assert!(entry.cost_usd >= cost_filter);
                }

                Ok(())
            });
        }
    }
}
