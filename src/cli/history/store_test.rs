//! Comprehensive unit tests for the history storage module
//!
//! This module provides extensive test coverage for history storage functionality including:
//! - History entry creation and validation
//! - Search functionality with various criteria
//! - Storage operations and data persistence
//! - Filtering and pagination
//! - Backup and restore operations
//! - Performance testing with large datasets
//! - Concurrent access and thread safety
//! - Edge cases and error handling

use super::store::EnhancedHistoryStore;
use super::{ExportFormat, HistoryEntry, HistorySearch, HistoryStats, HistoryStore, SortField};
use crate::{cli::error::InteractiveError, cli::session::SessionId};
use chrono::{DateTime, Duration, Utc};
use proptest::prelude::*;
use proptest::strategy::Just;
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::tempdir;
use uuid::Uuid;

/// Test fixture for creating history entries with known data
pub struct HistoryTestFixture {
    pub session_id: SessionId,
    pub entries: Vec<HistoryEntry>,
    pub temp_dir: tempfile::TempDir,
    pub storage_path: PathBuf,
}

impl HistoryTestFixture {
    /// Create a new test fixture with sample data
    pub fn new() -> Self {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let storage_path = temp_dir.path().join("test_history.json");
        let session_id = Uuid::new_v4();

        let mut entries = Vec::new();

        // Create diverse test entries with different command types and outcomes
        entries.push(
            HistoryEntry::new(
                session_id,
                "analyze_code".to_string(),
                vec!["--detailed".to_string(), "src/main.rs".to_string()],
                "Analysis complete: Found 3 potential improvements...".to_string(),
                true,
                2500,
            )
            .with_cost(0.050, 150, 300, "claude-3-opus".to_string())
            .with_tags(vec!["analysis".to_string(), "code-review".to_string()]),
        );

        entries.push(
            HistoryEntry::new(
                session_id,
                "generate_docs".to_string(),
                vec!["--format".to_string(), "markdown".to_string()],
                "# Documentation\n\nGenerated comprehensive documentation...".to_string(),
                true,
                1800,
            )
            .with_cost(0.025, 75, 150, "claude-3-sonnet".to_string())
            .with_tags(vec!["documentation".to_string(), "generation".to_string()]),
        );

        entries.push(
            HistoryEntry::new(
                session_id,
                "review_pr".to_string(),
                vec!["--pr-number".to_string(), "123".to_string()],
                "Pull request review completed. Approved with minor suggestions.".to_string(),
                true,
                1200,
            )
            .with_cost(0.015, 50, 100, "claude-3-haiku".to_string())
            .with_tags(vec!["review".to_string(), "pull-request".to_string()]),
        );

        entries.push(
            HistoryEntry::new(
                session_id,
                "debug_issue".to_string(),
                vec!["--error-log".to_string(), "app.log".to_string()],
                "".to_string(),
                false,
                3000,
            )
            .with_error("Failed to parse error log: file not found".to_string())
            .with_tags(vec!["debugging".to_string(), "error".to_string()]),
        );

        entries.push(
            HistoryEntry::new(
                session_id,
                "analyze_code".to_string(), // Duplicate command name
                vec!["--quick".to_string(), "src/lib.rs".to_string()],
                "Quick analysis: No major issues found.".to_string(),
                true,
                1000,
            )
            .with_cost(0.020, 60, 120, "claude-3-sonnet".to_string())
            .with_tags(vec!["analysis".to_string(), "quick".to_string()]),
        );

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
        let storage_path = temp_dir.path().join("large_test_history.json");
        let session_id = Uuid::new_v4();

        let mut entries = Vec::new();
        let commands = [
            "analyze", "generate", "review", "refactor", "test", "debug", "optimize", "search",
            "compile", "deploy",
        ];
        let models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"];
        let tags = [
            "urgent",
            "review",
            "testing",
            "production",
            "development",
            "maintenance",
            "feature",
            "bugfix",
        ];

        for i in 0..entry_count {
            let command_idx = i % commands.len();
            let model_idx = i % models.len();
            let tag_idx = i % tags.len();

            let mut entry = HistoryEntry::new(
                session_id,
                format!("{}_{}", commands[command_idx], i),
                vec![
                    format!("--arg-{}", i),
                    format!("value-{}", i),
                    format!("file-{}.rs", i),
                ],
                format!("Output for command {} iteration {}: Processing completed successfully with {} results.", commands[command_idx], i, i % 100),
                i % 10 != 0, // 90% success rate
                500 + (i as u64 * 50), // Varying duration
            );

            if i % 2 == 0 {
                entry = entry.with_cost(
                    0.001 + (i as f64 * 0.001), // Varying costs
                    10 + (i as u32 * 2),        // Varying input tokens
                    20 + (i as u32 * 3),        // Varying output tokens
                    models[model_idx].to_string(),
                );
            }

            if i % 3 == 0 {
                entry = entry.with_tags(vec![
                    tags[tag_idx].to_string(),
                    format!("batch-{}", i / 100),
                ]);
            }

            if !entry.success {
                entry = entry.with_error(format!(
                    "Error occurred in iteration {}: Simulated failure for testing",
                    i
                ));
            }

            entries.push(entry);
        }

        Self {
            session_id,
            entries,
            temp_dir,
            storage_path,
        }
    }

    /// Create fixture with time-distributed entries for temporal analysis
    pub fn with_time_series_data(days: i64) -> Self {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let storage_path = temp_dir.path().join("timeseries_test_history.json");
        let session_id = Uuid::new_v4();

        let mut entries = Vec::new();
        let base_time = Utc::now() - Duration::days(days);

        for day in 0..days {
            // Create 2-5 entries per day with different times
            let entries_per_day = 2 + (day % 4) as usize;
            for entry_num in 0..entries_per_day {
                let hour = 9 + (entry_num * 3) % 12; // Spread throughout work day
                let mut entry = HistoryEntry::new(
                    session_id,
                    format!("daily_task_{}_{}", day, entry_num),
                    vec![format!("--day-{}", day), format!("--task-{}", entry_num)],
                    format!(
                        "Daily task {} on day {}: Completed routine processing",
                        entry_num, day
                    ),
                    (day + entry_num as i64) % 5 != 0, // Some failures for realism
                    1000 + (day as u64 * 100) + (entry_num as u64 * 50),
                );

                // Set specific timestamp
                entry.timestamp = base_time + Duration::days(day) + Duration::hours(hour as i64);

                if day % 3 == 0 {
                    entry = entry.with_cost(
                        0.010 + (day as f64 * 0.002),
                        50 + (day as u32),
                        100 + (day as u32 * 2),
                        "claude-3-opus".to_string(),
                    );
                }

                if !entry.success {
                    entry = entry
                        .with_error(format!("Daily task failed on day {}: Simulated error", day));
                }

                entry = entry.with_tags(vec![
                    format!("day-{}", day),
                    if day % 7 == 0 { "weekly" } else { "daily" }.to_string(),
                ]);

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

    /// Create fixture with multiple sessions for cross-session testing
    pub fn with_multi_session_data() -> (Vec<SessionId>, Self) {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let storage_path = temp_dir.path().join("multi_session_test_history.json");

        let sessions: Vec<SessionId> = (0..4).map(|_| Uuid::new_v4()).collect();
        let mut all_entries = Vec::new();

        for (session_idx, &session_id) in sessions.iter().enumerate() {
            for cmd_idx in 0..6 {
                let mut entry = HistoryEntry::new(
                    session_id,
                    format!("session_{}_cmd_{}", session_idx, cmd_idx),
                    vec![format!("--session-{}", session_idx), format!("--cmd-{}", cmd_idx)],
                    format!(
                        "Session {} command {}: Processing completed",
                        session_idx, cmd_idx
                    ),
                    (session_idx + cmd_idx) % 7 != 0, // Varied success rate
                    1000 + (session_idx * 100 + cmd_idx * 50) as u64,
                );

                entry = entry
                    .with_cost(
                        0.01 * ((session_idx * 6 + cmd_idx) + 1) as f64,
                        50 + (cmd_idx * 10) as u32,
                        100 + (cmd_idx * 20) as u32,
                        format!("model-{}", session_idx % 3),
                    )
                    .with_tags(vec![
                        format!("session-{}", session_idx),
                        format!("batch-{}", cmd_idx / 3),
                    ]);

                if !entry.success {
                    entry = entry.with_error(format!(
                        "Error in session {} command {}",
                        session_idx, cmd_idx
                    ));
                }

                all_entries.push(entry);
            }
        }

        let fixture = Self {
            session_id: sessions[0], // Use first session as primary
            entries: all_entries,
            temp_dir,
            storage_path,
        };

        (sessions, fixture)
    }

    /// Create a history store with the fixture data
    pub async fn create_store(&self) -> HistoryStore {
        let mut store =
            HistoryStore::new(self.storage_path.clone()).expect("Failed to create test store");

        for entry in &self.entries {
            store
                .store_entry(entry.clone())
                .await
                .expect("Failed to store test entry");
        }

        store
    }

    /// Create enhanced history store with fixture data
    pub async fn create_enhanced_store(&self) -> EnhancedHistoryStore {
        let base_path = self.temp_dir.path().join("enhanced_store");
        let store =
            EnhancedHistoryStore::new(base_path).expect("Failed to create enhanced test store");

        // Note: Enhanced store is created empty for testing - individual tests will populate it as needed
        store
    }
}

/// Helper functions for test data generation and validation
pub mod test_helpers {
    use super::*;

    /// Create a history entry with minimal valid data
    pub fn minimal_history_entry(session_id: SessionId) -> HistoryEntry {
        HistoryEntry::new(
            session_id,
            "test_cmd".to_string(),
            vec!["arg".to_string()],
            "output".to_string(),
            true,
            100,
        )
    }

    /// Create a history entry with maximum realistic values
    pub fn maximal_history_entry(session_id: SessionId) -> HistoryEntry {
        HistoryEntry::new(
            session_id,
            "extremely_comprehensive_analysis_command_with_very_long_name_and_detailed_parameters".to_string(),
            vec![
                "--extensive-analysis".to_string(),
                "--deep-dive".to_string(),
                "--comprehensive-report".to_string(),
                "--all-formats".to_string(),
                "--verbose".to_string(),
                "very_large_file_with_extensive_content.rs".to_string(),
            ],
            "Extremely comprehensive analysis completed successfully.\n\nDetailed findings:\n- Analyzed 50,000 lines of code\n- Found 127 potential improvements\n- Generated 45 recommendations\n- Created comprehensive documentation\n- Performed security audit\n- Generated performance benchmarks\n\nComplete analysis report available with all metrics and recommendations.".to_string(),
            true,
            3600000, // 1 hour
        )
        .with_cost(99.999, 100000, 200000, "claude-3-opus-premium-ultra".to_string())
        .with_tags(vec![
            "comprehensive".to_string(),
            "analysis".to_string(),
            "security".to_string(),
            "performance".to_string(),
            "documentation".to_string(),
            "recommendations".to_string(),
        ])
    }

    /// Create entries with edge case values for boundary testing
    pub fn edge_case_entries(session_id: SessionId) -> Vec<HistoryEntry> {
        vec![
            // Zero duration
            HistoryEntry::new(
                session_id,
                "instant_cmd".to_string(),
                vec![],
                "instant".to_string(),
                true,
                0,
            ),
            // Empty output
            HistoryEntry::new(
                session_id,
                "silent_cmd".to_string(),
                vec!["--quiet".to_string()],
                "".to_string(),
                true,
                500,
            ),
            // Unicode in all fields
            HistoryEntry::new(
                session_id,
                "分析代码".to_string(),
                vec!["--文件".to_string(), "测试.rs".to_string()],
                "分析完成：发现了3个潜在的改进...".to_string(),
                true,
                1500,
            )
            .with_tags(vec!["中文".to_string(), "测试".to_string()]),
            // Special characters and symbols
            HistoryEntry::new(
                session_id,
                "cmd-with_special.chars@test!".to_string(),
                vec!["--path=/tmp/test$file&name".to_string(), "--regex=\\d+\\s*".to_string()],
                "Output with special chars: <>\"'&\nNew line\tTab\r\nWindows line ending"
                    .to_string(),
                true,
                2000,
            )
            .with_cost(0.00001, 1, 1, "model-v2.1@special".to_string()),
            // Very long strings to test storage limits
            HistoryEntry::new(
                session_id,
                "x".repeat(1000),        // Very long command name
                vec!["arg".repeat(500)], // Very long argument
                "y".repeat(10000),       // Very long output
                false,
                999999,
            )
            .with_error("z".repeat(1000)) // Very long error message
            .with_tags(vec!["long".repeat(100)]), // Very long tags
        ]
    }

    /// Generate entries for concurrent testing
    pub fn concurrent_test_entries(session_id: SessionId, count: usize) -> Vec<HistoryEntry> {
        (0..count)
            .map(|i| {
                HistoryEntry::new(
                    session_id,
                    format!("concurrent_cmd_{:04}", i),
                    vec![format!("--worker-{}", i), format!("--batch-{}", i / 10)],
                    format!("Concurrent processing {} completed successfully", i),
                    i % 20 != 0, // 95% success rate
                    100 + (i * 10) as u64,
                )
                .with_cost(
                    0.001 * (i + 1) as f64,
                    5 + (i % 100) as u32,
                    10 + (i % 200) as u32,
                    format!("model-{}", i % 4),
                )
                .with_tags(vec![
                    format!("concurrent-{}", i / 50),
                    "performance-test".to_string(),
                ])
            })
            .collect()
    }

    /// Create entries with specific patterns for search testing
    pub fn search_pattern_entries(session_id: SessionId) -> Vec<HistoryEntry> {
        vec![
            // Exact match targets
            HistoryEntry::new(
                session_id,
                "exact_match_command".to_string(),
                vec!["--exact".to_string()],
                "This is an exact match for searching".to_string(),
                true,
                1000,
            ),
            // Partial match targets
            HistoryEntry::new(
                session_id,
                "analyze_performance".to_string(),
                vec!["--performance".to_string()],
                "Performance analysis completed with detailed metrics".to_string(),
                true,
                1500,
            ),
            // Regex pattern targets
            HistoryEntry::new(
                session_id,
                "process_data_123".to_string(),
                vec!["--data-id=456".to_string()],
                "Processing data with ID 789 completed successfully".to_string(),
                true,
                2000,
            ),
            // Case sensitivity test
            HistoryEntry::new(
                session_id,
                "CaseSensitiveCommand".to_string(),
                vec!["--CamelCase".to_string()],
                "CamelCase output with Mixed CASE text".to_string(),
                true,
                1200,
            ),
            // Special character patterns
            HistoryEntry::new(
                session_id,
                "cmd.with.dots".to_string(),
                vec!["--pattern=*.rs".to_string()],
                "Found files: main.rs, lib.rs, test.rs".to_string(),
                true,
                1800,
            ),
        ]
    }

    /// Validate search results meet criteria
    pub fn validate_search_results(results: &[HistoryEntry], criteria: &HistorySearch) -> bool {
        for entry in results {
            // Check session ID filter
            if let Some(session_id) = criteria.session_id {
                if entry.session_id != session_id {
                    return false;
                }
            }

            // Check command pattern
            if let Some(pattern) = &criteria.command_pattern {
                if !entry.command_name.contains(pattern) {
                    return false;
                }
            }

            // Check output pattern
            if let Some(pattern) = &criteria.output_pattern {
                if !entry.output.contains(pattern) {
                    return false;
                }
            }

            // Check success filters
            if criteria.success_only && !entry.success {
                return false;
            }
            if criteria.failures_only && entry.success {
                return false;
            }

            // Check time range
            if let Some(since) = criteria.since {
                if entry.timestamp < since {
                    return false;
                }
            }
            if let Some(until) = criteria.until {
                if entry.timestamp > until {
                    return false;
                }
            }

            // Check duration range
            if let Some(min_duration) = criteria.min_duration_ms {
                if entry.duration_ms < min_duration {
                    return false;
                }
            }
            if let Some(max_duration) = criteria.max_duration_ms {
                if entry.duration_ms > max_duration {
                    return false;
                }
            }

            // Check cost range (only for entries with cost)
            if let Some(min_cost) = criteria.min_cost {
                if let Some(cost) = entry.cost_usd {
                    if cost < min_cost {
                        return false;
                    }
                } else if min_cost > 0.0 {
                    return false;
                }
            }
            if let Some(max_cost) = criteria.max_cost {
                if let Some(cost) = entry.cost_usd {
                    if cost > max_cost {
                        return false;
                    }
                } else if max_cost >= 0.0 {
                    // Entry without cost should pass max_cost filter
                    continue;
                }
            }

            // Check model filter
            if let Some(model) = &criteria.model {
                if let Some(entry_model) = &entry.model {
                    if entry_model != model {
                        return false;
                    }
                } else {
                    return false;
                }
            }

            // Check tags filter
            if !criteria.tags.is_empty() {
                let has_any_tag = criteria.tags.iter().any(|tag| entry.tags.contains(tag));
                if !has_any_tag {
                    return false;
                }
            }
        }

        // Check result limits and pagination
        if results.len() > criteria.limit {
            return false;
        }

        true
    }

    /// Validate history statistics calculation
    pub fn validate_history_stats(entries: &[HistoryEntry], stats: &HistoryStats) -> bool {
        if entries.is_empty() {
            return stats.total_entries == 0
                && stats.successful_commands == 0
                && stats.failed_commands == 0
                && stats.success_rate == 0.0
                && stats.total_cost == 0.0
                && stats.total_duration_ms == 0
                && stats.average_duration_ms == 0.0
                && stats.average_cost == 0.0;
        }

        let expected_total = entries.len();
        let expected_successful = entries.iter().filter(|e| e.success).count();
        let expected_failed = expected_total - expected_successful;
        let expected_success_rate = (expected_successful as f64 / expected_total as f64) * 100.0;

        let expected_total_cost: f64 = entries.iter().filter_map(|e| e.cost_usd).sum();
        let expected_total_duration: u64 = entries.iter().map(|e| e.duration_ms).sum();
        let expected_avg_duration = expected_total_duration as f64 / expected_total as f64;

        let cost_entries = entries.iter().filter(|e| e.cost_usd.is_some()).count();
        let expected_avg_cost = if cost_entries > 0 {
            expected_total_cost / cost_entries as f64
        } else {
            0.0
        };

        let total_match = stats.total_entries == expected_total;
        let successful_match = stats.successful_commands == expected_successful;
        let failed_match = stats.failed_commands == expected_failed;
        let success_rate_match = (stats.success_rate - expected_success_rate).abs() < 0.01;
        let cost_match = (stats.total_cost - expected_total_cost).abs() < 0.00001;
        let duration_match = stats.total_duration_ms == expected_total_duration;
        let avg_duration_match = (stats.average_duration_ms - expected_avg_duration).abs() < 0.1;
        let avg_cost_match = (stats.average_cost - expected_avg_cost).abs() < 0.00001;

        total_match
            && successful_match
            && failed_match
            && success_rate_match
            && cost_match
            && duration_match
            && avg_duration_match
            && avg_cost_match
    }

    /// Create performance benchmark entries for large-scale testing
    pub fn benchmark_entries(count: usize) -> Vec<HistoryEntry> {
        let session_id = Uuid::new_v4();
        (0..count)
            .map(|i| {
                HistoryEntry::new(
                    session_id,
                    format!("benchmark_cmd_{:08}", i),
                    vec![format!("--index={}", i), format!("--batch={}", i / 1000)],
                    format!(
                        "Benchmark iteration {} completed with {} operations",
                        i,
                        i % 1000
                    ),
                    i % 100 != 0, // 99% success rate for realistic benchmarking
                    50 + (i % 10000) as u64, // Varied duration up to 10 seconds
                )
                .with_cost(
                    0.0001 + (i as f64 / 10000000.0), // Small incremental costs
                    1 + (i % 1000) as u32,            // Varied input tokens
                    2 + (i % 2000) as u32,            // Varied output tokens
                    match i % 5 {
                        0 => "claude-3-opus",
                        1 => "claude-3-sonnet",
                        2 => "claude-3-haiku",
                        3 => "gpt-4",
                        _ => "local-model",
                    }
                    .to_string(),
                )
                .with_tags(vec![
                    format!("benchmark-{}", i / 10000),
                    "performance".to_string(),
                    if i % 2 == 0 { "even" } else { "odd" }.to_string(),
                ])
            })
            .collect()
    }

    /// Generate random search criteria for property testing
    pub fn random_search_criteria(
        session_id: Option<SessionId>,
    ) -> impl Strategy<Value = HistorySearch> {
        // Simplified version with fewer fields to avoid proptest tuple limit
        (
            any::<bool>(),
            any::<bool>(),
            1usize..500usize,
            0usize..1000usize,
            any::<bool>(),
        )
            .prop_map(
                move |(success_only, failures_only, limit, offset, sort_desc)| HistorySearch {
                    session_id,
                    command_pattern: None,
                    output_pattern: None,
                    error_pattern: None,
                    success_only,
                    failures_only,
                    since: None,
                    until: None,
                    min_duration_ms: None,
                    max_duration_ms: None,
                    min_cost: None,
                    max_cost: None,
                    model: None,
                    tags: Vec::new(),
                    limit,
                    offset,
                    sort_by: SortField::Timestamp,
                    sort_desc,
                },
            )
    }
}

/// Property-based test strategies for history testing
pub mod property_strategies {
    use super::*;

    /// Strategy for generating valid durations
    pub fn valid_duration() -> impl Strategy<Value = u64> {
        1u64..3600000u64 // 1ms to 1 hour
    }

    /// Strategy for generating realistic command names
    pub fn command_name() -> impl Strategy<Value = String> {
        prop::collection::vec(prop::string::string_regex("[a-z_]{1,15}").unwrap(), 1..4)
            .prop_map(|parts| parts.join("_"))
    }

    /// Strategy for generating command arguments
    pub fn command_args() -> impl Strategy<Value = Vec<String>> {
        prop::collection::vec(
            prop_oneof![
                prop::string::string_regex("--[a-z-]{2,10}").unwrap(),
                prop::string::string_regex("[a-zA-Z0-9._-]{1,20}").unwrap(),
            ],
            0..6,
        )
    }

    /// Strategy for generating output text
    pub fn output_text() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("".to_string()), // Empty output
            prop::string::string_regex("[A-Za-z0-9 .,!?\\n]{10,200}").unwrap(), // Normal output
            prop::string::string_regex("[A-Za-z0-9 .,!?\\n]{500,2000}").unwrap(), // Long output
        ]
    }

    /// Strategy for generating success status
    pub fn success_status() -> impl Strategy<Value = bool> {
        prop::bool::weighted(0.8) // 80% success rate
    }

    /// Strategy for generating valid costs
    pub fn valid_cost() -> impl Strategy<Value = f64> {
        prop_oneof![
            Just(0.0),                                                  // Free operations
            (0.001..0.1f64),                                            // Small costs
            (0.1..10.0f64),                                             // Medium costs
            (10.0..100.0f64).prop_map(|x| (x * 100.0).round() / 100.0), // Large costs, rounded
        ]
    }

    /// Strategy for generating token counts
    pub fn token_count() -> impl Strategy<Value = u32> {
        prop_oneof![
            Just(0u32),           // No tokens
            1u32..100u32,         // Small token counts
            100u32..10000u32,     // Medium token counts
            10000u32..1000000u32, // Large token counts
        ]
    }

    /// Strategy for generating model names
    pub fn model_name() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("claude-3-opus".to_string()),
            Just("claude-3-sonnet".to_string()),
            Just("claude-3-haiku".to_string()),
            prop::string::string_regex("model-[a-z0-9-]{5,20}").unwrap(),
        ]
    }

    /// Strategy for generating tags
    pub fn tag_list() -> impl Strategy<Value = Vec<String>> {
        prop::collection::vec(prop::string::string_regex("[a-z-]{3,15}").unwrap(), 0..5)
    }

    /// Strategy for generating past datetimes
    pub fn past_datetime() -> impl Strategy<Value = DateTime<Utc>> {
        (-365i64..0i64).prop_map(|days| Utc::now() + Duration::days(days))
    }

    /// Strategy for generating future datetimes
    pub fn future_datetime() -> impl Strategy<Value = DateTime<Utc>> {
        (0i64..365i64).prop_map(|days| Utc::now() + Duration::days(days))
    }

    /// Strategy for generating complete history entries
    pub fn history_entry(session_id: SessionId) -> impl Strategy<Value = HistoryEntry> {
        (
            command_name(),
            command_args(),
            output_text(),
            success_status(),
            valid_duration(),
            prop::option::of((valid_cost(), token_count(), token_count(), model_name())),
            prop::option::of(prop::string::string_regex("[A-Za-z0-9 .,!?]{10,100}").unwrap()),
            tag_list(),
        )
            .prop_map(
                move |(command, args, output, success, duration, cost_info, error_msg, tags)| {
                    let mut entry =
                        HistoryEntry::new(session_id, command, args, output, success, duration);

                    if let Some((cost, input_tokens, output_tokens, model)) = cost_info {
                        entry = entry.with_cost(cost, input_tokens, output_tokens, model);
                    }

                    if !success {
                        if let Some(error) = error_msg {
                            entry = entry.with_error(error);
                        }
                    }

                    if !tags.is_empty() {
                        entry = entry.with_tags(tags);
                    }

                    entry
                },
            )
    }
}

#[cfg(test)]
mod test_infrastructure_tests {
    use super::*;

    #[test]
    fn test_fixture_creation() {
        let fixture = HistoryTestFixture::new();

        assert!(!fixture.entries.is_empty());
        assert!(!fixture.storage_path.exists()); // Should not exist yet
        assert_eq!(fixture.entries.len(), 5);

        // Verify entries have the same session ID
        for entry in &fixture.entries {
            assert_eq!(entry.session_id, fixture.session_id);
        }

        // Verify entry diversity
        let success_count = fixture.entries.iter().filter(|e| e.success).count();
        let failure_count = fixture.entries.len() - success_count;
        assert!(success_count > 0);
        assert!(failure_count > 0);

        // Verify cost entries exist
        let cost_entries = fixture
            .entries
            .iter()
            .filter(|e| e.cost_usd.is_some())
            .count();
        assert!(cost_entries > 0);

        // Verify tag entries exist
        let tagged_entries = fixture
            .entries
            .iter()
            .filter(|e| !e.tags.is_empty())
            .count();
        assert!(tagged_entries > 0);
    }

    #[test]
    fn test_large_dataset_fixture() {
        let fixture = HistoryTestFixture::with_large_dataset(1000);

        assert_eq!(fixture.entries.len(), 1000);

        // Verify diversity in the dataset
        let mut unique_commands = std::collections::HashSet::new();
        let mut unique_models = std::collections::HashSet::new();

        for entry in &fixture.entries {
            if let Some(base_cmd) = entry.command_name.split('_').next() {
                unique_commands.insert(base_cmd);
            }
            if let Some(model) = &entry.model {
                unique_models.insert(model);
            }
        }

        assert!(unique_commands.len() >= 5);
        assert!(unique_models.len() >= 3);

        // Verify success rate is around 90%
        let success_count = fixture.entries.iter().filter(|e| e.success).count();
        let success_rate = success_count as f64 / fixture.entries.len() as f64;
        assert!(success_rate > 0.85 && success_rate < 0.95);
    }

    #[test]
    fn test_time_series_fixture() {
        let fixture = HistoryTestFixture::with_time_series_data(7);

        assert!(fixture.entries.len() >= 14); // At least 2 per day
        assert!(fixture.entries.len() <= 35); // At most 5 per day

        // Verify timestamps are distributed over time
        let mut timestamps: Vec<_> = fixture.entries.iter().map(|e| e.timestamp).collect();
        timestamps.sort();

        let time_span =
            timestamps.last().unwrap().timestamp() - timestamps.first().unwrap().timestamp();
        assert!(time_span >= 6 * 24 * 3600); // At least 6 days

        // Verify cost trend exists for some entries
        let cost_entries: Vec<_> = fixture.entries.iter().filter_map(|e| e.cost_usd).collect();
        assert!(!cost_entries.is_empty());
    }

    #[test]
    fn test_multi_session_fixture() {
        let (sessions, fixture) = HistoryTestFixture::with_multi_session_data();

        assert_eq!(sessions.len(), 4);
        assert_eq!(fixture.entries.len(), 24); // 4 sessions * 6 entries each

        // Verify each session has entries
        for &session_id in &sessions {
            let session_entries = fixture
                .entries
                .iter()
                .filter(|e| e.session_id == session_id)
                .count();
            assert_eq!(session_entries, 6);
        }

        // Verify cost calculation consistency
        for entry in &fixture.entries {
            assert!(entry.cost_usd.is_some());
            assert!(entry.input_tokens.is_some());
            assert!(entry.output_tokens.is_some());
        }
    }

    #[tokio::test]
    async fn test_fixture_store_creation() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        // Verify all entries were stored
        let stats = store.get_stats(None).await.unwrap();
        assert_eq!(stats.total_entries, fixture.entries.len());

        // Verify entry retrieval works
        for entry in &fixture.entries {
            let retrieved = store.get_entry(&entry.id);
            assert!(retrieved.is_some());
            let retrieved_entry = retrieved.unwrap();
            assert_eq!(retrieved_entry.command_name, entry.command_name);
            assert_eq!(retrieved_entry.session_id, entry.session_id);
        }
    }

    #[test]
    fn test_helper_functions() {
        let session_id = Uuid::new_v4();

        // Test minimal entry
        let minimal = test_helpers::minimal_history_entry(session_id);
        assert_eq!(minimal.session_id, session_id);
        assert_eq!(minimal.command_name, "test_cmd");
        assert!(minimal.success);

        // Test maximal entry
        let maximal = test_helpers::maximal_history_entry(session_id);
        assert_eq!(maximal.session_id, session_id);
        assert!(maximal.command_name.len() > minimal.command_name.len());
        assert!(maximal.output.len() > minimal.output.len());
        assert!(maximal.args.len() > minimal.args.len());
        assert!(maximal.duration_ms > minimal.duration_ms);

        // Test edge cases
        let edge_cases = test_helpers::edge_case_entries(session_id);
        assert!(!edge_cases.is_empty());
        assert_eq!(edge_cases.len(), 5);

        // Test concurrent entries
        let concurrent = test_helpers::concurrent_test_entries(session_id, 100);
        assert_eq!(concurrent.len(), 100);

        // Test search pattern entries
        let search_patterns = test_helpers::search_pattern_entries(session_id);
        assert_eq!(search_patterns.len(), 5);

        // Test benchmark entries
        let benchmark = test_helpers::benchmark_entries(1000);
        assert_eq!(benchmark.len(), 1000);
    }

    #[test]
    fn test_search_validation() {
        let session_id = Uuid::new_v4();
        let entries = vec![
            HistoryEntry::new(
                session_id,
                "test_command".to_string(),
                vec!["arg1".to_string()],
                "success output".to_string(),
                true,
                1000,
            ),
            HistoryEntry::new(
                session_id,
                "other_command".to_string(),
                vec!["arg2".to_string()],
                "failure output".to_string(),
                false,
                2000,
            ),
        ];

        // Test success filter validation
        let success_criteria = HistorySearch {
            success_only: true,
            ..Default::default()
        };
        let success_results: Vec<_> = entries.iter().filter(|e| e.success).cloned().collect();
        assert!(test_helpers::validate_search_results(
            &success_results,
            &success_criteria
        ));

        // Test command pattern validation
        let command_criteria = HistorySearch {
            command_pattern: Some("test".to_string()),
            ..Default::default()
        };
        let command_results: Vec<_> = entries
            .iter()
            .filter(|e| e.command_name.contains("test"))
            .cloned()
            .collect();
        assert!(test_helpers::validate_search_results(
            &command_results,
            &command_criteria
        ));

        // Test invalid results should fail validation
        assert!(!test_helpers::validate_search_results(
            &entries, // All entries, but filter is for success only
            &success_criteria
        ));
    }

    #[test]
    fn test_stats_validation() {
        let session_id = Uuid::new_v4();
        let entries = vec![
            HistoryEntry::new(
                session_id,
                "cmd1".to_string(),
                vec![],
                "output1".to_string(),
                true,
                1000,
            )
            .with_cost(0.10, 100, 200, "model1".to_string()),
            HistoryEntry::new(
                session_id,
                "cmd2".to_string(),
                vec![],
                "output2".to_string(),
                false,
                2000,
            )
            .with_cost(0.20, 150, 300, "model2".to_string()),
        ];

        let stats = HistoryStats {
            total_entries: 2,
            successful_commands: 1,
            failed_commands: 1,
            success_rate: 50.0,
            total_cost: 0.30,
            total_duration_ms: 3000,
            average_duration_ms: 1500.0,
            average_cost: 0.15,
            command_counts: HashMap::new(),
            model_usage: HashMap::new(),
            date_range: (Utc::now(), Utc::now()),
        };

        assert!(test_helpers::validate_history_stats(&entries, &stats));

        // Test with invalid stats
        let invalid_stats = HistoryStats {
            total_entries: 999, // Wrong count
            ..stats.clone()
        };

        assert!(!test_helpers::validate_history_stats(
            &entries,
            &invalid_stats
        ));
    }

    proptest! {
        #[test]
        fn property_entry_creation(
            cost in property_strategies::valid_cost(),
            input_tokens in property_strategies::token_count(),
            output_tokens in property_strategies::token_count(),
            duration in property_strategies::valid_duration(),
            command in property_strategies::command_name(),
            args in property_strategies::command_args(),
            output in property_strategies::output_text(),
            success in property_strategies::success_status(),
            model in property_strategies::model_name(),
            tags in property_strategies::tag_list(),
        ) {
            let session_id = Uuid::new_v4();
            let mut entry = HistoryEntry::new(
                session_id,
                command,
                args.clone(),
                output.clone(),
                success,
                duration,
            );

            if cost > 0.0 {
                entry = entry.with_cost(cost, input_tokens, output_tokens, model.clone());
            }

            if !tags.is_empty() {
                entry = entry.with_tags(tags.clone());
            }

            // Verify properties
            prop_assert_eq!(entry.session_id, session_id);
            prop_assert_eq!(entry.args, args);
            prop_assert_eq!(entry.output, output);
            prop_assert_eq!(entry.success, success);
            prop_assert_eq!(entry.duration_ms, duration);
            prop_assert!(!entry.id.is_empty());

            if cost > 0.0 {
                prop_assert_eq!(entry.cost_usd, Some(cost));
                prop_assert_eq!(entry.input_tokens, Some(input_tokens));
                prop_assert_eq!(entry.output_tokens, Some(output_tokens));
                prop_assert_eq!(entry.model, Some(model));
            }

            if !tags.is_empty() {
                prop_assert_eq!(entry.tags, tags);
            }
        }

        #[test]
        fn property_search_criteria_consistency(
            criteria in test_helpers::random_search_criteria(Some(Uuid::new_v4())),
        ) {
            // Test that search criteria don't have contradictory settings
            if criteria.success_only && criteria.failures_only {
                // This combination should never match anything
                prop_assert!(!criteria.success_only || !criteria.failures_only);
            }

            if let (Some(since), Some(until)) = (criteria.since, criteria.until) {
                // If both dates are specified, since should be <= until for valid range
                prop_assert!(since <= until);
            }

            if let (Some(min_duration), Some(max_duration)) = (criteria.min_duration_ms, criteria.max_duration_ms) {
                prop_assert!(min_duration <= max_duration);
            }

            if let (Some(min_cost), Some(max_cost)) = (criteria.min_cost, criteria.max_cost) {
                prop_assert!(min_cost <= max_cost);
            }

            prop_assert!(criteria.limit > 0);
        }
    }
}

/// Tests for core history operations (Task 2.2)
#[cfg(test)]
mod core_operations_tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[tokio::test]
    async fn test_store_entry_valid_data() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let entry = HistoryEntry::new(
            session_id,
            "test_command".to_string(),
            vec!["--arg1".to_string(), "value1".to_string()],
            "Command executed successfully".to_string(),
            true,
            1500,
        )
        .with_cost(0.025, 100, 200, "claude-3-opus".to_string())
        .with_tags(vec!["test".to_string(), "validation".to_string()]);

        let entry_id = entry.id.clone();
        let initial_count = store.entries.len();

        store.store_entry(entry.clone()).await.unwrap();

        // Verify entry was added
        assert_eq!(store.entries.len(), initial_count + 1);
        assert!(store.index.contains_key(&entry_id));

        // Verify entry data integrity
        let stored_entry = store.get_entry(&entry_id).unwrap();
        assert_eq!(stored_entry.session_id, entry.session_id);
        assert_eq!(stored_entry.command_name, entry.command_name);
        assert_eq!(stored_entry.args, entry.args);
        assert_eq!(stored_entry.output, entry.output);
        assert_eq!(stored_entry.success, entry.success);
        assert_eq!(stored_entry.duration_ms, entry.duration_ms);
        assert_eq!(stored_entry.cost_usd, entry.cost_usd);
        assert_eq!(stored_entry.input_tokens, entry.input_tokens);
        assert_eq!(stored_entry.output_tokens, entry.output_tokens);
        assert_eq!(stored_entry.model, entry.model);
        assert_eq!(stored_entry.tags, entry.tags);
        assert!(!stored_entry.id.is_empty());
    }

    #[tokio::test]
    async fn test_store_entry_minimal_data() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let minimal_entry = HistoryEntry::new(
            session_id,
            "cmd".to_string(),
            vec![],
            "".to_string(),
            false,
            0,
        );

        let entry_id = minimal_entry.id.clone();
        store.store_entry(minimal_entry.clone()).await.unwrap();

        let stored_entry = store.get_entry(&entry_id).unwrap();
        assert_eq!(stored_entry.session_id, session_id);
        assert_eq!(stored_entry.command_name, "cmd");
        assert!(stored_entry.args.is_empty());
        assert_eq!(stored_entry.output, "");
        assert!(!stored_entry.success);
        assert_eq!(stored_entry.duration_ms, 0);
        assert!(stored_entry.cost_usd.is_none());
        assert!(stored_entry.error.is_none());
        assert!(stored_entry.tags.is_empty());
    }

    #[tokio::test]
    async fn test_store_entry_with_error() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let error_entry = HistoryEntry::new(
            session_id,
            "failing_command".to_string(),
            vec!["--fail".to_string()],
            "Partial output before failure".to_string(),
            true, // Will be overridden by with_error
            1000,
        )
        .with_error("Command failed: file not found".to_string());

        let entry_id = error_entry.id.clone();
        store.store_entry(error_entry.clone()).await.unwrap();

        let stored_entry = store.get_entry(&entry_id).unwrap();
        assert!(!stored_entry.success); // Should be false due to error
        assert_eq!(
            stored_entry.error,
            Some("Command failed: file not found".to_string())
        );
        assert_eq!(stored_entry.output, "Partial output before failure");
    }

    #[tokio::test]
    async fn test_store_entry_edge_cases() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let edge_entries = test_helpers::edge_case_entries(session_id);

        for entry in edge_entries {
            let entry_id = entry.id.clone();
            store.store_entry(entry.clone()).await.unwrap();

            let stored_entry = store.get_entry(&entry_id).unwrap();
            assert_eq!(stored_entry.id, entry_id);
            assert_eq!(stored_entry.session_id, session_id);
        }

        // Verify all edge cases were stored
        assert_eq!(store.entries.len(), 5);
    }

    #[tokio::test]
    async fn test_update_entry_existing() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let original_entry = HistoryEntry::new(
            session_id,
            "original_cmd".to_string(),
            vec!["arg1".to_string()],
            "Original output".to_string(),
            true,
            1000,
        );

        let entry_id = original_entry.id.clone();
        store.store_entry(original_entry.clone()).await.unwrap();

        // Create updated entry with same ID
        let mut updated_entry = HistoryEntry::new(
            session_id,
            "updated_cmd".to_string(),
            vec!["arg1".to_string(), "arg2".to_string()],
            "Updated output with more information".to_string(),
            true,
            2000,
        )
        .with_cost(0.030, 150, 300, "claude-3-sonnet".to_string())
        .with_tags(vec!["updated".to_string()]);

        updated_entry.id = entry_id.clone(); // Keep same ID

        // Update the entry
        store
            .update_entry(&entry_id, updated_entry.clone())
            .await
            .unwrap();

        // Verify update
        let retrieved = store.get_entry(&entry_id).unwrap();
        assert_eq!(retrieved.command_name, "updated_cmd");
        assert_eq!(retrieved.args.len(), 2);
        assert_eq!(retrieved.output, "Updated output with more information");
        assert_eq!(retrieved.duration_ms, 2000);
        assert_eq!(retrieved.cost_usd, Some(0.030));
        assert_eq!(retrieved.tags, vec!["updated".to_string()]);

        // Verify only one entry exists (update, not add)
        assert_eq!(store.entries.len(), 1);
    }

    #[tokio::test]
    async fn test_update_entry_nonexistent() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let fake_entry = HistoryEntry::new(
            session_id,
            "fake_cmd".to_string(),
            vec![],
            "output".to_string(),
            true,
            1000,
        );

        let fake_id = "nonexistent_id_12345";
        let result = store.update_entry(fake_id, fake_entry).await;

        assert!(result.is_err());
        if let Err(InteractiveError::History(msg)) = result {
            assert!(msg.contains("not found"));
        } else {
            panic!("Expected History error");
        }
    }

    #[tokio::test]
    async fn test_get_entry_existing() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        // Get first entry from fixture
        let first_entry = &fixture.entries[0];
        let retrieved = store.get_entry(&first_entry.id);

        assert!(retrieved.is_some());
        let entry = retrieved.unwrap();
        assert_eq!(entry.id, first_entry.id);
        assert_eq!(entry.command_name, first_entry.command_name);
        assert_eq!(entry.session_id, first_entry.session_id);
    }

    #[tokio::test]
    async fn test_get_entry_nonexistent() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        let fake_id = "nonexistent_id_67890";
        let retrieved = store.get_entry(fake_id);

        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_entry_id_uniqueness() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let mut entry_ids = std::collections::HashSet::new();

        // Create many entries and verify unique IDs
        for i in 0..100 {
            let entry = HistoryEntry::new(
                session_id,
                format!("cmd_{}", i),
                vec![],
                "output".to_string(),
                true,
                1000,
            );

            let entry_id = entry.id.clone();
            assert!(entry_ids.insert(entry_id.clone())); // Should be unique

            store.store_entry(entry).await.unwrap();
        }

        assert_eq!(entry_ids.len(), 100);
        assert_eq!(store.entries.len(), 100);
    }

    #[tokio::test]
    async fn test_concurrent_store_operations() {
        let fixture = HistoryTestFixture::new();
        let store = Arc::new(Mutex::new(
            HistoryStore::new(fixture.storage_path.clone()).unwrap(),
        ));
        let session_id = Uuid::new_v4();

        // Create multiple concurrent tasks
        let mut handles = vec![];

        for i in 0..50 {
            let store_clone = Arc::clone(&store);
            let handle = tokio::spawn(async move {
                let entry = HistoryEntry::new(
                    session_id,
                    format!("concurrent_cmd_{}", i),
                    vec![format!("arg_{}", i)],
                    format!("Output for task {}", i),
                    i % 10 != 0, // 90% success rate
                    100 + i as u64 * 10,
                )
                .with_cost(
                    0.001 * (i + 1) as f64,
                    10 + i as u32,
                    20 + i as u32,
                    format!("model_{}", i % 3),
                );

                let mut store_lock = store_clone.lock().await;
                store_lock.store_entry(entry).await.unwrap();
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        let store_lock = store.lock().await;
        assert_eq!(store_lock.entries.len(), 50);

        // Verify all entries are accessible
        for entry in &store_lock.entries {
            let retrieved = store_lock.get_entry(&entry.id);
            assert!(retrieved.is_some());
        }
    }

    #[tokio::test]
    async fn test_entry_timestamp_ordering() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let base_time = Utc::now() - Duration::hours(5);

        // Create entries with specific timestamps
        for i in 0..10 {
            let mut entry = HistoryEntry::new(
                session_id,
                format!("timed_cmd_{}", i),
                vec![],
                format!("Output {}", i),
                true,
                1000,
            );

            // Set specific timestamp (every hour)
            entry.timestamp = base_time + Duration::hours(i as i64);
            store.store_entry(entry).await.unwrap();
        }

        // Verify entries maintain timestamp order when retrieved
        let all_entries = &store.entries;
        let mut timestamps: Vec<_> = all_entries.iter().map(|e| e.timestamp).collect();
        let original_timestamps = timestamps.clone();
        timestamps.sort();

        // Since we added in chronological order, they should match
        assert_eq!(timestamps, original_timestamps);
    }

    #[tokio::test]
    async fn test_store_entry_data_persistence() {
        let fixture = HistoryTestFixture::new();
        let session_id = Uuid::new_v4();

        let test_entry = HistoryEntry::new(
            session_id,
            "persistent_test".to_string(),
            vec!["--persist".to_string(), "data.txt".to_string()],
            "Persistence test completed successfully".to_string(),
            true,
            2500,
        )
        .with_cost(0.045, 200, 400, "claude-3-opus".to_string())
        .with_tags(vec!["persistence".to_string(), "test".to_string()]);

        let entry_id = test_entry.id.clone();

        // Create store, add entry, then drop it
        {
            let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();
            store.store_entry(test_entry.clone()).await.unwrap();
        } // Store goes out of scope

        // Create new store instance and verify persistence
        {
            let store = HistoryStore::new(fixture.storage_path.clone()).unwrap();
            let retrieved = store.get_entry(&entry_id);

            assert!(retrieved.is_some());
            let persisted_entry = retrieved.unwrap();
            assert_eq!(persisted_entry.command_name, "persistent_test");
            assert_eq!(persisted_entry.args.len(), 2);
            assert_eq!(
                persisted_entry.output,
                "Persistence test completed successfully"
            );
            assert_eq!(persisted_entry.cost_usd, Some(0.045));
            assert_eq!(persisted_entry.tags, vec!["persistence", "test"]);
        }
    }

    #[tokio::test]
    async fn test_entry_validation_constraints() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();

        // Test various constraint validations
        let valid_entry = HistoryEntry::new(
            session_id,
            "valid_command".to_string(),
            vec!["--arg".to_string()],
            "Valid output".to_string(),
            true,
            1000,
        );

        // Should succeed
        let result = store.store_entry(valid_entry).await;
        assert!(result.is_ok());

        // Test with very long strings (should still work)
        let long_entry = HistoryEntry::new(
            session_id,
            "x".repeat(1000),
            vec!["y".repeat(500)],
            "z".repeat(10000),
            true,
            999999,
        );

        let result = store.store_entry(long_entry).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_index_consistency_after_operations() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let mut entry_ids = Vec::new();

        // Add multiple entries
        for i in 0..20 {
            let entry = HistoryEntry::new(
                session_id,
                format!("indexed_cmd_{}", i),
                vec![],
                "output".to_string(),
                true,
                1000,
            );

            entry_ids.push(entry.id.clone());
            store.store_entry(entry).await.unwrap();
        }

        // Verify index consistency
        assert_eq!(store.entries.len(), 20);
        assert_eq!(store.index.len(), 20);

        // Verify all IDs are in index
        for id in &entry_ids {
            assert!(store.index.contains_key(id));
        }

        // Update some entries and verify index remains consistent
        for (i, id) in entry_ids.iter().take(5).enumerate() {
            let mut updated_entry = HistoryEntry::new(
                session_id,
                format!("updated_cmd_{}", i),
                vec![],
                "updated output".to_string(),
                false,
                2000,
            );
            updated_entry.id = id.clone();

            store.update_entry(id, updated_entry).await.unwrap();
        }

        // Index should still be consistent
        assert_eq!(store.entries.len(), 20);
        assert_eq!(store.index.len(), 20);

        // Verify updated entries are accessible
        for (i, id) in entry_ids.iter().take(5).enumerate() {
            let retrieved = store.get_entry(id).unwrap();
            assert_eq!(retrieved.command_name, format!("updated_cmd_{}", i));
            assert!(!retrieved.success);
        }
    }

    #[tokio::test]
    async fn test_large_scale_operations() {
        let fixture = HistoryTestFixture::with_large_dataset(1000);
        let store = fixture.create_store().await;

        // Verify all entries were stored
        assert_eq!(store.entries.len(), 1000);
        assert_eq!(store.index.len(), 1000);

        // Test retrieval performance
        let start = std::time::Instant::now();
        let mut successful_retrievals = 0;

        for entry in &fixture.entries {
            if store.get_entry(&entry.id).is_some() {
                successful_retrievals += 1;
            }
        }

        let duration = start.elapsed();

        assert_eq!(successful_retrievals, 1000);
        // Should be able to retrieve 1000 entries quickly
        assert!(duration.as_millis() < 100);
    }

    proptest! {
        #[test]
        fn property_store_entry_maintains_data_integrity(
            entry in property_strategies::history_entry(Uuid::new_v4()),
        ) {
            tokio_test::block_on(async {
                let fixture = HistoryTestFixture::new();
                let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

                let entry_id = entry.id.clone();
                let original_session = entry.session_id;
                let original_command = entry.command_name.clone();
                let original_success = entry.success;

                store.store_entry(entry.clone()).await.unwrap();

                let retrieved = store.get_entry(&entry_id).unwrap();

                // Verify core data integrity
                prop_assert_eq!(&retrieved.id, &entry_id);
                prop_assert_eq!(retrieved.session_id, original_session);
                prop_assert_eq!(&retrieved.command_name, &original_command);
                prop_assert_eq!(retrieved.success, original_success);

                // Verify index consistency
                prop_assert!(store.index.contains_key(&entry_id));
                prop_assert_eq!(store.index.len(), store.entries.len());

                Ok(())
            });
        }

        #[test]
        fn property_update_entry_preserves_constraints(
            original_entry in property_strategies::history_entry(Uuid::new_v4()),
            updated_entry in property_strategies::history_entry(Uuid::new_v4()),
        ) {
            tokio_test::block_on(async {
                let fixture = HistoryTestFixture::new();
                let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

                let entry_id = original_entry.id.clone();
                store.store_entry(original_entry).await.unwrap();

                let mut update_with_same_id = updated_entry;
                update_with_same_id.id = entry_id.clone();

                store.update_entry(&entry_id, update_with_same_id.clone()).await.unwrap();

                let retrieved = store.get_entry(&entry_id).unwrap();

                // Should have updated content but same ID
                prop_assert_eq!(&retrieved.id, &entry_id);
                prop_assert_eq!(&retrieved.command_name, &update_with_same_id.command_name);
                prop_assert_eq!(retrieved.session_id, update_with_same_id.session_id);

                // Should still have exactly one entry
                prop_assert_eq!(store.entries.len(), 1);

                Ok(())
            });
        }

        #[test]
        fn property_concurrent_operations_maintain_consistency(
            entries in prop::collection::vec(
                property_strategies::history_entry(Uuid::new_v4()),
                1..20
            ),
        ) {
            tokio_test::block_on(async {
                let fixture = HistoryTestFixture::new();
                let store = Arc::new(Mutex::new(
                    HistoryStore::new(fixture.storage_path.clone()).unwrap()
                ));

                let mut handles = vec![];
                let expected_count = entries.len();

                for entry in entries {
                    let store_clone = Arc::clone(&store);
                    let handle = tokio::spawn(async move {
                        let mut store_lock = store_clone.lock().await;
                        store_lock.store_entry(entry).await.unwrap();
                    });
                    handles.push(handle);
                }

                for handle in handles {
                    handle.await.unwrap();
                }

                let store_lock = store.lock().await;

                // All entries should be stored
                prop_assert_eq!(store_lock.entries.len(), expected_count);
                prop_assert_eq!(store_lock.index.len(), expected_count);

                // All entries should be retrievable
                for entry in &store_lock.entries {
                    prop_assert!(store_lock.get_entry(&entry.id).is_some());
                }

                Ok(())
            });
        }
    }
}

/// Tests for search and query functionality (Task 2.3)
#[cfg(test)]
mod search_query_tests {
    use super::*;

    #[tokio::test]
    async fn test_search_by_command_pattern_exact() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        // Search for exact command name
        let search_criteria = HistorySearch {
            command_pattern: Some("analyze_code".to_string()),
            ..Default::default()
        };

        let results = store.search(&search_criteria).await.unwrap();

        // Should find 2 entries with "analyze_code" command
        assert_eq!(results.len(), 2);
        for result in results {
            assert!(result.command_name.contains("analyze_code"));
        }
    }

    #[tokio::test]
    async fn test_search_by_command_pattern_partial() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        // Search for partial command pattern
        let search_criteria = HistorySearch {
            command_pattern: Some("analyze".to_string()),
            ..Default::default()
        };

        let results = store.search(&search_criteria).await.unwrap();

        // Should find all entries containing "analyze"
        assert_eq!(results.len(), 2);
        for result in results {
            assert!(result.command_name.contains("analyze"));
        }
    }

    #[tokio::test]
    async fn test_search_by_output_pattern() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        // Search in output text
        let search_criteria = HistorySearch {
            output_pattern: Some("Analysis".to_string()),
            ..Default::default()
        };

        let results = store.search(&search_criteria).await.unwrap();

        // Should find entry with "Analysis" in output
        assert_eq!(results.len(), 1);
        assert!(results[0].output.contains("Analysis"));
    }

    #[tokio::test]
    async fn test_search_by_error_pattern() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        // Search for entries with errors
        let search_criteria = HistorySearch {
            error_pattern: Some("Failed".to_string()),
            ..Default::default()
        };

        let results = store.search(&search_criteria).await.unwrap();

        // Should find entries with "Failed" in error message
        assert_eq!(results.len(), 1);
        assert!(results[0].error.as_ref().unwrap().contains("Failed"));
        assert!(!results[0].success);
    }

    #[tokio::test]
    async fn test_search_case_sensitivity() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let entries = vec![
            HistoryEntry::new(
                session_id,
                "CamelCaseCommand".to_string(),
                vec![],
                "CamelCase Output".to_string(),
                true,
                1000,
            ),
            HistoryEntry::new(
                session_id,
                "lowercase_command".to_string(),
                vec![],
                "lowercase output".to_string(),
                true,
                1000,
            ),
            HistoryEntry::new(
                session_id,
                "UPPERCASE_COMMAND".to_string(),
                vec![],
                "UPPERCASE OUTPUT".to_string(),
                true,
                1000,
            ),
        ];

        for entry in entries {
            store.store_entry(entry).await.unwrap();
        }

        // Test case-sensitive search
        let camel_search = HistorySearch {
            command_pattern: Some("CamelCase".to_string()),
            ..Default::default()
        };
        let camel_results = store.search(&camel_search).await.unwrap();
        assert_eq!(camel_results.len(), 1);

        // Test partial case search
        let case_search = HistorySearch {
            command_pattern: Some("case".to_string()),
            ..Default::default()
        };
        let case_results = store.search(&case_search).await.unwrap();
        assert_eq!(case_results.len(), 1); // Only lowercase_command

        // Test uppercase search
        let upper_search = HistorySearch {
            command_pattern: Some("UPPER".to_string()),
            ..Default::default()
        };
        let upper_results = store.search(&upper_search).await.unwrap();
        assert_eq!(upper_results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_with_special_characters() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let special_entries = vec![
            HistoryEntry::new(
                session_id,
                "cmd.with.dots".to_string(),
                vec!["--regex=*.rs".to_string()],
                "Found files: main.rs, lib.rs".to_string(),
                true,
                1000,
            ),
            HistoryEntry::new(
                session_id,
                "cmd-with-dashes".to_string(),
                vec!["--pattern=[a-z]+".to_string()],
                "Pattern matched 5 items".to_string(),
                true,
                1000,
            ),
            HistoryEntry::new(
                session_id,
                "cmd_with_underscores".to_string(),
                vec!["--search=$VAR".to_string()],
                "Variable value: $VAR=test".to_string(),
                true,
                1000,
            ),
        ];

        for entry in special_entries {
            store.store_entry(entry).await.unwrap();
        }

        // Test dot patterns
        let dot_search = HistorySearch {
            command_pattern: Some(".with.".to_string()),
            ..Default::default()
        };
        let dot_results = store.search(&dot_search).await.unwrap();
        assert_eq!(dot_results.len(), 1);

        // Test dash patterns
        let dash_search = HistorySearch {
            command_pattern: Some("-with-".to_string()),
            ..Default::default()
        };
        let dash_results = store.search(&dash_search).await.unwrap();
        assert_eq!(dash_results.len(), 1);

        // Test special character in output
        let dollar_search = HistorySearch {
            output_pattern: Some("$VAR".to_string()),
            ..Default::default()
        };
        let dollar_results = store.search(&dollar_search).await.unwrap();
        assert_eq!(dollar_results.len(), 1);

        // Test pattern in arguments
        let arg_search = HistorySearch {
            output_pattern: Some("[a-z]+".to_string()), // Search in output, not args
            ..Default::default()
        };
        let arg_results = store.search(&arg_search).await.unwrap();
        assert_eq!(arg_results.len(), 0); // [a-z]+ not in output, only in args
    }

    #[tokio::test]
    async fn test_search_with_unicode() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let unicode_entries = vec![
            HistoryEntry::new(
                session_id,
                "分析代码".to_string(),
                vec!["--文件".to_string(), "测试.rs".to_string()],
                "分析完成：发现了3个潜在的改进".to_string(),
                true,
                1500,
            ),
            HistoryEntry::new(
                session_id,
                "análisis_código".to_string(),
                vec!["--archivo".to_string(), "prueba.rs".to_string()],
                "Análisis completado: se encontraron 2 mejoras".to_string(),
                true,
                1200,
            ),
            HistoryEntry::new(
                session_id,
                "코드분석".to_string(),
                vec!["--파일".to_string(), "테스트.rs".to_string()],
                "분석 완료: 1개의 개선사항 발견".to_string(),
                true,
                1800,
            ),
        ];

        for entry in unicode_entries {
            store.store_entry(entry).await.unwrap();
        }

        // Test Chinese search
        let chinese_search = HistorySearch {
            command_pattern: Some("分析".to_string()),
            ..Default::default()
        };
        let chinese_results = store.search(&chinese_search).await.unwrap();
        assert_eq!(chinese_results.len(), 1);
        assert!(chinese_results[0].command_name.contains("分析"));

        // Test Spanish search
        let spanish_search = HistorySearch {
            output_pattern: Some("Análisis".to_string()),
            ..Default::default()
        };
        let spanish_results = store.search(&spanish_search).await.unwrap();
        assert_eq!(spanish_results.len(), 1);

        // Test Korean search
        let korean_search = HistorySearch {
            command_pattern: Some("코드".to_string()),
            ..Default::default()
        };
        let korean_results = store.search(&korean_search).await.unwrap();
        assert_eq!(korean_results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_combined_patterns() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        // Test combining command and output patterns
        let combined_search = HistorySearch {
            command_pattern: Some("analyze".to_string()),
            output_pattern: Some("complete".to_string()),
            ..Default::default()
        };

        let results = store.search(&combined_search).await.unwrap();

        // Should find entries that match both patterns
        for result in &results {
            assert!(result.command_name.contains("analyze"));
            assert!(result.output.contains("complete"));
        }

        // Test command pattern with success filter
        let success_search = HistorySearch {
            command_pattern: Some("debug".to_string()),
            success_only: true,
            ..Default::default()
        };

        let success_results = store.search(&success_search).await.unwrap();
        assert_eq!(success_results.len(), 0); // debug_issue is a failure

        // Test with failure filter
        let failure_search = HistorySearch {
            command_pattern: Some("debug".to_string()),
            failures_only: true,
            ..Default::default()
        };

        let failure_results = store.search(&failure_search).await.unwrap();
        assert_eq!(failure_results.len(), 1);
        assert!(!failure_results[0].success);
    }

    #[tokio::test]
    async fn test_search_empty_patterns() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        // Test empty pattern search (should match all)
        let empty_search = HistorySearch {
            command_pattern: Some("".to_string()),
            ..Default::default()
        };

        let results = store.search(&empty_search).await.unwrap();
        assert_eq!(results.len(), fixture.entries.len());

        // Test nonexistent pattern
        let nonexistent_search = HistorySearch {
            command_pattern: Some("nonexistent_command_xyz".to_string()),
            ..Default::default()
        };

        let no_results = store.search(&nonexistent_search).await.unwrap();
        assert_eq!(no_results.len(), 0);
    }

    #[tokio::test]
    async fn test_search_with_tags() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        // Search by tags
        let tag_search = HistorySearch {
            tags: vec!["analysis".to_string()],
            ..Default::default()
        };

        let results = store.search(&tag_search).await.unwrap();

        // Should find entries with "analysis" tag
        assert!(!results.is_empty());
        for result in &results {
            assert!(result.tags.contains(&"analysis".to_string()));
        }

        // Search by multiple tags (OR operation)
        let multi_tag_search = HistorySearch {
            tags: vec!["analysis".to_string(), "documentation".to_string()],
            ..Default::default()
        };

        let multi_results = store.search(&multi_tag_search).await.unwrap();
        assert!(multi_results.len() >= results.len()); // Should have at least as many

        // Search by nonexistent tag
        let no_tag_search = HistorySearch {
            tags: vec!["nonexistent_tag".to_string()],
            ..Default::default()
        };

        let no_tag_results = store.search(&no_tag_search).await.unwrap();
        assert_eq!(no_tag_results.len(), 0);
    }

    #[tokio::test]
    async fn test_search_result_validation() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        // Test various search criteria and validate results
        let test_searches = vec![
            HistorySearch {
                command_pattern: Some("analyze".to_string()),
                success_only: true,
                ..Default::default()
            },
            HistorySearch {
                output_pattern: Some("docs".to_string()),
                failures_only: false,
                ..Default::default()
            },
            HistorySearch {
                tags: vec!["review".to_string()],
                ..Default::default()
            },
        ];

        for search_criteria in test_searches {
            let results = store.search(&search_criteria).await.unwrap();

            // Validate that all results meet the search criteria
            assert!(test_helpers::validate_search_results(
                &results,
                &search_criteria
            ));
        }
    }

    #[tokio::test]
    async fn test_search_performance_large_dataset() {
        let fixture = HistoryTestFixture::with_large_dataset(10000);
        let store = fixture.create_store().await;

        let search_criteria = HistorySearch {
            command_pattern: Some("analyze".to_string()),
            success_only: true,
            limit: 100,
            ..Default::default()
        };

        // Measure search performance
        let start = std::time::Instant::now();
        let results = store.search(&search_criteria).await.unwrap();
        let duration = start.elapsed();

        // Should complete search quickly even with large dataset
        assert!(duration.as_millis() < 500); // Should take less than 500ms

        // Verify results are limited correctly
        assert!(results.len() <= 100);

        // Verify all results match criteria
        for result in &results {
            assert!(result.command_name.contains("analyze"));
            assert!(result.success);
        }
    }

    #[tokio::test]
    async fn test_search_with_special_patterns() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let pattern_entries = test_helpers::search_pattern_entries(session_id);

        for entry in pattern_entries {
            store.store_entry(entry).await.unwrap();
        }

        // Test exact match
        let exact_search = HistorySearch {
            output_pattern: Some("exact match for searching".to_string()),
            ..Default::default()
        };
        let exact_results = store.search(&exact_search).await.unwrap();
        assert_eq!(exact_results.len(), 1);

        // Test numeric patterns
        let numeric_search = HistorySearch {
            command_pattern: Some("123".to_string()),
            ..Default::default()
        };
        let numeric_results = store.search(&numeric_search).await.unwrap();
        assert_eq!(numeric_results.len(), 1);

        // Test mixed case patterns
        let mixed_case_search = HistorySearch {
            command_pattern: Some("CaseSensitive".to_string()),
            ..Default::default()
        };
        let mixed_results = store.search(&mixed_case_search).await.unwrap();
        assert_eq!(mixed_results.len(), 1);

        // Test dot patterns
        let dot_search = HistorySearch {
            command_pattern: Some(".with.".to_string()),
            ..Default::default()
        };
        let dot_results = store.search(&dot_search).await.unwrap();
        assert_eq!(dot_results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_performance_patterns() {
        let fixture = HistoryTestFixture::with_large_dataset(5000);
        let store = fixture.create_store().await;

        // Test performance of different search patterns
        let patterns = vec![
            ("short", "cmd"),
            ("medium", "analyze_performance"),
            ("long", "extremely_comprehensive_analysis_command"),
            ("common", "test"),
            ("rare", "xyzabc123"),
        ];

        for (name, pattern) in patterns {
            let search_criteria = HistorySearch {
                command_pattern: Some(pattern.to_string()),
                limit: 1000,
                ..Default::default()
            };

            let start = std::time::Instant::now();
            let results = store.search(&search_criteria).await.unwrap();
            let duration = start.elapsed();

            println!(
                "Pattern '{}' ({}): {} results in {:?}",
                pattern,
                name,
                results.len(),
                duration
            );

            // All patterns should complete reasonably quickly
            assert!(duration.as_millis() < 1000);

            // Verify results match pattern
            for result in &results {
                assert!(result.command_name.contains(pattern));
            }
        }
    }

    #[tokio::test]
    async fn test_search_relevance_ranking() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let now = Utc::now();

        // Create entries with different relevance for "analyze"
        let entries = vec![
            // Exact command match (highest relevance)
            HistoryEntry::new(
                session_id,
                "analyze".to_string(),
                vec![],
                "Direct analyze command".to_string(),
                true,
                1000,
            ),
            // Command contains pattern
            HistoryEntry::new(
                session_id,
                "analyze_code".to_string(),
                vec![],
                "Code analysis command".to_string(),
                true,
                1000,
            ),
            // Pattern in output only
            HistoryEntry::new(
                session_id,
                "review".to_string(),
                vec![],
                "Will analyze the code thoroughly".to_string(),
                true,
                1000,
            ),
            // Older entry with same pattern (lower relevance due to age)
            {
                let mut old_entry = HistoryEntry::new(
                    session_id,
                    "analyze_old".to_string(),
                    vec![],
                    "Old analysis".to_string(),
                    true,
                    1000,
                );
                old_entry.timestamp = now - Duration::days(30);
                old_entry
            },
        ];

        for entry in entries {
            store.store_entry(entry).await.unwrap();
        }

        // Search for "analyze"
        let search_criteria = HistorySearch {
            command_pattern: Some("analyze".to_string()),
            sort_by: SortField::Timestamp,
            sort_desc: true, // Most recent first
            ..Default::default()
        };

        let results = store.search(&search_criteria).await.unwrap();

        // Should find entries with "analyze" in command name
        assert_eq!(results.len(), 3); // Excludes "review" as it only has "analyze" in output

        // Verify sorting (most recent first)
        for i in 1..results.len() {
            assert!(results[i - 1].timestamp >= results[i].timestamp);
        }
    }

    #[tokio::test]
    async fn test_search_boundary_conditions() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        // Test with maximum limit
        let max_limit_search = HistorySearch {
            limit: usize::MAX,
            ..Default::default()
        };
        let max_results = store.search(&max_limit_search).await.unwrap();
        assert_eq!(max_results.len(), fixture.entries.len());

        // Test with zero limit (edge case)
        let zero_limit_search = HistorySearch {
            limit: 0,
            ..Default::default()
        };
        let zero_results = store.search(&zero_limit_search).await.unwrap();
        assert_eq!(zero_results.len(), 0);

        // Test with very large offset
        let large_offset_search = HistorySearch {
            offset: 1000000,
            limit: 10,
            ..Default::default()
        };
        let offset_results = store.search(&large_offset_search).await.unwrap();
        assert_eq!(offset_results.len(), 0);

        // Test searching in empty store
        let empty_store = HistoryStore::new(fixture.temp_dir.path().join("empty.json")).unwrap();
        let empty_search = HistorySearch {
            command_pattern: Some("any".to_string()),
            ..Default::default()
        };
        let empty_results = empty_store.search(&empty_search).await.unwrap();
        assert_eq!(empty_results.len(), 0);
    }

    proptest! {
        #[test]
        fn property_search_consistency(
            search_criteria in test_helpers::random_search_criteria(Some(Uuid::new_v4())),
        ) {
            tokio_test::block_on(async {
                let fixture = HistoryTestFixture::new();
                let store = fixture.create_store().await;

                let results = store.search(&search_criteria).await.unwrap();

                // Search results should never exceed limit
                prop_assert!(results.len() <= search_criteria.limit);

                // All results should meet the search criteria
                prop_assert!(test_helpers::validate_search_results(&results, &search_criteria));

                // Search should be deterministic - same criteria should give same results
                let results2 = store.search(&search_criteria).await.unwrap();
                prop_assert_eq!(results.len(), results2.len());

                Ok(())
            });
        }

        #[test]
        fn property_search_performance_scales(
            pattern_len in 1usize..50usize,
            dataset_size in 100usize..1000usize,
        ) {
            tokio_test::block_on(async {
                let fixture = HistoryTestFixture::with_large_dataset(dataset_size);
                let store = fixture.create_store().await;

                let pattern = "a".repeat(pattern_len);
                let search_criteria = HistorySearch {
                    command_pattern: Some(pattern),
                    limit: 100,
                    ..Default::default()
                };

                let start = std::time::Instant::now();
                let _results = store.search(&search_criteria).await.unwrap();
                let duration = start.elapsed();

                // Search should complete within reasonable time regardless of pattern length or dataset size
                prop_assert!(duration.as_millis() < 2000); // Max 2 seconds for any search

                Ok(())
            });
        }

        #[test]
        fn property_search_patterns_never_panic(
            pattern in prop::string::string_regex(".{0,100}").unwrap(),
        ) {
            tokio_test::block_on(async {
                let fixture = HistoryTestFixture::new();
                let store = fixture.create_store().await;

                let search_criteria = HistorySearch {
                    command_pattern: Some(pattern.clone()),
                    output_pattern: Some(pattern),
                    ..Default::default()
                };

                // Should never panic regardless of input pattern
                let result = store.search(&search_criteria).await;
                prop_assert!(result.is_ok());

                Ok(())
            });
        }
    }
}

/// Tests for filtering and pagination functionality (Task 2.4)
#[cfg(test)]
mod filtering_pagination_tests {
    use super::*;

    #[tokio::test]
    async fn test_filter_by_session_id_multiple_sessions() {
        let (sessions, fixture) = HistoryTestFixture::with_multi_session_data();
        let store = fixture.create_store().await;

        // Test filtering by each session ID
        for (idx, &session_id) in sessions.iter().enumerate() {
            let search_criteria = HistorySearch {
                session_id: Some(session_id),
                ..Default::default()
            };

            let results = store.search(&search_criteria).await.unwrap();

            // Should find exactly 6 entries per session
            assert_eq!(results.len(), 6);

            // Verify all results belong to the correct session
            for entry in &results {
                assert_eq!(entry.session_id, session_id);
                assert!(entry.command_name.contains(&format!("session_{}", idx)));
            }
        }

        // Test with non-existent session ID
        let fake_session = Uuid::new_v4();
        let fake_search = HistorySearch {
            session_id: Some(fake_session),
            ..Default::default()
        };

        let fake_results = store.search(&fake_search).await.unwrap();
        assert_eq!(fake_results.len(), 0);
    }

    #[tokio::test]
    async fn test_filter_by_date_ranges() {
        let fixture = HistoryTestFixture::with_time_series_data(14); // 2 weeks of data
        let store = fixture.create_store().await;

        let now = Utc::now();
        let one_week_ago = now - Duration::days(7);
        let two_weeks_ago = now - Duration::days(14);
        let three_days_ago = now - Duration::days(3);

        // Test filtering by "since" date
        let since_search = HistorySearch {
            since: Some(one_week_ago),
            ..Default::default()
        };

        let since_results = store.search(&since_search).await.unwrap();

        // Should only find entries from the last 7 days
        for entry in &since_results {
            assert!(entry.timestamp >= one_week_ago);
        }

        // Test filtering by "until" date
        let until_search = HistorySearch {
            until: Some(three_days_ago),
            ..Default::default()
        };

        let until_results = store.search(&until_search).await.unwrap();

        // Should only find entries older than 3 days
        for entry in &until_results {
            assert!(entry.timestamp <= three_days_ago);
        }

        // Test filtering by date range
        let range_search = HistorySearch {
            since: Some(one_week_ago),
            until: Some(three_days_ago),
            ..Default::default()
        };

        let range_results = store.search(&range_search).await.unwrap();

        // Should only find entries between 7 and 3 days ago
        for entry in &range_results {
            assert!(entry.timestamp >= one_week_ago);
            assert!(entry.timestamp <= three_days_ago);
        }

        // Test with future dates (should return empty)
        let future_search = HistorySearch {
            since: Some(now + Duration::days(1)),
            ..Default::default()
        };

        let future_results = store.search(&future_search).await.unwrap();
        assert_eq!(future_results.len(), 0);
    }

    #[tokio::test]
    async fn test_filter_by_time_periods() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();
        let base_time = Utc::now();

        // Create entries at specific time intervals
        let time_entries = vec![
            (base_time - Duration::minutes(5), "recent_cmd"),
            (base_time - Duration::hours(2), "hours_ago_cmd"),
            (base_time - Duration::days(1), "yesterday_cmd"),
            (base_time - Duration::days(7), "last_week_cmd"),
            (base_time - Duration::days(30), "last_month_cmd"),
        ];

        for (timestamp, cmd_name) in time_entries {
            let mut entry = HistoryEntry::new(
                session_id,
                cmd_name.to_string(),
                vec![],
                format!("Output for {}", cmd_name),
                true,
                1000,
            );
            entry.timestamp = timestamp;
            store.store_entry(entry).await.unwrap();
        }

        // Test last hour filter
        let last_hour = HistorySearch {
            since: Some(base_time - Duration::hours(1)),
            ..Default::default()
        };
        let hour_results = store.search(&last_hour).await.unwrap();
        assert_eq!(hour_results.len(), 1);
        assert_eq!(hour_results[0].command_name, "recent_cmd");

        // Test last 24 hours filter
        let last_day = HistorySearch {
            since: Some(base_time - Duration::days(1)),
            ..Default::default()
        };
        let day_results = store.search(&last_day).await.unwrap();
        assert_eq!(day_results.len(), 3); // recent, hours_ago, and yesterday

        // Test last week filter
        let last_week = HistorySearch {
            since: Some(base_time - Duration::days(7)),
            ..Default::default()
        };
        let week_results = store.search(&last_week).await.unwrap();
        assert_eq!(week_results.len(), 4); // All except last_month
    }

    #[tokio::test]
    async fn test_filter_by_command_types() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();

        // Create entries with different command types
        let command_types = vec![
            ("analyze_code", "analysis"),
            ("analyze_performance", "analysis"),
            ("generate_docs", "generation"),
            ("generate_tests", "generation"),
            ("review_pr", "review"),
            ("debug_issue", "debugging"),
        ];

        for (cmd_name, tag) in command_types {
            let entry = HistoryEntry::new(
                session_id,
                cmd_name.to_string(),
                vec![],
                format!("Output for {}", cmd_name),
                true,
                1000,
            )
            .with_tags(vec![tag.to_string()]);

            store.store_entry(entry).await.unwrap();
        }

        // Test filtering by command prefix
        let analyze_search = HistorySearch {
            command_pattern: Some("analyze".to_string()),
            ..Default::default()
        };
        let analyze_results = store.search(&analyze_search).await.unwrap();
        assert_eq!(analyze_results.len(), 2);

        // Test filtering by tags
        let generation_search = HistorySearch {
            tags: vec!["generation".to_string()],
            ..Default::default()
        };
        let generation_results = store.search(&generation_search).await.unwrap();
        assert_eq!(generation_results.len(), 2);

        // Test combining command pattern and tags
        let combined_search = HistorySearch {
            command_pattern: Some("generate".to_string()),
            tags: vec!["generation".to_string()],
            ..Default::default()
        };
        let combined_results = store.search(&combined_search).await.unwrap();
        assert_eq!(combined_results.len(), 2);
    }

    #[tokio::test]
    async fn test_filter_by_execution_status() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        // Test success_only filter
        let success_search = HistorySearch {
            success_only: true,
            ..Default::default()
        };
        let success_results = store.search(&success_search).await.unwrap();

        // Verify all results are successful
        for entry in &success_results {
            assert!(entry.success);
            assert!(entry.error.is_none());
        }

        // Test failures_only filter
        let failure_search = HistorySearch {
            failures_only: true,
            ..Default::default()
        };
        let failure_results = store.search(&failure_search).await.unwrap();

        // Verify all results are failures
        for entry in &failure_results {
            assert!(!entry.success);
            assert!(entry.error.is_some());
        }

        // Verify counts match
        assert_eq!(
            success_results.len() + failure_results.len(),
            fixture.entries.len()
        );

        // Test combining with other filters
        let failed_debug_search = HistorySearch {
            command_pattern: Some("debug".to_string()),
            failures_only: true,
            ..Default::default()
        };
        let failed_debug_results = store.search(&failed_debug_search).await.unwrap();
        assert_eq!(failed_debug_results.len(), 1);
        assert_eq!(failed_debug_results[0].command_name, "debug_issue");
    }

    #[tokio::test]
    async fn test_filter_by_duration() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();

        // Create entries with different durations
        let durations = vec![
            (100, "quick_cmd"),
            (500, "normal_cmd"),
            (1000, "slow_cmd"),
            (5000, "very_slow_cmd"),
            (10000, "extremely_slow_cmd"),
        ];

        for (duration_ms, cmd_name) in durations {
            let entry = HistoryEntry::new(
                session_id,
                cmd_name.to_string(),
                vec![],
                format!("Output for {} ({}ms)", cmd_name, duration_ms),
                true,
                duration_ms,
            );
            store.store_entry(entry).await.unwrap();
        }

        // Test minimum duration filter
        let min_duration_search = HistorySearch {
            min_duration_ms: Some(1000),
            ..Default::default()
        };
        let min_results = store.search(&min_duration_search).await.unwrap();
        assert_eq!(min_results.len(), 3); // slow, very_slow, extremely_slow

        // Test maximum duration filter
        let max_duration_search = HistorySearch {
            max_duration_ms: Some(1000),
            ..Default::default()
        };
        let max_results = store.search(&max_duration_search).await.unwrap();
        assert_eq!(max_results.len(), 3); // quick, normal, slow

        // Test duration range
        let range_search = HistorySearch {
            min_duration_ms: Some(500),
            max_duration_ms: Some(5000),
            ..Default::default()
        };
        let range_results = store.search(&range_search).await.unwrap();
        assert_eq!(range_results.len(), 3); // normal, slow, very_slow
    }

    #[tokio::test]
    async fn test_filter_by_cost() {
        let fixture = HistoryTestFixture::new();
        let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        let session_id = Uuid::new_v4();

        // Create entries with different costs
        let costs = vec![
            (None, "free_cmd"),
            (Some(0.001), "cheap_cmd"),
            (Some(0.01), "normal_cmd"),
            (Some(0.1), "expensive_cmd"),
            (Some(1.0), "very_expensive_cmd"),
        ];

        for (cost, cmd_name) in costs {
            let mut entry = HistoryEntry::new(
                session_id,
                cmd_name.to_string(),
                vec![],
                format!("Output for {}", cmd_name),
                true,
                1000,
            );

            if let Some(cost_usd) = cost {
                entry = entry.with_cost(cost_usd, 100, 200, "model".to_string());
            }

            store.store_entry(entry).await.unwrap();
        }

        // Test minimum cost filter (excludes free entries)
        let min_cost_search = HistorySearch {
            min_cost: Some(0.01),
            ..Default::default()
        };
        let min_results = store.search(&min_cost_search).await.unwrap();
        assert_eq!(min_results.len(), 3); // normal, expensive, very_expensive

        // Test maximum cost filter
        let max_cost_search = HistorySearch {
            max_cost: Some(0.1),
            ..Default::default()
        };
        let max_results = store.search(&max_cost_search).await.unwrap();
        assert_eq!(max_results.len(), 4); // free, cheap, normal, expensive

        // Test cost range
        let range_search = HistorySearch {
            min_cost: Some(0.001),
            max_cost: Some(0.1),
            ..Default::default()
        };
        let range_results = store.search(&range_search).await.unwrap();
        assert_eq!(range_results.len(), 3); // cheap, normal, expensive
    }

    #[tokio::test]
    async fn test_filter_by_model() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        // Test filtering by specific model
        let opus_search = HistorySearch {
            model: Some("claude-3-opus".to_string()),
            ..Default::default()
        };
        let opus_results = store.search(&opus_search).await.unwrap();

        // Verify all results use the specified model
        for entry in &opus_results {
            assert_eq!(entry.model, Some("claude-3-opus".to_string()));
        }

        // Test with non-existent model
        let fake_model_search = HistorySearch {
            model: Some("non-existent-model".to_string()),
            ..Default::default()
        };
        let fake_results = store.search(&fake_model_search).await.unwrap();
        assert_eq!(fake_results.len(), 0);
    }

    #[tokio::test]
    async fn test_pagination_with_various_page_sizes() {
        let fixture = HistoryTestFixture::with_large_dataset(100);
        let store = fixture.create_store().await;

        // Test different page sizes
        let page_sizes = vec![1, 5, 10, 20, 50];

        for page_size in page_sizes {
            let mut all_ids = Vec::new();
            let mut page_num = 0;

            loop {
                let search = HistorySearch {
                    limit: page_size,
                    offset: page_num * page_size,
                    ..Default::default()
                };

                let results = store.search(&search).await.unwrap();

                if results.is_empty() {
                    break;
                }

                // Verify page size (except possibly last page)
                if (page_num + 1) * page_size <= 100 {
                    assert_eq!(results.len(), page_size);
                } else {
                    assert!(results.len() <= page_size);
                }

                // Collect IDs
                all_ids.extend(results.iter().map(|e| e.id.clone()));

                page_num += 1;
            }

            // Verify we got all entries
            assert_eq!(all_ids.len(), 100);

            // Verify no duplicates
            let unique_ids: std::collections::HashSet<_> = all_ids.iter().collect();
            assert_eq!(unique_ids.len(), 100);
        }
    }

    #[tokio::test]
    async fn test_pagination_edge_cases() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        let total_entries = fixture.entries.len();

        // Test empty result set (offset beyond total)
        let beyond_search = HistorySearch {
            limit: 10,
            offset: total_entries + 10,
            ..Default::default()
        };
        let beyond_results = store.search(&beyond_search).await.unwrap();
        assert_eq!(beyond_results.len(), 0);

        // Test single page (limit > total)
        let single_page_search = HistorySearch {
            limit: total_entries + 10,
            offset: 0,
            ..Default::default()
        };
        let single_page_results = store.search(&single_page_search).await.unwrap();
        assert_eq!(single_page_results.len(), total_entries);

        // Test offset at boundary
        let boundary_search = HistorySearch {
            limit: 10,
            offset: total_entries - 1,
            ..Default::default()
        };
        let boundary_results = store.search(&boundary_search).await.unwrap();
        assert_eq!(boundary_results.len(), 1);

        // Test limit of 0 (edge case)
        let zero_limit_search = HistorySearch {
            limit: 0,
            offset: 0,
            ..Default::default()
        };
        let zero_results = store.search(&zero_limit_search).await.unwrap();
        assert_eq!(zero_results.len(), 0);

        // Test with filtering and pagination
        let filtered_page_search = HistorySearch {
            success_only: true,
            limit: 2,
            offset: 1,
            ..Default::default()
        };
        let filtered_results = store.search(&filtered_page_search).await.unwrap();
        assert!(filtered_results.len() <= 2);
        for entry in &filtered_results {
            assert!(entry.success);
        }
    }

    #[tokio::test]
    async fn test_pagination_consistency() {
        let fixture = HistoryTestFixture::with_large_dataset(50);
        let store = fixture.create_store().await;

        // Get all results at once
        let all_search = HistorySearch {
            limit: 50,
            offset: 0,
            sort_by: SortField::Timestamp,
            sort_desc: true,
            ..Default::default()
        };
        let all_results = store.search(&all_search).await.unwrap();

        // Get results in pages
        let mut paged_results = Vec::new();
        for page in 0..5 {
            let page_search = HistorySearch {
                limit: 10,
                offset: page * 10,
                sort_by: SortField::Timestamp,
                sort_desc: true,
                ..Default::default()
            };
            let page_results = store.search(&page_search).await.unwrap();
            paged_results.extend(page_results);
        }

        // Verify consistency
        assert_eq!(all_results.len(), paged_results.len());
        for (i, (all_entry, paged_entry)) in
            all_results.iter().zip(paged_results.iter()).enumerate()
        {
            assert_eq!(all_entry.id, paged_entry.id, "Mismatch at index {}", i);
        }
    }

    #[tokio::test]
    async fn test_complex_filtering_combinations() {
        let fixture = HistoryTestFixture::with_large_dataset(200);
        let store = fixture.create_store().await;

        // Complex filter: successful analyze commands with cost > 0.01,
        // duration < 2000ms, from the last 7 days
        let complex_search = HistorySearch {
            command_pattern: Some("analyze".to_string()),
            success_only: true,
            min_cost: Some(0.01),
            max_duration_ms: Some(2000),
            since: Some(Utc::now() - Duration::days(7)),
            tags: vec!["urgent".to_string(), "review".to_string()],
            limit: 20,
            offset: 0,
            sort_by: SortField::Cost,
            sort_desc: true,
            ..Default::default()
        };

        let results = store.search(&complex_search).await.unwrap();

        // Verify all results match all criteria
        for entry in &results {
            assert!(entry.command_name.contains("analyze"));
            assert!(entry.success);
            assert!(entry.cost_usd.unwrap_or(0.0) >= 0.01);
            assert!(entry.duration_ms <= 2000);
            assert!(entry.timestamp >= Utc::now() - Duration::days(7));
            assert!(
                entry.tags.contains(&"urgent".to_string())
                    || entry.tags.contains(&"review".to_string())
            );
        }

        // Verify sorting by cost (descending)
        for i in 1..results.len() {
            let prev_cost = results[i - 1].cost_usd.unwrap_or(0.0);
            let curr_cost = results[i].cost_usd.unwrap_or(0.0);
            assert!(prev_cost >= curr_cost);
        }
    }

    #[tokio::test]
    async fn test_filter_performance_with_large_dataset() {
        let fixture = HistoryTestFixture::with_large_dataset(10000);
        let store = fixture.create_store().await;

        // Test various filter combinations and measure performance
        let filter_tests = vec![
            (
                "Simple command filter",
                HistorySearch {
                    command_pattern: Some("analyze".to_string()),
                    limit: 100,
                    ..Default::default()
                },
            ),
            (
                "Date range filter",
                HistorySearch {
                    since: Some(Utc::now() - Duration::days(7)),
                    until: Some(Utc::now()),
                    limit: 100,
                    ..Default::default()
                },
            ),
            (
                "Complex multi-filter",
                HistorySearch {
                    command_pattern: Some("test".to_string()),
                    success_only: true,
                    min_cost: Some(0.001),
                    max_duration_ms: Some(5000),
                    tags: vec!["urgent".to_string()],
                    limit: 100,
                    ..Default::default()
                },
            ),
        ];

        for (name, search_criteria) in filter_tests {
            let start = std::time::Instant::now();
            let results = store.search(&search_criteria).await.unwrap();
            let duration = start.elapsed();

            println!("{}: {} results in {:?}", name, results.len(), duration);

            // All filters should complete quickly even with 10k entries
            assert!(duration.as_millis() < 1000);

            // Verify results are valid
            assert!(test_helpers::validate_search_results(
                &results,
                &search_criteria
            ));
        }
    }

    proptest! {
        #[test]
        fn property_filter_combinations_valid(
            criteria in test_helpers::random_search_criteria(Some(Uuid::new_v4())),
        ) {
            tokio_test::block_on(async {
                let fixture = HistoryTestFixture::with_large_dataset(100);
                let store = fixture.create_store().await;

                let results = store.search(&criteria).await.unwrap();

                // All results should match the criteria
                prop_assert!(test_helpers::validate_search_results(&results, &criteria));

                // Results should respect pagination
                prop_assert!(results.len() <= criteria.limit);

                // Contradictory filters should return empty results
                if criteria.success_only && criteria.failures_only {
                    prop_assert_eq!(results.len(), 0);
                }

                Ok(())
            });
        }

        #[test]
        fn property_pagination_consistency(
            total_items in 10usize..100usize,
            page_size in 1usize..20usize,
        ) {
            tokio_test::block_on(async {
                let fixture = HistoryTestFixture::with_large_dataset(total_items);
                let store = fixture.create_store().await;

                let mut all_ids = std::collections::HashSet::new();
                let mut page = 0;

                // Collect all items through pagination
                loop {
                    let search = HistorySearch {
                        limit: page_size,
                        offset: page * page_size,
                        ..Default::default()
                    };

                    let results = store.search(&search).await.unwrap();

                    if results.is_empty() {
                        break;
                    }

                    // Add IDs to set
                    for entry in results {
                        all_ids.insert(entry.id);
                    }

                    page += 1;
                }

                // Should have collected all unique items
                prop_assert_eq!(all_ids.len(), total_items);

                Ok(())
            });
        }
    }
}

/// Tests for storage management functionality (Task 2.5)
#[cfg(test)]
mod storage_management_tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::fs;
    use std::io::{Seek, Write};
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[tokio::test]
    async fn test_storage_rotation_size_limits() {
        let fixture = HistoryTestFixture::new();
        let storage_path = fixture.storage_path.clone();

        // Create a store with many entries to simulate large file
        let mut store = HistoryStore::new(storage_path.clone()).unwrap();
        let session_id = Uuid::new_v4();

        // Add entries until storage size exceeds a threshold
        for i in 0..1000 {
            let entry = HistoryEntry::new(
                session_id,
                format!("large_entry_{}", i),
                vec![format!("arg_{}", i); 10], // Multiple args to increase size
                "x".repeat(1000),               // Large output
                true,
                1000,
            )
            .with_cost(0.001 * i as f64, 100, 200, "model".to_string())
            .with_tags(vec![format!("tag_{}", i % 10); 5]); // Multiple tags

            store.store_entry(entry).await.unwrap();
        }

        // Check file size
        let metadata = tokio::fs::metadata(&storage_path).await.unwrap();
        let file_size = metadata.len();

        println!("Storage file size after 1000 entries: {} bytes", file_size);
        assert!(file_size > 100_000); // Should be reasonably large

        // Simulate rotation by archiving old entries
        let archive_path = storage_path.with_extension("archive.json");

        // Move half of the entries to archive
        let entries_to_archive = store.entries.len() / 2;
        let archived_entries: Vec<_> = store.entries.drain(..entries_to_archive).collect();

        // Save archived entries
        let archive_content = serde_json::to_string_pretty(&archived_entries).unwrap();
        tokio::fs::write(&archive_path, archive_content)
            .await
            .unwrap();

        // Rebuild index after rotation
        store.rebuild_index();

        // Save remaining entries
        store.save_entries().await.unwrap();

        // Verify rotation worked
        assert_eq!(store.entries.len(), 1000 - entries_to_archive);
        assert!(archive_path.exists());

        // Verify we can still access remaining entries
        for entry in &store.entries {
            let retrieved = store.get_entry(&entry.id);
            assert!(retrieved.is_some());
        }
    }

    #[tokio::test]
    async fn test_backup_creation_and_restoration() {
        let fixture = HistoryTestFixture::new();
        let store = fixture.create_store().await;

        let backup_dir = fixture.temp_dir.path().join("backups");
        tokio::fs::create_dir_all(&backup_dir).await.unwrap();

        // Create backup with timestamp
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let backup_path = backup_dir.join(format!("history_backup_{}.json", timestamp));

        // Backup current state
        store
            .export(&backup_path, ExportFormat::Json, None)
            .await
            .unwrap();

        assert!(backup_path.exists());

        // Verify backup content
        let backup_content = tokio::fs::read_to_string(&backup_path).await.unwrap();
        let backup_entries: Vec<HistoryEntry> = serde_json::from_str(&backup_content).unwrap();

        assert_eq!(backup_entries.len(), fixture.entries.len());

        // Simulate data loss - create new empty store
        let mut restored_store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

        // Clear any existing data
        restored_store.clear_all().await.unwrap();
        assert_eq!(restored_store.entries.len(), 0);

        // Restore from backup
        for entry in backup_entries {
            restored_store.store_entry(entry).await.unwrap();
        }

        // Verify restoration
        assert_eq!(restored_store.entries.len(), fixture.entries.len());

        // Verify data integrity after restoration
        for original_entry in &fixture.entries {
            let restored_entry = restored_store.get_entry(&original_entry.id);
            assert!(restored_entry.is_some());

            let restored = restored_entry.unwrap();
            assert_eq!(restored.command_name, original_entry.command_name);
            assert_eq!(restored.session_id, original_entry.session_id);
            assert_eq!(restored.output, original_entry.output);
        }
    }

    #[tokio::test]
    async fn test_incremental_backup() {
        let fixture = HistoryTestFixture::new();
        let mut store = fixture.create_store().await;

        let backup_dir = fixture.temp_dir.path().join("incremental_backups");
        tokio::fs::create_dir_all(&backup_dir).await.unwrap();

        // Create initial full backup
        let full_backup_path = backup_dir.join("full_backup.json");
        store
            .export(&full_backup_path, ExportFormat::Json, None)
            .await
            .unwrap();

        let initial_count = store.entries.len();

        // Track the last backup timestamp
        let last_backup_time = Utc::now();

        // Add new entries
        let session_id = Uuid::new_v4();
        for i in 0..5 {
            let entry = HistoryEntry::new(
                session_id,
                format!("new_cmd_{}", i),
                vec![],
                format!("New output {}", i),
                true,
                1000,
            );
            store.store_entry(entry).await.unwrap();
        }

        // Create incremental backup (only new entries since last backup)
        let incremental_search = HistorySearch {
            since: Some(last_backup_time),
            ..Default::default()
        };

        let incremental_backup_path = backup_dir.join("incremental_backup.json");
        store
            .export(
                &incremental_backup_path,
                ExportFormat::Json,
                Some(&incremental_search),
            )
            .await
            .unwrap();

        // Verify incremental backup contains only new entries
        let incremental_content = tokio::fs::read_to_string(&incremental_backup_path)
            .await
            .unwrap();
        let incremental_entries: Vec<HistoryEntry> =
            serde_json::from_str(&incremental_content).unwrap();

        assert_eq!(incremental_entries.len(), 5);
        for entry in &incremental_entries {
            assert!(entry.command_name.starts_with("new_cmd_"));
        }
    }

    #[tokio::test]
    async fn test_storage_cleanup_operations() {
        let fixture = HistoryTestFixture::with_time_series_data(30); // 30 days of data
        let mut store = fixture.create_store().await;

        let initial_count = store.entries.len();

        // Test pruning old entries (older than 14 days)
        let pruned_count = store.prune_old_entries(14).await.unwrap();

        assert!(pruned_count > 0);
        assert_eq!(store.entries.len(), initial_count - pruned_count);

        // Verify only recent entries remain
        for entry in &store.entries {
            assert!(entry.timestamp > Utc::now() - Duration::days(14));
        }

        // Test clearing specific session
        let session_id = store.entries.first().unwrap().session_id;
        let session_entries_before = store
            .entries
            .iter()
            .filter(|e| e.session_id == session_id)
            .count();

        store.clear_session(session_id).await.unwrap();

        let session_entries_after = store
            .entries
            .iter()
            .filter(|e| e.session_id == session_id)
            .count();

        assert_eq!(session_entries_after, 0);
        assert!(store.entries.len() < initial_count - pruned_count);
    }

    #[tokio::test]
    async fn test_archival_operations() {
        let fixture = HistoryTestFixture::with_large_dataset(500);
        let mut store = fixture.create_store().await;

        let archive_dir = fixture.temp_dir.path().join("archives");
        tokio::fs::create_dir_all(&archive_dir).await.unwrap();

        // Archive entries older than 7 days
        let cutoff_date = Utc::now() - Duration::days(7);
        let mut entries_to_archive = Vec::new();

        store.entries.retain(|entry| {
            if entry.timestamp < cutoff_date {
                entries_to_archive.push(entry.clone());
                false
            } else {
                true
            }
        });

        // Rebuild index after removing entries
        store.rebuild_index();

        // Save archived entries with date-based filename
        let archive_filename = format!(
            "history_archive_{}_to_{}.json",
            entries_to_archive
                .first()
                .map(|e| e.timestamp.format("%Y%m%d").to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            entries_to_archive
                .last()
                .map(|e| e.timestamp.format("%Y%m%d").to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );

        let archive_path = archive_dir.join(archive_filename);
        let archive_content = serde_json::to_string_pretty(&entries_to_archive).unwrap();
        tokio::fs::write(&archive_path, archive_content)
            .await
            .unwrap();

        // Save updated store
        store.save_entries().await.unwrap();

        // Verify archival
        assert!(archive_path.exists());
        assert!(entries_to_archive.len() > 0);

        // Verify we can read archived data
        let archived_content = tokio::fs::read_to_string(&archive_path).await.unwrap();
        let archived_entries: Vec<HistoryEntry> = serde_json::from_str(&archived_content).unwrap();
        assert_eq!(archived_entries.len(), entries_to_archive.len());
    }

    #[tokio::test]
    async fn test_storage_corruption_detection() {
        let fixture = HistoryTestFixture::new();
        let storage_path = fixture.storage_path.clone();

        // Create initial valid store
        {
            let store = fixture.create_store().await;
            // Store is automatically saved
        }

        // Corrupt the storage file
        let mut file = fs::OpenOptions::new()
            .write(true)
            .truncate(false)
            .open(&storage_path)
            .unwrap();

        // Write invalid JSON in the middle of the file
        file.seek(std::io::SeekFrom::Start(100)).unwrap();
        file.write_all(b"CORRUPTED_DATA_HERE").unwrap();
        file.sync_all().unwrap();
        drop(file);

        // Try to load corrupted store
        let load_result = HistoryStore::new(storage_path.clone());

        // Should fail to load due to corruption
        assert!(load_result.is_err());

        // Verify error is about parsing/corruption
        if let Err(e) = load_result {
            let error_msg = format!("{}", e);
            assert!(error_msg.contains("Failed to parse") || error_msg.contains("parse"));
        }
    }

    #[tokio::test]
    async fn test_storage_recovery_from_corruption() {
        let fixture = HistoryTestFixture::new();
        let storage_path = fixture.storage_path.clone();
        let backup_path = storage_path.with_extension("backup");

        // Create initial store with data
        let original_entries = {
            let store = fixture.create_store().await;

            // Create automatic backup
            store
                .export(&backup_path, ExportFormat::Json, None)
                .await
                .unwrap();

            store.entries.clone()
        };

        // Corrupt the main storage file
        tokio::fs::write(&storage_path, "{ INVALID JSON DATA")
            .await
            .unwrap();

        // Implement recovery logic
        let recovered_store = match HistoryStore::new(storage_path.clone()) {
            Ok(store) => store,
            Err(_) => {
                // Recovery: Try to load from backup
                println!("Main storage corrupted, attempting recovery from backup...");

                if backup_path.exists() {
                    // Read backup data
                    let backup_content = tokio::fs::read_to_string(&backup_path).await.unwrap();
                    let backup_entries: Vec<HistoryEntry> =
                        serde_json::from_str(&backup_content).unwrap();

                    // Create new store and restore from backup
                    let mut store = HistoryStore::new(storage_path.clone()).unwrap_or_else(|_| {
                        // If still can't create, use a recovery path
                        let recovery_path = storage_path.with_extension("recovered");
                        HistoryStore::new(recovery_path).unwrap()
                    });

                    // Clear any partial data
                    store.entries.clear();
                    store.index.clear();

                    // Restore from backup
                    for entry in backup_entries {
                        store.store_entry(entry).await.unwrap();
                    }

                    store
                } else {
                    panic!("No backup available for recovery");
                }
            }
        };

        // Verify recovery
        assert_eq!(recovered_store.entries.len(), original_entries.len());

        for original_entry in &original_entries {
            let recovered_entry = recovered_store.get_entry(&original_entry.id);
            assert!(recovered_entry.is_some());
        }
    }

    #[tokio::test]
    async fn test_partial_corruption_recovery() {
        let fixture = HistoryTestFixture::new();
        let storage_path = fixture.storage_path.clone();

        // Create store with multiple entries
        let mut store = HistoryStore::new(storage_path.clone()).unwrap();
        let session_id = Uuid::new_v4();

        // Add entries and track their JSON representation
        let mut valid_entries = Vec::new();
        for i in 0..10 {
            let entry = HistoryEntry::new(
                session_id,
                format!("cmd_{}", i),
                vec![],
                format!("Output {}", i),
                true,
                1000,
            );

            valid_entries.push(entry.clone());
            store.store_entry(entry).await.unwrap();
        }

        // Create a partially corrupted file (valid JSON array but some corrupted entries)
        let mut partial_data = vec![];

        // Add first 5 valid entries
        for entry in &valid_entries[..5] {
            partial_data.push(serde_json::to_value(entry).unwrap());
        }

        // Add corrupted entry (missing required fields)
        partial_data.push(serde_json::json!({
            "id": "corrupted_entry",
            "command_name": "bad_cmd"
            // Missing required fields
        }));

        // Add remaining valid entries
        for entry in &valid_entries[6..] {
            partial_data.push(serde_json::to_value(entry).unwrap());
        }

        // Write partially corrupted data
        let corrupted_content = serde_json::to_string_pretty(&partial_data).unwrap();
        tokio::fs::write(&storage_path, corrupted_content)
            .await
            .unwrap();

        // Try to load with recovery logic
        let result = std::panic::catch_unwind(|| {
            // This would be the actual recovery implementation
            let content = std::fs::read_to_string(&storage_path).unwrap();
            let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

            if let serde_json::Value::Array(entries) = parsed {
                let mut recovered_entries = Vec::new();
                let mut skipped_count = 0;

                for (idx, entry_value) in entries.iter().enumerate() {
                    match serde_json::from_value::<HistoryEntry>(entry_value.clone()) {
                        Ok(entry) => recovered_entries.push(entry),
                        Err(e) => {
                            println!("Skipping corrupted entry at index {}: {}", idx, e);
                            skipped_count += 1;
                        }
                    }
                }

                println!(
                    "Recovered {} entries, skipped {} corrupted entries",
                    recovered_entries.len(),
                    skipped_count
                );

                (recovered_entries, skipped_count)
            } else {
                panic!("Invalid storage format");
            }
        });

        assert!(result.is_ok());
        let (recovered_entries, skipped_count) = result.unwrap();

        // Should recover 9 valid entries and skip 1 corrupted
        assert_eq!(recovered_entries.len(), 9);
        assert_eq!(skipped_count, 1);
    }

    #[tokio::test]
    async fn test_automatic_backup_on_operations() {
        let fixture = HistoryTestFixture::new();
        let storage_path = fixture.storage_path.clone();
        let auto_backup_path = storage_path.with_extension("auto_backup");

        let mut store = HistoryStore::new(storage_path.clone()).unwrap();
        let session_id = Uuid::new_v4();

        // Simulate automatic backup every N operations
        let backup_frequency = 5;
        let mut operation_count = 0;

        for i in 0..20 {
            let entry = HistoryEntry::new(
                session_id,
                format!("cmd_{}", i),
                vec![],
                format!("Output {}", i),
                true,
                1000,
            );

            store.store_entry(entry).await.unwrap();
            operation_count += 1;

            // Create automatic backup
            if operation_count % backup_frequency == 0 {
                store
                    .export(&auto_backup_path, ExportFormat::Json, None)
                    .await
                    .unwrap();
                println!(
                    "Automatic backup created after {} operations",
                    operation_count
                );
            }
        }

        // Verify backup exists and is recent
        assert!(auto_backup_path.exists());

        let backup_content = tokio::fs::read_to_string(&auto_backup_path).await.unwrap();
        let backup_entries: Vec<HistoryEntry> = serde_json::from_str(&backup_content).unwrap();

        // Backup should have entries up to the last backup point
        assert_eq!(backup_entries.len(), 20); // Last backup was after all 20 entries
    }

    #[tokio::test]
    async fn test_storage_migration() {
        let fixture = HistoryTestFixture::new();
        let v1_path = fixture.temp_dir.path().join("history_v1.json");
        let v2_path = fixture.temp_dir.path().join("history_v2.json");

        // Create old format data (simulated v1)
        #[derive(Serialize, Deserialize)]
        struct HistoryEntryV1 {
            id: String,
            session_id: String, // String instead of UUID
            command: String,    // Different field name
            output: String,
            success: bool,
            timestamp: DateTime<Utc>,
        }

        let old_entries = vec![HistoryEntryV1 {
            id: Uuid::new_v4().to_string(),
            session_id: Uuid::new_v4().to_string(),
            command: "old_format_cmd".to_string(),
            output: "Old format output".to_string(),
            success: true,
            timestamp: Utc::now(),
        }];

        // Write v1 format
        let v1_content = serde_json::to_string_pretty(&old_entries).unwrap();
        tokio::fs::write(&v1_path, v1_content).await.unwrap();

        // Simulate migration logic
        let migrated_entries: Vec<HistoryEntry> = {
            let v1_data = tokio::fs::read_to_string(&v1_path).await.unwrap();
            let v1_entries: Vec<HistoryEntryV1> = serde_json::from_str(&v1_data).unwrap();

            v1_entries
                .into_iter()
                .map(|v1| {
                    // Migrate from v1 to current format
                    HistoryEntry {
                        id: v1.id,
                        session_id: Uuid::parse_str(&v1.session_id)
                            .unwrap_or_else(|_| Uuid::new_v4()),
                        command_name: v1.command, // Field rename
                        args: vec![],             // Default empty args
                        output: v1.output,
                        error: None,
                        cost_usd: None,
                        input_tokens: None,
                        output_tokens: None,
                        timestamp: v1.timestamp,
                        duration_ms: 0, // Default duration
                        success: v1.success,
                        model: None,
                        tags: vec![],
                    }
                })
                .collect()
        };

        // Save migrated data
        let mut store = HistoryStore::new(v2_path).unwrap();
        for entry in migrated_entries {
            store.store_entry(entry).await.unwrap();
        }

        // Verify migration
        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.entries[0].command_name, "old_format_cmd");
    }

    #[tokio::test]
    async fn test_concurrent_storage_operations() {
        let fixture = HistoryTestFixture::new();
        let storage_path = fixture.storage_path.clone();
        let store = Arc::new(Mutex::new(HistoryStore::new(storage_path).unwrap()));

        let mut handles = vec![];

        // Concurrent writes
        for i in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = tokio::spawn(async move {
                let session_id = Uuid::new_v4();
                let entry = HistoryEntry::new(
                    session_id,
                    format!("concurrent_write_{}", i),
                    vec![],
                    format!("Concurrent output {}", i),
                    true,
                    1000,
                );

                let mut store_lock = store_clone.lock().await;
                store_lock.store_entry(entry).await.unwrap();
            });
            handles.push(handle);
        }

        // Concurrent reads
        for i in 0..5 {
            let store_clone = Arc::clone(&store);
            let handle = tokio::spawn(async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

                let store_lock = store_clone.lock().await;
                let search = HistorySearch {
                    command_pattern: Some(format!("concurrent_write_{}", i)),
                    ..Default::default()
                };

                let results = store_lock.search(&search).await.unwrap();
                assert!(results.len() <= 1);
            });
            handles.push(handle);
        }

        // Wait for all operations
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify final state
        let store_lock = store.lock().await;
        assert_eq!(store_lock.entries.len(), 10);
    }

    proptest! {
        #[test]
        fn property_backup_restore_preserves_data(
            entries in prop::collection::vec(
                property_strategies::history_entry(Uuid::new_v4()),
                1..50
            ),
        ) {
            tokio_test::block_on(async {
                let fixture = HistoryTestFixture::new();
                let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();

                // Store all entries
                for entry in &entries {
                    store.store_entry(entry.clone()).await.unwrap();
                }

                // Create backup
                let backup_path = fixture.temp_dir.path().join("test_backup.json");
                store.export(&backup_path, ExportFormat::Json, None).await.unwrap();

                // Clear store
                store.clear_all().await.unwrap();
                prop_assert_eq!(store.entries.len(), 0);

                // Restore from backup
                let backup_content = tokio::fs::read_to_string(&backup_path).await.unwrap();
                let restored_entries: Vec<HistoryEntry> =
                    serde_json::from_str(&backup_content).unwrap();

                for entry in restored_entries {
                    store.store_entry(entry).await.unwrap();
                }

                // Verify all data is preserved
                prop_assert_eq!(store.entries.len(), entries.len());

                Ok(())
            });
        }

        #[test]
        fn property_storage_operations_maintain_consistency(
            operations in prop::collection::vec(
                prop::sample::select(vec!["add", "update", "clear_old"]),
                1..20
            ),
        ) {
            tokio_test::block_on(async {
                let fixture = HistoryTestFixture::new();
                let mut store = HistoryStore::new(fixture.storage_path.clone()).unwrap();
                let session_id = Uuid::new_v4();

                let mut expected_count = 0;
                let mut entry_ids = Vec::new();

                for (i, operation) in operations.iter().enumerate() {
                    match operation.as_ref() {
                        "add" => {
                            let entry = HistoryEntry::new(
                                session_id,
                                format!("cmd_{}", i),
                                vec![],
                                "output".to_string(),
                                true,
                                1000,
                            );

                            entry_ids.push(entry.id.clone());
                            store.store_entry(entry).await.unwrap();
                            expected_count += 1;
                        }
                        "update" => {
                            if !entry_ids.is_empty() {
                                let id = &entry_ids[i % entry_ids.len()];
                                if let Some(mut entry) = store.get_entry(id).cloned() {
                                    entry.output = format!("Updated output {}", i);
                                    store.update_entry(id, entry).await.unwrap();
                                }
                            }
                        }
                        "clear_old" => {
                            if expected_count > 5 {
                                let pruned = store.prune_old_entries(1000).await.unwrap();
                                expected_count -= pruned;
                            }
                        }
                        _ => {}
                    }
                }

                // Verify consistency
                prop_assert_eq!(store.entries.len(), expected_count);
                prop_assert_eq!(store.index.len(), expected_count);

                // Verify all indexed entries are accessible
                for (id, &index) in &store.index {
                    prop_assert!(index < store.entries.len());
                    prop_assert_eq!(&store.entries[index].id, id);
                }

                Ok(())
            });
        }
    }
}
