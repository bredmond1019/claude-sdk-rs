//! Property-based tests for analytics data integrity
//!
//! These tests use proptest to validate invariants and data integrity
//! across the analytics system under various conditions.

#[cfg(test)]
mod analytics_property_tests {
    use super::super::*;
    use crate::{
        cost::{CostEntry, CostTracker},
        history::{HistoryEntry, HistoryStore},
    };
    use chrono::{DateTime, Duration, Utc};
    use proptest::prelude::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::sync::RwLock;

    // Property test strategies for generating test data

    /// Strategy for generating valid cost values (0.0001 to 100.0)
    fn cost_strategy() -> impl Strategy<Value = f64> {
        prop::num::f64::POSITIVE.prop_map(|x| (x * 100.0).max(0.0001).min(100.0))
    }

    /// Strategy for generating token counts (1 to 100,000)
    fn token_strategy() -> impl Strategy<Value = u32> {
        1u32..100_000u32
    }

    /// Strategy for generating duration in milliseconds (1 to 300,000ms = 5 min)
    fn duration_strategy() -> impl Strategy<Value = u64> {
        1u64..300_000u64
    }

    /// Strategy for generating command names
    fn command_name_strategy() -> impl Strategy<Value = String> {
        prop::collection::vec("[a-z]{3,15}", 1..4).prop_map(|parts| parts.join("_"))
    }

    /// Strategy for generating model names
    fn model_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("claude-3-opus".to_string()),
            Just("claude-3-sonnet".to_string()),
            Just("claude-3-haiku".to_string()),
            Just("claude-2".to_string()),
        ]
    }

    /// Strategy for generating success/failure ratio
    fn success_strategy() -> impl Strategy<Value = bool> {
        prop::bool::weighted(0.85) // 85% success rate
    }

    /// Strategy for generating timestamps within the last 30 days
    fn timestamp_strategy() -> impl Strategy<Value = DateTime<Utc>> {
        let now = Utc::now();
        let start = now - Duration::days(30);
        let range = (now - start).num_seconds();

        (0..range).prop_map(move |offset| start + Duration::seconds(offset))
    }

    /// Strategy for generating analytics config
    fn analytics_config_strategy() -> impl Strategy<Value = AnalyticsConfig> {
        (
            prop::bool::ANY, // enable_real_time_alerts
            cost_strategy(), // cost_alert_threshold
            1u32..365u32,    // retention_days
            1u64..3600u64,   // dashboard_refresh_interval
        )
            .prop_map(|(alerts, threshold, retention, refresh)| AnalyticsConfig {
                enable_real_time_alerts: alerts,
                cost_alert_threshold: threshold,
                report_schedule: ReportSchedule::Weekly,
                retention_days: retention,
                dashboard_refresh_interval: refresh,
            })
    }

    /// Generate a collection of cost entries
    fn cost_entries_strategy() -> impl Strategy<Value = Vec<CostEntry>> {
        prop::collection::vec(
            (
                prop::strategy::Just(uuid::Uuid::new_v4()),
                command_name_strategy(),
                cost_strategy(),
                token_strategy(),
                token_strategy(),
                duration_strategy(),
                model_strategy(),
            ),
            1..50,
        )
        .prop_map(|entries| {
            entries
                .into_iter()
                .map(
                    |(session_id, command, cost, input_tokens, output_tokens, duration, model)| {
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
                .collect()
        })
    }

    /// Generate a collection of history entries
    fn history_entries_strategy() -> impl Strategy<Value = Vec<HistoryEntry>> {
        prop::collection::vec(
            (
                prop::strategy::Just(uuid::Uuid::new_v4()),
                command_name_strategy(),
                prop::collection::vec("[a-z0-9]{1,10}", 0..5), // args
                "[a-zA-Z0-9 ]{10,100}",                        // output
                success_strategy(),
                duration_strategy(),
                cost_strategy(),
                token_strategy(),
                token_strategy(),
                model_strategy(),
                timestamp_strategy(),
            ),
            1..50,
        )
        .prop_map(|entries| {
            entries
                .into_iter()
                .map(
                    |(
                        session_id,
                        command,
                        args,
                        output,
                        success,
                        duration,
                        cost,
                        input_tokens,
                        output_tokens,
                        model,
                        timestamp,
                    )| {
                        let mut entry =
                            HistoryEntry::new(session_id, command, args, output, success, duration);
                        entry.cost_usd = Some(cost);
                        entry.input_tokens = Some(input_tokens);
                        entry.output_tokens = Some(output_tokens);
                        entry.model = Some(model);
                        entry.timestamp = timestamp;
                        entry
                    },
                )
                .collect()
        })
    }

    /// Create a test analytics engine with the given data
    async fn create_test_engine_with_data(
        cost_entries: Vec<CostEntry>,
        history_entries: Vec<HistoryEntry>,
        config: AnalyticsConfig,
    ) -> Result<AnalyticsEngine> {
        let temp_dir = tempdir().unwrap();

        let cost_tracker = Arc::new(RwLock::new(
            CostTracker::new(temp_dir.path().join("costs.json")).unwrap(),
        ));

        let history_store = Arc::new(RwLock::new(
            HistoryStore::new(temp_dir.path().join("history.json")).unwrap(),
        ));

        // Populate cost tracker
        {
            let mut tracker = cost_tracker.write().await;
            for entry in cost_entries {
                tracker.record_cost(entry).await.unwrap();
            }
        }

        // Populate history store
        {
            let mut store = history_store.write().await;
            for entry in history_entries {
                store.store_entry(entry).await.unwrap();
            }
        }

        let engine = AnalyticsEngine::new(cost_tracker, history_store, config);
        Ok(engine)
    }

    // Property tests

    proptest! {
        /// Property: Total cost should always be the sum of individual costs
        #[test]
        fn prop_cost_summary_total_invariant(
            cost_entries in cost_entries_strategy(),
            config in analytics_config_strategy()
        ) {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let engine = create_test_engine_with_data(cost_entries.clone(), vec![], config).await.unwrap();
                let summary = engine.generate_summary(30).await.unwrap();

                let expected_total: f64 = cost_entries.iter().map(|e| e.cost_usd).sum();
                let actual_total = summary.cost_summary.total_cost;

                // Allow for floating point precision errors
                prop_assert!((expected_total - actual_total).abs() < 0.0001,
                    "Expected total {}, got {}", expected_total, actual_total);

                // Command count should match
                prop_assert_eq!(summary.cost_summary.command_count, cost_entries.len());

                // Average cost should be correct
                if !cost_entries.is_empty() {
                    let expected_avg = expected_total / cost_entries.len() as f64;
                    prop_assert!((summary.cost_summary.average_cost - expected_avg).abs() < 0.0001);
                }

                Ok(())
            }).unwrap();
        }

        /// Property: Success rate should be between 0 and 100 and match actual data
        #[test]
        fn prop_success_rate_invariant(
            history_entries in history_entries_strategy(),
            config in analytics_config_strategy()
        ) {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let engine = create_test_engine_with_data(vec![], history_entries.clone(), config).await.unwrap();
                let summary = engine.generate_summary(30).await.unwrap();

                let success_rate = summary.history_stats.success_rate;

                // Success rate should be in valid range
                prop_assert!(success_rate >= 0.0 && success_rate <= 100.0,
                    "Success rate {} not in range [0, 100]", success_rate);

                // Verify calculation
                let successful = history_entries.iter().filter(|e| e.success).count();
                let total = history_entries.len();
                let expected_rate = if total > 0 {
                    (successful as f64 / total as f64) * 100.0
                } else {
                    0.0
                };

                prop_assert!((success_rate - expected_rate).abs() < 0.1,
                    "Expected success rate {:.2}, got {:.2}", expected_rate, success_rate);

                Ok(())
            }).unwrap();
        }

        /// Property: Performance metrics should be internally consistent
        #[test]
        fn prop_performance_metrics_consistency(
            history_entries in history_entries_strategy(),
            config in analytics_config_strategy()
        ) {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let engine = create_test_engine_with_data(vec![], history_entries.clone(), config).await.unwrap();
                let summary = engine.generate_summary(30).await.unwrap();
                let metrics = &summary.performance_metrics;

                // Average response time should be non-negative
                prop_assert!(metrics.average_response_time >= 0.0);

                // Success rate should match history stats
                prop_assert!((metrics.success_rate - summary.history_stats.success_rate).abs() < 0.1);

                // Throughput should be positive if there are entries
                if !history_entries.is_empty() {
                    prop_assert!(metrics.throughput_commands_per_hour > 0.0);
                }

                // Peak usage hour should be in valid range
                prop_assert!(metrics.peak_usage_hour < 24);

                // Error rates should be in valid range [0, 100]
                for (_, error_rate) in &metrics.error_rate_by_command {
                    prop_assert!(*error_rate >= 0.0 && *error_rate <= 100.0,
                        "Error rate {} not in range [0, 100]", error_rate);
                }

                Ok(())
            }).unwrap();
        }

        /// Property: Model aggregations should be correct
        #[test]
        fn prop_model_aggregations_correct(
            cost_entries in cost_entries_strategy(),
            config in analytics_config_strategy()
        ) {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let engine = create_test_engine_with_data(cost_entries.clone(), vec![], config).await.unwrap();
                let summary = engine.generate_summary(30).await.unwrap();

                // Calculate expected model totals
                let mut expected_by_model: HashMap<String, f64> = HashMap::new();
                for entry in &cost_entries {
                    *expected_by_model.entry(entry.model.clone()).or_insert(0.0) += entry.cost_usd;
                }

                // Verify model totals
                for (model, expected_cost) in expected_by_model {
                    if let Some(actual_cost) = summary.cost_summary.by_model.get(&model) {
                        prop_assert!((expected_cost - actual_cost).abs() < 0.0001,
                            "Model {} cost mismatch: expected {}, got {}", model, expected_cost, actual_cost);
                    } else {
                        prop_assert!(expected_cost == 0.0, "Missing model {} with cost {}", model, expected_cost);
                    }
                }

                // Total of model costs should equal total cost
                let model_total: f64 = summary.cost_summary.by_model.values().sum();
                prop_assert!((model_total - summary.cost_summary.total_cost).abs() < 0.0001,
                    "Model total {} doesn't match cost total {}", model_total, summary.cost_summary.total_cost);

                Ok(())
            }).unwrap();
        }

        /// Property: Empty data sets should be handled gracefully
        #[test]
        fn prop_empty_data_handling(config in analytics_config_strategy()) {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let engine = create_test_engine_with_data(vec![], vec![], config).await.unwrap();
                let summary = engine.generate_summary(30).await.unwrap();

                // Should handle empty data gracefully
                prop_assert_eq!(summary.cost_summary.total_cost, 0.0);
                prop_assert_eq!(summary.cost_summary.command_count, 0);
                prop_assert_eq!(summary.cost_summary.average_cost, 0.0);
                prop_assert_eq!(summary.history_stats.total_entries, 0);
                prop_assert_eq!(summary.performance_metrics.average_response_time, 0.0);
                prop_assert_eq!(summary.performance_metrics.success_rate, 0.0);
                prop_assert!(summary.insights.is_empty());
                prop_assert!(summary.alerts.is_empty());

                // Dashboard should also handle empty data
                let dashboard = engine.get_dashboard_data().await.unwrap();
                prop_assert_eq!(dashboard.today_cost, 0.0);
                prop_assert_eq!(dashboard.today_commands, 0);
                prop_assert!(dashboard.recent_activity.is_empty());
                prop_assert!(dashboard.top_commands.is_empty());

                Ok(())
            }).unwrap();
        }
    }
}
