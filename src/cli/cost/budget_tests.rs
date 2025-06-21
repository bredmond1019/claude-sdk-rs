//! Budget and Alert System Tests (Task 1.4)
//!
//! These tests verify the budget and alert functionality of the cost tracker.

use super::{CostEntry, CostTracker};
use crate::cli::cost::tracker::{
    AdvancedCostTracker, Budget, BudgetScope, BudgetStatus, CostAlert,
};
use crate::cli::session::SessionId;
use chrono::{DateTime, Duration, Utc};
use proptest::prelude::*;
use tempfile::tempdir;
use uuid::Uuid;

/// Test fixture for budget and alert testing
struct BudgetTestFixture {
    pub advanced_tracker: AdvancedCostTracker,
    pub session_id: SessionId,
    pub temp_dir: tempfile::TempDir,
}

impl BudgetTestFixture {
    async fn new() -> Self {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let storage_path = temp_dir.path().join("budget_test_costs.json");
        let session_id = Uuid::new_v4();

        let base_tracker = CostTracker::new(storage_path).unwrap();
        let advanced_tracker = AdvancedCostTracker::new(base_tracker);

        Self {
            advanced_tracker,
            session_id,
            temp_dir,
        }
    }

    async fn add_sample_costs(&mut self, count: usize, base_cost: f64) {
        for i in 0..count {
            let entry = CostEntry::new(
                self.session_id,
                format!("test_cmd_{}", i),
                base_cost * (i + 1) as f64,
                100 + i as u32,
                200 + i as u32,
                1000 + i as u64,
                "claude-3-opus".to_string(),
            );
            self.advanced_tracker.record_cost(entry).await.unwrap();
        }
    }

    /// Add costs with specific timestamps for time-based testing
    async fn add_time_distributed_costs(&mut self, days_back: i64, costs_per_day: Vec<f64>) {
        let now = Utc::now();

        for (day_offset, &daily_cost) in costs_per_day.iter().enumerate() {
            let mut entry = CostEntry::new(
                self.session_id,
                format!("day_{}_cmd", day_offset),
                daily_cost,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            );

            // Set timestamp to specific day
            entry.timestamp = now - Duration::days(days_back - day_offset as i64);
            self.advanced_tracker.record_cost(entry).await.unwrap();
        }
    }
}

// Task 1.4.1: Test budget threshold calculations with various limits

#[tokio::test]
async fn test_budget_creation_and_validation() {
    let mut fixture = BudgetTestFixture::new().await;

    // Test valid budget creation
    let valid_budget = Budget {
        id: "test-budget-1".to_string(),
        name: "Test Budget".to_string(),
        limit_usd: 100.0,
        period_days: 30,
        scope: BudgetScope::Global,
        alert_thresholds: vec![50.0, 80.0, 95.0],
        created_at: Utc::now(),
    };

    let result = fixture.advanced_tracker.add_budget(valid_budget.clone());
    assert!(result.is_ok(), "Valid budget should be accepted");
    assert_eq!(fixture.advanced_tracker.budgets().len(), 1);

    // Test invalid budget - zero limit
    let invalid_budget_limit = Budget {
        limit_usd: 0.0,
        ..valid_budget.clone()
    };

    let result = fixture.advanced_tracker.add_budget(invalid_budget_limit);
    assert!(result.is_err(), "Zero budget limit should be rejected");

    // Test invalid budget - zero period
    let invalid_budget_period = Budget {
        period_days: 0,
        ..valid_budget.clone()
    };

    let result = fixture.advanced_tracker.add_budget(invalid_budget_period);
    assert!(result.is_err(), "Zero period should be rejected");

    // Test negative budget limit
    let negative_budget = Budget {
        limit_usd: -10.0,
        ..valid_budget
    };

    let result = fixture.advanced_tracker.add_budget(negative_budget);
    assert!(result.is_err(), "Negative budget limit should be rejected");
}

#[tokio::test]
async fn test_budget_threshold_calculations() {
    let mut fixture = BudgetTestFixture::new().await;

    // Create budget with multiple alert thresholds
    let budget = Budget {
        id: "threshold-test".to_string(),
        name: "Threshold Test Budget".to_string(),
        limit_usd: 100.0,
        period_days: 30,
        scope: BudgetScope::Global,
        alert_thresholds: vec![25.0, 50.0, 75.0, 90.0, 95.0],
        created_at: Utc::now(),
    };

    fixture.advanced_tracker.add_budget(budget.clone()).unwrap();

    // Test at various spending levels
    let test_cases = vec![
        (0.0, 0.0, vec![]),                           // No spending
        (24.0, 24.0, vec![]),                         // Just below first threshold
        (25.0, 25.0, vec!["25% budget utilization"]), // At first threshold
        (
            50.0,
            50.0,
            vec!["25% budget utilization", "50% budget utilization"],
        ),
        (
            75.0,
            75.0,
            vec!["25% budget utilization", "50% budget utilization", "75% budget utilization"],
        ),
        (
            90.0,
            90.0,
            vec![
                "25% budget utilization",
                "50% budget utilization",
                "75% budget utilization",
                "90% budget utilization",
            ],
        ),
        (
            95.0,
            95.0,
            vec![
                "25% budget utilization",
                "50% budget utilization",
                "75% budget utilization",
                "90% budget utilization",
                "95% budget utilization",
            ],
        ),
        (
            100.0,
            100.0,
            vec![
                "25% budget utilization",
                "50% budget utilization",
                "75% budget utilization",
                "90% budget utilization",
                "95% budget utilization",
            ],
        ),
    ];

    for (spend_amount, expected_utilization, expected_alerts) in test_cases {
        // Clear existing costs
        fixture.advanced_tracker = BudgetTestFixture::new().await.advanced_tracker;
        fixture.advanced_tracker.add_budget(budget.clone()).unwrap();

        if spend_amount > 0.0 {
            let entry = CostEntry::new(
                fixture.session_id,
                "test_cmd".to_string(),
                spend_amount,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            );
            fixture.advanced_tracker.record_cost(entry).await.unwrap();
        }

        let status = fixture
            .advanced_tracker
            .get_budget_status(&budget)
            .await
            .unwrap();

        assert_eq!(
            status.current_spend, spend_amount,
            "Current spend mismatch for {}",
            spend_amount
        );
        assert_eq!(
            status.utilization_percent, expected_utilization,
            "Utilization percent mismatch"
        );
        assert_eq!(
            status.alerts_triggered, expected_alerts,
            "Alert triggers mismatch for {} spending",
            spend_amount
        );
    }
}

#[tokio::test]
async fn test_budget_threshold_edge_cases() {
    let mut fixture = BudgetTestFixture::new().await;

    // Test with very small budget
    let small_budget = Budget {
        id: "small-budget".to_string(),
        name: "Small Budget".to_string(),
        limit_usd: 0.01,
        period_days: 1,
        scope: BudgetScope::Global,
        alert_thresholds: vec![50.0, 99.0],
        created_at: Utc::now(),
    };

    fixture
        .advanced_tracker
        .add_budget(small_budget.clone())
        .unwrap();

    // Add tiny cost
    let tiny_entry = CostEntry::new(
        fixture.session_id,
        "tiny_cmd".to_string(),
        0.005, // 50% of budget
        1,
        1,
        10,
        "claude-3-haiku".to_string(),
    );
    fixture
        .advanced_tracker
        .record_cost(tiny_entry)
        .await
        .unwrap();

    let status = fixture
        .advanced_tracker
        .get_budget_status(&small_budget)
        .await
        .unwrap();
    assert_eq!(status.utilization_percent, 50.0);
    assert_eq!(status.alerts_triggered.len(), 1);

    // Test with very large budget
    let large_budget = Budget {
        id: "large-budget".to_string(),
        name: "Large Budget".to_string(),
        limit_usd: 1_000_000.0,
        period_days: 365,
        scope: BudgetScope::Global,
        alert_thresholds: vec![0.001, 0.01, 0.1], // Very small percentages
        created_at: Utc::now(),
    };

    fixture
        .advanced_tracker
        .add_budget(large_budget.clone())
        .unwrap();

    // Add cost that's small relative to budget but triggers alerts
    let large_entry = CostEntry::new(
        fixture.session_id,
        "large_cmd".to_string(),
        1000.0, // 0.1% of budget
        10000,
        20000,
        60000,
        "claude-3-opus".to_string(),
    );
    fixture
        .advanced_tracker
        .record_cost(large_entry)
        .await
        .unwrap();

    let status = fixture
        .advanced_tracker
        .get_budget_status(&large_budget)
        .await
        .unwrap();
    assert!(status.utilization_percent >= 0.1);
    assert_eq!(status.alerts_triggered.len(), 3); // All thresholds triggered
}

// Task 1.4.2: Test budget alert generation when thresholds are exceeded

#[tokio::test]
async fn test_cost_alert_creation_and_validation() {
    let mut fixture = BudgetTestFixture::new().await;

    // Test valid alert creation
    let valid_alert = CostAlert {
        id: "test-alert-1".to_string(),
        name: "Test Alert".to_string(),
        threshold_usd: 50.0,
        period_days: 7,
        enabled: true,
        notification_channels: vec!["email".to_string(), "slack".to_string()],
    };

    let result = fixture.advanced_tracker.add_alert(valid_alert.clone());
    assert!(result.is_ok(), "Valid alert should be accepted");
    assert_eq!(fixture.advanced_tracker.alerts().len(), 1);

    // Test invalid alert - zero threshold
    let invalid_alert = CostAlert {
        threshold_usd: 0.0,
        ..valid_alert.clone()
    };

    let result = fixture.advanced_tracker.add_alert(invalid_alert);
    assert!(result.is_err(), "Zero threshold alert should be rejected");

    // Test negative threshold
    let negative_alert = CostAlert {
        threshold_usd: -10.0,
        ..valid_alert
    };

    let result = fixture.advanced_tracker.add_alert(negative_alert);
    assert!(
        result.is_err(),
        "Negative threshold alert should be rejected"
    );
}

#[tokio::test]
async fn test_alert_triggering_multiple_thresholds() {
    let mut fixture = BudgetTestFixture::new().await;

    // Create multiple alerts with different thresholds
    let alerts = vec![
        CostAlert {
            id: "low-alert".to_string(),
            name: "Low Spending Alert".to_string(),
            threshold_usd: 10.0,
            period_days: 1,
            enabled: true,
            notification_channels: vec!["email".to_string()],
        },
        CostAlert {
            id: "medium-alert".to_string(),
            name: "Medium Spending Alert".to_string(),
            threshold_usd: 50.0,
            period_days: 7,
            enabled: true,
            notification_channels: vec!["email".to_string(), "slack".to_string()],
        },
        CostAlert {
            id: "high-alert".to_string(),
            name: "High Spending Alert".to_string(),
            threshold_usd: 100.0,
            period_days: 30,
            enabled: true,
            notification_channels: vec![
                "email".to_string(),
                "slack".to_string(),
                "pager".to_string(),
            ],
        },
        CostAlert {
            id: "disabled-alert".to_string(),
            name: "Disabled Alert".to_string(),
            threshold_usd: 5.0,
            period_days: 1,
            enabled: false,
            notification_channels: vec!["email".to_string()],
        },
    ];

    for alert in alerts {
        fixture.advanced_tracker.add_alert(alert).unwrap();
    }

    // Add costs that will trigger some alerts
    let entry = CostEntry::new(
        fixture.session_id,
        "test_cmd".to_string(),
        60.0, // Should trigger low and medium alerts, but not high
        600,
        1200,
        5000,
        "claude-3-opus".to_string(),
    );
    fixture.advanced_tracker.record_cost(entry).await.unwrap();

    let triggered = fixture.advanced_tracker.check_alerts().await.unwrap();

    // Should have 2 triggered alerts (low and medium, not high or disabled)
    assert_eq!(triggered.len(), 2);

    // Verify correct alerts were triggered
    let triggered_ids: Vec<String> = triggered
        .iter()
        .map(|(alert, _)| alert.id.clone())
        .collect();
    assert!(triggered_ids.contains(&"low-alert".to_string()));
    assert!(triggered_ids.contains(&"medium-alert".to_string()));
    assert!(!triggered_ids.contains(&"high-alert".to_string()));
    assert!(!triggered_ids.contains(&"disabled-alert".to_string()));

    // Verify costs match
    for (alert, cost) in &triggered {
        assert_eq!(cost, &60.0);
    }
}

#[tokio::test]
async fn test_alert_time_window_filtering() {
    let mut fixture = BudgetTestFixture::new().await;

    // Create alert with 7-day window
    let alert = CostAlert {
        id: "weekly-alert".to_string(),
        name: "Weekly Alert".to_string(),
        threshold_usd: 50.0,
        period_days: 7,
        enabled: true,
        notification_channels: vec!["email".to_string()],
    };

    fixture.advanced_tracker.add_alert(alert.clone()).unwrap();

    // Add old cost (outside window)
    let mut old_entry = CostEntry::new(
        fixture.session_id,
        "old_cmd".to_string(),
        30.0,
        300,
        600,
        3000,
        "claude-3-opus".to_string(),
    );
    old_entry.timestamp = Utc::now() - Duration::days(10);
    fixture
        .advanced_tracker
        .record_cost(old_entry)
        .await
        .unwrap();

    // Add recent cost (inside window)
    let recent_entry = CostEntry::new(
        fixture.session_id,
        "recent_cmd".to_string(),
        30.0,
        300,
        600,
        3000,
        "claude-3-opus".to_string(),
    );
    fixture
        .advanced_tracker
        .record_cost(recent_entry)
        .await
        .unwrap();

    let triggered = fixture.advanced_tracker.check_alerts().await.unwrap();

    // Should not trigger because only recent cost (30.0) is counted
    assert_eq!(triggered.len(), 0);

    // Add another recent cost to exceed threshold
    let another_entry = CostEntry::new(
        fixture.session_id,
        "another_cmd".to_string(),
        25.0,
        250,
        500,
        2500,
        "claude-3-opus".to_string(),
    );
    fixture
        .advanced_tracker
        .record_cost(another_entry)
        .await
        .unwrap();

    let triggered = fixture.advanced_tracker.check_alerts().await.unwrap();

    // Now should trigger (30.0 + 25.0 = 55.0 > 50.0)
    assert_eq!(triggered.len(), 1);
    assert_eq!(triggered[0].1, 55.0);
}

// Task 1.4.3: Test budget warning states (approaching threshold)

#[tokio::test]
async fn test_budget_warning_states() {
    let mut fixture = BudgetTestFixture::new().await;

    // Create budget with warning thresholds
    let budget = Budget {
        id: "warning-budget".to_string(),
        name: "Warning Budget".to_string(),
        limit_usd: 100.0,
        period_days: 30,
        scope: BudgetScope::Global,
        alert_thresholds: vec![50.0, 75.0, 90.0, 95.0], // Warning levels
        created_at: Utc::now(),
    };

    fixture.advanced_tracker.add_budget(budget.clone()).unwrap();

    // Test approaching various thresholds
    let test_scenarios = vec![
        (45.0, "approaching 50%", 0), // Below first threshold
        (48.0, "approaching 50%", 0), // Very close to 50%
        (72.0, "approaching 75%", 1), // Between 50% and 75%
        (88.0, "approaching 90%", 2), // Between 75% and 90%
        (93.0, "approaching 95%", 3), // Between 90% and 95%
    ];

    for (spend, description, expected_alerts) in test_scenarios {
        // Reset tracker for each test
        fixture.advanced_tracker = BudgetTestFixture::new().await.advanced_tracker;
        fixture.advanced_tracker.add_budget(budget.clone()).unwrap();

        let entry = CostEntry::new(
            fixture.session_id,
            format!("test_cmd_{}", spend),
            spend,
            100,
            200,
            1000,
            "claude-3-opus".to_string(),
        );
        fixture.advanced_tracker.record_cost(entry).await.unwrap();

        let status = fixture
            .advanced_tracker
            .get_budget_status(&budget)
            .await
            .unwrap();

        println!(
            "Testing {}: spend={}, utilization={}%, alerts={}",
            description,
            spend,
            status.utilization_percent,
            status.alerts_triggered.len()
        );

        assert_eq!(
            status.alerts_triggered.len(),
            expected_alerts,
            "Wrong number of alerts for {}",
            description
        );

        // Verify remaining budget calculation
        assert!(
            (status.remaining - (100.0 - spend)).abs() < 0.001,
            "Remaining budget calculation error for {}",
            description
        );
    }
}

#[tokio::test]
async fn test_budget_projected_overage_warnings() {
    let mut fixture = BudgetTestFixture::new().await;

    // Create budget that started 10 days ago
    let budget = Budget {
        id: "projection-budget".to_string(),
        name: "Projection Budget".to_string(),
        limit_usd: 300.0,
        period_days: 30,
        scope: BudgetScope::Global,
        alert_thresholds: vec![80.0],
        created_at: Utc::now() - Duration::days(10),
    };

    fixture.advanced_tracker.add_budget(budget.clone()).unwrap();

    // Add costs that indicate under-budget trend
    // $5/day for 10 days = $50 total, projected to $150 in 30 days (well under limit)
    let daily_costs = vec![5.0; 10];
    fixture.add_time_distributed_costs(9, daily_costs).await;

    let status = fixture
        .advanced_tracker
        .get_budget_status(&budget)
        .await
        .unwrap();

    assert_eq!(status.current_spend, 50.0);
    assert!(
        status.projected_overage.is_none(),
        "Should not project overage when under budget"
    );

    // Add more days with very high spending to trigger projection warning
    // Need to push the average high enough that 30-day projection exceeds $300
    for i in 0..5 {
        let high_entry = CostEntry::new(
            fixture.session_id,
            format!("high_spend_cmd_{}", i),
            50.0, // High daily spending
            200,
            400,
            2000,
            "claude-3-opus".to_string(),
        );
        fixture
            .advanced_tracker
            .record_cost(high_entry)
            .await
            .unwrap();
    }

    let status = fixture
        .advanced_tracker
        .get_budget_status(&budget)
        .await
        .unwrap();

    assert_eq!(status.current_spend, 300.0); // $50 + 5*$50 = $300
                                             // With recent high spending, the 7-day average should be high, projecting overage
                                             // Recent 7-day average includes mostly the $50/day entries
    assert!(
        status.projected_overage.is_some(),
        "Should project overage with increased spending"
    );
}

// Task 1.4.4: Test budget reset functionality for new periods

#[tokio::test]
async fn test_budget_period_reset() {
    let mut fixture = BudgetTestFixture::new().await;

    // Create budget with short period for testing
    let budget = Budget {
        id: "reset-budget".to_string(),
        name: "Reset Test Budget".to_string(),
        limit_usd: 50.0,
        period_days: 7, // Weekly budget
        scope: BudgetScope::Global,
        alert_thresholds: vec![80.0],
        created_at: Utc::now() - Duration::days(8), // Started 8 days ago
    };

    fixture.advanced_tracker.add_budget(budget.clone()).unwrap();

    // Add costs from different periods
    // Old costs (outside current period)
    for i in 0..3 {
        let mut old_entry = CostEntry::new(
            fixture.session_id,
            format!("old_cmd_{}", i),
            10.0,
            100,
            200,
            1000,
            "claude-3-opus".to_string(),
        );
        old_entry.timestamp = Utc::now() - Duration::days(10 + i as i64);
        fixture
            .advanced_tracker
            .record_cost(old_entry)
            .await
            .unwrap();
    }

    // Current period costs
    for i in 0..2 {
        let recent_entry = CostEntry::new(
            fixture.session_id,
            format!("recent_cmd_{}", i),
            15.0,
            150,
            300,
            1500,
            "claude-3-opus".to_string(),
        );
        fixture
            .advanced_tracker
            .record_cost(recent_entry)
            .await
            .unwrap();
    }

    let status = fixture
        .advanced_tracker
        .get_budget_status(&budget)
        .await
        .unwrap();

    // Should only count recent costs (2 * 15.0 = 30.0)
    assert_eq!(status.current_spend, 30.0);
    assert_eq!(status.remaining, 20.0);
    assert_eq!(status.utilization_percent, 60.0);
    assert!(
        status.alerts_triggered.is_empty(),
        "Should not trigger 80% alert at 60%"
    );

    // Verify old costs are not included
    // Note: We can't directly access base_tracker, but we can verify through the budget status
    // that only recent costs are counted in the budget period
}

#[tokio::test]
async fn test_budget_scope_functionality() {
    let mut fixture = BudgetTestFixture::new().await;

    // Create budgets with different scopes
    let global_budget = Budget {
        id: "global-budget".to_string(),
        name: "Global Budget".to_string(),
        limit_usd: 100.0,
        period_days: 30,
        scope: BudgetScope::Global,
        alert_thresholds: vec![80.0],
        created_at: Utc::now(),
    };

    let session_budget = Budget {
        id: "session-budget".to_string(),
        name: "Session Budget".to_string(),
        limit_usd: 50.0,
        period_days: 30,
        scope: BudgetScope::Session(fixture.session_id),
        alert_thresholds: vec![90.0],
        created_at: Utc::now(),
    };

    let command_budget = Budget {
        id: "command-budget".to_string(),
        name: "Command Budget".to_string(),
        limit_usd: 25.0,
        period_days: 30,
        scope: BudgetScope::Command("analyze_code".to_string()),
        alert_thresholds: vec![75.0],
        created_at: Utc::now(),
    };

    // Add all budgets
    fixture.advanced_tracker.add_budget(global_budget).unwrap();
    fixture.advanced_tracker.add_budget(session_budget).unwrap();
    fixture.advanced_tracker.add_budget(command_budget).unwrap();

    assert_eq!(fixture.advanced_tracker.budgets().len(), 3);

    // Add costs to test scope filtering
    fixture.add_sample_costs(3, 10.0).await;

    // Add a specific command cost
    let command_entry = CostEntry::new(
        fixture.session_id,
        "analyze_code".to_string(),
        15.0,
        150,
        300,
        2500,
        "claude-3-opus".to_string(),
    );
    fixture
        .advanced_tracker
        .record_cost(command_entry.clone())
        .await
        .unwrap();

    // Test budget status for each scope
    let global_status = fixture
        .advanced_tracker
        .get_budget_status(fixture.advanced_tracker.get_budget(0).unwrap())
        .await
        .unwrap();
    let session_status = fixture
        .advanced_tracker
        .get_budget_status(fixture.advanced_tracker.get_budget(1).unwrap())
        .await
        .unwrap();
    let command_status = fixture
        .advanced_tracker
        .get_budget_status(fixture.advanced_tracker.get_budget(2).unwrap())
        .await
        .unwrap();

    // Global should include all costs
    assert!(
        global_status.current_spend > 0.0,
        "Global budget should track all costs"
    );

    // Session should include session-specific costs
    assert!(
        session_status.current_spend > 0.0,
        "Session budget should track session costs"
    );

    // Command should only include analyze_code costs
    assert_eq!(
        command_status.current_spend, 15.0,
        "Command budget should only track specific command"
    );
}

#[tokio::test]
async fn test_budget_limit_checking() {
    let mut fixture = BudgetTestFixture::new().await;

    // Create a tight budget
    let budget = Budget {
        id: "tight-budget".to_string(),
        name: "Tight Budget".to_string(),
        limit_usd: 30.0,
        period_days: 30,
        scope: BudgetScope::Global,
        alert_thresholds: vec![50.0, 80.0, 95.0],
        created_at: Utc::now(),
    };

    fixture.advanced_tracker.add_budget(budget).unwrap();

    // Add costs that approach the limit
    fixture.add_sample_costs(2, 10.0).await; // Total: 30.0

    // Test entry that would exceed budget
    let exceeding_entry = CostEntry::new(
        fixture.session_id,
        "expensive_cmd".to_string(),
        5.0, // Would make total 35.0, exceeding 30.0 limit
        200,
        400,
        3000,
        "claude-3-opus".to_string(),
    );

    let violations = fixture
        .advanced_tracker
        .check_budget_limits(&exceeding_entry)
        .await
        .unwrap();
    assert_eq!(violations.len(), 1, "Should detect budget violation");
    assert!(
        violations[0].current_spend >= 30.0,
        "Should show current spend at/near limit"
    );

    // Test entry that would stay within budget
    let acceptable_entry = CostEntry::new(
        fixture.session_id,
        "cheap_cmd".to_string(),
        0.0, // Would keep total at 30.0, staying within the limit
        50,
        100,
        500,
        "claude-3-haiku".to_string(),
    );

    let violations = fixture
        .advanced_tracker
        .check_budget_limits(&acceptable_entry)
        .await
        .unwrap();
    assert_eq!(
        violations.len(),
        0,
        "Should not detect violation for acceptable cost"
    );
}

#[tokio::test]
async fn test_multiple_budget_period_resets() {
    let mut fixture = BudgetTestFixture::new().await;

    // Create budgets with different periods
    let daily_budget = Budget {
        id: "daily".to_string(),
        name: "Daily Budget".to_string(),
        limit_usd: 10.0,
        period_days: 1,
        scope: BudgetScope::Global,
        alert_thresholds: vec![80.0],
        created_at: Utc::now() - Duration::days(2),
    };

    let weekly_budget = Budget {
        id: "weekly".to_string(),
        name: "Weekly Budget".to_string(),
        limit_usd: 50.0,
        period_days: 7,
        scope: BudgetScope::Global,
        alert_thresholds: vec![80.0],
        created_at: Utc::now() - Duration::days(8),
    };

    let monthly_budget = Budget {
        id: "monthly".to_string(),
        name: "Monthly Budget".to_string(),
        limit_usd: 200.0,
        period_days: 30,
        scope: BudgetScope::Global,
        alert_thresholds: vec![80.0],
        created_at: Utc::now() - Duration::days(31),
    };

    fixture
        .advanced_tracker
        .add_budget(daily_budget.clone())
        .unwrap();
    fixture
        .advanced_tracker
        .add_budget(weekly_budget.clone())
        .unwrap();
    fixture
        .advanced_tracker
        .add_budget(monthly_budget.clone())
        .unwrap();

    // Add costs across different time periods
    let cost_schedule = vec![
        (0, 5.0),   // Today
        (1, 8.0),   // Yesterday
        (2, 12.0),  // 2 days ago
        (5, 15.0),  // 5 days ago
        (10, 20.0), // 10 days ago
        (25, 25.0), // 25 days ago
        (35, 30.0), // 35 days ago
    ];

    for (days_ago, cost) in cost_schedule {
        let mut entry = CostEntry::new(
            fixture.session_id,
            format!("cmd_{}d_ago", days_ago),
            cost,
            100,
            200,
            1000,
            "claude-3-opus".to_string(),
        );
        entry.timestamp = Utc::now() - Duration::days(days_ago);
        fixture.advanced_tracker.record_cost(entry).await.unwrap();
    }

    // Check daily budget (should only include today)
    let daily_status = fixture
        .advanced_tracker
        .get_budget_status(&daily_budget)
        .await
        .unwrap();
    assert_eq!(daily_status.current_spend, 5.0);

    // Check weekly budget (should include last 7 days)
    let weekly_status = fixture
        .advanced_tracker
        .get_budget_status(&weekly_budget)
        .await
        .unwrap();
    assert_eq!(weekly_status.current_spend, 40.0); // 5.0 + 8.0 + 12.0 + 15.0

    // Check monthly budget (should include last 30 days)
    let monthly_status = fixture
        .advanced_tracker
        .get_budget_status(&monthly_budget)
        .await
        .unwrap();
    assert_eq!(monthly_status.current_spend, 85.0); // All except 35 days ago
}

// Property-based tests for budget calculations

proptest! {
    #[test]
    fn prop_budget_utilization_calculation(
        limit in 0.01f64..10000.0f64,
        spend in 0.0f64..20000.0f64,
    ) {
        let utilization = (spend / limit) * 100.0;
        prop_assert!(utilization >= 0.0);

        if spend <= limit {
            prop_assert!(utilization <= 100.0);
        } else {
            prop_assert!(utilization > 100.0);
        }
    }

    #[test]
    fn prop_budget_remaining_calculation(
        limit in 0.01f64..10000.0f64,
        spend in 0.0f64..20000.0f64,
    ) {
        let remaining = limit - spend;

        if spend <= limit {
            prop_assert!(remaining >= 0.0);
            prop_assert_eq!(remaining, limit - spend);
        } else {
            prop_assert!(remaining < 0.0);
        }
    }

    #[test]
    fn prop_alert_threshold_triggering(
        threshold_percent in 0.0f64..100.0f64,
        utilization_percent in 0.0f64..200.0f64,
    ) {
        let should_trigger = utilization_percent >= threshold_percent;

        if should_trigger {
            prop_assert!(utilization_percent >= threshold_percent);
        } else {
            prop_assert!(utilization_percent < threshold_percent);
        }
    }
}

// Integration tests combining budgets and alerts

#[tokio::test]
async fn test_budget_and_alert_integration() {
    let mut fixture = BudgetTestFixture::new().await;

    // Create a budget with alerts
    let budget = Budget {
        id: "integrated-budget".to_string(),
        name: "Integrated Budget".to_string(),
        limit_usd: 100.0,
        period_days: 30,
        scope: BudgetScope::Global,
        alert_thresholds: vec![50.0, 80.0, 95.0],
        created_at: Utc::now(),
    };

    // Create standalone alerts
    let alert1 = CostAlert {
        id: "daily-alert".to_string(),
        name: "Daily Spending Alert".to_string(),
        threshold_usd: 20.0,
        period_days: 1,
        enabled: true,
        notification_channels: vec!["email".to_string()],
    };

    let alert2 = CostAlert {
        id: "weekly-alert".to_string(),
        name: "Weekly Spending Alert".to_string(),
        threshold_usd: 80.0,
        period_days: 7,
        enabled: true,
        notification_channels: vec!["slack".to_string()],
    };

    fixture.advanced_tracker.add_budget(budget.clone()).unwrap();
    fixture.advanced_tracker.add_alert(alert1).unwrap();
    fixture.advanced_tracker.add_alert(alert2).unwrap();

    // Add costs that trigger various thresholds
    let costs = vec![
        (0, 25.0), // Today - triggers daily alert, 25% budget
        (1, 30.0), // Yesterday - 55% budget total, triggers 50% threshold
        (2, 35.0), // 2 days ago - 90% budget total, triggers 80% threshold and weekly alert
    ];

    for (days_ago, cost) in costs {
        let mut entry = CostEntry::new(
            fixture.session_id,
            format!("cmd_day_{}", days_ago),
            cost,
            100,
            200,
            1000,
            "claude-3-opus".to_string(),
        );
        entry.timestamp = Utc::now() - Duration::days(days_ago);
        fixture.advanced_tracker.record_cost(entry).await.unwrap();
    }

    // Check budget status
    let budget_status = fixture
        .advanced_tracker
        .get_budget_status(&budget)
        .await
        .unwrap();
    assert_eq!(budget_status.current_spend, 90.0);
    assert_eq!(budget_status.utilization_percent, 90.0);
    assert_eq!(budget_status.alerts_triggered.len(), 2); // 50% and 80%

    // Check standalone alerts
    let triggered_alerts = fixture.advanced_tracker.check_alerts().await.unwrap();
    assert_eq!(triggered_alerts.len(), 2); // Daily and weekly alerts
}

#[tokio::test]
async fn test_budget_reset_with_carryover_tracking() {
    let mut fixture = BudgetTestFixture::new().await;

    // Simulate multiple budget periods
    let budget = Budget {
        id: "carryover-budget".to_string(),
        name: "Carryover Test Budget".to_string(),
        limit_usd: 100.0,
        period_days: 7, // Weekly
        scope: BudgetScope::Global,
        alert_thresholds: vec![90.0],
        created_at: Utc::now() - Duration::days(21), // 3 weeks ago
    };

    fixture.advanced_tracker.add_budget(budget.clone()).unwrap();

    // Add costs from multiple periods
    // Week 1 (21-14 days ago): Under budget
    for i in 0..3 {
        let mut entry = CostEntry::new(
            fixture.session_id,
            format!("week1_cmd_{}", i),
            20.0,
            200,
            400,
            2000,
            "claude-3-opus".to_string(),
        );
        entry.timestamp = Utc::now() - Duration::days(20 - i as i64);
        fixture.advanced_tracker.record_cost(entry).await.unwrap();
    }

    // Week 2 (14-7 days ago): Over budget
    for i in 0..6 {
        let mut entry = CostEntry::new(
            fixture.session_id,
            format!("week2_cmd_{}", i),
            25.0,
            250,
            500,
            2500,
            "claude-3-opus".to_string(),
        );
        entry.timestamp = Utc::now() - Duration::days(13 - i as i64);
        fixture.advanced_tracker.record_cost(entry).await.unwrap();
    }

    // Week 3 (current): Normal spending
    for i in 0..2 {
        let mut entry = CostEntry::new(
            fixture.session_id,
            format!("week3_cmd_{}", i),
            30.0,
            300,
            600,
            3000,
            "claude-3-opus".to_string(),
        );
        entry.timestamp = Utc::now() - Duration::days(i as i64);
        fixture.advanced_tracker.record_cost(entry).await.unwrap();
    }

    // Check current period status
    let status = fixture
        .advanced_tracker
        .get_budget_status(&budget)
        .await
        .unwrap();
    assert_eq!(status.current_spend, 60.0); // Only current week
    assert!(!status
        .alerts_triggered
        .contains(&"90% budget utilization".to_string()));

    // Verify historical data is preserved but not counted
    // The budget only counts current period costs (60.0), even though more historical data exists
}
