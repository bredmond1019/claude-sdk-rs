//! Advanced cost tracking and analytics
//!
//! This module provides additional cost tracking capabilities including:
//! - Rate limiting based on cost thresholds
//! - Cost alerts and notifications
//! - Trend analysis and predictions
//! - Budget management

use super::{CostEntry, CostFilter, CostTracker};
use crate::{cli::error::InteractiveError, cli::error::Result, cli::session::SessionId};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Cost alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAlert {
    pub id: String,
    pub name: String,
    pub threshold_usd: f64,
    pub period_days: u32,
    pub enabled: bool,
    pub notification_channels: Vec<String>,
}

/// Budget configuration for sessions or commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    pub id: String,
    pub name: String,
    pub limit_usd: f64,
    pub period_days: u32,
    pub scope: BudgetScope,
    pub alert_thresholds: Vec<f64>, // Percentage thresholds (e.g., 50.0, 80.0, 95.0)
    pub created_at: DateTime<Utc>,
}

/// Scope for budget tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BudgetScope {
    Global,
    Session(SessionId),
    Command(String),
}

/// Cost trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTrend {
    pub period_days: u32,
    pub daily_costs: Vec<(DateTime<Utc>, f64)>,
    pub trend_direction: TrendDirection,
    pub projected_monthly_cost: f64,
    pub growth_rate: f64, // Percentage change per day
}

/// Direction of cost trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Budget status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetStatus {
    pub budget: Budget,
    pub current_spend: f64,
    pub remaining: f64,
    pub utilization_percent: f64,
    pub days_remaining: u32,
    pub projected_overage: Option<f64>,
    pub alerts_triggered: Vec<String>,
}

/// Advanced cost tracker with analytics capabilities
pub struct AdvancedCostTracker {
    base_tracker: CostTracker,
    budgets: Vec<Budget>,
    alerts: Vec<CostAlert>,
}

impl AdvancedCostTracker {
    /// Create a new advanced cost tracker
    pub fn new(base_tracker: CostTracker) -> Self {
        Self {
            base_tracker,
            budgets: Vec::new(),
            alerts: Vec::new(),
        }
    }

    /// Add a budget constraint
    pub fn add_budget(&mut self, budget: Budget) -> Result<()> {
        // Validate budget
        if budget.limit_usd <= 0.0 {
            return Err(InteractiveError::InvalidInput(
                "Budget limit must be positive".to_string(),
            ));
        }

        if budget.period_days == 0 {
            return Err(InteractiveError::InvalidInput(
                "Budget period must be at least 1 day".to_string(),
            ));
        }

        self.budgets.push(budget);
        Ok(())
    }

    /// Add a cost alert
    pub fn add_alert(&mut self, alert: CostAlert) -> Result<()> {
        if alert.threshold_usd <= 0.0 {
            return Err(InteractiveError::InvalidInput(
                "Alert threshold must be positive".to_string(),
            ));
        }

        self.alerts.push(alert);
        Ok(())
    }

    /// Check if a cost entry would exceed any budget limits
    pub async fn check_budget_limits(&self, entry: &CostEntry) -> Result<Vec<BudgetStatus>> {
        let mut violations = Vec::new();

        for budget in &self.budgets {
            let status = self.get_budget_status(budget).await?;

            // Check if this entry would cause a budget violation
            if status.current_spend + entry.cost_usd > budget.limit_usd {
                violations.push(status);
            }
        }

        Ok(violations)
    }

    /// Get budget status for a specific budget
    pub async fn get_budget_status(&self, budget: &Budget) -> Result<BudgetStatus> {
        let now = Utc::now();
        let period_start = now - Duration::days(budget.period_days as i64);

        let filter = CostFilter {
            since: Some(period_start),
            until: Some(now),
            session_id: match &budget.scope {
                BudgetScope::Session(id) => Some(*id),
                _ => None,
            },
            command_pattern: match &budget.scope {
                BudgetScope::Command(cmd) => Some(cmd.clone()),
                _ => None,
            },
            ..Default::default()
        };

        let summary = self.base_tracker.get_filtered_summary(&filter).await?;
        let current_spend = summary.total_cost;
        let remaining = budget.limit_usd - current_spend;
        let utilization_percent = (current_spend / budget.limit_usd) * 100.0;

        // Calculate projected overage based on trend
        let trend = self
            .calculate_cost_trend(budget.period_days, &filter)
            .await?;
        let days_remaining = budget.period_days;
        let projected_overage = if trend.projected_monthly_cost > budget.limit_usd {
            Some(trend.projected_monthly_cost - budget.limit_usd)
        } else {
            None
        };

        // Check which alert thresholds have been triggered
        let mut alerts_triggered = Vec::new();
        for threshold in &budget.alert_thresholds {
            if utilization_percent >= *threshold {
                alerts_triggered.push(format!("{}% budget utilization", threshold));
            }
        }

        Ok(BudgetStatus {
            budget: budget.clone(),
            current_spend,
            remaining,
            utilization_percent,
            days_remaining,
            projected_overage,
            alerts_triggered,
        })
    }

    /// Calculate cost trend analysis
    pub async fn calculate_cost_trend(
        &self,
        period_days: u32,
        filter: &CostFilter,
    ) -> Result<CostTrend> {
        let now = Utc::now();
        let period_start = now - Duration::days(period_days as i64);

        let mut daily_filter = filter.clone();
        let mut daily_costs = Vec::new();

        // Calculate daily costs for the period
        for day in 0..period_days {
            let day_start = period_start + Duration::days(day as i64);
            let day_end = day_start + Duration::days(1);

            daily_filter.since = Some(day_start);
            daily_filter.until = Some(day_end);

            let day_summary = self
                .base_tracker
                .get_filtered_summary(&daily_filter)
                .await?;
            daily_costs.push((day_start, day_summary.total_cost));
        }

        // Calculate trend direction and growth rate
        let total_costs: Vec<f64> = daily_costs.iter().map(|(_, cost)| *cost).collect();
        let (trend_direction, growth_rate) = self.calculate_trend_direction(&total_costs);

        // Project monthly cost based on current trend
        let recent_avg = if total_costs.len() >= 7 {
            total_costs.iter().rev().take(7).sum::<f64>() / 7.0
        } else {
            total_costs.iter().sum::<f64>() / total_costs.len().max(1) as f64
        };

        let projected_monthly_cost = recent_avg * 30.0 * (1.0 + growth_rate / 100.0);

        Ok(CostTrend {
            period_days,
            daily_costs,
            trend_direction,
            projected_monthly_cost,
            growth_rate,
        })
    }

    /// Get cost optimization recommendations
    pub async fn get_optimization_recommendations(&self) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        // Get global summary
        let global_summary = self.base_tracker.get_global_summary().await?;

        // Analyze by command
        let top_commands = self.base_tracker.get_top_commands(10).await?;
        if let Some((most_expensive_cmd, cost)) = top_commands.first() {
            if *cost > global_summary.total_cost * 0.3 {
                recommendations.push(format!(
                    "Command '{}' accounts for {:.1}% of total costs. Consider optimizing or reducing usage.",
                    most_expensive_cmd,
                    (*cost / global_summary.total_cost) * 100.0
                ));
            }
        }

        // Analyze by model
        let mut model_costs: Vec<_> = global_summary.by_model.iter().collect();
        model_costs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((most_expensive_model, cost)) = model_costs.first() {
            if **cost > global_summary.total_cost * 0.5 {
                recommendations.push(format!(
                    "Model '{}' accounts for {:.1}% of total costs. Consider using more cost-effective models for routine tasks.",
                    most_expensive_model,
                    (**cost / global_summary.total_cost) * 100.0
                ));
            }
        }

        // Analyze cost trends
        let trend = self
            .calculate_cost_trend(30, &CostFilter::default())
            .await?;
        match trend.trend_direction {
            TrendDirection::Increasing if trend.growth_rate > 10.0 => {
                recommendations.push(format!(
                    "Costs are increasing by {:.1}% per day. Review recent usage patterns and consider setting budgets.",
                    trend.growth_rate
                ));
            }
            _ => {}
        }

        // Analyze for many small commands
        let all_entries = self
            .base_tracker
            .get_entries(&CostFilter::default())
            .await?;
        let small_command_count = all_entries.iter().filter(|e| e.cost_usd < 0.01).count();

        if small_command_count > 10 {
            recommendations.push(format!(
                "Found {} commands with cost < $0.01. Consider batching small operations to reduce overhead.",
                small_command_count
            ));
        }

        // Budget recommendations
        if self.budgets.is_empty() {
            recommendations
                .push("Consider setting up budgets to track and control spending.".to_string());
        } else {
            for budget in &self.budgets {
                let status = self.get_budget_status(budget).await?;
                if status.utilization_percent > 80.0 {
                    recommendations.push(format!(
                        "Budget '{}' is {:.1}% utilized. Consider reviewing usage or increasing the budget.",
                        budget.name,
                        status.utilization_percent
                    ));
                }
            }
        }

        Ok(recommendations)
    }

    /// Export comprehensive cost report
    pub async fn export_detailed_report(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        // Write report header
        writeln!(file, "# Claude AI Interactive - Cost Report")?;
        writeln!(
            file,
            "Generated: {}\n",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )?;

        // Global summary
        let global_summary = self.base_tracker.get_global_summary().await?;
        writeln!(file, "## Global Summary")?;
        writeln!(file, "- Total Cost: ${:.4}", global_summary.total_cost)?;
        writeln!(file, "- Total Commands: {}", global_summary.command_count)?;
        writeln!(
            file,
            "- Average Cost per Command: ${:.4}",
            global_summary.average_cost
        )?;
        writeln!(file, "- Total Tokens: {}", global_summary.total_tokens)?;
        writeln!(
            file,
            "- Date Range: {} to {}",
            global_summary.date_range.0.format("%Y-%m-%d"),
            global_summary.date_range.1.format("%Y-%m-%d")
        )?;
        writeln!(file)?;

        // Top commands
        writeln!(file, "## Top Commands by Cost")?;
        let top_commands = self.base_tracker.get_top_commands(10).await?;
        for (i, (cmd, cost)) in top_commands.iter().enumerate() {
            writeln!(file, "{}. {} - ${:.4}", i + 1, cmd, cost)?;
        }
        writeln!(file)?;

        // Cost by model
        writeln!(file, "## Cost by Model")?;
        let mut model_costs: Vec<_> = global_summary.by_model.iter().collect();
        model_costs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (model, cost) in model_costs {
            writeln!(file, "- {}: ${:.4}", model, cost)?;
        }
        writeln!(file)?;

        // Budget status
        if !self.budgets.is_empty() {
            writeln!(file, "## Budget Status")?;
            for budget in &self.budgets {
                let status = self.get_budget_status(budget).await?;
                writeln!(file, "### {}", budget.name)?;
                writeln!(file, "- Limit: ${:.2}", budget.limit_usd)?;
                writeln!(file, "- Current Spend: ${:.4}", status.current_spend)?;
                writeln!(file, "- Remaining: ${:.4}", status.remaining)?;
                writeln!(file, "- Utilization: {:.1}%", status.utilization_percent)?;
                if let Some(overage) = status.projected_overage {
                    writeln!(file, "- Projected Overage: ${:.4}", overage)?;
                }
                writeln!(file)?;
            }
        }

        // Recommendations
        let recommendations = self.get_optimization_recommendations().await?;
        if !recommendations.is_empty() {
            writeln!(file, "## Optimization Recommendations")?;
            for (i, rec) in recommendations.iter().enumerate() {
                writeln!(file, "{}. {}", i + 1, rec)?;
            }
        }

        Ok(())
    }

    /// Get cost trend analysis (simpler interface)
    pub async fn analyze_cost_trend(&self, period_days: u32) -> Result<CostTrend> {
        let filter = CostFilter::default();
        self.calculate_cost_trend(period_days, &filter).await
    }

    /// Check alerts for violations
    pub async fn check_alerts(&self) -> Result<Vec<(CostAlert, f64)>> {
        let mut triggered = Vec::new();

        for alert in &self.alerts {
            if !alert.enabled {
                continue;
            }

            let period_start = Utc::now() - Duration::days(alert.period_days as i64);
            let filter = CostFilter {
                since: Some(period_start),
                ..Default::default()
            };

            let summary = self.base_tracker.get_filtered_summary(&filter).await?;

            if summary.total_cost > alert.threshold_usd {
                triggered.push((alert.clone(), summary.total_cost));
            }
        }

        Ok(triggered)
    }

    /// Generate comprehensive cost report (alias for export_detailed_report)
    pub async fn generate_report(&self, path: &std::path::Path) -> Result<()> {
        self.export_detailed_report(path).await
    }

    // Private helper methods

    fn calculate_trend_direction(&self, costs: &[f64]) -> (TrendDirection, f64) {
        if costs.len() < 2 {
            return (TrendDirection::Stable, 0.0);
        }

        // Simple linear regression to determine trend
        let n = costs.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = costs.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, cost) in costs.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (cost - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        let slope = if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        let growth_rate = if y_mean != 0.0 {
            (slope / y_mean) * 100.0
        } else {
            0.0
        };

        let direction = if slope > 0.1 {
            TrendDirection::Increasing
        } else if slope < -0.1 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        (direction, growth_rate)
    }

    // Public accessor methods for testing
    #[cfg(test)]
    pub fn budgets(&self) -> &Vec<Budget> {
        &self.budgets
    }

    #[cfg(test)]
    pub fn alerts(&self) -> &Vec<CostAlert> {
        &self.alerts
    }

    #[cfg(test)]
    pub async fn record_cost(&mut self, entry: CostEntry) -> Result<()> {
        self.base_tracker.record_cost(entry).await
    }

    /// Get number of budgets (for testing)
    #[cfg(test)]
    pub fn budget_count(&self) -> usize {
        self.budgets.len()
    }

    /// Get number of alerts (for testing)  
    #[cfg(test)]
    pub fn alert_count(&self) -> usize {
        self.alerts.len()
    }

    /// Get budget by index (for testing)
    #[cfg(test)]
    pub fn get_budget(&self, index: usize) -> Option<&Budget> {
        self.budgets.get(index)
    }

    /// Get cost entries with filtering (for testing)
    #[cfg(test)]
    pub async fn get_entries(&self, filter: &CostFilter) -> Result<Vec<CostEntry>> {
        self.base_tracker.get_entries(filter).await
    }

    /// Export to CSV (for testing)
    #[cfg(test)]
    pub async fn export_csv(&self, path: &PathBuf) -> Result<()> {
        self.base_tracker.export_csv(path).await
    }

    /// Get global summary (for testing)
    #[cfg(test)]
    pub async fn get_global_summary(&self) -> Result<CostSummary> {
        self.base_tracker.get_global_summary().await
    }
}

// Note: Types are already imported from super module

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use uuid::Uuid;

    async fn create_test_tracker() -> AdvancedCostTracker {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("test_costs.json");
        let base_tracker = CostTracker::new(storage_path).unwrap();
        AdvancedCostTracker::new(base_tracker)
    }

    #[tokio::test]
    async fn test_budget_creation() {
        let mut tracker = create_test_tracker().await;

        let budget = Budget {
            id: Uuid::new_v4().to_string(),
            name: "Monthly Budget".to_string(),
            limit_usd: 100.0,
            period_days: 30,
            scope: BudgetScope::Global,
            alert_thresholds: vec![50.0, 80.0, 95.0],
            created_at: Utc::now(),
        };

        tracker.add_budget(budget.clone()).unwrap();
        assert_eq!(tracker.budget_count(), 1);
        assert_eq!(tracker.get_budget(0).unwrap().name, "Monthly Budget");
    }

    #[tokio::test]
    async fn test_invalid_budget() {
        let mut tracker = create_test_tracker().await;

        // Test negative budget limit
        let invalid_budget = Budget {
            id: Uuid::new_v4().to_string(),
            name: "Invalid Budget".to_string(),
            limit_usd: -10.0,
            period_days: 30,
            scope: BudgetScope::Global,
            alert_thresholds: vec![],
            created_at: Utc::now(),
        };

        assert!(tracker.add_budget(invalid_budget).is_err());

        // Test zero period days
        let invalid_budget2 = Budget {
            id: Uuid::new_v4().to_string(),
            name: "Invalid Budget 2".to_string(),
            limit_usd: 100.0,
            period_days: 0,
            scope: BudgetScope::Global,
            alert_thresholds: vec![],
            created_at: Utc::now(),
        };

        assert!(tracker.add_budget(invalid_budget2).is_err());
    }

    #[tokio::test]
    async fn test_budget_status() {
        let mut tracker = create_test_tracker().await;
        let session_id = Uuid::new_v4();

        // Add budget
        let budget = Budget {
            id: Uuid::new_v4().to_string(),
            name: "Test Budget".to_string(),
            limit_usd: 50.0,
            period_days: 30,
            scope: BudgetScope::Global,
            alert_thresholds: vec![50.0, 80.0],
            created_at: Utc::now() - Duration::days(5),
        };
        tracker.add_budget(budget.clone()).unwrap();

        // Add some costs
        for i in 0..5 {
            tracker
                .base_tracker
                .record_cost(CostEntry::new(
                    session_id,
                    format!("cmd_{}", i),
                    5.0,
                    100,
                    200,
                    1000,
                    "claude-3-opus".to_string(),
                ))
                .await
                .unwrap();
        }

        let status = tracker.get_budget_status(&budget).await.unwrap();

        assert_eq!(status.budget.id, budget.id);
        assert!((status.current_spend - 25.0).abs() < 0.0001);
        assert!((status.remaining - 25.0).abs() < 0.0001);
        assert!((status.utilization_percent - 50.0).abs() < 0.1);
        assert_eq!(status.alerts_triggered.len(), 1); // 50% alert
    }

    #[tokio::test]
    async fn test_cost_alerts() {
        let mut tracker = create_test_tracker().await;

        let alert = CostAlert {
            id: Uuid::new_v4().to_string(),
            name: "High Cost Alert".to_string(),
            threshold_usd: 10.0,
            period_days: 1,
            enabled: true,
            notification_channels: vec!["email".to_string(), "slack".to_string()],
        };

        tracker.add_alert(alert.clone()).unwrap();
        assert_eq!(tracker.alert_count(), 1);

        // Add costs that should trigger alert
        let session_id = Uuid::new_v4();
        tracker
            .record_cost(CostEntry::new(
                session_id,
                "expensive_command".to_string(),
                15.0,
                500,
                1000,
                3000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        let triggered = tracker.check_alerts().await.unwrap();
        assert_eq!(triggered.len(), 1);
        assert_eq!(triggered[0].0.id, alert.id);
        assert!((triggered[0].1 - 15.0).abs() < 0.0001);
    }

    #[tokio::test]
    async fn test_cost_trend_analysis() {
        let mut tracker = create_test_tracker().await;
        let session_id = Uuid::new_v4();

        // Add costs over several days with increasing trend
        for day in 0..7 {
            let cost = 10.0 + (day as f64 * 2.0); // Increasing costs
            let timestamp = Utc::now() - Duration::days(6 - day);

            let mut entry = CostEntry::new(
                session_id,
                format!("cmd_day_{}", day),
                cost,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            );
            entry.timestamp = timestamp;

            tracker.record_cost(entry).await.unwrap();
        }

        let trend = tracker.analyze_cost_trend(7).await.unwrap();

        assert_eq!(trend.period_days, 7);
        assert!(matches!(trend.trend_direction, TrendDirection::Increasing));
        assert!(trend.growth_rate > 0.0);
        assert!(trend.projected_monthly_cost > 0.0);
        assert_eq!(trend.daily_costs.len(), 7);
    }

    #[tokio::test]
    async fn test_stable_trend() {
        let mut tracker = create_test_tracker().await;
        let session_id = Uuid::new_v4();

        // Add consistent costs
        for day in 0..5 {
            let timestamp = Utc::now() - Duration::days(4 - day);
            let mut entry = CostEntry::new(
                session_id,
                format!("cmd_{}", day),
                10.0, // Same cost each day
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            );
            entry.timestamp = timestamp;
            tracker.record_cost(entry).await.unwrap();
        }

        let trend = tracker.analyze_cost_trend(5).await.unwrap();
        assert!(matches!(trend.trend_direction, TrendDirection::Stable));
    }

    #[tokio::test]
    async fn test_session_budget() {
        let mut tracker = create_test_tracker().await;
        let session_id = Uuid::new_v4();

        // Add session-specific budget
        let budget = Budget {
            id: Uuid::new_v4().to_string(),
            name: "Session Budget".to_string(),
            limit_usd: 20.0,
            period_days: 30,
            scope: BudgetScope::Session(session_id),
            alert_thresholds: vec![90.0],
            created_at: Utc::now(),
        };
        tracker.add_budget(budget.clone()).unwrap();

        // Add costs to different sessions
        tracker
            .record_cost(CostEntry::new(
                session_id,
                "cmd1".to_string(),
                15.0,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        let other_session = Uuid::new_v4();
        tracker
            .record_cost(CostEntry::new(
                other_session,
                "cmd2".to_string(),
                10.0,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        let status = tracker.get_budget_status(&budget).await.unwrap();
        assert!((status.current_spend - 15.0).abs() < 0.0001); // Only session costs
        assert!((status.remaining - 5.0).abs() < 0.0001);
    }

    #[tokio::test]
    async fn test_command_budget() {
        let mut tracker = create_test_tracker().await;
        let session_id = Uuid::new_v4();

        // Add command-specific budget
        let budget = Budget {
            id: Uuid::new_v4().to_string(),
            name: "Analyze Command Budget".to_string(),
            limit_usd: 10.0,
            period_days: 7,
            scope: BudgetScope::Command("analyze".to_string()),
            alert_thresholds: vec![],
            created_at: Utc::now(),
        };
        tracker.add_budget(budget.clone()).unwrap();

        // Add various commands
        tracker
            .record_cost(CostEntry::new(
                session_id,
                "analyze".to_string(),
                8.0,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        tracker
            .record_cost(CostEntry::new(
                session_id,
                "generate".to_string(),
                5.0,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        let status = tracker.get_budget_status(&budget).await.unwrap();
        assert!((status.current_spend - 8.0).abs() < 0.0001); // Only analyze command
    }

    #[tokio::test]
    async fn test_optimization_recommendations() {
        let mut tracker = create_test_tracker().await;
        let session_id = Uuid::new_v4();

        // Add costs with patterns that should trigger recommendations

        // Many small commands
        for _i in 0..20 {
            tracker
                .base_tracker
                .record_cost(CostEntry::new(
                    session_id,
                    "small_cmd".to_string(),
                    0.001,
                    10,
                    20,
                    100,
                    "claude-3-haiku".to_string(),
                ))
                .await
                .unwrap();
        }

        // One expensive command
        tracker
            .record_cost(CostEntry::new(
                session_id,
                "expensive_cmd".to_string(),
                5.0,
                1000,
                2000,
                5000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        let recommendations = tracker.get_optimization_recommendations().await.unwrap();
        assert!(!recommendations.is_empty());

        // Should recommend batching small commands
        assert!(recommendations.iter().any(|r| r.contains("batch")));
    }

    #[tokio::test]
    async fn test_generate_report() {
        let mut tracker = create_test_tracker().await;
        let session_id = Uuid::new_v4();

        // Add some test data
        tracker
            .record_cost(CostEntry::new(
                session_id,
                "test_cmd".to_string(),
                1.5,
                150,
                300,
                2000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        // Add a budget
        let budget = Budget {
            id: Uuid::new_v4().to_string(),
            name: "Test Budget".to_string(),
            limit_usd: 10.0,
            period_days: 30,
            scope: BudgetScope::Global,
            alert_thresholds: vec![],
            created_at: Utc::now(),
        };
        tracker.add_budget(budget).unwrap();

        let temp_dir = tempdir().unwrap();
        let report_path = temp_dir.path().join("cost_report.md");

        tracker.generate_report(&report_path).await.unwrap();

        assert!(report_path.exists());
        let content = tokio::fs::read_to_string(&report_path).await.unwrap();
        assert!(content.contains("# Claude AI Interactive - Cost Report"));
        assert!(content.contains("Total Cost:"));
        assert!(content.contains("Budget Status"));
    }

    #[test]
    fn test_trend_calculation() {
        let tracker = AdvancedCostTracker {
            base_tracker: CostTracker::new(PathBuf::from("/tmp/test")).unwrap(),
            budgets: vec![],
            alerts: vec![],
        };

        // Test increasing trend
        let costs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (direction, growth_rate) = tracker.calculate_trend_direction(&costs);
        assert!(matches!(direction, TrendDirection::Increasing));
        assert!(growth_rate > 0.0);

        // Test decreasing trend
        let costs = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let (direction, growth_rate) = tracker.calculate_trend_direction(&costs);
        assert!(matches!(direction, TrendDirection::Decreasing));
        assert!(growth_rate < 0.0);

        // Test stable trend
        let costs = vec![2.0, 2.1, 2.0, 1.9, 2.0];
        let (direction, _) = tracker.calculate_trend_direction(&costs);
        assert!(matches!(direction, TrendDirection::Stable));

        // Test insufficient data
        let costs = vec![1.0];
        let (direction, growth_rate) = tracker.calculate_trend_direction(&costs);
        assert!(matches!(direction, TrendDirection::Stable));
        assert_eq!(growth_rate, 0.0);
    }

    #[tokio::test]
    async fn test_projected_overage() {
        let mut tracker = create_test_tracker().await;
        let session_id = Uuid::new_v4();

        // Add budget that starts 10 days ago
        let budget = Budget {
            id: Uuid::new_v4().to_string(),
            name: "Overage Test Budget".to_string(),
            limit_usd: 100.0,
            period_days: 30,
            scope: BudgetScope::Global,
            alert_thresholds: vec![],
            created_at: Utc::now() - Duration::days(10),
        };
        tracker.add_budget(budget.clone()).unwrap();

        // Add costs that would project over budget
        // $50 in 10 days = $5/day, projected to $150 in 30 days
        for i in 0..10 {
            let timestamp = Utc::now() - Duration::days(9 - i);
            let mut entry = CostEntry::new(
                session_id,
                format!("cmd_{}", i),
                5.0,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            );
            entry.timestamp = timestamp;
            tracker.record_cost(entry).await.unwrap();
        }

        let status = tracker.get_budget_status(&budget).await.unwrap();
        assert!(status.projected_overage.is_some());
        let overage = status.projected_overage.unwrap();
        assert!(overage > 0.0); // Should project overage
    }
}
