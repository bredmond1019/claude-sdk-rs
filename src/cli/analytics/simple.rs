//! Simple analytics implementation for integration testing
//!
//! This module provides a simplified analytics engine that can be used
//! for testing and basic analytics functionality.

use crate::{
    cli::cost::CostSummary,
    cli::history::{HistoryEntry, HistoryStats},
    cli::session::SessionId,
};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Simple analytics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleAnalyticsSummary {
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_cost: f64,
    pub total_commands: usize,
    pub success_rate: f64,
    pub insights: Vec<String>,
}

/// Simple session report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleSessionReport {
    pub session_id: SessionId,
    pub cost_summary: CostSummary,
    pub history_stats: HistoryStats,
    pub generated_at: DateTime<Utc>,
}

/// Simple dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleDashboardData {
    pub today_cost: f64,
    pub today_commands: usize,
    pub success_rate: f64,
    pub recent_activity: Vec<HistoryEntry>,
    pub top_commands: Vec<(String, f64)>,
    pub last_updated: DateTime<Utc>,
}

/// Simple analytics engine
pub struct SimpleAnalyticsEngine;

impl SimpleAnalyticsEngine {
    /// Create a new simple analytics engine
    pub fn new() -> Self {
        Self
    }

    /// Generate a simple analytics summary
    pub fn generate_summary(&self, _period_days: u32) -> SimpleAnalyticsSummary {
        SimpleAnalyticsSummary {
            period_start: Utc::now() - Duration::days(7),
            period_end: Utc::now(),
            total_cost: 0.5,
            total_commands: 10,
            success_rate: 85.0,
            insights: vec![
                "Total spending is within normal range".to_string(),
                "Command success rate is healthy".to_string(),
            ],
        }
    }

    /// Generate a simple session report
    pub fn generate_session_report(&self, session_id: SessionId) -> SimpleSessionReport {
        let now = Utc::now();

        SimpleSessionReport {
            session_id,
            cost_summary: CostSummary {
                total_cost: 0.25,
                command_count: 5,
                average_cost: 0.05,
                total_tokens: 1500,
                date_range: (now - Duration::hours(1), now),
                by_command: {
                    let mut map = HashMap::new();
                    map.insert("analyze".to_string(), 0.15);
                    map.insert("generate".to_string(), 0.10);
                    map
                },
                by_model: {
                    let mut map = HashMap::new();
                    map.insert("claude-3-opus".to_string(), 0.25);
                    map
                },
            },
            history_stats: HistoryStats {
                total_entries: 5,
                successful_commands: 4,
                failed_commands: 1,
                success_rate: 80.0,
                total_cost: 0.25,
                total_duration_ms: 15000,
                average_duration_ms: 3000.0,
                average_cost: 0.05,
                command_counts: {
                    let mut map = HashMap::new();
                    map.insert("analyze".to_string(), 3);
                    map.insert("generate".to_string(), 2);
                    map
                },
                model_usage: {
                    let mut map = HashMap::new();
                    map.insert("claude-3-opus".to_string(), 5);
                    map
                },
                date_range: (now - Duration::hours(1), now),
            },
            generated_at: now,
        }
    }

    /// Get simple dashboard data
    pub fn get_dashboard_data(&self) -> SimpleDashboardData {
        let now = Utc::now();

        SimpleDashboardData {
            today_cost: 0.15,
            today_commands: 3,
            success_rate: 100.0,
            recent_activity: vec![HistoryEntry::new(
                uuid::Uuid::new_v4(),
                "test-command".to_string(),
                vec!["--option".to_string()],
                "Test output".to_string(),
                true,
                1500,
            )],
            top_commands: vec![("analyze".to_string(), 0.08), ("generate".to_string(), 0.07)],
            last_updated: now,
        }
    }
}
