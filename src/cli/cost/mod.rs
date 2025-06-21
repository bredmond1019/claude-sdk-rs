//! Cost tracking and management
//!
//! This module handles:
//! - Tracking costs per command and session
//! - Aggregating cost data over time
//! - Cost formatting and display
//! - Export functionality

pub mod tracker;

#[cfg(test)]
pub mod tracker_test;

#[cfg(test)]
pub mod budget_tests;

#[cfg(test)]
pub mod trend_tests;

use crate::{cli::error::InteractiveError, cli::error::Result, cli::session::SessionId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// Cost tracking entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEntry {
    pub id: String,
    pub session_id: SessionId,
    pub command_name: String,
    pub cost_usd: f64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: u64,
    pub model: String,
}

impl CostEntry {
    /// Create a new cost entry
    pub fn new(
        session_id: SessionId,
        command_name: String,
        cost_usd: f64,
        input_tokens: u32,
        output_tokens: u32,
        duration_ms: u64,
        model: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            session_id,
            command_name,
            cost_usd,
            input_tokens,
            output_tokens,
            timestamp: Utc::now(),
            duration_ms,
            model,
        }
    }
}

/// Aggregated cost statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSummary {
    pub total_cost: f64,
    pub command_count: usize,
    pub average_cost: f64,
    pub total_tokens: u32,
    pub date_range: (DateTime<Utc>, DateTime<Utc>),
    pub by_command: HashMap<String, f64>,
    pub by_model: HashMap<String, f64>,
}

impl Default for CostSummary {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            total_cost: 0.0,
            command_count: 0,
            average_cost: 0.0,
            total_tokens: 0,
            date_range: (now, now),
            by_command: HashMap::new(),
            by_model: HashMap::new(),
        }
    }
}

/// Cost filter criteria
#[derive(Debug, Clone)]
pub struct CostFilter {
    pub session_id: Option<SessionId>,
    pub command_pattern: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
    pub min_cost: Option<f64>,
    pub max_cost: Option<f64>,
    pub model: Option<String>,
}

impl Default for CostFilter {
    fn default() -> Self {
        Self {
            session_id: None,
            command_pattern: None,
            since: None,
            until: None,
            min_cost: None,
            max_cost: None,
            model: None,
        }
    }
}

/// Cost tracker service with persistent storage
pub struct CostTracker {
    storage_path: PathBuf,
    entries: Vec<CostEntry>,
}

impl CostTracker {
    /// Create a new cost tracker with the specified storage path
    pub fn new(storage_path: PathBuf) -> Result<Self> {
        let mut tracker = Self {
            storage_path,
            entries: Vec::new(),
        };

        // Load existing entries if storage file exists
        tracker.load_entries()?;
        Ok(tracker)
    }

    /// Record a cost entry
    pub async fn record_cost(&mut self, entry: CostEntry) -> Result<()> {
        self.entries.push(entry);
        self.save_entries().await?;
        Ok(())
    }

    /// Get cost summary for a session
    pub async fn get_session_summary(&self, session_id: SessionId) -> Result<CostSummary> {
        let session_entries: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.session_id == session_id)
            .collect();

        self.calculate_summary(&session_entries)
    }

    /// Get global cost summary
    pub async fn get_global_summary(&self) -> Result<CostSummary> {
        let all_entries: Vec<_> = self.entries.iter().collect();
        self.calculate_summary(&all_entries)
    }

    /// Get filtered cost summary
    pub async fn get_filtered_summary(&self, filter: &CostFilter) -> Result<CostSummary> {
        let filtered_entries: Vec<_> = self
            .entries
            .iter()
            .filter(|e| self.matches_filter(e, filter))
            .collect();

        self.calculate_summary(&filtered_entries)
    }

    /// Get cost entries with filtering
    pub async fn get_entries(&self, filter: &CostFilter) -> Result<Vec<CostEntry>> {
        Ok(self
            .entries
            .iter()
            .filter(|e| self.matches_filter(e, filter))
            .cloned()
            .collect())
    }

    /// Get top spending commands
    pub async fn get_top_commands(&self, limit: usize) -> Result<Vec<(String, f64)>> {
        let mut command_costs: HashMap<String, f64> = HashMap::new();

        for entry in &self.entries {
            *command_costs
                .entry(entry.command_name.clone())
                .or_insert(0.0) += entry.cost_usd;
        }

        let mut sorted_commands: Vec<_> = command_costs.into_iter().collect();
        sorted_commands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_commands.truncate(limit);

        Ok(sorted_commands)
    }

    /// Export cost data to CSV
    pub async fn export_csv(&self, path: &PathBuf) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        // Write CSV header
        writeln!(file, "id,session_id,command_name,cost_usd,input_tokens,output_tokens,timestamp,duration_ms,model")?;

        // Write entries
        for entry in &self.entries {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{}",
                entry.id,
                entry.session_id,
                entry.command_name,
                entry.cost_usd,
                entry.input_tokens,
                entry.output_tokens,
                entry.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
                entry.duration_ms,
                entry.model
            )?;
        }

        Ok(())
    }

    /// Clear all cost data
    pub async fn clear_all(&mut self) -> Result<()> {
        self.entries.clear();
        self.save_entries().await?;
        Ok(())
    }

    /// Clear cost data for a specific session
    pub async fn clear_session(&mut self, session_id: SessionId) -> Result<()> {
        self.entries.retain(|e| e.session_id != session_id);
        self.save_entries().await?;
        Ok(())
    }

    // Private helper methods

    fn load_entries(&mut self) -> Result<()> {
        if !self.storage_path.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&self.storage_path)?;
        if content.trim().is_empty() {
            return Ok(());
        }

        self.entries = serde_json::from_str(&content).map_err(|e| {
            InteractiveError::CostTracking(format!("Failed to parse cost data: {}", e))
        })?;

        Ok(())
    }

    async fn save_entries(&self) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.storage_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let content = serde_json::to_string_pretty(&self.entries)?;
        tokio::fs::write(&self.storage_path, content).await?;

        Ok(())
    }

    fn matches_filter(&self, entry: &CostEntry, filter: &CostFilter) -> bool {
        if let Some(session_id) = filter.session_id {
            if entry.session_id != session_id {
                return false;
            }
        }

        if let Some(pattern) = &filter.command_pattern {
            if !entry.command_name.contains(pattern) {
                return false;
            }
        }

        if let Some(since) = filter.since {
            if entry.timestamp < since {
                return false;
            }
        }

        if let Some(until) = filter.until {
            if entry.timestamp > until {
                return false;
            }
        }

        if let Some(min_cost) = filter.min_cost {
            if entry.cost_usd < min_cost {
                return false;
            }
        }

        if let Some(max_cost) = filter.max_cost {
            if entry.cost_usd > max_cost {
                return false;
            }
        }

        if let Some(model) = &filter.model {
            if entry.model != *model {
                return false;
            }
        }

        true
    }

    fn calculate_summary(&self, entries: &[&CostEntry]) -> Result<CostSummary> {
        if entries.is_empty() {
            return Ok(CostSummary::default());
        }

        let total_cost: f64 = entries.iter().map(|e| e.cost_usd).sum();
        let command_count = entries.len();
        let average_cost = total_cost / command_count as f64;
        let total_tokens: u32 = entries
            .iter()
            .map(|e| e.input_tokens + e.output_tokens)
            .sum();

        let min_date = entries.iter().map(|e| e.timestamp).min().unwrap();
        let max_date = entries.iter().map(|e| e.timestamp).max().unwrap();

        let mut by_command: HashMap<String, f64> = HashMap::new();
        let mut by_model: HashMap<String, f64> = HashMap::new();

        for entry in entries {
            *by_command.entry(entry.command_name.clone()).or_insert(0.0) += entry.cost_usd;
            *by_model.entry(entry.model.clone()).or_insert(0.0) += entry.cost_usd;
        }

        Ok(CostSummary {
            total_cost,
            command_count,
            average_cost,
            total_tokens,
            date_range: (min_date, max_date),
            by_command,
            by_model,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use tempfile::tempdir;

    #[test]
    fn test_cost_entry_creation() {
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

        assert_eq!(entry.session_id, session_id);
        assert_eq!(entry.command_name, "test_command");
        assert_eq!(entry.cost_usd, 0.025);
        assert_eq!(entry.input_tokens, 100);
        assert_eq!(entry.output_tokens, 200);
        assert_eq!(entry.duration_ms, 1500);
        assert_eq!(entry.model, "claude-3-opus");
        assert!(!entry.id.is_empty());
    }

    #[test]
    fn test_cost_summary_default() {
        let summary = CostSummary::default();

        assert_eq!(summary.total_cost, 0.0);
        assert_eq!(summary.command_count, 0);
        assert_eq!(summary.average_cost, 0.0);
        assert_eq!(summary.total_tokens, 0);
        assert!(summary.by_command.is_empty());
        assert!(summary.by_model.is_empty());
    }

    #[tokio::test]
    async fn test_cost_tracker_new() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("cost_data.json");

        let tracker = CostTracker::new(storage_path.clone()).unwrap();
        assert_eq!(tracker.entries.len(), 0);
        assert_eq!(tracker.storage_path, storage_path);
    }

    #[tokio::test]
    async fn test_record_cost() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("cost_data.json");
        let mut tracker = CostTracker::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();
        let entry = CostEntry::new(
            session_id,
            "test_command".to_string(),
            0.03,
            150,
            300,
            2000,
            "claude-3-opus".to_string(),
        );

        tracker.record_cost(entry.clone()).await.unwrap();
        assert_eq!(tracker.entries.len(), 1);
        assert_eq!(tracker.entries[0].cost_usd, 0.03);
    }

    #[tokio::test]
    async fn test_get_session_summary() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("cost_data.json");
        let mut tracker = CostTracker::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add multiple entries for the same session
        for i in 0..5 {
            let entry = CostEntry::new(
                session_id,
                format!("command_{}", i),
                0.01 * (i + 1) as f64,
                50 + i as u32 * 10,
                100 + i as u32 * 20,
                1000 + i as u64 * 100,
                "claude-3-opus".to_string(),
            );
            tracker.record_cost(entry).await.unwrap();
        }

        let summary = tracker.get_session_summary(session_id).await.unwrap();

        assert_eq!(summary.command_count, 5);
        assert!((summary.total_cost - 0.15).abs() < 0.0001); // 0.01 + 0.02 + 0.03 + 0.04 + 0.05
        assert!((summary.average_cost - 0.03).abs() < 0.0001);
        assert_eq!(summary.total_tokens, 1050); // Sum of all input + output tokens
        assert_eq!(summary.by_command.len(), 5);
        assert_eq!(summary.by_model.len(), 1);
        assert!((summary.by_model.get("claude-3-opus").unwrap() - 0.15).abs() < 0.0001);
    }

    #[tokio::test]
    async fn test_get_global_summary() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("cost_data.json");
        let mut tracker = CostTracker::new(storage_path).unwrap();

        let session1 = Uuid::new_v4();
        let session2 = Uuid::new_v4();

        // Add entries for multiple sessions
        tracker
            .record_cost(CostEntry::new(
                session1,
                "cmd1".to_string(),
                0.05,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        tracker
            .record_cost(CostEntry::new(
                session2,
                "cmd2".to_string(),
                0.03,
                50,
                100,
                800,
                "claude-3-sonnet".to_string(),
            ))
            .await
            .unwrap();

        let summary = tracker.get_global_summary().await.unwrap();

        assert_eq!(summary.command_count, 2);
        assert!((summary.total_cost - 0.08).abs() < 0.0001);
        assert!((summary.average_cost - 0.04).abs() < 0.0001);
        assert_eq!(summary.total_tokens, 450);
        assert_eq!(summary.by_model.len(), 2);
    }

    #[tokio::test]
    async fn test_cost_filter() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("cost_data.json");
        let mut tracker = CostTracker::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add various entries
        tracker
            .record_cost(CostEntry::new(
                session_id,
                "analyze".to_string(),
                0.10,
                200,
                400,
                2000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        tracker
            .record_cost(CostEntry::new(
                session_id,
                "generate".to_string(),
                0.05,
                100,
                200,
                1000,
                "claude-3-sonnet".to_string(),
            ))
            .await
            .unwrap();

        tracker
            .record_cost(CostEntry::new(
                session_id,
                "analyze_detailed".to_string(),
                0.15,
                300,
                600,
                3000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        // Test filtering by minimum cost
        let filter = CostFilter {
            min_cost: Some(0.08),
            ..Default::default()
        };
        let summary = tracker.get_filtered_summary(&filter).await.unwrap();
        assert_eq!(summary.command_count, 2); // Only entries with cost >= 0.08

        // Test filtering by command pattern
        let filter = CostFilter {
            command_pattern: Some("analyze".to_string()),
            ..Default::default()
        };
        let summary = tracker.get_filtered_summary(&filter).await.unwrap();
        assert_eq!(summary.command_count, 2); // "analyze" and "analyze_detailed"

        // Test filtering by model
        let filter = CostFilter {
            model: Some("claude-3-sonnet".to_string()),
            ..Default::default()
        };
        let summary = tracker.get_filtered_summary(&filter).await.unwrap();
        assert_eq!(summary.command_count, 1);
        assert!((summary.total_cost - 0.05).abs() < 0.0001);
    }

    #[tokio::test]
    async fn test_get_top_commands() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("cost_data.json");
        let mut tracker = CostTracker::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add multiple entries with different costs
        tracker
            .record_cost(CostEntry::new(
                session_id,
                "expensive_cmd".to_string(),
                0.50,
                500,
                1000,
                5000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        tracker
            .record_cost(CostEntry::new(
                session_id,
                "cheap_cmd".to_string(),
                0.01,
                10,
                20,
                100,
                "claude-3-haiku".to_string(),
            ))
            .await
            .unwrap();

        tracker
            .record_cost(CostEntry::new(
                session_id,
                "medium_cmd".to_string(),
                0.10,
                100,
                200,
                1000,
                "claude-3-sonnet".to_string(),
            ))
            .await
            .unwrap();

        // Add another expensive_cmd to test aggregation
        tracker
            .record_cost(CostEntry::new(
                session_id,
                "expensive_cmd".to_string(),
                0.45,
                450,
                900,
                4500,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        let top_commands = tracker.get_top_commands(2).await.unwrap();

        assert_eq!(top_commands.len(), 2);
        assert_eq!(top_commands[0].0, "expensive_cmd");
        assert!((top_commands[0].1 - 0.95).abs() < 0.0001); // 0.50 + 0.45
        assert_eq!(top_commands[1].0, "medium_cmd");
        assert!((top_commands[1].1 - 0.10).abs() < 0.0001);
    }

    #[tokio::test]
    async fn test_export_csv() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("cost_data.json");
        let mut tracker = CostTracker::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();
        let entry = CostEntry::new(
            session_id,
            "test_export".to_string(),
            0.025,
            100,
            200,
            1500,
            "claude-3-opus".to_string(),
        );

        tracker.record_cost(entry).await.unwrap();

        let export_path = temp_dir.path().join("export.csv");
        tracker.export_csv(&export_path).await.unwrap();

        assert!(export_path.exists());

        let content = tokio::fs::read_to_string(&export_path).await.unwrap();
        assert!(content.contains("id,session_id,command_name,cost_usd"));
        assert!(content.contains("test_export"));
        assert!(content.contains("0.025"));
        assert!(content.contains("claude-3-opus"));
    }

    #[tokio::test]
    async fn test_clear_operations() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("cost_data.json");
        let mut tracker = CostTracker::new(storage_path).unwrap();

        let session1 = Uuid::new_v4();
        let session2 = Uuid::new_v4();

        // Add entries for two sessions
        tracker
            .record_cost(CostEntry::new(
                session1,
                "cmd1".to_string(),
                0.05,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        tracker
            .record_cost(CostEntry::new(
                session2,
                "cmd2".to_string(),
                0.03,
                50,
                100,
                800,
                "claude-3-sonnet".to_string(),
            ))
            .await
            .unwrap();

        assert_eq!(tracker.entries.len(), 2);

        // Test clear session
        tracker.clear_session(session1).await.unwrap();
        assert_eq!(tracker.entries.len(), 1);
        assert_eq!(tracker.entries[0].session_id, session2);

        // Test clear all
        tracker.clear_all().await.unwrap();
        assert_eq!(tracker.entries.len(), 0);
    }

    #[tokio::test]
    async fn test_persistence() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("cost_data.json");

        let session_id = Uuid::new_v4();

        // Create tracker and add entries
        {
            let mut tracker = CostTracker::new(storage_path.clone()).unwrap();
            tracker
                .record_cost(CostEntry::new(
                    session_id,
                    "persistent_cmd".to_string(),
                    0.075,
                    150,
                    300,
                    2000,
                    "claude-3-opus".to_string(),
                ))
                .await
                .unwrap();
        }

        // Create new tracker instance and verify data persisted
        {
            let tracker = CostTracker::new(storage_path).unwrap();
            assert_eq!(tracker.entries.len(), 1);
            assert_eq!(tracker.entries[0].command_name, "persistent_cmd");
            assert_eq!(tracker.entries[0].cost_usd, 0.075);
        }
    }

    #[tokio::test]
    async fn test_date_range_filtering() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("cost_data.json");
        let mut tracker = CostTracker::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();
        let now = Utc::now();

        // Add entry
        tracker
            .record_cost(CostEntry::new(
                session_id,
                "recent_cmd".to_string(),
                0.05,
                100,
                200,
                1000,
                "claude-3-opus".to_string(),
            ))
            .await
            .unwrap();

        // Test filtering with date range that includes the entry
        let filter = CostFilter {
            since: Some(now - Duration::hours(1)),
            until: Some(now + Duration::hours(1)),
            ..Default::default()
        };
        let entries = tracker.get_entries(&filter).await.unwrap();
        assert_eq!(entries.len(), 1);

        // Test filtering with date range that excludes the entry
        let filter = CostFilter {
            since: Some(now + Duration::hours(1)),
            ..Default::default()
        };
        let entries = tracker.get_entries(&filter).await.unwrap();
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn test_cost_filter_default() {
        let filter = CostFilter::default();

        assert!(filter.session_id.is_none());
        assert!(filter.command_pattern.is_none());
        assert!(filter.since.is_none());
        assert!(filter.until.is_none());
        assert!(filter.min_cost.is_none());
        assert!(filter.max_cost.is_none());
        assert!(filter.model.is_none());
    }

    #[tokio::test]
    async fn test_empty_tracker_operations() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("cost_data.json");
        let tracker = CostTracker::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Test operations on empty tracker
        let summary = tracker.get_session_summary(session_id).await.unwrap();
        assert_eq!(summary.command_count, 0);
        assert_eq!(summary.total_cost, 0.0);

        let global_summary = tracker.get_global_summary().await.unwrap();
        assert_eq!(global_summary.command_count, 0);

        let top_commands = tracker.get_top_commands(5).await.unwrap();
        assert!(top_commands.is_empty());

        let filter = CostFilter::default();
        let entries = tracker.get_entries(&filter).await.unwrap();
        assert!(entries.is_empty());
    }
}
