//! Command history storage and search
//!
//! This module handles:
//! - Storing command execution history
//! - Searching through historical data
//! - History export functionality
//! - Pagination and filtering

pub mod store;

#[cfg(test)]
pub mod store_test;

use crate::{cli::error::InteractiveError, cli::error::Result, cli::session::SessionId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// History entry for a command execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: String,
    pub session_id: SessionId,
    pub command_name: String,
    pub args: Vec<String>,
    pub output: String,
    pub error: Option<String>,
    pub cost_usd: Option<f64>,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: u64,
    pub success: bool,
    pub model: Option<String>,
    pub tags: Vec<String>,
}

impl HistoryEntry {
    /// Create a new history entry
    pub fn new(
        session_id: SessionId,
        command_name: String,
        args: Vec<String>,
        output: String,
        success: bool,
        duration_ms: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            session_id,
            command_name,
            args,
            output,
            error: None,
            cost_usd: None,
            input_tokens: None,
            output_tokens: None,
            timestamp: Utc::now(),
            duration_ms,
            success,
            model: None,
            tags: Vec::new(),
        }
    }

    /// Add cost information to the entry
    pub fn with_cost(
        mut self,
        cost_usd: f64,
        input_tokens: u32,
        output_tokens: u32,
        model: String,
    ) -> Self {
        self.cost_usd = Some(cost_usd);
        self.input_tokens = Some(input_tokens);
        self.output_tokens = Some(output_tokens);
        self.model = Some(model);
        self
    }

    /// Add error information to the entry
    pub fn with_error(mut self, error: String) -> Self {
        self.error = Some(error);
        self.success = false;
        self
    }

    /// Add tags to the entry
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Search criteria for history queries
#[derive(Debug, Clone)]
pub struct HistorySearch {
    pub session_id: Option<SessionId>,
    pub command_pattern: Option<String>,
    pub output_pattern: Option<String>,
    pub error_pattern: Option<String>,
    pub success_only: bool,
    pub failures_only: bool,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
    pub min_duration_ms: Option<u64>,
    pub max_duration_ms: Option<u64>,
    pub min_cost: Option<f64>,
    pub max_cost: Option<f64>,
    pub model: Option<String>,
    pub tags: Vec<String>,
    pub limit: usize,
    pub offset: usize,
    pub sort_by: SortField,
    pub sort_desc: bool,
}

impl Default for HistorySearch {
    fn default() -> Self {
        Self {
            session_id: None,
            command_pattern: None,
            output_pattern: None,
            error_pattern: None,
            success_only: false,
            failures_only: false,
            since: None,
            until: None,
            min_duration_ms: None,
            max_duration_ms: None,
            min_cost: None,
            max_cost: None,
            model: None,
            tags: Vec::new(),
            limit: 100,
            offset: 0,
            sort_by: SortField::Timestamp,
            sort_desc: true,
        }
    }
}

/// Fields available for sorting history results
#[derive(Debug, Clone)]
pub enum SortField {
    Timestamp,
    Duration,
    Cost,
    Command,
}

/// History statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryStats {
    pub total_entries: usize,
    pub successful_commands: usize,
    pub failed_commands: usize,
    pub success_rate: f64,
    pub total_cost: f64,
    pub total_duration_ms: u64,
    pub average_duration_ms: f64,
    pub average_cost: f64,
    pub command_counts: HashMap<String, usize>,
    pub model_usage: HashMap<String, usize>,
    pub date_range: (DateTime<Utc>, DateTime<Utc>),
}

/// History export format
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Html,
}

/// History store service with persistent storage
pub struct HistoryStore {
    storage_path: PathBuf,
    entries: Vec<HistoryEntry>,
    index: HashMap<String, usize>, // ID -> index mapping for fast lookups
}

impl HistoryStore {
    /// Create a new history store with the specified storage path
    pub fn new(storage_path: PathBuf) -> Result<Self> {
        let mut store = Self {
            storage_path,
            entries: Vec::new(),
            index: HashMap::new(),
        };

        // Load existing entries if storage file exists
        store.load_entries()?;
        Ok(store)
    }

    /// Store a history entry
    pub async fn store_entry(&mut self, entry: HistoryEntry) -> Result<()> {
        let id = entry.id.clone();
        let index = self.entries.len();

        self.entries.push(entry);
        self.index.insert(id, index);

        self.save_entries().await?;
        Ok(())
    }

    /// Update an existing history entry
    pub async fn update_entry(&mut self, id: &str, entry: HistoryEntry) -> Result<()> {
        if let Some(&index) = self.index.get(id) {
            self.entries[index] = entry;
            self.save_entries().await?;
            Ok(())
        } else {
            Err(InteractiveError::History(format!(
                "Entry with ID {} not found",
                id
            )))
        }
    }

    /// Get a specific history entry by ID
    pub fn get_entry(&self, id: &str) -> Option<&HistoryEntry> {
        self.index
            .get(id)
            .and_then(|&index| self.entries.get(index))
    }

    /// Search history entries
    pub async fn search(&self, criteria: &HistorySearch) -> Result<Vec<HistoryEntry>> {
        let mut results: Vec<_> = self
            .entries
            .iter()
            .filter(|e| self.matches_criteria(e, criteria))
            .cloned()
            .collect();

        // Sort results
        self.sort_entries(&mut results, &criteria.sort_by, criteria.sort_desc);

        // Apply pagination
        let start = criteria.offset;
        let _end = start + criteria.limit;
        results = results
            .into_iter()
            .skip(start)
            .take(criteria.limit)
            .collect();

        Ok(results)
    }

    /// Get history statistics
    pub async fn get_stats(&self, filter: Option<&HistorySearch>) -> Result<HistoryStats> {
        let entries: Vec<_> = if let Some(criteria) = filter {
            self.entries
                .iter()
                .filter(|e| self.matches_criteria(e, criteria))
                .collect()
        } else {
            self.entries.iter().collect()
        };

        if entries.is_empty() {
            let now = Utc::now();
            return Ok(HistoryStats {
                total_entries: 0,
                successful_commands: 0,
                failed_commands: 0,
                success_rate: 0.0,
                total_cost: 0.0,
                total_duration_ms: 0,
                average_duration_ms: 0.0,
                average_cost: 0.0,
                command_counts: HashMap::new(),
                model_usage: HashMap::new(),
                date_range: (now, now),
            });
        }

        let total_entries = entries.len();
        let successful_commands = entries.iter().filter(|e| e.success).count();
        let failed_commands = total_entries - successful_commands;
        let success_rate = (successful_commands as f64 / total_entries as f64) * 100.0;

        let total_cost: f64 = entries.iter().filter_map(|e| e.cost_usd).sum();
        let total_duration_ms: u64 = entries.iter().map(|e| e.duration_ms).sum();
        let average_duration_ms = total_duration_ms as f64 / total_entries as f64;
        let cost_count = entries.iter().filter(|e| e.cost_usd.is_some()).count();
        let average_cost = if cost_count > 0 {
            total_cost / cost_count as f64
        } else {
            0.0
        };

        let mut command_counts: HashMap<String, usize> = HashMap::new();
        let mut model_usage: HashMap<String, usize> = HashMap::new();

        for entry in &entries {
            *command_counts
                .entry(entry.command_name.clone())
                .or_insert(0) += 1;
            if let Some(model) = &entry.model {
                *model_usage.entry(model.clone()).or_insert(0) += 1;
            }
        }

        let min_date = entries.iter().map(|e| e.timestamp).min().unwrap();
        let max_date = entries.iter().map(|e| e.timestamp).max().unwrap();

        Ok(HistoryStats {
            total_entries,
            successful_commands,
            failed_commands,
            success_rate,
            total_cost,
            total_duration_ms,
            average_duration_ms,
            average_cost,
            command_counts,
            model_usage,
            date_range: (min_date, max_date),
        })
    }

    /// Export history data in the specified format
    pub async fn export(
        &self,
        path: &PathBuf,
        format: ExportFormat,
        filter: Option<&HistorySearch>,
    ) -> Result<()> {
        let entries = if let Some(criteria) = filter {
            self.search(criteria).await?
        } else {
            self.entries.clone()
        };

        match format {
            ExportFormat::Json => self.export_json(path, &entries).await,
            ExportFormat::Csv => self.export_csv(path, &entries).await,
            ExportFormat::Html => self.export_html(path, &entries).await,
        }
    }

    /// Get recent command history for a session
    pub async fn get_recent_commands(
        &self,
        session_id: SessionId,
        limit: usize,
    ) -> Result<Vec<HistoryEntry>> {
        let criteria = HistorySearch {
            session_id: Some(session_id),
            limit,
            sort_by: SortField::Timestamp,
            sort_desc: true,
            ..Default::default()
        };

        self.search(&criteria).await
    }

    /// Get command usage statistics
    pub async fn get_command_stats(&self) -> Result<Vec<(String, usize, f64, f64)>> {
        let mut command_stats: HashMap<String, (usize, f64, u64)> = HashMap::new();

        for entry in &self.entries {
            let (count, cost, duration) = command_stats
                .entry(entry.command_name.clone())
                .or_insert((0, 0.0, 0));
            *count += 1;
            *cost += entry.cost_usd.unwrap_or(0.0);
            *duration += entry.duration_ms;
        }

        let mut results: Vec<_> = command_stats
            .into_iter()
            .map(|(cmd, (count, cost, duration))| {
                let avg_duration = duration as f64 / count as f64;
                (cmd, count, cost, avg_duration)
            })
            .collect();

        results.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by usage count
        Ok(results)
    }

    /// Clear all history data
    pub async fn clear_all(&mut self) -> Result<()> {
        self.entries.clear();
        self.index.clear();
        self.save_entries().await?;
        Ok(())
    }

    /// Get all entries
    pub async fn get_all_entries(&self) -> Result<Vec<HistoryEntry>> {
        Ok(self.entries.clone())
    }

    /// Get entries for a specific session
    pub async fn get_session_entries(&self, session_id: &SessionId) -> Result<Vec<HistoryEntry>> {
        let entries: Vec<_> = self
            .entries
            .iter()
            .filter(|e| &e.session_id == session_id)
            .cloned()
            .collect();
        Ok(entries)
    }

    /// Clear history data for a specific session
    pub async fn clear_session(&mut self, session_id: SessionId) -> Result<()> {
        let mut removed_ids = Vec::new();

        self.entries.retain(|e| {
            if e.session_id == session_id {
                removed_ids.push(e.id.clone());
                false
            } else {
                true
            }
        });

        // Remove from index
        for id in removed_ids {
            self.index.remove(&id);
        }

        // Rebuild index
        self.rebuild_index();

        self.save_entries().await?;
        Ok(())
    }

    /// Prune old history entries beyond a certain age
    pub async fn prune_old_entries(&mut self, max_age_days: u32) -> Result<usize> {
        let cutoff_date = Utc::now() - chrono::Duration::days(max_age_days as i64);
        let initial_count = self.entries.len();

        let mut removed_ids = Vec::new();

        self.entries.retain(|e| {
            if e.timestamp < cutoff_date {
                removed_ids.push(e.id.clone());
                false
            } else {
                true
            }
        });

        // Remove from index
        for id in removed_ids {
            self.index.remove(&id);
        }

        // Rebuild index
        self.rebuild_index();

        self.save_entries().await?;

        Ok(initial_count - self.entries.len())
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
            InteractiveError::History(format!("Failed to parse history data: {}", e))
        })?;

        // Rebuild index
        self.rebuild_index();

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

    fn rebuild_index(&mut self) {
        self.index.clear();
        for (index, entry) in self.entries.iter().enumerate() {
            self.index.insert(entry.id.clone(), index);
        }
    }

    fn matches_criteria(&self, entry: &HistoryEntry, criteria: &HistorySearch) -> bool {
        if let Some(session_id) = criteria.session_id {
            if entry.session_id != session_id {
                return false;
            }
        }

        if let Some(pattern) = &criteria.command_pattern {
            if !entry.command_name.contains(pattern) {
                return false;
            }
        }

        if let Some(pattern) = &criteria.output_pattern {
            if !entry.output.contains(pattern) {
                return false;
            }
        }

        if let Some(pattern) = &criteria.error_pattern {
            if let Some(error) = &entry.error {
                if !error.contains(pattern) {
                    return false;
                }
            } else {
                return false;
            }
        }

        if criteria.success_only && !entry.success {
            return false;
        }

        if criteria.failures_only && entry.success {
            return false;
        }

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

        if let Some(min_cost) = criteria.min_cost {
            if let Some(cost) = entry.cost_usd {
                if cost < min_cost {
                    return false;
                }
            } else {
                return false;
            }
        }

        if let Some(max_cost) = criteria.max_cost {
            if let Some(cost) = entry.cost_usd {
                if cost > max_cost {
                    return false;
                }
            } else {
                return false;
            }
        }

        if let Some(model) = &criteria.model {
            if let Some(entry_model) = &entry.model {
                if entry_model != model {
                    return false;
                }
            } else {
                return false;
            }
        }

        if !criteria.tags.is_empty() {
            let has_any_tag = criteria.tags.iter().any(|tag| entry.tags.contains(tag));
            if !has_any_tag {
                return false;
            }
        }

        true
    }

    fn sort_entries(&self, entries: &mut [HistoryEntry], sort_by: &SortField, desc: bool) {
        entries.sort_by(|a, b| {
            let ordering = match sort_by {
                SortField::Timestamp => a.timestamp.cmp(&b.timestamp),
                SortField::Duration => a.duration_ms.cmp(&b.duration_ms),
                SortField::Cost => {
                    let a_cost = a.cost_usd.unwrap_or(0.0);
                    let b_cost = b.cost_usd.unwrap_or(0.0);
                    a_cost
                        .partial_cmp(&b_cost)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
                SortField::Command => a.command_name.cmp(&b.command_name),
            };

            if desc {
                ordering.reverse()
            } else {
                ordering
            }
        });
    }

    async fn export_json(&self, path: &PathBuf, entries: &[HistoryEntry]) -> Result<()> {
        let content = serde_json::to_string_pretty(entries)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }

    async fn export_csv(&self, path: &PathBuf, entries: &[HistoryEntry]) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        // Write CSV header
        writeln!(file, "id,session_id,command_name,args,output,error,cost_usd,input_tokens,output_tokens,timestamp,duration_ms,success,model,tags")?;

        // Write entries
        for entry in entries {
            let args = entry.args.join(";");
            let output = entry.output.replace('\n', "\\n").replace('"', "\"\"");
            let error = entry
                .error
                .as_deref()
                .unwrap_or("")
                .replace('\n', "\\n")
                .replace('"', "\"\"");
            let tags = entry.tags.join(";");

            writeln!(
                file,
                "{},{},{},\"{}\",\"{}\",\"{}\",{},{},{},{},{},{},{},\"{}\"",
                entry.id,
                entry.session_id,
                entry.command_name,
                args,
                output,
                error,
                entry.cost_usd.map(|c| c.to_string()).unwrap_or_default(),
                entry
                    .input_tokens
                    .map(|t| t.to_string())
                    .unwrap_or_default(),
                entry
                    .output_tokens
                    .map(|t| t.to_string())
                    .unwrap_or_default(),
                entry.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
                entry.duration_ms,
                entry.success,
                entry.model.as_deref().unwrap_or(""),
                tags
            )?;
        }

        Ok(())
    }

    async fn export_html(&self, path: &PathBuf, entries: &[HistoryEntry]) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        writeln!(file, "<!DOCTYPE html>")?;
        writeln!(
            file,
            "<html><head><title>Claude AI Interactive - Command History</title>"
        )?;
        writeln!(file, "<style>")?;
        writeln!(
            file,
            "  body {{ font-family: Arial, sans-serif; margin: 20px; }}"
        )?;
        writeln!(
            file,
            "  table {{ border-collapse: collapse; width: 100%; }}"
        )?;
        writeln!(
            file,
            "  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}"
        )?;
        writeln!(file, "  th {{ background-color: #f2f2f2; }}")?;
        writeln!(file, "  .success {{ color: green; }}")?;
        writeln!(file, "  .error {{ color: red; }}")?;
        writeln!(file, "  .output {{ max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}")?;
        writeln!(file, "</style></head><body>")?;

        writeln!(file, "<h1>Command History Report</h1>")?;
        writeln!(
            file,
            "<p>Generated: {}</p>",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )?;
        writeln!(file, "<p>Total Entries: {}</p>", entries.len())?;

        writeln!(file, "<table>")?;
        writeln!(file, "<tr><th>Timestamp</th><th>Session</th><th>Command</th><th>Duration</th><th>Cost</th><th>Status</th><th>Output</th></tr>")?;

        for entry in entries {
            let status_class = if entry.success { "success" } else { "error" };
            let status_text = if entry.success { "✓" } else { "✗" };
            let cost_text = entry
                .cost_usd
                .map(|c| format!("${:.4}", c))
                .unwrap_or_default();
            let output_preview = if entry.output.len() > 100 {
                format!("{}...", &entry.output[..100])
            } else {
                entry.output.clone()
            };

            writeln!(
                file,
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}ms</td><td>{}</td><td class=\"{}\">{}</td><td class=\"output\">{}</td></tr>",
                entry.timestamp.format("%Y-%m-%d %H:%M:%S"),
                entry.session_id,
                entry.command_name,
                entry.duration_ms,
                cost_text,
                status_class,
                status_text,
                html_escape(&output_preview)
            )?;
        }

        writeln!(file, "</table></body></html>")?;

        Ok(())
    }
}

// Helper function for HTML escaping
fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_history_entry_creation() {
        let session_id = Uuid::new_v4();
        let entry = HistoryEntry::new(
            session_id,
            "test_command".to_string(),
            vec!["--arg1".to_string(), "value1".to_string()],
            "Test output".to_string(),
            true,
            1500,
        );

        assert_eq!(entry.session_id, session_id);
        assert_eq!(entry.command_name, "test_command");
        assert_eq!(entry.args, vec!["--arg1", "value1"]);
        assert_eq!(entry.output, "Test output");
        assert!(entry.success);
        assert_eq!(entry.duration_ms, 1500);
        assert!(entry.error.is_none());
        assert!(entry.cost_usd.is_none());
        assert!(!entry.id.is_empty());
    }

    #[test]
    fn test_history_entry_builders() {
        let session_id = Uuid::new_v4();
        let entry = HistoryEntry::new(
            session_id,
            "command".to_string(),
            vec![],
            "output".to_string(),
            true,
            1000,
        )
        .with_cost(0.025, 100, 200, "claude-3-opus".to_string())
        .with_tags(vec!["test".to_string(), "example".to_string()]);

        assert_eq!(entry.cost_usd, Some(0.025));
        assert_eq!(entry.input_tokens, Some(100));
        assert_eq!(entry.output_tokens, Some(200));
        assert_eq!(entry.model, Some("claude-3-opus".to_string()));
        assert_eq!(entry.tags, vec!["test", "example"]);

        // Test error builder
        let error_entry = HistoryEntry::new(
            session_id,
            "failed_command".to_string(),
            vec![],
            "error output".to_string(),
            true,
            500,
        )
        .with_error("Command failed".to_string());

        assert_eq!(error_entry.error, Some("Command failed".to_string()));
        assert!(!error_entry.success); // Should be set to false
    }

    #[test]
    fn test_history_search_default() {
        let search = HistorySearch::default();

        assert!(search.session_id.is_none());
        assert!(search.command_pattern.is_none());
        assert!(!search.success_only);
        assert!(!search.failures_only);
        assert_eq!(search.limit, 100);
        assert_eq!(search.offset, 0);
        assert!(matches!(search.sort_by, SortField::Timestamp));
        assert!(search.sort_desc);
    }

    #[tokio::test]
    async fn test_history_store_new() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");

        let store = HistoryStore::new(storage_path.clone()).unwrap();
        assert_eq!(store.entries.len(), 0);
        assert_eq!(store.storage_path, storage_path);
        assert!(store.index.is_empty());
    }

    #[tokio::test]
    async fn test_store_and_retrieve_entry() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();
        let entry = HistoryEntry::new(
            session_id,
            "test_cmd".to_string(),
            vec!["arg1".to_string()],
            "Test output".to_string(),
            true,
            1000,
        );

        let entry_id = entry.id.clone();
        store.store_entry(entry).await.unwrap();

        assert_eq!(store.entries.len(), 1);
        assert!(store.index.contains_key(&entry_id));

        let retrieved = store.get_entry(&entry_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().command_name, "test_cmd");
    }

    #[tokio::test]
    async fn test_update_entry() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();
        let mut entry = HistoryEntry::new(
            session_id,
            "original_cmd".to_string(),
            vec![],
            "Original output".to_string(),
            true,
            1000,
        );

        let entry_id = entry.id.clone();
        store.store_entry(entry.clone()).await.unwrap();

        // Update the entry
        entry.command_name = "updated_cmd".to_string();
        entry.output = "Updated output".to_string();

        store.update_entry(&entry_id, entry).await.unwrap();

        let retrieved = store.get_entry(&entry_id).unwrap();
        assert_eq!(retrieved.command_name, "updated_cmd");
        assert_eq!(retrieved.output, "Updated output");
    }

    #[tokio::test]
    async fn test_search_basic() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add various entries
        for i in 0..10 {
            let entry = HistoryEntry::new(
                session_id,
                format!("command_{}", i),
                vec![],
                format!("Output {}", i),
                i % 2 == 0, // Even indices are successful
                1000 + i as u64 * 100,
            );
            store.store_entry(entry).await.unwrap();
        }

        // Search all entries
        let all_results = store.search(&HistorySearch::default()).await.unwrap();
        assert_eq!(all_results.len(), 10);

        // Search success only
        let success_search = HistorySearch {
            success_only: true,
            ..Default::default()
        };
        let success_results = store.search(&success_search).await.unwrap();
        assert_eq!(success_results.len(), 5);

        // Search failures only
        let failure_search = HistorySearch {
            failures_only: true,
            ..Default::default()
        };
        let failure_results = store.search(&failure_search).await.unwrap();
        assert_eq!(failure_results.len(), 5);
    }

    #[tokio::test]
    async fn test_search_with_patterns() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add entries with different patterns
        store
            .store_entry(HistoryEntry::new(
                session_id,
                "analyze".to_string(),
                vec![],
                "Analysis complete".to_string(),
                true,
                1000,
            ))
            .await
            .unwrap();

        store
            .store_entry(HistoryEntry::new(
                session_id,
                "generate".to_string(),
                vec![],
                "Generated code".to_string(),
                true,
                2000,
            ))
            .await
            .unwrap();

        store
            .store_entry(HistoryEntry::new(
                session_id,
                "analyze_detailed".to_string(),
                vec![],
                "Detailed analysis results".to_string(),
                true,
                3000,
            ))
            .await
            .unwrap();

        // Search by command pattern
        let search = HistorySearch {
            command_pattern: Some("analyze".to_string()),
            ..Default::default()
        };
        let results = store.search(&search).await.unwrap();
        assert_eq!(results.len(), 2);

        // Search by output pattern
        let search = HistorySearch {
            output_pattern: Some("analysis".to_string()),
            ..Default::default()
        };
        let results = store.search(&search).await.unwrap();
        assert_eq!(results.len(), 1); // Only "Detailed analysis results" contains lowercase "analysis"
    }

    #[tokio::test]
    async fn test_search_with_cost_filters() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add entries with different costs
        store
            .store_entry(
                HistoryEntry::new(
                    session_id,
                    "cheap_cmd".to_string(),
                    vec![],
                    "output".to_string(),
                    true,
                    1000,
                )
                .with_cost(0.01, 50, 100, "claude-3-haiku".to_string()),
            )
            .await
            .unwrap();

        store
            .store_entry(
                HistoryEntry::new(
                    session_id,
                    "medium_cmd".to_string(),
                    vec![],
                    "output".to_string(),
                    true,
                    2000,
                )
                .with_cost(0.05, 100, 200, "claude-3-sonnet".to_string()),
            )
            .await
            .unwrap();

        store
            .store_entry(
                HistoryEntry::new(
                    session_id,
                    "expensive_cmd".to_string(),
                    vec![],
                    "output".to_string(),
                    true,
                    3000,
                )
                .with_cost(0.10, 200, 400, "claude-3-opus".to_string()),
            )
            .await
            .unwrap();

        // Search by minimum cost
        let search = HistorySearch {
            min_cost: Some(0.04),
            ..Default::default()
        };
        let results = store.search(&search).await.unwrap();
        assert_eq!(results.len(), 2);

        // Search by cost range
        let search = HistorySearch {
            min_cost: Some(0.02),
            max_cost: Some(0.08),
            ..Default::default()
        };
        let results = store.search(&search).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].command_name, "medium_cmd");
    }

    #[tokio::test]
    async fn test_search_with_pagination() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add 20 entries
        for i in 0..20 {
            store
                .store_entry(HistoryEntry::new(
                    session_id,
                    format!("cmd_{}", i),
                    vec![],
                    format!("output_{}", i),
                    true,
                    1000,
                ))
                .await
                .unwrap();
        }

        // Get first page
        let search = HistorySearch {
            limit: 5,
            offset: 0,
            ..Default::default()
        };
        let page1 = store.search(&search).await.unwrap();
        assert_eq!(page1.len(), 5);

        // Get second page
        let search = HistorySearch {
            limit: 5,
            offset: 5,
            ..Default::default()
        };
        let page2 = store.search(&search).await.unwrap();
        assert_eq!(page2.len(), 5);

        // Verify no overlap
        let page1_ids: Vec<_> = page1.iter().map(|e| &e.id).collect();
        let page2_ids: Vec<_> = page2.iter().map(|e| &e.id).collect();
        assert!(page1_ids.iter().all(|id| !page2_ids.contains(id)));
    }

    #[tokio::test]
    async fn test_get_stats() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add mixed entries
        for i in 0..10 {
            let mut entry = HistoryEntry::new(
                session_id,
                format!("cmd_{}", i % 3),
                vec![],
                format!("output_{}", i),
                i % 4 != 0, // 75% success rate
                1000 + i as u64 * 100,
            );

            if i % 2 == 0 {
                entry =
                    entry.with_cost(0.01 * (i + 1) as f64, 50, 100, "claude-3-opus".to_string());
            }

            store.store_entry(entry).await.unwrap();
        }

        let stats = store.get_stats(None).await.unwrap();

        assert_eq!(stats.total_entries, 10);
        assert_eq!(stats.successful_commands, 7);
        assert_eq!(stats.failed_commands, 3);
        assert!((stats.success_rate - 70.0).abs() < 0.1);
        assert!(stats.total_cost > 0.0);
        assert_eq!(stats.total_duration_ms, 14500); // Sum of 1000, 1100, ..., 1900
        assert!((stats.average_duration_ms - 1450.0).abs() < 0.1);
        assert_eq!(stats.command_counts.len(), 3); // cmd_0, cmd_1, cmd_2
        assert_eq!(stats.model_usage.get("claude-3-opus"), Some(&5));
    }

    #[tokio::test]
    async fn test_export_formats() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add test entries
        for i in 0..3 {
            store
                .store_entry(
                    HistoryEntry::new(
                        session_id,
                        format!("export_test_{}", i),
                        vec![format!("arg_{}", i)],
                        format!("Output for test {}", i),
                        true,
                        1000 + i as u64 * 500,
                    )
                    .with_cost(
                        0.01 * (i + 1) as f64,
                        50,
                        100,
                        "claude-3-opus".to_string(),
                    ),
                )
                .await
                .unwrap();
        }

        // Test JSON export
        let json_path = temp_dir.path().join("export.json");
        store
            .export(&json_path, ExportFormat::Json, None)
            .await
            .unwrap();
        assert!(json_path.exists());
        let json_content = tokio::fs::read_to_string(&json_path).await.unwrap();
        assert!(json_content.contains("export_test_"));

        // Test CSV export
        let csv_path = temp_dir.path().join("export.csv");
        store
            .export(&csv_path, ExportFormat::Csv, None)
            .await
            .unwrap();
        assert!(csv_path.exists());
        let csv_content = tokio::fs::read_to_string(&csv_path).await.unwrap();
        assert!(csv_content.contains("command_name"));
        assert!(csv_content.contains("export_test_"));

        // Test HTML export
        let html_path = temp_dir.path().join("export.html");
        store
            .export(&html_path, ExportFormat::Html, None)
            .await
            .unwrap();
        assert!(html_path.exists());
        let html_content = tokio::fs::read_to_string(&html_path).await.unwrap();
        assert!(html_content.contains("<html>"));
        assert!(html_content.contains("export_test_"));
    }

    #[tokio::test]
    async fn test_get_recent_commands() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add entries with different timestamps
        for i in 0..10 {
            let mut entry = HistoryEntry::new(
                session_id,
                format!("recent_cmd_{}", i),
                vec![],
                format!("output_{}", i),
                true,
                1000,
            );
            // Simulate different timestamps
            entry.timestamp = Utc::now() - chrono::Duration::minutes(10 - i as i64);
            store.store_entry(entry).await.unwrap();
        }

        let recent = store.get_recent_commands(session_id, 5).await.unwrap();
        assert_eq!(recent.len(), 5);
        // Should be sorted by timestamp descending (most recent first)
        assert!(recent[0].command_name.contains("9"));
    }

    #[tokio::test]
    async fn test_get_command_stats() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add entries with repeated commands
        for i in 0..12 {
            let cmd_name = format!("cmd_{}", i % 3);
            store
                .store_entry(
                    HistoryEntry::new(
                        session_id,
                        cmd_name,
                        vec![],
                        "output".to_string(),
                        true,
                        1000 + (i % 3) as u64 * 500,
                    )
                    .with_cost(
                        0.01 * ((i % 3) + 1) as f64,
                        50,
                        100,
                        "claude-3-opus".to_string(),
                    ),
                )
                .await
                .unwrap();
        }

        let stats = store.get_command_stats().await.unwrap();

        assert_eq!(stats.len(), 3);
        // Each command appears 4 times
        assert_eq!(stats[0].1, 4);
        assert_eq!(stats[1].1, 4);
        assert_eq!(stats[2].1, 4);

        // Verify costs and durations are calculated correctly
        for (cmd, count, cost, avg_duration) in stats {
            assert_eq!(count, 4);
            if cmd == "cmd_0" {
                assert!((cost - 0.04).abs() < 0.0001); // 4 * 0.01
                assert!((avg_duration - 1000.0).abs() < 0.1);
            } else if cmd == "cmd_1" {
                assert!((cost - 0.08).abs() < 0.0001); // 4 * 0.02
                assert!((avg_duration - 1500.0).abs() < 0.1);
            } else if cmd == "cmd_2" {
                assert!((cost - 0.12).abs() < 0.0001); // 4 * 0.03
                assert!((avg_duration - 2000.0).abs() < 0.1);
            }
        }
    }

    #[tokio::test]
    async fn test_clear_operations() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session1 = Uuid::new_v4();
        let session2 = Uuid::new_v4();

        // Add entries for two sessions
        for i in 0..5 {
            store
                .store_entry(HistoryEntry::new(
                    session1,
                    format!("s1_cmd_{}", i),
                    vec![],
                    "output".to_string(),
                    true,
                    1000,
                ))
                .await
                .unwrap();

            store
                .store_entry(HistoryEntry::new(
                    session2,
                    format!("s2_cmd_{}", i),
                    vec![],
                    "output".to_string(),
                    true,
                    1000,
                ))
                .await
                .unwrap();
        }

        assert_eq!(store.entries.len(), 10);

        // Clear session 1
        store.clear_session(session1).await.unwrap();
        assert_eq!(store.entries.len(), 5);

        // Verify only session2 entries remain
        let search = HistorySearch {
            session_id: Some(session2),
            ..Default::default()
        };
        let remaining = store.search(&search).await.unwrap();
        assert_eq!(remaining.len(), 5);

        // Clear all
        store.clear_all().await.unwrap();
        assert_eq!(store.entries.len(), 0);
        assert!(store.index.is_empty());
    }

    #[tokio::test]
    async fn test_prune_old_entries() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add entries with different ages
        for i in 0..10 {
            let mut entry = HistoryEntry::new(
                session_id,
                format!("cmd_{}", i),
                vec![],
                "output".to_string(),
                true,
                1000,
            );
            // Half are old (10+ days), half are recent
            if i < 5 {
                entry.timestamp = Utc::now() - chrono::Duration::days(15);
            } else {
                entry.timestamp = Utc::now() - chrono::Duration::days(2);
            }
            store.store_entry(entry).await.unwrap();
        }

        // Prune entries older than 10 days
        let pruned = store.prune_old_entries(10).await.unwrap();

        assert_eq!(pruned, 5);
        assert_eq!(store.entries.len(), 5);

        // Verify only recent entries remain
        for entry in &store.entries {
            assert!(entry.timestamp > Utc::now() - chrono::Duration::days(10));
        }
    }

    #[tokio::test]
    async fn test_persistence() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");

        let session_id = Uuid::new_v4();
        let entry_id;

        // Create store, add entries, then drop
        {
            let mut store = HistoryStore::new(storage_path.clone()).unwrap();
            let entry = HistoryEntry::new(
                session_id,
                "persistent_cmd".to_string(),
                vec!["--persistent".to_string()],
                "Persistent output".to_string(),
                true,
                1234,
            )
            .with_cost(0.0456, 123, 456, "claude-3-opus".to_string());

            entry_id = entry.id.clone();
            store.store_entry(entry).await.unwrap();
        }

        // Create new store instance and verify data persisted
        {
            let store = HistoryStore::new(storage_path).unwrap();
            assert_eq!(store.entries.len(), 1);

            let entry = store.get_entry(&entry_id).unwrap();
            assert_eq!(entry.command_name, "persistent_cmd");
            assert_eq!(entry.args, vec!["--persistent"]);
            assert_eq!(entry.output, "Persistent output");
            assert!(entry.success);
            assert_eq!(entry.duration_ms, 1234);
            assert_eq!(entry.cost_usd, Some(0.0456));
            assert_eq!(entry.input_tokens, Some(123));
            assert_eq!(entry.output_tokens, Some(456));
            assert_eq!(entry.model, Some("claude-3-opus".to_string()));
        }
    }

    #[tokio::test]
    async fn test_search_with_tags() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add entries with different tags
        store
            .store_entry(
                HistoryEntry::new(
                    session_id,
                    "cmd1".to_string(),
                    vec![],
                    "output".to_string(),
                    true,
                    1000,
                )
                .with_tags(vec!["test".to_string(), "important".to_string()]),
            )
            .await
            .unwrap();

        store
            .store_entry(
                HistoryEntry::new(
                    session_id,
                    "cmd2".to_string(),
                    vec![],
                    "output".to_string(),
                    true,
                    1000,
                )
                .with_tags(vec!["test".to_string()]),
            )
            .await
            .unwrap();

        store
            .store_entry(
                HistoryEntry::new(
                    session_id,
                    "cmd3".to_string(),
                    vec![],
                    "output".to_string(),
                    true,
                    1000,
                )
                .with_tags(vec!["debug".to_string()]),
            )
            .await
            .unwrap();

        // Search by tag
        let search = HistorySearch {
            tags: vec!["test".to_string()],
            ..Default::default()
        };
        let results = store.search(&search).await.unwrap();
        assert_eq!(results.len(), 2);

        // Search by multiple tags (OR operation)
        let search = HistorySearch {
            tags: vec!["important".to_string(), "debug".to_string()],
            ..Default::default()
        };
        let results = store.search(&search).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_sorting() {
        let temp_dir = tempdir().unwrap();
        let storage_path = temp_dir.path().join("history_data.json");
        let mut store = HistoryStore::new(storage_path).unwrap();

        let session_id = Uuid::new_v4();

        // Add entries with different attributes
        store
            .store_entry(
                HistoryEntry::new(
                    session_id,
                    "b_cmd".to_string(),
                    vec![],
                    "output".to_string(),
                    true,
                    2000,
                )
                .with_cost(0.05, 100, 200, "model".to_string()),
            )
            .await
            .unwrap();

        store
            .store_entry(
                HistoryEntry::new(
                    session_id,
                    "a_cmd".to_string(),
                    vec![],
                    "output".to_string(),
                    true,
                    1000,
                )
                .with_cost(0.10, 100, 200, "model".to_string()),
            )
            .await
            .unwrap();

        store
            .store_entry(
                HistoryEntry::new(
                    session_id,
                    "c_cmd".to_string(),
                    vec![],
                    "output".to_string(),
                    true,
                    3000,
                )
                .with_cost(0.01, 100, 200, "model".to_string()),
            )
            .await
            .unwrap();

        // Sort by command name ascending
        let search = HistorySearch {
            sort_by: SortField::Command,
            sort_desc: false,
            ..Default::default()
        };
        let results = store.search(&search).await.unwrap();
        assert_eq!(results[0].command_name, "a_cmd");
        assert_eq!(results[1].command_name, "b_cmd");
        assert_eq!(results[2].command_name, "c_cmd");

        // Sort by duration descending
        let search = HistorySearch {
            sort_by: SortField::Duration,
            sort_desc: true,
            ..Default::default()
        };
        let results = store.search(&search).await.unwrap();
        assert_eq!(results[0].duration_ms, 3000);
        assert_eq!(results[1].duration_ms, 2000);
        assert_eq!(results[2].duration_ms, 1000);

        // Sort by cost ascending
        let search = HistorySearch {
            sort_by: SortField::Cost,
            sort_desc: false,
            ..Default::default()
        };
        let results = store.search(&search).await.unwrap();
        assert_eq!(results[0].cost_usd, Some(0.01));
        assert_eq!(results[1].cost_usd, Some(0.05));
        assert_eq!(results[2].cost_usd, Some(0.10));
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("Hello & World"), "Hello &amp; World");
        assert_eq!(
            html_escape("<script>alert('XSS')</script>"),
            "&lt;script&gt;alert(&#39;XSS&#39;)&lt;/script&gt;"
        );
        assert_eq!(
            html_escape("\"Quotes\" and 'apostrophes'"),
            "&quot;Quotes&quot; and &#39;apostrophes&#39;"
        );
    }
}
