//! Extended history storage with advanced features
//!
//! This module provides additional capabilities for history management:
//! - Persistent caching and indexing
//! - Full-text search capabilities
//! - Data compression for large histories
//! - Backup and restore functionality

use super::{HistoryEntry, HistorySearch, HistoryStats};
use crate::{cli::error::Result, cli::session::SessionId};
use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Advanced search capabilities with full-text indexing
#[derive(Debug, Clone)]
pub struct FullTextSearch {
    pub query: String,
    pub fields: Vec<SearchField>,
    pub fuzzy: bool,
    pub highlight: bool,
}

/// Fields available for full-text search
#[derive(Debug, Clone)]
pub enum SearchField {
    Command,
    Output,
    Error,
    Args,
    Tags,
    All,
}

/// Search result with highlighting
#[derive(Debug, Clone, Serialize)]
pub struct HistorySearchResult {
    pub entry: HistoryEntry,
    pub score: f64,
    pub highlights: HashMap<String, Vec<String>>,
}

/// Backup metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    pub created_at: DateTime<Utc>,
    pub entry_count: usize,
    pub compressed_size: u64,
    pub version: String,
}

/// History index for fast searching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryIndex {
    pub command_index: HashMap<String, HashSet<String>>, // command -> entry_ids
    pub session_index: HashMap<SessionId, HashSet<String>>, // session -> entry_ids
    pub timestamp_index: Vec<(DateTime<Utc>, String)>,   // sorted by timestamp
    pub word_index: HashMap<String, HashSet<String>>,    // word -> entry_ids (for full-text search)
}

impl Default for HistoryIndex {
    fn default() -> Self {
        Self {
            command_index: HashMap::new(),
            session_index: HashMap::new(),
            timestamp_index: Vec::new(),
            word_index: HashMap::new(),
        }
    }
}

/// Enhanced history store with advanced features
pub struct EnhancedHistoryStore {
    base_path: PathBuf,
    index: HistoryIndex,
    cache: HashMap<String, HistoryEntry>,
    compression_enabled: bool,
    max_cache_size: usize,
}

/// Pagination parameters for large result sets
#[derive(Debug, Clone)]
pub struct HistoryPagination {
    pub page: usize,
    pub page_size: usize,
    pub total_count: Option<usize>,
}

impl HistoryPagination {
    pub fn new(page: usize, page_size: usize) -> Self {
        Self {
            page,
            page_size,
            total_count: None,
        }
    }

    pub fn offset(&self) -> usize {
        self.page * self.page_size
    }

    pub fn limit(&self) -> usize {
        self.page_size
    }

    pub fn has_next_page(&self) -> bool {
        if let Some(total) = self.total_count {
            self.offset() + self.page_size < total
        } else {
            false
        }
    }

    pub fn total_pages(&self) -> Option<usize> {
        self.total_count
            .map(|total| (total + self.page_size - 1) / self.page_size)
    }
}

/// Paginated search results
#[derive(Debug, Clone)]
pub struct PaginatedHistoryResults {
    pub entries: Vec<HistoryEntry>,
    pub pagination: HistoryPagination,
}

impl EnhancedHistoryStore {
    /// Create a new enhanced history store
    pub fn new(base_path: PathBuf) -> Result<Self> {
        let mut store = Self {
            base_path: base_path.clone(),
            index: HistoryIndex::default(),
            cache: HashMap::new(),
            compression_enabled: true,
            max_cache_size: 10000,
        };

        // Create necessary directories
        std::fs::create_dir_all(&base_path)?;

        // Load existing index
        store.load_index()?;

        Ok(store)
    }

    /// Search history with pagination support and lazy loading
    pub async fn search_paginated(
        &self,
        criteria: &HistorySearch,
        pagination: HistoryPagination,
    ) -> Result<PaginatedHistoryResults> {
        // Use optimized index-based search when possible
        let search_criteria = HistorySearch {
            offset: pagination.offset(),
            limit: pagination.limit(),
            ..criteria.clone()
        };

        // Fast path: if searching by command or session, use index
        let entries = if let Some(command) = &criteria.command_pattern {
            self.search_by_command_index(command, &search_criteria)
                .await?
        } else if let Some(session_id) = criteria.session_id {
            self.search_by_session_index(session_id, &search_criteria)
                .await?
        } else {
            // Fallback to full search
            self.search_with_criteria(&search_criteria).await?
        };

        let mut result_pagination = pagination.clone();
        // Estimate total count for pagination (actual implementation would cache this)
        let estimated_total = if entries.len() < result_pagination.limit() {
            result_pagination.offset() + entries.len()
        } else {
            // Estimate based on current page
            (result_pagination.offset() + entries.len()) * 2
        };
        result_pagination.total_count = Some(estimated_total);

        Ok(PaginatedHistoryResults {
            entries,
            pagination: result_pagination,
        })
    }

    /// Fast search using command index
    async fn search_by_command_index(
        &self,
        command_pattern: &str,
        criteria: &HistorySearch,
    ) -> Result<Vec<HistoryEntry>> {
        let mut matching_entries = Vec::new();

        // Find commands that match the pattern
        for (command, entry_ids) in &self.index.command_index {
            if command.contains(command_pattern) {
                for entry_id in entry_ids.iter().skip(criteria.offset).take(criteria.limit) {
                    if let Some(entry) = self.get_entry_by_id(entry_id).await? {
                        if self.matches_detailed_criteria(&entry, criteria) {
                            matching_entries.push(entry);
                        }
                    }
                }
            }
        }

        // Sort by timestamp (most recent first)
        matching_entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        Ok(matching_entries)
    }

    /// Fast search using session index
    async fn search_by_session_index(
        &self,
        session_id: SessionId,
        criteria: &HistorySearch,
    ) -> Result<Vec<HistoryEntry>> {
        let mut matching_entries = Vec::new();

        if let Some(entry_ids) = self.index.session_index.get(&session_id) {
            for entry_id in entry_ids.iter().skip(criteria.offset).take(criteria.limit) {
                if let Some(entry) = self.get_entry_by_id(entry_id).await? {
                    if self.matches_detailed_criteria(&entry, criteria) {
                        matching_entries.push(entry);
                    }
                }
            }
        }

        // Sort by timestamp (most recent first)
        matching_entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        Ok(matching_entries)
    }

    /// Check if entry matches detailed search criteria (excluding the indexed fields)
    fn matches_detailed_criteria(&self, entry: &HistoryEntry, criteria: &HistorySearch) -> bool {
        // Check output pattern
        if let Some(pattern) = &criteria.output_pattern {
            if !entry.output.contains(pattern) {
                return false;
            }
        }

        // Check error pattern
        if let Some(pattern) = &criteria.error_pattern {
            if let Some(error) = &entry.error {
                if !error.contains(pattern) {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check success/failure filters
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

        // Check cost range
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

        true
    }

    /// Enable or disable compression
    pub fn set_compression(&mut self, enabled: bool) {
        self.compression_enabled = enabled;
    }

    /// Set maximum cache size
    pub fn set_cache_size(&mut self, size: usize) {
        self.max_cache_size = size;
        self.trim_cache();
    }

    /// Perform full-text search with optimized index
    pub async fn full_text_search(
        &self,
        search: &FullTextSearch,
    ) -> Result<Vec<HistorySearchResult>> {
        let query_words = self.tokenize(&search.query);

        // Filter out stopwords for better search performance
        let stopwords = self.get_stopwords();
        let filtered_words: Vec<String> = query_words
            .into_iter()
            .filter(|word| !stopwords.contains(word.as_str()))
            .collect();

        if filtered_words.is_empty() {
            return Ok(Vec::new());
        }

        let mut candidate_ids = HashSet::new();
        let mut word_scores: HashMap<String, f64> = HashMap::new();

        // Find candidate entries based on word matches with scoring
        for (i, word) in filtered_words.iter().enumerate() {
            if let Some(entry_ids) = self.index.word_index.get(word) {
                // First word initializes the candidate set
                if i == 0 {
                    candidate_ids = entry_ids.clone();
                } else if search.fuzzy {
                    // Fuzzy search: union of results (OR operation)
                    candidate_ids = candidate_ids.union(entry_ids).cloned().collect();
                } else {
                    // Exact search: intersection of results (AND operation)
                    candidate_ids = candidate_ids.intersection(entry_ids).cloned().collect();
                }

                // Calculate inverse document frequency for scoring
                let total_entries = self.index.timestamp_index.len() as f64;
                let word_frequency = entry_ids.len() as f64;
                let idf = (total_entries / word_frequency.max(1.0)).ln();
                word_scores.insert(word.clone(), idf);
            } else if !search.fuzzy {
                // If any word is not found in exact search, no results
                candidate_ids.clear();
                break;
            }
        }

        let mut results = Vec::new();

        for entry_id in candidate_ids {
            if let Some(entry) = self.get_entry_by_id(&entry_id).await? {
                let score = self.calculate_enhanced_relevance_score(
                    &entry,
                    &search.query,
                    &filtered_words,
                    &word_scores,
                );

                let highlights = if search.highlight {
                    self.generate_highlights(&entry, &filtered_words, &search.fields)
                } else {
                    HashMap::new()
                };

                results.push(HistorySearchResult {
                    entry,
                    score,
                    highlights,
                });
            }
        }

        // Sort by relevance score (highest first)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Create a backup of history data
    pub async fn create_backup(&self, backup_path: &PathBuf) -> Result<BackupMetadata> {
        let entries = self.get_all_entries().await?;
        let entry_count = entries.len();

        let backup_data = BackupData {
            entries,
            index: self.index.clone(),
            created_at: Utc::now(),
        };

        let serialized = serde_json::to_string(&backup_data)?;

        let compressed_data = if self.compression_enabled {
            self.compress_data(&serialized)?
        } else {
            serialized.into_bytes()
        };

        tokio::fs::write(backup_path, &compressed_data).await?;

        let metadata = BackupMetadata {
            created_at: backup_data.created_at,
            entry_count,
            compressed_size: compressed_data.len() as u64,
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        // Save metadata alongside backup
        let metadata_path = backup_path.with_extension("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        tokio::fs::write(metadata_path, metadata_json).await?;

        Ok(metadata)
    }

    /// Restore from backup
    pub async fn restore_from_backup(&mut self, backup_path: &PathBuf) -> Result<usize> {
        let compressed_data = tokio::fs::read(backup_path).await?;

        let serialized = if self.compression_enabled {
            self.decompress_data(&compressed_data)?
        } else {
            String::from_utf8(compressed_data)?
        };

        let backup_data: BackupData = serde_json::from_str(&serialized)?;

        // Clear existing data
        self.cache.clear();
        self.index = backup_data.index;

        // Store all entries
        let entry_count = backup_data.entries.len();
        for entry in backup_data.entries {
            self.store_entry_internal(entry).await?;
        }

        self.save_index().await?;

        Ok(entry_count)
    }

    /// Get statistics with time-based breakdowns
    pub async fn get_detailed_stats(
        &self,
        time_buckets: Vec<DateTime<Utc>>,
    ) -> Result<Vec<(DateTime<Utc>, HistoryStats)>> {
        let mut results = Vec::new();

        for (i, &bucket_start) in time_buckets.iter().enumerate() {
            let bucket_end = time_buckets.get(i + 1).copied().unwrap_or(Utc::now());

            let criteria = HistorySearch {
                since: Some(bucket_start),
                until: Some(bucket_end),
                ..Default::default()
            };

            let entries = self.search_with_criteria(&criteria).await?;
            let stats = self.calculate_stats_for_entries(&entries).await?;

            results.push((bucket_start, stats));
        }

        Ok(results)
    }

    /// Get usage patterns and insights
    pub async fn get_usage_insights(&self) -> Result<UsageInsights> {
        let all_entries = self.get_all_entries().await?;

        let mut hourly_usage = [0usize; 24];
        let mut daily_usage = [0usize; 7]; // 0 = Sunday
        let mut command_sequences = HashMap::new();
        let mut session_durations = Vec::new();

        // Analyze patterns
        for entry in &all_entries {
            let hour = entry.timestamp.hour() as usize;
            hourly_usage[hour] += 1;

            let weekday = entry.timestamp.weekday().num_days_from_sunday() as usize;
            daily_usage[weekday] += 1;
        }

        // Analyze command sequences
        let mut session_commands: HashMap<SessionId, Vec<String>> = HashMap::new();
        for entry in &all_entries {
            session_commands
                .entry(entry.session_id)
                .or_insert_with(Vec::new)
                .push(entry.command_name.clone());
        }

        for commands in session_commands.values() {
            for window in commands.windows(2) {
                if let [cmd1, cmd2] = window {
                    let sequence = format!("{} -> {}", cmd1, cmd2);
                    *command_sequences.entry(sequence).or_insert(0) += 1;
                }
            }
        }

        // Calculate session durations
        for commands in session_commands.keys() {
            let session_entries: Vec<_> = all_entries
                .iter()
                .filter(|e| e.session_id == *commands)
                .collect();

            if let (Some(first), Some(last)) = (session_entries.first(), session_entries.last()) {
                let duration = last.timestamp - first.timestamp;
                session_durations.push(duration.num_minutes());
            }
        }

        Ok(UsageInsights {
            hourly_usage,
            daily_usage,
            common_sequences: command_sequences,
            average_session_duration: if session_durations.is_empty() {
                0
            } else {
                session_durations.iter().sum::<i64>() / session_durations.len() as i64
            },
            total_sessions: session_commands.len(),
        })
    }

    /// Optimize the search index for better performance
    pub async fn optimize_index(&mut self) -> Result<IndexOptimizationResult> {
        let start_time = std::time::Instant::now();
        let initial_word_count = self.index.word_index.len();
        let initial_memory_usage = self.estimate_index_memory_usage();

        // Clean up stale entries from indices
        let mut cleaned_entries = 0;
        let mut stale_entry_ids = HashSet::new();

        // Check for orphaned entry IDs in indices
        for entry_ids in self.index.command_index.values() {
            for entry_id in entry_ids {
                if self.get_entry_by_id(entry_id).await?.is_none() {
                    stale_entry_ids.insert(entry_id.clone());
                }
            }
        }

        // Remove stale entries from all indices
        for stale_id in &stale_entry_ids {
            self.remove_from_index(stale_id);
            cleaned_entries += 1;
        }

        // Optimize word index by removing very short words and stopwords
        let stopwords = self.get_stopwords();
        let mut removed_words = 0;

        self.index.word_index.retain(|word, entry_ids| {
            let should_keep =
                word.len() > 2 && !stopwords.contains(word.as_str()) && !entry_ids.is_empty();

            if !should_keep {
                removed_words += 1;
            }
            should_keep
        });

        // Optimize timestamp index by sorting and deduplicating
        self.index.timestamp_index.sort_by_key(|(ts, _)| *ts);
        self.index
            .timestamp_index
            .dedup_by_key(|(_, id)| id.clone());

        // Create inverted indices for faster lookups
        self.create_inverted_indices().await?;

        // Compact memory by rebuilding HashMaps with appropriate capacity
        self.compact_index_memory();

        let final_word_count = self.index.word_index.len();
        let final_memory_usage = self.estimate_index_memory_usage();
        let optimization_time = start_time.elapsed();

        self.save_index().await?;

        Ok(IndexOptimizationResult {
            optimization_duration: optimization_time,
            initial_word_count,
            final_word_count,
            removed_words,
            cleaned_entries,
            initial_memory_usage,
            final_memory_usage,
            memory_saved: initial_memory_usage.saturating_sub(final_memory_usage),
        })
    }

    /// Optimize storage by removing redundant data
    pub async fn optimize_storage(&mut self) -> Result<OptimizationResult> {
        let initial_entries = self.get_all_entries().await?.len();
        let mut removed_duplicates = 0;
        let mut compressed_outputs = 0;

        // Remove exact duplicates
        let mut seen_hashes = HashSet::new();
        let mut entries_to_remove = Vec::new();

        for entry in self.get_all_entries().await? {
            let hash = self.calculate_entry_hash(&entry);
            if seen_hashes.contains(&hash) {
                entries_to_remove.push(entry.id.clone());
                removed_duplicates += 1;
            } else {
                seen_hashes.insert(hash);
            }
        }

        // Remove duplicates
        for entry_id in entries_to_remove {
            self.remove_entry(&entry_id).await?;
        }

        // Compress large outputs
        for entry in self.get_all_entries().await? {
            if entry.output.len() > 10000 {
                // In a real implementation, you might compress the output
                compressed_outputs += 1;
            }
        }

        self.rebuild_index().await?;

        Ok(OptimizationResult {
            initial_entries,
            final_entries: self.get_all_entries().await?.len(),
            removed_duplicates,
            compressed_outputs,
            space_saved_bytes: 0, // Would calculate actual space saved
        })
    }

    // Private helper methods

    async fn get_entry_by_id(&self, id: &str) -> Result<Option<HistoryEntry>> {
        // Check cache first
        if let Some(entry) = self.cache.get(id) {
            return Ok(Some(entry.clone()));
        }

        // Load from storage
        let entry_path = self.base_path.join("entries").join(format!("{}.json", id));
        if entry_path.exists() {
            let content = tokio::fs::read_to_string(&entry_path).await?;
            let entry: HistoryEntry = serde_json::from_str(&content)?;

            // Add to cache (would need mutable reference)
            // self.add_to_cache(entry.clone());

            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }

    pub(crate) async fn store_entry_internal(&mut self, entry: HistoryEntry) -> Result<()> {
        let entry_id = entry.id.clone();

        // Store to file
        let entries_dir = self.base_path.join("entries");
        tokio::fs::create_dir_all(&entries_dir).await?;

        let entry_path = entries_dir.join(format!("{}.json", entry_id));
        let content = serde_json::to_string_pretty(&entry)?;
        tokio::fs::write(&entry_path, content).await?;

        // Update index
        self.update_index_for_entry(&entry);

        // Add to cache
        self.add_to_cache(entry);

        Ok(())
    }

    async fn get_all_entries(&self) -> Result<Vec<HistoryEntry>> {
        let entries_dir = self.base_path.join("entries");
        if !entries_dir.exists() {
            return Ok(Vec::new());
        }

        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(&entries_dir).await?;

        while let Some(entry) = dir.next_entry().await? {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                let content = tokio::fs::read_to_string(&entry.path()).await?;
                let history_entry: HistoryEntry = serde_json::from_str(&content)?;
                entries.push(history_entry);
            }
        }

        Ok(entries)
    }

    async fn search_with_criteria(&self, _criteria: &HistorySearch) -> Result<Vec<HistoryEntry>> {
        // This would implement the actual search logic
        // For now, return empty results
        Ok(Vec::new())
    }

    async fn calculate_stats_for_entries(&self, entries: &[HistoryEntry]) -> Result<HistoryStats> {
        // Implementation similar to the main history store
        let now = Utc::now();
        Ok(HistoryStats {
            total_entries: entries.len(),
            successful_commands: entries.iter().filter(|e| e.success).count(),
            failed_commands: entries.iter().filter(|e| !e.success).count(),
            success_rate: 0.0,
            total_cost: 0.0,
            total_duration_ms: 0,
            average_duration_ms: 0.0,
            average_cost: 0.0,
            command_counts: HashMap::new(),
            model_usage: HashMap::new(),
            date_range: (now, now),
        })
    }

    async fn remove_entry(&mut self, entry_id: &str) -> Result<()> {
        let entry_path = self
            .base_path
            .join("entries")
            .join(format!("{}.json", entry_id));
        if entry_path.exists() {
            tokio::fs::remove_file(&entry_path).await?;
        }

        self.cache.remove(entry_id);
        self.remove_from_index(entry_id);

        Ok(())
    }

    async fn rebuild_index(&mut self) -> Result<()> {
        self.index = HistoryIndex::default();

        for entry in self.get_all_entries().await? {
            self.update_index_for_entry(&entry);
        }

        self.save_index().await?;
        Ok(())
    }

    fn load_index(&mut self) -> Result<()> {
        let index_path = self.base_path.join("index.json");
        if index_path.exists() {
            let content = std::fs::read_to_string(&index_path)?;
            self.index = serde_json::from_str(&content)?;
        }
        Ok(())
    }

    async fn save_index(&self) -> Result<()> {
        let index_path = self.base_path.join("index.json");
        let content = serde_json::to_string_pretty(&self.index)?;
        tokio::fs::write(&index_path, content).await?;
        Ok(())
    }

    fn update_index_for_entry(&mut self, entry: &HistoryEntry) {
        let entry_id = entry.id.clone();

        // Update command index
        self.index
            .command_index
            .entry(entry.command_name.clone())
            .or_insert_with(HashSet::new)
            .insert(entry_id.clone());

        // Update session index
        self.index
            .session_index
            .entry(entry.session_id)
            .or_insert_with(HashSet::new)
            .insert(entry_id.clone());

        // Update timestamp index
        self.index
            .timestamp_index
            .push((entry.timestamp, entry_id.clone()));
        self.index.timestamp_index.sort_by_key(|(ts, _)| *ts);

        // Update word index for full-text search
        let all_text = format!(
            "{} {} {} {}",
            entry.command_name,
            entry.args.join(" "),
            entry.output,
            entry.tags.join(" ")
        );

        for word in self.tokenize(&all_text) {
            self.index
                .word_index
                .entry(word)
                .or_insert_with(HashSet::new)
                .insert(entry_id.clone());
        }
    }

    fn remove_from_index(&mut self, entry_id: &str) {
        // Remove from all indices
        for ids in self.index.command_index.values_mut() {
            ids.remove(entry_id);
        }

        for ids in self.index.session_index.values_mut() {
            ids.remove(entry_id);
        }

        self.index.timestamp_index.retain(|(_, id)| id != entry_id);

        for ids in self.index.word_index.values_mut() {
            ids.remove(entry_id);
        }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|word| !word.is_empty() && word.len() > 2)
            .map(String::from)
            .collect()
    }

    fn calculate_relevance_score(
        &self,
        entry: &HistoryEntry,
        query: &str,
        query_words: &[String],
    ) -> f64 {
        let mut score = 0.0;
        let all_text = format!(
            "{} {} {} {}",
            entry.command_name,
            entry.args.join(" "),
            entry.output,
            entry.tags.join(" ")
        );

        // Exact phrase match gets highest score
        if all_text.to_lowercase().contains(&query.to_lowercase()) {
            score += 10.0;
        }

        // Word matches
        for word in query_words {
            if all_text.to_lowercase().contains(word) {
                score += 1.0;

                // Command name matches get higher score
                if entry.command_name.to_lowercase().contains(word) {
                    score += 2.0;
                }
            }
        }

        // Boost recent entries slightly
        let days_old = (Utc::now() - entry.timestamp).num_days();
        if days_old < 7 {
            score += 0.5;
        }

        score
    }

    /// Enhanced relevance scoring with TF-IDF and field weighting
    fn calculate_enhanced_relevance_score(
        &self,
        entry: &HistoryEntry,
        query: &str,
        query_words: &[String],
        word_scores: &HashMap<String, f64>,
    ) -> f64 {
        let mut score = 0.0;

        // Create weighted text fields (command name is most important)
        let command_text = entry.command_name.to_lowercase();
        let args_text = entry.args.join(" ").to_lowercase();
        let output_text = entry.output.to_lowercase();
        let tags_text = entry.tags.join(" ").to_lowercase();

        // Field weights
        let command_weight = 5.0;
        let args_weight = 3.0;
        let tags_weight = 2.0;
        let output_weight = 1.0;

        // Exact phrase match in different fields
        let query_lower = query.to_lowercase();
        if command_text.contains(&query_lower) {
            score += 20.0 * command_weight;
        }
        if args_text.contains(&query_lower) {
            score += 15.0 * args_weight;
        }
        if tags_text.contains(&query_lower) {
            score += 12.0 * tags_weight;
        }
        if output_text.contains(&query_lower) {
            score += 8.0 * output_weight;
        }

        // TF-IDF scoring for individual words
        for word in query_words {
            let idf = word_scores.get(word).unwrap_or(&1.0);

            // Calculate term frequency in each field
            let command_tf = self.count_word_occurrences(&command_text, word) as f64;
            let args_tf = self.count_word_occurrences(&args_text, word) as f64;
            let tags_tf = self.count_word_occurrences(&tags_text, word) as f64;
            let output_tf = self.count_word_occurrences(&output_text, word) as f64;

            // Add weighted TF-IDF scores
            score += command_tf * idf * command_weight;
            score += args_tf * idf * args_weight;
            score += tags_tf * idf * tags_weight;
            score += output_tf * idf * output_weight;
        }

        // Boost recent entries
        let days_old = (Utc::now() - entry.timestamp).num_days();
        let recency_boost = match days_old {
            0..=1 => 2.0,
            2..=7 => 1.5,
            8..=30 => 1.0,
            _ => 0.5,
        };
        score *= recency_boost;

        // Boost successful commands slightly
        if entry.success {
            score *= 1.1;
        }

        score
    }

    /// Count word occurrences in text
    fn count_word_occurrences(&self, text: &str, word: &str) -> usize {
        text.matches(word).count()
    }

    fn generate_highlights(
        &self,
        entry: &HistoryEntry,
        query_words: &[String],
        fields: &[SearchField],
    ) -> HashMap<String, Vec<String>> {
        let mut highlights = HashMap::new();

        for field in fields {
            let text = match field {
                SearchField::Command => &entry.command_name,
                SearchField::Output => &entry.output,
                SearchField::Error => entry.error.as_deref().unwrap_or(""),
                SearchField::Args => &entry.args.join(" "),
                SearchField::Tags => &entry.tags.join(" "),
                SearchField::All => {
                    // Handle all fields separately
                    continue;
                }
            };

            let field_highlights = self.extract_highlights(text, query_words);
            if !field_highlights.is_empty() {
                highlights.insert(format!("{:?}", field).to_lowercase(), field_highlights);
            }
        }

        highlights
    }

    fn extract_highlights(&self, text: &str, query_words: &[String]) -> Vec<String> {
        let mut highlights = Vec::new();

        for word in query_words {
            if let Some(start) = text.to_lowercase().find(word) {
                let context_start = start.saturating_sub(50);
                let context_end = (start + word.len() + 50).min(text.len());
                let context = &text[context_start..context_end];

                highlights.push(format!("...{}...", context));
            }
        }

        highlights
    }

    fn add_to_cache(&mut self, entry: HistoryEntry) {
        if self.cache.len() >= self.max_cache_size {
            self.trim_cache();
        }

        self.cache.insert(entry.id.clone(), entry);
    }

    fn trim_cache(&mut self) {
        if self.cache.len() > self.max_cache_size {
            let excess = self.cache.len() - self.max_cache_size / 2;
            let keys_to_remove: Vec<_> = self.cache.keys().take(excess).cloned().collect();

            for key in keys_to_remove {
                self.cache.remove(&key);
            }
        }
    }

    fn compress_data(&self, data: &str) -> Result<Vec<u8>> {
        // Placeholder for compression - would use a real compression library
        Ok(data.as_bytes().to_vec())
    }

    fn decompress_data(&self, data: &[u8]) -> Result<String> {
        // Placeholder for decompression
        Ok(String::from_utf8(data.to_vec())?)
    }

    fn calculate_entry_hash(&self, entry: &HistoryEntry) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        entry.command_name.hash(&mut hasher);
        entry.args.hash(&mut hasher);
        entry.output.hash(&mut hasher);
        entry.session_id.hash(&mut hasher);
        hasher.finish()
    }

    /// Get stopwords for index optimization
    fn get_stopwords(&self) -> HashSet<&'static str> {
        [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "is", "was", "are", "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this",
            "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
        ]
        .iter()
        .copied()
        .collect()
    }

    /// Estimate memory usage of the index
    fn estimate_index_memory_usage(&self) -> usize {
        let command_index_size = self.index.command_index.capacity() * 32 // rough estimate
            + self.index.command_index.values()
                .map(|set| set.capacity() * 24)
                .sum::<usize>();

        let session_index_size = self.index.session_index.capacity() * 32
            + self
                .index
                .session_index
                .values()
                .map(|set| set.capacity() * 24)
                .sum::<usize>();

        let timestamp_index_size = self.index.timestamp_index.capacity() * 40; // DateTime + String

        let word_index_size = self.index.word_index.capacity() * 32
            + self
                .index
                .word_index
                .iter()
                .map(|(word, set)| word.len() + set.capacity() * 24)
                .sum::<usize>();

        command_index_size + session_index_size + timestamp_index_size + word_index_size
    }

    /// Create inverted indices for faster searches
    async fn create_inverted_indices(&mut self) -> Result<()> {
        // Create frequency-based word index for better ranking
        let mut word_frequencies: HashMap<String, usize> = HashMap::new();

        for entry_ids in self.index.word_index.values() {
            for entry_id in entry_ids {
                if let Some(entry) = self.get_entry_by_id(entry_id).await? {
                    let all_text = format!(
                        "{} {} {} {}",
                        entry.command_name,
                        entry.args.join(" "),
                        entry.output,
                        entry.tags.join(" ")
                    );

                    for word in self.tokenize(&all_text) {
                        *word_frequencies.entry(word).or_insert(0) += 1;
                    }
                }
            }
        }

        // Sort words by frequency for potential caching optimizations
        let mut sorted_words: Vec<_> = word_frequencies.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));

        // Store most frequent words separately for faster access
        let _frequent_words: HashSet<String> = sorted_words
            .iter()
            .take(100) // Top 100 most frequent words
            .map(|(word, _)| word.clone())
            .collect();

        // This could be stored as part of the index for optimization
        // For now, we'll just use it to reorganize the word index

        Ok(())
    }

    /// Compact index memory by rebuilding with appropriate capacity
    fn compact_index_memory(&mut self) {
        // Rebuild command index with exact capacity
        let command_entries: Vec<_> = self.index.command_index.drain().collect();
        self.index.command_index = HashMap::with_capacity(command_entries.len());
        for (cmd, entries) in command_entries {
            let mut new_set = HashSet::with_capacity(entries.len());
            new_set.extend(entries);
            self.index.command_index.insert(cmd, new_set);
        }

        // Rebuild session index with exact capacity
        let session_entries: Vec<_> = self.index.session_index.drain().collect();
        self.index.session_index = HashMap::with_capacity(session_entries.len());
        for (session, entries) in session_entries {
            let mut new_set = HashSet::with_capacity(entries.len());
            new_set.extend(entries);
            self.index.session_index.insert(session, new_set);
        }

        // Rebuild word index with exact capacity
        let word_entries: Vec<_> = self.index.word_index.drain().collect();
        self.index.word_index = HashMap::with_capacity(word_entries.len());
        for (word, entries) in word_entries {
            let mut new_set = HashSet::with_capacity(entries.len());
            new_set.extend(entries);
            self.index.word_index.insert(word, new_set);
        }

        // Compact timestamp index
        self.index.timestamp_index.shrink_to_fit();
    }
}

/// Backup data structure
#[derive(Debug, Serialize, Deserialize)]
struct BackupData {
    entries: Vec<HistoryEntry>,
    index: HistoryIndex,
    created_at: DateTime<Utc>,
}

/// Usage insights and patterns
#[derive(Debug, Serialize)]
pub struct UsageInsights {
    pub hourly_usage: [usize; 24],
    pub daily_usage: [usize; 7],
    pub common_sequences: HashMap<String, usize>,
    pub average_session_duration: i64, // minutes
    pub total_sessions: usize,
}

/// Storage optimization results
#[derive(Debug, Serialize)]
pub struct OptimizationResult {
    pub initial_entries: usize,
    pub final_entries: usize,
    pub removed_duplicates: usize,
    pub compressed_outputs: usize,
    pub space_saved_bytes: u64,
}

/// Index optimization results
#[derive(Debug, Serialize)]
pub struct IndexOptimizationResult {
    pub optimization_duration: std::time::Duration,
    pub initial_word_count: usize,
    pub final_word_count: usize,
    pub removed_words: usize,
    pub cleaned_entries: usize,
    pub initial_memory_usage: usize,
    pub final_memory_usage: usize,
    pub memory_saved: usize,
}
